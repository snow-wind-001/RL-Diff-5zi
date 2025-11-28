import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

# ============ 全局配置 ============
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BOARD_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---- RL 阶段配置 ----
RL_MAX_EPISODES = 50000          # RL 最大训练局数
RL_GAMMA = 0.97                 # RL 折扣因子
RL_EPSILON = 0.2                # 自博弈时 exploration 概率
RL_EVAL_INTERVAL = 50           # 每隔多少局评估一次 vs 随机
RL_DEMO_INTERVAL = 50           # 每隔多少局打印一局 RL 对弈过程
RL_WINRATE_THRESHOLD = 0.7      # RL 胜率阈值，超过后开始扩散模型训练
RL_EVAL_GAMES = 20              # 评估时对局数
RL_RANDOM_RATIO = 0.3           # 前 30% 局数 RL vs Random，后 70% RL vs RL

# ---- 扩散阶段配置 ----
DIFFUSION_STEPS = 100            # 扩散时间步数
DIFF_MAX_EPISODES = 50000        # 扩散训练局数
DIFF_BATCH_SIZE = 2048           # 扩散训练 batch size
DIFF_REPLAY_CAPACITY = 50000    # 扩散经验池容量
DIFF_DEMO_INTERVAL = 1000         # 每隔多少局打印一局扩散对弈过程
DIFF_TRAIN_STEPS_PER_EP = 1     # 每局生成后，执行多少次梯度更新

# ---- 日志和保存配置 ----
LOG_DIR = "./logs"               # TensorBoard日志目录
MODEL_SAVE_DIR = "./models"      # 模型保存目录
SAVE_INTERVAL = 100              # 每隔多少轮保存一次模型
BEST_MODEL_METRIC = "winrate"    # 用于判断最佳模型的指标 ("winrate" 或 "loss")

# 随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 创建日志和模型保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_DIR, f"run_{timestamp}")
MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, f"run_{timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"日志目录: {LOG_DIR}")
print(f"模型保存目录: {MODEL_SAVE_DIR}")


# ============ 五子棋环境 ============
class FiveChessEnv:
    def __init__(self, board_size=BOARD_SIZE):
        self.M = board_size
        self.N = board_size
        # 空:0  先手:1  后手:-1
        self.chessState = np.zeros((self.M, self.N), dtype=np.int8)

    def reset(self):
        self.chessState.fill(0)
        return self.get_state()

    def get_state(self):
        return self.chessState.astype(np.float32).flatten()

    def step(self, action, player):
        i, j = action
        if self.chessState[i, j] != 0:
            raise ValueError(f"Invalid move at ({i}, {j})!")
        self.chessState[i, j] = player
        done, win_type = self.check_win(player)
        return self.get_state(), done, win_type

    def get_empty_positions(self):
        empties = np.argwhere(self.chessState == 0)
        return [tuple(pos) for pos in empties]

    def check_win(self, who):
        """
        检查当前棋盘上是否存在玩家 who (1 或 -1) 连续 5 个棋子的情况。
        四个方向全部包含：横、竖、斜线、反斜线。
        """
        M, N = self.M, self.N
        board = self.chessState

        for i in range(M):
            for j in range(N):
                if board[i, j] != who:
                    continue

                # 横向 —
                if j + 4 < N and np.all(board[i, j:j+5] == who):
                    return True, "—"

                # 纵向 |
                if i + 4 < M and np.all(board[i:i+5, j] == who):
                    return True, "|"

                # 正斜线 \
                if i + 4 < M and j + 4 < N:
                    ok = True
                    for k in range(5):
                        if board[i + k, j + k] != who:
                            ok = False
                            break
                    if ok:
                        return True, "\\"

                # 反斜线 /
                if i + 4 < M and j - 4 >= 0:
                    ok = True
                    for k in range(5):
                        if board[i + k, j - k] != who:
                            ok = False
                            break
                    if ok:
                        return True, "/"

        return False, ""

    def print_board(self):
        symbols = {1: "●", -1: "○", 0: " ."}
        for i in range(self.M):
            row = "".join(symbols[int(x)] for x in self.chessState[i])
            print(row)
        print()


# ============ 扩散经验回放 ============
class DiffReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.players = []
        self.advantages = []

    def add(self, state, action, player, advantage):
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.players.pop(0)
            self.advantages.pop(0)
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.int64))
        self.players.append(int(player))
        self.advantages.append(float(advantage))

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.states), size=batch_size)
        batch_states = [self.states[i] for i in idxs]
        batch_actions = [self.actions[i] for i in idxs]
        batch_players = [self.players[i] for i in idxs]
        batch_adv = [self.advantages[i] for i in idxs]
        return (
            np.stack(batch_states, axis=0),
            np.stack(batch_actions, axis=0),
            np.array(batch_players, dtype=np.int64),
            np.array(batch_adv, dtype=np.float32),
        )


# ============ RL 策略网络（CNN 输出 policy logits）===========
class RLPolicyNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, base_channels=64):
        super().__init__()
        self.board_size = board_size
        # 输入: [B,2,H,W]  (当前玩家棋子通道 + 对手棋子通道)
        self.conv1 = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, 1, kernel_size=1)  # 输出 [B,1,H,W] logits

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, board_player, board_opp):
        x = torch.cat([board_player, board_opp], dim=1)  # [B,2,H,W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.conv3(x)  # [B,1,H,W]
        return logits


# ============ 扩散策略网络：小型 U-Net ============
class UNetPolicy(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_timesteps=DIFFUSION_STEPS):
        super().__init__()
        self.num_timesteps = num_timesteps

        # 下采样
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)  # 10->5

        # 中间层
        self.conv_mid1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up1 = nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x_t, board_player, board_opp, t_scalar):
        B, _, H, W = x_t.shape
        t_norm = (t_scalar.float() + 1.0) / float(self.num_timesteps)
        t_plane = t_norm.view(B, 1, 1, 1).expand(-1, 1, H, W)

        x = torch.cat([x_t, board_player, board_opp, t_plane], dim=1)  # [B,4,H,W]

        h1 = F.relu(self.conv1(x))             # [B,C,10,10]
        h2 = F.relu(self.conv2(h1))            # [B,2C,5,5]
        h_mid = F.relu(self.conv_mid1(h2))     # [B,2C,5,5]
        h_up = self.up(h_mid)                  # [B,2C,10,10]
        h_cat = torch.cat([h_up, h1], dim=1)   # [B,3C,10,10]
        h = F.relu(self.conv_up1(h_cat))       # [B,C,10,10]
        out = self.conv_out(h)                 # [B,1,10,10]
        return out


# ============ Agent：先 RL (30% vs Random, 70% RL self-play) 再 扩散 ============
class RLPlusDiffusionAgent:
    def __init__(self,
                 board_size=BOARD_SIZE,
                 diffusion_steps=DIFFUSION_STEPS,
                 device=DEVICE):
        self.board_size = board_size
        self.device = device

        self.env = FiveChessEnv(board_size)
        self.diff_replay = DiffReplayBuffer(DIFF_REPLAY_CAPACITY)

        # RL 策略网络
        self.rl_policy = RLPolicyNet(board_size=board_size).to(self.device)
        self.rl_optimizer = optim.Adam(self.rl_policy.parameters(), lr=1e-4)

        # 扩散策略网络
        self.T = diffusion_steps
        self.diff_policy = UNetPolicy(in_channels=4,
                                      base_channels=32,
                                      num_timesteps=self.T).to(self.device)
        self.diff_optimizer = optim.Adam(self.diff_policy.parameters(), lr=1e-4)

        # 扩散超参数
        betas = torch.linspace(1e-4, 0.02, self.T, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # TensorBoard写入器
        self.writer = SummaryWriter(LOG_DIR)
        
        # 最佳模型跟踪
        self.best_rl_winrate = 0.0
        self.best_diff_loss = float('inf')
        self.training_start_time = time.time()

    # ---------- 工具：状态 -> 通道 ----------
    def _build_board_channels(self, state_flat, player):
        """
        state_flat: numpy [100] 或 torch [B,100]
        返回 board_player, board_opp: [B,1,H,W]
        """
        H = W = self.board_size
        if isinstance(state_flat, np.ndarray):
            board = torch.from_numpy(state_flat).float().view(1, 1, H, W).to(self.device)
            player_t = torch.tensor([player], dtype=torch.float32, device=self.device).view(1, 1, 1, 1)
        else:
            B = state_flat.shape[0]
            board = state_flat.float().view(B, 1, H, W).to(self.device)
            player_t = torch.ones((B, 1, 1, 1), dtype=torch.float32, device=self.device) * player

        board_player = (board == player_t).float()
        board_opp = (board == -player_t).float()
        return board_player, board_opp

    # ---------- 模型保存和加载 ----------
    def save_model(self, episode, stage, metrics=None):
        """保存模型权重"""
        rl_path = os.path.join(MODEL_SAVE_DIR, f"rl_policy_ep{episode}_{stage}.pth")
        diff_path = os.path.join(MODEL_SAVE_DIR, f"diff_policy_ep{episode}_{stage}.pth")
        
        torch.save({
            'episode': episode,
            'stage': stage,
            'rl_policy_state_dict': self.rl_policy.state_dict(),
            'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
            'diff_policy_state_dict': self.diff_policy.state_dict(),
            'diff_optimizer_state_dict': self.diff_optimizer.state_dict(),
            'metrics': metrics or {},
            'best_rl_winrate': self.best_rl_winrate,
            'best_diff_loss': self.best_diff_loss,
        }, os.path.join(MODEL_SAVE_DIR, f"checkpoint_ep{episode}_{stage}.pth"))
        
        torch.save(self.rl_policy.state_dict(), rl_path)
        torch.save(self.diff_policy.state_dict(), diff_path)
        
        print(f"模型已保存: {rl_path}, {diff_path}")

    def save_best_model(self, episode, stage, metrics):
        """保存最佳模型"""
        if stage == "rl" and metrics.get('winrate', 0) > self.best_rl_winrate:
            self.best_rl_winrate = metrics['winrate']
            best_rl_path = os.path.join(MODEL_SAVE_DIR, "best_rl_policy.pth")
            torch.save(self.rl_policy.state_dict(), best_rl_path)
            print(f"新的最佳RL模型已保存，胜率: {self.best_rl_winrate:.3f}")
            
        elif stage == "diffusion" and metrics.get('loss', float('inf')) < self.best_diff_loss:
            self.best_diff_loss = metrics['loss']
            best_diff_path = os.path.join(MODEL_SAVE_DIR, "best_diff_policy.pth")
            torch.save(self.diff_policy.state_dict(), best_diff_path)
            print(f"新的最佳扩散模型已保存，损失: {self.best_diff_loss:.4f}")

    def log_metrics(self, episode, stage, metrics):
        """记录指标到TensorBoard"""
        elapsed_time = time.time() - self.training_start_time
        
        for key, value in metrics.items():
            self.writer.add_scalar(f'{stage}/{key}', value, episode)
        
        self.writer.add_scalar('common/elapsed_time', elapsed_time, episode)
        self.writer.add_scalar('common/episode', episode, episode)
        
        # 记录学习率
        if stage == "rl":
            lr = self.rl_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('rl/learning_rate', lr, episode)
        elif stage == "diffusion":
            lr = self.diff_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('diffusion/learning_rate', lr, episode)

    # ---------- RL 对局（两种模式） ----------
    def play_one_rl_game_and_update(self, mode="vs_random",
                                     epsilon=RL_EPSILON, gamma=RL_GAMMA,
                                     max_steps=100):
        """
        mode:
            - "vs_random": 先手 RL，后手随机
            - "self_play": 先手/后手都用 RL（同一套参数）
        """
        self.env.reset()
        current_player = 1
        done = False
        step = 0

        log_probs = []
        rewards = []
        players = []
        states_for_diff = []
        actions_for_diff = []

        while not done and step < max_steps:
            state = self.env.get_state()
            empties = self.env.get_empty_positions()
            if not empties:
                break

            # ---- 决定当前玩家的下子策略 ----
            use_rl = False
            if mode == "vs_random":
                if current_player == 1:
                    use_rl = True    # RL 当先手
                else:
                    use_rl = False   # 后手随机
            elif mode == "self_play":
                use_rl = True        # 双方均用 RL

            if use_rl:
                # epsilon-greedy：探索 or 策略采样
                if np.random.rand() < epsilon:
                    action = random.choice(empties)
                    log_prob = None   # 探索步骤不参与梯度
                else:
                    board_player, board_opp = self._build_board_channels(state, current_player)
                    logits_map = self.rl_policy(board_player, board_opp)  # [1,1,H,W]
                    logits = logits_map.view(-1)                          # [H*W]

                    logits_list = []
                    for (i, j) in empties:
                        idx = i * self.board_size + j
                        logits_list.append(logits[idx])
                    logits_tensor = torch.stack(logits_list, dim=0)       # [num_empties]

                    log_probs_all = F.log_softmax(logits_tensor, dim=0)
                    probs_all = log_probs_all.exp()
                    idx_choice = torch.multinomial(probs_all, num_samples=1).item()
                    log_prob = log_probs_all[idx_choice]
                    action = empties[idx_choice]
            else:
                # 纯随机玩家
                action = random.choice(empties)
                log_prob = None

            next_state, done, win_type = self.env.step(action, current_player)

            rewards.append(0.0)  # 先占位，之后根据 winner 统一赋值
            players.append(current_player)
            log_probs.append(log_prob)
            states_for_diff.append(state)
            actions_for_diff.append(action)

            if done:
                winner = current_player
                break

            current_player = -1 if current_player == 1 else 1
            step += 1

        if not done:
            winner = 0  # 平局

        T_len = len(rewards)
        if T_len == 0:
            return winner, 0

        # 基于 winner 和 player 计算每步 reward
        if winner == 0:
            rewards = [0.0] * T_len
        else:
            rewards = [1.0 if players[t] == winner else -1.0 for t in range(T_len)]

        # 折扣回报 G_t
        returns = np.zeros(T_len, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T_len)):
            running = rewards[t] + gamma * running
            returns[t] = running

        baseline = returns.mean()
        advantages = returns - baseline

        # ---- RL 参数更新（REINFORCE）----
        loss_rl = 0.0
        count_terms = 0
        for t in range(T_len):
            log_prob = log_probs[t]
            if log_prob is None:
                continue  # 随机/探索步骤不参与梯度
            adv = advantages[t]
            loss_rl += -log_prob * adv
            count_terms += 1

        if count_terms > 0:
            loss_rl = loss_rl / count_terms
            self.rl_optimizer.zero_grad()
            loss_rl.backward()
            self.rl_optimizer.step()

        # ---- 为扩散阶段存数据（好/坏棋都存，用优势作权重）----
        for t in range(T_len):
            self.diff_replay.add(
                states_for_diff[t],
                actions_for_diff[t],
                players[t],
                advantages[t]
            )

        return winner, T_len

    # ---------- RL 评估 (RL vs Random) ----------
    def evaluate_rl_vs_random(self, num_games=RL_EVAL_GAMES, max_steps=100):
        self.rl_policy.eval()
        win_count = 0
        draw_count = 0

        for _ in range(num_games):
            env = FiveChessEnv(self.board_size)
            env.reset()
            current_player = 1
            done = False
            step = 0

            while not done and step < max_steps:
                empties = env.get_empty_positions()
                if not empties:
                    break

                state = env.get_state()
                if current_player == 1:
                    # RL 贪心
                    board_player, board_opp = self._build_board_channels(state, current_player)
                    with torch.no_grad():
                        logits_map = self.rl_policy(board_player, board_opp)
                    logits = logits_map.view(-1).cpu().numpy()
                    best_score = -1e9
                    best_action = None
                    for (i, j) in empties:
                        idx = i * self.board_size + j
                        if logits[idx] > best_score:
                            best_score = logits[idx]
                            best_action = (i, j)
                    action = best_action
                else:
                    # 随机玩家
                    action = random.choice(empties)

                _, done, win_type = env.step(action, current_player)
                if done:
                    winner = current_player
                    break

                current_player = -1 if current_player == 1 else 1
                step += 1

            if not done:
                winner = 0

            if winner == 1:
                win_count += 1
            elif winner == 0:
                draw_count += 1

        winrate = win_count / num_games
        print(f"[RL Eval] win={win_count}, draw={draw_count}, total={num_games}, winrate={winrate:.3f}")
        return winrate

    # ---------- RL 演示对局（RL vs Random） ----------
    def demo_game_rl(self, max_steps=100):
        print("\n=== RL 策略测试对局（RL vs Random） ===")
        env = FiveChessEnv(self.board_size)
        env.reset()
        current_player = 1
        done = False
        step = 0

        env.print_board()

        while not done and step < max_steps:
            print(f"Step {step + 1}, Player {current_player}")
            empties = env.get_empty_positions()
            if not empties:
                print("无子可下，平局。")
                break

            state = env.get_state()
            if current_player == 1:
                board_player, board_opp = self._build_board_channels(state, current_player)
                with torch.no_grad():
                    logits_map = self.rl_policy(board_player, board_opp)
                logits = logits_map.view(-1).cpu().numpy()
                best_score = -1e9
                best_action = None
                for (i, j) in empties:
                    idx = i * self.board_size + j
                    if logits[idx] > best_score:
                        best_score = logits[idx]
                        best_action = (i, j)
                action = best_action
            else:
                action = random.choice(empties)

            _, done, win_type = env.step(action, current_player)
            env.print_board()
            if done:
                print(f"Player {current_player} wins! ({win_type})")
                break

            current_player = -1 if current_player == 1 else 1
            step += 1

        if not done:
            print("对局结束：平局。")

    # ---------- 扩散前向/反向 ----------
    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def p_sample_step(self, x_t, t, board_player, board_opp):
        B = x_t.shape[0]
        t_scalar = torch.full((B,), t, device=self.device, dtype=torch.long)
        eps_pred = self.diff_policy(x_t, board_player, board_opp, t_scalar)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

        mean = coef1 * (x_t - coef2 * eps_pred)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        return x_prev

    def sample_action_from_diffusion(self, state, player, temperature=0.8, num_steps=None):
        if num_steps is None:
            num_steps = self.T

        H = W = self.board_size
        state_tensor = torch.from_numpy(state).float().view(1, 1, H, W).to(self.device)
        player_t = torch.tensor([player], dtype=torch.float32, device=self.device).view(1, 1, 1, 1)
        board_player = (state_tensor == player_t).float()
        board_opp = (state_tensor == -player_t).float()

        x_t = torch.randn(1, 1, H, W, device=self.device)
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                x_t = self.p_sample_step(x_t, t, board_player, board_opp)
        x0 = x_t[0, 0].cpu().numpy()

        empties = self.env.get_empty_positions()
        if not empties:
            return None

        logits = x0.reshape(-1)
        mask = np.full_like(logits, -1e9, dtype=np.float32)
        for (i, j) in empties:
            idx = i * W + j
            mask[idx] = logits[idx]

        logits_masked = mask / max(temperature, 1e-3)
        probs = np.exp(logits_masked - np.max(logits_masked))
        psum = probs.sum()
        if psum <= 0:
            return random.choice(empties)
        probs = probs / psum
        idx = np.random.choice(len(probs), p=probs)
        i, j = divmod(int(idx), W)
        return (i, j)

    # ---------- 扩散训练 step ----------
    def diffusion_train_step(self, batch_states, batch_actions, batch_players, batch_advantages):
        self.diff_policy.train()
        B = batch_states.shape[0]
        H = W = self.board_size

        states = torch.from_numpy(batch_states).float().to(self.device)
        actions = torch.from_numpy(batch_actions).long().to(self.device)
        players = torch.from_numpy(batch_players).float().to(self.device)
        advantages = torch.from_numpy(batch_advantages).float().to(self.device)

        board = states.view(B, 1, H, W)
        player_t = players.view(B, 1, 1, 1)
        board_player = (board == player_t).float()
        board_opp = (board == -player_t).float()

        x0 = torch.zeros(B, 1, H, W, device=self.device)
        idx_i = actions[:, 0]
        idx_j = actions[:, 1]
        x0[torch.arange(B), 0, idx_i, idx_j] = 1.0

        t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        eps_pred = self.diff_policy(x_t, board_player, board_opp, t)

        adv = advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        w = torch.sigmoid(adv)

        loss_per = ((noise - eps_pred) ** 2).mean(dim=[1, 2, 3])
        loss = (loss_per * w).mean()

        self.diff_optimizer.zero_grad()
        loss.backward()
        self.diff_optimizer.step()
        return loss.item()

    # ---------- 用 RL 策略生成扩散训练数据 ----------
    def play_one_game_for_diffusion(self, gamma=RL_GAMMA, max_steps=100):
        self.env.reset()
        current_player = 1
        done = False
        step = 0

        states = []
        actions = []
        players = []

        while not done and step < max_steps:
            state = self.env.get_state()
            empties = self.env.get_empty_positions()
            if not empties:
                break

            board_player, board_opp = self._build_board_channels(state, current_player)
            with torch.no_grad():
                logits_map = self.rl_policy(board_player, board_opp)
            logits = logits_map.view(-1).cpu().numpy()
            best_score = -1e9
            best_action = None
            for (i, j) in empties:
                idx = i * self.board_size + j
                if logits[idx] > best_score:
                    best_score = logits[idx]
                    best_action = (i, j)
            action = best_action

            _, done, win_type = self.env.step(action, current_player)

            states.append(state)
            actions.append(action)
            players.append(current_player)

            if done:
                winner = current_player
                break

            current_player = -1 if current_player == 1 else 1
            step += 1

        if not done:
            winner = 0

        T_len = len(states)
        if T_len == 0:
            return winner, 0

        if winner == 0:
            rewards = [0.0] * T_len
        else:
            rewards = [1.0 if players[t] == winner else -1.0 for t in range(T_len)]

        returns = np.zeros(T_len, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T_len)):
            running = rewards[t] + gamma * running
            returns[t] = running

        baseline = returns.mean()
        advantages = returns - baseline

        for t in range(T_len):
            self.diff_replay.add(states[t], actions[t], players[t], advantages[t])

        return winner, T_len

    # ---------- 扩散策略测试对局 ----------
    def demo_game_diffusion(self, max_steps=100):
        print("\n=== 扩散策略测试对局（Diffusion vs Random） ===")
        env = FiveChessEnv(self.board_size)
        env.reset()
        current_player = 1
        done = False
        step = 0

        env.print_board()

        while not done and step < max_steps:
            print(f"Step {step + 1}, Player {current_player}")
            empties = env.get_empty_positions()
            if not empties:
                print("无子可下，平局。")
                break

            state = env.get_state()
            if current_player == 1:
                # 扩散策略采样
                self.env.chessState = env.chessState.copy()
                action = self.sample_action_from_diffusion(state, current_player, temperature=0.7)
                if action is None or env.chessState[action[0], action[1]] != 0:
                    action = random.choice(empties)
            else:
                # 随机玩家
                action = random.choice(empties)

            _, done, win_type = env.step(action, current_player)
            env.print_board()
            if done:
                print(f"Player {current_player} wins! ({win_type})")
                break

            current_player = -1 if current_player == 1 else 1
            step += 1

        if not done:
            print("对局结束：平局。")

    # ---------- 总训练流程 ----------
    def train(self):
        print("========== 阶段 1：RL 策略训练 ==========")
        start_diffusion = False
        rl_random_episodes = int(RL_MAX_EPISODES * RL_RANDOM_RATIO)
        
        # RL训练统计
        rl_win_count = 0
        rl_draw_count = 0
        rl_total_steps = 0

        for ep in range(1, RL_MAX_EPISODES + 1):
            # 前 30% 用 RL vs Random，后 70% 用 RL vs RL 自博弈
            if ep <= rl_random_episodes:
                mode = "vs_random"
            else:
                mode = "self_play"

            winner, steps = self.play_one_rl_game_and_update(mode=mode)
            
            # 统计
            rl_total_steps += steps
            if winner == 1:
                rl_win_count += 1
            elif winner == 0:
                rl_draw_count += 1

            # 定期记录指标和保存模型
            if ep % RL_EVAL_INTERVAL == 0:
                winrate = self.evaluate_rl_vs_random()
                
                # 记录指标
                metrics = {
                    'winrate': winrate,
                    'win_count': rl_win_count,
                    'draw_count': rl_draw_count,
                    'avg_steps': rl_total_steps / ep,
                    'replay_size': len(self.diff_replay)
                }
                self.log_metrics(ep, "rl", metrics)
                
                # 保存最佳模型
                self.save_best_model(ep, "rl", {'winrate': winrate})
                
                # 定期保存检查点
                if ep % SAVE_INTERVAL == 0:
                    self.save_model(ep, "rl", metrics)
                
                if winrate >= RL_WINRATE_THRESHOLD:
                    print(f"RL 胜率达到阈值 {RL_WINRATE_THRESHOLD:.2f}，开始扩散模型训练阶段。")
                    start_diffusion = True
                    break

            if ep % RL_DEMO_INTERVAL == 0:
                print(f"\n[RL] Episode {ep}, mode={mode}, last winner={winner}, steps={steps}, "
                      f"diff_replay_size={len(self.diff_replay)}")
                self.demo_game_rl()

        if not start_diffusion:
            print("RL 训练达到最大轮次，仍未达到阈值，依然开始扩散训练（可能策略较弱）。")
            # 保存最终RL模型
            self.save_model(RL_MAX_EPISODES, "rl_final", {
                'winrate': rl_win_count / RL_MAX_EPISODES,
                'win_count': rl_win_count,
                'draw_count': rl_draw_count
            })

        print("\n========== 阶段 2：扩散策略训练 ==========")

        # 扩散数据预热：经验池太小时，先用 RL 多生成几局
        while len(self.diff_replay) < DIFF_BATCH_SIZE:
            winner, steps = self.play_one_game_for_diffusion()
            print(f"预热扩散数据：winner={winner}, steps={steps}, replay_size={len(self.diff_replay)}")

        # 扩散训练统计
        diff_win_count = 0
        diff_draw_count = 0
        diff_total_steps = 0
        diff_losses = []

        for ep in range(1, DIFF_MAX_EPISODES + 1):
            winner, steps = self.play_one_game_for_diffusion()
            
            # 统计
            diff_total_steps += steps
            if winner == 1:
                diff_win_count += 1
            elif winner == 0:
                diff_draw_count += 1

            # 执行扩散训练步骤
            episode_losses = []
            for _ in range(DIFF_TRAIN_STEPS_PER_EP):
                if len(self.diff_replay) >= DIFF_BATCH_SIZE:
                    (batch_states,
                     batch_actions,
                     batch_players,
                     batch_advantages) = self.diff_replay.sample(DIFF_BATCH_SIZE)
                    loss = self.diffusion_train_step(batch_states,
                                                     batch_actions,
                                                     batch_players,
                                                     batch_advantages)
                    episode_losses.append(loss)
                else:
                    loss = None
            
            # 计算平均损失
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            diff_losses.append(avg_loss)

            # 定期记录指标和保存模型
            if ep % SAVE_INTERVAL == 0:
                metrics = {
                    'loss': avg_loss,
                    'win_count': diff_win_count,
                    'draw_count': diff_draw_count,
                    'avg_steps': diff_total_steps / ep,
                    'replay_size': len(self.diff_replay),
                    'avg_loss_100': np.mean(diff_losses[-100:]) if len(diff_losses) >= 100 else np.mean(diff_losses)
                }
                self.log_metrics(ep, "diffusion", metrics)
                
                # 保存最佳模型
                self.save_best_model(ep, "diffusion", {'loss': avg_loss})
                
                # 保存检查点
                self.save_model(ep, "diffusion", metrics)

            if ep % DIFF_DEMO_INTERVAL == 0:
                print(f"\n[Diffusion] Episode {ep}, last winner={winner}, steps={steps}, "
                      f"replay_size={len(self.diff_replay)}"
                      + (f", loss={avg_loss:.4f}" if avg_loss is not None else ""))
                self.demo_game_diffusion()

        # 保存最终扩散模型
        self.save_model(DIFF_MAX_EPISODES, "diffusion_final", {
            'loss': np.mean(diff_losses) if diff_losses else 0.0,
            'win_count': diff_win_count,
            'draw_count': diff_draw_count
        })
        
        # 关闭TensorBoard写入器
        self.writer.close()
        print(f"训练完成！日志保存在: {LOG_DIR}")
        print(f"模型保存在: {MODEL_SAVE_DIR}")


# ============ 主程序 ============
if __name__ == "__main__":
    agent = RLPlusDiffusionAgent()
    agent.train()
    print("\n=== 最终扩散策略演示对局 ===")
    agent.demo_game_diffusion()

