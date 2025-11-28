import random
import numpy as np
import torch
import torch.nn.functional as F

from config import RL_GAMMA, RL_EPSILON, RL_EVAL_GAMES
from environment import FiveChessEnv


class RLTrainer:
    """RL训练相关的方法"""

    def __init__(self, agent):
        self.agent = agent

    def play_one_rl_game_and_update(self, mode="vs_random",
                                     epsilon=RL_EPSILON, gamma=RL_GAMMA,
                                     max_steps=100):
        """
        RL 对局并更新参数

        mode:
            - "vs_random": 先手 RL，后手随机
            - "self_play": 先手/后手都用 RL（同一套参数）
        """
        self.agent.env.reset()
        current_player = 1
        done = False
        step = 0

        log_probs = []
        rewards = []
        players = []
        states_for_diff = []
        actions_for_diff = []

        while not done and step < max_steps:
            state = self.agent.env.get_state()
            empties = self.agent.env.get_empty_positions()
            if not empties:
                break

            # 决定当前玩家的下子策略
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
                    board_player, board_opp = self.agent._build_board_channels(state, current_player)
                    logits_map = self.agent.rl_policy(board_player, board_opp)  # [1,1,H,W]
                    logits = logits_map.view(-1)                          # [H*W]

                    logits_list = []
                    for (i, j) in empties:
                        idx = i * self.agent.board_size + j
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

            next_state, done, win_type = self.agent.env.step(action, current_player)

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

        # RL 参数更新（REINFORCE）
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
            self.agent.rl_optimizer.zero_grad()
            loss_rl.backward()
            self.agent.rl_optimizer.step()

        # 为扩散阶段存数据（好/坏棋都存，用优势作权重）
        for t in range(T_len):
            self.agent.diff_replay.add(
                states_for_diff[t],
                actions_for_diff[t],
                players[t],
                advantages[t]
            )

        return winner, T_len

    def evaluate_rl_vs_random(self, num_games=RL_EVAL_GAMES, max_steps=100):
        """评估RL策略vs随机策略的胜率"""
        self.agent.rl_policy.eval()
        win_count = 0
        draw_count = 0

        for _ in range(num_games):
            env = FiveChessEnv(self.agent.board_size)
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
                    board_player, board_opp = self.agent._build_board_channels(state, current_player)
                    with torch.no_grad():
                        logits_map = self.agent.rl_policy(board_player, board_opp)
                    logits = logits_map.view(-1).cpu().numpy()
                    best_score = -1e9
                    best_action = None
                    for (i, j) in empties:
                        idx = i * self.agent.board_size + j
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

    def demo_game_rl(self, max_steps=100):
        """RL策略演示对局（RL vs Random）"""
        print("\n=== RL 策略测试对局（RL vs Random） ===")
        env = FiveChessEnv(self.agent.board_size)
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
                board_player, board_opp = self.agent._build_board_channels(state, current_player)
                with torch.no_grad():
                    logits_map = self.agent.rl_policy(board_player, board_opp)
                logits = logits_map.view(-1).cpu().numpy()
                best_score = -1e9
                best_action = None
                for (i, j) in empties:
                    idx = i * self.agent.board_size + j
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