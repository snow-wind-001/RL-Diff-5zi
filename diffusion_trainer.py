import numpy as np
import random
import torch

from config import RL_GAMMA


class DiffusionTrainer:
    """扩散训练相关的方法"""

    def __init__(self, agent):
        self.agent = agent

    def play_one_game_for_diffusion(self, gamma=RL_GAMMA, max_steps=100):
        """用 RL 策略生成扩散训练数据"""
        self.agent.env.reset()
        current_player = 1
        done = False
        step = 0

        states = []
        actions = []
        players = []

        while not done and step < max_steps:
            state = self.agent.env.get_state()
            empties = self.agent.env.get_empty_positions()
            if not empties:
                break

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

            _, done, win_type = self.agent.env.step(action, current_player)

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
            self.agent.diff_replay.add(states[t], actions[t], players[t], advantages[t])

        return winner, T_len

    def sample_action_from_diffusion(self, state, player, temperature=0.8, num_steps=None):
        """从扩散策略中采样动作"""
        if num_steps is None:
            num_steps = self.agent.T

        H = W = self.agent.board_size
        state_tensor = torch.from_numpy(state).float().view(1, 1, H, W).to(self.agent.device)
        player_t = torch.tensor([player], dtype=torch.float32, device=self.agent.device).view(1, 1, 1, 1)
        board_player = (state_tensor == player_t).float()
        board_opp = (state_tensor == -player_t).float()

        x_t = torch.randn(1, 1, H, W, device=self.agent.device)
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                x_t = self.agent.p_sample_step(x_t, t, board_player, board_opp)
        x0 = x_t[0, 0].cpu().numpy()

        empties = self.agent.env.get_empty_positions()
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

    def demo_game_diffusion(self, max_steps=100):
        """扩散策略演示对局"""
        print("\n=== 扩散策略测试对局（Diffusion vs Random） ===")
        from environment import FiveChessEnv
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
                # 扩散策略采样
                self.agent.env.chessState = env.chessState.copy()
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