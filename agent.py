import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import (
    BOARD_SIZE, DEVICE, RL_GAMMA, RL_EPSILON, RL_EVAL_GAMES,
    DIFFUSION_STEPS, DIFF_BATCH_SIZE, SAVE_INTERVAL,
    LOG_DIR, MODEL_SAVE_DIR, DIFF_REPLAY_CAPACITY
)
from environment import FiveChessEnv
from replay_buffer import DiffReplayBuffer
from networks import RLPolicyNet, UNetPolicy


class RLPlusDiffusionAgent:
    """Agent：先 RL (30% vs Random, 70% RL self-play) 再 扩散"""

    def __init__(self, board_size=BOARD_SIZE, diffusion_steps=DIFFUSION_STEPS, device=DEVICE):
        self.board_size = board_size
        self.device = device

        self.env = FiveChessEnv(board_size)
        self.diff_replay = DiffReplayBuffer(50000)

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

    def _build_board_channels(self, state_flat, player):
        """状态 -> 通道转换"""
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

    def q_sample(self, x0, t, noise):
        """扩散前向采样"""
        B = x0.shape[0]
        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def p_sample_step(self, x_t, t, board_player, board_opp):
        """扩散反向采样步骤"""
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

    def diffusion_train_step(self, batch_states, batch_actions, batch_players, batch_advantages):
        """扩散训练步骤"""
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

    def close(self):
        """关闭资源"""
        self.writer.close()