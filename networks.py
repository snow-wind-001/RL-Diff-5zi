import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BOARD_SIZE, DIFFUSION_STEPS


class RLPolicyNet(nn.Module):
    """RL 策略网络（CNN 输出 policy logits）"""

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


class UNetPolicy(nn.Module):
    """扩散策略网络：小型 U-Net"""

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