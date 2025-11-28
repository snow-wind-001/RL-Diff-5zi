import os
import torch
import numpy as np
import random
from datetime import datetime

# ============ 全局配置 ============
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 基本配置
BOARD_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL 阶段配置
RL_MAX_EPISODES = 50000          # RL 最大训练局数
RL_GAMMA = 0.97                 # RL 折扣因子
RL_EPSILON = 0.2                # 自博弈时 exploration 概率
RL_EVAL_INTERVAL = 50           # 每隔多少局评估一次 vs 随机
RL_DEMO_INTERVAL = 50           # 每隔多少局打印一局 RL 对弈过程
RL_WINRATE_THRESHOLD = 0.7      # RL 胜率阈值，超过后开始扩散模型训练
RL_EVAL_GAMES = 20              # 评估时对局数
RL_RANDOM_RATIO = 0.3           # 前 30% 局数 RL vs Random，后 70% RL vs RL

# 扩散阶段配置
DIFFUSION_STEPS = 100            # 扩散时间步数
DIFF_MAX_EPISODES = 50000        # 扩散训练局数
DIFF_BATCH_SIZE = 2048           # 扩散训练 batch size
DIFF_REPLAY_CAPACITY = 50000    # 扩散经验池容量
DIFF_DEMO_INTERVAL = 1000         # 每隔多少局打印一局扩散对弈过程
DIFF_TRAIN_STEPS_PER_EP = 1     # 每局生成后，执行多少次梯度更新

# 日志和保存配置
LOG_DIR = "./logs"               # TensorBoard日志目录
MODEL_SAVE_DIR = "./models"      # 模型保存目录
SAVE_INTERVAL = 100              # 每隔多少轮保存一次模型
BEST_MODEL_METRIC = "winrate"    # 用于判断最佳模型的指标 ("winrate" 或 "loss")

# 随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 创建日志和模型保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_DIR, f"run_{timestamp}")
MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, f"run_{timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 为其他模块导出的变量
__all__ = [
    'BOARD_SIZE', 'DEVICE', 'RL_MAX_EPISODES', 'RL_GAMMA', 'RL_EPSILON',
    'RL_EVAL_INTERVAL', 'RL_DEMO_INTERVAL', 'RL_WINRATE_THRESHOLD',
    'RL_EVAL_GAMES', 'RL_RANDOM_RATIO', 'DIFFUSION_STEPS', 'DIFF_MAX_EPISODES',
    'DIFF_BATCH_SIZE', 'DIFF_REPLAY_CAPACITY', 'DIFF_DEMO_INTERVAL',
    'DIFF_TRAIN_STEPS_PER_EP', 'LOG_DIR', 'MODEL_SAVE_DIR', 'SAVE_INTERVAL',
    'BEST_MODEL_METRIC', 'RANDOM_SEED'
]