# RL-Diff-5zi 重构后的代码结构

## 概述

本项目将原始的单文件 `RL-Diff-5v2.py` 重构为模块化的代码结构，提高了代码的可维护性和可扩展性。项目实现了强化学习(RL)结合扩散模型的五子棋算法。

## 文件结构

```
RL-Diff-5zi/
├── config.py                 # 全局配置参数
├── environment.py           # 五子棋游戏环境
├── replay_buffer.py         # 扩散经验回放缓冲区
├── networks.py             # RL和扩散策略网络
├── agent.py                # 主要的Agent类
├── rl_trainer.py           # RL训练相关方法
├── diffusion_trainer.py    # 扩散训练相关方法
├── train.py                # 主训练程序
├── test_refactored.py      # 重构代码测试脚本
├── README_refactored.md     # 本说明文档
└── RL-Diff-5v2.py         # 原始单文件（保留）
```

## 模块说明

### 1. config.py
- **功能**: 包含所有全局配置参数
- **主要配置**:
  - `BOARD_SIZE`: 棋盘大小 (默认10x10)
  - `DEVICE`: 计算设备 (CPU/GPU)
  - RL训练参数: `RL_MAX_EPISODES`, `RL_GAMMA`, `RL_EPSILON` 等
  - 扩散训练参数: `DIFFUSION_STEPS`, `DIFF_MAX_EPISODES` 等
  - 日志和保存配置: `LOG_DIR`, `MODEL_SAVE_DIR` 等

### 2. environment.py
- **功能**: 五子棋游戏环境实现
- **主要类**: `FiveChessEnv`
- **方法**:
  - `reset()`: 重置棋盘
  - `step()`: 执行一步动作
  - `check_win()`: 检查获胜条件
  - `get_empty_positions()`: 获取空位置
  - `print_board()`: 打印棋盘

### 3. replay_buffer.py
- **功能**: 扩散经验回放缓冲区
- **主要类**: `DiffReplayBuffer`
- **方法**:
  - `add()`: 添加经验到缓冲区
  - `sample()`: 采样批次数据

### 4. networks.py
- **功能**: 神经网络架构定义
- **主要类**:
  - `RLPolicyNet`: RL策略网络 (CNN)
  - `UNetPolicy`: 扩散策略网络 (小型U-Net)

### 5. agent.py
- **功能**: 核心Agent类，整合所有组件
- **主要类**: `RLPlusDiffusionAgent`
- **功能**:
  - 管理RL和扩散策略网络
  - 处理扩散前向/反向采样
  - 模型保存和加载
  - TensorBoard日志记录

### 6. rl_trainer.py
- **功能**: RL训练相关方法
- **主要类**: `RLTrainer`
- **方法**:
  - `play_one_rl_game_and_update()`: 进行一局RL游戏并更新参数
  - `evaluate_rl_vs_random()`: 评估RL策略vs随机策略
  - `demo_game_rl()`: RL策略演示对局

### 7. diffusion_trainer.py
- **功能**: 扩散训练相关方法
- **主要类**: `DiffusionTrainer`
- **方法**:
  - `play_one_game_for_diffusion()`: 生成扩散训练数据
  - `sample_action_from_diffusion()`: 从扩散策略采样动作
  - `demo_game_diffusion()`: 扩散策略演示对局

### 8. train.py
- **功能**: 主训练程序入口
- **流程**:
  1. RL策略训练阶段
  2. 扩散策略训练阶段
  3. 模型保存和日志记录

## 使用方法

### 1. 环境要求
```bash
pip install torch numpy tensorboard
```

### 2. 运行训练
```bash
python train.py
```

### 3. 测试重构代码
```bash
python test_refactored.py
```

### 4. 查看训练日志
```bash
tensorboard --logdir=logs/
```

## 主要改进

1. **模块化设计**: 将单文件拆分为功能明确的模块
2. **代码复用**: 减少重复代码，提高可维护性
3. **职责分离**: 每个模块专注于特定功能
4. **易于扩展**: 可以轻松添加新的算法或网络结构
5. **测试友好**: 模块化结构便于单元测试
6. **配置集中**: 所有参数在config.py中统一管理

## 训练流程

### 阶段1: RL策略训练
- 前30%局数: RL vs Random
- 后70%局数: RL vs RL (self-play)
- 当胜率达到阈值(0.7)时进入下一阶段

### 阶段2: 扩散策略训练
- 使用RL策略生成训练数据
- 基于经验回放训练扩散模型
- 生成高质量的棋步

## 输出文件

- **日志**: `./logs/run_[timestamp]/`
- **模型**: `./models/run_[timestamp]/`
- **最佳RL模型**: `best_rl_policy.pth`
- **最佳扩散模型**: `best_diff_policy.pth`

## 注意事项

1. 首次运行会自动创建logs和models目录
2. 训练过程会定期保存检查点
3. 可通过config.py调整所有超参数
4. 支持CPU和GPU训练（自动检测）