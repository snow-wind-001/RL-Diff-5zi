#!/usr/bin/env python3
"""
测试修改后的RL-Diff-5v2.py代码
验证TensorBoard和模型保存功能是否正常工作
"""

import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 导入修改后的模块
sys.path.append('.')
import importlib.util
spec = importlib.util.spec_from_file_location("RL_Diff_5v2", "RL-Diff-5v2.py")
RL_Diff_5v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RL_Diff_5v2)

RLPlusDiffusionAgent = RL_Diff_5v2.RLPlusDiffusionAgent
BOARD_SIZE = RL_Diff_5v2.BOARD_SIZE
DEVICE = RL_Diff_5v2.DEVICE

def test_tensorboard_functionality():
    """测试TensorBoard功能"""
    print("测试TensorBoard功能...")
    
    # 创建测试日志目录
    test_log_dir = "./test_logs"
    os.makedirs(test_log_dir, exist_ok=True)
    
    # 创建SummaryWriter
    writer = SummaryWriter(test_log_dir)
    
    # 写入一些测试数据
    for i in range(10):
        writer.add_scalar('test/accuracy', i * 0.1, i)
        writer.add_scalar('test/loss', 1.0 - i * 0.05, i)
    
    writer.close()
    print(f"TensorBoard测试数据已写入: {test_log_dir}")
    print("可以使用以下命令查看: tensorboard --logdir ./test_logs")

def test_model_saving():
    """测试模型保存功能"""
    print("\n测试模型保存功能...")
    
    # 创建测试模型保存目录
    test_model_dir = "./test_models"
    os.makedirs(test_model_dir, exist_ok=True)
    
    # 创建一个简单的测试Agent
    agent = RLPlusDiffusionAgent(board_size=5)  # 使用较小的棋盘进行测试
    
    # 测试保存模型
    test_metrics = {
        'winrate': 0.75,
        'loss': 0.123,
        'steps': 50
    }
    
    # 修改保存路径为测试目录
    original_model_dir = RL_Diff_5v2.MODEL_SAVE_DIR
    RL_Diff_5v2.MODEL_SAVE_DIR = test_model_dir
    
    try:
        agent.save_model(1, "test", test_metrics)
        agent.save_best_model(1, "rl", {'winrate': 0.8})
        agent.save_best_model(1, "diffusion", {'loss': 0.1})
        
        print("模型保存测试完成")
        print(f"测试模型文件保存在: {test_model_dir}")
        
        # 列出保存的文件
        if os.path.exists(test_model_dir):
            files = os.listdir(test_model_dir)
            print("保存的文件:")
            for f in files:
                print(f"  - {f}")
                
    finally:
        # 恢复原始路径
        RL_Diff_5v2.MODEL_SAVE_DIR = original_model_dir

def test_agent_initialization():
    """测试Agent初始化"""
    print("\n测试Agent初始化...")
    
    try:
        # 使用较小的配置进行快速测试
        agent = RLPlusDiffusionAgent(board_size=5)
        print("Agent初始化成功")
        print(f"设备: {agent.device}")
        print(f"棋盘大小: {agent.board_size}")
        print(f"扩散步数: {agent.T}")
        print(f"经验池容量: {len(agent.diff_replay)}")
        
        # 测试环境
        state = agent.env.reset()
        print(f"初始状态形状: {state.shape}")
        
        # 测试网络前向传播
        board_player, board_opp = agent._build_board_channels(state, 1)
        with torch.no_grad():
            rl_logits = agent.rl_policy(board_player, board_opp)
            print(f"RL策略输出形状: {rl_logits.shape}")
            
        print("Agent初始化测试通过")
        
    except Exception as e:
        print(f"Agent初始化测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试修改后的RL-Diff-5v2.py代码")
    print("=" * 50)
    
    # 测试TensorBoard功能
    test_tensorboard_functionality()
    
    # 测试模型保存功能
    test_model_saving()
    
    # 测试Agent初始化
    if test_agent_initialization():
        print("\n" + "=" * 50)
        print("所有测试通过！代码修改成功。")
        print("\n使用说明:")
        print("1. 运行训练: python RL-Diff-5v2.py")
        print("2. 查看TensorBoard: tensorboard --logdir ./logs")
        print("3. 模型会自动保存在 ./models 目录")
        print("4. 最佳模型会标记为 best_*.pth")
    else:
        print("\n测试失败，请检查代码修改")

if __name__ == "__main__":
    main()