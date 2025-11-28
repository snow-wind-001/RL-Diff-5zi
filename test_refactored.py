#!/usr/bin/env python3

import sys
import traceback

def test_imports():
    """测试所有模块的导入"""
    try:
        print("Testing imports...")

        # Test config
        from config import (
            BOARD_SIZE, DEVICE, RL_MAX_EPISODES, RL_GAMMA,
            DIFFUSION_STEPS, LOG_DIR, MODEL_SAVE_DIR
        )
        print(f"✓ Config imported: BOARD_SIZE={BOARD_SIZE}, DEVICE={DEVICE}")

        # Test environment
        from environment import FiveChessEnv
        env = FiveChessEnv()
        state = env.reset()
        print(f"✓ Environment imported and working: state shape={state.shape}")

        # Test networks
        from networks import RLPolicyNet, UNetPolicy
        rl_net = RLPolicyNet()
        unet = UNetPolicy()
        print(f"✓ Networks imported successfully")

        # Test replay buffer
        from replay_buffer import DiffReplayBuffer
        buffer = DiffReplayBuffer(100)
        buffer.add(state, (0, 0), 1, 0.5)
        print(f"✓ Replay buffer working: buffer size={len(buffer)}")

        # Test agent
        from agent import RLPlusDiffusionAgent
        agent = RLPlusDiffusionAgent()
        print(f"✓ Agent initialized successfully")

        # Test trainers
        from rl_trainer import RLTrainer
        from diffusion_trainer import DiffusionTrainer
        rl_trainer = RLTrainer(agent)
        diff_trainer = DiffusionTrainer(agent)
        print(f"✓ Trainers imported and initialized")

        # Test train module
        from train import main
        print(f"✓ Train module imported (main function available)")

        print("\n🎉 All imports successful! The refactored code structure is working.")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        print("\nTesting basic functionality...")

        from config import BOARD_SIZE
        from environment import FiveChessEnv
        from agent import RLPlusDiffusionAgent

        # Create environment and agent
        env = FiveChessEnv(BOARD_SIZE)
        agent = RLPlusDiffusionAgent(BOARD_SIZE, device='cpu')  # Use CPU for testing

        # Test basic RL game
        from rl_trainer import RLTrainer
        rl_trainer = RLTrainer(agent)

        # Play a quick RL game
        winner, steps = rl_trainer.play_one_rl_game_and_update(mode="vs_random", epsilon=0.1)
        print(f"✓ RL game completed: winner={winner}, steps={steps}")

        # Test diffusion replay buffer
        if len(agent.diff_replay) > 0:
            print(f"✓ Diffusion replay buffer has data: {len(agent.diff_replay)} entries")

        print("\n✅ Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("REFACTORED CODE STRUCTURE TEST")
    print("=" * 60)

    # Test imports
    import_success = test_imports()

    if import_success:
        # Test basic functionality
        func_success = test_basic_functionality()

        if func_success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("The refactored code structure is ready for use.")
            print("Run 'python train.py' to start training.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  IMPORTS OK BUT FUNCTIONALITY ISSUES")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("❌ IMPORT FAILURES - PLEASE FIX BEFORE USING")
        print("=" * 60)
        sys.exit(1)