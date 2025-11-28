import os
import numpy as np

from config import (
    BOARD_SIZE, DEVICE, RL_MAX_EPISODES, RL_EVAL_INTERVAL, RL_DEMO_INTERVAL,
    RL_WINRATE_THRESHOLD, RL_RANDOM_RATIO, DIFF_MAX_EPISODES, DIFF_BATCH_SIZE,
    DIFF_DEMO_INTERVAL, DIFF_TRAIN_STEPS_PER_EP, SAVE_INTERVAL, LOG_DIR,
    MODEL_SAVE_DIR
)
from agent import RLPlusDiffusionAgent
from rl_trainer import RLTrainer
from diffusion_trainer import DiffusionTrainer


def main():
    """主训练函数"""
    print(f"Using device: {DEVICE}")
    print(f"日志目录: {LOG_DIR}")
    print(f"模型保存目录: {MODEL_SAVE_DIR}")

    # 初始化agent和trainer
    agent = RLPlusDiffusionAgent(board_size=BOARD_SIZE, device=DEVICE)
    rl_trainer = RLTrainer(agent)
    diffusion_trainer = DiffusionTrainer(agent)

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

        winner, steps = rl_trainer.play_one_rl_game_and_update(mode=mode)

        # 统计
        rl_total_steps += steps
        if winner == 1:
            rl_win_count += 1
        elif winner == 0:
            rl_draw_count += 1

        # 定期记录指标和保存模型
        if ep % RL_EVAL_INTERVAL == 0:
            winrate = rl_trainer.evaluate_rl_vs_random()

            # 记录指标
            metrics = {
                'winrate': winrate,
                'win_count': rl_win_count,
                'draw_count': rl_draw_count,
                'avg_steps': rl_total_steps / ep,
                'replay_size': len(agent.diff_replay)
            }
            agent.log_metrics(ep, "rl", metrics)

            # 保存最佳模型
            agent.save_best_model(ep, "rl", {'winrate': winrate})

            # 定期保存检查点
            if ep % SAVE_INTERVAL == 0:
                agent.save_model(ep, "rl", metrics)

            if winrate >= RL_WINRATE_THRESHOLD:
                print(f"RL 胜率达到阈值 {RL_WINRATE_THRESHOLD:.2f}，开始扩散模型训练阶段。")
                start_diffusion = True
                break

        if ep % RL_DEMO_INTERVAL == 0:
            print(f"\n[RL] Episode {ep}, mode={mode}, last winner={winner}, steps={steps}, "
                  f"diff_replay_size={len(agent.diff_replay)}")
            rl_trainer.demo_game_rl()

    if not start_diffusion:
        print("RL 训练达到最大轮次，仍未达到阈值，依然开始扩散训练（可能策略较弱）。")
        # 保存最终RL模型
        agent.save_model(RL_MAX_EPISODES, "rl_final", {
            'winrate': rl_win_count / RL_MAX_EPISODES,
            'win_count': rl_win_count,
            'draw_count': rl_draw_count
        })

    print("\n========== 阶段 2：扩散策略训练 ==========")

    # 扩散数据预热：经验池太小时，先用 RL 多生成几局
    while len(agent.diff_replay) < DIFF_BATCH_SIZE:
        winner, steps = diffusion_trainer.play_one_game_for_diffusion()
        print(f"预热扩散数据：winner={winner}, steps={steps}, replay_size={len(agent.diff_replay)}")

    # 扩散训练统计
    diff_win_count = 0
    diff_draw_count = 0
    diff_total_steps = 0
    diff_losses = []

    for ep in range(1, DIFF_MAX_EPISODES + 1):
        winner, steps = diffusion_trainer.play_one_game_for_diffusion()

        # 统计
        diff_total_steps += steps
        if winner == 1:
            diff_win_count += 1
        elif winner == 0:
            diff_draw_count += 1

        # 执行扩散训练步骤
        episode_losses = []
        for _ in range(DIFF_TRAIN_STEPS_PER_EP):
            if len(agent.diff_replay) >= DIFF_BATCH_SIZE:
                (batch_states,
                 batch_actions,
                 batch_players,
                 batch_advantages) = agent.diff_replay.sample(DIFF_BATCH_SIZE)
                loss = agent.diffusion_train_step(batch_states,
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
                'replay_size': len(agent.diff_replay),
                'avg_loss_100': np.mean(diff_losses[-100:]) if len(diff_losses) >= 100 else np.mean(diff_losses)
            }
            agent.log_metrics(ep, "diffusion", metrics)

            # 保存最佳模型
            agent.save_best_model(ep, "diffusion", {'loss': avg_loss})

            # 保存检查点
            agent.save_model(ep, "diffusion", metrics)

        if ep % DIFF_DEMO_INTERVAL == 0:
            print(f"\n[Diffusion] Episode {ep}, last winner={winner}, steps={steps}, "
                  f"replay_size={len(agent.diff_replay)}"
                  + (f", loss={avg_loss:.4f}" if avg_loss is not None else ""))
            diffusion_trainer.demo_game_diffusion()

    # 保存最终扩散模型
    agent.save_model(DIFF_MAX_EPISODES, "diffusion_final", {
        'loss': np.mean(diff_losses) if diff_losses else 0.0,
        'win_count': diff_win_count,
        'draw_count': diff_draw_count
    })

    # 关闭资源
    agent.close()
    print(f"训练完成！日志保存在: {LOG_DIR}")
    print(f"模型保存在: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()