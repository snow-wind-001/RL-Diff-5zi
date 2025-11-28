#!/usr/bin/env python3
"""
简化版五子棋对战系统（无GUI依赖）
支持人机对战和AI模型对战
"""

import sys
import os
import random
import numpy as np
from typing import Optional, Tuple, List

# 尝试导入本地模块
try:
    from config import BOARD_SIZE, LOG_DIR, MODEL_SAVE_DIR
    from environment import FiveChessEnv
    from agent import RLPlusDiffusionAgent
    from rl_trainer import RLTrainer
    from diffusion_trainer import DiffusionTrainer
    MODELS_AVAILABLE = True
except ImportError:
    print("⚠️  模型模块导入失败，将使用简化版本")
    BOARD_SIZE = 10
    MODELS_AVAILABLE = False


class SimpleChessBoard:
    """简单棋盘类"""

    def __init__(self, board_size=10):
        self.board_size = board_size
        self.reset_board()

    def reset_board(self):
        """重置棋盘"""
        self.chess_state = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.last_move = None

    def print_board(self):
        """打印棋盘"""
        print(f"\n{' ' * 4}", end="")
        for i in range(self.board_size):
            print(f" {chr(65+i)}", end="")
        print()

        for i in range(self.board_size):
            print(f" {i:2d} ", end="")
            for j in range(self.board_size):
                if self.chess_state[i, j] == 1:
                    print(" ●", end="")
                elif self.chess_state[i, j] == -1:
                    print(" ○", end="")
                else:
                    print(" ·", end="")
            print()

        print(f"{' ' * 4}", end="")
        for i in range(self.board_size):
            print(f" {chr(65+i)}", end="")
        print()

    def is_valid_move(self, row, col):
        """检查移动是否有效"""
        return (0 <= row < self.board_size and
                0 <= col < self.board_size and
                self.chess_state[row, col] == 0)

    def make_move(self, row, col):
        """落子"""
        if not self.is_valid_move(row, col):
            return False, "无效移动"

        self.chess_state[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_history.append((row, col, self.current_player))

        # 检查获胜
        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            return True, f"玩家{self.current_player}获胜！"

        # 检查平局
        if len(self.get_empty_positions()) == 0:
            self.game_over = True
            self.winner = None
            return True, "平局！"

        # 切换玩家
        self.current_player = -self.current_player
        return True, f"玩家{self.current_player}回合"

    def check_win(self, player):
        """检查玩家是否获胜"""
        board = self.chess_state
        M, N = self.board_size, self.board_size

        for i in range(M):
            for j in range(N):
                if board[i, j] != player:
                    continue

                # 横向
                if j + 4 < N and all(board[i, j+k] == player for k in range(5)):
                    return True

                # 纵向
                if i + 4 < M and all(board[i+k, j] == player for k in range(5)):
                    return True

                # 斜线
                if i + 4 < M and j + 4 < N:
                    if all(board[i+k, j+k] == player for k in range(5)):
                        return True

                # 反斜线
                if i + 4 < M and j - 4 >= 0:
                    if all(board[i+k, j-k] == player for k in range(5)):
                        return True

        return False

    def get_empty_positions(self):
        """获取空位"""
        positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.chess_state[i, j] == 0:
                    positions.append((i, j))
        return positions

    def get_state(self):
        """获取棋盘状态"""
        return self.chess_state.copy()


class AIPlayer:
    """AI玩家基类"""

    def __init__(self, name: str):
        self.name = name

    def get_move(self, board_state: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """获取AI的下一步动作"""
        raise NotImplementedError


class RandomAI(AIPlayer):
    """随机AI"""

    def __init__(self):
        super().__init__("随机AI")

    def get_move(self, board_state: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """随机选择空位"""
        empty_positions = []
        for i in range(board_state.shape[0]):
            for j in range(board_state.shape[1]):
                if board_state[i, j] == 0:
                    empty_positions.append((i, j))

        if empty_positions:
            return random.choice(empty_positions)
        return None


class ConsolePlayer(AIPlayer):
    """控制台玩家（人类）"""

    def __init__(self, name: str):
        super().__init__(name)

    def get_move(self, board_state: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """从控制台获取人类输入"""
        while True:
            try:
                user_input = input(f"请输入位置 (例如: B5 或 1,4): ").strip().upper()

                # 解析输入
                if ',' in user_input:
                    # 格式: "1,4"
                    row, col = map(int, user_input.split(','))
                else:
                    # 格式: "B5"
                    if len(user_input) >= 2:
                        col = ord(user_input[0]) - ord('A')
                        row = int(user_input[1:])
                    else:
                        print("输入格式错误，请重试")
                        continue

                # 检查有效性
                if (0 <= row < board_state.shape[0] and
                    0 <= col < board_state.shape[1] and
                    board_state[row, col] == 0):
                    return (row, col)
                else:
                    print("该位置无效或已被占用，请重试")

            except (ValueError, IndexError):
                print("输入格式错误，请重试")
                continue


class SimpleBattleSystem:
    """简化版对战系统"""

    def __init__(self):
        self.board = SimpleChessBoard(BOARD_SIZE)
        self.players = {}

    def setup_players(self):
        """设置玩家"""
        print("\n=== 五子棋对战系统 ===\n")

        print("选择对战模式：")
        print("1. 人机对战")
        print("2. AI对战")
        print("3. 人人对战")

        while True:
            try:
                mode = int(input("请选择模式 (1-3): "))
                if 1 <= mode <= 3:
                    break
                print("请输入1-3之间的数字")
            except ValueError:
                print("请输入有效的数字")

        print("\n选择先手 (1 或 2): ", end="")
        while True:
            try:
                first_player = int(input())
                if first_player in [1, 2]:
                    break
                print("请输入1或2")
            except ValueError:
                print("请输入有效的数字")

        # 设置玩家
        if mode == 1:  # 人机对战
            if first_player == 1:
                self.players[1] = ConsolePlayer("人类玩家1")
                self.players[-1] = self.select_ai()
            else:
                self.players[1] = self.select_ai()
                self.players[-1] = ConsolePlayer("人类玩家2")

        elif mode == 2:  # AI对战
            self.players[1] = self.select_ai("先手AI")
            self.players[-1] = self.select_ai("后手AI")

        else:  # 人人对战
            self.players[1] = ConsolePlayer("人类玩家1")
            self.players[-1] = ConsolePlayer("人类玩家2")

        print(f"\n对战设置完成：")
        print(f"玩家1 (先手 ●): {self.players[1].name}")
        print(f"玩家2 (后手 ○): {self.players[-1].name}")

    def select_ai(self, default_name="AI玩家") -> AIPlayer:
        """选择AI类型"""
        print(f"\n选择{default_name}类型：")
        print("1. 随机AI")
        if MODELS_AVAILABLE:
            print("2. RL AI (强化学习)")
            print("3. 扩散AI")
        else:
            print("2. RL AI (强化学习) [不可用]")
            print("3. 扩散AI [不可用]")

        while True:
            try:
                ai_type = int(input(f"请选择AI类型 ({'1-3' if MODELS_AVAILABLE else '1'}): "))
                if ai_type == 1:
                    return RandomAI()
                elif MODELS_AVAILABLE and ai_type == 2:
                    return self.create_rl_ai()
                elif MODELS_AVAILABLE and ai_type == 3:
                    return self.create_diffusion_ai()
                else:
                    print("无效选择")
            except ValueError:
                print("请输入有效的数字")

    def create_rl_ai(self) -> AIPlayer:
        """创建RL AI"""
        print("查找可用的RL模型...")

        if not MODELS_AVAILABLE:
            print("RL模型不可用，使用随机AI")
            return RandomAI()

        # 查找最佳RL模型
        best_model = None
        best_run = None

        if os.path.exists(MODEL_SAVE_DIR):
            for run_dir in os.listdir(MODEL_SAVE_DIR):
                run_path = os.path.join(MODEL_SAVE_DIR, run_dir)
                if os.path.isdir(run_path):
                    rl_model = os.path.join(run_path, "best_rl_policy.pth")
                    if os.path.exists(rl_model):
                        best_model = rl_model
                        best_run = run_dir
                        break

        if best_model:
            print(f"找到RL模型: {best_run}")
            try:
                from networks import RLPolicyNet
                import torch

                model = RLPolicyNet(BOARD_SIZE)
                model.load_state_dict(torch.load(best_model, map_location='cpu'))
                model.eval()

                class CustomRLAI(AIPlayer):
                    def __init__(self, model, name="RL AI"):
                        super().__init__(name)
                        self.model = model

                    def get_move(self, board_state: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
                        import torch.nn.functional as F

                        # 构建输入特征
                        board_tensor = torch.from_numpy(board_state).float().view(1, 1, BOARD_SIZE, BOARD_SIZE)
                        player_tensor = torch.tensor([[player]], dtype=torch.float32).view(1, 1, 1, 1)

                        board_player = (board_tensor == player_tensor).float()
                        board_opp = (board_tensor == -player_tensor).float()

                        with torch.no_grad():
                            logits = self.model(board_player, board_opp)
                            logits = logits.view(-1).cpu().numpy()

                        # 选择最佳空位
                        empty_positions = []
                        for i in range(BOARD_SIZE):
                            for j in range(BOARD_SIZE):
                                if board_state[i, j] == 0:
                                    empty_positions.append((i, j, logits[i * BOARD_SIZE + j]))

                        if empty_positions:
                            empty_positions.sort(key=lambda x: x[2], reverse=True)
                            return empty_positions[0][:2]
                        return None

                return CustomRLAI(model, f"RL AI ({best_run})")

            except Exception as e:
                print(f"RL模型加载失败: {e}，使用随机AI")
                return RandomAI()
        else:
            print("未找到RL模型，使用随机AI")
            return RandomAI()

    def create_diffusion_ai(self) -> AIPlayer:
        """创建扩散AI"""
        print("查找可用的扩散模型...")

        if not MODELS_AVAILABLE:
            print("扩散模型不可用，使用随机AI")
            return RandomAI()

        # 查找最佳扩散模型
        best_model = None
        best_run = None

        if os.path.exists(MODEL_SAVE_DIR):
            for run_dir in os.listdir(MODEL_SAVE_DIR):
                run_path = os.path.join(MODEL_SAVE_DIR, run_dir)
                if os.path.isdir(run_path):
                    diff_model = os.path.join(run_path, "best_diff_policy.pth")
                    if os.path.exists(diff_model):
                        best_model = diff_model
                        best_run = run_dir
                        break

        if best_model:
            print(f"找到扩散模型: {best_run}")
            # 简化版扩散AI，实际应该实现完整的扩散采样
            return RandomAI()  # 暂时用随机AI代替
        else:
            print("未找到扩散模型，使用随机AI")
            return RandomAI()

    def play_game(self):
        """进行游戏"""
        print("\n=== 游戏开始 ===\n")

        self.board.reset_board()
        self.board.print_board()

        while not self.board.game_over:
            current_player = self.board.current_player
            current_player_name = self.players[current_player].name
            player_symbol = "●" if current_player == 1 else "○"

            print(f"\n{current_player_name} ({player_symbol}) 的回合:")

            if isinstance(self.players[current_player], ConsolePlayer):
                # 人类玩家
                self.board.print_board()
                move = self.players[current_player].get_move(self.board.get_state(), current_player)
            else:
                # AI玩家
                print(f"{current_player_name}正在思考...")
                move = self.players[current_player].get_move(self.board.get_state(), current_player)

            if move:
                row, col = move
                success, message = self.board.make_move(row, col)
                print(f"{current_player_name} {player_symbol}: {chr(65+col)}{row} -> {message}")

                if self.board.last_move:
                    print(f"最后落子: {chr(65+self.board.last_move[1])}{self.board.last_move[0]}")

                self.board.print_board()
            else:
                print("无效移动，请重试")

        # 游戏结束
        self.show_game_result()

    def show_game_result(self):
        """显示游戏结果"""
        print("\n" + "="*30)
        if self.board.winner:
            winner_name = self.players[self.board.winner].name
            winner_symbol = "●" if self.board.winner == 1 else "○"
            print(f"🎉 {winner_name} ({winner_symbol}) 获胜！")
        else:
            print("🤝 游戏平局！")

        print(f"总回合数: {len(self.board.move_history)}")
        print("="*30)

    def run(self):
        """运行对战系统"""
        print("欢迎使用五子棋对战系统！")

        while True:
            try:
                self.setup_players()
                self.play_game()

                # 询问是否再来一局
                choice = input("\n是否再来一局？(y/n): ").strip().lower()
                if choice != 'y' and choice != 'yes':
                    print("感谢使用五子棋对战系统，再见！")
                    break

            except KeyboardInterrupt:
                print("\n\n游戏被中断，感谢使用！")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                print("请重新开始...")


def main():
    """主函数"""
    try:
        system = SimpleBattleSystem()
        system.run()
    except Exception as e:
        print(f"程序启动失败: {e}")
        print("请检查依赖是否正确安装")


if __name__ == "__main__":
    main()