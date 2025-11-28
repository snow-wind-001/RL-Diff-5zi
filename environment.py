import numpy as np
from config import BOARD_SIZE


class FiveChessEnv:
    """五子棋游戏环境"""

    def __init__(self, board_size=BOARD_SIZE):
        self.M = board_size
        self.N = board_size
        # 空:0  先手:1  后手:-1
        self.chessState = np.zeros((self.M, self.N), dtype=np.int8)

    def reset(self):
        """重置棋盘状态"""
        self.chessState.fill(0)
        return self.get_state()

    def get_state(self):
        """获取当前棋盘状态（扁平化）"""
        return self.chessState.astype(np.float32).flatten()

    def step(self, action, player):
        """执行一步动作"""
        i, j = action
        if self.chessState[i, j] != 0:
            raise ValueError(f"Invalid move at ({i}, {j})!")
        self.chessState[i, j] = player
        done, win_type = self.check_win(player)
        return self.get_state(), done, win_type

    def get_empty_positions(self):
        """获取所有空位置"""
        empties = np.argwhere(self.chessState == 0)
        return [tuple(pos) for pos in empties]

    def check_win(self, who):
        """
        检查当前棋盘上是否存在玩家 who (1 或 -1) 连续 5 个棋子的情况。
        四个方向全部包含：横、竖、斜线、反斜线。
        """
        M, N = self.M, self.N
        board = self.chessState

        for i in range(M):
            for j in range(N):
                if board[i, j] != who:
                    continue

                # 横向 —
                if j + 4 < N and np.all(board[i, j:j+5] == who):
                    return True, "—"

                # 纵向 |
                if i + 4 < M and np.all(board[i:i+5, j] == who):
                    return True, "|"

                # 正斜线 \
                if i + 4 < M and j + 4 < N:
                    ok = True
                    for k in range(5):
                        if board[i + k, j + k] != who:
                            ok = False
                            break
                    if ok:
                        return True, "\\"

                # 反斜线 /
                if i + 4 < M and j - 4 >= 0:
                    ok = True
                    for k in range(5):
                        if board[i + k, j - k] != who:
                            ok = False
                            break
                    if ok:
                        return True, "/"

        return False, ""

    def print_board(self):
        """打印当前棋盘状态"""
        symbols = {1: "●", -1: "○", 0: " ."}
        for i in range(self.M):
            row = "".join(symbols[int(x)] for x in self.chessState[i])
            print(row)
        print()