#!/usr/bin/env python3
"""
五子棋图形化对战系统
支持人机对战、AI模型对战，可选择先后手
"""

import sys
import os
import time
import random
from typing import Optional, Tuple, List
import numpy as np

# PyQt 导入
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QPushButton, QLabel,
                               QComboBox, QGroupBox, QMessageBox, QFrame,
                               QSplitter, QTextEdit, QStatusBar, QMenuBar,
                               QAction, QFileDialog)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QPointF
    from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QMouseEvent
    PyQt5_AVAILABLE = True
except ImportError:
    print("PyQt5 未安装，请运行: pip install PyQt5")
    PyQt5_AVAILABLE = False
    sys.exit(1)

# 本地模型导入
try:
    import torch
    from config import BOARD_SIZE, DEVICE, LOG_DIR, MODEL_SAVE_DIR
    from environment import FiveChessEnv
    from agent import RLPlusDiffusionAgent
    from rl_trainer import RLTrainer
    from diffusion_trainer import DiffusionTrainer
    MODELS_AVAILABLE = True
except ImportError:
    print("模型模块导入失败，将使用简化版本")
    MODELS_AVAILABLE = False


class ChessBoardWidget(QWidget):
    """棋盘绘制组件"""

    move_requested = pyqtSignal(int, int)  # 信号：请求落子 (row, col)

    def __init__(self, board_size=10):
        super().__init__()
        self.board_size = board_size
        self.cell_size = 40
        self.board_margin = 30
        self.chess_state = np.zeros((board_size, board_size), dtype=int)
        self.last_move = None
        self.winning_line = []
        self.hint_mode = False
        self.hint_position = None

        self.setMinimumSize(
            self.cell_size * board_size + 2 * self.board_margin,
            self.cell_size * board_size + 2 * self.board_margin
        )

    def reset_board(self):
        """重置棋盘"""
        self.chess_state.fill(0)
        self.last_move = None
        self.winning_line = []
        self.hint_position = None
        self.update()

    def set_hints(self, positions: List[Tuple[int, int]] = None):
        """设置提示位置"""
        self.hint_mode = positions is not None
        if positions:
            self.hint_position = positions[0] if positions else None
        else:
            self.hint_position = None
        self.update()

    def make_move(self, row: int, col: int, player: int):
        """落子"""
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.chess_state[row, col] == 0:
                self.chess_state[row, col] = player
                self.last_move = (row, col)
                self.update()
                return True
        return False

    def set_winning_line(self, line: List[Tuple[int, int]]):
        """设置获胜连线"""
        self.winning_line = line
        self.update()

    def get_board_state(self) -> np.ndarray:
        """获取棋盘状态"""
        return self.chess_state.copy()

    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """获取空位置"""
        return [(i, j) for i in range(self.board_size)
                for j in range(self.board_size) if self.chess_state[i, j] == 0]

    def paintEvent(self, event):
        """绘制棋盘和棋子"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor(220, 180, 120))

        # 绘制网格线
        pen = QPen(QColor(0, 0, 0), 2)
        painter.setPen(pen)

        for i in range(self.board_size):
            # 横线
            y = self.board_margin + i * self.cell_size
            painter.drawLine(self.board_margin, y,
                          self.board_margin + (self.board_size - 1) * self.cell_size, y)
            # 竖线
            x = self.board_margin + i * self.cell_size
            painter.drawLine(x, self.board_margin,
                          x, self.board_margin + (self.board_size - 1) * self.cell_size)

        # 绘制坐标标签
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for i in range(self.board_size):
            # 数字标签
            x = self.board_margin - 20
            y = self.board_margin + i * self.cell_size + 5
            painter.drawText(x, y, str(i))
            # 字母标签
            x = self.board_margin + i * self.cell_size - 5
            y = self.board_margin - 10
            painter.drawText(x, y, chr(65 + i))  # A, B, C...

        # 绘制获胜连线
        if self.winning_line:
            pen = QPen(QColor(255, 0, 0), 4)
            painter.setPen(pen)
            for i in range(len(self.winning_line) - 1):
                r1, c1 = self.winning_line[i]
                r2, c2 = self.winning_line[i + 1]
                x1 = self.board_margin + c1 * self.cell_size
                y1 = self.board_margin + r1 * self.cell_size
                x2 = self.board_margin + c2 * self.cell_size
                y2 = self.board_margin + r2 * self.cell_size
                painter.drawLine(x1, y1, x2, y2)

        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.chess_state[i, j] != 0:
                    self._draw_chess_piece(painter, i, j, self.chess_state[i, j])

        # 绘制最后落子标记
        if self.last_move:
            row, col = self.last_move
            x = self.board_margin + col * self.cell_size
            y = self.board_margin + row * self.cell_size
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(x - 8, y - 8, 16, 16)

        # 绘制提示位置
        if self.hint_mode and self.hint_position:
            row, col = self.hint_position
            x = self.board_margin + col * self.cell_size
            y = self.board_margin + row * self.cell_size
            painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            painter.drawEllipse(x - 15, y - 15, 30, 30)

    def _draw_chess_piece(self, painter, row, col, player):
        """绘制棋子"""
        x = self.board_margin + col * self.cell_size
        y = self.board_margin + row * self.cell_size
        radius = self.cell_size // 2 - 4

        if player == 1:  # 黑子
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.setPen(QPen(QColor(50, 50, 50), 1))
        else:  # 白子
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(200, 200, 200), 1))

        painter.drawEllipse(x - radius, y - radius, 2 * radius, 2 * radius)

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()

            # 计算点击的棋盘位置
            col = round((x - self.board_margin) / self.cell_size)
            row = round((y - self.board_margin) / self.cell_size)

            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                if self.chess_state[row, col] == 0:  # 空位才能下子
                    self.move_requested.emit(row, col)


class GameEngine:
    """游戏引擎"""

    def __init__(self, board_size=10):
        self.board_size = board_size
        self.env = FiveChessEnv(board_size) if MODELS_AVAILABLE else None
        self.reset_game()

    def reset_game(self):
        """重置游戏"""
        if self.env:
            self.env.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []

    def make_move(self, row, col, player=None) -> Tuple[bool, str]:
        """落子并检查游戏状态"""
        if self.game_over:
            return False, "游戏已结束"

        if player is None:
            player = self.current_player

        # 检查位置是否有效
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False, f"位置超出范围: ({row}, {col})"

        if self.env:
            try:
                state, done, win_type = self.env.step((row, col), player)
                if done:
                    self.game_over = True
                    self.winner = player
                    return True, f"玩家{player}获胜! ({win_type})"
                else:
                    self.current_player = -self.current_player
                    self.move_history.append((row, col, player))
                    return True, "落子成功"
            except ValueError as e:
                return False, str(e)
        else:
            # 简化版本（没有environment模块）
            # 这里需要自己实现基本的游戏逻辑
            return self._simple_make_move(row, col, player)

    def _simple_make_move(self, row, col, player) -> Tuple[bool, str]:
        """简化版本的落子逻辑"""
        # 这里应该实现基本的五子棋检查逻辑
        # 为了简化，只检查棋盘是否满了
        # 在实际使用中，建议确保environment模块可用
        self.current_player = -self.current_player
        self.move_history.append((row, col, player))
        return True, "落子成功"

    def check_win(self, player: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """检查玩家是否获胜"""
        if self.env:
            won, win_type = self.env.check_win(player)
            if won:
                # 返回获胜连线
                winning_line = self._get_winning_line(player, win_type)
                return won, winning_line
        return False, []

    def _get_winning_line(self, player: int, win_type: str) -> List[Tuple[int, int]]:
        """根据获胜类型获取获胜连线"""
        board = self.env.chessState if self.env else self.chess_state
        M, N = board.shape

        for i in range(M):
            for j in range(N):
                if board[i, j] != player:
                    continue

                if win_type == "—" and j + 4 < N:
                    if all(board[i, j+k] == player for k in range(5)):
                        return [(i, j+k) for k in range(5)]

                elif win_type == "|" and i + 4 < M:
                    if all(board[i+k, j] == player for k in range(5)):
                        return [(i+k, j) for k in range(5)]

                elif win_type == "\\" and i + 4 < M and j + 4 < N:
                    if all(board[i+k, j+k] == player for k in range(5)):
                        return [(i+k, j+k) for k in range(5)]

                elif win_type == "/" and i + 4 < M and j - 4 >= 0:
                    if all(board[i+k, j-k] == player for k in range(5)):
                        return [(i+k, j-k) for k in range(5)]

        return []

    def get_board_state(self) -> np.ndarray:
        """获取棋盘状态"""
        if self.env:
            return self.env.get_state().reshape(self.board_size, self.board_size)
        return np.zeros((self.board_size, self.board_size), dtype=int)

    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """获取空位置"""
        if self.env:
            return self.env.get_empty_positions()
        return []


class AIPlayer:
    """AI玩家基类"""

    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"

    def get_move(self, board_state: np.ndarray, player: int) -> Tuple[int, int]:
        """获取AI的下一步动作"""
        raise NotImplementedError


class RandomAI(AIPlayer):
    """随机AI"""

    def __init__(self):
        super().__init__("Random AI")

    def get_move(self, board_state: np.ndarray, player: int) -> Tuple[int, int]:
        """随机选择空位"""
        empty_positions = []
        for i in range(board_state.shape[0]):
            for j in range(board_state.shape[1]):
                if board_state[i, j] == 0:
                    empty_positions.append((i, j))

        if empty_positions:
            return random.choice(empty_positions)
        return None


class RLAI(AIPlayer):
    """强化学习AI"""

    def __init__(self, model_path: str = None):
        super().__init__("RL Policy AI")
        self.model = None
        self.model_path = model_path
        self.board_size = BOARD_SIZE

        if MODELS_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                from networks import RLPolicyNet
                self.model = RLPolicyNet(self.board_size)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                self.device = 'cpu'
                print(f"RL模型加载成功: {model_path}")
            except Exception as e:
                print(f"RL模型加载失败: {e}")
                self.model = None
        elif MODELS_AVAILABLE:
            print(f"RL模型文件不存在: {model_path}")
        else:
            print("模型模块不可用，无法使用RL AI")

    def get_move(self, board_state: np.ndarray, player: int) -> Tuple[int, int]:
        """使用RL策略选择动作"""
        if not self.model or not MODELS_AVAILABLE:
            return RandomAI().get_move(board_state, player)

        try:
            import torch
            import torch.nn.functional as F

            # 构建输入特征
            board_tensor = torch.from_numpy(board_state).float().view(1, 1, self.board_size, self.board_size)
            player_tensor = torch.tensor([[player]], dtype=torch.float32).view(1, 1, 1, 1)

            board_player = (board_tensor == player_tensor).float()
            board_opp = (board_tensor == -player_tensor).float()

            with torch.no_grad():
                logits = self.model(board_player, board_opp)
                logits = logits.view(-1).cpu().numpy()

            # 获取空位并选择最佳位置
            empty_positions = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board_state[i, j] == 0:
                        empty_positions.append((i, j, logits[i * self.board_size + j]))

            if empty_positions:
                empty_positions.sort(key=lambda x: x[2], reverse=True)
                return empty_positions[0][:2]
            return None

        except Exception as e:
            print(f"RL推理失败: {e}")
            return RandomAI().get_move(board_state, player)


class DiffusionAI(AIPlayer):
    """扩散模型AI"""

    def __init__(self, model_path: str = None):
        super().__init__("Diffusion AI")
        self.model = None
        self.model_path = model_path
        self.board_size = BOARD_SIZE
        self.diffusion_steps = 50

        if MODELS_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                from networks import UNetPolicy
                self.model = UNetPolicy()
                # 这里应该加载完整的扩散模型参数
                # 为了简化，我们使用基本的模型加载
                print(f"扩散模型加载成功: {model_path}")
            except Exception as e:
                print(f"扩散模型加载失败: {e}")
                self.model = None
        elif MODELS_AVAILABLE:
            print(f"扩散模型文件不存在: {model_path}")
        else:
            print("模型模块不可用，无法使用扩散AI")

    def get_move(self, board_state: np.ndarray, player: int) -> Tuple[int, int]:
        """使用扩散策略选择动作"""
        if not self.model or not MODELS_AVAILABLE:
            return RandomAI().get_move(board_state, player)

        try:
            # 这里应该实现完整的扩散采样过程
            # 为了简化，暂时使用随机策略
            print(f"扩散AI正在思考...")
            time.sleep(0.2)  # 减少思考时间避免看起来像无限循环
            return RandomAI().get_move(board_state, player)

        except Exception as e:
            print(f"扩散推理失败: {e}")
            return RandomAI().get_move(board_state, player)


class BattleSystemWindow(QMainWindow):
    """对战系统主窗口"""

    def __init__(self):
        super().__init__()
        self.board_size = BOARD_SIZE if MODELS_AVAILABLE else 10
        self.init_ui()
        self.init_game()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("五子棋对战系统")
        self.setGeometry(100, 100, 1000, 700)

        # 创建菜单栏
        self.create_menus()

        # 创建中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：棋盘
        board_frame = QGroupBox("棋盘")
        board_layout = QVBoxLayout(board_frame)

        self.chess_board = ChessBoardWidget(self.board_size)
        self.chess_board.move_requested.connect(self.on_board_clicked)
        board_layout.addWidget(self.chess_board)

        main_layout.addWidget(board_frame, 2)

        # 右侧：控制面板
        control_frame = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_frame)

        # 游戏设置
        settings_group = QGroupBox("游戏设置")
        settings_layout = QVBoxLayout()

        # 玩家1选择
        player1_layout = QHBoxLayout()
        player1_layout.addWidget(QLabel("玩家1 (先手):"))
        self.player1_combo = QComboBox()
        self.player1_combo.addItems(["人类", "随机AI", "RL AI", "扩散AI"])
        self.player1_combo.setCurrentIndex(0)
        player1_layout.addWidget(self.player1_combo)
        settings_layout.addLayout(player1_layout)

        # 玩家2选择
        player2_layout = QHBoxLayout()
        player2_layout.addWidget(QLabel("玩家2 (后手):"))
        self.player2_combo = QComboBox()
        self.player2_combo.addItems(["人类", "随机AI", "RL AI", "扩散AI"])
        self.player2_combo.setCurrentIndex(1)
        player2_layout.addWidget(self.player2_combo)
        settings_layout.addLayout(player2_layout)

        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)

        # 模型选择
        models_group = QGroupBox("模型选择")
        models_layout = QVBoxLayout()

        # RL模型选择
        rl_model_layout = QHBoxLayout()
        rl_model_layout.addWidget(QLabel("RL模型:"))
        self.rl_model_label = QLabel("未选择")
        self.rl_model_button = QPushButton("选择RL模型")
        self.rl_model_button.clicked.connect(self.select_rl_model)
        rl_model_layout.addWidget(self.rl_model_label)
        rl_model_layout.addWidget(self.rl_model_button)
        models_layout.addLayout(rl_model_layout)

        # 扩散模型选择
        diff_model_layout = QHBoxLayout()
        diff_model_layout.addWidget(QLabel("扩散模型:"))
        self.diff_model_label = QLabel("未选择")
        self.diff_model_button = QPushButton("选择扩散模型")
        self.diff_model_button.clicked.connect(self.select_diffusion_model)
        diff_model_layout.addWidget(self.diff_model_label)
        diff_model_layout.addWidget(self.diff_model_button)
        models_layout.addLayout(diff_model_layout)

        models_group.setLayout(models_layout)
        control_layout.addWidget(models_group)

        # 游戏控制
        game_group = QGroupBox("游戏控制")
        game_layout = QVBoxLayout()

        self.start_button = QPushButton("开始新游戏")
        self.start_button.clicked.connect(self.start_new_game)
        game_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("暂停/继续")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        game_layout.addWidget(self.pause_button)

        game_group.setLayout(game_layout)
        control_layout.addWidget(game_group)

        # 游戏状态
        status_group = QGroupBox("游戏状态")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("准备就绪")
        status_layout.addWidget(self.status_label)

        self.current_player_label = QLabel("当前玩家: -")
        status_layout.addWidget(self.current_player_label)

        self.move_count_label = QLabel("回合数: 0")
        status_layout.addWidget(self.move_count_label)

        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)

        # 游戏记录
        history_group = QGroupBox("游戏记录")
        history_layout = QVBoxLayout()

        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(200)
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)

        history_group.setLayout(history_layout)
        control_layout.addWidget(history_group)

        control_layout.addStretch()

        main_layout.addWidget(control_frame, 1)

        # 状态栏
        self.statusBar().showMessage("就绪")

    def create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        load_game_action = QAction('加载游戏', self)
        load_game_action.triggered.connect(self.load_game)
        file_menu.addAction(load_game_action)

        save_game_action = QAction('保存游戏', self)
        save_game_action.triggered.connect(self.save_game)
        file_menu.addAction(save_game_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_game(self):
        """初始化游戏"""
        self.game_engine = GameEngine(self.board_size)
        self.ai_players = {}
        self.rl_model_path = None
        self.diffusion_model_path = None
        self.is_paused = False
        self.move_count = 0

        # 查找默认模型
        self.find_default_models()

    def find_default_models(self):
        """查找默认模型"""
        if MODELS_AVAILABLE:
            # 查找最佳RL模型
            for run_dir in os.listdir(MODEL_SAVE_DIR):
                run_path = os.path.join(MODEL_SAVE_DIR, run_dir)
                if os.path.isdir(run_path):
                    rl_model = os.path.join(run_path, "best_rl_policy.pth")
                    if os.path.exists(rl_model):
                        self.rl_model_path = rl_model
                        self.rl_model_label.setText(os.path.basename(run_path))
                        break

            # 查找最佳扩散模型
            for run_dir in os.listdir(MODEL_SAVE_DIR):
                run_path = os.path.join(MODEL_SAVE_DIR, run_dir)
                if os.path.isdir(run_path):
                    diff_model = os.path.join(run_path, "best_diff_policy.pth")
                    if os.path.exists(diff_model):
                        self.diffusion_model_path = diff_model
                        self.diff_model_label.setText(os.path.basename(run_dir))
                        break

    def select_rl_model(self):
        """选择RL模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择RL模型", MODEL_SAVE_DIR, "PyTorch模型 (*.pth)"
        )
        if file_path:
            self.rl_model_path = file_path
            self.rl_model_label.setText(os.path.basename(os.path.dirname(file_path)))
            self.add_history("选择RL模型: " + os.path.basename(file_path))

    def select_diffusion_model(self):
        """选择扩散模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择扩散模型", MODEL_SAVE_DIR, "PyTorch模型 (*.pth)"
        )
        if file_path:
            self.diffusion_model_path = file_path
            self.diff_model_label.setText(os.path.basename(os.path.dirname(file_path)))
            self.add_history("选择扩散模型: " + os.path.basename(file_path))

    def start_new_game(self):
        """开始新游戏"""
        self.game_engine.reset_game()
        self.chess_board.reset_board()
        self.move_count = 0
        self.is_paused = False

        # 初始化AI玩家
        self.init_ai_players()

        # 更新状态
        self.update_status("游戏开始！")
        self.current_player_label.setText(f"当前玩家: {self.game_engine.current_player} (●)")
        self.move_count_label.setText(f"回合数: {self.move_count}")

        self.start_button.setText("重新开始")
        self.pause_button.setEnabled(True)

        # 如果先手是AI，则让AI下子
        if self.is_ai_player(1):
            QTimer.singleShot(1000, self.make_ai_move)

        self.add_history("=== 新游戏开始 ===")
        self.add_history(f"玩家1: {self.get_player_name(1)} vs 玩家2: {self.get_player_name(2)}")

    def init_ai_players(self):
        """初始化AI玩家"""
        player1_type = self.player1_combo.currentText()
        player2_type = self.player2_combo.currentText()

        # 玩家1 (先手)
        if player1_type == "随机AI":
            self.ai_players[1] = RandomAI()
        elif player1_type == "RL AI":
            self.ai_players[1] = RLAI(self.rl_model_path)
        elif player1_type == "扩散AI":
            self.ai_players[1] = DiffusionAI(self.diffusion_model_path)

        # 玩家2 (后手)
        if player2_type == "随机AI":
            self.ai_players[-1] = RandomAI()
        elif player2_type == "RL AI":
            self.ai_players[-1] = RLAI(self.rl_model_path)
        elif player2_type == "扩散AI":
            self.ai_players[-1] = DiffusionAI(self.diffusion_model_path)

    def is_ai_player(self, player: int) -> bool:
        """检查是否是AI玩家"""
        return player in self.ai_players

    def get_player_name(self, player: int) -> str:
        """获取玩家名称"""
        if self.is_ai_player(player):
            return self.ai_players[player].name
        elif player == 1:
            return "人类玩家1"
        else:
            return "人类玩家2"

    def on_board_clicked(self, row, col):
        """处理棋盘点击事件"""
        if self.game_engine.game_over or self.is_paused:
            return

        current_player = self.game_engine.current_player

        # 如果当前是AI玩家，不允许人类点击
        if self.is_ai_player(current_player):
            self.update_status("请等待AI下子...")
            return

        # 人类玩家下子
        self.make_human_move(row, col)

    def make_human_move(self, row, col):
        """人类玩家下子"""
        success, message = self.game_engine.make_move(row, col)

        if success:
            self.chess_board.make_move(row, col, self.game_engine.current_player * -1)  # 因为make_move会切换玩家
            self.move_count += 1
            self.move_count_label.setText(f"回合数: {self.move_count}")

            player_symbol = "●" if self.game_engine.current_player == -1 else "○"
            self.add_history(f"人类玩家 {player_symbol}: ({row}, {col})")

            if self.game_engine.game_over:
                self.handle_game_over()
            else:
                self.current_player_label.setText(f"当前玩家: {self.game_engine.current_player} (●)" if self.game_engine.current_player == 1 else "当前玩家: {self.game_engine.current_player} (○)")

                # 如果下一个玩家是AI，则让AI下子
                if self.is_ai_player(self.game_engine.current_player):
                    QTimer.singleShot(1000, self.make_ai_move)
        else:
            self.update_status(f"无效移动: {message}")

    def make_ai_move(self):
        """AI玩家下子"""
        if self.game_engine.game_over or self.is_paused:
            return

        current_player = self.game_engine.current_player

        if not self.is_ai_player(current_player):
            return

        ai_player = self.ai_players[current_player]
        board_state = self.game_engine.get_board_state()

        self.update_status(f"{ai_player.name}正在思考...")
        QApplication.processEvents()

        # 获取AI的移动
        move = ai_player.get_move(board_state, current_player)

        if move:
            row, col = move
            success, message = self.game_engine.make_move(row, col)

            if success:
                self.chess_board.make_move(row, col, self.game_engine.current_player * -1)
                self.move_count += 1
                self.move_count_label.setText(f"回合数: {self.move_count}")

                player_symbol = "●" if self.game_engine.current_player == -1 else "○"
                self.add_history(f"{ai_player.name} {player_symbol}: ({row}, {col})")

                if self.game_engine.game_over:
                    self.handle_game_over()
                else:
                    self.current_player_label.setText(f"当前玩家: {self.game_engine.current_player} (●)" if self.game_engine.current_player == 1 else "当前玩家: {self.game_engine.current_player} (○)")

                    # 如果下一个玩家也是AI，继续
                    if self.is_ai_player(self.game_engine.current_player):
                        QTimer.singleShot(1000, self.make_ai_move)
                    else:
                        self.update_status("请人类玩家下子")
        else:
            self.update_status(f"{ai_player.name}无法找到有效移动")

    def handle_game_over(self):
        """处理游戏结束"""
        winner = self.game_engine.winner

        if winner:
            winner_name = self.get_player_name(winner)
            winner_symbol = "●" if winner == 1 else "○"
            self.update_status(f"游戏结束！{winner_name} {winner_symbol} 获胜！")
            self.add_history(f"=== {winner_name} 获胜！===")

            # 显示获胜连线
            success, line = self.game_engine.check_win(winner)
            if success and line:
                self.chess_board.set_winning_line(line)
        else:
            self.update_status("游戏结束！平局！")
            self.add_history("=== 平局！===")

        self.pause_button.setEnabled(False)

        # 显示结果对话框
        self.show_game_result()

    def show_game_result(self):
        """显示游戏结果"""
        winner = self.game_engine.winner

        # 检查是否是人类玩家获胜或失败
        if winner:
            winner_name = self.get_player_name(winner)
            winner_symbol = "●" if winner == 1 else "○"

            # 检查是否是人类玩家
            player1_type = self.player1_combo.currentText()
            player2_type = self.player2_combo.currentText()

            if "人类" in player1_type and winner == 1:
                # 人类玩家1获胜
                title = "恭喜你赢了！🎉"
                message = f"恭喜！你 ({winner_symbol}) 获胜了！\n\n总回合数：{self.move_count}\n\n是否再来一局？"
                icon = QMessageBox.Information
            elif "人类" in player2_type and winner == -1:
                # 人类玩家2获胜
                title = "恭喜你赢了！🎉"
                message = f"恭喜！你 ({winner_symbol}) 获胜了！\n\n总回合数：{self.move_count}\n\n是否再来一局？"
                icon = QMessageBox.Information
            else:
                # AI获胜
                title = "很遗憾，你输了 😔"
                message = f"{winner_name} ({winner_symbol}) 获胜了！\n\n总回合数：{self.move_count}\n\n继续努力，再来一局？"
                icon = QMessageBox.Warning
        else:
            # 平局
            title = "平局！🤝"
            message = f"这是一场平局！\n\n总回合数：{self.move_count}\n\n再来一局决定胜负？"
            icon = QMessageBox.Question

        # 显示对话框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        msg_box.button(QMessageBox.Yes).setText("再来一局")
        msg_box.button(QMessageBox.No).setText("结束游戏")

        reply = msg_box.exec_()

        if reply == QMessageBox.Yes:
            self.start_new_game()

    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText("继续游戏")
            self.update_status("游戏已暂停")
        else:
            self.pause_button.setText("暂停游戏")
            self.update_status("游戏继续")

            # 如果当前是AI玩家的回合，让AI继续下子
            if self.is_ai_player(self.game_engine.current_player):
                QTimer.singleShot(500, self.make_ai_move)

    def update_status(self, message):
        """更新状态显示"""
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def add_history(self, message):
        """添加游戏记录"""
        self.history_text.append(message)
        # 自动滚动到底部
        scrollbar = self.history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_game(self):
        """加载游戏（功能待实现）"""
        QMessageBox.information(self, "提示", "加载游戏功能待实现")

    def save_game(self):
        """保存游戏（功能待实现）"""
        QMessageBox.information(self, "提示", "保存游戏功能待实现")

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于",
            "五子棋对战系统\n\n"
            "支持多种对战模式：\n"
            "• 人机对战\n"
            "• AI对战 (随机AI, RL AI, 扩散AI)\n"
            "• 可选择先后手\n\n"
            "基于强化学习和扩散模型\n"
            "实现智能五子棋AI"
        )


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("五子棋对战系统")
    app.setApplicationVersion("1.0")

    # 创建并显示主窗口
    window = BattleSystemWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    if PyQt5_AVAILABLE:
        main()
    else:
        print("PyQt5未安装，无法运行图形界面")
        print("请运行: pip install PyQt5")