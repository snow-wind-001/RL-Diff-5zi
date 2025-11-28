#!/usr/bin/env python3
"""
五子棋对战系统启动器
自动选择可用的版本并启动
"""

import sys
import os

def check_gui_availability():
    """检查GUI是否可用"""
    try:
        from PyQt5.QtWidgets import QApplication
        from gui_battle_system import BattleSystemWindow
        return True, None
    except ImportError:
        return False, "PyQt5未安装，请运行: pip install PyQt5"
    except Exception as e:
        return False, f"GUI加载失败: {e}"

def start_gui_version():
    """启动GUI版本"""
    try:
        print("启动图形化对战系统...")
        from PyQt5.QtWidgets import QApplication
        from gui_battle_system import BattleSystemWindow

        app = QApplication(sys.argv)
        app.setApplicationName("五子棋对战系统")
        app.setApplicationVersion("1.0")

        window = BattleSystemWindow()
        window.show()

        print("✅ 图形界面已启动，请在窗口中进行游戏")
        return app.exec_()
    except Exception as e:
        print(f"❌ GUI启动失败: {e}")
        return 1

def start_cli_version():
    """启动命令行版本"""
    try:
        print("启动命令行对战系统...")
        from simple_battle_system import main as cli_main

        print("✅ 命令行版本已启动")
        return cli_main()
    except Exception as e:
        print(f"❌ CLI启动失败: {e}")
        return 1

def show_usage():
    """显示使用说明"""
    print("""
五子棋对战系统启动器

用法:
    python start_battle.py [选项]

选项:
    --gui      强制启动GUI版本
    --cli       强制启动命令行版本
    --help      显示此帮助信息

如果不指定选项，程序将自动选择可用的版本

系统要求:
    GUI版本: pip install PyQt5
    CLI版本: 仅需基本Python环境
""")

def main():
    """主函数"""
    args = sys.argv[1:]

    # 处理帮助选项
    if '--help' in args or '-h' in args:
        show_usage()
        return 0

    # 处理强制选项
    force_gui = '--gui' in args
    force_cli = '--cli' in args

    if force_gui and force_cli:
        print("❌ 不能同时指定 --gui 和 --cli")
        return 1

    if force_gui:
        # 强制启动GUI版本
        gui_available, error = check_gui_availability()
        if not gui_available:
            print(f"❌ GUI版本不可用: {error}")
            return 1
        return start_gui_version()

    if force_cli:
        # 强制启动CLI版本
        return start_cli_version()

    # 自动选择版本
    print("🔍 正在检测可用的版本...")

    gui_available, error = check_gui_availability()

    if gui_available:
        print("✅ GUI版本可用，启动图形界面...")
        return start_gui_version()
    else:
        print(f"⚠️  GUI版本不可用: {error}")
        print("📱 启动命令行版本...")
        return start_cli_version()

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序启动失败: {e}")
        sys.exit(1)