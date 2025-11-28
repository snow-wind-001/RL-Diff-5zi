#!/usr/bin/env python3
"""
RL-Diff-5zi 项目启动器
沈阳理工大学装备工程学院深度学习课题组
Shenyang Ligong University - School of Equipment Engineering
Deep Learning Research Group
"""

import sys
import os
import subprocess
import time
from typing import Dict, List

def print_banner():
    """打印项目横幅"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    RL-Diff-5zi 五子棋AI系统                                   ║
║                                                                              ║
║  🏛️  沈阳理工大学装备工程学院深度学习课题组                                ║
║  🧠  Deep Learning Research Group                                         ║
║                                                                              ║
║  🚀 强化学习 + 扩散模型融合的五子棋AI                                 ║
║  🎮 完整的训练系统 + 图形化对战界面                                    ║
║  📚 模块化代码架构，便于研究与应用                                      ║
╚════════════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies() -> Dict[str, bool]:
    """检查依赖项"""
    deps = {
        'torch': False,
        'numpy': False,
        'tensorboard': False,
        'PyQt5': False
    }

    print("🔍 检查系统依赖...")

    try:
        import torch
        deps['torch'] = True
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: 未安装 (pip install torch)")

    try:
        import numpy as np
        deps['numpy'] = True
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy: 未安装 (pip install numpy)")

    try:
        import tensorboard
        deps['tensorboard'] = True
        print("✅ TensorBoard: 已安装")
    except ImportError:
        print("❌ TensorBoard: 未安装 (pip install tensorboard)")

    try:
        from PyQt5.QtWidgets import QApplication
        deps['PyQt5'] = True
        print("✅ PyQt5: 已安装")
    except ImportError:
        print("❌ PyQt5: 未安装 (pip install PyQt5)")

    return deps

def check_project_files() -> bool:
    """检查项目文件完整性"""
    print("\n📁 检查项目文件...")

    required_files = [
        'config.py',
        'environment.py',
        'networks.py',
        'agent.py',
        'replay_buffer.py',
        'rl_trainer.py',
        'diffusion_trainer.py',
        'train.py',
        'gui_battle_system.py',
        'simple_battle_system.py',
        'start_battle.py',
        'README.md'
    ]

    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 缺失")
            missing_files.append(file)

    return len(missing_files) == 0

def check_models() -> List[str]:
    """检查可用模型"""
    print("\n🧠 检查可用模型...")

    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("📁 模型目录不存在，需要先训练模型")
        return []

    available_models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            rl_model = os.path.join(model_path, 'best_rl_policy.pth')
            diff_model = os.path.join(model_path, 'best_diff_policy.pth')

            has_rl = os.path.exists(rl_model)
            has_diff = os.path.exists(diff_model)

            if has_rl or has_diff:
                model_info = f"{item}"
                if has_rl:
                    model_info += " [RL]"
                if has_diff:
                    model_info += " [Diffusion]"
                print(f"✅ {model_info}")
                available_models.append(item)

    if not available_models:
        print("📁 未找到训练好的模型")

    return available_models

def show_menu(deps: Dict[str, bool], models: List[str]):
    """显示主菜单"""
    print("\n" + "="*70)
    print("🚀 RL-Diff-5zi 项目启动菜单")
    print("="*70)

    print("\n📚 项目信息:")
    print("1. 📖 查看项目README")
    print("2. 🔧 查看架构说明")
    print("3. 🎮 查看对战系统指南")
    print("4. 🔍 查看修复总结")

    print("\n🏃 训练任务:")
    print("5. 🚀 开始完整训练 (RL + Diffusion)")
    print("6. 🧠 查看训练进度 (TensorBoard)")

    print("\n🎮 对战系统:")
    if deps['PyQt5']:
        print("7. 🖥️  启动图形化对战系统")
    print("8. 💻 启动命令行对战系统")

    if models:
        print(f"\n📊 可用模型: {', '.join(models)}")

    print("\n❓ 其他:")
    print("9. 🔧 安装缺失依赖")
    print("10. ❓ 帮助信息")
    print("0. 🚪 退出")

    print("\n" + "-"*70)

def handle_choice(choice: str, deps: Dict[str, bool]):
    """处理用户选择"""
    try:
        choice = int(choice)
    except ValueError:
        print("❌ 请输入有效的数字")
        return

    if choice == 0:
        print("\n👋 感谢使用 RL-Diff-5zi 项目！")
        sys.exit(0)

    elif choice == 1:
        open_readme()

    elif choice == 2:
        open_file('README_refactored.md')

    elif choice == 3:
        open_file('README_battle_system.md')

    elif choice == 4:
        open_file('FIXES_SUMMARY.md')

    elif choice == 5:
        start_training()

    elif choice == 6:
        start_tensorboard()

    elif choice == 7:
        if deps['PyQt5']:
            start_gui_battle()
        else:
            print("❌ PyQt5未安装，无法启动图形化界面")
            print("💡 运行: pip install PyQt5")

    elif choice == 8:
        start_cli_battle()

    elif choice == 9:
        install_dependencies(deps)

    elif choice == 10:
        show_help()

    else:
        print("❌ 无效选择，请重试")

def open_readme():
    """打开README文件"""
    print("\n📖 打开项目README...")
    if os.name == 'nt':  # Windows
        os.startfile('README.md')
    elif os.name == 'posix':  # macOS/Linux
        try:
            subprocess.run(['xdg-open', 'README.md'], check=False)
        except:
            try:
                subprocess.run(['open', 'README.md'], check=False)
            except:
                print("💡 请手动打开 README.md 文件")

def open_file(filename: str):
    """打开指定文件"""
    if os.path.exists(filename):
        print(f"\n📖 打开 {filename}...")
        if os.name == 'nt':  # Windows
            os.startfile(filename)
        elif os.name == 'posix':  # macOS/Linux
            try:
                subprocess.run(['xdg-open', filename], check=False)
            except:
                try:
                    subprocess.run(['open', filename], check=False)
                except:
                    print(f"💡 请手动打开 {filename} 文件")
    else:
        print(f"❌ 文件不存在: {filename}")

def start_training():
    """开始训练"""
    print("\n🚀 启动完整训练流程...")
    print("💡 这将开始RL + Diffusion模型的完整训练")
    print("💡 训练时间较长，建议使用GPU加速")

    try:
        subprocess.run([sys.executable, 'train.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ 训练失败，请检查环境配置")
    except FileNotFoundError:
        print("❌ 未找到训练脚本")

def start_tensorboard():
    """启动TensorBoard"""
    print("\n📊 启动TensorBoard监控...")
    print("💡 将在浏览器中打开训练进度监控")

    try:
        # 查找最新的日志目录
        logs_dir = 'logs'
        if os.path.exists(logs_dir):
            log_dirs = [d for d in os.listdir(logs_dir) if d.startswith('run_')]
            if log_dirs:
                latest_log = max(log_dirs)
                log_path = os.path.join(logs_dir, latest_log)
                print(f"📈 打开训练日志: {log_path}")
                subprocess.Popen(['tensorboard', '--logdir', log_path])
                print("🌐 TensorBoard正在启动，请等待浏览器打开...")
                time.sleep(2)
                return

        print("❌ 未找到训练日志，请先运行训练")
    except Exception as e:
        print(f"❌ TensorBoard启动失败: {e}")

def start_gui_battle():
    """启动图形化对战系统"""
    print("\n🖥️ 启动图形化对战系统...")
    try:
        subprocess.run([sys.executable, 'start_battle.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ 图形化界面启动失败")
    except FileNotFoundError:
        print("❌ 未找到对战系统文件")

def start_cli_battle():
    """启动命令行对战系统"""
    print("\n💻 启动命令行对战系统...")
    try:
        subprocess.run([sys.executable, 'simple_battle_system.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ 命令行对战系统启动失败")
    except FileNotFoundError:
        print("❌ 未找到对战系统文件")

def install_dependencies(deps: Dict[str, bool]):
    """安装缺失依赖"""
    print("\n🔧 安装缺失的依赖...")

    missing_deps = []
    for dep, installed in deps.items():
        if not installed:
            missing_deps.append(dep)

    if not missing_deps:
        print("✅ 所有依赖都已安装")
        return

    install_commands = {
        'torch': 'pip install torch',
        'numpy': 'pip install numpy',
        'tensorboard': 'pip install tensorboard',
        'PyQt5': 'pip install PyQt5'
    }

    for dep in missing_deps:
        if dep in install_commands:
            cmd = install_commands[dep]
            print(f"🔧 安装 {dep}: {cmd}")
            try:
                subprocess.run(cmd.split(), check=True)
                print(f"✅ {dep} 安装成功")
            except subprocess.CalledProcessError:
                print(f"❌ {dep} 安装失败")

def show_help():
    """显示帮助信息"""
    help_text = """
🚀 RL-Diff-5zi 项目帮助

📚 项目文档:
- README.md: 完整项目说明
- README_refactored.md: 代码架构说明
- README_battle_system.md: 对战系统使用指南
- FIXES_SUMMARY.md: 问题修复总结

🏃 训练命令:
- python train.py: 开始完整训练流程
- tensorboard --logdir=logs/: 查看训练进度

🎮 对战系统:
- python start_battle.py: 智能启动器(推荐)
- python gui_battle_system.py: 图形化界面
- python simple_battle_system.py: 命令行界面

📦 依赖要求:
- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- TensorBoard >= 2.7.0
- PyQt5 >= 5.15.0 (图形界面)

🔧 快速开始:
1. 选择菜单项 5 开始训练模型
2. 训练完成后选择菜单项 7 或 8 开始对战
3. 享受AI对战体验！

💡 如遇问题，请查看相关文档或提交Issue。
"""
    print(help_text)

def main():
    """主函数"""
    print_banner()

    # 检查依赖
    deps = check_dependencies()

    # 检查项目文件
    files_ok = check_project_files()
    if not files_ok:
        print("\n❌ 项目文件不完整，请确保所有必需文件存在")
        sys.exit(1)

    # 检查可用模型
    models = check_models()

    # 显示菜单
    while True:
        show_menu(deps, models)
        choice = input("\n🎯 请选择操作 (0-10): ").strip()
        handle_choice(choice, deps)

        if choice == '0':
            break

        input("\n按Enter键继续...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用 RL-Diff-5zi 项目！")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序运行错误: {e}")
        sys.exit(1)