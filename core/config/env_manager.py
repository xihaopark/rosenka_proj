#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_manager.py
Conda和venv环境管理器
自动检测、创建和管理Python环境
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class EnvironmentManager:
    def __init__(self):
        self.conda_available = self._check_conda()
        self.current_venv = self._detect_current_venv()
        self.project_root = Path.cwd()
        
    def _check_conda(self):
        """检查conda是否可用"""
        try:
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _detect_current_venv(self):
        """检测当前虚拟环境"""
        venv_info = {}
        
        # 检查是否在虚拟环境中
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_info['in_venv'] = True
            venv_info['venv_path'] = sys.prefix
            
            # 检查是否是conda环境
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            if conda_env:
                venv_info['type'] = 'conda'
                venv_info['name'] = conda_env
            else:
                venv_info['type'] = 'venv'
                venv_info['name'] = os.path.basename(sys.prefix)
        else:
            venv_info['in_venv'] = False
            venv_info['type'] = 'system'
        
        return venv_info
    
    def print_status(self):
        """打印当前环境状态"""
        print("🔍 环境状态检测")
        print("=" * 60)
        
        print(f"Python版本: {sys.version}")
        print(f"Python路径: {sys.executable}")
        print(f"工作目录: {self.project_root}")
        
        print(f"\n📊 Conda状态: {'✅ 可用' if self.conda_available else '❌ 不可用'}")
        
        if self.current_venv['in_venv']:
            print(f"📦 当前环境: {self.current_venv['type']} - {self.current_venv['name']}")
            print(f"📁 环境路径: {self.current_venv.get('venv_path', 'unknown')}")
        else:
            print("⚠️ 当前在系统Python环境中")
    
    def list_conda_envs(self):
        """列出conda环境"""
        if not self.conda_available:
            print("❌ Conda不可用")
            return []
        
        try:
            result = subprocess.run(['conda', 'env', 'list', '--json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                envs = []
                for env_path in data['envs']:
                    env_name = os.path.basename(env_path)
                    if env_path == data.get('envs_dirs', [None])[0]:
                        env_name = 'base'
                    envs.append({'name': env_name, 'path': env_path})
                return envs
            else:
                print(f"❌ 获取conda环境失败: {result.stderr}")
                return []
        except Exception as e:
            print(f"❌ 解析conda环境失败: {e}")
            return []
    
    def list_venv_envs(self):
        """查找venv环境"""
        venv_dirs = [
            Path.home() / '.virtualenvs',
            Path.cwd() / 'venv',
            Path.cwd() / '.venv',
            Path('/venv'),  # 容器环境常见路径
        ]
        
        venvs = []
        for venv_dir in venv_dirs:
            if venv_dir.exists():
                if venv_dir.name in ['venv', '.venv']:
                    # 单个venv环境
                    if (venv_dir / 'bin' / 'python').exists() or (venv_dir / 'Scripts' / 'python.exe').exists():
                        venvs.append({
                            'name': venv_dir.name,
                            'path': str(venv_dir),
                            'active': str(venv_dir) == self.current_venv.get('venv_path')
                        })
                else:
                    # 多个venv环境目录
                    for env_path in venv_dir.iterdir():
                        if env_path.is_dir():
                            if (env_path / 'bin' / 'python').exists() or (env_path / 'Scripts' / 'python.exe').exists():
                                venvs.append({
                                    'name': env_path.name,
                                    'path': str(env_path),
                                    'active': str(env_path) == self.current_venv.get('venv_path')
                                })
        
        return venvs
    
    def print_all_environments(self):
        """打印所有环境"""
        print("\n🐍 Conda环境列表:")
        print("-" * 40)
        conda_envs = self.list_conda_envs()
        if conda_envs:
            for env in conda_envs:
                active = "✅" if env['name'] == self.current_venv.get('name') else "  "
                print(f"{active} {env['name']:<15} {env['path']}")
        else:
            print("❌ 没有找到conda环境")
        
        print("\n📦 Venv环境列表:")
        print("-" * 40)
        venv_envs = self.list_venv_envs()
        if venv_envs:
            for env in venv_envs:
                active = "✅" if env['active'] else "  "
                print(f"{active} {env['name']:<15} {env['path']}")
        else:
            print("❌ 没有找到venv环境")
    
    def create_conda_env(self, env_name, python_version="3.10"):
        """创建conda环境"""
        if not self.conda_available:
            print("❌ Conda不可用，无法创建conda环境")
            return False
        
        print(f"🔧 创建conda环境: {env_name}")
        try:
            cmd = ['conda', 'create', '-n', env_name, f'python={python_version}', '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ 成功创建conda环境: {env_name}")
                return True
            else:
                print(f"❌ 创建conda环境失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ 创建conda环境异常: {e}")
            return False
    
    def install_packages_in_conda(self, env_name, packages):
        """在conda环境中安装包"""
        if not self.conda_available:
            print("❌ Conda不可用")
            return False
        
        print(f"📦 在conda环境 {env_name} 中安装包...")
        
        # 分离conda包和pip包
        conda_packages = [
            'numpy', 'pandas', 'matplotlib', 'scikit-learn', 
            'jupyter', 'ipython', 'scipy'
        ]
        
        pip_packages = [
            'streamlit', 'paddleocr', 'easyocr', 'rapidfuzz',
            'transformers', 'PyMuPDF'
        ]
        
        # 安装conda包
        conda_to_install = [pkg for pkg in packages if any(cp in pkg for cp in conda_packages)]
        if conda_to_install:
            try:
                cmd = ['conda', 'install', '-n', env_name] + conda_to_install + ['-y']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Conda包安装成功: {conda_to_install}")
                else:
                    print(f"⚠️ Conda包安装部分失败: {result.stderr}")
            except Exception as e:
                print(f"❌ Conda包安装异常: {e}")
        
        # 安装pip包
        pip_to_install = [pkg for pkg in packages if pkg not in conda_to_install]
        if pip_to_install:
            try:
                cmd = ['conda', 'run', '-n', env_name, 'pip', 'install'] + pip_to_install
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Pip包安装成功: {pip_to_install}")
                else:
                    print(f"⚠️ Pip包安装部分失败: {result.stderr}")
            except Exception as e:
                print(f"❌ Pip包安装异常: {e}")
        
        return True
    
    def create_hybrid_environment(self):
        """创建混合环境配置"""
        print("\n🔧 创建混合环境配置")
        print("=" * 60)
        
        # 方案1：创建专门的conda环境用于GPU和科学计算
        if self.conda_available:
            conda_env_name = "rosenka_gpu"
            print(f"📊 方案1: 创建conda环境 '{conda_env_name}' 用于GPU和科学计算")
            
            if self.create_conda_env(conda_env_name):
                # 在conda环境中安装GPU相关包
                gpu_packages = [
                    'numpy<2.0',  # 固定numpy版本
                    'opencv',
                    'pytorch',
                    'torchvision',
                    'torchaudio',
                    'cudatoolkit',
                    'pandas',
                    'matplotlib',
                    'scikit-learn',
                    'jupyter'
                ]
                
                print("安装GPU和科学计算包...")
                self.install_packages_in_conda(conda_env_name, gpu_packages)
                
                # 使用pip安装特殊包
                special_packages = [
                    'paddlepaddle-gpu',
                    'paddleocr',
                    'easyocr',
                    'streamlit',
                    'PyMuPDF',
                    'rapidfuzz'
                ]
                
                print("使用pip安装特殊包...")
                try:
                    cmd = ['conda', 'run', '-n', conda_env_name, 'pip', 'install'] + special_packages
                    subprocess.run(cmd, check=True)
                    print("✅ 特殊包安装完成")
                except Exception as e:
                    print(f"⚠️ 特殊包安装部分失败: {e}")
        
        # 方案2：优化现有venv环境
        print(f"\n📦 方案2: 优化现有venv环境 'main'")
        if self.current_venv['name'] == 'main':
            print("✅ 当前在main环境中，直接优化")
            self._fix_current_venv()
        else:
            print("⚠️ 需要激活main环境后运行优化")
    
    def _fix_current_venv(self):
        """修复当前venv环境"""
        print("🔧 修复当前venv环境...")
        
        # 修复numpy兼容性
        fix_commands = [
            ['pip', 'uninstall', 'opencv-python', 'opencv-contrib-python', 'opencv-python-headless', '-y'],
            ['pip', 'install', 'numpy<2.0', '--force-reinstall'],
            ['pip', 'install', 'opencv-python-headless'],
            ['pip', 'install', 'pillow', 'pytesseract', 'jieba', 'rapidfuzz', 'transformers'],
        ]
        
        # 检查GPU
        gpu_available = self._check_gpu()
        if gpu_available:
            fix_commands.extend([
                ['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'],
                ['pip', 'install', 'paddlepaddle-gpu'],
                ['pip', 'install', 'paddleocr', 'easyocr']
            ])
        else:
            fix_commands.extend([
                ['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'],
                ['pip', 'install', 'paddlepaddle'],
                ['pip', 'install', 'paddleocr', 'easyocr']
            ])
        
        for cmd in fix_commands:
            try:
                print(f"执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ 成功")
                else:
                    print(f"⚠️ 部分失败: {result.stderr[:100]}...")
            except Exception as e:
                print(f"❌ 异常: {e}")
    
    def _check_gpu(self):
        """检查GPU可用性"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def generate_activation_scripts(self):
        """生成环境激活脚本"""
        print("\n📝 生成环境激活脚本")
        print("=" * 60)
        
        # Conda环境激活脚本
        conda_script = """#!/bin/bash
# activate_conda.sh - 激活conda环境
echo "🐍 激活conda环境 rosenka_gpu"
conda activate rosenka_gpu
echo "✅ 当前环境: $(conda info --envs | grep '*')"
echo "🚀 可以运行GPU加速的OCR任务"
echo ""
echo "使用方法:"
echo "  python test_gpu_ocr_integration.py  # 测试GPU OCR"
echo "  python launch_gpu_app.py           # 启动应用"
"""
        
        # Venv环境激活脚本
        venv_script = """#!/bin/bash
# activate_venv.sh - 激活venv环境
echo "📦 激活venv环境 main"
source /venv/main/bin/activate
echo "✅ 当前环境: main"
echo "🔧 适合开发和调试"
echo ""
echo "使用方法:"
echo "  python check_environment.py        # 检查环境"
echo "  python fix_environment.py          # 修复环境"
echo "  python launch_modern_app.py        # 启动应用"
"""
        
        # 项目启动脚本
        project_script = """#!/bin/bash
# start_project.sh - 智能项目启动
echo "🚀 路線価図项目启动器"
echo "========================"

# 检查可用环境
if conda info --envs | grep -q "rosenka_gpu"; then
    echo "🐍 发现conda环境，使用GPU加速模式"
    conda activate rosenka_gpu
    python launch_gpu_app.py
elif [ -d "/venv/main" ]; then
    echo "📦 使用venv环境"
    source /venv/main/bin/activate
    python launch_modern_app.py
else
    echo "⚠️ 未找到合适的环境，使用系统Python"
    python launch_modern_app.py
fi
"""
        
        # 写入脚本文件
        scripts = [
            ('activate_conda.sh', conda_script),
            ('activate_venv.sh', venv_script),
            ('start_project.sh', project_script)
        ]
        
        for filename, content in scripts:
            with open(filename, 'w') as f:
                f.write(content)
            os.chmod(filename, 0o755)
            print(f"✅ 创建了 {filename}")
    
    def provide_recommendations(self):
        """提供使用建议"""
        print("\n💡 环境使用建议")
        print("=" * 60)
        
        print("🎯 推荐配置:")
        print("1. Conda环境 'rosenka_gpu' - 用于GPU加速和生产环境")
        print("   - 优点: 更好的包管理，GPU支持更稳定")
        print("   - 用途: 运行GPU加速的OCR任务")
        print("   - 激活: conda activate rosenka_gpu")
        
        print("\n2. Venv环境 'main' - 用于开发和调试")
        print("   - 优点: 轻量级，启动快")
        print("   - 用途: 代码开发，环境调试")
        print("   - 激活: source /venv/main/bin/activate")
        
        print("\n🔄 环境切换策略:")
        print("- 开发调试 → 使用venv main环境")
        print("- GPU训练/推理 → 使用conda rosenka_gpu环境")
        print("- 生产部署 → 使用conda环境")
        
        print("\n📋 快速命令:")
        print("./start_project.sh        # 智能启动项目")
        print("./activate_conda.sh       # 激活conda环境")
        print("./activate_venv.sh        # 激活venv环境")

def main():
    """主函数"""
    print("🔧 Conda & Venv 环境管理器")
    print("=" * 60)
    
    manager = EnvironmentManager()
    
    # 1. 检测当前状态
    manager.print_status()
    
    # 2. 列出所有环境
    manager.print_all_environments()
    
    # 3. 创建混合环境配置
    print("\n" + "=" * 60)
    response = input("是否创建/优化混合环境配置？(y/n): ").lower().strip()
    if response == 'y':
        manager.create_hybrid_environment()
        manager.generate_activation_scripts()
    
    # 4. 提供建议
    manager.provide_recommendations()
    
    print("\n🎉 环境管理完成！")

if __name__ == "__main__":
    main() 