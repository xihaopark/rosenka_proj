#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_environment.py
修复Python环境问题
主要解决NumPy版本兼容性和缺失包安装
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
        else:
            print("❌ 失败")
            if result.stderr.strip():
                print(f"错误: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def fix_numpy_compatibility():
    """修复NumPy兼容性问题"""
    print("="*60)
    print(" 修复NumPy兼容性问题")
    print("="*60)
    
    # 1. 卸载可能冲突的包
    packages_to_uninstall = ['opencv-python', 'opencv-contrib-python', 'opencv-python-headless']
    for package in packages_to_uninstall:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    # 2. 降级NumPy到兼容版本
    run_command("pip install 'numpy<2.0' --force-reinstall", "降级NumPy到1.x版本")
    
    # 3. 重新安装OpenCV
    run_command("pip install opencv-python-headless", "重新安装OpenCV")

def install_missing_packages():
    """安装缺失的包"""
    print("\n" + "="*60)
    print(" 安装缺失的包")
    print("="*60)
    
    # 基础包
    basic_packages = [
        "pillow",
        "pytesseract",
        "jieba",
        "transformers",
        "rapidfuzz"
    ]
    
    for package in basic_packages:
        run_command(f"pip install {package}", f"安装 {package}")

def install_gpu_packages():
    """安装GPU加速包"""
    print("\n" + "="*60)
    print(" 安装GPU加速包")
    print("="*60)
    
    # 检查是否有GPU
    gpu_available = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_available = result.returncode == 0
    except:
        pass
    
    if gpu_available:
        print("✅ 检测到GPU，安装GPU版本")
        
        # PyTorch GPU版本
        run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "安装PyTorch GPU版本"
        )
        
        # PaddlePaddle GPU版本
        run_command("pip install paddlepaddle-gpu", "安装PaddlePaddle GPU版本")
        
        # PaddleOCR
        run_command("pip install paddleocr", "安装PaddleOCR")
        
        # EasyOCR
        run_command("pip install easyocr", "安装EasyOCR")
        
    else:
        print("⚠️ 未检测到GPU，安装CPU版本")
        
        # PyTorch CPU版本
        run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "安装PyTorch CPU版本"
        )
        
        # PaddlePaddle CPU版本
        run_command("pip install paddlepaddle", "安装PaddlePaddle CPU版本")
        
        # PaddleOCR
        run_command("pip install paddleocr", "安装PaddleOCR")
        
        # EasyOCR (CPU模式)
        run_command("pip install easyocr", "安装EasyOCR")

def verify_installation():
    """验证安装"""
    print("\n" + "="*60)
    print(" 验证安装")
    print("="*60)
    
    test_imports = [
        ("import numpy", "NumPy"),
        ("import cv2", "OpenCV"),
        ("import torch", "PyTorch"),
        ("from PIL import Image", "Pillow"),
        ("import pytesseract", "Tesseract"),
        ("import rapidfuzz", "RapidFuzz"),
        ("import jieba", "Jieba"),
    ]
    
    for import_cmd, name in test_imports:
        try:
            exec(import_cmd)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
    
    # 测试OCR引擎
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR")
    except ImportError:
        print("❌ PaddleOCR")
    
    try:
        import easyocr
        print("✅ EasyOCR")
    except ImportError:
        print("❌ EasyOCR")

def create_fixed_requirements():
    """创建修复后的requirements文件"""
    print("\n" + "="*60)
    print(" 创建固定版本requirements")
    print("="*60)
    
    fixed_requirements = """# 修复版本的requirements文件
# 解决NumPy兼容性问题

# 核心数值计算 (固定版本避免兼容性问题)
numpy<2.0
opencv-python-headless>=4.8.0
Pillow>=9.0.0
scikit-image>=0.19.0
matplotlib>=3.5.0
pandas>=1.5.0
plotly>=5.0.0

# 深度学习框架
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0

# OCR引擎
paddleocr>=2.7.0
easyocr>=1.7.0
pytesseract>=0.3.10

# PDF处理
PyMuPDF>=1.23.0

# 文本处理
rapidfuzz>=3.0.0
jieba>=0.42.1

# Web框架
streamlit>=1.28.0

# 工具库
tqdm>=4.65.0
pyyaml>=6.0
"""
    
    with open('requirements_fixed.txt', 'w', encoding='utf-8') as f:
        f.write(fixed_requirements)
    
    print("✅ 创建了 requirements_fixed.txt")

def main():
    """主修复流程"""
    print("🔧 Python环境修复工具")
    print("="*60)
    
    print("⚠️ 检测到NumPy版本兼容性问题")
    print("将执行以下修复步骤:")
    print("1. 修复NumPy兼容性问题")
    print("2. 安装缺失的包")
    print("3. 安装GPU加速包")
    print("4. 验证安装")
    print("5. 创建固定版本requirements")
    
    response = input("\n是否继续？(y/n): ").lower().strip()
    if response != 'y':
        print("取消修复")
        return
    
    # 执行修复步骤
    fix_numpy_compatibility()
    install_missing_packages()
    install_gpu_packages()
    verify_installation()
    create_fixed_requirements()
    
    print("\n" + "="*60)
    print(" 修复完成")
    print("="*60)
    print("🎉 环境修复完成！")
    print("\n下一步:")
    print("1. 重新运行环境检查: python check_environment.py")
    print("2. 测试GPU OCR功能: python test_gpu_ocr_integration.py")
    print("3. 启动应用: python launch_gpu_app.py")

if __name__ == "__main__":
    main() 