#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_environment.py
检查当前Python环境状态
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_python_environment():
    """检查Python环境"""
    print_separator("Python环境信息")
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"Python路径列表:")
    for i, path in enumerate(sys.path[:5]):  # 只显示前5个
        print(f"  {i+1}. {path}")
    
    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✅ 在虚拟环境中: {sys.prefix}")
    else:
        print(f"⚠️ 在系统Python环境中: {sys.prefix}")
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda环境: {conda_env}")
    else:
        print("⚠️ 未检测到Conda环境")

def check_installed_packages():
    """检查已安装的包"""
    print_separator("已安装的关键包")
    
    # 需要检查的包
    packages_to_check = [
        'numpy', 'opencv-python', 'pillow', 'pandas', 'matplotlib',
        'streamlit', 'plotly', 'PyMuPDF', 'rapidfuzz',
        'torch', 'torchvision', 'paddlepaddle', 'paddleocr', 
        'easyocr', 'pytesseract', 'transformers', 'jieba'
    ]
    
    installed = {}
    failed = []
    
    for package in packages_to_check:
        try:
            # 尝试导入包
            if package == 'opencv-python':
                import cv2
                version = cv2.__version__
            elif package == 'PyMuPDF':
                import fitz
                version = fitz.version[0]
            elif package == 'paddlepaddle':
                import paddle
                version = paddle.__version__
            else:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
            
            installed[package] = version
            print(f"✅ {package}: {version}")
            
        except ImportError as e:
            failed.append(package)
            print(f"❌ {package}: 未安装 ({str(e)[:50]}...)")
    
    return installed, failed

def check_gpu_status():
    """检查GPU状态"""
    print_separator("GPU状态检查")
    
    # 检查NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 可用")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver Version' in line:
                    print(f"  {line.strip()}")
        else:
            print("❌ nvidia-smi 命令失败")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ 未找到nvidia-smi命令")
    
    # 检查PyTorch GPU支持
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch 未安装")

def check_ocr_engines():
    """检查OCR引擎"""
    print_separator("OCR引擎状态")
    
    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR 可用")
        # 尝试初始化（但不实际使用GPU）
        try:
            ocr = PaddleOCR(use_gpu=False, show_log=False)
            print("✅ PaddleOCR 初始化成功")
        except Exception as e:
            print(f"⚠️ PaddleOCR 初始化失败: {e}")
    except ImportError:
        print("❌ PaddleOCR 未安装")
    
    # EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR 可用")
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR 初始化成功")
        except Exception as e:
            print(f"⚠️ EasyOCR 初始化失败: {e}")
    except ImportError:
        print("❌ EasyOCR 未安装")
    
    # Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract 可用: {version}")
    except Exception as e:
        print(f"❌ Tesseract 不可用: {e}")

def check_project_files():
    """检查项目文件"""
    print_separator("项目文件检查")
    
    # 检查关键文件
    key_files = [
        'app/processors/gpu_ocr_processor.py',
        'app/processors/enhanced_address_matcher.py',
        'app/processors/simple_processor.py',
        'app/ui/modern_rosenka_app.py',
        'rosenka_data',
        'requirements.txt',
        'requirements_gpu.txt'
    ]
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"✅ {file_path} ({size} bytes)")
            else:
                items = len(list(path.iterdir())) if path.is_dir() else 0
                print(f"✅ {file_path}/ ({items} items)")
        else:
            print(f"❌ {file_path} 不存在")

def test_imports():
    """测试关键模块导入"""
    print_separator("模块导入测试")
    
    test_modules = [
        ('app.processors.gpu_ocr_processor', 'GPUOCREngine'),
        ('app.processors.enhanced_address_matcher', 'SmartAddressMatcher'),
        ('app.processors.simple_processor', 'SimplePDFProcessor'),
    ]
    
    # 添加项目路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "app"))
    sys.path.insert(0, str(project_root / "app" / "processors"))
    
    for module_name, class_name in test_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
        except AttributeError as e:
            print(f"⚠️ {module_name}.{class_name}: {e}")

def provide_recommendations(failed_packages):
    """提供修复建议"""
    print_separator("修复建议")
    
    if failed_packages:
        print("🔧 需要安装的包:")
        print("pip install " + " ".join(failed_packages))
        
        # GPU相关建议
        if any(pkg in failed_packages for pkg in ['torch', 'paddlepaddle', 'paddleocr']):
            print("\n🚀 GPU加速包安装建议:")
            print("# 安装GPU版本PyTorch")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n# 安装GPU版本PaddlePaddle")
            print("pip install paddlepaddle-gpu")
            print("pip install paddleocr")
    
    print("\n📋 完整环境设置:")
    print("1. 运行GPU环境设置脚本:")
    print("   chmod +x setup_gpu_environment.sh")
    print("   ./setup_gpu_environment.sh")
    print("\n2. 或手动安装:")
    print("   pip install -r requirements_gpu.txt")
    print("\n3. 测试环境:")
    print("   python test_gpu_ocr_integration.py")
    print("\n4. 启动应用:")
    print("   python launch_gpu_app.py")

def main():
    """主函数"""
    print("🔍 Python环境诊断工具")
    print(f"工作目录: {os.getcwd()}")
    
    # 执行所有检查
    check_python_environment()
    installed, failed = check_installed_packages()
    check_gpu_status()
    check_ocr_engines()
    check_project_files()
    test_imports()
    provide_recommendations(failed)
    
    print_separator("诊断完成")
    print(f"✅ 已安装包: {len(installed)}")
    print(f"❌ 缺失包: {len(failed)}")
    
    if not failed:
        print("🎉 环境状态良好，可以运行项目！")
    else:
        print("⚠️ 需要安装缺失的包才能正常运行")

if __name__ == "__main__":
    main() 