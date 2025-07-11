#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modern_detection.py
测试现代化文字检测方案
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.processors.lightweight_detector import LightweightTextDetector
from app.processors.modern_text_detector import ModernTextDetector

def test_detection_methods():
    """测试不同检测方法"""
    
    # 测试图像路径
    test_image_path = "test_rosenka_map.jpg"
    
    print("=" * 50)
    print("文字检测方法对比测试")
    print("=" * 50)
    
    # 方法1: 轻量级检测（零深度学习依赖）
    print("\n1. 轻量级检测器 (OpenCV + Tesseract)")
    print("-" * 30)
    try:
        detector1 = LightweightTextDetector()
        print("✅ 初始化成功")
        print("特点: 零深度学习依赖，稳定可靠")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
    
    # 方法2: 现代检测器（CRAFT + Tesseract）
    print("\n2. 现代检测器 (CRAFT + Tesseract)")
    print("-" * 30)
    try:
        detector2 = ModernTextDetector(use_trocr=False)
        print("✅ 初始化成功")
        print("特点: 专业文字检测，无需PaddlePaddle")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
    
    # 方法3: 现代检测器（CRAFT + TrOCR）
    print("\n3. 现代检测器 (CRAFT + TrOCR)")
    print("-" * 30)
    try:
        detector3 = ModernTextDetector(use_trocr=True)
        print("✅ 初始化成功")
        print("特点: 最先进的Transformer OCR")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("提示: TrOCR需要下载模型，可选功能")

if __name__ == "__main__":
    test_detection_methods()