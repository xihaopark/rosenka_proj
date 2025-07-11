#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_enhanced_detection.py
测试增强版检测效果
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enhanced_text_detector import EnhancedTextDetector
from image_preprocessor import ImagePreprocessor

def visualize_enhanced_results(image, results):
    """可视化增强检测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # 不同类型的结果
    titles = ['横向文字', '竖向文字', '带圈数字', '小文字']
    result_keys = ['horizontal_text', 'vertical_text', 'circled_numbers', 'small_text']
    colors = ['red', 'blue', 'green', 'yellow']
    
    for idx, (ax, title, key, color) in enumerate(zip(axes.flat, titles, result_keys, colors)):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{title} ({len(results.get(key, []))}个)')
        
        # 绘制检测框
        for det in results.get(key, []):
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加文字
                if 'text' in det:
                    ax.text(x1, y1-5, det['text'], 
                           color=color, fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', alpha=0.7))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_on_your_image(image_path):
    """测试您的图像"""
    print("=" * 50)
    print("增强版文字检测测试")
    print("=" * 50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"原始图像尺寸: {image.shape}")
    
    # 预处理
    print("\n1. 图像预处理...")
    preprocessor = ImagePreprocessor()
    enhanced_image = preprocessor.enhance_low_resolution(image)
    print(f"增强后尺寸: {enhanced_image.shape}")
    
    # 检测
    print("\n2. 执行增强检测...")
    detector = EnhancedTextDetector(use_trocr=True)
    results = detector.detect_all_enhanced(enhanced_image)
    
    # 统计
    print("\n3. 检测结果统计:")
    for key, detections in results.items():
        print(f"   - {key}: {len(detections)}个")
    
    # 可视化
    print("\n4. 可视化结果...")
    visualize_enhanced_results(enhanced_image, results)

if __name__ == "__main__":
    # 测试您的图像
    test_on_your_image("your_rosenka_image.jpg")