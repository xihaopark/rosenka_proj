#!/usr/bin/env python3
"""
创建测试PDF文件，模拟路線価图内容
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def create_route_price_test_page():
    """创建模拟路線価图页面"""
    # 创建A4大小的白色画布 (2480x3508 像素, 300 DPI)
    width, height = 2480, 3508
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 尝试使用默认字体
    try:
        # 使用较大的字体
        font_large = ImageFont.truetype("Arial.ttf", 60)
        font_medium = ImageFont.truetype("Arial.ttf", 40)
        font_small = ImageFont.truetype("Arial.ttf", 30)
    except:
        # 如果没有Arial字体，使用默认字体
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 添加标题
    draw.text((100, 50), "路線価図テスト - Route Price Map Test", fill='black', font=font_large)
    
    # 绘制网格背景（模拟真实路線価图）
    grid_color = (220, 220, 220)
    for x in range(0, width, 200):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 200):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # 添加各种类型的路線価信息
    test_data = [
        # 典型路線価格式
        {"text": "115E", "pos": (300, 300), "font": font_large, "color": "black"},
        {"text": "120万", "pos": (600, 300), "font": font_large, "color": "black"},
        {"text": "95A", "pos": (900, 300), "font": font_large, "color": "black"},
        {"text": "180", "pos": (1200, 300), "font": font_large, "color": "black"},
        
        # 小数和复杂格式
        {"text": "12.5万", "pos": (300, 500), "font": font_medium, "color": "black"},
        {"text": "2,500", "pos": (600, 500), "font": font_medium, "color": "black"},
        {"text": "R07", "pos": (900, 500), "font": font_medium, "color": "black"},
        {"text": "No.15", "pos": (1200, 500), "font": font_medium, "color": "black"},
        
        # 地名和用途地域
        {"text": "住宅地", "pos": (300, 700), "font": font_medium, "color": "blue"},
        {"text": "商業地域", "pos": (600, 700), "font": font_medium, "color": "blue"},
        {"text": "工業専用", "pos": (900, 700), "font": font_medium, "color": "blue"},
        {"text": "準工業", "pos": (1200, 700), "font": font_medium, "color": "blue"},
        
        # 小字体数字（挑战识别）
        {"text": "85", "pos": (300, 900), "font": font_small, "color": "black"},
        {"text": "92B", "pos": (450, 900), "font": font_small, "color": "black"},
        {"text": "15万", "pos": (600, 900), "font": font_small, "color": "black"},
        {"text": "1.5", "pos": (750, 900), "font": font_small, "color": "black"},
        
        # 混合格式
        {"text": "255E18.5万", "pos": (300, 1100), "font": font_medium, "color": "red"},
        {"text": "3-5", "pos": (600, 1100), "font": font_medium, "color": "red"},
        {"text": "北側", "pos": (750, 1100), "font": font_medium, "color": "green"},
        {"text": "南側", "pos": (900, 1100), "font": font_medium, "color": "green"},
        
        # 地址信息
        {"text": "東京都渋谷区", "pos": (300, 1300), "font": font_medium, "color": "purple"},
        {"text": "原宿1-2-3", "pos": (600, 1300), "font": font_medium, "color": "purple"},
        {"text": "JR原宿駅", "pos": (900, 1300), "font": font_medium, "color": "purple"},
        {"text": "徒歩5分", "pos": (1200, 1300), "font": font_medium, "color": "purple"},
    ]
    
    # 绘制所有文本
    for item in test_data:
        # 添加文本框背景
        bbox = draw.textbbox(item["pos"], item["text"], font=item["font"])
        padding = 5
        draw.rectangle([
            bbox[0] - padding, bbox[1] - padding,
            bbox[2] + padding, bbox[3] + padding
        ], fill='white', outline='gray', width=1)
        
        # 添加文本
        draw.text(item["pos"], item["text"], fill=item["color"], font=item["font"])
    
    # 添加一些形状标记（模拟路線価图的符号）
    # 圆形标记
    for i, (x, y) in enumerate([(500, 1500), (800, 1500), (1100, 1500)]):
        draw.ellipse([x-20, y-20, x+20, y+20], fill='red', outline='black', width=2)
        draw.text((x-10, y-10), str(i+1), fill='white', font=font_small)
    
    # 方形标记
    for i, (x, y) in enumerate([(500, 1600), (800, 1600), (1100, 1600)]):
        draw.rectangle([x-20, y-20, x+20, y+20], fill='blue', outline='black', width=2)
        draw.text((x-10, y-10), chr(65+i), fill='white', font=font_small)
    
    # 添加页面信息
    draw.text((100, height-100), "テストページ 1/1 - 路線価図OCRテスト用", fill='gray', font=font_small)
    
    return image

def create_test_pdf():
    """创建测试PDF文件"""
    print("📄 创建测试PDF文件...")
    
    # 创建测试图像
    test_image = create_route_price_test_page()
    
    # 保存为图像（用于预览）
    test_image.save("/Users/park/code/rosenka_proj/test_page.jpg", "JPEG", quality=95)
    print("💾 测试图像已保存: test_page.jpg")
    
    # 转换为PDF
    test_image.save("/Users/park/code/rosenka_proj/test.pdf", "PDF", resolution=300.0)
    print("💾 测试PDF已保存: test.pdf")
    
    # 显示文件信息
    import os
    file_size = os.path.getsize("/Users/park/code/rosenka_proj/test.pdf")
    print(f"📊 PDF文件大小: {file_size:,} bytes")
    
    return True

def main():
    """主函数"""
    print("🚀 创建路線価图测试PDF文件")
    print("=" * 50)
    
    try:
        success = create_test_pdf()
        
        if success:
            print("\n✅ 测试PDF创建成功!")
            print("\n📋 创建的文件:")
            print("  - test.pdf (PDF格式)")
            print("  - test_page.jpg (图像预览)")
            print("\n💡 现在可以运行OCR测试:")
            print("  python test_single_pdf.py")
        else:
            print("\n❌ 创建失败")
            
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()