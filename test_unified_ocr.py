#!/usr/bin/env python3
"""
测试统一OCR引擎
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像"""
    # 创建白色背景图像
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加不同类型的文本
    test_texts = [
        # 数字测试
        ("115E", (50, 80), 2, 3),      # 路線価格式
        ("120万", (300, 80), 2, 3),     # 万单位
        ("95A", (500, 80), 2, 3),       # 带字母
        
        # 纯数字
        ("1200", (50, 180), 2, 3),
        ("85", (300, 180), 2, 3),
        ("150", (500, 180), 2, 3),
        
        # 日文文字
        ("住宅地", (50, 280), 2, 3),
        ("商業", (300, 280), 2, 3),
        ("路線価", (500, 280), 2, 3),
        
        # 混合文本
        ("12.5万", (50, 380), 2, 3),
        ("R07", (300, 380), 2, 3),
        ("2-3", (500, 380), 2, 3),
    ]
    
    for text, pos, scale, thickness in test_texts:
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    
    return image

def test_unified_ocr():
    """测试统一OCR引擎"""
    print("🚀 测试统一OCR引擎...")
    
    try:
        from core.ocr.unified_ocr_engine import UnifiedOCREngine
        
        # 创建测试图像
        image = create_test_image()
        cv2.imwrite("unified_test.jpg", image)
        print("💾 测试图像已保存: unified_test.jpg")
        
        # 初始化统一OCR引擎
        print("🔧 初始化统一OCR引擎...")
        ocr_engine = UnifiedOCREngine(
            use_gpu=False,  # 使用CPU模式避免GPU问题
            enable_number_enhancement=True,
            confidence_threshold=0.3
        )
        
        # 检测文本区域
        print("🔍 检测文本区域...")
        regions = ocr_engine.detect_text_regions(image)
        
        print(f"📊 检测到 {len(regions)} 个文本区域:")
        
        # 分析结果
        number_count = 0
        text_count = 0
        
        for i, region in enumerate(regions):
            text = region.get('text', '')
            confidence = region.get('confidence', 0)
            engine = region.get('engine', 'unknown')
            is_enhanced = region.get('is_number_enhanced', False)
            
            # 判断是否包含数字
            has_digit = any(c.isdigit() for c in text)
            if has_digit:
                number_count += 1
                status = "🔢"
            else:
                text_count += 1
                status = "🔤"
            
            enhancement = " (增强)" if is_enhanced else ""
            
            print(f"  {i+1}. {status} '{text}' (置信度: {confidence:.3f}, 引擎: {engine}{enhancement})")
        
        # 创建可视化结果
        vis_image = image.copy()
        
        for region in regions:
            bbox = region['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # 判断颜色
            text = region.get('text', '')
            has_digit = any(c.isdigit() for c in text)
            is_enhanced = region.get('is_number_enhanced', False)
            
            if is_enhanced:
                color = (0, 255, 0)  # 绿色 - 数字增强
            elif has_digit:
                color = (255, 165, 0)  # 橙色 - 包含数字
            else:
                color = (255, 0, 0)  # 红色 - 纯文字
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # 添加标签
            label = f"{text} ({region.get('confidence', 0):.2f})"
            cv2.putText(vis_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite("unified_results.jpg", vis_image)
        print("💾 可视化结果已保存: unified_results.jpg")
        
        # 输出统计
        print(f"\n📈 识别统计:")
        print(f"  数字文本: {number_count} 个")
        print(f"  纯文字: {text_count} 个")
        print(f"  总计: {len(regions)} 个")
        
        # 获取引擎信息
        info = ocr_engine.get_engine_info()
        print(f"\n🔧 引擎信息:")
        print(f"  名称: {info['name']}")
        print(f"  版本: {info['version']}")
        print(f"  GPU启用: {info['gpu_enabled']}")
        print(f"  数字增强: {info['number_enhancement']}")
        print(f"  置信度阈值: {info['confidence_threshold']}")
        
        if len(regions) > 0:
            print("\n✅ 统一OCR引擎测试成功！")
            return True
        else:
            print("\n⚠️  未检测到任何文本，可能需要调整参数")
            return False
        
    except Exception as e:
        print(f"❌ 统一OCR引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_ocr():
    """测试简单OCR引擎"""
    print("\n🧪 测试简单OCR引擎...")
    
    try:
        from core.ocr.fixed_simple_ocr import FixedSimpleOCR
        
        # 创建简单测试图像
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(image, "115E", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        cv2.putText(image, "120", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        
        # 初始化简单OCR
        ocr = FixedSimpleOCR(lang='japan')
        
        # 检测文本
        regions = ocr.detect_text_regions(image)
        
        print(f"📊 简单OCR检测到 {len(regions)} 个文本区域:")
        for i, region in enumerate(regions):
            text = region.get('text', '')
            confidence = region.get('confidence', 0)
            print(f"  {i+1}. '{text}' (置信度: {confidence:.3f})")
        
        return len(regions) > 0
        
    except Exception as e:
        print(f"❌ 简单OCR测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始OCR系统测试...")
    
    # 测试简单OCR
    simple_success = test_simple_ocr()
    
    # 测试统一OCR
    unified_success = test_unified_ocr()
    
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print('='*60)
    
    print(f"简单OCR测试: {'✅ 通过' if simple_success else '❌ 失败'}")
    print(f"统一OCR测试: {'✅ 通过' if unified_success else '❌ 失败'}")
    
    if simple_success and unified_success:
        print("\n🎉 OCR系统修复成功！")
        print("💡 现在可以处理路線価图的数字识别了")
    elif simple_success:
        print("\n⚠️  基础OCR可用，但统一引擎需要进一步调试")
    else:
        print("\n❌ OCR系统仍需修复")
    
    print("\n📁 生成的文件:")
    print("  - unified_test.jpg (统一OCR测试图像)")
    print("  - unified_results.jpg (统一OCR可视化结果)")

if __name__ == "__main__":
    main()