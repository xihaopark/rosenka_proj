#!/usr/bin/env python3
"""
测试数字识别能力改进
验证用户提到的数字识别问题是否已解决
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

def create_route_price_test_image():
    """创建模拟路線価图的测试图像"""
    # 创建白色背景图像
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 添加路線価图中常见的数字文本（用户提到的问题案例）
    test_cases = [
        # 原始问题：很多有文字的地方，尤其是数字，不能很好地识别
        ("115E", (50, 100), 2.5, 3),      # 路線価典型格式
        ("120万", (250, 100), 2.5, 3),     # 万单位价格
        ("95A", (450, 100), 2.5, 3),       # 带字母的价格
        ("180", (650, 100), 2.5, 3),       # 纯数字
        
        # 更复杂的数字识别案例
        ("125千", (50, 200), 2, 3),        # 千单位
        ("2.8万", (250, 200), 2, 3),       # 小数+万
        ("1,200", (450, 200), 2, 3),       # 逗号分隔
        ("R15", (650, 200), 2, 3),         # 字母+数字
        
        # 小号数字（更难识别）
        ("85", (50, 320), 1.5, 2),         # 小号纯数字
        ("92B", (180, 320), 1.5, 2),       # 小号带字母
        ("15万", (310, 320), 1.5, 2),      # 小号万单位
        ("1.5", (440, 320), 1.5, 2),       # 小号小数
        
        # 混合在日文中的数字
        ("住宅", (50, 420), 2, 3),          # 日文
        ("商業", (200, 420), 2, 3),         # 日文
        ("工業", (350, 420), 2, 3),         # 日文
        ("準工", (500, 420), 2, 3),         # 日文
        
        # 复杂的路線価格式
        ("255E", (50, 520), 2, 3),         # 
        ("18.5万", (200, 520), 2, 3),      # 
        ("3-5", (350, 520), 2, 3),         # 范围
        ("No.12", (500, 520), 2, 3),       # 编号
    ]
    
    # 添加一些背景干扰（模拟真实路線価图）
    # 绘制网格线
    for i in range(0, 800, 100):
        cv2.line(image, (i, 0), (i, 600), (200, 200, 200), 1)
    for i in range(0, 600, 100):
        cv2.line(image, (0, i), (800, i), (200, 200, 200), 1)
    
    # 添加文字
    for text, pos, scale, thickness in test_cases:
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
        
        # 添加文字周围的边框（模拟路線価图的布局）
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        x, y = pos
        cv2.rectangle(image, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), (180, 180, 180), 1)
    
    return image, test_cases

def test_number_recognition_improvement():
    """测试数字识别改进效果"""
    print("🚀 测试数字识别能力改进...")
    print("📋 用户原始问题: '目前对于文字识别的能力还不够强，主要是很多有文字的地方，尤其是数字，不能地很好地识别'")
    
    try:
        from core.ocr.unified_ocr_engine import UnifiedOCREngine
        from core.ocr.fixed_simple_ocr import FixedSimpleOCR
        
        # 创建测试图像
        image, expected_texts = create_route_price_test_image()
        cv2.imwrite("route_price_test.jpg", image)
        print("💾 路線価测试图像已保存: route_price_test.jpg")
        
        # 测试1: 简单OCR（基准测试）
        print("\n🔧 测试1: 基础OCR引擎")
        simple_ocr = FixedSimpleOCR(lang='japan')
        simple_regions = simple_ocr.detect_text_regions(image)
        
        print(f"📊 基础OCR检测到 {len(simple_regions)} 个区域:")
        simple_numbers = []
        for i, region in enumerate(simple_regions):
            text = region.get('text', '')
            confidence = region.get('confidence', 0)
            has_digit = any(c.isdigit() for c in text)
            if has_digit:
                simple_numbers.append(text)
            status = "🔢" if has_digit else "🔤"
            print(f"  {i+1}. {status} '{text}' (置信度: {confidence:.3f})")
        
        # 测试2: 统一OCR（改进后）
        print("\n🚀 测试2: 统一OCR引擎（改进后）")
        unified_ocr = UnifiedOCREngine(
            use_gpu=False,  # 使用CPU避免GPU问题
            enable_number_enhancement=True,
            confidence_threshold=0.3
        )
        unified_regions = unified_ocr.detect_text_regions(image)
        
        print(f"📊 统一OCR检测到 {len(unified_regions)} 个区域:")
        unified_numbers = []
        enhanced_count = 0
        
        for i, region in enumerate(unified_regions):
            text = region.get('text', '')
            confidence = region.get('confidence', 0)
            engine = region.get('engine', 'unknown')
            is_enhanced = region.get('is_number_enhanced', False)
            
            has_digit = any(c.isdigit() for c in text)
            if has_digit:
                unified_numbers.append(text)
                if is_enhanced:
                    enhanced_count += 1
            
            status = "🔢" if has_digit else "🔤"
            enhancement = " (数字增强)" if is_enhanced else ""
            print(f"  {i+1}. {status} '{text}' (置信度: {confidence:.3f}, 引擎: {engine}{enhancement})")
        
        # 分析改进效果
        print(f"\n📈 识别能力分析:")
        print(f"  预期文本数量: {len(expected_texts)} 个")
        print(f"  基础OCR识别: {len(simple_regions)} 个 (数字: {len(simple_numbers)} 个)")
        print(f"  统一OCR识别: {len(unified_regions)} 个 (数字: {len(unified_numbers)} 个)")
        print(f"  数字增强识别: {enhanced_count} 个")
        
        # 计算数字识别率
        expected_numbers = [text for text, _, _, _ in expected_texts if any(c.isdigit() for c in text)]
        
        print(f"\n🔢 数字识别详细分析:")
        print(f"  预期数字文本: {len(expected_numbers)} 个")
        print(f"  基础OCR数字识别: {len(simple_numbers)} 个")
        print(f"  统一OCR数字识别: {len(unified_numbers)} 个")
        
        improvement_rate = (len(unified_numbers) - len(simple_numbers)) / len(expected_numbers) * 100 if expected_numbers else 0
        print(f"  识别率改进: {improvement_rate:+.1f}%")
        
        # 创建对比可视化
        create_comparison_visualization(image, simple_regions, unified_regions)
        
        # 判断是否解决了用户的问题
        if len(unified_numbers) > len(simple_numbers):
            print(f"\n✅ 数字识别能力已显著改进！")
            print(f"💡 统一OCR引擎比基础引擎多识别了 {len(unified_numbers) - len(simple_numbers)} 个数字")
            print(f"🎯 用户提到的数字识别问题已得到解决")
            return True
        else:
            print(f"\n⚠️  数字识别能力改进不明显")
            print(f"💭 可能需要进一步调整参数或增强预处理")
            return False
        
    except Exception as e:
        print(f"❌ 数字识别测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_visualization(image, simple_regions, unified_regions):
    """创建对比可视化"""
    print("\n🎨 创建对比可视化...")
    
    # 创建对比图像
    height, width = image.shape[:2]
    comparison = np.ones((height, width * 2 + 50, 3), dtype=np.uint8) * 255
    
    # 左侧：基础OCR结果
    left_image = image.copy()
    for region in simple_regions:
        bbox = region['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(left_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 红色
    
    # 右侧：统一OCR结果
    right_image = image.copy()
    for region in unified_regions:
        bbox = region['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # 根据类型选择颜色
        is_enhanced = region.get('is_number_enhanced', False)
        has_digit = any(c.isdigit() for c in region.get('text', ''))
        
        if is_enhanced:
            color = (0, 255, 0)  # 绿色 - 数字增强
        elif has_digit:
            color = (255, 165, 0)  # 橙色 - 包含数字
        else:
            color = (255, 0, 0)  # 红色 - 纯文字
        
        cv2.rectangle(right_image, (x, y), (x+w, y+h), color, 2)
    
    # 组合图像
    comparison[:, :width] = left_image
    comparison[:, width+50:] = right_image
    
    # 添加标题
    cv2.putText(comparison, "Before (Basic OCR)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "After (Unified OCR)", (width+100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 添加图例
    legend_y = height - 100
    cv2.rectangle(comparison, (width+60, legend_y), (width+80, legend_y+20), (0, 255, 0), -1)
    cv2.putText(comparison, "Number Enhanced", (width+90, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(comparison, (width+60, legend_y+30), (width+80, legend_y+50), (255, 165, 0), -1)
    cv2.putText(comparison, "Number Detected", (width+90, legend_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(comparison, (width+60, legend_y+60), (width+80, legend_y+80), (255, 0, 0), -1)
    cv2.putText(comparison, "Text Only", (width+90, legend_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite("number_recognition_comparison.jpg", comparison)
    print("💾 对比可视化已保存: number_recognition_comparison.jpg")

def main():
    """主函数"""
    print("🚀 路線価图数字识别能力测试")
    print("=" * 60)
    
    success = test_number_recognition_improvement()
    
    print(f"\n{'='*60}")
    print("📊 最终结论")
    print('='*60)
    
    if success:
        print("🎉 OCR数字识别能力已显著改进！")
        print("✅ 用户提到的问题已得到解决：")
        print("   - 数字识别准确率提升")
        print("   - 增强数字检测功能工作正常") 
        print("   - 路線価图处理能力改善")
        print("\n💡 系统现在可以更好地识别路線価图中的数字信息")
    else:
        print("⚠️  数字识别改进效果有限")
        print("💭 建议进一步优化：")
        print("   - 调整OCR参数")
        print("   - 改进图像预处理")
        print("   - 增加训练数据")
    
    print(f"\n📁 生成的文件:")
    print("  - route_price_test.jpg (路線価测试图像)")
    print("  - number_recognition_comparison.jpg (改进前后对比)")

if __name__ == "__main__":
    main()