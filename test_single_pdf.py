#!/usr/bin/env python3
"""
测试单个PDF文件的OCR识别
Test single PDF file OCR recognition
"""

import sys
from pathlib import Path
import logging
import time
import json
import cv2
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.database.database_manager import DatabaseManager
from core.pdf.pdf_processor import PDFProcessor
from core.ocr.unified_ocr_engine import UnifiedOCREngine
from core.detection.circle_detector import CircleDetector

def test_single_pdf(pdf_path: str):
    """测试单个PDF文件"""
    print(f"🚀 开始测试PDF文件: {pdf_path}")
    print("=" * 60)
    
    # 检查文件是否存在
    if not Path(pdf_path).exists():
        print(f"❌ 错误: 文件不存在 - {pdf_path}")
        return False
    
    try:
        # 1. 初始化处理器
        print("\n📋 初始化处理器...")
        pdf_processor = PDFProcessor()
        ocr_engine = UnifiedOCREngine(
            use_gpu=False,  # 使用CPU避免GPU问题
            enable_number_enhancement=True,
            confidence_threshold=0.3
        )
        circle_detector = CircleDetector()
        
        # 创建输出目录
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 2. 处理PDF文件
        print(f"\n📄 处理PDF文件...")
        start_time = time.time()
        
        # 提取PDF页面
        images = pdf_processor.extract_images_from_pdf(pdf_path)
        print(f"📊 提取到 {len(images)} 页")
        
        # 处理结果统计
        all_results = []
        total_text_count = 0
        total_number_count = 0
        
        # 3. 逐页处理
        for page_num, image in enumerate(images, 1):
            print(f"\n📖 处理第 {page_num} 页...")
            
            # OCR识别
            ocr_results = ocr_engine.detect_text_regions(image)
            
            # 统计结果
            page_text_count = len(ocr_results)
            page_number_count = sum(1 for r in ocr_results if any(c.isdigit() for c in r.get('text', '')))
            
            total_text_count += page_text_count
            total_number_count += page_number_count
            
            print(f"  ✅ 检测到 {page_text_count} 个文本区域")
            print(f"  🔢 其中数字: {page_number_count} 个")
            
            # 显示前10个识别结果
            print(f"\n  📝 识别结果示例:")
            for i, result in enumerate(ocr_results[:10]):
                text = result.get('text', '')
                confidence = result.get('confidence', 0)
                engine = result.get('engine', 'unknown')
                is_enhanced = result.get('is_number_enhanced', False)
                
                has_digit = any(c.isdigit() for c in text)
                status = "🔢" if has_digit else "🔤"
                enhancement = " (增强)" if is_enhanced else ""
                
                print(f"    {i+1}. {status} '{text}' (置信度: {confidence:.3f}, 引擎: {engine}{enhancement})")
            
            if len(ocr_results) > 10:
                print(f"    ... 还有 {len(ocr_results) - 10} 个结果")
            
            # 保存页面结果
            page_results = {
                'page': page_num,
                'total_regions': page_text_count,
                'number_regions': page_number_count,
                'results': ocr_results
            }
            all_results.append(page_results)
            
            # 创建可视化
            vis_image = create_visualization(image, ocr_results)
            vis_path = output_dir / f"page_{page_num}_visualization.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            print(f"  💾 可视化已保存: {vis_path}")
            
            # 保存原始图像
            raw_path = output_dir / f"page_{page_num}_raw.jpg"
            cv2.imwrite(str(raw_path), image)
        
        # 4. 保存完整结果
        results_path = output_dir / "ocr_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整结果已保存: {results_path}")
        
        # 5. 显示统计信息
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n📊 处理统计:")
        print(f"  总页数: {len(images)} 页")
        print(f"  总文本区域: {total_text_count} 个")
        print(f"  总数字区域: {total_number_count} 个")
        if total_text_count > 0:
            print(f"  数字占比: {total_number_count/total_text_count*100:.1f}%")
        else:
            print(f"  数字占比: 无法计算 (没有检测到文本)")
        print(f"  处理时间: {processing_time:.2f} 秒")
        if len(images) > 0:
            print(f"  平均速度: {len(images)/processing_time:.2f} 页/秒")
        else:
            print(f"  平均速度: 无法计算 (没有页面)")
        
        # 6. 创建汇总报告
        summary = {
            'pdf_file': pdf_path,
            'total_pages': len(images),
            'total_text_regions': total_text_count,
            'total_number_regions': total_number_count,
            'number_ratio': total_number_count/total_text_count if total_text_count > 0 else 0,
            'processing_time': processing_time,
            'pages_per_second': len(images)/processing_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 测试完成!")
        print(f"📁 所有结果已保存到: {output_dir}")
        
        # 7. 测试数据库存储（可选）
        if input("\n是否将结果存储到数据库? (y/n): ").lower() == 'y':
            db_path = "test_results.db"
            db_manager = DatabaseManager(db_path)
            
            for page_data in all_results:
                page_num = page_data['page']
                for result in page_data['results']:
                    db_manager.insert_ocr_result(
                        filename=Path(pdf_path).name,
                        page_number=page_num,
                        result=result
                    )
            
            print(f"✅ 结果已存储到数据库: {db_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_visualization(image, ocr_results):
    """创建可视化结果"""
    vis_image = image.copy()
    
    for result in ocr_results:
        bbox = result['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # 根据类型选择颜色
        text = result.get('text', '')
        has_digit = any(c.isdigit() for c in text)
        is_enhanced = result.get('is_number_enhanced', False)
        
        if is_enhanced:
            color = (0, 255, 0)  # 绿色 - 数字增强
            thickness = 3
        elif has_digit:
            color = (255, 165, 0)  # 橙色 - 包含数字
            thickness = 2
        else:
            color = (255, 0, 0)  # 红色 - 纯文字
            thickness = 2
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, thickness)
        
        # 添加标签（只显示置信度高的）
        if result.get('confidence', 0) > 0.5:
            label = f"{text[:20]}{'...' if len(text) > 20 else ''}"
            font_scale = 0.5
            font_thickness = 1
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # 添加背景
            cv2.rectangle(vis_image, (x, y-text_height-5), (x+text_width+2, y), color, -1)
            
            # 添加文本
            cv2.putText(vis_image, label, (x+1, y-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return vis_image

def main():
    """主函数"""
    print("🔍 路線価図OCR测试工具 v1.0")
    print("=" * 60)
    
    # 测试指定的PDF文件
    pdf_path = "/Users/park/code/rosenka_proj/test.pdf"
    
    success = test_single_pdf(pdf_path)
    
    if success:
        print("\n🎉 测试成功完成!")
        print("\n💡 提示:")
        print("  1. 查看 test_output/ 目录中的可视化结果")
        print("  2. 检查 ocr_results.json 了解详细识别结果")
        print("  3. 查看 processing_summary.json 了解处理统计")
    else:
        print("\n❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main()