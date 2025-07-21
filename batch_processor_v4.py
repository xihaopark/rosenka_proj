#!/usr/bin/env python3
"""
Batch Processor v4 - 路線価図统一处理系统
路線価図検索システム - Route Price Map Search System

核心功能：
1. PDF → 图像提取 → PaddleOCR文字识别 → 坐标信息
2. 自动生成可视化图片（高亮识别结果）
3. 存储到统一数据库（供日文客户端搜索）
4. 无需输出目录，全自动处理
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import time
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database.database_manager import DatabaseManager
from core.pdf.pdf_processor import PDFProcessor
from core.ocr.unified_ocr_engine import UnifiedOCREngine
from core.detection.circle_detector import CircleDetector
from core.utils.image_utils import enhance_image_for_ocr

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedRouteMapProcessor:
    """
    统一的路線価図处理系统
    """
    
    def __init__(self, 
                 input_dir: str,
                 db_path: str = "rosenka_unified.db",
                 use_gpu: bool = True):
        """
        Initialize unified processor.
        
        Args:
            input_dir: Directory containing PDF files
            db_path: Database file path
            use_gpu: Whether to use GPU acceleration
        """
        self.input_dir = Path(input_dir)
        self.db_path = db_path
        self.use_gpu = use_gpu
        
        # 创建可视化输出目录
        self.visualization_dir = Path("visualizations")
        self.visualization_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager(db_path)
        self.pdf_processor = PDFProcessor()
        self.ocr_engine = UnifiedOCREngine(
            use_gpu=use_gpu,
            enable_number_enhancement=True,
            confidence_threshold=0.3
        )
        self.circle_detector = CircleDetector()
        
        logger.info(f"🗺️ 路線価図统一处理系统 v4 初始化完成")
        logger.info(f"📁 输入目录: {self.input_dir}")
        logger.info(f"🗄️ 数据库: {self.db_path}")
        logger.info(f"🎨 可视化目录: {self.visualization_dir}")
        logger.info(f"🚀 GPU加速: {use_gpu}")
    
    def preprocess_image(self, image):
        """
        Enhanced preprocessing for route price maps.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # 检查输入图像的通道数
        if len(image.shape) == 2:
            # 灰度图像，转换为BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # 单通道图像，转换为BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA图像，转换为BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Apply multiple enhancement techniques
        enhanced = enhance_image_for_ocr(image)
        
        # Additional preprocessing specific to route price maps
        # Remove background lines and noise
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced
        
        # Apply morphological operations to remove thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for OCR
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return cleaned_bgr
    
    def create_visualization(self, image, text_regions, circles, pdf_name, page_num):
        """
        创建可视化图片，高亮显示识别结果
        
        Args:
            image: 原始图像
            text_regions: 文字识别结果
            circles: 圆形检测结果
            pdf_name: PDF文件名
            page_num: 页码
            
        Returns:
            可视化图片路径
        """
        vis_image = image.copy()
        
        # 绘制文字区域（绿色框）
        for region in text_regions:
            bbox = region['bbox']
            text = region['text']
            confidence = region['confidence']
            
            # 绘制边界框
            cv2.rectangle(vis_image, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         (0, 255, 0), 2)
            
            # 添加文字标签
            label = f"{text} ({confidence:.2f})"
            cv2.putText(vis_image, label, 
                       (bbox['x'], bbox['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制圆形区域（红色圆）
        for circle in circles:
            center = (circle['x'], circle['y'])
            radius = circle['radius']
            confidence = circle['confidence']
            
            # 绘制圆形
            cv2.circle(vis_image, center, radius, (0, 0, 255), 2)
            cv2.circle(vis_image, center, 2, (255, 0, 0), -1)
            
            # 添加置信度标签
            label = f"Circle ({confidence:.2f})"
            cv2.putText(vis_image, label, 
                       (center[0] - 30, center[1] - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 保存可视化图片
        vis_filename = f"{pdf_name}_page_{page_num}_visualization.jpg"
        vis_path = self.visualization_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_image)
        
        logger.info(f"🎨 可视化图片已保存: {vis_path}")
        return str(vis_path)
    
    def extract_location_info(self, pdf_path: Path):
        """
        从PDF路径提取位置信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            位置信息字典
        """
        path_parts = pdf_path.parts
        
        # 尝试从路径提取位置信息
        # 例如: /data/大阪府/吹田市/藤白台１/43009.pdf
        location_info = {
            'prefecture': '',
            'city': '',
            'district': '',
            'filename': pdf_path.name
        }
        
        if len(path_parts) >= 3:
            location_info['district'] = path_parts[-2]  # 藤白台１
        if len(path_parts) >= 4:
            location_info['city'] = path_parts[-3]     # 吹田市
        if len(path_parts) >= 5:
            location_info['prefecture'] = path_parts[-4]  # 大阪府
        
        return location_info
    
    def process_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing results
        """
        logger.info(f"📄 处理PDF: {pdf_path.name}")
        
        try:
            # 提取位置信息
            location_info = self.extract_location_info(pdf_path)
            
            # Extract images from PDF
            images = self.pdf_processor.extract_images_from_pdf(str(pdf_path))
            
            if not images:
                logger.warning(f"⚠️ 无法从PDF提取图像: {pdf_path.name}")
                return {
                    'pdf_path': str(pdf_path),
                    'status': 'no_images',
                    'pages_processed': 0,
                    'total_text_regions': 0,
                    'total_circles': 0,
                    'processing_time': 0
                }
            
            # Store PDF info in database
            pdf_id = self.db_manager.add_pdf_file(
                filename=pdf_path.name,
                filepath=str(pdf_path),
                total_pages=len(images)
            )
            
            total_text_regions = 0
            total_circles = 0
            start_time = time.time()
            
            # Process each page
            for page_num, image in enumerate(images):
                logger.info(f"📃 处理页面 {page_num + 1}/{len(images)}")
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
                
                # Detect text regions using PaddleOCR
                text_regions = self.ocr_engine.detect_text_regions(processed_image)
                
                # Detect circles
                circles = self.circle_detector.detect_circles(processed_image)
                
                # Create visualization
                vis_path = self.create_visualization(
                    image, text_regions, circles, 
                    pdf_path.stem, page_num
                )
                
                # Store results in database
                page_id = self.db_manager.add_page(
                    pdf_id=pdf_id,
                    page_number=page_num + 1,
                    image_width=image.shape[1],
                    image_height=image.shape[0]
                )
                
                # Store text regions with location info
                for region in text_regions:
                    self.db_manager.add_text_region(
                        page_id=page_id,
                        x=region['bbox']['x'],
                        y=region['bbox']['y'],
                        width=region['bbox']['width'],
                        height=region['bbox']['height'],
                        text=region['text'],
                        confidence=region['confidence'],
                        engine=region['engine']
                    )
                
                # Store circles
                for circle in circles:
                    self.db_manager.add_circle(
                        page_id=page_id,
                        x=circle['x'],
                        y=circle['y'],
                        radius=circle['radius'],
                        confidence=circle['confidence']
                    )
                
                total_text_regions += len(text_regions)
                total_circles += len(circles)
                
                logger.info(f"📊 页面 {page_num + 1}: {len(text_regions)} 文字区域, {len(circles)} 圆形")
            
            processing_time = time.time() - start_time
            
            result = {
                'pdf_path': str(pdf_path),
                'status': 'success',
                'pages_processed': len(images),
                'total_text_regions': total_text_regions,
                'total_circles': total_circles,
                'processing_time': processing_time,
                'location_info': location_info
            }
            
            logger.info(f"✅ 完成 {pdf_path.name}: {total_text_regions} 文字区域, {total_circles} 圆形 ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 处理失败 {pdf_path.name}: {e}")
            return {
                'pdf_path': str(pdf_path),
                'status': 'error',
                'error': str(e),
                'pages_processed': 0,
                'total_text_regions': 0,
                'total_circles': 0,
                'processing_time': 0
            }
    
    def run_batch_processing(self) -> List[Dict]:
        """
        Run batch processing on all PDF files in input directory.
        
        Returns:
            List of processing results
        """
        logger.info("🚀 开始批量处理路線価図")
        
        # Find all PDF files
        pdf_files = list(self.input_dir.glob("**/*.pdf"))  # 递归搜索所有PDF
        
        if not pdf_files:
            logger.warning(f"⚠️ 在 {self.input_dir} 中未找到PDF文件")
            return []
        
        logger.info(f"📁 找到 {len(pdf_files)} 个PDF文件")
        
        results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"🔄 处理文件 {i}/{len(pdf_files)}: {pdf_file.name}")
            
            result = self.process_pdf(pdf_file)
            results.append(result)
            
            # Save progress
            self.save_progress(results)
        
        # Generate summary
        self.generate_summary(results)
        
        logger.info("🎉 批量处理完成！")
        return results
    
    def save_progress(self, results: List[Dict]):
        """Save processing progress to JSON file."""
        progress_file = self.visualization_dir / "processing_progress.json"
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self, results: List[Dict]):
        """Generate processing summary."""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        total_text_regions = sum(r['total_text_regions'] for r in successful)
        total_circles = sum(r['total_circles'] for r in successful)
        total_time = sum(r['processing_time'] for r in successful)
        
        summary = {
            'total_files': len(results),
            'successful_files': len(successful),
            'failed_files': len(failed),
            'total_text_regions': total_text_regions,
            'total_circles': total_circles,
            'total_processing_time': total_time,
            'average_time_per_file': total_time / len(successful) if successful else 0,
            'failed_files': [r['pdf_path'] for r in failed],
            'database_path': self.db_path,
            'visualization_dir': str(self.visualization_dir)
        }
        
        # Save summary
        summary_file = self.visualization_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("📊 处理汇总:")
        logger.info(f"  📁 总文件数: {summary['total_files']}")
        logger.info(f"  ✅ 成功: {summary['successful_files']}")
        logger.info(f"  ❌ 失败: {summary['failed_files']}")
        logger.info(f"  📝 总文字区域: {summary['total_text_regions']}")
        logger.info(f"  ⭕ 总圆形: {summary['total_circles']}")
        logger.info(f"  ⏱️ 总时间: {summary['total_processing_time']:.2f}s")
        logger.info(f"  📈 平均时间/文件: {summary['average_time_per_file']:.2f}s")
        logger.info(f"  🗄️ 数据库: {self.db_path}")
        logger.info(f"  🎨 可视化目录: {self.visualization_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="路線価図统一处理系统 v4")
    parser.add_argument("input_dir", help="包含PDF文件的输入目录")
    parser.add_argument("--db", default="rosenka_unified.db", help="数据库文件路径")
    parser.add_argument("--cpu", action="store_true", help="使用CPU而非GPU")
    
    args = parser.parse_args()
    
    # Initialize and run unified processor
    processor = UnifiedRouteMapProcessor(
        input_dir=args.input_dir,
        db_path=args.db,
        use_gpu=not args.cpu
    )
    
    results = processor.run_batch_processing()
    
    print(f"\n🎉 路線価図处理完成！")
    print(f"🗄️ 数据库: {args.db}")
    print(f"🎨 可视化图片: visualizations/")
    print(f"📊 处理结果: {len([r for r in results if r['status'] == 'success'])} 成功, {len([r for r in results if r['status'] == 'error'])} 失败")

if __name__ == "__main__":
    main() 