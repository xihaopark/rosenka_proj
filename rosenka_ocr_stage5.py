#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosenka OCR Stage 5 - Advanced Pipeline
路線価図OCR系统 Stage 5 主处理管线

整合所有Stage 5组件的主处理管线：
1. 增强图像预处理 (EnhancedImagePreprocessor)
2. 多尺度OCR检测 (MultiScaleOCRDetector)  
3. 空间智能分析 (SpatialIntelligenceEngine)
4. 智能后处理 (IntelligentPostProcessor)
5. 可视化结果生成

特性:
- 高召回率路線价和街区番号检测
- 空间上下文感知分类
- 智能结果优化和质量控制
- 完整的调试和可视化支持
"""

import cv2
import numpy as np
import logging
import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import os

# 导入Stage 5组件
from core.ocr.enhanced_image_preprocessor import EnhancedImagePreprocessor
from core.ocr.multi_scale_ocr_detector import MultiScaleOCRDetector
from core.ocr.spatial_intelligence_engine import SpatialIntelligenceEngine
from core.ocr.intelligent_post_processor import IntelligentPostProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RosenkaOCRStage5:
    """路線価図OCR系统 Stage 5 主处理管线"""
    
    def __init__(self, use_gpu: bool = False, debug_mode: bool = False, 
                 output_dir: str = "stage5_output"):
        """
        初始化Stage 5 OCR系统
        
        Args:
            use_gpu: 是否使用GPU加速
            debug_mode: 是否启用调试模式
            output_dir: 输出目录
        """
        self.use_gpu = use_gpu
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建调试目录
        if debug_mode:
            self.debug_dir = self.output_dir / "debug"
            self.debug_dir.mkdir(exist_ok=True)
        
        # 初始化所有组件
        logger.info("初始化Stage 5组件...")
        self.preprocessor = EnhancedImagePreprocessor(debug_mode=debug_mode)
        self.detector = MultiScaleOCRDetector(use_gpu=use_gpu, debug_mode=debug_mode)
        self.spatial_engine = SpatialIntelligenceEngine(debug_mode=debug_mode)
        self.post_processor = IntelligentPostProcessor(debug_mode=debug_mode)
        
        # 性能统计
        self.performance_stats = {
            'total_pages': 0,
            'total_time': 0,
            'total_detections': 0,
            'successful_pages': 0
        }
        
        logger.info(f"Stage 5系统初始化完成 (GPU: {use_gpu}, Debug: {debug_mode})")
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理结果字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        logger.info(f"开始处理PDF: {pdf_path.name}")
        start_time = time.time()
        
        try:
            # 1. PDF转图像
            images = self._pdf_to_images(pdf_path)
            logger.info(f"提取了 {len(images)} 页图像")
            
            # 2. 处理每一页
            all_results = []
            for page_num, image in enumerate(images):
                logger.info(f"处理第 {page_num + 1} 页...")
                
                page_result = self._process_single_page(
                    image, f"{pdf_path.stem}_page_{page_num + 1}"
                )
                page_result['page_number'] = page_num + 1
                page_result['pdf_name'] = pdf_path.name
                all_results.append(page_result)
                
                self.performance_stats['total_pages'] += 1
                if page_result['success']:
                    self.performance_stats['successful_pages'] += 1
                self.performance_stats['total_detections'] += len(page_result.get('detections', []))
            
            # 3. 整合结果
            pdf_result = {
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.name,
                'total_pages': len(images),
                'page_results': all_results,
                'processing_time': time.time() - start_time,
                'success': True,
                'summary': self._generate_summary(all_results)
            }
            
            # 4. 保存结果
            self._save_pdf_results(pdf_result)
            
            # 5. 生成可视化
            if self.debug_mode:
                self._generate_visualization(images, all_results, pdf_path.stem)
            
            logger.info(f"PDF处理完成: {pdf_path.name}, 耗时: {pdf_result['processing_time']:.2f}秒")
            return pdf_result
            
        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            return {
                'pdf_path': str(pdf_path),
                'pdf_name': pdf_path.name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """将PDF转换为图像列表"""
        images = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 设置高分辨率
                mat = fitz.Matrix(2.0, 2.0)  # 2倍缩放
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为numpy数组
                img_data = pix.tobytes("ppm")
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    images.append(image)
                else:
                    logger.warning(f"第 {page_num + 1} 页转换失败")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF转图像失败: {e}")
            raise
        
        return images
    
    def _process_single_page(self, image: np.ndarray, page_name: str) -> Dict:
        """处理单页图像"""
        try:
            page_start_time = time.time()
            
            # 1. 图像预处理
            logger.debug(f"开始预处理: {page_name}")
            preprocessed_variants = self.preprocessor.preprocess_for_ocr(image, page_name)
            
            # 2. 多尺度OCR检测
            logger.debug(f"开始OCR检测: {page_name}")
            all_detections = []
            for variant_name, processed_image in preprocessed_variants:
                detections = self.detector.detect_multi_scale(processed_image, f"{page_name}_{variant_name}")
                all_detections.extend(detections)
            
            # 去重初步处理
            unique_detections = self._deduplicate_initial(all_detections)
            logger.debug(f"OCR检测完成: {len(unique_detections)}个候选")
            
            # 3. 空间智能分析
            logger.debug(f"开始空间分析: {page_name}")
            spatial_result = self.spatial_engine.analyze_spatial_context(
                image, unique_detections, page_name
            )
            
            # 4. 智能后处理
            logger.debug(f"开始后处理: {page_name}")
            final_detections = self.post_processor.process_detections(
                spatial_result['classified_detections'], page_name
            )
            
            # 5. 结果整理
            page_result = {
                'page_name': page_name,
                'success': True,
                'processing_time': time.time() - page_start_time,
                'preprocessing_variants': len(preprocessed_variants),
                'raw_detections': len(all_detections),
                'unique_detections': len(unique_detections),
                'spatial_analysis': spatial_result,
                'detections': final_detections,
                'statistics': self._calculate_page_statistics(final_detections)
            }
            
            logger.debug(f"页面处理完成: {page_name}, {len(final_detections)}个最终检测")
            return page_result
            
        except Exception as e:
            logger.error(f"页面处理失败 {page_name}: {e}")
            return {
                'page_name': page_name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - page_start_time if 'page_start_time' in locals() else 0
            }
    
    def _deduplicate_initial(self, detections: List[Dict]) -> List[Dict]:
        """初步去重（简单的重叠检测）"""
        if not detections:
            return []
        
        unique = []
        for detection in detections:
            # 检查是否与已有检测重叠
            is_duplicate = False
            for existing in unique:
                if self._calculate_overlap_ratio(detection, existing) > 0.7:
                    # 保留置信度更高的
                    if detection.get('confidence', 0) > existing.get('confidence', 0):
                        unique.remove(existing)
                        unique.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def _calculate_overlap_ratio(self, det1: Dict, det2: Dict) -> float:
        """计算两个检测的重叠比例"""
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        # 转换为矩形
        x1_min = min(p[0] for p in bbox1)
        y1_min = min(p[1] for p in bbox1)
        x1_max = max(p[0] for p in bbox1)
        y1_max = max(p[1] for p in bbox1)
        
        x2_min = min(p[0] for p in bbox2)
        y2_min = min(p[1] for p in bbox2)
        x2_max = max(p[0] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        # 计算重叠区域
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = overlap_x * overlap_y
        
        # 计算联合区域
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - overlap_area
        
        return overlap_area / union_area if union_area > 0 else 0
    
    def _calculate_page_statistics(self, detections: List[Dict]) -> Dict:
        """计算页面统计信息"""
        if not detections:
            return {}
        
        # 按类型统计
        type_counts = {}
        confidence_scores = []
        
        for detection in detections:
            detection_type = detection.get('format_type', 'unknown')
            type_counts[detection_type] = type_counts.get(detection_type, 0) + 1
            confidence_scores.append(detection.get('final_comprehensive_score', 0))
        
        return {
            'total_detections': len(detections),
            'type_distribution': type_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'high_confidence_count': sum(1 for score in confidence_scores if score > 0.8),
            'valid_detections': sum(1 for det in detections if det.get('quality_score', 0) > 0.5)
        }
    
    def _generate_summary(self, page_results: List[Dict]) -> Dict:
        """生成处理摘要"""
        successful_pages = [r for r in page_results if r.get('success', False)]
        
        total_detections = sum(len(r.get('detections', [])) for r in successful_pages)
        total_processing_time = sum(r.get('processing_time', 0) for r in page_results)
        
        # 汇总类型分布
        overall_type_counts = {}
        for page_result in successful_pages:
            stats = page_result.get('statistics', {})
            type_dist = stats.get('type_distribution', {})
            for type_name, count in type_dist.items():
                overall_type_counts[type_name] = overall_type_counts.get(type_name, 0) + count
        
        return {
            'total_pages': len(page_results),
            'successful_pages': len(successful_pages),
            'failed_pages': len(page_results) - len(successful_pages),
            'total_detections': total_detections,
            'total_processing_time': total_processing_time,
            'average_time_per_page': total_processing_time / len(page_results) if page_results else 0,
            'average_detections_per_page': total_detections / len(successful_pages) if successful_pages else 0,
            'type_distribution': overall_type_counts
        }
    
    def _save_pdf_results(self, pdf_result: Dict):
        """保存PDF处理结果"""
        # 保存详细结果
        results_path = self.output_dir / f"{pdf_result['pdf_name']}_stage5_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存简化版结果（仅检测数据）
        simplified_result = {
            'pdf_name': pdf_result['pdf_name'],
            'total_pages': pdf_result['total_pages'],
            'summary': pdf_result['summary'],
            'detections_by_page': []
        }
        
        for page_result in pdf_result.get('page_results', []):
            if page_result.get('success'):
                page_detections = []
                for detection in page_result.get('detections', []):
                    page_detections.append({
                        'text': detection['text'],
                        'bbox': detection['bbox'],
                        'type': detection.get('format_type', 'unknown'),
                        'confidence': detection.get('final_comprehensive_score', 0),
                        'spatial_type': detection.get('spatial_type', 'unknown')
                    })
                simplified_result['detections_by_page'].append({
                    'page_number': page_result['page_number'],
                    'detections': page_detections
                })
        
        simplified_path = self.output_dir / f"{pdf_result['pdf_name']}_detections.json"
        with open(simplified_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已保存: {results_path}, {simplified_path}")
    
    def _generate_visualization(self, images: List[np.ndarray], 
                               page_results: List[Dict], 
                               pdf_name: str):
        """生成可视化结果"""
        viz_dir = self.output_dir / "visualizations" / pdf_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image, page_result) in enumerate(zip(images, page_results)):
            if not page_result.get('success'):
                continue
            
            page_num = i + 1
            viz_image = self._create_visualization_image(image, page_result)
            
            # 保存可视化图像
            viz_path = viz_dir / f"page_{page_num:02d}_visualization.jpg"
            cv2.imwrite(str(viz_path), viz_image)
            
            logger.debug(f"可视化已保存: {viz_path}")
    
    def _create_visualization_image(self, image: np.ndarray, page_result: Dict) -> np.ndarray:
        """创建可视化图像"""
        viz_image = image.copy()
        
        # 颜色映射
        color_map = {
            'block_number': (0, 255, 0),      # 绿色 - 街区番号
            'route_price': (255, 0, 0),       # 蓝色 - 路線価
            'complex_address': (0, 255, 255), # 黄色 - 复合地址
            'partial_number': (255, 165, 0),  # 橙色 - 部分数字
            'unknown': (128, 128, 128)        # 灰色 - 未知
        }
        
        # 绘制检测结果
        detections = page_result.get('detections', [])
        for detection in detections:
            bbox = detection['bbox']
            text = detection['text']
            format_type = detection.get('format_type', 'unknown')
            confidence = detection.get('final_comprehensive_score', 0)
            spatial_type = detection.get('spatial_type', '')
            
            # 选择颜色
            color = color_map.get(format_type, (255, 255, 255))
            
            # 绘制边界框
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(viz_image, [points], True, color, 2)
            
            # 绘制文本标签
            label = f"{text}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            # 计算标签位置
            label_x = int(min(p[0] for p in bbox))
            label_y = int(min(p[1] for p in bbox)) - 10
            if label_y < 20:
                label_y = int(max(p[1] for p in bbox)) + 20
            
            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(viz_image, (label_x, label_y - label_h - 5), 
                         (label_x + label_w, label_y + 5), color, -1)
            
            # 绘制标签文字
            cv2.putText(viz_image, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加图例
        viz_image = self._add_legend(viz_image, color_map, page_result)
        
        return viz_image
    
    def _add_legend(self, image: np.ndarray, color_map: Dict, page_result: Dict) -> np.ndarray:
        """添加图例"""
        h, w = image.shape[:2]
        
        # 在右上角添加图例
        legend_x = w - 300
        legend_y = 30
        
        # 统计信息
        stats = page_result.get('statistics', {})
        total_detections = stats.get('total_detections', 0)
        
        # 绘制背景
        legend_h = len(color_map) * 25 + 60
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + legend_h), (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + legend_h), (255, 255, 255), 2)
        
        # 标题
        cv2.putText(image, f"Stage 5 Results ({total_detections})", 
                   (legend_x, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 图例项
        y_offset = 35
        for type_name, color in color_map.items():
            count = stats.get('type_distribution', {}).get(type_name, 0)
            if count > 0:
                # 绘制颜色框
                cv2.rectangle(image, (legend_x, legend_y + y_offset), 
                             (legend_x + 15, legend_y + y_offset + 15), color, -1)
                
                # 绘制文字
                label = f"{type_name}: {count}"
                cv2.putText(image, label, (legend_x + 20, legend_y + y_offset + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset += 20
        
        return image
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        stats = self.performance_stats.copy()
        
        if stats['total_pages'] > 0:
            stats['success_rate'] = stats['successful_pages'] / stats['total_pages']
            stats['average_time_per_page'] = stats['total_time'] / stats['total_pages']
            stats['average_detections_per_page'] = stats['total_detections'] / stats['successful_pages'] if stats['successful_pages'] > 0 else 0
        
        return stats

def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rosenka OCR Stage 5 - Advanced Pipeline')
    parser.add_argument('--input', required=True, help='输入PDF文件路径')
    parser.add_argument('--output', default='stage5_output', help='输出目录')
    parser.add_argument('--gpu', action='store_true', help='启用GPU加速')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 初始化系统
    ocr_system = RosenkaOCRStage5(
        use_gpu=args.gpu,
        debug_mode=args.debug,
        output_dir=args.output
    )
    
    # 处理PDF
    result = ocr_system.process_pdf(args.input)
    
    # 打印摘要
    if result['success']:
        summary = result['summary']
        print(f"\n=== Stage 5 处理完成 ===")
        print(f"PDF: {result['pdf_name']}")
        print(f"页数: {summary['total_pages']} (成功: {summary['successful_pages']})")
        print(f"总检测数: {summary['total_detections']}")
        print(f"处理时间: {summary['total_processing_time']:.2f}秒")
        print(f"平均每页: {summary['average_detections_per_page']:.1f}个检测")
        
        print(f"\n类型分布:")
        for type_name, count in summary['type_distribution'].items():
            print(f"  {type_name}: {count}")
        
        print(f"\n结果已保存至: {args.output}")
    else:
        print(f"处理失败: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()