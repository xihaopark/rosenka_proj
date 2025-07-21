#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosenka OCR Stage 5 - Simplified Version
路線価図OCR系统 Stage 5 简化版本

整合v4.0系统和Stage 5新功能的简化实现
使用现有的统一OCR引擎，添加Stage 5的可视化和分析功能
"""

import cv2
import numpy as np
import logging
import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF
import os

# 导入v4.0的工作组件
from core.ocr.unified_ocr_engine import UnifiedOCREngine

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RosenkaOCRStage5Simple:
    """路線価図OCR系统 Stage 5 简化版本"""
    
    def __init__(self, use_gpu: bool = False, debug_mode: bool = True, 
                 output_dir: str = "stage5_simple_output"):
        """
        初始化Stage 5简化版OCR系统
        
        Args:
            use_gpu: 是否使用GPU加速
            debug_mode: 是否启用调试模式
            output_dir: 输出目录
        """
        self.use_gpu = use_gpu
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建可视化目录
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # 初始化v4.0的统一OCR引擎
        logger.info("初始化Stage 5简化版系统...")
        self.ocr_engine = UnifiedOCREngine(
            use_gpu=use_gpu,
            enable_number_enhancement=True,
            confidence_threshold=0.3
        )
        
        # 路線価图专用模式匹配
        self.patterns = {
            'block_number': r'^\d{1,3}$',  # 街区番号：1-3位数字
            'route_price': r'^\d{1,4}[A-G]$',  # 路線価：数字+字母A-G
            'complex_number': r'^\d{1,3}-\d{1,3}[A-G]?$',  # 复合番号
            'price_with_unit': r'^\d{1,4}万?[A-G]?$',  # 带万字单位
            'reference_code': r'^[A-Z]\d{1,3}$',  # 参考编号：字母+数字
        }
        
        logger.info(f"Stage 5简化版系统初始化完成 (GPU: {use_gpu}, Debug: {debug_mode})")
    
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
            
            # 1. 使用v4.0的OCR引擎进行检测
            logger.debug(f"开始OCR检测: {page_name}")
            detections = self.ocr_engine.detect_text_regions(image)
            
            # 2. Stage 5增强：分类和分析
            logger.debug(f"开始分类分析: {page_name}")
            enhanced_detections = self._enhance_detections(detections)
            
            # 3. 过滤和优化
            logger.debug(f"开始过滤优化: {page_name}")
            final_detections = self._filter_and_optimize(enhanced_detections)
            
            # 4. 结果整理
            page_result = {
                'page_name': page_name,
                'success': True,
                'processing_time': time.time() - page_start_time,
                'raw_detections': len(detections),
                'enhanced_detections': len(enhanced_detections),
                'final_detections': len(final_detections),
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
    
    def _enhance_detections(self, detections: List[Dict]) -> List[Dict]:
        """增强检测结果（Stage 5 功能）"""
        enhanced = []
        
        for detection in detections:
            text = detection.get('text', '').strip()
            if not text:
                continue
            
            # 添加分类信息
            classification = self._classify_text(text)
            
            # 计算增强评分
            enhanced_score = self._calculate_enhanced_score(detection, classification)
            
            # 创建增强的检测结果
            enhanced_detection = detection.copy()
            enhanced_detection.update({
                'classification': classification,
                'enhanced_score': enhanced_score,
                'is_target_type': classification in ['block_number', 'route_price', 'complex_number'],
                'stage5_processed': True
            })
            
            enhanced.append(enhanced_detection)
        
        return enhanced
    
    def _classify_text(self, text: str) -> str:
        """分类文本类型"""
        import re
        
        # 清理文本
        cleaned_text = text.strip()
        
        # 检查各种模式
        for pattern_name, pattern in self.patterns.items():
            if re.match(pattern, cleaned_text):
                return pattern_name
        
        # 特殊情况处理
        if cleaned_text.isdigit():
            return 'pure_number'
        elif cleaned_text.isalpha() and len(cleaned_text) == 1:
            return 'single_letter'
        elif any(char in cleaned_text for char in '-/'):
            return 'complex_identifier'
        
        return 'unknown'
    
    def _calculate_enhanced_score(self, detection: Dict, classification: str) -> float:
        """计算增强评分"""
        base_score = detection.get('confidence', 0.5)
        
        # 根据分类调整评分
        if classification in ['block_number', 'route_price']:
            base_score *= 1.3  # 提高目标类型的评分
        elif classification == 'complex_number':
            base_score *= 1.2
        elif classification in ['pure_number', 'single_letter']:
            base_score *= 1.1
        elif classification == 'unknown':
            base_score *= 0.8  # 降低未知类型的评分
        
        # 文本长度调整
        text = detection.get('text', '')
        if 1 <= len(text) <= 6:  # 合理的地址长度
            base_score *= 1.1
        elif len(text) > 10:  # 过长可能是误识别
            base_score *= 0.7
        
        return min(base_score, 1.0)  # 限制在1.0以内
    
    def _filter_and_optimize(self, detections: List[Dict]) -> List[Dict]:
        """过滤和优化检测结果"""
        # 1. 基本过滤
        filtered = []
        for detection in detections:
            text = detection.get('text', '')
            enhanced_score = detection.get('enhanced_score', 0)
            
            # 过滤条件
            if (len(text) >= 1 and 
                enhanced_score >= 0.2 and 
                detection.get('classification') != 'unknown'):
                filtered.append(detection)
        
        # 2. 去重处理
        deduplicated = self._remove_duplicates(filtered)
        
        # 3. 按评分排序
        sorted_detections = sorted(deduplicated, 
                                 key=lambda x: x.get('enhanced_score', 0), 
                                 reverse=True)
        
        return sorted_detections
    
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """去除重复检测"""
        if not detections:
            return []
        
        unique = []
        for detection in detections:
            is_duplicate = False
            
            for existing in unique:
                # 检查文本相似性和位置接近性
                if (detection['text'] == existing['text'] and 
                    self._are_positions_close(detection, existing)):
                    # 保留评分更高的
                    if detection.get('enhanced_score', 0) > existing.get('enhanced_score', 0):
                        unique.remove(existing)
                        unique.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def _are_positions_close(self, det1: Dict, det2: Dict) -> bool:
        """检查两个检测的位置是否接近"""
        try:
            bbox1 = det1.get('bbox', [])
            bbox2 = det2.get('bbox', [])
            
            if not bbox1 or not bbox2:
                return False
            
            # 计算中心点
            center1 = self._calculate_center(bbox1)
            center2 = self._calculate_center(bbox2)
            
            # 计算距离
            distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            
            return distance < 50  # 50像素以内认为是重复
            
        except:
            return False
    
    def _calculate_center(self, bbox: List) -> Tuple[float, float]:
        """计算边界框中心点"""
        if isinstance(bbox[0], list):
            # 四个点的格式
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
        else:
            # 可能是其他格式，尝试解析
            xs = [bbox[0], bbox[2]]
            ys = [bbox[1], bbox[3]]
        
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _calculate_page_statistics(self, detections: List[Dict]) -> Dict:
        """计算页面统计信息"""
        if not detections:
            return {}
        
        # 按类型统计
        type_counts = {}
        confidence_scores = []
        
        for detection in detections:
            classification = detection.get('classification', 'unknown')
            type_counts[classification] = type_counts.get(classification, 0) + 1
            confidence_scores.append(detection.get('enhanced_score', 0))
        
        return {
            'total_detections': len(detections),
            'type_distribution': type_counts,
            'average_score': sum(confidence_scores) / len(confidence_scores),
            'high_score_count': sum(1 for score in confidence_scores if score > 0.8),
            'target_type_count': sum(1 for det in detections 
                                   if det.get('is_target_type', False))
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
        results_path = self.output_dir / f"{pdf_result['pdf_name']}_stage5_simple_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存简化版结果
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
                        'bbox': detection.get('bbox', []),
                        'classification': detection.get('classification', 'unknown'),
                        'enhanced_score': detection.get('enhanced_score', 0),
                        'is_target_type': detection.get('is_target_type', False)
                    })
                simplified_result['detections_by_page'].append({
                    'page_number': page_result['page_number'],
                    'detections': page_detections
                })
        
        simplified_path = self.output_dir / f"{pdf_result['pdf_name']}_detections_simple.json"
        with open(simplified_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已保存: {results_path}, {simplified_path}")
    
    def _generate_visualization(self, images: List[np.ndarray], 
                               page_results: List[Dict], 
                               pdf_name: str):
        """生成可视化结果"""
        viz_pdf_dir = self.viz_dir / pdf_name
        viz_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image, page_result) in enumerate(zip(images, page_results)):
            if not page_result.get('success'):
                continue
            
            page_num = i + 1
            viz_image = self._create_visualization_image(image, page_result)
            
            # 保存可视化图像
            viz_path = viz_pdf_dir / f"page_{page_num:02d}_stage5_visualization.jpg"
            cv2.imwrite(str(viz_path), viz_image)
            
            logger.info(f"可视化已保存: {viz_path}")
    
    def _create_visualization_image(self, image: np.ndarray, page_result: Dict) -> np.ndarray:
        """创建可视化图像"""
        viz_image = image.copy()
        
        # 颜色映射 - Stage 5 特色配色
        color_map = {
            'block_number': (0, 255, 0),      # 绿色 - 街区番号
            'route_price': (255, 0, 0),       # 蓝色 - 路線価
            'complex_number': (0, 255, 255),  # 黄色 - 复合番号
            'pure_number': (255, 165, 0),     # 橙色 - 纯数字
            'single_letter': (255, 0, 255),   # 紫色 - 单字母
            'unknown': (128, 128, 128)        # 灰色 - 未知
        }
        
        # 绘制检测结果
        detections = page_result.get('detections', [])
        for detection in detections:
            bbox = detection.get('bbox', [])
            text = detection['text']
            classification = detection.get('classification', 'unknown')
            enhanced_score = detection.get('enhanced_score', 0)
            is_target = detection.get('is_target_type', False)
            
            if not bbox:
                continue
            
            # 选择颜色
            color = color_map.get(classification, (255, 255, 255))
            
            # 如果是目标类型，使用更亮的颜色
            if is_target:
                color = tuple(min(255, int(c * 1.2)) for c in color)
            
            # 绘制边界框
            try:
                if isinstance(bbox[0], list):
                    # 四个点的格式
                    points = np.array(bbox, dtype=np.int32)
                    cv2.polylines(viz_image, [points], True, color, 2)
                    
                    # 计算标签位置
                    label_x = int(min(p[0] for p in bbox))
                    label_y = int(min(p[1] for p in bbox)) - 10
                else:
                    # 矩形格式 [x, y, w, h]
                    x, y, w, h = map(int, bbox[:4])
                    cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
                    label_x, label_y = x, y - 10
                
                if label_y < 20:
                    label_y = int(max(p[1] for p in bbox)) + 20 if isinstance(bbox[0], list) else y + h + 20
                
                # 绘制文本标签
                label = f"{text}"
                if enhanced_score > 0:
                    label += f" ({enhanced_score:.2f})"
                if is_target:
                    label = "★" + label
                
                # 绘制标签背景
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(viz_image, (label_x, label_y - label_h - 5), 
                             (label_x + label_w, label_y + 5), color, -1)
                
                # 绘制标签文字
                cv2.putText(viz_image, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                logger.warning(f"绘制检测框失败: {e}")
        
        # 添加图例
        viz_image = self._add_legend(viz_image, color_map, page_result)
        
        return viz_image
    
    def _add_legend(self, image: np.ndarray, color_map: Dict, page_result: Dict) -> np.ndarray:
        """添加图例"""
        h, w = image.shape[:2]
        
        # 在右上角添加图例
        legend_x = w - 320
        legend_y = 30
        
        # 统计信息
        stats = page_result.get('statistics', {})
        total_detections = stats.get('total_detections', 0)
        target_count = stats.get('target_type_count', 0)
        
        # 绘制背景
        legend_h = len(color_map) * 25 + 80
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + legend_h), (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + legend_h), (255, 255, 255), 2)
        
        # 标题
        cv2.putText(image, f"Stage 5 Results", 
                   (legend_x, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Total: {total_detections}, Target: {target_count}", 
                   (legend_x, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 图例项
        y_offset = 55
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

def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rosenka OCR Stage 5 - Simplified Version')
    parser.add_argument('--input', required=True, help='输入PDF文件路径')
    parser.add_argument('--output', default='stage5_simple_output', help='输出目录')
    parser.add_argument('--gpu', action='store_true', help='启用GPU加速')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 初始化系统
    ocr_system = RosenkaOCRStage5Simple(
        use_gpu=args.gpu,
        debug_mode=args.debug,
        output_dir=args.output
    )
    
    # 处理PDF
    result = ocr_system.process_pdf(args.input)
    
    # 打印摘要
    if result['success']:
        summary = result['summary']
        print(f"\n=== Stage 5 简化版处理完成 ===")
        print(f"PDF: {result['pdf_name']}")
        print(f"页数: {summary['total_pages']} (成功: {summary['successful_pages']})")
        print(f"总检测数: {summary['total_detections']}")
        print(f"处理时间: {summary['total_processing_time']:.2f}秒")
        print(f"平均每页: {summary['average_detections_per_page']:.1f}个检测")
        
        print(f"\n类型分布:")
        for type_name, count in summary['type_distribution'].items():
            print(f"  {type_name}: {count}")
        
        print(f"\n结果已保存至: {args.output}")
        print(f"可视化图像保存至: {args.output}/visualizations/")
    else:
        print(f"处理失败: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()