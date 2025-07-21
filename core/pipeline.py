"""
路線価図检索系统核心处理管道
整合所有组件进行端到端处理
"""

import cv2
import numpy as np
import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import time

# 导入各个组件
from .preprocessing.image_enhancer import ImageEnhancer
from .detection.scene_text.fusion_engine import TextDetectionFusion
from .detection.circle_detection.yolo_circle_detector import YOLOCircleDetector
from .detection.circle_detection.circle_ocr import CircleOCR
from .search.spatial_indexer import SpatialIndexer
from .utils.gpu_manager import GPUManager

class RosenkaV2Pipeline:
    """路線価図检索系统v2.0核心管道"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
        # 性能监控
        self.processing_stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_processing_time': 0.0
        }
    
    def _init_components(self):
        """初始化所有组件"""
        try:
            # GPU管理器
            self.gpu_manager = GPUManager()
            self.gpu_manager.optimize_memory_usage()
            
            # 图像增强器
            self.image_enhancer = ImageEnhancer()
            
            # 文字检测融合引擎
            self.text_detector = TextDetectionFusion(
                self.config.get('text_detection', {})
            )
            
            # 圆形检测器
            self.circle_detector = YOLOCircleDetector(
                self.config.get('circle_model_path')
            )
            
            # 圆内文字识别
            self.circle_ocr = CircleOCR()
            
            # 空间索引器
            self.spatial_indexer = SpatialIndexer()
            
            self.logger.info("所有组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    async def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        try:
            # 1. 加载图像
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 2. 图像预处理
            processed_image = self._preprocess_image(image)
            
            # 3. 并行检测
            detection_results = await self._parallel_detection(processed_image)
            
            # 4. 空间关系构建
            spatial_results = self._build_spatial_relationships(
                detection_results, processed_image.shape
            )
            
            # 5. 结果整合
            final_results = self._integrate_results(
                detection_results, spatial_results, image_path
            )
            
            # 6. 更新统计信息
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=True)
            
            final_results['processing_time'] = processing_time
            final_results['status'] = 'success'
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"处理图像失败 {image_path}: {e}")
            self._update_stats(time.time() - start_time, success=False)
            
            return {
                'image_path': image_path,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像文件: {image_path}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"加载图像失败: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            # 使用图像增强器处理
            enhanced_image = self.image_enhancer.enhance_map_image(image)
            return enhanced_image
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return image
    
    async def _parallel_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """并行检测"""
        async def detect_text():
            """文字检测任务"""
            try:
                return self.text_detector.detect_text_parallel(image)
            except Exception as e:
                self.logger.error(f"文字检测失败: {e}")
                return []
        
        async def detect_circles():
            """圆形检测任务"""
            try:
                circles = self.circle_detector.detect_with_fallback(image)
                # 识别圆内文字
                circles_with_text = self.circle_ocr.recognize_circle_text(image, circles)
                return circles_with_text
            except Exception as e:
                self.logger.error(f"圆形检测失败: {e}")
                return []
        
        # 并行执行检测任务
        text_task = asyncio.create_task(detect_text())
        circle_task = asyncio.create_task(detect_circles())
        
        # 等待所有任务完成
        text_detections, circle_detections = await asyncio.gather(
            text_task, circle_task, return_exceptions=True
        )
        
        # 处理异常结果
        if isinstance(text_detections, Exception):
            self.logger.error(f"文字检测异常: {text_detections}")
            text_detections = []
        
        if isinstance(circle_detections, Exception):
            self.logger.error(f"圆形检测异常: {circle_detections}")
            circle_detections = []
        
        return {
            'text_detections': text_detections,
            'circle_detections': circle_detections
        }
    
    def _build_spatial_relationships(self, detection_results: Dict, image_shape: Tuple) -> Dict:
        """构建空间关系"""
        try:
            text_detections = detection_results.get('text_detections', [])
            circle_detections = detection_results.get('circle_detections', [])
            
            # 构建空间索引
            spatial_index = self.spatial_indexer.build_index(
                text_detections, circle_detections, image_shape
            )
            
            # 计算空间关系
            relationships = self.spatial_indexer.calculate_relationships(
                text_detections, circle_detections
            )
            
            return {
                'spatial_index': spatial_index,
                'relationships': relationships
            }
            
        except Exception as e:
            self.logger.error(f"空间关系构建失败: {e}")
            return {'spatial_index': None, 'relationships': []}
    
    def _integrate_results(self, detection_results: Dict, spatial_results: Dict, 
                          image_path: str) -> Dict[str, Any]:
        """整合处理结果"""
        # 统计信息
        text_stats = self._calculate_text_stats(detection_results['text_detections'])
        circle_stats = self._calculate_circle_stats(detection_results['circle_detections'])
        
        return {
            'image_path': image_path,
            'text_detections': detection_results['text_detections'],
            'circle_detections': detection_results['circle_detections'],
            'spatial_relationships': spatial_results['relationships'],
            'statistics': {
                'text_stats': text_stats,
                'circle_stats': circle_stats,
                'total_detections': len(detection_results['text_detections']) + 
                                  len(detection_results['circle_detections'])
            }
        }
    
    def _calculate_text_stats(self, text_detections: List[Dict]) -> Dict:
        """计算文字检测统计"""
        if not text_detections:
            return {'count': 0, 'avg_confidence': 0.0}
        
        total_confidence = sum(det.get('confidence', 0) for det in text_detections)
        avg_confidence = total_confidence / len(text_detections)
        
        detector_counts = {}
        for det in text_detections:
            detector = det.get('detector', 'unknown')
            detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        return {
            'count': len(text_detections),
            'avg_confidence': avg_confidence,
            'detector_distribution': detector_counts
        }
    
    def _calculate_circle_stats(self, circle_detections: List[Dict]) -> Dict:
        """计算圆形检测统计"""
        if not circle_detections:
            return {'count': 0, 'recognized_count': 0, 'avg_confidence': 0.0}
        
        recognized_count = sum(1 for det in circle_detections if det.get('text'))
        total_confidence = sum(det.get('confidence', 0) for det in circle_detections)
        avg_confidence = total_confidence / len(circle_detections)
        
        return {
            'count': len(circle_detections),
            'recognized_count': recognized_count,
            'recognition_rate': recognized_count / len(circle_detections),
            'avg_confidence': avg_confidence
        }
    
    def _update_stats(self, processing_time: float, success: bool):
        """更新处理统计信息"""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['success_count'] += 1
        else:
            self.processing_stats['error_count'] += 1
        
        # 更新平均处理时间
        total_time = (self.processing_stats['avg_processing_time'] * 
                     (self.processing_stats['total_processed'] - 1) + processing_time)
        self.processing_stats['avg_processing_time'] = total_time / self.processing_stats['total_processed']
    
    async def process_batch(self, image_paths: List[str], 
                           max_concurrent: int = 4) -> List[Dict[str, Any]]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            max_concurrent: 最大并发数
            
        Returns:
            处理结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_path: str):
            async with semaphore:
                return await self.process_single_image(image_path)
        
        # 创建任务
        tasks = [process_with_semaphore(path) for path in image_paths]
        
        # 批量执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'image_path': image_paths[i],
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存处理结果
        
        Args:
            results: 处理结果
            output_path: 输出路径
        """
        try:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = self._make_serializable(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def visualize_results(self, image_path: str, results: Dict[str, Any], 
                         output_path: str):
        """
        可视化处理结果
        
        Args:
            image_path: 原始图像路径
            results: 处理结果
            output_path: 输出图像路径
        """
        try:
            # 加载原始图像
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法加载图像: {image_path}")
                return
            
            vis_image = image.copy()
            
            # 绘制文字检测结果
            text_detections = results.get('text_detections', [])
            for det in text_detections:
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    cv2.rectangle(vis_image, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
            
            # 绘制圆形检测结果
            circle_detections = results.get('circle_detections', [])
            for det in circle_detections:
                center = det.get('center', (0, 0))
                radius = det.get('radius', 0)
                text = det.get('text', '')
                
                # 绘制圆形
                cv2.circle(vis_image, center, radius, (0, 0, 255), 2)
                
                # 绘制文字
                if text:
                    cv2.putText(vis_image, text, 
                              (center[0] - 20, center[1] - radius - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 保存可视化结果
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"可视化结果已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.processing_stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_processing_time': 0.0
        } 