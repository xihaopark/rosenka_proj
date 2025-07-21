"""
多模型融合引擎
整合CRAFT、TextSnake、DBNet++等文字检测器的结果
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 导入各个检测器
from .craft_detector import CRAFTDetector
try:
    from .textsnake_detector import TextSnakeDetector
except ImportError:
    TextSnakeDetector = None
try:
    from .dbnet_detector import DBNetDetector
except ImportError:
    DBNetDetector = None

class TextDetectionFusion:
    """文字检测融合引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 初始化各个检测器
        self.detectors = {}
        self._init_detectors()
        
        # 融合参数
        self.nms_threshold = self.config.get('nms_threshold', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.max_detections = self.config.get('max_detections', 1000)
        
        # 权重配置
        self.detector_weights = {
            'craft': self.config.get('craft_weight', 1.0),
            'textsnake': self.config.get('textsnake_weight', 0.8),
            'dbnet': self.config.get('dbnet_weight', 0.9)
        }
    
    def _init_detectors(self):
        """初始化各个检测器"""
        try:
            # CRAFT检测器
            self.detectors['craft'] = CRAFTDetector(
                self.config.get('craft_model_path')
            )
            self.logger.info("CRAFT检测器初始化成功")
        except Exception as e:
            self.logger.error(f"CRAFT检测器初始化失败: {e}")
        
        # TextSnake检测器
        if TextSnakeDetector:
            try:
                self.detectors['textsnake'] = TextSnakeDetector(
                    self.config.get('textsnake_model_path')
                )
                self.logger.info("TextSnake检测器初始化成功")
            except Exception as e:
                self.logger.error(f"TextSnake检测器初始化失败: {e}")
        
        # DBNet检测器
        if DBNetDetector:
            try:
                self.detectors['dbnet'] = DBNetDetector(
                    self.config.get('dbnet_model_path')
                )
                self.logger.info("DBNet检测器初始化成功")
            except Exception as e:
                self.logger.error(f"DBNet检测器初始化失败: {e}")
    
    def detect_text_parallel(self, image: np.ndarray) -> List[Dict]:
        """
        并行文字检测
        
        Args:
            image: 输入图像
            
        Returns:
            融合后的检测结果
        """
        if not self.detectors:
            self.logger.error("没有可用的检测器")
            return []
        
        # 并行执行各个检测器
        with ThreadPoolExecutor(max_workers=len(self.detectors)) as executor:
            futures = {}
            
            for name, detector in self.detectors.items():
                future = executor.submit(self._safe_detect, detector, image, name)
                futures[future] = name
            
            # 收集结果
            detection_results = {}
            for future in futures:
                name = futures[future]
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    detection_results[name] = result
                except Exception as e:
                    self.logger.error(f"{name}检测失败: {e}")
                    detection_results[name] = []
        
        # 融合结果
        fused_results = self.fuse_detections(detection_results)
        
        return fused_results
    
    def _safe_detect(self, detector, image: np.ndarray, detector_name: str) -> List[Dict]:
        """
        安全的检测执行
        
        Args:
            detector: 检测器实例
            image: 输入图像
            detector_name: 检测器名称
            
        Returns:
            检测结果
        """
        try:
            if detector_name == 'craft':
                boxes = detector.detect_text_regions(image)
                return self._convert_craft_results(boxes, detector_name)
            elif detector_name == 'textsnake':
                results = detector.detect_curved_text(image)
                return self._convert_textsnake_results(results, detector_name)
            elif detector_name == 'dbnet':
                results = detector.detect_oriented_text(image)
                return self._convert_dbnet_results(results, detector_name)
            else:
                return []
        except Exception as e:
            self.logger.error(f"{detector_name}检测执行失败: {e}")
            return []
    
    def _convert_craft_results(self, boxes: List[np.ndarray], detector_name: str) -> List[Dict]:
        """转换CRAFT检测结果格式"""
        results = []
        for i, box in enumerate(boxes):
            if box is not None and len(box) == 4:
                results.append({
                    'id': f"{detector_name}_{i}",
                    'polygon': box.astype(np.float32),
                    'bbox': self._polygon_to_bbox(box),
                    'confidence': 0.8,  # CRAFT默认置信度
                    'detector': detector_name,
                    'text_type': 'regular',
                    'area': self._calculate_polygon_area(box)
                })
        return results
    
    def _convert_textsnake_results(self, results: List, detector_name: str) -> List[Dict]:
        """转换TextSnake检测结果格式"""
        converted = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                converted.append({
                    'id': f"{detector_name}_{i}",
                    'polygon': result.get('polygon', []),
                    'bbox': result.get('bbox', []),
                    'confidence': result.get('confidence', 0.7),
                    'detector': detector_name,
                    'text_type': 'curved',
                    'area': result.get('area', 0)
                })
            else:
                # 处理其他格式
                converted.append({
                    'id': f"{detector_name}_{i}",
                    'polygon': result if isinstance(result, np.ndarray) else [],
                    'bbox': self._polygon_to_bbox(result) if isinstance(result, np.ndarray) else [],
                    'confidence': 0.7,
                    'detector': detector_name,
                    'text_type': 'curved',
                    'area': self._calculate_polygon_area(result) if isinstance(result, np.ndarray) else 0
                })
        return converted
    
    def _convert_dbnet_results(self, results: List, detector_name: str) -> List[Dict]:
        """转换DBNet检测结果格式"""
        converted = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                converted.append({
                    'id': f"{detector_name}_{i}",
                    'polygon': result.get('polygon', []),
                    'bbox': result.get('bbox', []),
                    'confidence': result.get('confidence', 0.75),
                    'detector': detector_name,
                    'text_type': 'oriented',
                    'area': result.get('area', 0)
                })
            else:
                converted.append({
                    'id': f"{detector_name}_{i}",
                    'polygon': result if isinstance(result, np.ndarray) else [],
                    'bbox': self._polygon_to_bbox(result) if isinstance(result, np.ndarray) else [],
                    'confidence': 0.75,
                    'detector': detector_name,
                    'text_type': 'oriented',
                    'area': self._calculate_polygon_area(result) if isinstance(result, np.ndarray) else 0
                })
        return converted
    
    def fuse_detections(self, detection_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        融合多个检测器的结果
        
        Args:
            detection_results: 各检测器的结果
            
        Returns:
            融合后的结果
        """
        all_detections = []
        
        # 收集所有检测结果
        for detector_name, detections in detection_results.items():
            # 应用检测器权重
            weight = self.detector_weights.get(detector_name, 1.0)
            
            for detection in detections:
                detection = detection.copy()
                detection['confidence'] *= weight
                detection['weighted_confidence'] = detection['confidence']
                all_detections.append(detection)
        
        if not all_detections:
            return []
        
        # 按置信度排序
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 应用NMS
        nms_results = self.apply_nms(all_detections)
        
        # 过滤低置信度检测
        filtered_results = [
            det for det in nms_results 
            if det['confidence'] >= self.confidence_threshold
        ]
        
        # 限制最大检测数量
        if len(filtered_results) > self.max_detections:
            filtered_results = filtered_results[:self.max_detections]
        
        return filtered_results
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            NMS后的结果
        """
        if not detections:
            return []
        
        # 转换为numpy数组以便处理
        boxes = []
        scores = []
        
        for det in detections:
            if len(det['bbox']) == 4:
                boxes.append(det['bbox'])
                scores.append(det['confidence'])
        
        if not boxes:
            return detections
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # 使用OpenCV的NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # 返回保留的检测结果
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集区域
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _polygon_to_bbox(self, polygon: np.ndarray) -> List[float]:
        """将多边形转换为边界框"""
        if len(polygon) == 0:
            return [0, 0, 0, 0]
        
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        
        return [
            float(np.min(x_coords)),
            float(np.min(y_coords)),
            float(np.max(x_coords)),
            float(np.max(y_coords))
        ]
    
    def _calculate_polygon_area(self, polygon: np.ndarray) -> float:
        """计算多边形面积"""
        if len(polygon) < 3:
            return 0.0
        
        return float(cv2.contourArea(polygon))
    
    def detect_with_adaptive_threshold(self, image: np.ndarray) -> List[Dict]:
        """
        自适应阈值检测
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果
        """
        # 分析图像特征
        image_features = self._analyze_image_features(image)
        
        # 根据图像特征调整参数
        self._adjust_parameters_by_features(image_features)
        
        # 执行检测
        results = self.detect_text_parallel(image)
        
        return results
    
    def _analyze_image_features(self, image: np.ndarray) -> Dict:
        """分析图像特征"""
        # 计算图像统计信息
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'mean_brightness': np.mean(gray),
            'brightness_std': np.std(gray),
            'contrast': np.max(gray) - np.min(gray),
            'image_size': image.shape[:2],
            'aspect_ratio': image.shape[1] / image.shape[0]
        }
        
        return features
    
    def _adjust_parameters_by_features(self, features: Dict):
        """根据图像特征调整参数"""
        # 根据亮度调整置信度阈值
        if features['mean_brightness'] < 100:  # 暗图像
            self.confidence_threshold *= 0.9
        elif features['mean_brightness'] > 200:  # 亮图像
            self.confidence_threshold *= 1.1
        
        # 根据对比度调整NMS阈值
        if features['contrast'] < 100:  # 低对比度
            self.nms_threshold *= 0.9
        
        # 根据图像大小调整最大检测数量
        total_pixels = features['image_size'][0] * features['image_size'][1]
        if total_pixels > 2000000:  # 大图像
            self.max_detections = int(self.max_detections * 1.5)
    
    def visualize_fusion_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        可视化融合结果
        
        Args:
            image: 原始图像
            results: 检测结果
            
        Returns:
            可视化图像
        """
        vis_image = image.copy()
        
        # 不同检测器使用不同颜色
        colors = {
            'craft': (0, 255, 0),      # 绿色
            'textsnake': (255, 0, 0),  # 蓝色
            'dbnet': (0, 0, 255),      # 红色
            'fused': (255, 255, 0)     # 青色
        }
        
        for result in results:
            detector = result.get('detector', 'fused')
            color = colors.get(detector, (255, 255, 255))
            
            # 绘制边界框
            bbox = result['bbox']
            if len(bbox) == 4:
                cv2.rectangle(vis_image, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
            
            # 绘制多边形
            if len(result['polygon']) > 0:
                polygon = result['polygon'].astype(np.int32)
                cv2.polylines(vis_image, [polygon], True, color, 1)
            
            # 添加标签
            label = f"{detector}:{result['confidence']:.2f}"
            cv2.putText(vis_image, label, 
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def get_detection_statistics(self, results: List[Dict]) -> Dict:
        """
        获取检测统计信息
        
        Args:
            results: 检测结果
            
        Returns:
            统计信息
        """
        if not results:
            return {}
        
        detector_counts = {}
        confidence_stats = {}
        
        for result in results:
            detector = result.get('detector', 'unknown')
            confidence = result.get('confidence', 0)
            
            # 统计各检测器的检测数量
            detector_counts[detector] = detector_counts.get(detector, 0) + 1
            
            # 统计置信度
            if detector not in confidence_stats:
                confidence_stats[detector] = []
            confidence_stats[detector].append(confidence)
        
        # 计算平均置信度
        avg_confidence = {}
        for detector, confidences in confidence_stats.items():
            avg_confidence[detector] = np.mean(confidences)
        
        return {
            'total_detections': len(results),
            'detector_counts': detector_counts,
            'average_confidence': avg_confidence,
            'overall_avg_confidence': np.mean([r['confidence'] for r in results])
        } 