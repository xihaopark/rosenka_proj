#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_detector.py
圆形检测器 - 路線価図検索システム
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CircleDetector:
    """圆形检测器"""
    
    def __init__(self, min_radius: int = 5, max_radius: int = 50):
        """
        初始化圆形检测器
        
        Args:
            min_radius: 最小半径
            max_radius: 最大半径
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        logger.info(f"圆形检测器初始化完成 (半径范围: {min_radius}-{max_radius})")
    
    def detect_circles(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的圆形
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的圆形列表
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 图像预处理
            processed = self._preprocess_image(gray)
            
            # 多尺度圆形检测
            circles = self._detect_circles_multi_scale(processed)
            
            # 过滤和验证圆形
            validated_circles = self._validate_circles(circles, image.shape)
            
            logger.info(f"检测到 {len(validated_circles)} 个圆形")
            return validated_circles
            
        except Exception as e:
            logger.error(f"圆形检测失败: {e}")
            return []
    
    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            gray: 灰度图像
            
        Returns:
            预处理后的图像
        """
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def _detect_circles_multi_scale(self, image: np.ndarray) -> List[Dict]:
        """
        多尺度圆形检测
        
        Args:
            image: 预处理后的图像
            
        Returns:
            检测到的圆形列表
        """
        all_circles = []
        
        # 多种检测配置
        configs = [
            # 检测小圆圈
            {'dp': 1, 'minDist': 15, 'param1': 100, 'param2': 20, 'minRadius': 5, 'maxRadius': 20},
            {'dp': 1, 'minDist': 20, 'param1': 80, 'param2': 25, 'minRadius': 8, 'maxRadius': 25},
            {'dp': 1, 'minDist': 25, 'param1': 60, 'param2': 30, 'minRadius': 10, 'maxRadius': 30},
            
            # 检测中等圆圈
            {'dp': 1, 'minDist': 30, 'param1': 100, 'param2': 35, 'minRadius': 15, 'maxRadius': 35},
            {'dp': 2, 'minDist': 40, 'param1': 80, 'param2': 40, 'minRadius': 20, 'maxRadius': 40},
            
            # 检测大圆圈
            {'dp': 2, 'minDist': 50, 'param1': 120, 'param2': 50, 'minRadius': 25, 'maxRadius': 50}
        ]
        
        for i, config in enumerate(configs):
            circles = cv2.HoughCircles(
                image,
                cv2.HOUGH_GRADIENT,
                **config
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # 计算圆形质量分数
                    quality_score = self._evaluate_circle_quality(image, x, y, r)
                    
                    all_circles.append({
                        'x': x,
                        'y': y,
                        'radius': r,
                        'confidence': quality_score,
                        'config_id': i
                    })
        
        return all_circles
    
    def _evaluate_circle_quality(self, image: np.ndarray, x: int, y: int, r: int) -> float:
        """
        评估圆形质量
        
        Args:
            image: 图像
            x: 圆心x坐标
            y: 圆心y坐标
            r: 半径
            
        Returns:
            质量分数 (0-1)
        """
        try:
            # 创建圆形掩码
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 计算圆形区域的统计信息
            circle_region = cv2.bitwise_and(image, image, mask=mask)
            
            # 计算边缘强度
            edges = cv2.Canny(image, 50, 150)
            edge_region = cv2.bitwise_and(edges, edges, mask=mask)
            
            # 计算质量指标
            total_pixels = cv2.countNonZero(mask)
            edge_pixels = cv2.countNonZero(edge_region)
            
            if total_pixels == 0:
                return 0.0
            
            # 边缘密度
            edge_density = edge_pixels / total_pixels
            
            # 圆形度评估
            perimeter = 2 * np.pi * r
            expected_edge_pixels = perimeter * 0.8  # 假设80%的周长有边缘
            circularity = min(edge_pixels / expected_edge_pixels, 1.0) if expected_edge_pixels > 0 else 0.0
            
            # 综合质量分数
            quality = (edge_density * 0.4 + circularity * 0.6)
            
            return min(quality, 1.0)
            
        except Exception as e:
            logger.warning(f"圆形质量评估失败: {e}")
            return 0.5  # 默认质量分数
    
    def _validate_circles(self, circles: List[Dict], image_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        验证和过滤圆形
        
        Args:
            circles: 初步检测的圆形列表
            image_shape: 图像尺寸
            
        Returns:
            验证后的圆形列表
        """
        validated = []
        h, w = image_shape[:2]
        
        for circle in circles:
            x, y, r = circle['x'], circle['y'], circle['r']
            
            # 检查边界
            if (x - r < 0 or x + r >= w or y - r < 0 or y + r >= h):
                continue
            
            # 检查半径范围
            if r < self.min_radius or r > self.max_radius:
                continue
            
            # 检查质量分数
            if circle['confidence'] < 0.3:
                continue
            
            validated.append(circle)
        
        # 去除重叠的圆形
        final_circles = self._remove_overlapping_circles(validated)
        
        return final_circles
    
    def _remove_overlapping_circles(self, circles: List[Dict]) -> List[Dict]:
        """
        去除重叠的圆形
        
        Args:
            circles: 圆形列表
            
        Returns:
            去重后的圆形列表
        """
        if not circles:
            return []
        
        # 按置信度排序
        circles.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_circles = []
        
        for circle in circles:
            is_duplicate = False
            
            for existing in final_circles:
                # 计算中心点距离
                dist = np.sqrt((circle['x'] - existing['x'])**2 + (circle['y'] - existing['y'])**2)
                
                # 如果距离小于半径之和的一半，认为是重复
                min_dist = (circle['radius'] + existing['radius']) / 2
                if dist < min_dist:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_circles.append(circle)
        
        return final_circles
    
    def visualize_circles(self, image: np.ndarray, circles: List[Dict]) -> np.ndarray:
        """
        可视化检测到的圆形
        
        Args:
            image: 原始图像
            circles: 检测到的圆形
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        for circle in circles:
            x, y, r = circle['x'], circle['y'], circle['radius']
            confidence = circle['confidence']
            
            # 绘制圆形
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            
            # 绘制圆心
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)
            
            # 添加置信度标签
            label = f"{confidence:.2f}"
            cv2.putText(result_image, label, 
                       (x - 20, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return result_image 