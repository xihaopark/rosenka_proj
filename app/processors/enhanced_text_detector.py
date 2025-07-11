#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_text_detector.py
增强版文字检测 - 处理竖排文字和特殊符号
"""

import cv2
import numpy as np
from .modern_text_detector import ModernTextDetector
from typing import List, Dict, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)

class EnhancedTextDetector(ModernTextDetector):
    """增强版检测器 - 专门处理路線価図特征"""
    
    def __init__(self, use_trocr: bool = True):
        super().__init__(use_trocr)
        
    def detect_with_rotation(self, image: np.ndarray) -> List[Dict]:
        """多角度检测以捕获竖排文字"""
        all_detections = []
        
        # 原始图像检测
        detections_0 = self.detect_text_regions(image)
        all_detections.extend(self._transform_detections(detections_0, 0))
        
        # 旋转90度检测（捕获竖排文字）
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        detections_90 = self.detect_text_regions(rotated_90)
        all_detections.extend(self._transform_detections(detections_90, 90, image.shape))
        
        # 旋转270度检测（另一种竖排方向）
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        detections_270 = self.detect_text_regions(rotated_270)
        all_detections.extend(self._transform_detections(detections_270, 270, image.shape))
        
        # TODO: 合并和去重
        return all_detections
    
    def detect_circled_numbers(self, image: np.ndarray) -> List[Dict]:
        """专门检测带圆圈的数字"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c[0], c[1], c[2]
                box = {
                    "x1": max(0, x - r), "y1": max(0, y - r),
                    "x2": min(image.shape[1], x + r), "y2": min(image.shape[0], y + r)
                }
                detections.append({
                    'bbox': (box['x1'], box['y1'], box['x2'], box['y2']),
                    'type': 'circled_number',
                    'center': (int(x), int(y)),
                    'radius': int(r)
                })
        return detections
    
    def detect_small_text_enhanced(self, image: np.ndarray) -> List[Dict]:
        """增强小文字检测"""
        enhanced_image, scale = self._enhance_image_for_small_text(image)
        
        detections = self.detect_text_regions(enhanced_image)

        for det in detections:
            # Scale coordinates back to original image size
            if 'polygon' in det:
                det['polygon'] = (np.array(det['polygon']) / scale).astype(int).tolist()
                x_coords = [p[0] for p in det['polygon']]
                y_coords = [p[1] for p in det['polygon']]
                det['x1'], det['y1'] = int(min(x_coords)), int(min(y_coords))
                det['x2'], det['y2'] = int(max(x_coords)), int(max(y_coords))
        return detections

    def _enhance_image_for_small_text(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """图像增强以更好地检测小文字"""
        scale = 2.0
        upscaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), scale
    
    def _transform_detections(self, detections: List[Dict], angle: int,
                              original_shape: Optional[Tuple[int, ...]] = None) -> List[Dict]:
        """添加旋转信息并转换坐标回原图"""
        if angle == 0:
            for det in detections:
                det['rotation'] = 0
            return detections

        if original_shape is None:
            logger.warning("Rotation angle is non-zero but original_shape is None. Cannot transform coordinates.")
            return detections
        
        h, w = original_shape[:2]
        transformed_detections = []
        for det in detections:
            det['rotation'] = angle
            points = np.array(det['polygon'])
            if angle == 90:
                new_points = np.array([[h - p[1], p[0]] for p in points])
            elif angle == 270:
                new_points = np.array([[p[1], w - p[0]] for p in points])
            else:
                new_points = points
            
            det['polygon'] = new_points.tolist()
            x_coords = [p[0] for p in new_points]
            y_coords = [p[1] for p in new_points]
            det['x1'], det['y1'] = int(min(x_coords)), int(min(y_coords))
            det['x2'], det['y2'] = int(max(x_coords)), int(max(y_coords))
            transformed_detections.append(det)
        return transformed_detections
    
    def detect_all_enhanced(self, image: np.ndarray) -> Dict[str, List]:
        """完整的增强检测流程"""
        results = {
            'horizontal_text': [], 'vertical_text': [],
            'circled_numbers': [], 'small_text': [], 'route_prices': []
        }
        
        # Detect all region types first
        rotated_regions = self.detect_with_rotation(image)
        circled_regions = self.detect_circled_numbers(image)
        small_text_regions = self.detect_small_text_enhanced(image)

        # Process circled numbers
        for det in circled_regions:
            box_for_rec = {'x1':det['bbox'][0], 'y1':det['bbox'][1], 'x2':det['bbox'][2], 'y2':det['bbox'][3]}
            text, conf = self.recognize_text(image, box_for_rec)
            if text:
                det.update({'text': text, 'confidence': conf})
                results['circled_numbers'].append(det)

        # Process rotated text
        for det in rotated_regions:
            text, conf = self.recognize_text(image, det)
            if text:
                det.update({'text': text, 'confidence': conf})
                if det.get('rotation', 0) == 0:
                    results['horizontal_text'].append(det)
                else:
                    results['vertical_text'].append(det)
                if self._is_route_price(text):
                    results['route_prices'].append(det)

        # Process small text
        for det in small_text_regions:
            text, conf = self.recognize_text(image, det)
            if text and len(text) <= 5:
                det.update({'text': text, 'confidence': conf})
                results['small_text'].append(det)
        
        return results
    
    def _is_route_price(self, text: str) -> bool:
        """判断是否为路线价"""
        return bool(re.match(r'^\d{2,3}[A-G]$', text.strip())) 