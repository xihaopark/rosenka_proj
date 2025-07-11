#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_text_detector.py
增强版文字检测 - 处理竖排文字和特殊符号
"""

import cv2
import numpy as np
from modern_text_detector import ModernTextDetector
from typing import List, Dict, Tuple
import logging

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
        all_detections.extend(self._add_rotation_info(detections_0, 0))
        
        # 旋转90度检测（捕获竖排文字）
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        detections_90 = self.detect_text_regions(rotated_90)
        all_detections.extend(self._add_rotation_info(detections_90, 90, image.shape))
        
        # 旋转270度检测（另一种竖排方向）
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        detections_270 = self.detect_text_regions(rotated_270)
        all_detections.extend(self._add_rotation_info(detections_270, 270, image.shape))
        
        # 合并和去重
        return self._merge_rotated_detections(all_detections)
    
    def detect_circled_numbers(self, image: np.ndarray) -> List[Dict]:
        """专门检测带圆圈的数字"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=40
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                x, y, r = circle
                # 提取圆形区域
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(image.shape[1], x + r)
                y2 = min(image.shape[0], y + r)
                
                roi = image[y1:y2, x1:x2]
                
                # 识别圆内内容
                text, conf = self.recognize_text(image, {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                if text:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'confidence': conf,
                        'type': 'circled_number',
                        'center': (int(x), int(y)),
                        'radius': int(r)
                    })
        
        return detections
    
    def detect_small_text_enhanced(self, image: np.ndarray) -> List[Dict]:
        """增强小文字检测"""
        # 预处理增强
        enhanced = self._enhance_image_for_small_text(image)
        
        # 使用更细致的参数检测
        original_threshold = self.craft.text_threshold
        self.craft.text_threshold = 0.3  # 更低的阈值
        
        detections = self.detect_text_regions(enhanced)
        
        # 恢复原参数
        self.craft.text_threshold = original_threshold
        
        return detections
    
    def _enhance_image_for_small_text(self, image: np.ndarray) -> np.ndarray:
        """图像增强以更好地检测小文字"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 上采样（提高分辨率）
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(upscaled)
        
        # 3. 锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 转回彩色（CRAFT需要）
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def _add_rotation_info(self, detections: List[Dict], angle: int, 
                          original_shape: Tuple = None) -> List[Dict]:
        """添加旋转信息并转换坐标"""
        for det in detections:
            det['rotation'] = angle
            
            # 如果是旋转的检测，需要转换坐标回原图
            if angle == 90 and original_shape:
                h, w = original_shape[:2]
                # 转换坐标
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                det['x1'] = h - y2
                det['y1'] = x1
                det['x2'] = h - y1
                det['y2'] = x2
                
            elif angle == 270 and original_shape:
                h, w = original_shape[:2]
                # 转换坐标
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                det['x1'] = y1
                det['y1'] = w - x2
                det['x2'] = y2
                det['y2'] = w - x1
        
        return detections
    
    def _merge_rotated_detections(self, detections: List[Dict]) -> List[Dict]:
        """合并不同角度的检测结果"""
        # 简单的NMS去重
        # TODO: 实现更智能的合并策略
        return detections
    
    def detect_all_enhanced(self, image: np.ndarray) -> Dict[str, List]:
        """完整的增强检测流程"""
        results = {
            'horizontal_text': [],      # 横向文字
            'vertical_text': [],        # 竖向文字
            'circled_numbers': [],      # 带圈数字
            'small_text': [],           # 小文字
            'route_prices': []          # 路线价
        }
        
        # 1. 多角度检测
        all_rotated = self.detect_with_rotation(image)
        
        # 2. 带圈数字检测
        circled = self.detect_circled_numbers(image)
        results['circled_numbers'] = circled
        
        # 3. 小文字增强检测
        small_text = self.detect_small_text_enhanced(image)
        
        # 4. 分类结果
        for det in all_rotated:
            text, conf = self.recognize_text(image, det)
            if text:
                det['text'] = text
                det['confidence'] = conf
                
                # 根据旋转角度分类
                if det.get('rotation', 0) == 0:
                    results['horizontal_text'].append(det)
                else:
                    results['vertical_text'].append(det)
                
                # 检查是否为路线价
                if self._is_route_price(text):
                    results['route_prices'].append(det)
        
        # 5. 处理小文字检测结果
        for det in small_text:
            text, conf = self.recognize_text(image, det)
            if text and len(text) <= 5:  # 假设小文字不超过5个字符
                det['text'] = text
                det['confidence'] = conf
                results['small_text'].append(det)
        
        return results
    
    def _is_route_price(self, text: str) -> bool:
        """判断是否为路线价"""
        import re
        # 路线价模式：数字+字母，如 100D, 150E
        return bool(re.match(r'^\d+[A-Z]$', text))