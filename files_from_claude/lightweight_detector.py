#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightweight_detector.py
轻量级文字检测 - 仅使用OpenCV和Tesseract
无需任何深度学习框架
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LightweightTextDetector:
    """轻量级文字检测器 - 零深度学习依赖"""
    
    def __init__(self):
        self.min_area = 100  # 最小文字区域面积
        self.max_area = 50000  # 最大文字区域面积
        
    def detect_text_regions_opencv(self, image: np.ndarray) -> List[Dict]:
        """使用OpenCV的MSER和形态学操作检测文字"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 方法1: MSER检测
        mser_regions = self._detect_with_mser(gray)
        
        # 方法2: 边缘检测 + 轮廓
        edge_regions = self._detect_with_edges(gray)
        
        # 方法3: 形态学操作
        morph_regions = self._detect_with_morphology(gray)
        
        # 合并所有检测结果
        all_regions = mser_regions + edge_regions + morph_regions
        
        # NMS去重
        return self._nms_regions(all_regions)
    
    def _detect_with_mser(self, gray: np.ndarray) -> List[Dict]:
        """MSER文字区域检测"""
        mser = cv2.MSER_create(
            _delta=5,
            _min_area=self.min_area,
            _max_area=self.max_area,
            _max_variation=0.5
        )
        
        regions, _ = mser.detectRegions(gray)
        
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if self._is_valid_text_region(w, h):
                boxes.append({
                    'x1': x, 'y1': y, 
                    'x2': x + w, 'y2': y + h,
                    'method': 'MSER'
                })
        
        return boxes
    
    def _detect_with_edges(self, gray: np.ndarray) -> List[Dict]:
        """边缘检测方法"""
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学闭运算连接文字
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_text_region(w, h):
                boxes.append({
                    'x1': x, 'y1': y,
                    'x2': x + w, 'y2': y + h,
                    'method': 'Edge'
                })
        
        return boxes
    
    def _detect_with_morphology(self, gray: np.ndarray) -> List[Dict]:
        """形态学操作检测"""
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 水平和垂直投影
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 合并
        combined = cv2.bitwise_or(horizontal, vertical)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_text_region(w, h):
                boxes.append({
                    'x1': x, 'y1': y,
                    'x2': x + w, 'y2': y + h,
                    'method': 'Morphology'
                })
        
        return boxes
    
    def _is_valid_text_region(self, width: int, height: int) -> bool:
        """判断是否为有效的文字区域"""
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # 过滤条件
        if area < self.min_area or area > self.max_area:
            return False
        
        # 文字的宽高比通常在合理范围内
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            return False
        
        return True
    
    def _nms_regions(self, regions: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """非极大值抑制"""
        if not regions:
            return []
        
        # 转换为numpy数组
        boxes = np.array([[r['x1'], r['y1'], r['x2'], r['y2']] for r in regions])
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按面积排序
        order = areas.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [regions[i] for i in keep]
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """完整的检测和识别流程"""
        # 1. 检测文字区域
        regions = self.detect_text_regions_opencv(image)
        
        # 2. 对每个区域进行OCR
        results = []
        for region in regions:
            x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            try:
                # Tesseract OCR
                text = pytesseract.image_to_string(
                    roi, 
                    lang='jpn+eng',
                    config='--psm 8'
                ).strip()
                
                if text:
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'method': f"OpenCV-{region['method']}+Tesseract"
                    })
            except Exception as e:
                logger.error(f"OCR失败: {e}")
                continue
        
        return results