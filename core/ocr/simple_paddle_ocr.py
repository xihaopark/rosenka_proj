#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_paddle_ocr.py
简化的PaddleOCR引擎 - 兼容新版本API
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class SimplePaddleOCR:
    """
    简化的PaddleOCR引擎
    使用最基本的参数确保兼容性
    """
    
    def __init__(self, lang: str = 'japan'):
        """
        初始化简化PaddleOCR引擎
        
        Args:
            lang: 语言代码
        """
        self.lang = lang
        
        try:
            # 使用最基本的参数初始化
            self.ocr = PaddleOCR(lang=lang)
            logger.info(f"SimplePaddleOCR initialized successfully (Lang: {lang})")
        except Exception as e:
            logger.error(f"Failed to initialize SimplePaddleOCR: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        检测文本区域
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的文本区域列表
        """
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)
            
            # 运行OCR
            results = self.ocr.ocr(processed_image)
            
            if not results or not results[0]:
                return []
            
            detected_regions = []
            
            for line in results[0]:
                if len(line) >= 2:
                    # 提取坐标和文本
                    coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence)
                    
                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # 过滤置信度太低的结果
                        if confidence >= 0.3:
                            # 转换坐标为整数
                            coords_int = [[int(x), int(y)] for x, y in coords]
                            
                            # 计算边界框
                            x_coords = [coord[0] for coord in coords_int]
                            y_coords = [coord[1] for coord in coords_int]
                            
                            bbox = {
                                'x': min(x_coords),
                                'y': min(y_coords),
                                'width': max(x_coords) - min(x_coords),
                                'height': max(y_coords) - min(y_coords)
                            }
                            
                            region = {
                                'bbox': bbox,
                                'text': text,
                                'confidence': confidence,
                                'coordinates': coords_int,
                                'engine': 'simple_paddleocr'
                            }
                            
                            detected_regions.append(region)
            
            logger.info(f"SimplePaddleOCR detected {len(detected_regions)} text regions")
            return detected_regions
            
        except Exception as e:
            logger.error(f"Error in SimplePaddleOCR text detection: {e}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 转换为RGB（PaddleOCR期望RGB格式）
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def get_engine_info(self) -> Dict:
        """获取引擎信息"""
        return {
            'name': 'SimplePaddleOCR',
            'version': 'Compatible',
            'language': self.lang
        }