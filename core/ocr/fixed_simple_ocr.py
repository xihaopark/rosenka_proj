#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixed_simple_ocr.py
修复后的简单OCR引擎 - 兼容新版本PaddleOCR API
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class FixedSimpleOCR:
    """
    修复后的简单OCR引擎
    兼容新版本PaddleOCR API
    """
    
    def __init__(self, lang: str = 'japan'):
        """
        初始化OCR引擎
        
        Args:
            lang: 语言代码
        """
        self.lang = lang
        
        try:
            # 使用最基本的参数初始化
            self.ocr = PaddleOCR(lang=lang)
            logger.info(f"FixedSimpleOCR initialized successfully (Lang: {lang})")
        except Exception as e:
            logger.error(f"Failed to initialize FixedSimpleOCR: {e}")
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
            
            if not results:
                return []
            
            # 处理新版本API格式
            return self._parse_ocr_results(results)
            
        except Exception as e:
            logger.error(f"Error in FixedSimpleOCR text detection: {e}")
            return []
    
    def _parse_ocr_results(self, results) -> List[Dict]:
        """
        解析OCR结果，兼容新旧API格式
        
        Args:
            results: OCR原始结果
            
        Returns:
            标准化的文本区域列表
        """
        detected_regions = []
        
        try:
            # 新版本API格式（字典格式）
            if isinstance(results[0], dict) and 'rec_texts' in results[0]:
                logger.info("Using new API format (dict)")
                
                texts = results[0]['rec_texts']
                scores = results[0]['rec_scores']
                polys = results[0]['rec_polys']
                
                for text, score, poly in zip(texts, scores, polys):
                    # 过滤置信度太低和空文本
                    if score >= 0.3 and text.strip():
                        # 转换坐标为整数
                        coords_int = [[int(x), int(y)] for x, y in poly]
                        
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
                            'confidence': score,
                            'coordinates': coords_int,
                            'engine': 'fixed_simple_ocr'
                        }
                        
                        detected_regions.append(region)
            
            # 旧版本API格式（列表格式）
            elif isinstance(results[0], list):
                logger.info("Using old API format (list)")
                
                for line in results[0]:
                    if len(line) >= 2:
                        coords = line[0]
                        text_info = line[1]
                        
                        if text_info and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            # 过滤置信度太低和空文本
                            if confidence >= 0.3 and text.strip():
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
                                    'engine': 'fixed_simple_ocr'
                                }
                                
                                detected_regions.append(region)
            
            logger.info(f"FixedSimpleOCR detected {len(detected_regions)} valid text regions")
            return detected_regions
            
        except Exception as e:
            logger.error(f"Error parsing OCR results: {e}")
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
            'name': 'FixedSimpleOCR',
            'version': 'Compatible',
            'language': self.lang
        }
