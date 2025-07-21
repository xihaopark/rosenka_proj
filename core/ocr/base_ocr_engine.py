#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_ocr_engine.py
OCR引擎基类 - 定义OCR引擎的标准接口
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BaseOCREngine(ABC):
    """OCR引擎基类"""
    
    def __init__(self):
        """初始化OCR引擎"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的文本区域
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的文本区域列表
        """
        pass
    
    @abstractmethod
    def recognize_text(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        识别指定区域中的文本
        
        Args:
            image: 输入图像
            regions: 文本区域列表
            
        Returns:
            识别结果列表
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict:
        """
        获取引擎信息
        
        Returns:
            引擎信息字典
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理（可选实现）
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        return image
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        验证输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            图像是否有效
        """
        if image is None:
            return False
        
        if len(image.shape) < 2:
            return False
        
        if image.size == 0:
            return False
        
        return True 