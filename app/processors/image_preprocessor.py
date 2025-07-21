#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_preprocessor.py
图像预处理器 - 提高低分辨率图像的检测效果
"""

import cv2
import numpy as np

class ImagePreprocessor:
    """图像预处理器"""
    
    @staticmethod
    def enhance_low_resolution(image: np.ndarray) -> np.ndarray:
        """增强低分辨率图像"""
        # 1. 超分辨率（简单版）
        height, width = image.shape[:2]
        if width < 1000 or height < 1000:
            # 使用INTER_CUBIC上采样
            scale = max(2000 / width, 2000 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        
        # 2. 去噪
        denoised_image = image
        if len(image.shape) == 3:
            # fastNlMeansDenoisingColored is very slow, use a faster alternative for now
            # denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            pass
        else:
            # denoised_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            pass
        
        # 3. 增强对比度
        lab = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. 锐化
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    @staticmethod
    def prepare_for_vertical_text(image: np.ndarray) -> np.ndarray:
        """专门为竖排文字优化"""
        # 应用特殊的形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return processed 