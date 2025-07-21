#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_utils.py
图像处理工具 - 路線価図検索システム
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    为OCR优化的图像增强
    
    Args:
        image: 输入图像
        
    Returns:
        增强后的图像
    """
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 锐化
        kernel = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 二值化
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        logger.error(f"图像增强失败: {e}")
        return image

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        
    Returns:
        调整后的图像
    """
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        logger.error(f"图像尺寸调整失败: {e}")
        return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    图像归一化
    
    Args:
        image: 输入图像
        
    Returns:
        归一化后的图像
    """
    try:
        # 转换为float32
        normalized = image.astype(np.float32) / 255.0
        return normalized
    except Exception as e:
        logger.error(f"图像归一化失败: {e}")
        return image

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    为OCR优化的图像预处理
    
    Args:
        image: 输入图像
        
    Returns:
        预处理后的图像
    """
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return processed
        
    except Exception as e:
        logger.error(f"OCR预处理失败: {e}")
        return image

def enhance_text_regions(image: np.ndarray, text_regions: list) -> np.ndarray:
    """
    增强文本区域
    
    Args:
        image: 输入图像
        text_regions: 文本区域列表，每个元素为 [x, y, w, h]
        
    Returns:
        增强后的图像
    """
    try:
        result = image.copy()
        
        for region in text_regions:
            x, y, w, h = region
            
            # 提取文本区域
            text_roi = image[y:y+h, x:x+w]
            
            # 对文本区域进行特殊处理
            enhanced_roi = _enhance_text_roi(text_roi)
            
            # 将处理后的区域放回原图
            result[y:y+h, x:x+w] = enhanced_roi
        
        return result
        
    except Exception as e:
        logger.error(f"文本区域增强失败: {e}")
        return image

def _enhance_text_roi(roi: np.ndarray) -> np.ndarray:
    """
    增强单个文本区域
    
    Args:
        roi: 文本区域图像
        
    Returns:
        增强后的文本区域
    """
    try:
        # 转换为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 锐化
        kernel = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        logger.warning(f"文本区域增强失败: {e}")
        return roi

def enhance_map_image(image: np.ndarray) -> np.ndarray:
    """
    专门为地图图像进行增强处理
    
    Args:
        image: 输入地图图像
        
    Returns:
        增强后的地图图像
    """
    try:
        # 1. 基础图像增强
        enhanced = enhance_image_for_ocr(image)
        
        # 2. 地图特定的处理
        # 增强对比度以突出文字和线条
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道进行更强的CLAHE处理
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. 锐化处理以增强文字边缘
        kernel = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 混合原图和锐化结果
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
        
    except Exception as e:
        logger.error(f"地图图像增强失败: {e}")
        return image 