#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Image Preprocessor for Stage 5 Rosenka OCR System
专门针对路線価図的高级图像预处理器

主要功能:
1. 多版本图像生成 - 生成6-8个优化版本
2. 页眉页脚自动检测和移除
3. 自适应二值化处理  
4. 反色图像生成（黑底白字 → 白底黑字）
5. 形态学线条去除（保留文字）
6. 多尺度图像缩放
7. 噪声降低和对比度增强
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedImagePreprocessor:
    """Stage 5 增强图像预处理器"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化预处理器
        
        Args:
            debug_mode: 是否启用调试模式（保存中间结果）
        """
        self.debug_mode = debug_mode
        self.debug_dir = Path("debug_preprocessing") if debug_mode else None
        
        # 预处理参数
        self.header_height_ratio = 0.15  # 页眉高度比例 
        self.footer_height_ratio = 0.05  # 页脚高度比例
        
        # 线条检测参数
        self.min_line_length = 30
        self.max_line_gap = 10
        
        # 缩放比例设置
        self.scales = [1.0, 1.5, 2.0]
        
        if self.debug_mode and self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"调试模式启用，中间结果保存至: {self.debug_dir}")
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                          page_name: str = "page") -> List[Tuple[str, np.ndarray]]:
        """
        生成多个预处理版本以提高OCR检测率
        
        Args:
            image: 输入图像
            page_name: 页面名称（用于调试）
            
        Returns:
            List of (variant_name, processed_image) tuples
        """
        logger.info(f"开始预处理 {page_name}, 原始尺寸: {image.shape}")
        
        results = []
        
        try:
            # 1. 原始图像
            results.append(("original", image.copy()))
            
            # 2. 去除页眉页脚
            no_header_footer = self.remove_header_footer(image)
            results.append(("no_header_footer", no_header_footer))
            
            # 3. 自适应二值化
            binary = self.adaptive_binarize(no_header_footer)
            results.append(("binary", binary))
            
            # 4. 反色图像（检测黑底白字）
            inverted = cv2.bitwise_not(binary)
            results.append(("inverted", inverted))
            
            # 5. 形态学处理去除细线
            no_lines = self.remove_thin_lines(binary)
            results.append(("no_lines", no_lines))
            
            # 6. 反色的去线版本
            inverted_no_lines = cv2.bitwise_not(no_lines)
            results.append(("inverted_no_lines", inverted_no_lines))
            
            # 7. 对比度增强版本
            enhanced = self.enhance_contrast(no_header_footer)
            results.append(("enhanced", enhanced))
            
            # 8. 噪声去除版本
            denoised = self.remove_noise(binary)
            results.append(("denoised", denoised))
            
            # 保存调试图像
            if self.debug_mode:
                self._save_debug_images(results, page_name)
            
            logger.info(f"预处理完成，生成 {len(results)} 个版本")
            return results
            
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            # 返回原始图像作为fallback
            return [("original", image)]
    
    def remove_header_footer(self, image: np.ndarray) -> np.ndarray:
        """
        自动检测并去除页眉页脚区域
        
        Args:
            image: 输入图像
            
        Returns:
            去除页眉页脚的图像
        """
        h, w = image.shape[:2]
        
        # 计算裁剪区域
        header_pixels = int(h * self.header_height_ratio)
        footer_pixels = int(h * self.footer_height_ratio)
        
        # 智能检测页眉边界（查找空白行）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 检测页眉实际边界
        actual_header = self._detect_content_boundary(gray[:header_pixels*2], 'top')
        actual_footer = h - self._detect_content_boundary(gray[h-footer_pixels*2:], 'bottom')
        
        # 使用检测到的边界，但有最小保护区域
        start_y = max(header_pixels//2, actual_header)
        end_y = min(h - footer_pixels//2, actual_footer)
        
        cropped = image[start_y:end_y, :]
        
        logger.debug(f"页眉页脚移除: {h}x{w} -> {cropped.shape[0]}x{cropped.shape[1]}")
        return cropped
    
    def _detect_content_boundary(self, region: np.ndarray, direction: str) -> int:
        """
        检测内容边界（查找空白区域结束的位置）
        
        Args:
            region: 要检测的区域
            direction: 'top' 或 'bottom'
            
        Returns:
            边界位置
        """
        # 计算每行的非零像素数量
        if direction == 'top':
            rows = range(region.shape[0])
        else:
            rows = range(region.shape[0]-1, -1, -1)
        
        for i, row_idx in enumerate(rows):
            row = region[row_idx, :]
            non_zero_count = np.count_nonzero(row < 240)  # 非白色像素
            
            # 如果找到有内容的行
            if non_zero_count > region.shape[1] * 0.05:  # 至少5%的像素有内容
                if direction == 'top':
                    return max(0, row_idx - 5)  # 留一些边距
                else:
                    return region.shape[0] - max(0, row_idx - 5)
        
        # 如果没找到，返回默认值
        return region.shape[0] // 2 if direction == 'top' else region.shape[0] // 2
    
    def adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        自适应二值化处理
        
        Args:
            image: 输入图像
            
        Returns:
            二值化图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用自适应阈值 - 对地图图像效果更好
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        
        return binary
    
    def remove_thin_lines(self, image: np.ndarray) -> np.ndarray:
        """
        去除细线（道路、区划线）但保留文字
        
        Args:
            image: 二值化图像
            
        Returns:
            去除线条的图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测水平线条
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, h_kernel)
        
        # 检测垂直线条  
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        v_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, v_kernel)
        
        # 检测斜线（45度）
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 合并所有线条
        all_lines = cv2.add(h_lines, v_lines)
        
        # 从原图减去线条
        result = cv2.subtract(image, all_lines)
        
        # 形态学操作恢复可能被误删的文字
        restore_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, restore_kernel)
        
        return result
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强对比度
        
        Args:
            image: 输入图像
            
        Returns:
            对比度增强的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE (限制对比度自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        去除噪声
        
        Args:
            image: 输入图像
            
        Returns:
            去噪的图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 中值滤波去除椒盐噪声
        denoised = cv2.medianBlur(image, 3)
        
        # 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        return denoised
    
    def create_scaled_versions(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        创建多尺度版本
        
        Args:
            image: 输入图像
            
        Returns:
            多尺度图像列表
        """
        scaled_versions = []
        
        for scale in self.scales:
            if scale == 1.0:
                scaled_versions.append((f"scale_{scale}", image.copy()))
            else:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                scaled_versions.append((f"scale_{scale}", scaled))
        
        return scaled_versions
    
    def _save_debug_images(self, results: List[Tuple[str, np.ndarray]], page_name: str):
        """
        保存调试图像
        
        Args:
            results: 处理结果列表
            page_name: 页面名称
        """
        if not self.debug_dir:
            return
        
        for variant_name, processed_image in results:
            filename = f"{page_name}_{variant_name}.jpg"
            filepath = self.debug_dir / filename
            
            try:
                cv2.imwrite(str(filepath), processed_image)
                logger.debug(f"保存调试图像: {filepath}")
            except Exception as e:
                logger.warning(f"保存调试图像失败 {filepath}: {e}")

# 使用示例
if __name__ == "__main__":
    # 测试预处理器
    preprocessor = EnhancedImagePreprocessor(debug_mode=True)
    
    # 假设有测试图像
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        if image is not None:
            results = preprocessor.preprocess_for_ocr(image, "test")
            print(f"生成了 {len(results)} 个预处理版本")
            for name, img in results:
                print(f"- {name}: {img.shape}")
        else:
            print("无法读取测试图像")
    else:
        print(f"测试图像不存在: {test_image_path}")