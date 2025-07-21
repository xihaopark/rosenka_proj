"""
图像增强模块
提供图像预处理和增强功能，包括去噪、锐化、对比度调整等
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """图像增强器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化图像增强器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.denoise_strength = self.config.get('denoise_strength', 10)
        self.sharpen_strength = self.config.get('sharpen_strength', 1.5)
        self.contrast_factor = self.config.get('contrast_factor', 1.2)
        self.brightness_offset = self.config.get('brightness_offset', 10)
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        对图像进行综合增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        try:
            # 1. 去噪
            denoised = self._denoise(image)
            
            # 2. 锐化
            sharpened = self._sharpen(denoised)
            
            # 3. 对比度调整
            enhanced = self._adjust_contrast(sharpened)
            
            # 4. 亮度调整
            final = self._adjust_brightness(enhanced)
            
            return final
            
        except Exception as e:
            logger.error(f"图像增强失败: {e}")
            return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        try:
            # 使用非局部均值去噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image, 
                None, 
                self.denoise_strength, 
                self.denoise_strength, 
                7, 
                21
            )
            return denoised
        except Exception as e:
            logger.warning(f"去噪失败，使用原图: {e}")
            return image
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        图像锐化
        
        Args:
            image: 输入图像
            
        Returns:
            锐化后的图像
        """
        try:
            # 创建锐化核
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 混合原图和锐化结果
            result = cv2.addWeighted(image, 1-self.sharpen_strength, 
                                   sharpened, self.sharpen_strength, 0)
            return result
        except Exception as e:
            logger.warning(f"锐化失败，使用原图: {e}")
            return image
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像对比度
        
        Args:
            image: 输入图像
            
        Returns:
            调整后的图像
        """
        try:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 对L通道进行CLAHE处理
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 合并通道
            enhanced_lab = cv2.merge([l, a, b])
            
            # 转换回BGR
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"对比度调整失败，使用原图: {e}")
            return image
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像亮度
        
        Args:
            image: 输入图像
            
        Returns:
            调整后的图像
        """
        try:
            # 调整亮度
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 调整V通道
            v = cv2.add(v, self.brightness_offset)
            v = np.clip(v, 0, 255)
            
            # 合并通道
            enhanced_hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"亮度调整失败，使用原图: {e}")
            return image
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        为OCR优化的图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 1. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 2. 去噪
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 3. 自适应阈值处理
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 4. 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"OCR预处理失败: {e}")
            return image
    
    def enhance_text_regions(self, image: np.ndarray, 
                           text_regions: list) -> np.ndarray:
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
                enhanced_roi = self._enhance_text_roi(text_roi)
                
                # 将处理后的区域放回原图
                result[y:y+h, x:x+w] = enhanced_roi
            
            return result
            
        except Exception as e:
            logger.error(f"文本区域增强失败: {e}")
            return image
    
    def _enhance_text_roi(self, roi: np.ndarray) -> np.ndarray:
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
    
    def resize_image(self, image: np.ndarray, 
                    target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            调整后的图像
        """
        try:
            resized = cv2.resize(image, target_size, 
                               interpolation=cv2.INTER_CUBIC)
            return resized
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
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
    
    def enhance_map_image(self, image: np.ndarray) -> np.ndarray:
        """
        专门为地图图像进行增强处理
        
        Args:
            image: 输入地图图像
            
        Returns:
            增强后的地图图像
        """
        try:
            # 1. 基础图像增强
            enhanced = self.enhance_image(image)
            
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