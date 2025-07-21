"""
OCR回退方案
当主要OCR引擎不可用时使用的简化OCR
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)

class FallbackOCR:
    """OCR回退方案"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 价格模式匹配
        self.price_patterns = [
            r'^\d+[A-Z]?$',  # 115E, 120, 95A等
            r'^\d+万$',      # 120万等
            r'^\d+千$',      # 95千等
            r'^\d+\.\d+$',   # 115.5等
        ]
        
        # 数字模板匹配
        self.digit_templates = self._create_digit_templates()
    
    def _create_digit_templates(self) -> Dict[str, np.ndarray]:
        """创建数字模板"""
        templates = {}
        
        # 创建0-9的数字模板
        for i in range(10):
            # 创建简单的数字图像
            img = np.zeros((20, 12), dtype=np.uint8)
            cv2.putText(img, str(i), (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            templates[str(i)] = img
        
        # 添加字母模板
        for letter in ['A', 'B', 'C', 'D', 'E']:
            img = np.zeros((20, 12), dtype=np.uint8)
            cv2.putText(img, letter, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            templates[letter] = img
        
        return templates
    
    def recognize_text(self, image: np.ndarray) -> Optional[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果
        """
        try:
            # 图像预处理
            processed = self._preprocess_image(image)
            
            # 尝试多种识别方法
            results = []
            
            # 方法1: 模板匹配
            template_result = self._template_matching(processed)
            if template_result:
                results.append(template_result)
            
            # 方法2: 轮廓分析
            contour_result = self._contour_analysis(processed)
            if contour_result:
                results.append(contour_result)
            
            # 方法3: 连通组件分析
            cc_result = self._connected_component_analysis(processed)
            if cc_result:
                results.append(cc_result)
            
            # 选择最佳结果
            if results:
                best_result = max(results, key=lambda x: x.get('confidence', 0))
                return best_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"文字识别失败: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 反色处理
        inverted = cv2.bitwise_not(gray)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(inverted)
        
        # 二值化
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _template_matching(self, image: np.ndarray) -> Optional[Dict]:
        """模板匹配方法"""
        try:
            best_match = None
            best_confidence = 0
            
            for char, template in self.digit_templates.items():
                # 模板匹配
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = char
            
            if best_confidence > 0.6:  # 阈值
                return {
                    'text': best_match,
                    'confidence': best_confidence,
                    'method': 'template_matching'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"模板匹配失败: {e}")
            return None
    
    def _contour_analysis(self, image: np.ndarray) -> Optional[Dict]:
        """轮廓分析方法"""
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 分析轮廓特征
            text_candidates = []
            
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 过滤小轮廓
                if area < 50:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 分析宽高比
                aspect_ratio = w / h if h > 0 else 0
                
                # 数字通常有特定的宽高比
                if 0.3 < aspect_ratio < 2.0:
                    # 提取ROI
                    roi = image[y:y+h, x:x+w]
                    
                    # 简单的数字识别
                    digit = self._recognize_digit_from_roi(roi)
                    if digit:
                        text_candidates.append(digit)
            
            if text_candidates:
                text = ''.join(text_candidates)
                return {
                    'text': text,
                    'confidence': 0.7,
                    'method': 'contour_analysis'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"轮廓分析失败: {e}")
            return None
    
    def _connected_component_analysis(self, image: np.ndarray) -> Optional[Dict]:
        """连通组件分析"""
        try:
            # 连通组件分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
            
            text_candidates = []
            
            for i in range(1, num_labels):  # 跳过背景
                # 获取组件统计信息
                x, y, w, h, area = stats[i]
                
                # 过滤小组件
                if area < 30 or w < 5 or h < 5:
                    continue
                
                # 提取组件ROI
                component = (labels == i).astype(np.uint8) * 255
                roi = component[y:y+h, x:x+w]
                
                # 识别数字
                digit = self._recognize_digit_from_roi(roi)
                if digit:
                    text_candidates.append(digit)
            
            if text_candidates:
                text = ''.join(text_candidates)
                return {
                    'text': text,
                    'confidence': 0.6,
                    'method': 'connected_component'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"连通组件分析失败: {e}")
            return None
    
    def _recognize_digit_from_roi(self, roi: np.ndarray) -> Optional[str]:
        """从ROI识别数字"""
        try:
            # 简单的数字识别逻辑
            # 基于像素密度和形状特征
            
            # 计算像素密度
            total_pixels = roi.shape[0] * roi.shape[1]
            white_pixels = np.sum(roi > 0)
            density = white_pixels / total_pixels if total_pixels > 0 else 0
            
            # 计算形状特征
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 获取最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓特征
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 基于特征进行简单分类
            if density > 0.3 and circularity < 0.8:
                # 可能是数字
                return self._classify_digit_by_features(roi, density, circularity)
            
            return None
            
        except Exception as e:
            self.logger.error(f"ROI数字识别失败: {e}")
            return None
    
    def _classify_digit_by_features(self, roi: np.ndarray, density: float, circularity: float) -> Optional[str]:
        """基于特征分类数字"""
        # 这是一个简化的分类逻辑
        # 在实际应用中，可以使用更复杂的特征提取和分类算法
        
        # 基于密度和圆形度的简单分类
        if density > 0.5:
            if circularity > 0.6:
                return "8"  # 圆形数字
            else:
                return "1"  # 直线数字
        elif density > 0.3:
            return "2"  # 中等密度
        else:
            return "0"  # 低密度
    
    def validate_price_text(self, text: str) -> Optional[str]:
        """验证价格文本"""
        if not text:
            return None
        
        # 清理文本
        cleaned_text = re.sub(r'[^\w\d]', '', text)
        
        # 检查是否匹配价格模式
        for pattern in self.price_patterns:
            if re.match(pattern, cleaned_text):
                return cleaned_text
        
        return None 