#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_number_ocr.py
增强数字识别OCR引擎 - 专门针对路線価图数字识别优化
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger.warning("EasyOCR not available, will use PaddleOCR only")
    EASYOCR_AVAILABLE = False

class EnhancedNumberOCR:
    """
    增强数字识别OCR引擎
    专门针对路線価图中的价格数字进行优化
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        初始化增强数字识别引擎
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu
        
        # 初始化多个OCR引擎
        self._init_ocr_engines()
        
        # 数字模式
        self.number_patterns = [
            r'^\d+[A-Z]?$',      # 115E, 120, 95A等
            r'^\d+万$',          # 120万等  
            r'^\d+千$',          # 95千等
            r'^\d+\.\d+$',       # 115.5等
            r'^\d+,\d+$',        # 1,200等
            r'^\d+[\.\,]\d+万$', # 12.5万等
        ]
        
        logger.info("增强数字识别OCR引擎初始化完成")
    
    def _init_ocr_engines(self):
        """初始化多个OCR引擎"""
        try:
            # PaddleOCR - 使用新的兼容版本
            self.paddle_ocr = PaddleOCR(lang='japan')
            
            # EasyOCR - 作为备用引擎
            if EASYOCR_AVAILABLE:
                self.easy_ocr = easyocr.Reader(['ja', 'en'], gpu=self.use_gpu)
            else:
                self.easy_ocr = None
            
            logger.info("多OCR引擎初始化成功")
            
        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            raise
    
    def enhance_number_regions(self, image: np.ndarray) -> np.ndarray:
        """
        专门针对数字区域的图像增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        try:
            # 1. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 2. 去噪 - 使用双边滤波保护边缘
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 3. 对比度增强 - 更激进的CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(denoised)
            
            # 4. 锐化 - 增强数字边缘
            kernel = np.array([[-1, -1, -1, -1, -1],
                             [-1,  2,  2,  2, -1],
                             [-1,  2,  8,  2, -1],
                             [-1,  2,  2,  2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 5. 自适应二值化 - 适应不同光照条件
            binary = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 6. 形态学操作 - 连接断裂的数字
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 7. 最终清理 - 去除噪点
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            final = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return final
            
        except Exception as e:
            logger.error(f"数字区域增强失败: {e}")
            return image
    
    def detect_number_regions(self, image: np.ndarray) -> List[Dict]:
        """
        检测可能包含数字的区域
        
        Args:
            image: 输入图像
            
        Returns:
            数字区域列表
        """
        try:
            # 图像预处理
            enhanced = self.enhance_number_regions(image)
            
            # 使用多个OCR引擎进行检测
            regions = []
            
            # 1. PaddleOCR检测
            paddle_regions = self._detect_with_paddle(enhanced)
            regions.extend(paddle_regions)
            
            # 2. EasyOCR检测
            if self.easy_ocr:
                easy_regions = self._detect_with_easy(enhanced)
                regions.extend(easy_regions)
            
            # 3. 传统CV方法检测
            cv_regions = self._detect_with_cv(enhanced)
            regions.extend(cv_regions)
            
            # 合并和过滤结果
            merged_regions = self._merge_overlapping_regions(regions)
            validated_regions = self._validate_number_regions(merged_regions)
            
            logger.info(f"检测到 {len(validated_regions)} 个数字区域")
            return validated_regions
            
        except Exception as e:
            logger.error(f"数字区域检测失败: {e}")
            return []
    
    def _detect_with_paddle(self, image: np.ndarray) -> List[Dict]:
        """使用PaddleOCR检测"""
        try:
            # 转换为RGB
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.paddle_ocr.ocr(rgb_image)
            
            regions = []
            if results:
                # 处理新版本API格式（字典格式）
                if isinstance(results[0], dict) and 'rec_texts' in results[0]:
                    texts = results[0]['rec_texts']
                    scores = results[0]['rec_scores']
                    polys = results[0]['rec_polys']
                    
                    for text, score, poly in zip(texts, scores, polys):
                        # 过滤置信度太低和空文本
                        if score >= 0.3 and text.strip():
                            # 检查是否可能是数字
                            if self._is_potential_number(text):
                                # 转换坐标为整数
                                coords_int = [[int(x), int(y)] for x, y in poly]
                                
                                # 计算边界框
                                x_coords = [coord[0] for coord in coords_int]
                                y_coords = [coord[1] for coord in coords_int]
                                
                                bbox = {
                                    'x': int(min(x_coords)),
                                    'y': int(min(y_coords)),
                                    'width': int(max(x_coords) - min(x_coords)),
                                    'height': int(max(y_coords) - min(y_coords))
                                }
                                
                                regions.append({
                                    'bbox': bbox,
                                    'text': text,
                                    'confidence': score,
                                    'engine': 'paddleocr',
                                    'coordinates': coords_int
                                })
                
                # 处理旧版本API格式（列表格式）
                elif isinstance(results[0], list):
                    for line in results[0]:
                        if len(line) >= 2:
                            coords = line[0]
                            text_info = line[1]
                            
                            if text_info and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                                
                                # 检查是否可能是数字
                                if self._is_potential_number(text):
                                    # 计算边界框
                                    x_coords = [coord[0] for coord in coords]
                                    y_coords = [coord[1] for coord in coords]
                                    
                                    bbox = {
                                        'x': int(min(x_coords)),
                                        'y': int(min(y_coords)),
                                        'width': int(max(x_coords) - min(x_coords)),
                                        'height': int(max(y_coords) - min(y_coords))
                                    }
                                    
                                    regions.append({
                                        'bbox': bbox,
                                        'text': text,
                                        'confidence': confidence,
                                        'engine': 'paddleocr',
                                        'coordinates': coords
                                    })
            
            return regions
            
        except Exception as e:
            logger.error(f"PaddleOCR检测失败: {e}")
            return []
    
    def _detect_with_easy(self, image: np.ndarray) -> List[Dict]:
        """使用EasyOCR检测"""
        if not self.easy_ocr:
            return []
        
        try:
            results = self.easy_ocr.readtext(image)
            
            regions = []
            for result in results:
                coords, text, confidence = result
                
                # 检查是否可能是数字
                if self._is_potential_number(text):
                    # 计算边界框
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    
                    bbox = {
                        'x': int(min(x_coords)),
                        'y': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords))
                    }
                    
                    regions.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence,
                        'engine': 'easyocr',
                        'coordinates': coords
                    })
            
            return regions
            
        except Exception as e:
            logger.error(f"EasyOCR检测失败: {e}")
            return []
    
    def _detect_with_cv(self, image: np.ndarray) -> List[Dict]:
        """使用传统CV方法检测"""
        try:
            regions = []
            
            # 查找轮廓
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 过滤小轮廓
                if area < 100:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 检查宽高比
                aspect_ratio = w / h if h > 0 else 0
                
                # 数字通常有特定的宽高比
                if 0.2 < aspect_ratio < 3.0 and w > 10 and h > 10:
                    # 提取ROI进行简单验证
                    roi = image[y:y+h, x:x+w]
                    
                    # 计算白色像素比例
                    white_pixels = np.sum(roi > 0)
                    total_pixels = roi.shape[0] * roi.shape[1]
                    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # 如果白色像素比例合理，认为可能是数字
                    if 0.1 < white_ratio < 0.8:
                        regions.append({
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'text': '',  # CV方法不直接识别文字
                            'confidence': white_ratio,
                            'engine': 'cv_contour',
                            'coordinates': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        })
            
            return regions
            
        except Exception as e:
            logger.error(f"CV方法检测失败: {e}")
            return []
    
    def _is_potential_number(self, text: str) -> bool:
        """检查文本是否可能是数字"""
        if not text:
            return False
        
        # 清理文本
        cleaned_text = re.sub(r'[^\w\d万千,\.A-Z]', '', text)
        
        # 检查是否包含数字
        if not re.search(r'\d', cleaned_text):
            return False
        
        # 检查是否匹配数字模式
        for pattern in self.number_patterns:
            if re.match(pattern, cleaned_text):
                return True
        
        # 检查是否主要由数字组成
        digit_ratio = len(re.findall(r'\d', cleaned_text)) / len(cleaned_text)
        return digit_ratio > 0.5
    
    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """合并重叠的区域"""
        if not regions:
            return []
        
        # 按置信度排序
        sorted_regions = sorted(regions, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        
        for region in sorted_regions:
            bbox = region['bbox']
            is_duplicate = False
            
            for existing in merged:
                existing_bbox = existing['bbox']
                
                # 计算重叠面积
                overlap = self._calculate_overlap(bbox, existing_bbox)
                
                # 如果重叠超过50%，认为是重复
                if overlap > 0.5:
                    # 保留置信度更高的
                    if region['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(region)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(region)
        
        return merged
    
    def _calculate_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """计算两个边界框的重叠率"""
        try:
            x1_min, y1_min = bbox1['x'], bbox1['y']
            x1_max, y1_max = x1_min + bbox1['width'], y1_min + bbox1['height']
            
            x2_min, y2_min = bbox2['x'], bbox2['y']
            x2_max, y2_max = x2_min + bbox2['width'], y2_min + bbox2['height']
            
            # 计算交集
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            
            overlap_area = x_overlap * y_overlap
            
            # 计算并集
            area1 = bbox1['width'] * bbox1['height']
            area2 = bbox2['width'] * bbox2['height']
            union_area = area1 + area2 - overlap_area
            
            # 返回IoU
            return overlap_area / union_area if union_area > 0 else 0
            
        except Exception as e:
            logger.error(f"重叠计算失败: {e}")
            return 0
    
    def _validate_number_regions(self, regions: List[Dict]) -> List[Dict]:
        """验证数字区域"""
        validated = []
        
        for region in regions:
            # 检查置信度
            if region['confidence'] < 0.3:
                continue
            
            # 检查边界框尺寸
            bbox = region['bbox']
            if bbox['width'] < 8 or bbox['height'] < 8:
                continue
            
            # 检查宽高比
            aspect_ratio = bbox['width'] / bbox['height']
            if aspect_ratio > 5.0 or aspect_ratio < 0.1:
                continue
            
            # 如果有文本，验证文本内容
            if region['text'] and not self._is_potential_number(region['text']):
                continue
            
            validated.append(region)
        
        return validated
    
    def recognize_numbers_in_regions(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """在指定区域识别数字"""
        results = []
        
        for region in regions:
            try:
                bbox = region['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # 扩展边界框以获得更好的识别效果
                padding = 5
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(image.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(image.shape[0] - y_pad, h + 2 * padding)
                
                # 提取ROI
                roi = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # 对ROI进行增强
                enhanced_roi = self.enhance_number_regions(roi)
                
                # 使用多个引擎识别
                recognition_results = []
                
                # PaddleOCR识别
                paddle_result = self._recognize_with_paddle(enhanced_roi)
                if paddle_result:
                    recognition_results.append(paddle_result)
                
                # EasyOCR识别
                if self.easy_ocr:
                    easy_result = self._recognize_with_easy(enhanced_roi)
                    if easy_result:
                        recognition_results.append(easy_result)
                
                # 选择最佳结果
                best_result = self._select_best_recognition(recognition_results)
                
                if best_result:
                    result = region.copy()
                    result.update(best_result)
                    results.append(result)
                
            except Exception as e:
                logger.error(f"区域数字识别失败: {e}")
                continue
        
        return results
    
    def _recognize_with_paddle(self, roi: np.ndarray) -> Optional[Dict]:
        """使用PaddleOCR识别ROI"""
        try:
            if len(roi.shape) == 2:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            else:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            results = self.paddle_ocr.ocr(rgb_roi)
            
            if results:
                texts = []
                confidences = []
                
                # 处理新版本API格式（字典格式）
                if isinstance(results[0], dict) and 'rec_texts' in results[0]:
                    result_texts = results[0]['rec_texts']
                    result_scores = results[0]['rec_scores']
                    
                    for text, score in zip(result_texts, result_scores):
                        if score >= 0.3 and text.strip():
                            if self._is_potential_number(text):
                                texts.append(text)
                                confidences.append(score)
                
                # 处理旧版本API格式（列表格式）
                elif isinstance(results[0], list):
                    for line in results[0]:
                        if len(line) >= 2:
                            text_info = line[1]
                            if text_info and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                                
                                if self._is_potential_number(text):
                                    texts.append(text)
                                    confidences.append(confidence)
                
                if texts:
                    # 合并文本
                    combined_text = ''.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    return {
                        'text': combined_text,
                        'confidence': avg_confidence,
                        'engine': 'paddleocr'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"PaddleOCR ROI识别失败: {e}")
            return None
    
    def _recognize_with_easy(self, roi: np.ndarray) -> Optional[Dict]:
        """使用EasyOCR识别ROI"""
        if not self.easy_ocr:
            return None
        
        try:
            results = self.easy_ocr.readtext(roi)
            
            if results:
                # 合并所有识别结果
                texts = []
                confidences = []
                
                for result in results:
                    _, text, confidence = result
                    
                    if self._is_potential_number(text):
                        texts.append(text)
                        confidences.append(confidence)
                
                if texts:
                    # 合并文本
                    combined_text = ''.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    return {
                        'text': combined_text,
                        'confidence': avg_confidence,
                        'engine': 'easyocr'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"EasyOCR ROI识别失败: {e}")
            return None
    
    def _select_best_recognition(self, results: List[Dict]) -> Optional[Dict]:
        """选择最佳识别结果"""
        if not results:
            return None
        
        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # 选择置信度最高且文本合理的结果
        for result in sorted_results:
            text = result['text']
            
            # 验证文本
            validated_text = self._validate_and_clean_number_text(text)
            if validated_text:
                result['text'] = validated_text
                return result
        
        return None
    
    def _validate_and_clean_number_text(self, text: str) -> Optional[str]:
        """验证和清理数字文本"""
        if not text:
            return None
        
        # 清理文本
        cleaned = re.sub(r'[^\w\d万千,\.A-Z]', '', text)
        
        # 检查是否匹配数字模式
        for pattern in self.number_patterns:
            if re.match(pattern, cleaned):
                return cleaned
        
        # 如果不匹配模式，但主要是数字，返回数字部分
        digits = re.findall(r'\d+', cleaned)
        if digits:
            main_number = digits[0]
            
            # 检查是否有单位
            suffix = re.findall(r'[万千A-Z]', cleaned)
            if suffix:
                return main_number + suffix[0]
            else:
                return main_number
        
        return None