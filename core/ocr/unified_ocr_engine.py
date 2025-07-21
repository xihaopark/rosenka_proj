#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_ocr_engine.py
统一OCR引擎 - 集成增强数字识别和通用文字识别
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .fixed_simple_ocr import FixedSimpleOCR
from .enhanced_number_ocr import EnhancedNumberOCR
from .base_ocr_engine import BaseOCREngine

logger = logging.getLogger(__name__)

class UnifiedOCREngine(BaseOCREngine):
    """
    统一OCR引擎
    结合通用文字识别和专门的数字识别能力
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 enable_number_enhancement: bool = True,
                 confidence_threshold: float = 0.3):
        """
        初始化统一OCR引擎
        
        Args:
            use_gpu: 是否使用GPU加速
            enable_number_enhancement: 是否启用数字增强识别
            confidence_threshold: 置信度阈值
        """
        super().__init__()
        
        self.use_gpu = use_gpu
        self.enable_number_enhancement = enable_number_enhancement
        self.confidence_threshold = confidence_threshold
        
        # 初始化通用OCR引擎
        self.general_ocr = FixedSimpleOCR(lang='japan')
        
        # 初始化数字增强OCR引擎
        if enable_number_enhancement:
            try:
                self.number_ocr = EnhancedNumberOCR(use_gpu=use_gpu)
                logger.info("数字增强OCR引擎初始化成功")
            except Exception as e:
                logger.warning(f"数字增强OCR引擎初始化失败，将使用通用引擎: {e}")
                self.number_ocr = None
                self.enable_number_enhancement = False
        else:
            self.number_ocr = None
        
        logger.info(f"统一OCR引擎初始化完成 (数字增强: {self.enable_number_enhancement})")
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        检测文本区域（通用+数字增强）
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的文本区域列表
        """
        try:
            all_regions = []
            
            # 1. 通用文字识别
            general_regions = self.general_ocr.detect_text_regions(image)
            all_regions.extend(general_regions)
            
            # 2. 数字增强识别
            if self.enable_number_enhancement and self.number_ocr:
                number_regions = self.number_ocr.detect_number_regions(image)
                
                # 对数字区域进行详细识别
                enhanced_number_regions = self.number_ocr.recognize_numbers_in_regions(
                    image, number_regions
                )
                
                # 标记为数字增强结果
                for region in enhanced_number_regions:
                    region['is_number_enhanced'] = True
                
                all_regions.extend(enhanced_number_regions)
            
            # 3. 合并和去重
            merged_regions = self._merge_and_deduplicate(all_regions)
            
            # 4. 过滤低质量结果
            filtered_regions = self._filter_by_quality(merged_regions)
            
            logger.info(f"统一OCR检测到 {len(filtered_regions)} 个文本区域")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"统一OCR文本检测失败: {e}")
            return []
    
    def _merge_and_deduplicate(self, regions: List[Dict]) -> List[Dict]:
        """合并和去重文本区域"""
        if not regions:
            return []
        
        # 按置信度排序
        sorted_regions = sorted(regions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        merged = []
        
        for region in sorted_regions:
            bbox = region['bbox']
            is_duplicate = False
            
            for existing in merged:
                existing_bbox = existing['bbox']
                
                # 计算重叠率
                overlap = self._calculate_bbox_overlap(bbox, existing_bbox)
                
                # 如果重叠超过阈值
                if overlap > 0.6:
                    # 选择更好的结果
                    better_region = self._select_better_region(region, existing)
                    
                    if better_region == region:
                        # 替换现有结果
                        merged.remove(existing)
                        merged.append(region)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(region)
        
        return merged
    
    def _calculate_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
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
            
            # 计算较小区域的面积
            area1 = bbox1['width'] * bbox1['height']
            area2 = bbox2['width'] * bbox2['height']
            min_area = min(area1, area2)
            
            # 返回重叠率（相对于较小区域）
            return overlap_area / min_area if min_area > 0 else 0
            
        except Exception as e:
            logger.error(f"边界框重叠计算失败: {e}")
            return 0
    
    def _select_better_region(self, region1: Dict, region2: Dict) -> Dict:
        """选择更好的识别结果"""
        # 优先级规则：
        # 1. 数字增强结果优先
        # 2. 置信度更高
        # 3. 文本内容更完整
        
        # 检查是否为数字增强结果
        is_number1 = region1.get('is_number_enhanced', False)
        is_number2 = region2.get('is_number_enhanced', False)
        
        if is_number1 and not is_number2:
            return region1
        elif is_number2 and not is_number1:
            return region2
        
        # 比较置信度
        conf1 = region1.get('confidence', 0)
        conf2 = region2.get('confidence', 0)
        
        if abs(conf1 - conf2) > 0.1:  # 置信度差异明显
            return region1 if conf1 > conf2 else region2
        
        # 比较文本长度（更完整的文本）
        text1 = region1.get('text', '')
        text2 = region2.get('text', '')
        
        if len(text1) != len(text2):
            return region1 if len(text1) > len(text2) else region2
        
        # 默认返回置信度更高的
        return region1 if conf1 >= conf2 else region2
    
    def _filter_by_quality(self, regions: List[Dict]) -> List[Dict]:
        """根据质量过滤结果"""
        filtered = []
        
        for region in regions:
            # 检查置信度
            confidence = region.get('confidence', 0)
            if confidence < self.confidence_threshold:
                continue
            
            # 检查边界框
            bbox = region['bbox']
            if bbox['width'] < 5 or bbox['height'] < 5:
                continue
            
            # 检查文本内容
            text = region.get('text', '').strip()
            if not text:
                continue
            
            # 检查宽高比
            aspect_ratio = bbox['width'] / bbox['height']
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            filtered.append(region)
        
        return filtered
    
    def enhance_image_for_ocr(self, image: np.ndarray, region_type: str = 'general') -> np.ndarray:
        """
        根据区域类型增强图像
        
        Args:
            image: 输入图像
            region_type: 区域类型 ('general' 或 'number')
            
        Returns:
            增强后的图像
        """
        if region_type == 'number' and self.number_ocr:
            return self.number_ocr.enhance_number_regions(image)
        else:
            return self.general_ocr.preprocess_image(image)
    
    def recognize_text(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        识别指定区域中的文本（实现抽象方法）
        
        Args:
            image: 输入图像
            regions: 文本区域列表
            
        Returns:
            识别结果列表
        """
        results = []
        
        for region in regions:
            try:
                bbox = region['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # 提取ROI
                roi = image[y:y+h, x:x+w]
                
                # 根据区域类型选择识别方法
                if region.get('is_number_enhanced', False) and self.number_ocr:
                    # 使用数字增强识别
                    enhanced_roi = self.number_ocr.enhance_number_regions(roi)
                    # 数字识别已经在detect_text_regions中完成
                    results.append(region)
                else:
                    # 使用通用识别
                    enhanced_roi = self.general_ocr.preprocess_image(roi)
                    # 这里可以添加额外的识别逻辑
                    results.append(region)
                
            except Exception as e:
                logger.error(f"区域文本识别失败: {e}")
                continue
        
        return results
    
    def recognize_text_in_regions(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        在指定区域识别文本（兼容性方法）
        
        Args:
            image: 输入图像
            regions: 文本区域列表
            
        Returns:
            识别结果
        """
        return self.recognize_text(image, regions)
    
    def get_engine_info(self) -> Dict:
        """获取引擎信息"""
        info = {
            'name': 'UnifiedOCR',
            'version': '1.0',
            'gpu_enabled': self.use_gpu,
            'number_enhancement': self.enable_number_enhancement,
            'confidence_threshold': self.confidence_threshold,
            'general_engine': self.general_ocr.get_engine_info() if self.general_ocr else None
        }
        
        if self.number_ocr:
            info['number_engine'] = 'EnhancedNumberOCR'
        
        return info