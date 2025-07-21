#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modern_text_detector.py
现代化的文字检测方案 - 无需PaddlePaddle
使用CRAFT进行文字检测，TrOCR进行识别
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# CRAFT文字检测
from craft_text_detector import Craft

# Hugging Face transformers
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# 已有的Tesseract作为备份
import pytesseract

logger = logging.getLogger(__name__)

@dataclass
class ModernTextDetection:
    """现代文字检测结果"""
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    method: str
    text_type: Optional[str] = None

class ModernTextDetector:
    """现代化文字检测器 - 稳定可靠"""
    
    def __init__(self, use_trocr: bool = False):
        """
        初始化检测器
        Args:
            use_trocr: 是否使用TrOCR（需要下载模型，约1GB）
        """
        # CRAFT检测器
        self.craft = Craft(
            output_dir=None,
            crop_type="box",
            cuda=False,
            export_extra=False
        )
        
        # OCR选项
        self.use_trocr = use_trocr
        if use_trocr:
            self._init_trocr()
        else:
            # 使用Tesseract作为OCR引擎
            self.ocr_method = "Tesseract"
            
    def _init_trocr(self):
        """初始化TrOCR（可选）"""
        try:
            # 使用小模型以提高速度
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-small-printed"
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-small-printed"
            )
            self.ocr_method = "TrOCR"
            logger.info("TrOCR初始化成功")
        except Exception as e:
            logger.warning(f"TrOCR初始化失败，降级到Tesseract: {e}")
            self.use_trocr = False
            self.ocr_method = "Tesseract"
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """使用CRAFT检测文字区域"""
        # CRAFT预期BGR格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # 检测文字区域
        prediction_result = self.craft.detect_text(image)
        
        # 提取边界框
        boxes = []
        for box in prediction_result['boxes']:
            # box格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            
            bbox = {
                'x1': int(min(x_coords)),
                'y1': int(min(y_coords)),
                'x2': int(max(x_coords)),
                'y2': int(max(y_coords)),
                'polygon': box.tolist()
            }
            boxes.append(bbox)
        
        return boxes
    
    def recognize_text(self, image: np.ndarray, bbox: Dict) -> Tuple[str, float]:
        """识别单个文字区域"""
        # 裁剪区域
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "", 0.0
        
        if self.use_trocr and hasattr(self, 'trocr_model'):
            return self._recognize_with_trocr(roi)
        else:
            return self._recognize_with_tesseract(roi)
    
    def _recognize_with_tesseract(self, roi: np.ndarray) -> Tuple[str, float]:
        """使用Tesseract识别"""
        try:
            # 转为灰度图
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Tesseract配置
            config = '--oem 3 --psm 8 -l jpn+eng'
            
            # 获取详细数据
            data = pytesseract.image_to_data(
                gray, config=config, output_type=pytesseract.Output.DICT
            )
            
            # 提取文本和置信度
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i]) if str(data['conf'][i]).replace('.', '').isdigit() else 0
                
                if text and conf > 0:
                    texts.append(text)
                    confidences.append(conf)
            
            if texts:
                combined_text = ' '.join(texts)
                avg_conf = sum(confidences) / len(confidences) / 100.0
                return combined_text, avg_conf
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Tesseract识别失败: {e}")
            return "", 0.0
    
    def _recognize_with_trocr(self, roi: np.ndarray) -> Tuple[str, float]:
        """使用TrOCR识别"""
        try:
            # 转为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            # 处理图像
            pixel_values = self.trocr_processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values
            
            # 生成文本
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # TrOCR不直接提供置信度，使用固定值
            return text, 0.9
            
        except Exception as e:
            logger.error(f"TrOCR识别失败: {e}")
            return "", 0.0
    
    def detect_and_recognize(self, image: np.ndarray) -> List[ModernTextDetection]:
        """完整的检测和识别流程"""
        # 1. 检测文字区域
        boxes = self.detect_text_regions(image)
        
        # 2. 识别每个区域
        results = []
        for box in boxes:
            text, confidence = self.recognize_text(image, box)
            
            if text and confidence > 0.3:
                detection = ModernTextDetection(
                    bbox=(box['x1'], box['y1'], box['x2'], box['y2']),
                    text=text,
                    confidence=confidence,
                    method=self.ocr_method,
                    text_type=self._classify_text(text, box)
                )
                results.append(detection)
        
        return results
    
    def _classify_text(self, text: str, box: Dict) -> str:
        """分类文字类型"""
        import re
        
        # 路线价模式
        if re.match(r'^\d+[A-Z]$', text) or re.match(r'^\d{2,3}$', text):
            return 'price'
        
        # 地址模式
        if any(char in text for char in ['市', '区', '町', '丁目']):
            return 'address'
        
        # 基于大小判断
        width = box['x2'] - box['x1']
        height = box['y2'] - box['y1']
        
        if width > height * 3:
            return 'street'
        
        return 'other' 