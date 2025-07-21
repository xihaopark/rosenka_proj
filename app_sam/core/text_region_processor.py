"""
text_region_processor.py
文本区域后处理和OCR映射
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TextSegment:
    """文本片段"""
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    image: np.ndarray
    confidence: float
    text: str = ""
    ocr_confidence: float = 0.0
    page_num: int = 0
    pdf_path: str = ""

class TextRegionProcessor:
    """文本区域处理器"""
    
    def __init__(self, ocr_engine='paddleocr'):
        self.ocr_engine = self._init_ocr(ocr_engine)
        
    def _init_ocr(self, engine_type):
        """初始化OCR引擎"""
        if engine_type == 'paddleocr':
            from paddleocr import PaddleOCR
            return PaddleOCR(use_angle_cls=True, lang='japan')
        else:
            # 其他OCR引擎
            pass
    
    def process_sam_regions(self, image: np.ndarray, sam_regions: List[Dict]) -> List[TextSegment]:
        """
        处理SAM分割的区域
        
        1. 提取每个区域的图像
        2. 进行OCR识别
        3. 创建文本片段对象
        """
        text_segments = []
        
        for region in sam_regions:
            # 提取区域
            segment = self._extract_region(image, region)
            
            # OCR识别
            if segment is not None:
                text_result = self._ocr_region(segment.image)
                segment.text = text_result['text']
                segment.ocr_confidence = text_result['confidence']
                
                text_segments.append(segment)
        
        return text_segments
    
    def _extract_region(self, image: np.ndarray, region: Dict) -> TextSegment:
        """提取单个区域"""
        bbox = region['bbox']  # x, y, w, h
        x, y, w, h = bbox
        
        # 提取区域图像
        region_image = image[y:y+h, x:x+w].copy()
        
        # 应用mask
        if 'segmentation' in region:
            mask = region['segmentation'][y:y+h, x:x+w]
            # 创建白色背景
            white_bg = np.ones_like(region_image) * 255
            # 应用mask
            region_image = np.where(mask[..., None], region_image, white_bg)
        
        return TextSegment(
            mask=region.get('segmentation', None),
            bbox=bbox,
            image=region_image,
            confidence=region.get('predicted_iou', 1.0)
        )
    
    def _ocr_region(self, image: np.ndarray) -> Dict:
        """对区域进行OCR识别"""
        if self.ocr_engine is None:
            return {'text': '', 'confidence': 0.0}
        
        try:
            # PaddleOCR
            result = self.ocr_engine.ocr(image, cls=True)
            
            # 提取文本
            texts = []
            confidences = []
            
            for line in result[0] if result[0] else []:
                if len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    confidences.append(confidence)
            
            # 合并文本
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"OCR错误: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def create_text_map(self, text_segments: List[TextSegment], 
                       page_size: Tuple[int, int]) -> Dict:
        """
        创建文本映射图
        
        Returns:
            包含所有文本位置和内容的映射
        """
        text_map = {
            'page_size': page_size,
            'segments': []
        }
        
        for segment in text_segments:
            x, y, w, h = segment.bbox
            
            # 归一化坐标
            norm_x = x / page_size[0]
            norm_y = y / page_size[1]
            norm_w = w / page_size[0]
            norm_h = h / page_size[1]
            
            text_map['segments'].append({
                'text': segment.text,
                'bbox': segment.bbox,
                'normalized_bbox': (norm_x, norm_y, norm_w, norm_h),
                'confidence': segment.confidence,
                'ocr_confidence': segment.ocr_confidence,
                'center': ((x + w/2) / page_size[0], (y + h/2) / page_size[1])
            })
        
        return text_map