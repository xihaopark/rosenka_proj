#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_processor.py
简化版CV处理器 - 只保留核心的OCR和图像处理功能
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from typing import List, Dict, Tuple, Optional
import pytesseract
from rapidfuzz import fuzz
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= 数据结构 =========================

@dataclass
class AddressLocation:
    """地址位置信息"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    pdf_name: str
    page_num: int
    prefecture: str
    city: str
    district: str
    method: str = "Tesseract"

# ========================= 简化版PDF处理器 =========================

class SimplePDFProcessor:
    """简化版PDF处理器"""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.check_dependencies()
    
    def check_dependencies(self):
        """检查依赖关系"""
        try:
            available_langs = pytesseract.get_languages()
            self.tesseract_available = 'jpn' in available_langs
            if not self.tesseract_available:
                logger.warning("Japanese language pack not found, using English")
        except:
            self.tesseract_available = False
            logger.error("Tesseract not available")
        
        # 简化版本不使用EasyOCR，只使用Tesseract
        self.easyocr_available = False
    
    def pdf_to_images(self, pdf_path: str) -> Dict[int, np.ndarray]:
        """将PDF转换为图像"""
        try:
            doc = fitz.open(pdf_path)
            images = {}
            
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                images[page_num] = np.array(img)
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            return {}
    
    def extract_addresses(self, image: np.ndarray, pdf_name: str, page_num: int,
                         prefecture: str, city: str, district: str) -> List[AddressLocation]:
        """从图像中提取地址信息"""
        addresses = []
        
        if self.tesseract_available:
            addresses = self._extract_with_tesseract(
                image, pdf_name, page_num, prefecture, city, district
            )
        else:
            logger.warning("Tesseract不可用，无法提取文本")
        
        return self._deduplicate_addresses(addresses)
    
    def _extract_with_tesseract(self, image: np.ndarray, pdf_name: str, page_num: int,
                               prefecture: str, city: str, district: str) -> List[AddressLocation]:
        """使用Tesseract提取文本"""
        results = []
        h, w = image.shape[:2]
        
        # 使用多种窗口大小进行扫描
        window_sizes = [512, 1024]  # 简化窗口大小
        
        for window_size in window_sizes:
            stride = window_size // 2
            
            for y in range(0, h - window_size + 1, stride):
                for x in range(0, w - window_size + 1, stride):
                    roi = image[y:y+window_size, x:x+window_size]
                    
                    # 图像预处理
                    processed_roi = self._preprocess_image(roi)
                    
                    try:
                        # 使用Tesseract进行OCR
                        data = pytesseract.image_to_data(
                            processed_roi,
                            lang='jpn' if self.tesseract_available else 'eng',
                            output_type=pytesseract.Output.DICT,
                            config='--psm 6'  # 统一的文本块
                        )
                        
                        # 将单词组合成行
                        lines = self._group_words_to_lines(data)
                        
                        for line_text, line_bbox, conf in lines:
                            if self._is_meaningful_text(line_text):
                                # 转换为绝对坐标
                                abs_bbox = (
                                    x + line_bbox[0],
                                    y + line_bbox[1],
                                    x + line_bbox[2],
                                    y + line_bbox[3]
                                )
                                
                                results.append(AddressLocation(
                                    text=line_text,
                                    bbox=abs_bbox,
                                    confidence=float(conf),
                                    pdf_name=pdf_name,
                                    page_num=int(page_num),
                                    prefecture=prefecture,
                                    city=city,
                                    district=district,
                                    method="Tesseract"
                                ))
                    except Exception as e:
                        logger.debug(f"OCR提取失败: {e}")
                        continue
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 降噪
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _group_words_to_lines(self, ocr_data: Dict) -> List[Tuple[str, Tuple, float]]:
        """将OCR识别的单词组合成行"""
        from collections import defaultdict
        
        lines = defaultdict(list)
        n = len(ocr_data['text'])
        
        for i in range(n):
            text = ocr_data['text'][i].strip()
            if not text:
                continue
            
            # 检查置信度
            conf = float(ocr_data['conf'][i]) if str(ocr_data['conf'][i]).replace('.', '').isdigit() else 0
            if conf < 30:  # 置信度阈值
                continue
            
            # 行ID
            line_id = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
            
            lines[line_id].append({
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': conf
            })
        
        # 组合每行的文本
        results = []
        for words in lines.values():
            if not words:
                continue
            
            # 按x坐标排序
            words.sort(key=lambda w: w['left'])
            
            # 合并文本
            text = ''.join(w['text'] for w in words)
            
            # 计算边界框
            min_x = min(w['left'] for w in words)
            min_y = min(w['top'] for w in words)
            max_x = max(w['left'] + w['width'] for w in words)
            max_y = max(w['top'] + w['height'] for w in words)
            
            # 平均置信度
            avg_conf = sum(w['conf'] for w in words) / len(words)
            
            results.append((text, (int(min_x), int(min_y), int(max_x), int(max_y)), float(avg_conf)))
        
        return results
    
    def _is_meaningful_text(self, text: str) -> bool:
        """判断文本是否有意义"""
        text = text.strip()
        if len(text) < 2:
            return False
        
        # 有意义的模式
        patterns = [
            r'\d+[-ー]\d+',          # 番地格式 (88-7)
            r'.*[町丁目番地号]',      # 包含地名标记
            r'\d+[A-Z]',             # 路线价格 (120E)
            r'[東西南北].*\d',        # 方位+数字
            r'\d{2,}',               # 多位数字
            r'[一二三四五六七八九十]+丁目',  # 丁目
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        # 包含足够的汉字
        if len(re.findall(r'[\u4e00-\u9fff]', text)) >= 2:
            return True
        
        return False
    
    def _deduplicate_addresses(self, addresses: List[AddressLocation]) -> List[AddressLocation]:
        """去除重复的地址"""
        if not addresses:
            return []
        
        unique = []
        used = set()
        
        for i, addr1 in enumerate(addresses):
            if i in used:
                continue
            
            # 找到重叠的地址
            group = [addr1]
            for j, addr2 in enumerate(addresses[i+1:], i+1):
                if j in used:
                    continue
                
                # 计算重叠度
                overlap = self._calculate_overlap(addr1.bbox, addr2.bbox)
                if overlap > 0.5:  # 重叠阈值
                    group.append(addr2)
                    used.add(j)
            
            # 选择最佳的地址（文本最长且置信度最高）
            best = max(group, key=lambda a: (len(a.text), a.confidence))
            unique.append(best)
        
        return unique
    
    def _calculate_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个边界框的重叠度"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算交集
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        intersection = x_overlap * y_overlap
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

# ========================= 简化版搜索引擎 =========================

class SimpleSearchEngine:
    """简化版搜索引擎"""
    
    def __init__(self, addresses: List[AddressLocation]):
        self.addresses = addresses
    
    def search(self, query: str, threshold: float = 70.0) -> List[Dict]:
        """搜索地址"""
        results = []
        
        for addr in self.addresses:
            similarity = self._calculate_similarity(query, addr.text)
            if similarity >= threshold:
                results.append({
                    'address': addr,
                    'similarity': similarity,
                    'text': addr.text,
                    'bbox': addr.bbox,
                    'confidence': addr.confidence,
                    'pdf_name': addr.pdf_name,
                    'page_num': addr.page_num,
                    'prefecture': addr.prefecture,
                    'city': addr.city,
                    'district': addr.district,
                    'method': addr.method
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """计算文本相似度"""
        # 精确匹配
        if query in text or text in query:
            return 100.0
        
        # 模糊匹配
        return fuzz.partial_ratio(query, text)

# ========================= 工具函数 =========================

def create_simple_processor(dpi: int = 300) -> SimplePDFProcessor:
    """创建简化版处理器"""
    return SimplePDFProcessor(dpi)

def process_pdf_simple(pdf_path: str, prefecture: str, city: str, district: str, 
                      dpi: int = 300) -> List[AddressLocation]:
    """简化版PDF处理函数"""
    processor = create_simple_processor(dpi)
    
    # 转换PDF为图像
    images = processor.pdf_to_images(pdf_path)
    if not images:
        return []
    
    # 提取所有地址
    all_addresses = []
    for page_num, image in images.items():
        addresses = processor.extract_addresses(
            image, pdf_path, page_num, prefecture, city, district
        )
        all_addresses.extend(addresses)
    
    return all_addresses

def search_addresses_simple(addresses: List[AddressLocation], query: str, 
                          threshold: float = 70.0) -> List[Dict]:
    """简化版地址搜索"""
    engine = SimpleSearchEngine(addresses)
    return engine.search(query, threshold) 