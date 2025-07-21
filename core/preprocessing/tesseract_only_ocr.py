#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tesseract_only_ocr.py
只使用Tesseract的OCR处理器
避免NumPy兼容性问题
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float

class TesseractOnlyOCR:
    """只使用Tesseract的OCR处理器"""
    
    def __init__(self, lang: str = 'jpn+eng'):
        """
        初始化Tesseract OCR处理器
        Args:
            lang: 语言设置，默认日语+英语
        """
        self.lang = lang
        self.logger = logging.getLogger(__name__)
        
        # 检查Tesseract是否可用
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"✅ Tesseract版本: {version}")
            
            # 检查语言支持
            languages = pytesseract.get_languages()
            if 'jpn' in languages:
                self.logger.info("✅ 日语支持可用")
            else:
                self.logger.warning("⚠️ 日语支持不可用，使用英语")
                self.lang = 'eng'
                
        except Exception as e:
            self.logger.error(f"❌ Tesseract初始化失败: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 去噪
        denoised = cv2.medianBlur(enhanced, 3)
        
        # 二值化
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """简单文本提取"""
        try:
            # 预处理图像
            processed = self.preprocess_image(image)
            
            # 配置Tesseract
            config = f'--oem 3 --psm 6 -l {self.lang}'
            
            # 提取文本
            text = pytesseract.image_to_string(processed, config=config)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"文本提取失败: {e}")
            return ""
    
    def extract_text_detailed(self, image: np.ndarray) -> List[OCRResult]:
        """详细文本提取（包含位置信息）"""
        results = []
        
        try:
            # 预处理图像
            processed = self.preprocess_image(image)
            
            # 配置Tesseract
            config = f'--oem 3 --psm 6 -l {self.lang}'
            
            # 获取详细信息
            data = pytesseract.image_to_data(
                processed, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            # 解析结果
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and data['conf'][i] > 30:  # 置信度阈值
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    confidence = data['conf'][i] / 100.0
                    
                    results.append(OCRResult(
                        text=text,
                        bbox=bbox,
                        confidence=confidence
                    ))
            
        except Exception as e:
            self.logger.error(f"详细文本提取失败: {e}")
        
        return results
    
    def process_pdf_page(self, pdf_path: str, page_num: int, dpi: int = 300) -> List[OCRResult]:
        """处理PDF页面"""
        results = []
        
        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 渲染为图像
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # 转换为numpy数组
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 提取文本
            results = self.extract_text_detailed(image)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"PDF页面处理失败: {e}")
        
        return results
    
    def batch_process_images(self, image_paths: List[str]) -> Dict[str, List[OCRResult]]:
        """批量处理图像"""
        results = {}
        
        for image_path in image_paths:
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.warning(f"无法读取图像: {image_path}")
                    continue
                
                # 提取文本
                ocr_results = self.extract_text_detailed(image)
                results[image_path] = ocr_results
                
                self.logger.info(f"✅ 处理完成: {image_path} ({len(ocr_results)} 个文本块)")
                
            except Exception as e:
                self.logger.error(f"图像处理失败 {image_path}: {e}")
                results[image_path] = []
        
        return results
    
    def create_optimized_config(self, mode: str = 'document') -> str:
        """创建优化的Tesseract配置"""
        configs = {
            'document': f'--oem 3 --psm 6 -l {self.lang}',  # 文档模式
            'single_line': f'--oem 3 --psm 8 -l {self.lang}',  # 单行模式
            'single_word': f'--oem 3 --psm 10 -l {self.lang}',  # 单词模式
            'sparse': f'--oem 3 --psm 11 -l {self.lang}',  # 稀疏文本
            'raw_line': f'--oem 3 --psm 13 -l {self.lang}'  # 原始行模式
        }
        
        return configs.get(mode, configs['document'])

def test_tesseract_ocr():
    """测试Tesseract OCR"""
    print("🧪 测试Tesseract OCR...")
    
    try:
        # 创建OCR处理器
        ocr = TesseractOnlyOCR()
        
        # 创建测试图像
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, 'Test OCR', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 测试文本提取
        text = ocr.extract_text_simple(test_image)
        print(f"✅ 提取文本: '{text}'")
        
        # 测试详细提取
        results = ocr.extract_text_detailed(test_image)
        print(f"✅ 详细结果: {len(results)} 个文本块")
        
        for result in results:
            print(f"   文本: '{result.text}', 置信度: {result.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 Tesseract OCR处理器")
    print("=" * 50)
    
    # 测试OCR
    if test_tesseract_ocr():
        print("✅ Tesseract OCR处理器可用")
    else:
        print("❌ Tesseract OCR处理器不可用")

if __name__ == "__main__":
    main() 