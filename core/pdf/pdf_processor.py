#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_processor.py
PDF处理器 - 路線価図検索システム
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
from PIL import Image
import io

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF处理器"""
    
    def __init__(self, dpi: int = 300):
        """
        初始化PDF处理器
        
        Args:
            dpi: 图像分辨率
        """
        self.dpi = dpi
        logger.info(f"PDF处理器初始化完成 (DPI: {dpi})")
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        从PDF中提取图像
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的图像列表
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 计算缩放矩阵
                zoom = self.dpi / 72.0  # 72 DPI是PDF的标准分辨率
                mat = fitz.Matrix(zoom, zoom)
                
                # 渲染页面为图像
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # 转换为numpy数组
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    images.append(image)
                    logger.info(f"页面 {page_num + 1}: 图像尺寸 {image.shape}")
                else:
                    logger.warning(f"页面 {page_num + 1}: 图像转换失败")
            
            doc.close()
            logger.info(f"PDF处理完成: {len(images)} 页")
            return images
            
        except Exception as e:
            logger.error(f"PDF图像提取失败: {e}")
            return []
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        获取PDF文件信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            PDF信息字典
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                'page_count': len(doc),
                'pages': []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_info = {
                    'page_num': page_num,
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'rotation': page.rotation
                }
                info['pages'].append(page_info)
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"获取PDF信息失败: {e}")
            return {}
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF中提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            文本信息列表
        """
        try:
            doc = fitz.open(pdf_path)
            text_info = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本块
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_info.append({
                                    'text': span["text"],
                                    'bbox': span["bbox"],
                                    'page': page_num + 1,
                                    'font': span.get("font", ""),
                                    'size': span.get("size", 0)
                                })
            
            doc.close()
            return text_info
            
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            return []
    
    def pdf_to_images(self, pdf_path: str) -> Dict[int, np.ndarray]:
        """
        将PDF转换为图像字典
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            页面编号到图像的映射
        """
        images = self.extract_images_from_pdf(pdf_path)
        return {i: img for i, img in enumerate(images)} 