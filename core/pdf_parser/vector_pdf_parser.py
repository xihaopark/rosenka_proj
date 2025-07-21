#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矢量PDF解析器 - 专门处理路線価図PDF
基于PyMuPDF，提取矢量数据而非简单渲染
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class VectorText:
    """矢量文字对象"""
    text: str
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    font: str
    size: float
    rotation: float = 0.0
    confidence: float = 1.0

@dataclass
class VectorPath:
    """矢量路径对象"""
    points: List[Tuple[float, float]]
    stroke_width: float
    stroke_color: Tuple[int, int, int]
    fill_color: Optional[Tuple[int, int, int]] = None
    path_type: str = "line"  # line, circle, polygon

@dataclass
class PriceCircle:
    """价格圆形标记"""
    center: Tuple[float, float]
    radius: float
    text: Optional[str] = None
    confidence: float = 0.0
    circle_type: str = "price_marker"

class VectorPDFParser:
    """矢量PDF解析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        解析PDF结构，提取矢量数据
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            解析结果
        """
        try:
            # 检查文件是否存在
            if not Path(pdf_path).exists():
                self.logger.error(f"PDF文件不存在: {pdf_path}")
                return {}
            
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            self.logger.info(f"成功打开PDF: {pdf_path}")
            self.logger.info(f"PDF页数: {len(doc)}")
            
            # 检查页面
            if len(doc) == 0:
                self.logger.error("PDF没有页面")
                doc.close()
                return {}
            
            page = doc[0]  # 获取第一页
            if page is None:
                self.logger.error("无法获取PDF页面")
                doc.close()
                return {}
            
            self.logger.info(f"页面尺寸: {page.rect}")
            
            # 1. 提取矢量路径（道路、边界等）
            vector_paths = self._extract_vector_paths(page)
            
            # 2. 提取文字对象
            vector_texts = self._extract_vector_texts(page)
            
            # 3. 提取图形对象（圆形等）
            vector_shapes = self._extract_vector_shapes(page)
            
            # 4. 分类处理
            classified_data = self._classify_elements(
                vector_paths, vector_texts, vector_shapes
            )
            
            # 5. 构建空间关系
            spatial_relations = self._build_spatial_relations(classified_data)
            
            doc.close()
            
            return {
                'vector_paths': vector_paths,
                'vector_texts': vector_texts,
                'vector_shapes': vector_shapes,
                'classified_data': classified_data,
                'spatial_relations': spatial_relations,
                'page_info': {
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'rotation': page.rotation
                }
            }
            
        except Exception as e:
            self.logger.error(f"PDF解析失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}
    
    def _extract_vector_paths(self, page) -> List[VectorPath]:
        """提取矢量路径"""
        paths = []
        
        try:
            # 获取所有绘图对象
            drawings = page.get_drawings()
            
            for item in drawings:
                if item['type'] == 'path':
                    # 道路线条
                    path = VectorPath(
                        points=item['points'],
                        stroke_width=item.get('width', 1.0),
                        stroke_color=item.get('stroke', (0, 0, 0)),
                        fill_color=item.get('fill'),
                        path_type='line'
                    )
                    paths.append(path)
                    
                elif item['type'] == 'circle':
                    # 圆形对象
                    path = VectorPath(
                        points=[item['center']],
                        stroke_width=item.get('width', 1.0),
                        stroke_color=item.get('stroke', (0, 0, 0)),
                        fill_color=item.get('fill'),
                        path_type='circle'
                    )
                    paths.append(path)
            
            self.logger.info(f"提取到 {len(paths)} 个矢量路径")
            return paths
            
        except Exception as e:
            self.logger.error(f"提取矢量路径失败: {e}")
            return []
    
    def _extract_vector_texts(self, page) -> List[VectorText]:
        """提取矢量文字"""
        texts = []
        
        try:
            # 获取文字字典
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # 文字块
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = VectorText(
                                text=span['text'].strip(),
                                bbox=span['bbox'],  # (x, y, width, height)
                                font=span['font'],
                                size=span['size'],
                                rotation=span.get('rotation', 0.0)
                            )
                            texts.append(text)
            
            self.logger.info(f"提取到 {len(texts)} 个文字对象")
            return texts
            
        except Exception as e:
            self.logger.error(f"提取矢量文字失败: {e}")
            return []
    
    def _extract_vector_shapes(self, page) -> List[Dict]:
        """提取图形对象"""
        shapes = []
        
        try:
            drawings = page.get_drawings()
            
            for item in drawings:
                if item['type'] == 'circle':
                    # 可能是价格标记
                    shape = {
                        'type': 'circle',
                        'center': item['center'],
                        'radius': item.get('radius', 10.0),
                        'fill': item.get('fill'),
                        'stroke': item.get('stroke'),
                        'stroke_width': item.get('width', 1.0)
                    }
                    shapes.append(shape)
                    
                elif item['type'] == 'rect':
                    # 矩形对象
                    shape = {
                        'type': 'rect',
                        'bbox': item['bbox'],
                        'fill': item.get('fill'),
                        'stroke': item.get('stroke')
                    }
                    shapes.append(shape)
            
            self.logger.info(f"提取到 {len(shapes)} 个图形对象")
            return shapes
            
        except Exception as e:
            self.logger.error(f"提取图形对象失败: {e}")
            return []
    
    def _classify_elements(self, paths: List[VectorPath], 
                          texts: List[VectorText], 
                          shapes: List[Dict]) -> Dict[str, Any]:
        """分类处理各种元素"""
        
        classified = {
            'road_network': [],
            'address_labels': [],
            'price_markers': [],
            'district_names': [],
            'other_texts': []
        }
        
        # 1. 分类文字
        for text in texts:
            if self._is_address_label(text):
                classified['address_labels'].append(text)
            elif self._is_district_name(text):
                classified['district_names'].append(text)
            elif self._is_price_text(text):
                classified['price_markers'].append(text)
            else:
                classified['other_texts'].append(text)
        
        # 2. 分类路径（道路网络）
        for path in paths:
            if path.path_type == 'line' and path.stroke_width > 0.5:
                classified['road_network'].append(path)
        
        # 3. 处理圆形标记
        for shape in shapes:
            if shape['type'] == 'circle' and shape.get('fill') == (0, 0, 0):
                # 黑色填充圆形，可能是价格标记
                circle = PriceCircle(
                    center=shape['center'],
                    radius=shape['radius'],
                    circle_type='price_marker'
                )
                classified['price_markers'].append(circle)
        
        return classified
    
    def _is_address_label(self, text: VectorText) -> bool:
        """判断是否为地址标签"""
        # 地址特征：包含"丁目"、"番地"等
        address_keywords = ['丁目', '番地', '号', '条', '町']
        return any(keyword in text.text for keyword in address_keywords)
    
    def _is_district_name(self, text: VectorText) -> bool:
        """判断是否为区域名称"""
        # 区域名称特征：较长，不包含数字
        return len(text.text) > 3 and not any(c.isdigit() for c in text.text)
    
    def _is_price_text(self, text: VectorText) -> bool:
        """判断是否为价格文本"""
        # 价格特征：数字+字母（如"115E"）
        import re
        pattern = r'^\d{2,3}[A-G]$'
        return bool(re.match(pattern, text.text))
    
    def _build_spatial_relations(self, classified_data: Dict) -> List[Dict]:
        """构建空间关系"""
        relations = []
        
        # 分析地址标签与价格标记的关系
        for address in classified_data['address_labels']:
            for price in classified_data['price_markers']:
                distance = self._calculate_distance(address.bbox, price.center)
                
                relation = {
                    'element1': 'address',
                    'element1_data': address.text,
                    'element2': 'price',
                    'element2_data': price.text if hasattr(price, 'text') else '',
                    'distance': distance,
                    'relation_type': 'near' if distance < 100 else 'far'
                }
                relations.append(relation)
        
        return relations
    
    def _calculate_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个元素之间的距离"""
        # 简化计算：中心点距离
        center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
        center2 = bbox2 if len(bbox2) == 2 else (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)
        
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def extract_price_circles_with_text(self, pdf_path: str) -> List[PriceCircle]:
        """提取价格圆形标记及其文字"""
        try:
            # 解析PDF
            result = self.parse_pdf_structure(pdf_path)
            
            price_circles = []
            
            # 从分类数据中提取价格标记
            for item in result.get('classified_data', {}).get('price_markers', []):
                if isinstance(item, PriceCircle):
                    # 尝试识别圆内文字
                    text = self._recognize_circle_text(item, result)
                    item.text = text
                    price_circles.append(item)
            
            return price_circles
            
        except Exception as e:
            self.logger.error(f"提取价格圆形失败: {e}")
            return []
    
    def _recognize_circle_text(self, circle: PriceCircle, parse_result: Dict) -> Optional[str]:
        """识别圆内文字"""
        # 在圆形附近的文字中寻找价格文本
        circle_center = circle.center
        circle_radius = circle.radius
        
        for text in parse_result.get('vector_texts', []):
            text_center = (text.bbox[0] + text.bbox[2]/2, text.bbox[1] + text.bbox[3]/2)
            distance = ((circle_center[0] - text_center[0])**2 + 
                       (circle_center[1] - text_center[1])**2)**0.5
            
            # 如果文字在圆形内部或附近
            if distance < circle_radius * 1.5:
                if self._is_price_text(text):
                    return text.text
        
        return None
    
    def render_to_image_with_annotations(self, pdf_path: str, 
                                       output_path: str = None) -> np.ndarray:
        """渲染PDF为图像并添加标注"""
        try:
            # 解析PDF
            result = self.parse_pdf_structure(pdf_path)
            
            # 渲染PDF页面
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放
            img_data = pix.tobytes("png")
            
            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 添加标注
            annotated_image = self._add_annotations(image, result)
            
            if output_path:
                cv2.imwrite(output_path, annotated_image)
            
            doc.close()
            return annotated_image
            
        except Exception as e:
            self.logger.error(f"渲染PDF失败: {e}")
            return None
    
    def _add_annotations(self, image: np.ndarray, parse_result: Dict) -> np.ndarray:
        """在图像上添加标注"""
        annotated = image.copy()
        
        # 标注文字区域
        for text in parse_result.get('vector_texts', []):
            x, y, w, h = text.bbox
            cv2.rectangle(annotated, (int(x), int(y)), 
                         (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(annotated, text.text[:10], (int(x), int(y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 标注价格圆形
        for circle in parse_result.get('classified_data', {}).get('price_markers', []):
            if isinstance(circle, PriceCircle):
                center = (int(circle.center[0]), int(circle.center[1]))
                radius = int(circle.radius)
                cv2.circle(annotated, center, radius, (255, 0, 0), 2)
                if circle.text:
                    cv2.putText(annotated, circle.text, 
                               (center[0] - 20, center[1] - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return annotated

def main():
    """测试函数"""
    parser = VectorPDFParser()
    
    # 测试PDF解析
    pdf_path = "rosenka_data/大阪府/吹田市/藤白台１/43009.pdf"
    
    if Path(pdf_path).exists():
        print("🔍 开始解析PDF...")
        result = parser.parse_pdf_structure(pdf_path)
        
        print(f"📊 解析结果:")
        print(f"   矢量路径: {len(result.get('vector_paths', []))}")
        print(f"   文字对象: {len(result.get('vector_texts', []))}")
        print(f"   图形对象: {len(result.get('vector_shapes', []))}")
        
        # 显示分类结果
        classified = result.get('classified_data', {})
        print(f"   地址标签: {len(classified.get('address_labels', []))}")
        print(f"   价格标记: {len(classified.get('price_markers', []))}")
        print(f"   道路网络: {len(classified.get('road_network', []))}")
        
        # 渲染带标注的图像
        annotated_image = parser.render_to_image_with_annotations(pdf_path, "annotated_map.png")
        if annotated_image is not None:
            print("✅ 已生成标注图像: annotated_map.png")
    else:
        print(f"❌ PDF文件不存在: {pdf_path}")

if __name__ == "__main__":
    main() 