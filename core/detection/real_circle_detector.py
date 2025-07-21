#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_circle_detector.py
真正的圆圈数字检测器 - 先检测圆圈，再识别数字
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from typing import List, Dict, Tuple
import easyocr
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealCircleDetector:
    """真正的圆圈检测器"""
    
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['ja', 'en'])
        
    def detect_circles_in_image(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的圆圈"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 方法1: HoughCircles检测
        circles_hough = self.detect_with_hough_circles(gray)
        
        # 方法2: 轮廓检测
        circles_contour = self.detect_with_contours(gray)
        
        # 合并结果
        all_circles = circles_hough + circles_contour
        
        # 去重和过滤
        filtered_circles = self.filter_and_merge_circles(all_circles)
        
        return filtered_circles
    
    def detect_with_hough_circles(self, gray: np.ndarray) -> List[Dict]:
        """使用HoughCircles检测圆形"""
        circles = []
        
        # 多种参数配置，专门检测小圆圈
        configs = [
            # 检测很小的圆圈 (半径5-20)
            {'dp': 1, 'minDist': 15, 'param1': 100, 'param2': 20, 'minRadius': 5, 'maxRadius': 20},
            {'dp': 1, 'minDist': 20, 'param1': 80, 'param2': 25, 'minRadius': 8, 'maxRadius': 25},
            {'dp': 1, 'minDist': 25, 'param1': 60, 'param2': 30, 'minRadius': 10, 'maxRadius': 30},
            
            # 检测中等圆圈 (半径15-35)
            {'dp': 1, 'minDist': 30, 'param1': 100, 'param2': 35, 'minRadius': 15, 'maxRadius': 35},
            {'dp': 2, 'minDist': 40, 'param1': 80, 'param2': 40, 'minRadius': 20, 'maxRadius': 40},
        ]
        
        for config in configs:
            detected = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                **config
            )
            
            if detected is not None:
                detected = np.round(detected[0, :]).astype("int")
                for (x, y, r) in detected:
                    # 评估圆圈质量
                    quality = self.evaluate_circle_quality(gray, x, y, r)
                    if quality > 0.3:  # 质量阈值
                        circles.append({
                            'center': (x, y),
                            'radius': r,
                            'quality': quality,
                            'method': 'hough'
                        })
        
        return circles
    
    def detect_with_contours(self, gray: np.ndarray) -> List[Dict]:
        """使用轮廓检测圆形"""
        circles = []
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 计算轮廓面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 100 or area > 2000:  # 面积过滤
                continue
                
            if perimeter < 20:  # 周长过滤
                continue
            
            # 计算圆形度
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.6:  # 圆形度阈值
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y, radius = int(x), int(y), int(radius)
                
                if 5 <= radius <= 35:  # 半径范围
                    circles.append({
                        'center': (x, y),
                        'radius': radius,
                        'quality': circularity,
                        'method': 'contour'
                    })
        
        return circles
    
    def evaluate_circle_quality(self, gray: np.ndarray, x: int, y: int, r: int) -> float:
        """评估圆圈质量"""
        h, w = gray.shape
        
        # 边界检查
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            return 0.0
        
        # 创建圆形掩码
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # 创建圆环掩码（边界）
        ring_mask = np.zeros_like(gray)
        cv2.circle(ring_mask, (x, y), r, 255, 2)
        
        # 计算边缘强度
        edges = cv2.Canny(gray, 50, 150)
        ring_edges = cv2.bitwise_and(edges, ring_mask)
        ring_strength = np.sum(ring_edges) / (2 * np.pi * r + 1)
        
        # 计算内部一致性
        roi = gray[y-r:y+r, x-r:x+r]
        if roi.size > 0:
            internal_variance = np.var(roi)
            # 圆圈内部应该相对均匀
            consistency = max(0, 1 - internal_variance / 1000)
        else:
            consistency = 0
        
        # 综合质量分数
        quality = ring_strength * 0.6 + consistency * 0.4
        
        return min(quality, 1.0)
    
    def filter_and_merge_circles(self, circles: List[Dict]) -> List[Dict]:
        """过滤和合并圆圈"""
        if not circles:
            return []
        
        # 按质量排序
        circles.sort(key=lambda x: x['quality'], reverse=True)
        
        filtered = []
        for circle in circles:
            # 检查是否与已有圆圈重叠
            overlap = False
            for existing in filtered:
                dist = np.sqrt((circle['center'][0] - existing['center'][0])**2 + 
                             (circle['center'][1] - existing['center'][1])**2)
                if dist < (circle['radius'] + existing['radius']) * 0.7:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(circle)
        
        return filtered
    
    def extract_and_recognize_circle_content(self, image: np.ndarray, circle: Dict) -> str:
        """提取并识别圆圈内容"""
        x, y = circle['center']
        r = circle['radius']
        
        # 提取圆圈区域（稍微扩大）
        expand_factor = 1.2
        expanded_r = int(r * expand_factor)
        
        x1 = max(0, x - expanded_r)
        y1 = max(0, y - expanded_r)
        x2 = min(image.shape[1], x + expanded_r)
        y2 = min(image.shape[0], y + expanded_r)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # 图像预处理
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        # 放大图像
        scale_factor = 4
        roi_large = cv2.resize(roi_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        roi_enhanced = clahe.apply(roi_large)
        
        # OCR识别
        try:
            results = self.ocr_reader.readtext(roi_enhanced)
            if results:
                # 选择置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]
                
                # 过滤：只保留数字或简单字符
                if text.isdigit() or text in ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']:
                    return text
                    
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
        
        return ""
    
    def process_pdf_for_circles(self, pdf_path: str) -> List[Dict]:
        """处理PDF文件，检测圆圈数字"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []
        
        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            
            # 高分辨率转换
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat)  # type: ignore
            img_data = pix.tobytes("png")
            
            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            doc.close()
            
            # 去除表头（上部15%）
            header_height = int(image.shape[0] * 0.15)
            image = image[header_height:, :]
            
            # 检测圆圈
            circles = self.detect_circles_in_image(image)
            
            # 识别圆圈内容
            results = []
            for circle in circles:
                text = self.extract_and_recognize_circle_content(image, circle)
                if text:  # 只保留成功识别的结果
                    x, y = circle['center']
                    r = circle['radius']
                    
                    # 计算bbox（相对于去除表头后的图像）
                    bbox = [x - r, y - r, x + r, y + r]
                    
                    # 调整到原始图像坐标
                    bbox[1] += header_height
                    bbox[3] += header_height
                    
                    results.append({
                        'text': text,
                        'bbox': bbox,
                        'center': (x, y + header_height),
                        'radius': r,
                        'confidence': circle['quality'],
                        'detection_type': 'real_circle',
                        'method': circle['method'],
                        'pdf_path': pdf_path
                    })
            
            logger.info(f"在 {pdf_path} 中检测到 {len(results)} 个圆圈数字")
            return results
            
        except Exception as e:
            logger.error(f"处理PDF失败: {e}")
            return []

def test_real_circle_detector():
    """测试真正的圆圈检测器"""
    detector = RealCircleDetector()
    
    # 测试藤白台1文件夹中的PDF
    test_folder = "rosenka_data/大阪府/吹田市/藤白台１"
    
    if not os.path.exists(test_folder):
        print(f"❌ 测试文件夹不存在: {test_folder}")
        return
    
    pdf_files = [f for f in os.listdir(test_folder) if f.endswith('.pdf')]
    print(f"🔍 找到 {len(pdf_files)} 个PDF文件")
    
    total_circles = 0
    for pdf_file in pdf_files[:3]:  # 测试前3个文件
        pdf_path = os.path.join(test_folder, pdf_file)
        print(f"\n📄 处理: {pdf_file}")
        
        results = detector.process_pdf_for_circles(pdf_path)
        total_circles += len(results)
        
        print(f"   🔴 检测到 {len(results)} 个圆圈数字:")
        for result in results:
            print(f"      • {result['text']} (置信度: {result['confidence']:.3f}, 方法: {result['method']})")
    
    print(f"\n🎯 总计检测到 {total_circles} 个真正的圆圈数字")

if __name__ == "__main__":
    test_real_circle_detector() 