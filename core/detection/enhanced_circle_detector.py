#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import logging
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CircleDetection:
    """圆形检测结果"""
    center: Tuple[int, int]
    radius: int
    bbox: Tuple[int, int, int, int]
    roi: np.ndarray
    confidence: float
    inner_text: str = ""
    ocr_confidence: float = 0.0

class EnhancedCircleDetector:
    """增强的圆形检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.init_ocr_engines()
    
    def init_ocr_engines(self):
        """初始化OCR引擎"""
        self.ocr_engines = {}
        
        # 初始化EasyOCR
        try:
            import easyocr
            self.ocr_engines['easyocr'] = easyocr.Reader(['ja', 'en'])
            self.logger.info("✓ EasyOCR 初始化成功")
        except Exception as e:
            self.logger.error(f"EasyOCR 初始化失败: {e}")
    
    def detect_circles_multi_scale(self, image):
        """多尺度圆形检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 预处理：增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
        
        # 多种检测配置
        configs = [
            # 检测小圆圈（如①②③）
            {'dp': 1, 'minDist': 15, 'param1': 100, 'param2': 20, 'minRadius': 5, 'maxRadius': 30},
            {'dp': 1, 'minDist': 20, 'param1': 80, 'param2': 25, 'minRadius': 8, 'maxRadius': 35},
            {'dp': 1, 'minDist': 25, 'param1': 60, 'param2': 30, 'minRadius': 10, 'maxRadius': 40},
            
            # 检测中等圆圈
            {'dp': 1, 'minDist': 30, 'param1': 100, 'param2': 35, 'minRadius': 15, 'maxRadius': 60},
            {'dp': 2, 'minDist': 40, 'param1': 80, 'param2': 40, 'minRadius': 20, 'maxRadius': 80},
            
            # 检测大圆圈
            {'dp': 2, 'minDist': 50, 'param1': 120, 'param2': 50, 'minRadius': 30, 'maxRadius': 120}
        ]
        
        all_circles = []
        
        for i, config in enumerate(configs):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                **config
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # 计算圆形质量分数
                    quality_score = self.evaluate_circle_quality(blurred, x, y, r)
                    
                    all_circles.append({
                        'center': (x, y),
                        'radius': r,
                        'quality': quality_score,
                        'config_id': i
                    })
        
        # 按质量分数排序并去重
        all_circles.sort(key=lambda x: x['quality'], reverse=True)
        filtered_circles = self.filter_overlapping_circles(all_circles)
        
        return filtered_circles
    
    def evaluate_circle_quality(self, gray_image, x, y, r):
        """评估圆形检测质量"""
        h, w = gray_image.shape
        
        # 检查边界
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            return 0.0
        
        # 创建圆形掩码
        mask = np.zeros_like(gray_image)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # 创建圆环掩码（边界）
        ring_mask = np.zeros_like(gray_image)
        cv2.circle(ring_mask, (x, y), r, 255, 2)
        
        # 计算圆形内部和边界的梯度强度
        edges = cv2.Canny(gray_image, 50, 150)
        
        # 圆环上的边缘强度
        ring_edges = cv2.bitwise_and(edges, ring_mask)
        ring_strength = np.sum(ring_edges) / (2 * np.pi * r + 1)
        
        # 圆形内部的变化程度
        roi = gray_image[y-r:y+r, x-r:x+r]
        if roi.size > 0:
            internal_variance = np.var(roi)
        else:
            internal_variance = 0
        
        # 综合质量分数
        quality = ring_strength * 0.7 + min(internal_variance / 100, 10) * 0.3
        
        return quality
    
    def filter_overlapping_circles(self, circles, min_distance=20):
        """过滤重叠的圆形"""
        if not circles:
            return []
        
        filtered = []
        used = set()
        
        for i, circle in enumerate(circles):
            if i in used:
                continue
            
            x1, y1 = circle['center']
            r1 = circle['radius']
            
            # 检查是否与已选择的圆形重叠
            overlap = False
            for existing in filtered:
                x2, y2 = existing['center']
                r2 = existing['radius']
                
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < min_distance or distance < (r1 + r2) * 0.8:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(circle)
                used.add(i)
        
        return filtered
    
    def extract_circle_roi(self, image, circle_info, expand_factor=1.8):
        """提取圆形区域"""
        x, y = circle_info['center']
        r = circle_info['radius']
        
        # 扩大提取区域
        expanded_r = int(r * expand_factor)
        
        x1 = max(0, x - expanded_r)
        y1 = max(0, y - expanded_r)
        x2 = min(image.shape[1], x + expanded_r)
        y2 = min(image.shape[0], y + expanded_r)
        
        roi = image[y1:y2, x1:x2]
        
        return roi, (x1, y1, x2, y2)
    
    def preprocess_circle_roi(self, roi):
        """预处理圆形区域以提高OCR识别率"""
        if roi.size == 0:
            return roi
        
        # 转换为灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 放大图像
        scale_factor = 4
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 高斯模糊去噪
        denoised = cv2.GaussianBlur(enlarged, (3, 3), 0)
        
        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_circled_numbers(self, image):
        """检测带圆圈的数字"""
        # 检测圆形
        circles = self.detect_circles_multi_scale(image)
        self.logger.info(f"检测到 {len(circles)} 个潜在圆形")
        
        circle_detections = []
        
        for i, circle_info in enumerate(circles):
            self.logger.info(f"处理圆形 {i+1}/{len(circles)}")
            
            # 提取ROI
            roi, bbox = self.extract_circle_roi(image, circle_info)
            
            if roi.size == 0:
                continue
            
            # 预处理ROI
            processed_roi = self.preprocess_circle_roi(roi)
            
            # OCR识别
            text, confidence = self.ocr_circle_content(processed_roi)
            
            # 创建检测结果
            detection = CircleDetection(
                center=circle_info['center'],
                radius=circle_info['radius'],
                bbox=bbox,
                roi=roi,
                confidence=circle_info['quality'],
                inner_text=text,
                ocr_confidence=confidence
            )
            
            circle_detections.append(detection)
        
        return circle_detections
    
    def ocr_circle_content(self, roi):
        """OCR识别圆形内容"""
        if 'easyocr' not in self.ocr_engines:
            return "", 0.0
        
        try:
            # 使用EasyOCR
            results = self.ocr_engines['easyocr'].readtext(roi)
            
            if not results:
                return "", 0.0
            
            # 选择置信度最高的结果
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            confidence = best_result[2]
            
            # 特殊处理：检测可能的带圆圈数字
            # 如果识别结果包含数字，可能是带圆圈的数字
            if any(c.isdigit() for c in text):
                return text, confidence
            
            # 检测可能的日文数字符号
            japanese_numbers = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
            for num in japanese_numbers:
                if num in text:
                    return text, confidence
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {e}")
            return "", 0.0
    
    def process_pdf_page(self, pdf_path, page_num=0):
        """处理PDF页面"""
        self.logger.info(f"处理PDF: {pdf_path}, 页面: {page_num}")
        
        # 打开PDF
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # 转换为超高分辨率图像
        mat = fitz.Matrix(4.0, 4.0)  # 4倍放大
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        self.logger.info(f"超高分辨率图像尺寸: {image.shape}")
        
        # 检测带圆圈的数字
        circle_detections = self.detect_circled_numbers(image)
        
        return circle_detections, image
    
    def visualize_circle_detections(self, image, detections, output_path):
        """可视化圆形检测结果"""
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x, y = detection.center
            r = detection.radius
            
            # 绘制圆形
            cv2.circle(vis_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
            
            # 绘制边界框
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # 绘制文本信息
            if detection.inner_text:
                label = f"{i+1}: {detection.inner_text} ({detection.ocr_confidence:.2f})"
            else:
                label = f"{i+1}: ? ({detection.confidence:.2f})"
            
            # 计算标签位置
            label_y = y1 - 10 if y1 > 20 else y2 + 20
            
            # 绘制标签背景
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, label_y - text_height - 5), 
                         (x1 + text_width, label_y + 5), (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(vis_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 保存结果
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"可视化结果保存到: {output_path}")
        
        return vis_image
    
    def save_circle_rois(self, detections, output_dir="circle_rois"):
        """保存圆形区域的ROI图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, detection in enumerate(detections):
            # 保存原始ROI
            roi_path = os.path.join(output_dir, f"circle_{i+1}_original.jpg")
            cv2.imwrite(roi_path, detection.roi)
            
            # 保存处理后的ROI
            processed_roi = self.preprocess_circle_roi(detection.roi)
            processed_path = os.path.join(output_dir, f"circle_{i+1}_processed.jpg")
            cv2.imwrite(processed_path, processed_roi)
            
            self.logger.info(f"保存ROI: {roi_path} 和 {processed_path}")

def main():
    """主函数"""
    detector = EnhancedCircleDetector()
    
    # 测试PDF文件
    pdf_path = "rosenka_data/大阪府/吹田市/藤白台１/43012.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    try:
        # 处理PDF
        start_time = time.time()
        detections, image = detector.process_pdf_page(pdf_path, 0)
        process_time = time.time() - start_time
        
        print(f"\n🎯 圆形检测结果统计:")
        print(f"检测到圆形数量: {len(detections)}")
        print(f"处理时间: {process_time:.2f}秒")
        
        # 显示检测结果
        print(f"\n📝 检测到的圆形内容:")
        for i, detection in enumerate(detections):
            print(f"{i+1:2d}. 中心: {detection.center}, 半径: {detection.radius}")
            print(f"    内容: '{detection.inner_text}' (OCR置信度: {detection.ocr_confidence:.2f})")
            print(f"    检测质量: {detection.confidence:.2f}")
            print()
        
        # 可视化结果
        output_path = "enhanced_circle_detection.jpg"
        detector.visualize_circle_detections(image, detections, output_path)
        
        # 保存ROI图像
        detector.save_circle_rois(detections)
        
        print(f"\n✅ 处理完成！")
        print(f"✅ 可视化结果: {output_path}")
        print(f"✅ ROI图像保存在: circle_rois/ 目录")
        
        # 保存结果为JSON
        json_output = []
        for detection in detections:
            json_output.append({
                'center': detection.center,
                'radius': detection.radius,
                'bbox': detection.bbox,
                'inner_text': detection.inner_text,
                'ocr_confidence': detection.ocr_confidence,
                'detection_quality': detection.confidence
            })
        
        with open('circle_detection_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: circle_detection_results.json")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 