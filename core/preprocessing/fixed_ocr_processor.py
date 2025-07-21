#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import logging
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    method: str
    detection_type: str = "text"  # text, circle, patch
    
class FixedOCRProcessor:
    """修复后的OCR处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ocr_engines = {}
        self.init_ocr_engines()
    
    def init_ocr_engines(self):
        """初始化OCR引擎"""
        
        # 初始化EasyOCR (主要引擎)
        try:
            import easyocr
            self.ocr_engines['easyocr'] = easyocr.Reader(['ja', 'en'])
            self.logger.info("✓ EasyOCR 初始化成功")
        except Exception as e:
            self.logger.error(f"EasyOCR 初始化失败: {e}")
        
        # 初始化PaddleOCR (使用正确参数)
        try:
            import paddleocr
            # 修复参数：移除use_gpu，使用正确的参数名
            self.ocr_engines['paddleocr'] = paddleocr.PaddleOCR(
                use_textline_orientation=True,  # 替代use_angle_cls
                lang='japan',
                show_log=False
            )
            self.logger.info("✓ PaddleOCR 初始化成功")
        except Exception as e:
            self.logger.error(f"PaddleOCR 初始化失败: {e}")
    
    def detect_circles(self, image):
        """检测图像中的圆形"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 使用多种参数组合检测圆形
        circle_configs = [
            # 配置1: 检测较小的圆形
            {'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 25, 'minRadius': 8, 'maxRadius': 60},
            # 配置2: 检测中等大小的圆形
            {'dp': 1, 'minDist': 30, 'param1': 100, 'param2': 35, 'minRadius': 15, 'maxRadius': 80},
            # 配置3: 检测较大的圆形
            {'dp': 2, 'minDist': 40, 'param1': 80, 'param2': 40, 'minRadius': 25, 'maxRadius': 120}
        ]
        
        all_circles = []
        for config in circle_configs:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                **config
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                all_circles.extend(circles)
        
        # 去重：合并距离很近的圆形
        if all_circles:
            all_circles = self.merge_nearby_circles(all_circles)
        
        # 提取圆形区域
        circle_regions = []
        for (x, y, r) in all_circles:
            # 扩大提取区域以包含更多上下文
            expand_factor = 1.5
            expanded_r = int(r * expand_factor)
            
            x1, y1 = max(0, x - expanded_r), max(0, y - expanded_r)
            x2, y2 = min(image.shape[1], x + expanded_r), min(image.shape[0], y + expanded_r)
            
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                circle_regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (x, y),
                    'radius': r,
                    'roi': roi
                })
        
        return circle_regions
    
    def merge_nearby_circles(self, circles, min_distance=30):
        """合并距离很近的圆形"""
        if len(circles) <= 1:
            return circles
        
        merged = []
        used = set()
        
        for i, (x1, y1, r1) in enumerate(circles):
            if i in used:
                continue
                
            # 找到所有距离很近的圆形
            group = [(x1, y1, r1)]
            used.add(i)
            
            for j, (x2, y2, r2) in enumerate(circles):
                if j in used:
                    continue
                    
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < min_distance:
                    group.append((x2, y2, r2))
                    used.add(j)
            
            # 计算组的平均位置和半径
            avg_x = int(np.mean([x for x, y, r in group]))
            avg_y = int(np.mean([y for x, y, r in group]))
            avg_r = int(np.mean([r for x, y, r in group]))
            
            merged.append((avg_x, avg_y, avg_r))
        
        return merged
    
    def create_patches(self, image, patch_size=512, overlap=0.3):
        """创建重叠的图像patches"""
        h, w = image.shape[:2]
        stride = int(patch_size * (1 - overlap))
        
        patches = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + patch_size, w)
                y2 = min(y + patch_size, h)
                
                # 确保patch有足够的大小
                if x2 - x < patch_size // 2 or y2 - y < patch_size // 2:
                    continue
                
                patch = image[y:y2, x:x2]
                patches.append({
                    'image': patch,
                    'bbox': (x, y, x2, y2),
                    'offset': (x, y)
                })
        
        return patches
    
    def ocr_with_easyocr(self, image):
        """使用EasyOCR进行识别"""
        if 'easyocr' not in self.ocr_engines:
            return []
        
        try:
            results = self.ocr_engines['easyocr'].readtext(image)
            
            detections = []
            for result in results:
                bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                # 过滤低置信度结果
                if confidence < 0.3:
                    continue
                
                # 转换bbox格式
                x1 = int(min([p[0] for p in bbox]))
                y1 = int(min([p[1] for p in bbox]))
                x2 = int(max([p[0] for p in bbox]))
                y2 = int(max([p[1] for p in bbox]))
                
                detections.append(DetectionResult(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    method='easyocr'
                ))
            
            return detections
        except Exception as e:
            self.logger.error(f"EasyOCR 识别失败: {e}")
            return []
    
    def ocr_with_paddleocr(self, image):
        """使用PaddleOCR进行识别"""
        if 'paddleocr' not in self.ocr_engines:
            return []
        
        try:
            results = self.ocr_engines['paddleocr'].ocr(image, cls=True)
            
            detections = []
            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # 过滤低置信度结果
                    if confidence < 0.3:
                        continue
                    
                    # 转换bbox格式
                    x1 = int(min([p[0] for p in bbox]))
                    y1 = int(min([p[1] for p in bbox]))
                    x2 = int(max([p[0] for p in bbox]))
                    y2 = int(max([p[1] for p in bbox]))
                    
                    detections.append(DetectionResult(
                        text=text,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        method='paddleocr'
                    ))
            
            return detections
        except Exception as e:
            self.logger.error(f"PaddleOCR 识别失败: {e}")
            return []
    
    def enhance_image_for_ocr(self, image):
        """增强图像以提高OCR识别率"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用CLAHE (对比度限制自适应直方图均衡)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 二值化
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def process_pdf_page(self, pdf_path, page_num=0):
        """处理PDF页面"""
        self.logger.info(f"处理PDF: {pdf_path}, 页面: {page_num}")
        
        # 打开PDF
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # 转换为高分辨率图像
        mat = fitz.Matrix(3.0, 3.0)  # 3倍放大提高识别精度
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        self.logger.info(f"图像尺寸: {image.shape}")
        
        # 检测圆形区域
        self.logger.info("检测圆形区域...")
        circles = self.detect_circles(image)
        self.logger.info(f"发现 {len(circles)} 个圆形区域")
        
        # 创建patches
        self.logger.info("创建图像patches...")
        patches = self.create_patches(image)
        self.logger.info(f"创建了 {len(patches)} 个patches")
        
        all_detections = []
        
        # 处理圆形区域
        for i, circle in enumerate(circles):
            self.logger.info(f"处理圆形区域 {i+1}/{len(circles)}")
            
            # 增强圆形区域图像
            enhanced_roi = self.enhance_image_for_ocr(circle['roi'])
            
            # 使用两种OCR引擎
            easyocr_results = self.ocr_with_easyocr(circle['roi'])
            paddleocr_results = self.ocr_with_paddleocr(enhanced_roi)
            
            # 合并结果
            all_circle_results = easyocr_results + paddleocr_results
            
            # 调整坐标到原图
            for det in all_circle_results:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (
                    x1 + circle['bbox'][0],
                    y1 + circle['bbox'][1],
                    x2 + circle['bbox'][0],
                    y2 + circle['bbox'][1]
                )
                det.detection_type = 'circle'
            
            all_detections.extend(all_circle_results)
        
        # 处理部分patches（限制数量以控制处理时间）
        max_patches = min(20, len(patches))
        sample_patches = patches[:max_patches]
        
        for i, patch in enumerate(sample_patches):
            self.logger.info(f"处理patch {i+1}/{len(sample_patches)}")
            
            # 增强patch图像
            enhanced_patch = self.enhance_image_for_ocr(patch['image'])
            
            # 使用EasyOCR (主要引擎)
            easyocr_results = self.ocr_with_easyocr(patch['image'])
            
            # 调整坐标到原图
            for det in easyocr_results:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (
                    x1 + patch['offset'][0],
                    y1 + patch['offset'][1],
                    x2 + patch['offset'][0],
                    y2 + patch['offset'][1]
                )
                det.detection_type = 'patch'
            
            all_detections.extend(easyocr_results)
        
        # 去重和后处理
        all_detections = self.remove_duplicate_detections(all_detections)
        
        return all_detections, image
    
    def remove_duplicate_detections(self, detections, iou_threshold=0.3):
        """去除重复的检测结果"""
        if not detections:
            return detections
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for det in detections:
            is_duplicate = False
            
            for existing in filtered:
                # 计算IoU
                iou = self.calculate_iou(det.bbox, existing.bbox)
                
                # 如果IoU高且文本相似，认为是重复
                if iou > iou_threshold and self.text_similarity(det.text, existing.text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def text_similarity(self, text1, text2):
        """计算文本相似度"""
        if text1 == text2:
            return 1.0
        
        # 简单的字符级相似度
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_results(self, image, detections, output_path):
        """可视化识别结果"""
        vis_image = image.copy()
        
        # 绘制检测结果
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            
            # 根据类型选择颜色
            if det.detection_type == 'circle':
                color = (0, 255, 0)  # 绿色 - 圆形检测
            elif det.detection_type == 'patch':
                color = (255, 0, 0)  # 蓝色 - patch检测
            else:
                color = (0, 0, 255)  # 红色 - 其他
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制文本信息
            label = f"{i+1}: {det.text} ({det.confidence:.2f})"
            
            # 计算文本位置
            label_y = y1 - 10 if y1 > 20 else y2 + 20
            
            # 绘制文本背景
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, label_y - text_height - 5), 
                         (x1 + text_width, label_y + 5), color, -1)
            
            # 绘制文本
            cv2.putText(vis_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存结果
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"可视化结果保存到: {output_path}")
        
        return vis_image

def main():
    """主函数"""
    processor = FixedOCRProcessor()
    
    # 测试PDF文件
    pdf_path = "rosenka_data/大阪府/吹田市/藤白台１/43012.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    try:
        # 处理PDF
        start_time = time.time()
        detections, image = processor.process_pdf_page(pdf_path, 0)
        process_time = time.time() - start_time
        
        print(f"\n🎯 识别结果统计:")
        print(f"总检测数量: {len(detections)}")
        print(f"处理时间: {process_time:.2f}秒")
        
        # 按类型统计
        circle_count = len([d for d in detections if d.detection_type == 'circle'])
        patch_count = len([d for d in detections if d.detection_type == 'patch'])
        
        print(f"圆形区域检测: {circle_count}")
        print(f"Patch区域检测: {patch_count}")
        
        # 按OCR引擎统计
        easyocr_count = len([d for d in detections if d.method == 'easyocr'])
        paddleocr_count = len([d for d in detections if d.method == 'paddleocr'])
        
        print(f"EasyOCR识别: {easyocr_count}")
        print(f"PaddleOCR识别: {paddleocr_count}")
        
        # 显示识别到的文本
        print(f"\n📝 识别到的文本:")
        for i, det in enumerate(detections[:30]):  # 显示前30个
            print(f"{i+1:2d}. {det.text} (置信度: {det.confidence:.2f}, 方法: {det.method}, 类型: {det.detection_type})")
        
        # 可视化结果
        output_path = "fixed_ocr_result.jpg"
        processor.visualize_results(image, detections, output_path)
        
        print(f"\n✅ 处理完成！结果已保存到: {output_path}")
        
        # 保存结果为JSON
        json_output = []
        for det in detections:
            json_output.append({
                'text': det.text,
                'bbox': det.bbox,
                'confidence': det.confidence,
                'method': det.method,
                'type': det.detection_type
            })
        
        with open('fixed_ocr_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: fixed_ocr_results.json")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 