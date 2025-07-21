import sys
import os
import cv2
import numpy as np
import torch
import fitz  # PyMuPDF
from PIL import Image
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
import math

# 设置环境路径
sys.path.insert(0, '/venv/main/lib/python3.10/site-packages')

try:
    # 尝试导入深度学习模型
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from segment_anything import sam_model_registry, SamPredictor
    import ultralytics
    from ultralytics import YOLO
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"模型导入失败: {e}")
    MODELS_AVAILABLE = False

@dataclass
class DetectionResult:
    """检测结果数据类"""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    detection_type: str  # 'circle_number', 'address', 'text'
    center_x: int
    center_y: int

class AdvancedOCRProcessor:
    """先进的OCR处理器"""
    
    def __init__(self, gpu_enabled=True):
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.device = 'cuda' if self.gpu_enabled else 'cpu'
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 模型初始化标志
        self.models_loaded = False
        self.sam_predictor = None
        self.donut_processor = None
        self.donut_model = None
        self.yolo_model = None
        
        # Patch处理参数
        self.patch_size = 640
        self.overlap_ratio = 0.2
        self.min_confidence = 0.3
        
        if MODELS_AVAILABLE:
            self._load_models()
    
    def _load_models(self):
        """加载所有需要的模型"""
        try:
            self.logger.info("开始加载模型...")
            
            # 加载SAM2模型（用于图像分割）
            if self._load_sam_model():
                self.logger.info("SAM模型加载成功")
            
            # 加载Donut模型（用于文档理解）
            if self._load_donut_model():
                self.logger.info("Donut模型加载成功")
            
            # 加载优化的YOLO模型（用于小物体检测）
            if self._load_yolo_model():
                self.logger.info("YOLO模型加载成功")
            
            self.models_loaded = True
            self.logger.info("所有模型加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.models_loaded = False
    
    def _load_sam_model(self) -> bool:
        """加载SAM模型"""
        try:
            # 这里应该加载实际的SAM2模型
            # sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            # sam.to(device=self.device)
            # self.sam_predictor = SamPredictor(sam)
            self.logger.info("SAM模型占位符加载")
            return True
        except Exception as e:
            self.logger.error(f"SAM模型加载失败: {e}")
            return False
    
    def _load_donut_model(self) -> bool:
        """加载Donut模型"""
        try:
            # 加载预训练的Donut模型
            # self.donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            # self.donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            # self.donut_model.to(self.device)
            self.logger.info("Donut模型占位符加载")
            return True
        except Exception as e:
            self.logger.error(f"Donut模型加载失败: {e}")
            return False
    
    def _load_yolo_model(self) -> bool:
        """加载优化的YOLO模型"""
        try:
            # 加载针对小物体优化的YOLO模型
            # self.yolo_model = YOLO('yolov8n.pt')  # 或者使用自定义训练的模型
            self.logger.info("YOLO模型占位符加载")
            return True
        except Exception as e:
            self.logger.error(f"YOLO模型加载失败: {e}")
            return False
    
    def extract_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        将图像切分成重叠的patches
        
        Args:
            image: 输入图像
            
        Returns:
            patches和其在原图中的位置
        """
        patches = []
        h, w = image.shape[:2]
        
        step_size = int(self.patch_size * (1 - self.overlap_ratio))
        
        for y in range(0, h - self.patch_size + 1, step_size):
            for x in range(0, w - self.patch_size + 1, step_size):
                # 确保不超出边界
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                
                patch = image[y:y_end, x:x_end]
                
                # 如果patch太小，跳过
                if patch.shape[0] < self.patch_size // 2 or patch.shape[1] < self.patch_size // 2:
                    continue
                
                patches.append((patch, (x, y)))
        
        return patches
    
    def detect_circle_numbers(self, image: np.ndarray) -> List[DetectionResult]:
        """
        检测带圆圈的数字
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 使用计算机视觉方法检测圆形
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 使用HoughCircles检测圆形
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 提取圆形区域
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # 提取圆形内的数字
                roi = cv2.bitwise_and(gray, gray, mask=mask)
                
                # 使用简单的OCR识别数字
                # 这里应该调用更精确的数字识别模型
                detected_text = self._recognize_number_in_circle(roi, x, y, r)
                
                if detected_text:
                    result = DetectionResult(
                        text=detected_text,
                        bbox=(x-r, y-r, x+r, y+r),
                        confidence=0.8,  # 临时置信度
                        detection_type='circle_number',
                        center_x=x,
                        center_y=y
                    )
                    results.append(result)
        
        return results
    
    def _recognize_number_in_circle(self, roi: np.ndarray, x: int, y: int, r: int) -> Optional[str]:
        """
        识别圆形内的数字
        
        Args:
            roi: 圆形区域图像
            x, y: 圆心坐标
            r: 半径
            
        Returns:
            识别的数字文本
        """
        # 提取圆形内部区域
        center_roi = roi[max(0, y-r//2):min(roi.shape[0], y+r//2), 
                         max(0, x-r//2):min(roi.shape[1], x+r//2)]
        
        if center_roi.size == 0:
            return None
        
        # 二值化处理
        _, binary = cv2.threshold(center_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 简单的数字识别（这里应该使用更精确的模型）
        # 计算白色像素比例来判断可能的数字
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        if white_pixels / total_pixels > 0.1:  # 如果有足够的前景像素
            # 这里应该调用专门的数字识别模型
            # 暂时返回占位符
            return "1"  # 临时返回
        
        return None
    
    def process_with_donut(self, image: np.ndarray) -> List[DetectionResult]:
        """
        使用Donut模型处理文档图像
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        results = []
        
        if not self.models_loaded or not self.donut_model:
            self.logger.warning("Donut模型未加载，跳过处理")
            return results
        
        try:
            # 将numpy数组转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 使用Donut模型进行文档理解
            # inputs = self.donut_processor(pil_image, return_tensors="pt")
            # outputs = self.donut_model.generate(**inputs)
            # text = self.donut_processor.decode(outputs[0], skip_special_tokens=True)
            
            # 临时占位符
            self.logger.info("Donut模型处理占位符")
            
        except Exception as e:
            self.logger.error(f"Donut处理失败: {e}")
        
        return results
    
    def process_with_yolo(self, image: np.ndarray) -> List[DetectionResult]:
        """
        使用优化的YOLO模型检测小物体
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        results = []
        
        if not self.models_loaded or not self.yolo_model:
            self.logger.warning("YOLO模型未加载，跳过处理")
            return results
        
        try:
            # 使用YOLO进行检测
            # detections = self.yolo_model(image)
            
            # 临时占位符
            self.logger.info("YOLO模型处理占位符")
            
        except Exception as e:
            self.logger.error(f"YOLO处理失败: {e}")
        
        return results
    
    def segment_with_sam(self, image: np.ndarray) -> List[np.ndarray]:
        """
        使用SAM进行图像分割
        
        Args:
            image: 输入图像
            
        Returns:
            分割后的图像区域列表
        """
        segments = []
        
        if not self.models_loaded or not self.sam_predictor:
            self.logger.warning("SAM模型未加载，跳过处理")
            return [image]  # 返回原图
        
        try:
            # 使用SAM进行分割
            # self.sam_predictor.set_image(image)
            # masks, _, _ = self.sam_predictor.predict()
            
            # 临时占位符
            self.logger.info("SAM模型处理占位符")
            segments = [image]  # 临时返回原图
            
        except Exception as e:
            self.logger.error(f"SAM处理失败: {e}")
            segments = [image]
        
        return segments
    
    def process_pdf_page_advanced(self, pdf_path: str, page_num: int) -> List[DetectionResult]:
        """
        使用先进pipeline处理PDF页面
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码
            
        Returns:
            检测结果列表
        """
        all_results = []
        
        try:
            # 打开PDF并提取页面
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 提高分辨率
            mat = fitz.Matrix(3.0, 3.0)  # 3倍放大
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            doc.close()
            
            if image is None:
                self.logger.error(f"无法读取页面 {page_num}")
                return all_results
            
            self.logger.info(f"处理页面 {page_num}，图像尺寸: {image.shape}")
            
            # 1. 使用SAM进行图像分割
            segments = self.segment_with_sam(image)
            
            # 2. 对每个分割区域进行patch处理
            for segment in segments:
                patches = self.extract_patches(segment)
                
                for patch, (offset_x, offset_y) in patches:
                    # 3. 检测带圆圈的数字
                    circle_results = self.detect_circle_numbers(patch)
                    
                    # 4. 使用Donut进行文档理解
                    donut_results = self.process_with_donut(patch)
                    
                    # 5. 使用YOLO进行小物体检测
                    yolo_results = self.process_with_yolo(patch)
                    
                    # 调整坐标到原图坐标系
                    for result in circle_results + donut_results + yolo_results:
                        result.bbox = (
                            result.bbox[0] + offset_x,
                            result.bbox[1] + offset_y,
                            result.bbox[2] + offset_x,
                            result.bbox[3] + offset_y
                        )
                        result.center_x += offset_x
                        result.center_y += offset_y
                    
                    all_results.extend(circle_results + donut_results + yolo_results)
            
            # 6. 后处理：去重和置信度过滤
            all_results = self._post_process_results(all_results)
            
        except Exception as e:
            self.logger.error(f"处理PDF页面失败: {e}")
        
        return all_results
    
    def _post_process_results(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """
        后处理检测结果：去重、过滤低置信度结果
        
        Args:
            results: 原始检测结果
            
        Returns:
            处理后的结果
        """
        # 过滤低置信度结果
        filtered_results = [r for r in results if r.confidence >= self.min_confidence]
        
        # 简单的去重：如果两个检测框重叠度很高，保留置信度更高的
        final_results = []
        
        for result in filtered_results:
            is_duplicate = False
            
            for existing in final_results:
                if self._calculate_iou(result.bbox, existing.bbox) > 0.5:
                    if result.confidence > existing.confidence:
                        # 替换现有结果
                        final_results.remove(existing)
                        final_results.append(result)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_results.append(result)
        
        return final_results
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1, bbox2: 边界框 (x1, y1, x2, y2)
            
        Returns:
            IoU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def save_results_to_database(self, results: List[DetectionResult], pdf_path: str, page_num: int, db_path: str):
        """
        保存检测结果到数据库
        
        Args:
            results: 检测结果列表
            pdf_path: PDF文件路径
            page_num: 页码
            db_path: 数据库路径
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建表（如果不存在）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS advanced_detections (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    center_x INTEGER,
                    center_y INTEGER,
                    confidence REAL,
                    detection_type TEXT,
                    pdf_path TEXT,
                    page_num INTEGER,
                    created_at TEXT
                )
            ''')
            
            # 插入检测结果
            for result in results:
                detection_id = f"{pdf_path}_{page_num}_{result.center_x}_{result.center_y}"
                
                cursor.execute('''
                    INSERT OR REPLACE INTO advanced_detections
                    (id, text, bbox_x1, bbox_y1, bbox_x2, bbox_y2, center_x, center_y,
                     confidence, detection_type, pdf_path, page_num, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_id, result.text,
                    result.bbox[0], result.bbox[1], result.bbox[2], result.bbox[3],
                    result.center_x, result.center_y, result.confidence,
                    result.detection_type, pdf_path, page_num,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"已保存 {len(results)} 个检测结果到数据库")
            
        except Exception as e:
            self.logger.error(f"保存结果到数据库失败: {e}")

# 使用示例
if __name__ == "__main__":
    processor = AdvancedOCRProcessor(gpu_enabled=True)
    
    # 处理示例PDF
    pdf_path = "rosenka_data/大阪府/吹田市/藤白台１/43012.pdf"
    
    if os.path.exists(pdf_path):
        results = processor.process_pdf_page_advanced(pdf_path, 0)
        print(f"检测到 {len(results)} 个对象")
        
        for result in results:
            print(f"类型: {result.detection_type}, 文本: {result.text}, 置信度: {result.confidence:.2f}")
    else:
        print(f"PDF文件不存在: {pdf_path}") 