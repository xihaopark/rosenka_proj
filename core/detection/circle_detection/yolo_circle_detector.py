"""
YOLOv8圆形检测器
专门用于检测路線価図中的圆形价格标记
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
import os

class YOLOCircleDetector:
    """YOLOv8圆形检测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 检测参数
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        self.max_detections = 100
        
        # 初始化YOLO模型
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.logger.info(f"加载自定义YOLOv8模型: {model_path}")
            else:
                # 使用预训练模型
                self.model = YOLO('yolov8m.pt')
                self.logger.info("使用预训练YOLOv8模型")
        except Exception as e:
            self.logger.error(f"YOLO模型初始化失败: {e}")
            self.model = None
        
        # 颜色过滤参数
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 50])
    
    def detect_price_circles(self, image: np.ndarray) -> List[Dict]:
        """
        检测价格圆形
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的圆形列表
        """
        try:
            if self.model is None:
                self.logger.warning("YOLO模型未初始化，使用Hough圆检测")
                return self.detect_circles_with_hough(image)
            
            # 1. 图像预处理
            processed_image = self.preprocess_for_circle_detection(image)
            
            # 2. YOLOv8检测
            results = self.model.predict(
                processed_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # 3. 后处理结果
            circles = self.postprocess_detections(results, image.shape)
            
            # 4. 进一步过滤和验证
            validated_circles = self.validate_circles(circles, image)
            
            return validated_circles
            
        except Exception as e:
            self.logger.error(f"圆形检测失败: {e}")
            # 回退到Hough圆检测
            return self.detect_circles_with_hough(image)
    
    def preprocess_for_circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        圆形检测预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 1. 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 2. 提取黑色区域
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        
        # 3. 形态学操作增强圆形特征
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. 去噪
        denoised = cv2.medianBlur(enhanced, 5)
        
        # 5. 转换回BGR格式用于YOLO
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def postprocess_detections(self, results, image_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        后处理检测结果
        
        Args:
            results: YOLO检测结果
            image_shape: 原图像尺寸
            
        Returns:
            处理后的圆形检测结果
        """
        circles = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # 计算圆心和半径
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(min(x2 - x1, y2 - y1) / 2)
                    
                    # 验证圆形特征
                    if self.is_valid_circle(center_x, center_y, radius, image_shape):
                        circles.append({
                            'center': (center_x, center_y),
                            'radius': radius,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'area': np.pi * radius * radius
                        })
        
        return circles
    
    def is_valid_circle(self, center_x: int, center_y: int, radius: int, 
                       image_shape: Tuple[int, int, int]) -> bool:
        """
        验证圆形是否有效
        
        Args:
            center_x: 圆心x坐标
            center_y: 圆心y坐标
            radius: 半径
            image_shape: 图像尺寸
            
        Returns:
            是否为有效圆形
        """
        h, w = image_shape[:2]
        
        # 检查圆心是否在图像范围内
        if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
            return False
        
        # 检查半径是否合理
        if radius < 5 or radius > 100:  # 价格圆形通常在这个范围内
            return False
        
        # 检查圆形是否完全在图像内
        if (center_x - radius < 0 or center_x + radius >= w or 
            center_y - radius < 0 or center_y + radius >= h):
            return False
        
        return True
    
    def validate_circles(self, circles: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        进一步验证圆形检测结果
        
        Args:
            circles: 初步检测的圆形列表
            image: 原始图像
            
        Returns:
            验证后的圆形列表
        """
        validated = []
        
        for circle in circles:
            if self.validate_circle_appearance(circle, image):
                validated.append(circle)
        
        # 按置信度排序
        validated.sort(key=lambda x: x['confidence'], reverse=True)
        
        return validated
    
    def validate_circle_appearance(self, circle: Dict, image: np.ndarray) -> bool:
        """
        验证圆形外观特征
        
        Args:
            circle: 圆形信息
            image: 原始图像
            
        Returns:
            是否通过外观验证
        """
        center_x, center_y = circle['center']
        radius = circle['radius']
        
        # 提取圆形区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # 提取圆形内的像素
        roi = cv2.bitwise_and(image, image, mask=mask)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 计算圆形内的平均亮度
        circle_pixels = roi_gray[mask > 0]
        if len(circle_pixels) == 0:
            return False
        
        avg_brightness = np.mean(circle_pixels)
        
        # 价格圆形通常是黑底，亮度较低
        if avg_brightness > 100:  # 阈值可调整
            return False
        
        # 检查圆形的填充程度
        filled_ratio = len(circle_pixels) / (np.pi * radius * radius)
        if filled_ratio < 0.7:  # 至少70%的区域应该被填充
            return False
        
        return True
    
    def detect_circles_with_hough(self, image: np.ndarray) -> List[Dict]:
        """
        使用霍夫圆检测作为备用方法
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的圆形列表
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 霍夫圆检测
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
        
        results = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 验证圆形
                if self.is_valid_circle(x, y, r, image.shape):
                    results.append({
                        'center': (x, y),
                        'radius': r,
                        'bbox': (x-r, y-r, x+r, y+r),
                        'confidence': 0.7,  # 霍夫圆检测的固定置信度
                        'area': np.pi * r * r,
                        'method': 'hough'
                    })
        
        return results
    
    def combine_detections(self, yolo_circles: List[Dict], 
                          hough_circles: List[Dict]) -> List[Dict]:
        """
        融合YOLO和霍夫圆检测结果
        
        Args:
            yolo_circles: YOLO检测结果
            hough_circles: 霍夫圆检测结果
            
        Returns:
            融合后的检测结果
        """
        all_circles = yolo_circles + hough_circles
        
        # 去重 - 移除重叠的圆形
        final_circles = []
        
        for circle in all_circles:
            is_duplicate = False
            
            for existing in final_circles:
                # 计算中心点距离
                dist = np.sqrt((circle['center'][0] - existing['center'][0])**2 + 
                              (circle['center'][1] - existing['center'][1])**2)
                
                # 如果距离小于半径之和的一半，认为是重复
                if dist < (circle['radius'] + existing['radius']) / 2:
                    is_duplicate = True
                    # 保留置信度更高的
                    if circle['confidence'] > existing['confidence']:
                        final_circles.remove(existing)
                        final_circles.append(circle)
                    break
            
            if not is_duplicate:
                final_circles.append(circle)
        
        return final_circles
    
    def detect_with_fallback(self, image: np.ndarray) -> List[Dict]:
        """
        带备用方法的圆形检测
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果
        """
        # 首先尝试YOLO检测
        yolo_results = self.detect_price_circles(image)
        
        # 如果YOLO检测结果不足，使用霍夫圆检测补充
        if len(yolo_results) < 5:  # 假设至少应该有5个圆形
            hough_results = self.detect_circles_with_hough(image)
            combined_results = self.combine_detections(yolo_results, hough_results)
            return combined_results
        
        return yolo_results
    
    def visualize_detections(self, image: np.ndarray, circles: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            circles: 检测到的圆形
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            confidence = circle['confidence']
            
            # 绘制圆形
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            
            # 绘制圆心
            cv2.circle(result_image, center, 2, (0, 0, 255), -1)
            
            # 添加置信度标签
            label = f"{confidence:.2f}"
            cv2.putText(result_image, label, 
                       (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return result_image 