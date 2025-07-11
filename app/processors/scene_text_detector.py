"""
scene_text_detector.py
使用PaddleOCR的检测模块 - 专为场景文字优化
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np

class PaddleSceneTextDetector:
    """PaddleOCR场景文字检测器"""
    
    def __init__(self):
        # 只使用检测功能，关闭识别以提高速度
        self.ocr = PaddleOCR(
            use_angle_cls=True,      # 文字方向分类
            lang='japan',            # 日语模型
            det_algorithm='DB++',    # 最新的DB++算法
            det_db_thresh=0.3,       # 检测阈值
            det_db_box_thresh=0.5,   # 框阈值
            det_db_unclip_ratio=1.6, # 扩展检测框
            max_batch_size=10,
            use_gpu=True,
            rec=False,               # 只检测不识别
            cls=True,
            show_log=False
        )
        
    def detect_text(self, image: np.ndarray):
        """检测所有文字区域"""
        # PaddleOCR检测
        dt_boxes = self.ocr.ocr(image, rec=False, cls=True)
        
        if not dt_boxes or not dt_boxes[0]:
            return []
        
        results = []
        for box in dt_boxes[0]:
            # box格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(box, dtype=np.int32)
            
            # 转换为矩形框
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])
            
            # 计算角度
            angle = self._calculate_angle(points)
            
            results.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'polygon': points.tolist(),
                'angle': angle,
                'confidence': 0.95  # PaddleOCR不返回置信度
            })
        
        return results
    
    def detect_by_text_type(self, image: np.ndarray):
        """按文字类型分类检测"""
        all_boxes = self.detect_text(image)
        
        categorized = {
            'numbers': [],      # 纯数字（路线价）
            'japanese': [],     # 日文地名
            'mixed': [],        # 混合文字
            'english': []       # 英文
        }
        
        for box in all_boxes:
            # 裁剪文字区域
            x1, y1, x2, y2 = box['bbox']
            text_region = image[y1:y2, x1:x2]
            
            # 简单分类
            text_type = self._classify_text_region(text_region, box)
            categorized[text_type].append(box)
        
        return categorized
    
    def _calculate_angle(self, points):
        """计算文字角度"""
        # 使用上边缘计算角度
        edge = points[1] - points[0]
        angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
        return angle
    
    def _classify_text_region(self, region, box):
        """分类文字区域类型"""
        # 基于区域大小和形状的简单分类
        bbox = box['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratio = w / h if h > 0 else 1
        
        # 路线价特征：小且接近正方形
        if 20 < w < 80 and 20 < h < 80 and 0.7 < aspect_ratio < 1.3:
            return 'numbers'
        # 长条形可能是地名
        elif aspect_ratio > 2:
            return 'japanese'
        else:
            return 'mixed' 