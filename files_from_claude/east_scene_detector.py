"""
east_scene_detector.py
使用EAST进行旋转文字检测
"""

import tensorflow as tf
import cv2
import numpy as np

class EASTDetector:
    """EAST旋转文字检测器"""
    
    def __init__(self, model_path='frozen_east_text_detection.pb'):
        self.model = cv2.dnn.readNet(model_path)
        self.output_layers = ['feature_fusion/Conv_7/Sigmoid', 
                             'feature_fusion/concat_3']
        
    def detect(self, image: np.ndarray, min_confidence=0.5):
        """检测任意角度的文字"""
        # 获取原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # EAST需要尺寸是32的倍数
        new_h = (orig_h // 32) * 32
        new_w = (orig_w // 32) * 32
        rW = orig_w / new_w
        rH = orig_h / new_h
        
        # 调整图像大小
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # 构建blob
        blob = cv2.dnn.blobFromImage(
            image_resized, 1.0, (new_w, new_h),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        
        # 前向传播
        self.model.setInput(blob)
        scores, geometry = self.model.forward(self.output_layers)
        
        # 解码预测
        boxes, confidences = self._decode_predictions(scores, geometry)
        
        # NMS
        indices = cv2.dnn.NMSBoxesRotated(
            boxes, confidences, min_confidence, 0.4
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # 获取旋转框的顶点
                vertices = cv2.boxPoints(boxes[i])
                vertices = np.array(vertices, dtype=np.float32)
                
                # 缩放回原始尺寸
                vertices[:, 0] *= rW
                vertices[:, 1] *= rH
                
                results.append({
                    'polygon': vertices.astype(int).tolist(),
                    'confidence': float(confidences[i]),
                    'angle': boxes[i][4]  # 旋转角度
                })
        
        return results
    
    def _decode_predictions(self, scores, geometry):
        """解码EAST输出"""
        # 获取分数和几何信息
        num_rows, num_cols = scores.shape[2:4]
        boxes = []
        confidences = []
        
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            
            for x in range(num_cols):
                if scores_data[x] < 0.5:
                    continue
                
                # 计算偏移
                offset_x = x * 4.0
                offset_y = y * 4.0
                
                # 提取角度和尺寸
                angle = angles_data[x]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                
                # 计算中心点
                end_x = int(offset_x + cos_a * x_data1[x] + sin_a * x_data2[x])
                end_y = int(offset_y - sin_a * x_data1[x] + cos_a * x_data2[x])
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                rect = ((offset_x + end_x / 2, offset_y + end_y / 2),
                       (w, h), -angle * 180.0 / np.pi)
                
                boxes.append(rect)
                confidences.append(float(scores_data[x]))
        
        return boxes, confidences