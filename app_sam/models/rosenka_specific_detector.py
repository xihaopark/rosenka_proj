"""
路線価図专用检测器
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple

class RosenkaSpecificDetector:
    """针对路線価図优化的检测器"""
    
    def detect_text_candidates(self, image: np.ndarray) -> List[Dict]:
        """检测可能的文本候选区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. 使用MSER检测文字区域
        mser = cv2.MSER_create(
            5,
            30,
            1000,
            0.25,
            0.2
        )
        
        regions, _ = mser.detectRegions(gray)
        candidates = []
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # 路線価図特征过滤
            aspect_ratio = w / h if h > 0 else 0
            
            # 数字通常是方形的
            if 0.3 < aspect_ratio < 3.0 and 20 < w < 200 and 15 < h < 100:
                candidates.append({
                    'bbox': (x, y, w, h),
                    'type': 'number' if aspect_ratio < 1.5 else 'text'
                })
        
        # 2. 使用形态学操作找文字行
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 水平膨胀连接文字
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
        
        contours, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and 10 < h < 50:  # 文字行特征
                candidates.append({
                    'bbox': (x, y, w, h),
                    'type': 'text_line'
                })
        
        return self._merge_overlapping(candidates)
    
    def _merge_overlapping(self, candidates: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """合并重叠的候选区域"""
        if not candidates:
            return []
        
        # 按面积排序
        candidates = sorted(candidates, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        merged = []
        used = set()
        
        for i, cand1 in enumerate(candidates):
            if i in used:
                continue
            
            x1, y1, w1, h1 = cand1['bbox']
            group = [cand1]
            
            for j, cand2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = cand2['bbox']
                
                # 计算IoU
                xx1 = max(x1, x2)
                yy1 = max(y1, y2)
                xx2 = min(x1 + w1, x2 + w2)
                yy2 = min(y1 + h1, y2 + h2)
                
                if xx2 > xx1 and yy2 > yy1:
                    intersection = (xx2 - xx1) * (yy2 - yy1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union = area1 + area2 - intersection
                    iou = intersection / union
                    
                    if iou > iou_thresh:
                        group.append(cand2)
                        used.add(j)
            
            # 合并组内区域
            if len(group) > 1:
                xs = [c['bbox'][0] for c in group]
                ys = [c['bbox'][1] for c in group]
                x2s = [c['bbox'][0] + c['bbox'][2] for c in group]
                y2s = [c['bbox'][1] + c['bbox'][3] for c in group]
                
                merged.append({
                    'bbox': (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)),
                    'type': 'merged',
                    'count': len(group)
                })
            else:
                merged.append(cand1)
        
        return merged