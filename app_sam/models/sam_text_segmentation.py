"""
sam_text_segmentation.py
使用 SAM 进行路線価図文本区域分割
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SAMTextSegmenter:
    """基于 SAM 的文本区域分割器"""
    
    def __init__(self, model_type='vit_h', checkpoint_path='sam_vit_h_4b8939.pth'):
        """
        初始化 SAM 模型
        
        Args:
            model_type: 模型类型 ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: 模型权重路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载 SAM 模型
        logger.info(f"加载 SAM 模型: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        # 创建自动mask生成器
        self.mask_generator = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=64,  # 增加采样密度（原32）
        pred_iou_thresh=0.86,  # 稍微降低阈值（原0.88）
        stability_score_thresh=0.90,  # 稍微降低（原0.92）
        crop_n_layers=2,  # 增加层数（原1）
        crop_n_points_downscale_factor=1,  # 减少下采样（原2）
        min_mask_region_area=30,  # 降低最小面积（原100）
    )
        
        # 创建交互式预测器
        self.predictor = SamPredictor(self.sam)
        
    def segment_page(self, image: np.ndarray) -> List[Dict]:
        """
        对整页进行自动分割
        
        Args:
            image: 输入图像 (H, W, 3)
            
        Returns:
            分割结果列表，每个包含:
            - segmentation: 分割掩码
            - bbox: 边界框 (x, y, w, h)
            - area: 区域面积
            - predicted_iou: 预测的IOU
            - stability_score: 稳定性分数
        """
        logger.info("开始自动分割...")
        
        # 生成所有masks
        masks = self.mask_generator.generate(image)
        
        # 过滤出可能的文本区域
        text_regions = self._filter_text_regions(masks, image)
        
        logger.info(f"检测到 {len(text_regions)} 个潜在文本区域")
        
        return text_regions
    
    def _filter_text_regions(self, masks: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        过滤出文本区域
        
        基于以下特征:
        1. 宽高比
        2. 区域大小
        3. 边缘特征
        4. 颜色一致性
        """
        text_regions = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h
            
            # 计算区域特征
            features = self._compute_region_features(mask, bbox, image)
            
            # 判断是否为文本区域
            if self._is_text_region(features):
                text_regions.append({
                    **mask_data,
                    'features': features,
                    'region_type': 'text'
                })
        
        return text_regions
    
    def segment_page_hybrid(self, image: np.ndarray) -> List[Dict]:
        """混合方法：SAM + 传统CV"""
        # 1. 先用SAM
        sam_regions = self.segment_page(image)
        
        # 2. 如果SAM结果太少，使用传统方法
        if len(sam_regions) < 10:  # 阈值可调
            from app_sam.models.rosenka_specific_detector import RosenkaSpecificDetector
            detector = RosenkaSpecificDetector()
            
            # 获取候选区域
            candidates = detector.detect_text_candidates(image)
            
            # 转换为SAM格式
            for cand in candidates:
                x, y, w, h = cand['bbox']
                
                # 创建mask
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[y:y+h, x:x+w] = True
                
                sam_regions.append({
                    'segmentation': mask,
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'predicted_iou': 0.9,
                    'stability_score': 0.9,
                    'region_type': 'traditional_' + cand['type']
                })
        
        return sam_regions
    
    def _compute_region_features(self, mask: np.ndarray, bbox: Tuple, image: np.ndarray) -> Dict:
        """计算区域特征"""
        # 强制转换为int，防止切片报错
        x, y, w, h = [int(v) for v in bbox]
        
        # 提取区域
        region = image[y:y+h, x:x+w]
        region_mask = mask[y:y+h, x:x+w]
        
        # 计算特征
        features = {
            'aspect_ratio': w / h if h > 0 else 0,
            'area': mask.sum(),
            'relative_area': mask.sum() / (image.shape[0] * image.shape[1]),
            'density': mask.sum() / (w * h) if w * h > 0 else 0,
        }
        
        # 计算边缘密度（文本区域通常有更多边缘）
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_region, 50, 150)
        features['edge_density'] = edges.sum() / (w * h) if w * h > 0 else 0
        
        # 计算颜色统计
        masked_pixels = region[region_mask > 0]
        if len(masked_pixels) > 0:
            features['color_std'] = np.std(masked_pixels, axis=0).mean()
        else:
            features['color_std'] = 0
        
        return features
    
    def _is_text_region(self, features: Dict) -> bool:
        """判断是否为文本区域"""
        # 基于路線価図的特点调整阈值
        aspect_ratio = features['aspect_ratio']
        edge_density = features['edge_density']
        area = features['area']
        
        # 文本区域的典型特征
        is_text = (
            # 宽高比在合理范围内
            (0.1 < aspect_ratio < 20) and
            # 有一定的边缘密度
            (edge_density > 500) and
            # 面积不能太小
            (area > 200) and
            # 面积不能太大（排除背景）
            (features['relative_area'] < 0.3)
        )
        
        return is_text
    
    def segment_interactive(self, image: np.ndarray, points: List[Tuple[int, int]], 
                          labels: List[int]) -> Dict:
        """
        交互式分割 - 通过点击指定区域
        
        Args:
            image: 输入图像
            points: 点击的坐标列表 [(x1, y1), (x2, y2), ...]
            labels: 点的标签列表 [1, 1, 0, ...] (1=前景, 0=背景)
            
        Returns:
            分割结果
        """
        self.predictor.set_image(image)
        
        # 转换为numpy数组
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # 选择最佳mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            'segmentation': best_mask,
            'score': scores[best_idx],
            'all_masks': masks
        }
    
    def refine_text_regions(self, image: np.ndarray, initial_regions: List[Dict]) -> List[Dict]:
        """
        细化文本区域 - 二次处理
        
        1. 合并相邻的文本区域
        2. 分割过大的区域
        3. 调整边界
        """
        refined_regions = []
        
        # 合并相邻区域
        merged_regions = self._merge_adjacent_regions(initial_regions)
        
        for region in merged_regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # 如果区域过大，尝试细分
            if w * h > 50000:  # 阈值可调
                sub_regions = self._split_large_region(image, region)
                refined_regions.extend(sub_regions)
            else:
                # 调整边界
                adjusted_region = self._adjust_boundaries(image, region)
                refined_regions.append(adjusted_region)
        
        return refined_regions
    
    def _merge_adjacent_regions(self, regions: List[Dict], distance_threshold: int = 20) -> List[Dict]:
        """合并相邻的文本区域"""
        if not regions:
            return []
        
        # 按位置排序
        sorted_regions = sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        merged = []
        current_group = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            region = sorted_regions[i]
            
            # 检查是否与当前组相邻
            if self._are_adjacent(current_group[-1], region, distance_threshold):
                current_group.append(region)
            else:
                # 合并当前组并开始新组
                if current_group:
                    merged.append(self._merge_region_group(current_group))
                current_group = [region]
        
        # 处理最后一组
        if current_group:
            merged.append(self._merge_region_group(current_group))
        
        return merged
    
    def _are_adjacent(self, region1: Dict, region2: Dict, threshold: int) -> bool:
        """判断两个区域是否相邻"""
        x1, y1, w1, h1 = region1['bbox']
        x2, y2, w2, h2 = region2['bbox']
        
        # 计算最短距离
        x_distance = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_distance = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        
        return x_distance < threshold and y_distance < threshold
    
    def _merge_region_group(self, regions: List[Dict]) -> Dict:
        """合并一组区域"""
        # 计算合并后的边界框
        x_min = min(r['bbox'][0] for r in regions)
        y_min = min(r['bbox'][1] for r in regions)
        x_max = max(r['bbox'][0] + r['bbox'][2] for r in regions)
        y_max = max(r['bbox'][1] + r['bbox'][3] for r in regions)
        
        # 合并mask
        h_total = y_max - y_min
        w_total = x_max - x_min
        merged_mask = np.zeros((h_total, w_total), dtype=bool)
        
        for region in regions:
            x, y, w, h = region['bbox']
            x_rel, y_rel = x - x_min, y - y_min
            
            # 提取原始mask的相应部分
            original_mask = region['segmentation']
            mask_part = original_mask[y:y+h, x:x+w]
            
            # 放置到合并mask中
            merged_mask[y_rel:y_rel+h, x_rel:x_rel+w] |= mask_part
        
        return {
            'bbox': (x_min, y_min, w_total, h_total),
            'segmentation': merged_mask,
            'merged_from': len(regions),
            'region_type': 'text_merged'
        }