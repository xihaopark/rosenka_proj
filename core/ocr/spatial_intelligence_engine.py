#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Intelligence Engine for Stage 5 Rosenka OCR System
空间智能分析引擎 - 基于地图空间信息的文字分类和验证

主要功能:
1. 道路线条检测和提取 - 识别地图中的道路网络
2. 封闭区域（街区）边界识别 - 找到建筑街区边界
3. 文本空间位置分类 - 区分街区番号vs路線価
4. 邻近关系分析 - 检测重复和相关联的地址
5. 地理一致性验证 - 验证地址的合理性
6. 空间规则引擎 - 应用路線価图的空间规则
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import math

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialIntelligenceEngine:
    """Stage 5 空间智能分析引擎"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化空间智能分析引擎
        
        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        self.debug_dir = Path("debug_spatial") if debug_mode else None
        
        # 道路检测参数
        self.road_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 100,
            'min_line_length': 50,
            'max_line_gap': 10,
            'line_merge_threshold': 20,  # 平行线合并阈值
            'angle_tolerance': 5  # 角度容差（度）
        }
        
        # 区域分析参数
        self.region_params = {
            'contour_min_area': 1000,
            'contour_max_area': 50000,
            'proximity_threshold': 30,  # 文字与线条的邻近阈值
            'block_center_threshold': 0.7  # 街区中心判定阈值
        }
        
        # 空间分类规则
        self.spatial_rules = {
            'route_price_max_distance_to_road': 50,  # 路線価到道路的最大距离
            'block_number_min_distance_to_road': 20,  # 街区番号到道路的最小距离
            'duplicate_detection_radius': 100,  # 重复检测半径
            'cluster_max_distance': 150  # 聚类最大距离
        }
        
        if self.debug_mode and self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"调试模式启用，空间分析结果保存至: {self.debug_dir}")
    
    def analyze_spatial_context(self, image: np.ndarray, 
                               detections: List[Dict],
                               image_name: str = "image") -> Dict:
        """
        分析图像的空间上下文
        
        Args:
            image: 输入图像
            detections: OCR检测结果
            image_name: 图像名称
            
        Returns:
            空间分析结果，包含道路网络、区域信息等
        """
        logger.info(f"开始空间上下文分析: {image_name}")
        
        # 1. 道路网络检测
        road_network = self.detect_road_network(image)
        
        # 2. 封闭区域检测
        regions = self.detect_closed_regions(image, road_network)
        
        # 3. 文本空间分类
        classified_detections = self.classify_text_spatial_position(
            detections, road_network, regions
        )
        
        # 4. 邻近关系分析
        proximity_analysis = self.analyze_proximity_relationships(
            classified_detections
        )
        
        # 5. 重复检测和过滤
        filtered_detections = self.detect_and_filter_duplicates(
            classified_detections
        )
        
        # 6. 地理一致性验证
        validated_detections = self.validate_geographic_consistency(
            filtered_detections, road_network, regions
        )
        
        # 构建结果
        spatial_result = {
            'road_network': road_network,
            'regions': regions,
            'classified_detections': validated_detections,
            'proximity_analysis': proximity_analysis,
            'image_name': image_name,
            'total_detections': len(validated_detections),
            'statistics': self._calculate_statistics(validated_detections)
        }
        
        # 保存调试信息
        if self.debug_mode:
            self._save_spatial_debug(image, spatial_result, image_name)
        
        logger.info(f"空间分析完成: {len(validated_detections)}个有效检测")
        return spatial_result
    
    def detect_road_network(self, image: np.ndarray) -> Dict:
        """
        检测道路网络
        
        Args:
            image: 输入图像
            
        Returns:
            道路网络信息
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Canny边缘检测
        edges = cv2.Canny(
            gray, 
            self.road_detection_params['canny_low'], 
            self.road_detection_params['canny_high']
        )
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.road_detection_params['hough_threshold'],
            minLineLength=self.road_detection_params['min_line_length'],
            maxLineGap=self.road_detection_params['max_line_gap']
        )
        
        if lines is None:
            return {'raw_lines': [], 'grouped_lines': [], 'intersections': []}
        
        # 线条分组和合并
        grouped_lines = self._group_parallel_lines(lines)
        
        # 找到交叉点
        intersections = self._find_intersections(grouped_lines)
        
        road_network = {
            'raw_lines': lines.tolist() if lines is not None else [],
            'grouped_lines': grouped_lines,
            'intersections': intersections,
            'line_count': len(grouped_lines)
        }
        
        logger.debug(f"检测到 {len(grouped_lines)} 条道路线段，{len(intersections)} 个交叉点")
        return road_network
    
    def _group_parallel_lines(self, lines: np.ndarray) -> List[Dict]:
        """将平行的线条分组合并"""
        if lines is None or len(lines) == 0:
            return []
        
        grouped = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            x1, y1, x2, y2 = line1[0]
            
            # 计算线条的角度和长度
            angle1 = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            length1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 寻找平行线条
            parallel_group = [line1[0]]
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                angle2 = math.atan2(y4 - y3, x4 - x3) * 180 / math.pi
                
                # 检查角度差异
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                # 如果角度相近且距离合适，则合并
                if angle_diff < self.road_detection_params['angle_tolerance']:
                    # 检查距离
                    dist = self._line_to_line_distance(line1[0], line2[0])
                    if dist < self.road_detection_params['line_merge_threshold']:
                        parallel_group.append(line2[0])
                        used.add(j)
            
            # 合并平行线组
            merged_line = self._merge_parallel_lines(parallel_group)
            grouped.append({
                'line': merged_line,
                'original_count': len(parallel_group),
                'angle': angle1,
                'length': self._line_length(merged_line)
            })
            
            used.add(i)
        
        return grouped
    
    def _line_to_line_distance(self, line1: List, line2: List) -> float:
        """计算两条线段之间的距离"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 计算线段中点
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        
        # 返回中点距离
        return math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
    
    def _merge_parallel_lines(self, lines: List[List]) -> List:
        """合并一组平行线"""
        if len(lines) == 1:
            return lines[0]
        
        # 找到最远的两个端点
        all_points = []
        for line in lines:
            all_points.extend([(line[0], line[1]), (line[2], line[3])])
        
        # 计算最小外接线段
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        # 使用第一条线的方向
        base_line = lines[0]
        dx = base_line[2] - base_line[0]
        dy = base_line[3] - base_line[1]
        
        # 找到在该方向上的最远点
        if abs(dx) > abs(dy):  # 主要是水平线
            min_x_idx = xs.index(min(xs))
            max_x_idx = xs.index(max(xs))
            return [all_points[min_x_idx][0], all_points[min_x_idx][1],
                   all_points[max_x_idx][0], all_points[max_x_idx][1]]
        else:  # 主要是垂直线
            min_y_idx = ys.index(min(ys))
            max_y_idx = ys.index(max(ys))
            return [all_points[min_y_idx][0], all_points[min_y_idx][1],
                   all_points[max_y_idx][0], all_points[max_y_idx][1]]
    
    def _line_length(self, line: List) -> float:
        """计算线段长度"""
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _find_intersections(self, grouped_lines: List[Dict]) -> List[Tuple]:
        """找到线段交叉点"""
        intersections = []
        
        for i, line1_info in enumerate(grouped_lines):
            for j, line2_info in enumerate(grouped_lines[i+1:], i+1):
                line1 = line1_info['line']
                line2 = line2_info['line']
                
                intersection = self._line_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)
        
        return intersections
    
    def _line_intersection(self, line1: List, line2: List) -> Optional[Tuple]:
        """计算两条线段的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 计算交点
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行线
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # 检查交点是否在线段上
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (int(ix), int(iy))
        
        return None
    
    def detect_closed_regions(self, image: np.ndarray, 
                             road_network: Dict) -> List[Dict]:
        """
        检测封闭区域（街区）
        
        Args:
            image: 输入图像
            road_network: 道路网络信息
            
        Returns:
            封闭区域列表
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            # 计算区域属性
            area = cv2.contourArea(contour)
            
            # 过滤太小或太大的区域
            if (area < self.region_params['contour_min_area'] or 
                area > self.region_params['contour_max_area']):
                continue
            
            # 计算边界框和中心点
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # 计算几何属性
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            regions.append({
                'id': i,
                'contour': contour.tolist(),
                'area': area,
                'center': center,
                'bbox': (x, y, w, h),
                'perimeter': perimeter,
                'circularity': circularity
            })
        
        logger.debug(f"检测到 {len(regions)} 个封闭区域")
        return regions
    
    def classify_text_spatial_position(self, detections: List[Dict],
                                     road_network: Dict,
                                     regions: List[Dict]) -> List[Dict]:
        """
        基于空间位置对文本进行分类
        
        Args:
            detections: OCR检测结果
            road_network: 道路网络
            regions: 封闭区域
            
        Returns:
            空间分类后的检测结果
        """
        classified = []
        
        for detection in detections:
            # 获取文本中心点
            bbox = detection['bbox']
            center = self._bbox_center(bbox)
            
            # 计算到道路的距离
            road_distance = self._distance_to_roads(center, road_network['grouped_lines'])
            
            # 检查是否在封闭区域内
            containing_region = self._find_containing_region(center, regions)
            
            # 基于空间规则进行分类
            spatial_type = self._classify_by_spatial_rules(
                detection, center, road_distance, containing_region
            )
            
            # 添加空间信息
            enhanced_detection = detection.copy()
            enhanced_detection.update({
                'center': center,
                'road_distance': road_distance,
                'containing_region': containing_region,
                'spatial_type': spatial_type,
                'spatial_confidence': self._calculate_spatial_confidence(
                    detection, spatial_type, road_distance, containing_region
                )
            })
            
            classified.append(enhanced_detection)
        
        return classified
    
    def _bbox_center(self, bbox: List) -> Tuple[float, float]:
        """计算边界框中心点"""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _distance_to_roads(self, point: Tuple[float, float], 
                          grouped_lines: List[Dict]) -> float:
        """计算点到道路的最小距离"""
        if not grouped_lines:
            return float('inf')
        
        min_distance = float('inf')
        
        for line_info in grouped_lines:
            line = line_info['line']
            distance = self._point_to_line_distance(point, line)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line: List) -> float:
        """计算点到线段的距离"""
        px, py = point
        x1, y1, x2, y2 = line
        
        # 线段长度
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        # 计算投影参数
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length**2))
        
        # 投影点
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 返回距离
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def _find_containing_region(self, point: Tuple[float, float], 
                               regions: List[Dict]) -> Optional[Dict]:
        """找到包含该点的区域"""
        px, py = point
        
        for region in regions:
            contour = np.array(region['contour'])
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                return region
        
        return None
    
    def _classify_by_spatial_rules(self, detection: Dict, 
                                  center: Tuple[float, float],
                                  road_distance: float,
                                  containing_region: Optional[Dict]) -> str:
        """基于空间规则进行分类"""
        text = detection['text']
        original_type = detection.get('type', 'unknown')
        
        # 规则1: 路線価通常靠近道路
        if (original_type == 'route_price' and 
            road_distance <= self.spatial_rules['route_price_max_distance_to_road']):
            return 'route_price_confirmed'
        
        # 规则2: 街区番号通常在封闭区域内且远离道路
        if (original_type == 'block_number' and 
            containing_region is not None and
            road_distance >= self.spatial_rules['block_number_min_distance_to_road']):
            return 'block_number_confirmed'
        
        # 规则3: 基于位置重新分类
        if road_distance < 30:
            if text.isdigit() or any(c.isalpha() for c in text):
                return 'likely_route_price'
        elif containing_region is not None and road_distance > 50:
            if text.isdigit():
                return 'likely_block_number'
        
        # 规则4: 特殊情况
        if original_type in ['pure_number', 'unknown']:
            if containing_region is not None:
                return 'possible_block_number'
            elif road_distance < 50:
                return 'possible_route_price'
        
        return f"spatial_{original_type}"
    
    def _calculate_spatial_confidence(self, detection: Dict, 
                                    spatial_type: str,
                                    road_distance: float,
                                    containing_region: Optional[Dict]) -> float:
        """计算空间分类的置信度"""
        base_confidence = detection.get('final_score', detection.get('confidence', 0.5))
        
        # 根据空间分类调整置信度
        if 'confirmed' in spatial_type:
            spatial_bonus = 0.3
        elif 'likely' in spatial_type:
            spatial_bonus = 0.2
        elif 'possible' in spatial_type:
            spatial_bonus = 0.1
        else:
            spatial_bonus = 0.0
        
        # 根据位置特征调整
        if 'route_price' in spatial_type:
            # 路線价：越靠近道路越好
            distance_factor = max(0, 1 - road_distance / 100)
            spatial_bonus += distance_factor * 0.1
        elif 'block_number' in spatial_type:
            # 街区番号：在区域内且远离道路更好
            if containing_region is not None:
                spatial_bonus += 0.15
            if road_distance > 30:
                spatial_bonus += 0.1
        
        return min(1.0, base_confidence + spatial_bonus)
    
    def analyze_proximity_relationships(self, detections: List[Dict]) -> Dict:
        """分析邻近关系"""
        if not detections:
            return {'clusters': [], 'relationships': []}
        
        # 提取中心点
        centers = np.array([det['center'] for det in detections])
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(
            eps=self.spatial_rules['cluster_max_distance'], 
            min_samples=2
        ).fit(centers)
        
        # 分析聚类结果
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # 分析关系
        relationships = []
        for cluster_id, indices in clusters.items():
            if cluster_id == -1:  # 噪声点
                continue
            
            cluster_detections = [detections[i] for i in indices]
            cluster_analysis = self._analyze_cluster(cluster_detections)
            relationships.append({
                'cluster_id': cluster_id,
                'detection_indices': indices,
                'analysis': cluster_analysis
            })
        
        return {
            'clusters': clusters,
            'relationships': relationships,
            'noise_points': clusters.get(-1, [])
        }
    
    def _analyze_cluster(self, cluster_detections: List[Dict]) -> Dict:
        """分析一个聚类的特征"""
        types = [det.get('spatial_type', 'unknown') for det in cluster_detections]
        texts = [det['text'] for det in cluster_detections]
        
        # 统计类型分布
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # 检测可能的重复
        text_counts = {}
        for text in texts:
            text_counts[text] = text_counts.get(text, 0) + 1
        
        duplicates = {text: count for text, count in text_counts.items() if count > 1}
        
        return {
            'size': len(cluster_detections),
            'type_distribution': type_counts,
            'potential_duplicates': duplicates,
            'dominant_type': max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else 'unknown'
        }
    
    def detect_and_filter_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """检测并过滤重复的检测结果"""
        if not detections:
            return []
        
        filtered = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
            
            # 查找重复项
            duplicates = [det1]
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._is_duplicate(det1, det2):
                    duplicates.append(det2)
                    used_indices.add(j)
            
            # 选择最佳检测结果
            best_detection = self._select_best_duplicate(duplicates)
            filtered.append(best_detection)
            
            used_indices.add(i)
        
        return filtered
    
    def _is_duplicate(self, det1: Dict, det2: Dict) -> bool:
        """判断两个检测是否为重复"""
        # 文本相似性
        if det1['text'] == det2['text']:
            # 距离检查
            dist = math.sqrt((det1['center'][0] - det2['center'][0])**2 + 
                           (det1['center'][1] - det2['center'][1])**2)
            return dist < self.spatial_rules['duplicate_detection_radius']
        
        return False
    
    def _select_best_duplicate(self, duplicates: List[Dict]) -> Dict:
        """从重复检测中选择最佳的一个"""
        return max(duplicates, key=lambda x: x.get('spatial_confidence', 
                                                  x.get('final_score', 
                                                       x.get('confidence', 0))))
    
    def validate_geographic_consistency(self, detections: List[Dict],
                                      road_network: Dict,
                                      regions: List[Dict]) -> List[Dict]:
        """验证地理一致性"""
        validated = []
        
        for detection in detections:
            # 基本验证规则
            is_valid = True
            validation_issues = []
            
            # 验证1: 路線价应该在道路附近
            if 'route_price' in detection.get('spatial_type', ''):
                if detection['road_distance'] > self.spatial_rules['route_price_max_distance_to_road']:
                    validation_issues.append('route_price_too_far_from_road')
            
            # 验证2: 街区番号应该在区域内
            if 'block_number' in detection.get('spatial_type', ''):
                if detection['containing_region'] is None:
                    validation_issues.append('block_number_outside_region')
            
            # 验证3: 文本格式一致性
            text = detection['text']
            spatial_type = detection.get('spatial_type', '')
            
            if 'route_price' in spatial_type:
                # 路線价应该有数字
                if not any(c.isdigit() for c in text):
                    validation_issues.append('route_price_no_digits')
            elif 'block_number' in spatial_type:
                # 街区番号应该主要是数字
                if not text.isdigit() and '-' not in text:
                    validation_issues.append('block_number_non_numeric')
            
            # 更新检测结果
            detection['validation_issues'] = validation_issues
            detection['is_valid'] = len(validation_issues) == 0
            
            # 调整置信度
            if validation_issues:
                penalty = len(validation_issues) * 0.1
                detection['spatial_confidence'] = max(0, 
                    detection.get('spatial_confidence', 0.5) - penalty)
            
            validated.append(detection)
        
        return validated
    
    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """计算统计信息"""
        if not detections:
            return {}
        
        spatial_types = [det.get('spatial_type', 'unknown') for det in detections]
        type_counts = {}
        for t in spatial_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        valid_count = sum(1 for det in detections if det.get('is_valid', True))
        avg_confidence = sum(det.get('spatial_confidence', 0) for det in detections) / len(detections)
        
        return {
            'total_detections': len(detections),
            'valid_detections': valid_count,
            'invalid_detections': len(detections) - valid_count,
            'type_distribution': type_counts,
            'average_confidence': avg_confidence,
            'validation_rate': valid_count / len(detections) if detections else 0
        }
    
    def _save_spatial_debug(self, image: np.ndarray, 
                           spatial_result: Dict, 
                           image_name: str):
        """保存空间分析调试信息"""
        if not self.debug_dir:
            return
        
        # 创建可视化图像
        debug_image = image.copy()
        
        # 绘制道路网络
        for line_info in spatial_result['road_network']['grouped_lines']:
            line = line_info['line']
            x1, y1, x2, y2 = line
            cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # 绘制交叉点
        for intersection in spatial_result['road_network']['intersections']:
            cv2.circle(debug_image, intersection, 5, (0, 255, 255), -1)
        
        # 绘制区域
        for region in spatial_result['regions']:
            contour = np.array(region['contour'])
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            center = region['center']
            cv2.circle(debug_image, center, 3, (0, 255, 0), -1)
        
        # 绘制检测结果
        for detection in spatial_result['classified_detections']:
            bbox = detection['bbox']
            center = detection['center']
            spatial_type = detection.get('spatial_type', 'unknown')
            is_valid = detection.get('is_valid', True)
            
            # 根据类型和有效性选择颜色
            if not is_valid:
                color = (128, 128, 128)  # 灰色：无效
            elif 'route_price' in spatial_type:
                color = (255, 0, 0)      # 蓝色：路線价
            elif 'block_number' in spatial_type:
                color = (0, 255, 0)      # 绿色：街区番号
            else:
                color = (0, 255, 255)    # 黄色：其他
            
            # 绘制边界框
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(debug_image, [points], True, color, 2)
            
            # 标注文本
            label = f"{detection['text']}({spatial_type[:8]})"
            cv2.putText(debug_image, label, (int(center[0]), int(center[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 保存图像
        debug_path = self.debug_dir / f"{image_name}_spatial_analysis.jpg"
        cv2.imwrite(str(debug_path), debug_image)
        
        # 保存JSON数据
        json_path = self.debug_dir / f"{image_name}_spatial_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(spatial_result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.debug(f"空间分析调试信息已保存: {debug_path}, {json_path}")

# 使用示例
if __name__ == "__main__":
    # 测试空间智能分析引擎
    engine = SpatialIntelligenceEngine(debug_mode=True)
    
    # 模拟测试数据
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)
    test_detections = [
        {
            'text': '115E',
            'bbox': [[100, 100], [130, 100], [130, 120], [100, 120]],
            'confidence': 0.9,
            'type': 'route_price'
        },
        {
            'text': '42',
            'bbox': [[200, 200], [220, 200], [220, 220], [200, 220]],
            'confidence': 0.8,
            'type': 'block_number'
        }
    ]
    
    result = engine.analyze_spatial_context(test_image, test_detections, "test")
    print(f"空间分析完成：")
    print(f"- 道路线段: {result['road_network']['line_count']}")
    print(f"- 区域数量: {len(result['regions'])}")
    print(f"- 有效检测: {result['statistics']['valid_detections']}")