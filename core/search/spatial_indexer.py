"""
空间索引器
用于构建和查询空间关系
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import networkx as nx

try:
    import rtree.index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False

class SpatialIndexer:
    """空间索引器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rtree_available = RTREE_AVAILABLE
        
        # 空间索引
        self.text_index = None
        self.circle_index = None
        self.kdtree = None
        
        # 关系图
        self.relationship_graph = nx.Graph()
        
        # 空间参数
        self.max_distance = 100  # 最大关联距离
        self.angle_threshold = 30  # 角度阈值（度）
    
    def build_index(self, text_detections: List[Dict], 
                   circle_detections: List[Dict], 
                   image_shape: Tuple[int, int]) -> Dict:
        """
        构建空间索引
        
        Args:
            text_detections: 文字检测结果
            circle_detections: 圆形检测结果
            image_shape: 图像尺寸
            
        Returns:
            空间索引信息
        """
        try:
            # 构建R-tree索引（如果可用）
            if self.rtree_available:
                self._build_rtree_index(text_detections, circle_detections)
            
            # 构建KD-tree索引
            self._build_kdtree_index(text_detections, circle_detections)
            
            # 构建关系图
            self._build_relationship_graph(text_detections, circle_detections)
            
            return {
                'text_count': len(text_detections),
                'circle_count': len(circle_detections),
                'rtree_available': self.rtree_available,
                'kdtree_built': self.kdtree is not None,
                'image_shape': image_shape
            }
            
        except Exception as e:
            self.logger.error(f"构建空间索引失败: {e}")
            return {}
    
    def _build_rtree_index(self, text_detections: List[Dict], 
                          circle_detections: List[Dict]):
        """构建R-tree索引"""
        if not self.rtree_available:
            return
        
        try:
            # 文字检测索引
            self.text_index = rtree.index.Index()
            for i, detection in enumerate(text_detections):
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    self.text_index.insert(i, bbox)
            
            # 圆形检测索引
            self.circle_index = rtree.index.Index()
            for i, detection in enumerate(circle_detections):
                center = detection.get('center', (0, 0))
                radius = detection.get('radius', 0)
                # 将圆形转换为边界框
                bbox = [
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius
                ]
                self.circle_index.insert(i, bbox)
            
            self.logger.info("R-tree索引构建完成")
            
        except Exception as e:
            self.logger.error(f"R-tree索引构建失败: {e}")
    
    def _build_kdtree_index(self, text_detections: List[Dict], 
                           circle_detections: List[Dict]):
        """构建KD-tree索引"""
        try:
            # 收集所有中心点
            points = []
            point_types = []
            point_indices = []
            
            # 文字检测点
            for i, detection in enumerate(text_detections):
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    points.append([center_x, center_y])
                    point_types.append('text')
                    point_indices.append(i)
            
            # 圆形检测点
            for i, detection in enumerate(circle_detections):
                center = detection.get('center', (0, 0))
                points.append([center[0], center[1]])
                point_types.append('circle')
                point_indices.append(i)
            
            if points:
                self.kdtree = cKDTree(np.array(points))
                self.point_types = point_types
                self.point_indices = point_indices
                
                self.logger.info(f"KD-tree索引构建完成，包含{len(points)}个点")
            
        except Exception as e:
            self.logger.error(f"KD-tree索引构建失败: {e}")
    
    def _build_relationship_graph(self, text_detections: List[Dict], 
                                 circle_detections: List[Dict]):
        """构建关系图"""
        try:
            self.relationship_graph.clear()
            
            # 添加文字节点
            for i, detection in enumerate(text_detections):
                self.relationship_graph.add_node(
                    f"text_{i}",
                    type='text',
                    detection=detection,
                    center=self._get_detection_center(detection)
                )
            
            # 添加圆形节点
            for i, detection in enumerate(circle_detections):
                self.relationship_graph.add_node(
                    f"circle_{i}",
                    type='circle',
                    detection=detection,
                    center=detection.get('center', (0, 0))
                )
            
            # 添加边（基于空间距离）
            self._add_spatial_edges()
            
            self.logger.info(f"关系图构建完成: {self.relationship_graph.number_of_nodes()}个节点, "
                           f"{self.relationship_graph.number_of_edges()}条边")
            
        except Exception as e:
            self.logger.error(f"关系图构建失败: {e}")
    
    def _get_detection_center(self, detection: Dict) -> Tuple[float, float]:
        """获取检测结果的中心点"""
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return (0, 0)
    
    def _add_spatial_edges(self):
        """添加空间关系边"""
        nodes = list(self.relationship_graph.nodes(data=True))
        
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                # 计算距离
                center1 = data1['center']
                center2 = data2['center']
                distance = np.sqrt((center1[0] - center2[0])**2 + 
                                 (center1[1] - center2[1])**2)
                
                # 如果距离在阈值内，添加边
                if distance <= self.max_distance:
                    # 计算关系强度
                    strength = 1.0 - (distance / self.max_distance)
                    
                    # 添加边
                    self.relationship_graph.add_edge(
                        node1, node2,
                        distance=distance,
                        strength=strength,
                        type='spatial'
                    )
    
    def calculate_relationships(self, text_detections: List[Dict], 
                              circle_detections: List[Dict]) -> List[Dict]:
        """
        计算空间关系
        
        Args:
            text_detections: 文字检测结果
            circle_detections: 圆形检测结果
            
        Returns:
            关系列表
        """
        relationships = []
        
        try:
            # 为每个圆形查找最近的文字
            for i, circle in enumerate(circle_detections):
                circle_center = circle.get('center', (0, 0))
                
                # 查找附近的文字
                nearby_texts = self._find_nearby_texts(circle_center, text_detections)
                
                # 分析每个附近文字的关系
                for text_info in nearby_texts:
                    relationship = self._analyze_relationship(circle, text_info)
                    if relationship:
                        relationships.append(relationship)
            
            # 按关系强度排序
            relationships.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"计算空间关系失败: {e}")
            return []
    
    def _find_nearby_texts(self, center: Tuple[float, float], 
                          text_detections: List[Dict]) -> List[Dict]:
        """查找附近的文字"""
        nearby_texts = []
        
        for i, text_detection in enumerate(text_detections):
            text_center = self._get_detection_center(text_detection)
            distance = np.sqrt((center[0] - text_center[0])**2 + 
                             (center[1] - text_center[1])**2)
            
            if distance <= self.max_distance:
                nearby_texts.append({
                    'index': i,
                    'detection': text_detection,
                    'center': text_center,
                    'distance': distance
                })
        
        # 按距离排序
        nearby_texts.sort(key=lambda x: x['distance'])
        
        return nearby_texts
    
    def _analyze_relationship(self, circle: Dict, text_info: Dict) -> Optional[Dict]:
        """分析圆形和文字的关系"""
        try:
            circle_center = circle.get('center', (0, 0))
            text_center = text_info['center']
            distance = text_info['distance']
            
            # 计算方向角度
            angle = np.arctan2(text_center[1] - circle_center[1], 
                             text_center[0] - circle_center[0])
            angle_degrees = np.degrees(angle)
            
            # 计算关系置信度
            confidence = self._calculate_relationship_confidence(
                circle, text_info['detection'], distance, angle_degrees
            )
            
            if confidence > 0.3:  # 置信度阈值
                return {
                    'circle_index': circle.get('index', -1),
                    'text_index': text_info['index'],
                    'circle_center': circle_center,
                    'text_center': text_center,
                    'distance': distance,
                    'angle': angle_degrees,
                    'confidence': confidence,
                    'relationship_type': self._classify_relationship(distance, angle_degrees),
                    'circle_text': circle.get('text', ''),
                    'text_content': text_info['detection'].get('text', '')
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"分析关系失败: {e}")
            return None
    
    def _calculate_relationship_confidence(self, circle: Dict, text_detection: Dict, 
                                         distance: float, angle: float) -> float:
        """计算关系置信度"""
        # 基础置信度（基于距离）
        distance_confidence = max(0, 1.0 - (distance / self.max_distance))
        
        # 角度置信度（垂直或水平方向的文字关系更强）
        angle_confidence = 1.0
        normalized_angle = abs(angle) % 90
        if normalized_angle > 45:
            normalized_angle = 90 - normalized_angle
        
        if normalized_angle <= self.angle_threshold:
            angle_confidence = 1.0 - (normalized_angle / self.angle_threshold) * 0.3
        
        # 大小匹配置信度
        size_confidence = 1.0
        text_bbox = text_detection.get('bbox', [])
        if len(text_bbox) == 4:
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            circle_radius = circle.get('radius', 0)
            
            # 文字大小与圆形大小的匹配程度
            size_ratio = min(text_width, text_height) / (circle_radius * 2)
            if 0.3 <= size_ratio <= 2.0:
                size_confidence = 1.0
            else:
                size_confidence = 0.7
        
        # 综合置信度
        final_confidence = (distance_confidence * 0.5 + 
                          angle_confidence * 0.3 + 
                          size_confidence * 0.2)
        
        return final_confidence
    
    def _classify_relationship(self, distance: float, angle: float) -> str:
        """分类关系类型"""
        if distance <= 30:
            return 'adjacent'  # 相邻
        elif distance <= 60:
            return 'nearby'    # 附近
        else:
            return 'distant'   # 远距离
    
    def query_nearby_elements(self, center: Tuple[float, float], 
                            radius: float = 50) -> List[Dict]:
        """查询附近的元素"""
        if not self.kdtree:
            return []
        
        try:
            # 使用KD-tree查询
            indices = self.kdtree.query_ball_point(center, radius)
            
            results = []
            for idx in indices:
                results.append({
                    'type': self.point_types[idx],
                    'index': self.point_indices[idx],
                    'distance': np.sqrt((self.kdtree.data[idx][0] - center[0])**2 + 
                                      (self.kdtree.data[idx][1] - center[1])**2)
                })
            
            # 按距离排序
            results.sort(key=lambda x: x['distance'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"查询附近元素失败: {e}")
            return []
    
    def find_text_circle_pairs(self, text_detections: List[Dict], 
                              circle_detections: List[Dict]) -> List[Dict]:
        """查找文字-圆形配对"""
        pairs = []
        
        try:
            relationships = self.calculate_relationships(text_detections, circle_detections)
            
            # 为每个圆形找到最佳匹配的文字
            circle_matched = set()
            text_matched = set()
            
            for relation in relationships:
                circle_idx = relation['circle_index']
                text_idx = relation['text_index']
                
                # 避免重复匹配
                if circle_idx not in circle_matched and text_idx not in text_matched:
                    pairs.append({
                        'circle': circle_detections[circle_idx] if circle_idx < len(circle_detections) else None,
                        'text': text_detections[text_idx] if text_idx < len(text_detections) else None,
                        'relationship': relation
                    })
                    
                    circle_matched.add(circle_idx)
                    text_matched.add(text_idx)
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"查找配对失败: {e}")
            return []
    
    def get_spatial_statistics(self) -> Dict:
        """获取空间统计信息"""
        stats = {
            'total_nodes': self.relationship_graph.number_of_nodes(),
            'total_edges': self.relationship_graph.number_of_edges(),
            'rtree_available': self.rtree_available,
            'kdtree_available': self.kdtree is not None
        }
        
        if self.relationship_graph.number_of_nodes() > 0:
            # 计算连通组件
            connected_components = list(nx.connected_components(self.relationship_graph))
            stats['connected_components'] = len(connected_components)
            
            # 计算平均度
            degrees = dict(self.relationship_graph.degree())
            stats['average_degree'] = sum(degrees.values()) / len(degrees)
            
            # 节点类型统计
            node_types = {}
            for node, data in self.relationship_graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            stats['node_types'] = node_types
        
        return stats 