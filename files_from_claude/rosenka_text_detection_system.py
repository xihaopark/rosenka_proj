"""
rosenka_text_detection_system.py
路線価図文字检测完整系统
"""

class RosenkaTextDetectionSystem:
    """路線価図文字检测系统"""
    
    def __init__(self):
        # 主检测器：PaddleOCR（最稳定）
        self.paddle_detector = PaddleSceneTextDetector()
        
        # 备用：EAST（处理特殊角度）
        self.east_detector = None  # 按需加载
        
        # 后处理器
        self.postprocessor = TextPostProcessor()
        
    def detect_all_text(self, image: np.ndarray):
        """检测所有文字"""
        # 1. PaddleOCR检测
        paddle_results = self.paddle_detector.detect_by_text_type(image)
        
        # 2. 后处理优化
        optimized_results = self.postprocessor.optimize_detections(
            paddle_results, image
        )
        
        return optimized_results
    
    def detect_specific_elements(self, image: np.ndarray):
        """检测特定元素"""
        all_detections = self.detect_all_text(image)
        
        # 分类结果
        elements = {
            'route_prices': [],    # 路线价（29E, 30E等）
            'street_names': [],    # 街道名
            'landmarks': [],       # 地标
            'addresses': []        # 地址
        }
        
        # 基于位置、大小、内容分类
        for text_type, boxes in all_detections.items():
            if text_type == 'numbers':
                # 小的数字框可能是路线价
                for box in boxes:
                    if self._is_route_price_box(box):
                        elements['route_prices'].append(box)
            elif text_type == 'japanese':
                # 日文可能是地名或地址
                for box in boxes:
                    if self._is_street_name(box):
                        elements['street_names'].append(box)
                    else:
                        elements['addresses'].append(box)
        
        return elements
    
    def _is_route_price_box(self, box):
        """判断是否为路线价框"""
        bbox = box['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # 路线价特征：小、正方形、通常在道路附近
        return 20 < w < 80 and 20 < h < 80 and abs(w - h) < 20
    
    def _is_street_name(self, box):
        """判断是否为街道名"""
        # 街道名通常是横向长条
        bbox = box['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w / h > 3  # 宽高比大于3