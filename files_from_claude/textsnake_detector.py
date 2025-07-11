"""
textsnake_detector.py
TextSnake - 可以检测曲线文字
"""

class TextSnakeDetector:
    """TextSnake曲线文字检测器"""
    
    def __init__(self):
        # 适合检测沿道路的弯曲文字
        self.model = self._load_textsnake_model()
        
    def detect_curved_text(self, image: np.ndarray):
        """检测曲线文字"""
        # TextSnake可以检测沿道路弯曲的文字
        # 返回文字中心线和半径
        pass