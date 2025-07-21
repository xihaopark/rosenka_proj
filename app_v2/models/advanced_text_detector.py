"""
advanced_text_detector.py
高级文本检测器 - 集成多种SOTA模型
"""

import torch
import torch.nn as nn
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np

class AdvancedTextDetector:
    """高级文本检测器"""
    
    def __init__(self, model_type='dbnet'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._init_model()
        
    def _init_model(self):
        """初始化模型"""
        if self.model_type == 'dbnet':
            return self._init_dbnet()
        elif self.model_type == 'craft':
            return self._init_craft()
        elif self.model_type == 'textsnake':
            return self._init_textsnake()
        else:
            return self._init_maskrcnn()
    
    def _init_dbnet(self):
        """初始化DBNet模型"""
        # DBNet: Differentiable Binarization
        # 适合检测多方向文本
        from models.dbnet import DBNet
        
        model = DBNet(backbone='resnet50')
        model.load_state_dict(torch.load('weights/dbnet_r50.pth'))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _init_craft(self):
        """初始化CRAFT模型"""
        # CRAFT: Character Region Awareness for Text Detection
        # 适合检测字符级别的文本
        from models.craft import CRAFT
        
        model = CRAFT()
        model.load_state_dict(torch.load('weights/craft.pth'))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _init_maskrcnn(self):
        """初始化Mask R-CNN模型"""
        # 使用Detectron2的预训练模型
        cfg = get_cfg()
        cfg.merge_from_file("configs/text_detection_maskrcnn.yaml")
        cfg.MODEL.WEIGHTS = "weights/text_maskrcnn.pth"
        cfg.MODEL.DEVICE = str(self.device)
        
        return DefaultPredictor(cfg)
    
    def detect(self, image: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """检测文本区域"""
        if self.model_type == 'dbnet':
            return self._detect_dbnet(image, threshold)
        elif self.model_type == 'craft':
            return self._detect_craft(image, threshold)
        else:
            return self._detect_maskrcnn(image, threshold)
    
    def _detect_dbnet(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """使用DBNet检测"""
        # 预处理
        img_tensor = self._preprocess_dbnet(image)
        
        with torch.no_grad():
            preds = self.model(img_tensor)
            
        # 后处理获取文本框
        boxes = self._postprocess_dbnet(preds, image.shape, threshold)
        
        results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            region = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(box[4]) if len(box) > 4 else threshold,
                'polygon': box[:8].reshape(-1, 2).tolist() if len(box) >= 8 else None
            }
            results.append(region)
        
        return results