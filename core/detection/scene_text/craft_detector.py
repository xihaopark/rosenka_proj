"""
CRAFT文字检测器
用于检测地图中的常规文字区域
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging
import urllib.request
import os

class CRAFT(nn.Module):
    """CRAFT模型实现"""
    
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        
        # VGG16 backbone
        self.basenet = self._make_vgg16_layers()
        
        # Feature pyramid network
        self.upconv1 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Output layers
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.conv_link = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_vgg16_layers(self):
        """构建VGG16层"""
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        # 这里应该加载实际的预训练权重
        # 为了演示，我们跳过这一步
        pass
    
    def forward(self, x):
        """前向传播"""
        # VGG16 features
        sources = []
        for k in range(len(self.basenet)):
            x = self.basenet[k](x)
            if k in [3, 8, 15, 22, 29]:  # 保存特定层的输出
                sources.append(x)
        
        # Feature pyramid
        y = F.relu(self.bn1(self.upconv1(sources[-1])))
        y = F.relu(self.bn2(self.upconv2(y)))
        y = F.relu(self.bn3(self.upconv3(y)))
        y = F.relu(self.bn4(self.upconv4(y)))
        
        # 调整到32通道
        y = F.interpolate(y, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=False)
        if y.size(1) != 32:
            # 使用卷积层调整通道数
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Conv2d(y.size(1), 32, 1).to(y.device)
            y = self.channel_adapter(y)
        
        # 输出
        region_score = self.conv_cls(y)
        affinity_score = self.conv_link(y)
        
        return region_score, affinity_score

class CRAFTDetector:
    """CRAFT文字检测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRAFT(pretrained=True).to(self.device)
        self.model.eval()
        
        self.logger = logging.getLogger(__name__)
        
        # 检测参数
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"成功加载CRAFT模型: {model_path}")
        except Exception as e:
            self.logger.error(f"加载CRAFT模型失败: {e}")
    
    def detect_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        检测文字区域
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的文字框列表
        """
        try:
            # 图像预处理
            img_resized, target_ratio, size_heatmap = self.resize_aspect_ratio(
                image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
            )
            
            ratio_h = ratio_w = 1 / target_ratio
            
            # 转换为tensor
            x = self.normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # 模型推理
            with torch.no_grad():
                y, feature = self.model(x)
                
            # 后处理
            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = feature[0, :, :, 0].cpu().data.numpy()
            
            # 获取文字框
            boxes, polys = self.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text)
            
            # 坐标映射回原图
            boxes = self.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = self.adjustResultCoordinates(polys, ratio_w, ratio_h)
            
            return boxes
            
        except Exception as e:
            self.logger.error(f"CRAFT文字检测失败: {e}")
            return []
    
    def resize_aspect_ratio(self, img, square_size, interpolation, mag_ratio=1):
        """调整图像尺寸保持宽高比"""
        height, width, channel = img.shape
        
        # 计算目标尺寸
        target_size = mag_ratio * max(height, width)
        
        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)
        target_h, target_w = int(height * ratio), int(width * ratio)
        
        # 调整为32的倍数
        target_h = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
        target_w = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)
        
        # 调整图像尺寸
        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        
        # 创建正方形画布
        target_h32, target_w32 = target_h, target_w
        if target_h32 < square_size or target_w32 < square_size:
            target_h32 = square_size
            target_w32 = square_size
            
        proc = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
        proc[:target_h, :target_w, :] = img
        
        return proc, ratio, (target_h, target_w)
    
    def normalizeMeanVariance(self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """图像标准化"""
        img = in_img.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img
    
    def getDetBoxes(self, textmap, linkmap, text_threshold, link_threshold, low_text):
        """从热力图中提取文字框"""
        # 简化的文字框提取算法
        boxes = []
        
        # 二值化
        text_score_comb = np.clip(textmap + linkmap, 0, 1)
        ret, text_score = cv2.threshold(text_score_comb, low_text, 1, cv2.THRESH_BINARY)
        
        # 连通组件分析
        labels, stats = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)[1:3]
        
        for k in range(1, labels.max() + 1):
            # 获取连通区域
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue
                
            # 获取边界框
            x = stats[k, cv2.CC_STAT_LEFT]
            y = stats[k, cv2.CC_STAT_TOP]
            w = stats[k, cv2.CC_STAT_WIDTH]
            h = stats[k, cv2.CC_STAT_HEIGHT]
            
            # 创建四边形
            box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            boxes.append(box)
        
        return boxes, boxes
    
    def adjustResultCoordinates(self, polys, ratio_w, ratio_h):
        """调整坐标到原图"""
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * 2, ratio_h * 2)
        return polys
    
    def detect_with_confidence(self, image: np.ndarray, confidence_threshold: float = 0.7) -> List[dict]:
        """
        带置信度的文字检测
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            
        Returns:
            检测结果列表，包含位置和置信度
        """
        boxes = self.detect_text_regions(image)
        
        results = []
        for i, box in enumerate(boxes):
            # 计算置信度（简化版本）
            confidence = 0.8  # 实际应该基于模型输出计算
            
            if confidence >= confidence_threshold:
                results.append({
                    'box': box,
                    'confidence': confidence,
                    'type': 'regular_text',
                    'detector': 'CRAFT'
                })
        
        return results 