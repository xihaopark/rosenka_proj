"""
圆内文字识别模块
专门用于识别圆形价格标记中的文字
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from typing import List, Dict, Tuple, Optional
import logging

class CircleOCR:
    """圆内文字识别器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化PaddleOCR
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='japan',
                use_gpu=False,  # 改为CPU模式避免GPU问题
                show_log=False
            )
            self.logger.info("PaddleOCR初始化成功")
        except Exception as e:
            self.logger.error(f"PaddleOCR初始化失败: {e}")
            # 尝试使用Tesseract作为备选
            try:
                import pytesseract
                self.ocr_engine = 'tesseract'
                self.logger.info("使用Tesseract作为OCR引擎")
            except ImportError:
                self.logger.error("Tesseract不可用")
                self.ocr_engine = None
        
        # 价格文本验证正则表达式
        self.price_patterns = [
            r'^\d+[A-Z]?$',  # 115E, 120, 95A等
            r'^\d+万$',      # 120万等
            r'^\d+千$',      # 95千等
        ]
    
    def recognize_circle_text(self, image: np.ndarray, circles: List[Dict]) -> List[Dict]:
        """
        识别圆内文字
        
        Args:
            image: 原始图像
            circles: 圆形检测结果
            
        Returns:
            包含文字识别结果的圆形列表
        """
        if not self.ocr_engine:
            self.logger.error("OCR引擎未初始化")
            return circles
        
        results = []
        
        for circle in circles:
            try:
                # 提取圆形ROI
                roi = self.extract_circle_roi(image, circle)
                
                if roi is None:
                    continue
                
                # 预处理ROI
                processed_roi = self.preprocess_circle_roi(roi)
                
                # OCR识别
                text_result = self.perform_ocr(processed_roi)
                
                # 验证和清理文本
                validated_text = self.validate_price_text(text_result)
                
                # 更新圆形信息
                circle_with_text = circle.copy()
                circle_with_text.update({
                    'text': validated_text,
                    'ocr_confidence': text_result.get('confidence', 0.0) if text_result else 0.0,
                    'is_valid_price': validated_text is not None
                })
                
                results.append(circle_with_text)
                
            except Exception as e:
                self.logger.error(f"圆内文字识别失败: {e}")
                circle['text'] = None
                circle['ocr_confidence'] = 0.0
                circle['is_valid_price'] = False
                results.append(circle)
        
        return results
    
    def extract_circle_roi(self, image: np.ndarray, circle: Dict) -> Optional[np.ndarray]:
        """
        提取圆形感兴趣区域
        
        Args:
            image: 原始图像
            circle: 圆形信息
            
        Returns:
            圆形ROI图像
        """
        try:
            center_x, center_y = circle['center']
            radius = circle['radius']
            
            # 扩大ROI以包含更多上下文
            expanded_radius = int(radius * 1.2)
            
            # 计算ROI边界
            x1 = max(0, center_x - expanded_radius)
            y1 = max(0, center_y - expanded_radius)
            x2 = min(image.shape[1], center_x + expanded_radius)
            y2 = min(image.shape[0], center_y + expanded_radius)
            
            # 提取ROI
            roi = image[y1:y2, x1:x2]
            
            # 调整圆心坐标到ROI坐标系
            roi_center_x = center_x - x1
            roi_center_y = center_y - y1
            
            # 创建圆形掩码
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (roi_center_x, roi_center_y), radius, 255, -1)
            
            # 应用掩码
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
            
            return roi_masked
            
        except Exception as e:
            self.logger.error(f"提取圆形ROI失败: {e}")
            return None
    
    def preprocess_circle_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        预处理圆形ROI
        
        Args:
            roi: 圆形ROI图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 反色处理（黑底白字 -> 白底黑字）
        inverted = cv2.bitwise_not(gray)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(inverted)
        
        # 尺寸调整 - OCR在较大尺寸下效果更好
        height, width = denoised.shape
        if height < 64 or width < 64:
            scale_factor = max(64/height, 64/width, 2.0)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            denoised = cv2.resize(denoised, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作清理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def perform_ocr(self, processed_roi: np.ndarray) -> Optional[Dict]:
        """
        执行OCR识别
        
        Args:
            processed_roi: 预处理后的ROI
            
        Returns:
            OCR识别结果
        """
        try:
            if isinstance(self.ocr_engine, str) and self.ocr_engine == 'tesseract':
                # 使用Tesseract
                import pytesseract
                text = pytesseract.image_to_string(processed_roi, lang='jpn', config='--psm 6')
                return {'text': text.strip(), 'confidence': 0.8}
            else:
                # PaddleOCR识别
                ocr_results = self.ocr_engine.ocr(processed_roi, cls=True)
                
                if not ocr_results or not ocr_results[0]:
                    return None
                
                # 提取最佳结果
                best_result = None
                best_confidence = 0
            
            for line in ocr_results[0]:
                if len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            'text': text,
                            'confidence': confidence,
                            'bbox': line[0]
                        }
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {e}")
            return None
    
    def validate_price_text(self, ocr_result: Optional[Dict]) -> Optional[str]:
        """
        验证和清理价格文本
        
        Args:
            ocr_result: OCR识别结果
            
        Returns:
            验证后的价格文本
        """
        if not ocr_result or not ocr_result.get('text'):
            return None
        
        text = ocr_result['text'].strip()
        
        # 清理常见的OCR错误
        text = self.clean_ocr_text(text)
        
        # 验证价格格式
        if self.is_valid_price_format(text):
            return text
        
        return None
    
    def clean_ocr_text(self, text: str) -> str:
        """
        清理OCR文本中的常见错误
        
        Args:
            text: 原始OCR文本
            
        Returns:
            清理后的文本
        """
        # 移除空格
        text = text.replace(' ', '')
        
        # 常见字符替换
        replacements = {
            'O': '0',  # 字母O替换为数字0
            'o': '0',
            'I': '1',  # 字母I替换为数字1
            'l': '1',
            'S': '5',  # 字母S替换为数字5
            'G': '6',  # 字母G替换为数字6
            'B': '8',  # 字母B替换为数字8
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def is_valid_price_format(self, text: str) -> bool:
        """
        验证价格格式是否有效
        
        Args:
            text: 待验证的文本
            
        Returns:
            是否为有效的价格格式
        """
        # 检查是否匹配任何价格模式
        for pattern in self.price_patterns:
            if re.match(pattern, text):
                return True
        
        # 额外检查：纯数字且长度合理
        if text.isdigit() and 1 <= len(text) <= 6:
            return True
        
        return False
    
    def recognize_single_circle(self, image: np.ndarray, circle: Dict) -> Dict:
        """
        识别单个圆形中的文字
        
        Args:
            image: 原始图像
            circle: 圆形信息
            
        Returns:
            包含文字识别结果的圆形信息
        """
        result = self.recognize_circle_text(image, [circle])
        return result[0] if result else circle
    
    def batch_recognize(self, image: np.ndarray, circles: List[Dict], 
                       batch_size: int = 10) -> List[Dict]:
        """
        批量识别圆内文字
        
        Args:
            image: 原始图像
            circles: 圆形列表
            batch_size: 批处理大小
            
        Returns:
            识别结果列表
        """
        results = []
        
        for i in range(0, len(circles), batch_size):
            batch = circles[i:i + batch_size]
            batch_results = self.recognize_circle_text(image, batch)
            results.extend(batch_results)
        
        return results
    
    def get_recognition_statistics(self, circles: List[Dict]) -> Dict:
        """
        获取识别统计信息
        
        Args:
            circles: 识别结果列表
            
        Returns:
            统计信息
        """
        total_circles = len(circles)
        recognized_circles = sum(1 for c in circles if c.get('text'))
        valid_prices = sum(1 for c in circles if c.get('is_valid_price'))
        
        avg_confidence = 0
        if recognized_circles > 0:
            avg_confidence = sum(c.get('ocr_confidence', 0) for c in circles if c.get('text')) / recognized_circles
        
        return {
            'total_circles': total_circles,
            'recognized_circles': recognized_circles,
            'valid_prices': valid_prices,
            'recognition_rate': recognized_circles / total_circles if total_circles > 0 else 0,
            'valid_price_rate': valid_prices / total_circles if total_circles > 0 else 0,
            'average_confidence': avg_confidence
        } 