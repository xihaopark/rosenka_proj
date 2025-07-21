#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Scale OCR Detector for Stage 5 Rosenka OCR System
多尺度OCR检测器 - 专门针对路線価図的高召回率文字检测

主要功能:
1. 多尺度并行检测 (1.0x, 1.5x, 2.0x, 2.5x)
2. 多引擎融合 (PaddleOCR + EasyOCR + Tesseract)
3. 参数优化配置（高召回率模式）
4. 坐标系统标准化和结果融合
5. 路線価图专用模式匹配
6. 置信度重新评分机制
"""

import cv2
import numpy as np
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import time

# OCR引擎导入
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiScaleOCRDetector:
    """Stage 5 多尺度OCR检测器"""
    
    def __init__(self, use_gpu: bool = False, debug_mode: bool = False):
        """
        初始化多尺度OCR检测器
        
        Args:
            use_gpu: 是否使用GPU加速
            debug_mode: 是否启用调试模式
        """
        self.use_gpu = use_gpu
        self.debug_mode = debug_mode
        self.debug_dir = Path("debug_detection") if debug_mode else None
        
        # 初始化OCR引擎
        self.engines = {}
        self._initialize_ocr_engines()
        
        # 多尺度参数配置
        self.scales = [1.0, 1.5, 2.0, 2.5]
        
        # 路線価图专用模式匹配
        self.patterns = {
            'block_number': r'^\d{1,3}$',  # 街区番号：1-3位数字
            'route_price': r'^\d{1,4}[A-G]$',  # 路線价：数字+字母A-G
            'complex_number': r'^\d{1,3}-\d{1,3}[A-G]?$',  # 复合番号
            'price_with_unit': r'^\d{1,4}万?[A-G]?$',  # 带万字单位
            'reference_code': r'^[A-Z]\d{1,3}$',  # 参考编号：字母+数字
        }
        
        # OCR配置参数
        self.ocr_configs = self._get_optimized_configs()
        
        if self.debug_mode and self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"调试模式启用，检测结果保存至: {self.debug_dir}")
    
    def _initialize_ocr_engines(self):
        """初始化OCR引擎"""
        
        # 初始化PaddleOCR（主引擎）
        if PADDLE_AVAILABLE:
            try:
                # 使用最简化配置
                self.engines['paddle'] = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan'
                )
                logger.info("PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
        
        # 初始化EasyOCR（辅助引擎）
        if EASYOCR_AVAILABLE:
            try:
                self.engines['easy'] = easyocr.Reader(
                    ['ja', 'en'], 
                    gpu=self.use_gpu,
                    verbose=False
                )
                logger.info("EasyOCR 初始化成功")
            except Exception as e:
                logger.error(f"EasyOCR 初始化失败: {e}")
        
        # Tesseract配置（备用引擎）
        if TESSERACT_AVAILABLE:
            self.engines['tesseract'] = True
            logger.info("Tesseract 可用")
        
        if not self.engines:
            raise RuntimeError("没有可用的OCR引擎！请安装PaddleOCR、EasyOCR或Tesseract")
    
    def _get_optimized_configs(self) -> Dict:
        """获取优化的OCR配置"""
        return {
            "paddle_high_recall": {
                "det_db_thresh": 0.1,
                "det_db_box_thresh": 0.3,
                "det_db_unclip_ratio": 1.8,
                "rec_batch_num": 30
            },
            "easy_precise": {
                "width_ths": 0.7,
                "height_ths": 0.7,
                "decoder": 'beamsearch'
            },
            "tesseract_number": {
                "config": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFG-"
            }
        }
    
    def detect_multi_scale(self, image: np.ndarray, 
                          image_name: str = "image") -> List[Dict]:
        """
        多尺度OCR检测
        
        Args:
            image: 输入图像
            image_name: 图像名称（用于调试）
            
        Returns:
            检测结果列表，每个结果包含text, bbox, confidence, scale, engine等信息
        """
        logger.info(f"开始多尺度OCR检测: {image_name}")
        start_time = time.time()
        
        all_results = []
        
        # 对每个尺度进行检测
        for scale in self.scales:
            logger.debug(f"处理尺度 {scale}x")
            
            # 缩放图像
            if scale == 1.0:
                scaled_image = image.copy()
            else:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(
                    image, (new_w, new_h), 
                    interpolation=cv2.INTER_CUBIC
                )
            
            # 使用所有可用引擎检测
            scale_results = self._detect_with_all_engines(scaled_image, scale)
            
            # 坐标归一化到原始尺寸
            normalized_results = self._normalize_coordinates(scale_results, scale)
            
            all_results.extend(normalized_results)
        
        # 结果融合和去重
        merged_results = self._merge_multi_scale_results(all_results)
        
        # 分类和评分
        classified_results = self._classify_and_score(merged_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"多尺度检测完成: {len(classified_results)}个结果, 耗时: {elapsed_time:.2f}秒")
        
        # 保存调试信息
        if self.debug_mode:
            self._save_detection_debug(image, classified_results, image_name)
        
        return classified_results
    
    def _detect_with_all_engines(self, image: np.ndarray, scale: float) -> List[Dict]:
        """使用所有可用引擎进行检测"""
        results = []
        
        # PaddleOCR检测
        if 'paddle' in self.engines:
            paddle_results = self._detect_with_paddle(image, scale)
            results.extend(paddle_results)
        
        # EasyOCR检测
        if 'easy' in self.engines:
            easy_results = self._detect_with_easy(image, scale)
            results.extend(easy_results)
        
        # Tesseract检测
        if 'tesseract' in self.engines:
            tesseract_results = self._detect_with_tesseract(image, scale)
            results.extend(tesseract_results)
        
        return results
    
    def _detect_with_paddle(self, image: np.ndarray, scale: float) -> List[Dict]:
        """使用PaddleOCR进行检测"""
        results = []
        
        try:
            # 尝试不同的API调用方式
            try:
                # 新版API
                ocr_results = self.engines['paddle'].predict(image)
            except:
                try:
                    # 旧版API
                    ocr_results = self.engines['paddle'].ocr(image)
                except:
                    # 最简单的调用
                    ocr_results = self.engines['paddle'](image)
            
            # 处理结果
            if ocr_results and len(ocr_results) > 0:
                # 检查结果格式
                if isinstance(ocr_results, dict) and 'rec_texts' in ocr_results:
                    # 新API格式
                    texts = ocr_results['rec_texts']
                    scores = ocr_results.get('rec_scores', [0.5] * len(texts))
                    bboxes = ocr_results.get('rec_polys', [])
                    
                    for i, (text, confidence) in enumerate(zip(texts, scores)):
                        if confidence < 0.1 or not text.strip():
                            continue
                        
                        # 生成一个简单的边界框
                        bbox = bboxes[i] if i < len(bboxes) else [[0, 0], [100, 0], [100, 20], [0, 20]]
                        
                        results.append({
                            'text': text.strip(),
                            'bbox': bbox,
                            'confidence': confidence,
                            'scale': scale,
                            'engine': 'paddle'
                        })
                        
                elif isinstance(ocr_results, list) and ocr_results:
                    # 旧API格式
                    for item in ocr_results:
                        if isinstance(item, list) and len(item) >= 2:
                            bbox, (text, confidence) = item[:2]
                            
                            if confidence < 0.1 or not text.strip():
                                continue
                            
                            results.append({
                                'text': text.strip(),
                                'bbox': bbox,
                                'confidence': confidence,
                                'scale': scale,
                                'engine': 'paddle'
                            })
                
        except Exception as e:
            logger.warning(f"PaddleOCR检测失败 (scale={scale}): {e}")
        
        return results
    
    def _detect_with_easy(self, image: np.ndarray, scale: float) -> List[Dict]:
        """使用EasyOCR进行检测"""
        results = []
        
        try:
            easy_results = self.engines['easy'].readtext(image)
            
            for bbox, text, confidence in easy_results:
                # 过滤低置信度和空文本
                if confidence < 0.1 or not text.strip():
                    continue
                
                # 转换bbox格式为PaddleOCR兼容格式
                bbox_converted = [[bbox[0][0], bbox[0][1]], 
                                [bbox[1][0], bbox[1][1]],
                                [bbox[2][0], bbox[2][1]], 
                                [bbox[3][0], bbox[3][1]]]
                
                results.append({
                    'text': text.strip(),
                    'bbox': bbox_converted,
                    'confidence': confidence,
                    'scale': scale,
                    'engine': 'easy'
                })
                
        except Exception as e:
            logger.warning(f"EasyOCR检测失败 (scale={scale}): {e}")
        
        return results
    
    def _detect_with_tesseract(self, image: np.ndarray, scale: float) -> List[Dict]:
        """使用Tesseract进行检测"""
        results = []
        
        try:
            # 配置Tesseract用于数字和字母识别
            config = self.ocr_configs['tesseract_number']['config']
            
            # 获取详细检测数据
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                # 过滤低置信度和空文本
                if confidence < 30 or not text:
                    continue
                
                # 构建边界框
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                
                results.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence / 100.0,  # 转换为0-1范围
                    'scale': scale,
                    'engine': 'tesseract'
                })
                
        except Exception as e:
            logger.warning(f"Tesseract检测失败 (scale={scale}): {e}")
        
        return results
    
    def _normalize_coordinates(self, results: List[Dict], scale: float) -> List[Dict]:
        """将坐标归一化到原始图像尺寸"""
        if scale == 1.0:
            return results
        
        normalized = []
        for result in results:
            normalized_result = result.copy()
            
            # 缩放坐标
            bbox = result['bbox']
            normalized_bbox = [[x/scale, y/scale] for x, y in bbox]
            normalized_result['bbox'] = normalized_bbox
            
            normalized.append(normalized_result)
        
        return normalized
    
    def _merge_multi_scale_results(self, results: List[Dict]) -> List[Dict]:
        """合并多尺度检测结果，去除重复"""
        if not results:
            return []
        
        # 按照空间位置聚类
        merged = []
        used_indices = set()
        
        for i, result1 in enumerate(results):
            if i in used_indices:
                continue
            
            # 寻找可以合并的检测结果
            merge_group = [result1]
            bbox1 = self._bbox_to_rect(result1['bbox'])
            
            for j, result2 in enumerate(results[i+1:], i+1):
                if j in used_indices:
                    continue
                
                bbox2 = self._bbox_to_rect(result2['bbox'])
                
                # 检查是否应该合并
                if self._should_merge_detections(result1, result2, bbox1, bbox2):
                    merge_group.append(result2)
                    used_indices.add(j)
            
            # 合并组内结果
            merged_result = self._merge_detection_group(merge_group)
            merged.append(merged_result)
            used_indices.add(i)
        
        return merged
    
    def _should_merge_detections(self, det1: Dict, det2: Dict, 
                                bbox1: Tuple, bbox2: Tuple) -> bool:
        """判断两个检测结果是否应该合并"""
        
        # 计算重叠度
        overlap_ratio = self._calculate_overlap_ratio(bbox1, bbox2)
        
        # 高重叠度的结果应该合并
        if overlap_ratio > 0.7:
            return True
        
        # 相邻的数字和字母（路線价模式）
        if overlap_ratio > 0.1:
            text1, text2 = det1['text'], det2['text']
            
            # 数字+字母的组合
            if (text1.isdigit() and text2.isalpha() and len(text2) == 1) or \
               (text2.isdigit() and text1.isalpha() and len(text1) == 1):
                return True
        
        return False
    
    def _merge_detection_group(self, group: List[Dict]) -> Dict:
        """合并一组检测结果"""
        if len(group) == 1:
            return group[0]
        
        # 选择最高置信度的结果作为基础
        base_result = max(group, key=lambda x: x['confidence'])
        
        # 合并文本（处理分离的数字+字母）
        texts = [det['text'] for det in group]
        merged_text = self._merge_texts(texts)
        
        # 计算平均置信度
        avg_confidence = sum(det['confidence'] for det in group) / len(group)
        
        # 合并边界框
        all_bboxes = [det['bbox'] for det in group]
        merged_bbox = self._merge_bboxes(all_bboxes)
        
        # 收集引擎信息
        engines = list(set(det['engine'] for det in group))
        
        return {
            'text': merged_text,
            'bbox': merged_bbox,
            'confidence': avg_confidence,
            'scale': base_result['scale'],
            'engine': '+'.join(engines),
            'merged_count': len(group)
        }
    
    def _merge_texts(self, texts: List[str]) -> str:
        """智能合并文本"""
        if len(texts) == 1:
            return texts[0]
        
        # 去除重复
        unique_texts = list(dict.fromkeys(texts))  # 保持顺序的去重
        
        # 如果只有一个独特文本，返回它
        if len(unique_texts) == 1:
            return unique_texts[0]
        
        # 尝试智能合并（数字+字母）
        digits = [t for t in unique_texts if t.isdigit()]
        letters = [t for t in unique_texts if t.isalpha() and len(t) == 1]
        
        if digits and letters:
            # 组合最长的数字和字母
            longest_digit = max(digits, key=len)
            return longest_digit + letters[0]
        
        # 否则连接所有文本
        return ''.join(unique_texts)
    
    def _merge_bboxes(self, bboxes: List[List]) -> List:
        """合并多个边界框"""
        all_points = []
        for bbox in bboxes:
            all_points.extend(bbox)
        
        # 找到外接矩形
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    
    def _classify_and_score(self, results: List[Dict]) -> List[Dict]:
        """对检测结果进行分类和重新评分"""
        classified = []
        
        for result in results:
            # 添加分类信息
            result['type'] = self._classify_text_type(result['text'])
            
            # 重新评分（基于模式匹配和置信度）
            result['final_score'] = self._calculate_final_score(result)
            
            classified.append(result)
        
        # 按最终得分排序
        classified.sort(key=lambda x: x['final_score'], reverse=True)
        
        return classified
    
    def _classify_text_type(self, text: str) -> str:
        """分类文本类型"""
        # 清理文本
        cleaned_text = text.strip()
        
        # 检查各种模式
        for pattern_name, pattern in self.patterns.items():
            if re.match(pattern, cleaned_text):
                return pattern_name
        
        # 特殊情况处理
        if cleaned_text.isdigit():
            return 'pure_number'
        elif cleaned_text.isalpha():
            return 'pure_letter'
        elif any(char in cleaned_text for char in '-/'):
            return 'complex_identifier'
        
        return 'unknown'
    
    def _calculate_final_score(self, result: Dict) -> float:
        """计算最终评分"""
        base_score = result['confidence']
        
        # 根据类型调整评分
        text_type = result['type']
        if text_type in ['block_number', 'route_price']:
            base_score *= 1.2  # 提高目标类型的评分
        elif text_type == 'unknown':
            base_score *= 0.8  # 降低未知类型的评分
        
        # 根据引擎调整评分
        if 'paddle' in result['engine']:
            base_score *= 1.1  # PaddleOCR通常更准确
        
        # 如果是多引擎合并结果，提高评分
        if '+' in result['engine']:
            base_score *= 1.15
        
        return min(base_score, 1.0)  # 限制在1.0以内
    
    def _bbox_to_rect(self, bbox: List) -> Tuple[int, int, int, int]:
        """将边界框转换为矩形格式 (x, y, w, h)"""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        
        return (x, y, w, h)
    
    def _calculate_overlap_ratio(self, rect1: Tuple, rect2: Tuple) -> float:
        """计算两个矩形的重叠比例"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # 计算重叠区域
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # 计算总面积
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - overlap_area
        
        return overlap_area / union_area if union_area > 0 else 0
    
    def _save_detection_debug(self, image: np.ndarray, results: List[Dict], 
                             image_name: str):
        """保存检测调试信息"""
        if not self.debug_dir:
            return
        
        # 保存检测结果的可视化图像
        debug_image = image.copy()
        
        for i, result in enumerate(results):
            bbox = result['bbox']
            text = result['text']
            confidence = result['final_score']
            text_type = result.get('type', 'unknown')
            
            # 根据类型选择颜色
            color_map = {
                'block_number': (0, 255, 0),    # 绿色
                'route_price': (255, 0, 0),     # 蓝色
                'pure_number': (0, 255, 255),   # 黄色
                'unknown': (128, 128, 128)      # 灰色
            }
            color = color_map.get(text_type, (255, 255, 255))
            
            # 绘制边界框
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(debug_image, [points], True, color, 2)
            
            # 添加文本标签
            label = f"{text}({confidence:.2f})"
            cv2.putText(debug_image, label, (int(bbox[0][0]), int(bbox[0][1]-5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存图像
        debug_path = self.debug_dir / f"{image_name}_detection_result.jpg"
        cv2.imwrite(str(debug_path), debug_image)
        
        # 保存JSON结果
        json_path = self.debug_dir / f"{image_name}_detection_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.debug(f"调试信息已保存: {debug_path}, {json_path}")

# 使用示例
if __name__ == "__main__":
    # 测试多尺度检测器
    detector = MultiScaleOCRDetector(debug_mode=True)
    
    # 假设有测试图像
    test_image_path = "test_image.jpg"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        if image is not None:
            results = detector.detect_multi_scale(image, "test")
            print(f"检测到 {len(results)} 个文本区域")
            for result in results:
                print(f"- {result['text']} ({result['type']}) - 得分: {result['final_score']:.3f}")
        else:
            print("无法读取测试图像")
    else:
        print(f"测试图像不存在: {test_image_path}")