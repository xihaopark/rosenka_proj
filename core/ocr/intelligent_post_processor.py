#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Post Processor for Stage 5 Rosenka OCR System
智能后处理器 - 高级结果优化和质量控制

主要功能:
1. 破损文字智能修复 - 修复OCR识别错误
2. 邻近检测结果合并 - 合并分离的数字+字母
3. 异常值检测和过滤 - 移除明显错误的结果
4. 置信度重新评分 - 基于多种因素重新计算可信度
5. 格式标准化 - 统一输出格式
6. 质量控制 - 最终结果验证
"""

import cv2
import numpy as np
import logging
import re
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
import math
from collections import Counter
import difflib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentPostProcessor:
    """Stage 5 智能后处理器"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化智能后处理器
        
        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        self.debug_dir = Path("debug_postprocessing") if debug_mode else None
        
        # 路線価图专用模式和规则
        self.rosenka_patterns = {
            'valid_block_number': r'^\d{1,3}$',  # 街区番号
            'valid_route_price': r'^\d{1,4}[A-G]$',  # 路線価
            'valid_complex_address': r'^\d{1,3}-\d{1,3}[A-G]?$',  # 复合地址
            'valid_reference': r'^[A-Z]\d{1,3}$',  # 参考编号
            'partial_number': r'^\d+$',  # 纯数字
            'partial_letter': r'^[A-G]$',  # 单个字母
        }
        
        # 常见OCR错误映射
        self.error_corrections = {
            # 数字常见错误
            'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5',
            'Z': '2', 'z': '2', 'B': '8', 'g': '9', 'G': '6',
            # 字母常见错误
            '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B', '6': 'G'
        }
        
        # 合并参数
        self.merge_params = {
            'max_merge_distance': 50,  # 最大合并距离
            'max_merge_gap': 30,       # 最大合并间隙
            'text_height_ratio': 0.5,  # 文字高度比例阈值
            'alignment_tolerance': 20   # 对齐容差
        }
        
        # 过滤参数
        self.filter_params = {
            'min_confidence': 0.1,     # 最低置信度
            'min_text_length': 1,      # 最短文本长度
            'max_text_length': 20,     # 最长文本长度
            'noise_patterns': [r'^[\.,:;!\?]+$', r'^[-_=\+]+$'],  # 噪声模式
        }
        
        # 常见有效地址词汇
        self.valid_vocabulary = {
            'numbers': set('0123456789'),
            'letters': set('ABCDEFG'),
            'separators': set('-/'),
            'units': {'万', '千', '円'}
        }
        
        if self.debug_mode and self.debug_dir:
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"调试模式启用，后处理结果保存至: {self.debug_dir}")
    
    def process_detections(self, detections: List[Dict], 
                          image_name: str = "image") -> List[Dict]:
        """
        处理检测结果的主函数
        
        Args:
            detections: 输入的检测结果
            image_name: 图像名称
            
        Returns:
            处理后的检测结果
        """
        logger.info(f"开始智能后处理: {image_name}, 输入{len(detections)}个检测")
        
        if not detections:
            return []
        
        # 1. 预过滤 - 移除明显的噪声
        filtered_detections = self._prefilter_noise(detections)
        logger.debug(f"预过滤后: {len(filtered_detections)}个检测")
        
        # 2. 文字修复 - 修正常见OCR错误
        corrected_detections = self._correct_ocr_errors(filtered_detections)
        logger.debug(f"错误修正后: {len(corrected_detections)}个检测")
        
        # 3. 智能合并 - 合并分离的文字
        merged_detections = self._intelligent_merge(corrected_detections)
        logger.debug(f"智能合并后: {len(merged_detections)}个检测")
        
        # 4. 格式验证和修复
        validated_detections = self._validate_and_fix_formats(merged_detections)
        logger.debug(f"格式验证后: {len(validated_detections)}个检测")
        
        # 5. 重复检测和去重
        deduplicated_detections = self._advanced_deduplication(validated_detections)
        logger.debug(f"去重后: {len(deduplicated_detections)}个检测")
        
        # 6. 最终评分和排序
        final_detections = self._final_scoring_and_ranking(deduplicated_detections)
        logger.debug(f"最终评分后: {len(final_detections)}个检测")
        
        # 7. 质量控制
        quality_controlled = self._quality_control(final_detections)
        
        logger.info(f"智能后处理完成: {len(quality_controlled)}个有效检测")
        
        # 保存调试信息
        if self.debug_mode:
            self._save_processing_debug(detections, quality_controlled, image_name)
        
        return quality_controlled
    
    def _prefilter_noise(self, detections: List[Dict]) -> List[Dict]:
        """预过滤明显的噪声"""
        filtered = []
        
        for detection in detections:
            text = detection.get('text', '').strip()
            confidence = detection.get('confidence', 0)
            
            # 基本过滤条件
            if (len(text) < self.filter_params['min_text_length'] or
                len(text) > self.filter_params['max_text_length'] or
                confidence < self.filter_params['min_confidence']):
                continue
            
            # 噪声模式过滤
            is_noise = False
            for pattern in self.filter_params['noise_patterns']:
                if re.match(pattern, text):
                    is_noise = True
                    break
            
            if is_noise:
                continue
            
            # 空白字符过滤
            if not text or text.isspace():
                continue
            
            filtered.append(detection)
        
        return filtered
    
    def _correct_ocr_errors(self, detections: List[Dict]) -> List[Dict]:
        """修正常见的OCR错误"""
        corrected = []
        
        for detection in detections:
            original_text = detection.get('text', '')
            corrected_text = self._apply_corrections(original_text)
            
            # 如果文本被修正，更新检测结果
            if corrected_text != original_text:
                corrected_detection = detection.copy()
                corrected_detection['text'] = corrected_text
                corrected_detection['original_text'] = original_text
                corrected_detection['was_corrected'] = True
                
                # 稍微降低置信度，因为进行了修正
                if 'confidence' in corrected_detection:
                    corrected_detection['confidence'] *= 0.95
                
                corrected.append(corrected_detection)
            else:
                corrected.append(detection)
        
        return corrected
    
    def _apply_corrections(self, text: str) -> str:
        """应用OCR错误修正"""
        corrected = text
        
        # 基于上下文的智能修正
        corrected = self._context_aware_correction(corrected)
        
        # 字符级修正
        corrected = self._character_level_correction(corrected)
        
        # 模式基修正
        corrected = self._pattern_based_correction(corrected)
        
        return corrected
    
    def _context_aware_correction(self, text: str) -> str:
        """基于上下文的智能修正"""
        # 如果看起来像路線価（数字+字母），确保最后一个字符是有效字母
        if re.match(r'^\d+[A-Za-z]$', text):
            last_char = text[-1].upper()
            if last_char in 'ABCDEFG':
                return text[:-1] + last_char
            elif last_char in self.error_corrections:
                corrected_char = self.error_corrections[last_char]
                if corrected_char in 'ABCDEFG':
                    return text[:-1] + corrected_char
        
        # 如果看起来像纯数字，修正数字错误
        if text.isalnum():
            corrected = ''.join(
                self.error_corrections.get(c, c) if c.isalpha() and not c.isdigit() else c
                for c in text
            )
            # 验证是否为有效数字
            if corrected.isdigit():
                return corrected
        
        return text
    
    def _character_level_correction(self, text: str) -> str:
        """字符级错误修正"""
        corrected_chars = []
        
        for i, char in enumerate(text):
            # 基于位置的修正策略
            if i == len(text) - 1 and char.isalpha():
                # 最后一个字符应该是A-G
                upper_char = char.upper()
                if upper_char in 'ABCDEFG':
                    corrected_chars.append(upper_char)
                elif upper_char in self.error_corrections:
                    corrected = self.error_corrections[upper_char]
                    if corrected in 'ABCDEFG':
                        corrected_chars.append(corrected)
                    else:
                        corrected_chars.append(char)
                else:
                    corrected_chars.append(char)
            elif char.isdigit() or char.isalpha():
                # 应用标准修正
                corrected_chars.append(self.error_corrections.get(char, char))
            else:
                corrected_chars.append(char)
        
        return ''.join(corrected_chars)
    
    def _pattern_based_correction(self, text: str) -> str:
        """基于模式的修正"""
        # 路線価模式修正
        route_price_match = re.match(r'^(\d+)([A-Za-z0-9])$', text)
        if route_price_match:
            number_part = route_price_match.group(1)
            letter_part = route_price_match.group(2).upper()
            
            # 修正字母部分
            if letter_part.isdigit():
                # 数字被误识别为字母
                digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B', '6': 'G'}
                if letter_part in digit_to_letter:
                    letter_part = digit_to_letter[letter_part]
            
            # 确保是有效的借地权字母
            if letter_part in 'ABCDEFG':
                return number_part + letter_part
        
        # 街区番号模式修正（纯数字）
        if re.match(r'^[\dOIlSZBG]+$', text):
            corrected = text
            for wrong, right in self.error_corrections.items():
                if wrong.isalpha() and right.isdigit():
                    corrected = corrected.replace(wrong, right)
            if corrected.isdigit():
                return corrected
        
        return text
    
    def _intelligent_merge(self, detections: List[Dict]) -> List[Dict]:
        """智能合并分离的检测结果"""
        if len(detections) <= 1:
            return detections
        
        merged = []
        used_indices = set()
        
        for i, detection1 in enumerate(detections):
            if i in used_indices:
                continue
            
            # 寻找可以合并的检测
            merge_candidates = [detection1]
            
            for j, detection2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._should_merge_intelligently(detection1, detection2):
                    merge_candidates.append(detection2)
                    used_indices.add(j)
            
            # 合并候选项
            if len(merge_candidates) > 1:
                merged_detection = self._merge_detections_intelligently(merge_candidates)
                merged.append(merged_detection)
            else:
                merged.append(detection1)
            
            used_indices.add(i)
        
        return merged
    
    def _should_merge_intelligently(self, det1: Dict, det2: Dict) -> bool:
        """判断两个检测是否应该智能合并"""
        # 获取边界框信息
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        center1 = self._bbox_center(bbox1)
        center2 = self._bbox_center(bbox2)
        
        # 计算距离
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 距离过远不合并
        if distance > self.merge_params['max_merge_distance']:
            return False
        
        # 检查文本兼容性
        text1 = det1.get('text', '')
        text2 = det2.get('text', '')
        
        # 数字+字母组合（路線価）
        if (text1.isdigit() and text2.isalpha() and len(text2) == 1) or \
           (text2.isdigit() and text1.isalpha() and len(text1) == 1):
            return self._check_alignment(bbox1, bbox2)
        
        # 分离的数字部分
        if text1.isdigit() and text2.isdigit():
            # 检查是否可能是分离的街区番号
            if len(text1) <= 2 and len(text2) <= 2:
                return self._check_horizontal_alignment(bbox1, bbox2)
        
        # 破损的文字修复
        if self._are_fragments_of_same_text(text1, text2):
            return True
        
        return False
    
    def _check_alignment(self, bbox1: List, bbox2: List) -> bool:
        """检查两个边界框是否对齐"""
        # 计算垂直对齐
        y1_min = min(p[1] for p in bbox1)
        y1_max = max(p[1] for p in bbox1)
        y2_min = min(p[1] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        # 检查垂直重叠
        overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        height1 = y1_max - y1_min
        height2 = y2_max - y2_min
        
        # 重叠度应该足够高
        overlap_ratio = overlap / max(height1, height2)
        return overlap_ratio > self.merge_params['text_height_ratio']
    
    def _check_horizontal_alignment(self, bbox1: List, bbox2: List) -> bool:
        """检查水平对齐"""
        x1_min = min(p[0] for p in bbox1)
        x1_max = max(p[0] for p in bbox1)
        x2_min = min(p[0] for p in bbox2)
        x2_max = max(p[0] for p in bbox2)
        
        # 计算水平间隙
        gap = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        return gap <= self.merge_params['max_merge_gap']
    
    def _are_fragments_of_same_text(self, text1: str, text2: str) -> bool:
        """判断是否是同一文字的片段"""
        # 使用序列匹配检查相似性
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity > 0.6 and len(text1) + len(text2) <= 10
    
    def _merge_detections_intelligently(self, detections: List[Dict]) -> Dict:
        """智能合并多个检测结果"""
        if len(detections) == 1:
            return detections[0]
        
        # 按水平位置排序
        sorted_detections = sorted(detections, key=lambda d: min(p[0] for p in d['bbox']))
        
        # 合并文本
        merged_text = self._merge_texts_intelligently([d['text'] for d in sorted_detections])
        
        # 合并边界框
        all_bboxes = [d['bbox'] for d in sorted_detections]
        merged_bbox = self._merge_bboxes(all_bboxes)
        
        # 计算平均置信度
        confidences = [d.get('confidence', 0.5) for d in sorted_detections]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 选择最佳的其他属性
        base_detection = max(sorted_detections, key=lambda d: d.get('confidence', 0))
        
        # 构建合并结果
        merged = base_detection.copy()
        merged.update({
            'text': merged_text,
            'bbox': merged_bbox,
            'confidence': avg_confidence,
            'merged_from': len(detections),
            'merge_source': 'intelligent_merge'
        })
        
        return merged
    
    def _merge_texts_intelligently(self, texts: List[str]) -> str:
        """智能合并文本"""
        if len(texts) == 1:
            return texts[0]
        
        # 去重但保持顺序
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        # 数字+字母合并（路線価）
        digits = [t for t in unique_texts if t.isdigit()]
        letters = [t for t in unique_texts if t.isalpha() and len(t) == 1 and t in 'ABCDEFG']
        
        if digits and letters:
            # 组合最长的数字和第一个字母
            longest_digit = max(digits, key=len)
            return longest_digit + letters[0]
        
        # 数字合并（街区番号分离）
        if all(t.isdigit() for t in unique_texts):
            return ''.join(unique_texts)
        
        # 其他情况直接连接
        return ''.join(unique_texts)
    
    def _merge_bboxes(self, bboxes: List[List]) -> List:
        """合并边界框"""
        all_points = []
        for bbox in bboxes:
            all_points.extend(bbox)
        
        # 计算外接矩形
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    
    def _bbox_center(self, bbox: List) -> Tuple[float, float]:
        """计算边界框中心"""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _validate_and_fix_formats(self, detections: List[Dict]) -> List[Dict]:
        """验证和修复格式"""
        validated = []
        
        for detection in detections:
            text = detection.get('text', '')
            
            # 尝试修复格式
            fixed_text = self._fix_text_format(text)
            
            # 验证格式
            format_type = self._validate_format(fixed_text)
            
            if format_type != 'invalid':
                # 更新检测结果
                updated_detection = detection.copy()
                updated_detection['text'] = fixed_text
                updated_detection['format_type'] = format_type
                
                if fixed_text != text:
                    updated_detection['format_fixed'] = True
                    updated_detection['original_text'] = text
                
                validated.append(updated_detection)
        
        return validated
    
    def _fix_text_format(self, text: str) -> str:
        """修复文本格式"""
        # 移除空白字符
        cleaned = re.sub(r'\s+', '', text)
        
        # 修复常见格式问题
        # 1. 多余的符号
        cleaned = re.sub(r'[^\w\-万千円]', '', cleaned)
        
        # 2. 重复字符
        cleaned = re.sub(r'(.)\1{2,}', r'\1', cleaned)
        
        # 3. 大小写标准化
        if re.match(r'\d+[a-z]$', cleaned):
            cleaned = cleaned[:-1] + cleaned[-1].upper()
        
        return cleaned
    
    def _validate_format(self, text: str) -> str:
        """验证文本格式"""
        for format_name, pattern in self.rosenka_patterns.items():
            if re.match(pattern, text):
                return format_name.replace('valid_', '')
        
        return 'invalid'
    
    def _advanced_deduplication(self, detections: List[Dict]) -> List[Dict]:
        """高级去重处理"""
        if not detections:
            return []
        
        # 按文本内容分组
        text_groups = {}
        for i, detection in enumerate(detections):
            text = detection['text']
            if text not in text_groups:
                text_groups[text] = []
            text_groups[text].append((i, detection))
        
        deduplicated = []
        
        for text, group in text_groups.items():
            if len(group) == 1:
                # 唯一文本，直接保留
                deduplicated.append(group[0][1])
            else:
                # 重复文本，选择最佳的
                best_detection = self._select_best_detection([det for _, det in group])
                deduplicated.append(best_detection)
        
        return deduplicated
    
    def _select_best_detection(self, candidates: List[Dict]) -> Dict:
        """从候选检测中选择最佳的"""
        # 评分函数
        def score_detection(detection):
            score = 0
            
            # 基础置信度
            score += detection.get('confidence', 0) * 100
            
            # 空间分析置信度
            score += detection.get('spatial_confidence', 0) * 50
            
            # 格式匹配奖励
            if detection.get('format_type') in ['block_number', 'route_price']:
                score += 30
            
            # 多引擎合并奖励
            if '+' in detection.get('engine', ''):
                score += 20
            
            # 智能合并奖励
            if detection.get('merge_source') == 'intelligent_merge':
                score += 15
            
            return score
        
        return max(candidates, key=score_detection)
    
    def _final_scoring_and_ranking(self, detections: List[Dict]) -> List[Dict]:
        """最终评分和排序"""
        for detection in detections:
            final_score = self._calculate_comprehensive_score(detection)
            detection['final_comprehensive_score'] = final_score
        
        # 按得分排序
        return sorted(detections, key=lambda d: d['final_comprehensive_score'], reverse=True)
    
    def _calculate_comprehensive_score(self, detection: Dict) -> float:
        """计算综合得分"""
        score = 0.0
        
        # 1. OCR置信度 (40%)
        ocr_confidence = detection.get('confidence', 0.5)
        score += ocr_confidence * 0.4
        
        # 2. 空间分析得分 (30%)
        spatial_confidence = detection.get('spatial_confidence', 0.5)
        score += spatial_confidence * 0.3
        
        # 3. 格式匹配得分 (20%)
        format_type = detection.get('format_type', 'invalid')
        format_score = 0.8 if format_type in ['block_number', 'route_price'] else 0.3
        score += format_score * 0.2
        
        # 4. 处理质量得分 (10%)
        quality_score = 0.5  # 基础得分
        
        # 智能合并奖励
        if detection.get('merge_source') == 'intelligent_merge':
            quality_score += 0.2
        
        # 错误修正奖励
        if detection.get('was_corrected') or detection.get('format_fixed'):
            quality_score += 0.1
        
        # 验证通过奖励
        if detection.get('is_valid', True):
            quality_score += 0.2
        
        score += min(quality_score, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _quality_control(self, detections: List[Dict]) -> List[Dict]:
        """质量控制"""
        quality_controlled = []
        
        for detection in detections:
            # 质量检查
            quality_issues = self._check_quality(detection)
            
            # 添加质量信息
            detection['quality_issues'] = quality_issues
            detection['quality_score'] = 1.0 - len(quality_issues) * 0.2
            
            # 只保留高质量的检测
            if detection['quality_score'] >= 0.3:  # 最多1-2个质量问题
                quality_controlled.append(detection)
        
        return quality_controlled
    
    def _check_quality(self, detection: Dict) -> List[str]:
        """检查检测质量"""
        issues = []
        
        text = detection.get('text', '')
        confidence = detection.get('confidence', 0)
        
        # 1. 置信度检查
        if confidence < 0.3:
            issues.append('low_confidence')
        
        # 2. 文本长度检查
        if len(text) > 15:
            issues.append('text_too_long')
        
        # 3. 格式检查
        if detection.get('format_type') == 'invalid':
            issues.append('invalid_format')
        
        # 4. 空间一致性检查
        if not detection.get('is_valid', True):
            issues.append('spatial_inconsistency')
        
        # 5. 字符合法性检查
        valid_chars = self.valid_vocabulary['numbers'] | self.valid_vocabulary['letters'] | self.valid_vocabulary['separators']
        if any(c not in valid_chars and c not in self.valid_vocabulary['units'] for c in text):
            issues.append('invalid_characters')
        
        return issues
    
    def _save_processing_debug(self, original_detections: List[Dict], 
                              final_detections: List[Dict], 
                              image_name: str):
        """保存处理调试信息"""
        if not self.debug_dir:
            return
        
        debug_info = {
            'image_name': image_name,
            'processing_stages': {
                'original_count': len(original_detections),
                'final_count': len(final_detections),
                'reduction_ratio': 1 - len(final_detections) / len(original_detections) if original_detections else 0
            },
            'original_detections': original_detections,
            'final_detections': final_detections,
            'statistics': {
                'format_distribution': Counter(d.get('format_type', 'unknown') for d in final_detections),
                'average_confidence': sum(d.get('confidence', 0) for d in final_detections) / len(final_detections) if final_detections else 0,
                'average_comprehensive_score': sum(d.get('final_comprehensive_score', 0) for d in final_detections) / len(final_detections) if final_detections else 0
            }
        }
        
        # 保存JSON
        json_path = self.debug_dir / f"{image_name}_postprocessing_debug.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2, default=str)
        
        logger.debug(f"后处理调试信息已保存: {json_path}")

# 使用示例
if __name__ == "__main__":
    # 测试智能后处理器
    processor = IntelligentPostProcessor(debug_mode=True)
    
    # 模拟测试数据
    test_detections = [
        {
            'text': '115',
            'bbox': [[100, 100], [120, 100], [120, 115], [100, 115]],
            'confidence': 0.9,
            'type': 'partial_number'
        },
        {
            'text': 'E',
            'bbox': [[125, 100], [135, 100], [135, 115], [125, 115]],
            'confidence': 0.8,
            'type': 'partial_letter'
        },
        {
            'text': '42',
            'bbox': [[200, 200], [220, 200], [220, 220], [200, 220]],
            'confidence': 0.85,
            'type': 'pure_number'
        }
    ]
    
    result = processor.process_detections(test_detections, "test")
    print(f"后处理完成: {len(result)}个检测")
    for det in result:
        print(f"- {det['text']} (格式: {det.get('format_type', 'unknown')}, 得分: {det.get('final_comprehensive_score', 0):.3f})")