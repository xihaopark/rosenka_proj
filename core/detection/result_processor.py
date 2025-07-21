#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import json
import sqlite3
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultProcessor:
    """结果处理器 - 将检测结果标准化并保存"""
    
    def __init__(self, db_path="rosenka_detection_results.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建检测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id TEXT PRIMARY KEY,
                pdf_path TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                text_content TEXT,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                center_x INTEGER,
                center_y INTEGER,
                confidence REAL,
                detection_type TEXT,
                ocr_method TEXT,
                circle_radius INTEGER,
                patch_info TEXT,
                created_at TEXT,
                UNIQUE(pdf_path, page_num, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_content ON detection_results(text_content)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pdf_path ON detection_results(pdf_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_type ON detection_results(detection_type)')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"数据库初始化完成: {self.db_path}")
    
    def process_fixed_ocr_results(self, json_file="fixed_ocr_results.json", pdf_path="", page_num=0):
        """处理fixed_ocr_processor的结果"""
        if not os.path.exists(json_file):
            self.logger.error(f"结果文件不存在: {json_file}")
            return []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        processed_results = []
        
        for result in results:
            # 生成唯一ID
            result_id = self.generate_result_id(pdf_path, page_num, result)
            
            # 计算中心点
            bbox = result['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            processed_result = {
                'id': result_id,
                'pdf_path': pdf_path,
                'page_num': page_num,
                'text_content': result['text'],
                'bbox': bbox,
                'center': [center_x, center_y],
                'confidence': float(result['confidence']),
                'detection_type': result['type'],
                'ocr_method': result['method'],
                'circle_radius': None,
                'patch_info': None,
                'created_at': datetime.now().isoformat()
            }
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def process_circle_detection_results(self, circle_rois_dir="circle_rois", pdf_path="", page_num=0):
        """处理圆形检测结果"""
        if not os.path.exists(circle_rois_dir):
            self.logger.error(f"圆形ROI目录不存在: {circle_rois_dir}")
            return []
        
        processed_results = []
        
        # 遍历circle_rois目录
        for roi_file in sorted(os.listdir(circle_rois_dir)):
            if roi_file.endswith('_original.jpg'):
                # 提取圆形编号
                circle_num = roi_file.split('_')[1]
                
                # 尝试读取对应的处理后图像
                processed_file = f"circle_{circle_num}_processed.jpg"
                processed_path = os.path.join(circle_rois_dir, processed_file)
                
                if os.path.exists(processed_path):
                    # 使用EasyOCR识别处理后的图像
                    text_content = self.ocr_roi_image(processed_path)
                    
                    # 生成结果记录
                    result_id = f"circle_{pdf_path.replace('/', '_')}_{page_num}_{circle_num}"
                    
                    processed_result = {
                        'id': result_id,
                        'pdf_path': pdf_path,
                        'page_num': page_num,
                        'text_content': text_content,
                        'bbox': None,  # 圆形检测暂时没有精确的bbox
                        'center': None,  # 需要从检测结果中获取
                        'confidence': 0.8,  # 默认置信度
                        'detection_type': 'circle',
                        'ocr_method': 'easyocr',
                        'circle_radius': None,  # 需要从检测结果中获取
                        'patch_info': {
                            'original_roi': os.path.join(circle_rois_dir, roi_file),
                            'processed_roi': processed_path
                        },
                        'created_at': datetime.now().isoformat()
                    }
                    
                    processed_results.append(processed_result)
        
        return processed_results
    
    def ocr_roi_image(self, image_path):
        """对ROI图像进行OCR识别"""
        try:
            import easyocr
            reader = easyocr.Reader(['ja', 'en'])
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # OCR识别
            results = reader.readtext(image)
            
            if not results:
                return ""
            
            # 选择置信度最高的结果
            best_result = max(results, key=lambda x: x[2])
            return best_result[1]
            
        except Exception as e:
            self.logger.error(f"OCR识别失败 {image_path}: {e}")
            return ""
    
    def generate_result_id(self, pdf_path, page_num, result):
        """生成唯一的结果ID"""
        # 使用文件路径、页码和边界框信息生成哈希
        bbox = result.get('bbox', [0, 0, 0, 0])
        text = result.get('text', '')
        
        id_string = f"{pdf_path}_{page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{text}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def save_to_database(self, results):
        """保存结果到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        
        for result in results:
            try:
                # 处理patch_info
                patch_info_json = json.dumps(result.get('patch_info'), ensure_ascii=False) if result.get('patch_info') else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO detection_results 
                    (id, pdf_path, page_num, text_content, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                     center_x, center_y, confidence, detection_type, ocr_method, circle_radius, 
                     patch_info, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['id'],
                    result['pdf_path'],
                    result['page_num'],
                    result['text_content'],
                    result['bbox'][0] if result['bbox'] else None,
                    result['bbox'][1] if result['bbox'] else None,
                    result['bbox'][2] if result['bbox'] else None,
                    result['bbox'][3] if result['bbox'] else None,
                    result['center'][0] if result['center'] else None,
                    result['center'][1] if result['center'] else None,
                    result['confidence'],
                    result['detection_type'],
                    result['ocr_method'],
                    result['circle_radius'],
                    patch_info_json,
                    result['created_at']
                ))
                
                saved_count += 1
                
            except sqlite3.Error as e:
                self.logger.error(f"保存结果失败: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"成功保存 {saved_count} 条结果到数据库")
        return saved_count
    
    def save_to_json(self, results, output_file="standardized_results.json"):
        """保存结果到JSON文件"""
        try:
            # 确保所有数据都是JSON可序列化的
            serializable_results = []
            
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.integer, np.int64)):
                        serializable_result[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        serializable_result[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    else:
                        serializable_result[key] = value
                
                serializable_results.append(serializable_result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果已保存到: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存JSON失败: {e}")
            return False
    
    def create_summary_report(self, results):
        """创建摘要报告"""
        if not results:
            return {}
        
        # 统计信息
        total_count = len(results)
        
        # 按检测类型统计
        type_counts = {}
        for result in results:
            detection_type = result.get('detection_type', 'unknown')
            type_counts[detection_type] = type_counts.get(detection_type, 0) + 1
        
        # 按OCR方法统计
        method_counts = {}
        for result in results:
            ocr_method = result.get('ocr_method', 'unknown')
            method_counts[ocr_method] = method_counts.get(ocr_method, 0) + 1
        
        # 置信度统计
        confidences = [result.get('confidence', 0) for result in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # 文本内容统计
        text_contents = [result.get('text_content', '') for result in results if result.get('text_content')]
        unique_texts = len(set(text_contents))
        
        summary = {
            'total_detections': total_count,
            'detection_types': type_counts,
            'ocr_methods': method_counts,
            'average_confidence': round(avg_confidence, 3),
            'unique_text_count': unique_texts,
            'processing_time': datetime.now().isoformat()
        }
        
        return summary
    
    def search_results(self, query_text, limit=10):
        """搜索结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 模糊搜索
        cursor.execute('''
            SELECT * FROM detection_results 
            WHERE text_content LIKE ? 
            ORDER BY confidence DESC 
            LIMIT ?
        ''', (f'%{query_text}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # 转换为字典格式
        columns = ['id', 'pdf_path', 'page_num', 'text_content', 'bbox_x1', 'bbox_y1', 
                  'bbox_x2', 'bbox_y2', 'center_x', 'center_y', 'confidence', 
                  'detection_type', 'ocr_method', 'circle_radius', 'patch_info', 'created_at']
        
        formatted_results = []
        for row in results:
            result_dict = dict(zip(columns, row))
            # 重构bbox和center
            if all(result_dict[f'bbox_{axis}'] is not None for axis in ['x1', 'y1', 'x2', 'y2']):
                result_dict['bbox'] = [
                    result_dict['bbox_x1'], result_dict['bbox_y1'],
                    result_dict['bbox_x2'], result_dict['bbox_y2']
                ]
            if result_dict['center_x'] is not None and result_dict['center_y'] is not None:
                result_dict['center'] = [result_dict['center_x'], result_dict['center_y']]
            
            formatted_results.append(result_dict)
        
        return formatted_results

def main():
    """主函数"""
    processor = ResultProcessor()
    
    # 处理fixed_ocr_processor的结果
    pdf_path = "rosenka_data/大阪府/吹田市/藤白台１/43012.pdf"
    page_num = 0
    
    print("🔄 处理OCR识别结果...")
    ocr_results = processor.process_fixed_ocr_results("fixed_ocr_results.json", pdf_path, page_num)
    print(f"✅ 处理了 {len(ocr_results)} 条OCR结果")
    
    # 处理圆形检测结果
    print("🔄 处理圆形检测结果...")
    circle_results = processor.process_circle_detection_results("circle_rois", pdf_path, page_num)
    print(f"✅ 处理了 {len(circle_results)} 条圆形检测结果")
    
    # 合并结果
    all_results = ocr_results + circle_results
    print(f"📊 总计 {len(all_results)} 条检测结果")
    
    # 保存到数据库
    print("💾 保存到数据库...")
    saved_count = processor.save_to_database(all_results)
    
    # 保存到JSON
    print("📄 保存到JSON...")
    processor.save_to_json(all_results, "standardized_detection_results.json")
    
    # 创建摘要报告
    print("📈 生成摘要报告...")
    summary = processor.create_summary_report(all_results)
    
    with open("detection_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 处理摘要:")
    print(f"总检测数量: {summary['total_detections']}")
    print(f"检测类型: {summary['detection_types']}")
    print(f"OCR方法: {summary['ocr_methods']}")
    print(f"平均置信度: {summary['average_confidence']}")
    print(f"唯一文本数: {summary['unique_text_count']}")
    
    # 测试搜索功能
    print(f"\n🔍 测试搜索功能:")
    test_queries = ["藤白", "1", "道路"]
    
    for query in test_queries:
        results = processor.search_results(query, limit=5)
        print(f"搜索 '{query}': 找到 {len(results)} 条结果")
        for i, result in enumerate(results[:3]):  # 只显示前3个
            print(f"  {i+1}. {result['text_content']} (置信度: {result['confidence']})")
    
    print(f"\n✅ 所有结果已标准化并保存!")
    print(f"✅ 数据库: {processor.db_path}")
    print(f"✅ JSON文件: standardized_detection_results.json")
    print(f"✅ 摘要报告: detection_summary.json")

if __name__ == "__main__":
    main() 