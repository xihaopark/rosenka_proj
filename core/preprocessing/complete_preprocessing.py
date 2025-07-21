#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complete_preprocessing.py
完整的预处理流程 - 集成真正圆圈检测器
"""

import os
import json
import sqlite3
from pathlib import Path
import logging
from typing import List, Dict
from core.detection.real_circle_detector import RealCircleDetector

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePreprocessor:
    """完整的预处理器"""
    
    def __init__(self, data_dir: str = "rosenka_data"):
        self.data_dir = data_dir
        self.detector = RealCircleDetector()
        self.db_path = "rosenka_detection_results.db"
        
    def scan_all_pdfs(self) -> List[Dict]:
        """扫描所有PDF文件，使用真正圆圈检测器"""
        results = []
        pdf_count = 0
        
        logger.info("🔍 开始使用真正圆圈检测器扫描所有PDF文件...")
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdf_count += 1
                    
                    logger.info(f"📄 处理第{pdf_count}个PDF: {pdf_path}")
                    
                    try:
                        # 使用真正圆圈检测器
                        circle_results = self.detector.process_pdf_for_circles(pdf_path)
                        
                        # 解析地址信息
                        path_parts = Path(pdf_path).parts
                        prefecture = ""
                        city = ""
                        district = ""
                        
                        if len(path_parts) >= 3:
                            prefecture = path_parts[-3]
                            city = path_parts[-2]
                            district = path_parts[-1]
                        
                        for result in circle_results:
                            result.update({
                                'pdf_file': pdf_path,
                                'prefecture': prefecture,
                                'city': city,
                                'district': district,
                                'file_name': file
                            })
                            results.append(result)
                        
                        logger.info(f"   🔴 检测到 {len(circle_results)} 个圆圈数字")
                        
                    except Exception as e:
                        logger.error(f"处理PDF失败 {pdf_path}: {e}")
                        continue
        
        logger.info(f"🎯 总计处理 {pdf_count} 个PDF文件，检测到 {len(results)} 个真正的圆圈数字")
        return results
    
    def create_database(self, results: List[Dict]):
        """创建新的数据库"""
        logger.info("💾 创建新的数据库...")
        
        # 删除旧数据库
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            logger.info("🗑️ 删除旧数据库")
        
        # 创建新数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS circle_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT NOT NULL,
                center TEXT NOT NULL,
                radius INTEGER NOT NULL,
                method TEXT NOT NULL,
                pdf_file TEXT NOT NULL,
                prefecture TEXT,
                city TEXT,
                district TEXT,
                file_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 插入数据
        for result in results:
            cursor.execute('''
                INSERT INTO circle_detections 
                (text, confidence, bbox, center, radius, method, pdf_file, prefecture, city, district, file_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['text'],
                result['confidence'],
                json.dumps(result['bbox']),
                json.dumps(result['center']),
                result['radius'],
                result['method'],
                result['pdf_file'],
                result.get('prefecture', ''),
                result.get('city', ''),
                result.get('district', ''),
                result.get('file_name', '')
            ))
        
        conn.commit()
        
        # 统计信息
        cursor.execute("SELECT COUNT(*) FROM circle_detections")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT method, COUNT(*) FROM circle_detections GROUP BY method")
        method_stats = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"✅ 数据库创建完成，共插入 {total_count} 条记录")
        for method, count in method_stats:
            logger.info(f"   - {method}: {count} 条")
    
    def save_results_json(self, results: List[Dict]):
        """保存结果到JSON文件"""
        json_path = "real_circle_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 结果已保存到 {json_path}")
    
    def run_complete_preprocessing(self):
        """运行完整的预处理流程"""
        logger.info("🚀 开始完整预处理流程...")
        
        # 1. 扫描所有PDF
        results = self.scan_all_pdfs()
        
        # 2. 保存JSON结果
        self.save_results_json(results)
        
        # 3. 创建数据库
        self.create_database(results)
        
        logger.info("🎉 完整预处理流程完成！")
        
        # 4. 显示统计信息
        self.show_statistics()
    
    def show_statistics(self):
        """显示统计信息"""
        logger.info("📊 统计信息:")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总数
        cursor.execute("SELECT COUNT(*) FROM circle_detections")
        total = cursor.fetchone()[0]
        logger.info(f"   总检测数: {total}")
        
        # 按方法统计
        cursor.execute("SELECT method, COUNT(*) FROM circle_detections GROUP BY method ORDER BY COUNT(*) DESC")
        methods = cursor.fetchall()
        logger.info("   按检测方法:")
        for method, count in methods:
            logger.info(f"     - {method}: {count}")
        
        # 按地区统计
        cursor.execute("SELECT prefecture, city, COUNT(*) FROM circle_detections GROUP BY prefecture, city ORDER BY COUNT(*) DESC LIMIT 10")
        regions = cursor.fetchall()
        logger.info("   按地区统计 (前10):")
        for prefecture, city, count in regions:
            logger.info(f"     - {prefecture} {city}: {count}")
        
        # 置信度统计
        cursor.execute("SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM circle_detections")
        conf_stats = cursor.fetchone()
        logger.info(f"   置信度统计: 平均={conf_stats[0]:.3f}, 最小={conf_stats[1]:.3f}, 最大={conf_stats[2]:.3f}")
        
        conn.close()

def main():
    """主函数"""
    preprocessor = CompletePreprocessor()
    preprocessor.run_complete_preprocessing()

if __name__ == "__main__":
    main() 