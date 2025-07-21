#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_preprocessor.py
PDF预处理系统 - 批量解析PDF文件，提取地址和坐标
建立索引数据库供实时查询使用
"""

import os
import sys
import json
import sqlite3
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import hashlib

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app" / "processors"))

import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF
import io

# OCR引擎导入
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from rapidfuzz import fuzz, process
import jieba

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================= 数据结构 =========================

@dataclass
class AddressEntry:
    """地址条目"""
    id: str                          # 唯一ID
    text: str                        # 地址文本
    normalized_text: str             # 标准化地址文本
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    center_point: Tuple[int, int]    # 中心点坐标
    confidence: float                # 置信度
    pdf_path: str                    # PDF文件路径
    page_num: int                    # 页码
    prefecture: str                  # 都道府县
    city: str                        # 市区町村  
    district: str                    # 町丁目
    sub_district: str                # 具体小区域
    ocr_method: str                  # OCR方法
    created_at: str                  # 创建时间

@dataclass
class ProcessingResult:
    """处理结果"""
    pdf_path: str
    total_addresses: int
    processing_time: float
    success: bool
    error_message: str = ""

# ========================= 地址标准化器 =========================

class JapaneseAddressNormalizer:
    """日文地址标准化器"""
    
    def __init__(self):
        # 数字映射
        self.number_map = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '－': '-', '−': '-', '‐': '-', '‑': '-', '‒': '-'
        }
        
        # 地址关键词
        self.address_keywords = [
            '丁目', '番地', '番', '号', '町', '区', '市', '府', '県',
            '丁', '地', '１', '２', '３', '４', '５', '６', '７', '８', '９', '０'
        ]
        
        # 初始化jieba
        jieba.initialize()
    
    def normalize_text(self, text: str) -> str:
        """标准化文本"""
        if not text:
            return ""
        
        # Unicode标准化
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # 转换数字
        for full, half in self.number_map.items():
            text = text.replace(full, half)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_address_components(self, text: str) -> Dict[str, str]:
        """提取地址组件"""
        normalized = self.normalize_text(text)
        components = {}
        
        # 提取数字模式
        import re
        
        # 丁目
        chome_match = re.search(r'(\d+)丁目', normalized)
        if chome_match:
            components['chome'] = chome_match.group(1)
        
        # 番地
        banchi_match = re.search(r'(\d+)番地?', normalized)
        if banchi_match:
            components['banchi'] = banchi_match.group(1)
        
        # 号
        go_match = re.search(r'(\d+)号', normalized)
        if go_match:
            components['go'] = go_match.group(1)
        
        # 连续数字模式 (1-2-3)
        number_pattern = re.search(r'(\d+[-‐]\d+[-‐]\d+)', normalized)
        if number_pattern:
            components['full_number'] = number_pattern.group(1)
        
        return components
    
    def is_valid_address(self, text: str) -> bool:
        """验证是否为有效地址"""
        if not text or len(text.strip()) < 2:
            return False
        
        normalized = self.normalize_text(text)
        
        # 检查是否包含地址关键词
        for keyword in self.address_keywords:
            if keyword in normalized:
                return True
        
        # 检查是否包含数字
        if any(char.isdigit() for char in normalized):
            return True
        
        return False

# ========================= OCR处理器 =========================

class BatchOCRProcessor:
    """批量OCR处理器"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.normalizer = JapaneseAddressNormalizer()
        self._init_ocr_engines()
    
    def _init_ocr_engines(self):
        """初始化OCR引擎"""
        self.ocr_engines = {}
        
        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engines['paddleocr'] = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan',
                    use_gpu=self.use_gpu,
                    show_log=False
                )
                logger.info("✅ PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_engines['easyocr'] = easyocr.Reader(['ja', 'en'], gpu=self.use_gpu, verbose=False)
                logger.info("✅ EasyOCR 初始化成功")
            except Exception as e:
                logger.error(f"EasyOCR 初始化失败: {e}")
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                self.ocr_engines['tesseract'] = 'available'
                logger.info("✅ Tesseract 可用")
            except Exception as e:
                logger.error(f"Tesseract 不可用: {e}")
    
    def process_pdf(self, pdf_path: str, prefecture: str, city: str, district: str) -> List[AddressEntry]:
        """处理单个PDF文件"""
        addresses = []
        
        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 转换为图像
                zoom = 2.0  # 提高分辨率
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # 转换为numpy数组
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                image = np.array(img)
                
                # OCR识别
                page_addresses = self._extract_addresses_from_image(
                    image, pdf_path, page_num, prefecture, city, district
                )
                addresses.extend(page_addresses)
            
            doc.close()
            logger.info(f"处理完成 {pdf_path}: {len(addresses)} 个地址")
            
        except Exception as e:
            logger.error(f"处理PDF失败 {pdf_path}: {e}")
        
        return addresses
    
    def _extract_addresses_from_image(self, image: np.ndarray, pdf_path: str, page_num: int,
                                    prefecture: str, city: str, district: str) -> List[AddressEntry]:
        """从图像中提取地址"""
        addresses = []
        
        # 使用最佳可用的OCR引擎
        if 'paddleocr' in self.ocr_engines:
            addresses.extend(self._extract_with_paddleocr(image, pdf_path, page_num, prefecture, city, district))
        elif 'easyocr' in self.ocr_engines:
            addresses.extend(self._extract_with_easyocr(image, pdf_path, page_num, prefecture, city, district))
        elif 'tesseract' in self.ocr_engines:
            addresses.extend(self._extract_with_tesseract(image, pdf_path, page_num, prefecture, city, district))
        
        return addresses
    
    def _extract_with_paddleocr(self, image: np.ndarray, pdf_path: str, page_num: int,
                               prefecture: str, city: str, district: str) -> List[AddressEntry]:
        """使用PaddleOCR提取地址"""
        addresses = []
        
        try:
            results = self.ocr_engines['paddleocr'].ocr(image, cls=True)
            
            if results and results[0]:
                for line in results[0]:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # 验证是否为有效地址
                    if self.normalizer.is_valid_address(text):
                        # 计算边界框
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        bbox = (int(min(x_coords)), int(min(y_coords)), 
                               int(max(x_coords)), int(max(y_coords)))
                        
                        # 计算中心点
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2
                        
                        # 创建地址条目
                        address_entry = AddressEntry(
                            id=self._generate_id(pdf_path, page_num, text, bbox),
                            text=text,
                            normalized_text=self.normalizer.normalize_text(text),
                            bbox=bbox,
                            center_point=(center_x, center_y),
                            confidence=confidence,
                            pdf_path=pdf_path,
                            page_num=page_num,
                            prefecture=prefecture,
                            city=city,
                            district=district,
                            sub_district=self._extract_sub_district(text),
                            ocr_method='PaddleOCR',
                            created_at=datetime.now().isoformat()
                        )
                        addresses.append(address_entry)
        
        except Exception as e:
            logger.error(f"PaddleOCR处理失败: {e}")
        
        return addresses
    
    def _extract_with_easyocr(self, image: np.ndarray, pdf_path: str, page_num: int,
                             prefecture: str, city: str, district: str) -> List[AddressEntry]:
        """使用EasyOCR提取地址"""
        addresses = []
        
        try:
            results = self.ocr_engines['easyocr'].readtext(image)
            
            for bbox_points, text, confidence in results:
                if self.normalizer.is_valid_address(text):
                    # 计算边界框
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    bbox = (int(min(x_coords)), int(min(y_coords)), 
                           int(max(x_coords)), int(max(y_coords)))
                    
                    # 计算中心点
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    address_entry = AddressEntry(
                        id=self._generate_id(pdf_path, page_num, text, bbox),
                        text=text,
                        normalized_text=self.normalizer.normalize_text(text),
                        bbox=bbox,
                        center_point=(center_x, center_y),
                        confidence=confidence,
                        pdf_path=pdf_path,
                        page_num=page_num,
                        prefecture=prefecture,
                        city=city,
                        district=district,
                        sub_district=self._extract_sub_district(text),
                        ocr_method='EasyOCR',
                        created_at=datetime.now().isoformat()
                    )
                    addresses.append(address_entry)
        
        except Exception as e:
            logger.error(f"EasyOCR处理失败: {e}")
        
        return addresses
    
    def _extract_with_tesseract(self, image: np.ndarray, pdf_path: str, page_num: int,
                               prefecture: str, city: str, district: str) -> List[AddressEntry]:
        """使用Tesseract提取地址"""
        addresses = []
        
        try:
            config = r'--oem 3 --psm 6 -l jpn+eng'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and data['conf'][i] > 30 and self.normalizer.is_valid_address(text):
                    bbox = (data['left'][i], data['top'][i],
                           data['left'][i] + data['width'][i],
                           data['top'][i] + data['height'][i])
                    
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    address_entry = AddressEntry(
                        id=self._generate_id(pdf_path, page_num, text, bbox),
                        text=text,
                        normalized_text=self.normalizer.normalize_text(text),
                        bbox=bbox,
                        center_point=(center_x, center_y),
                        confidence=data['conf'][i] / 100.0,
                        pdf_path=pdf_path,
                        page_num=page_num,
                        prefecture=prefecture,
                        city=city,
                        district=district,
                        sub_district=self._extract_sub_district(text),
                        ocr_method='Tesseract',
                        created_at=datetime.now().isoformat()
                    )
                    addresses.append(address_entry)
        
        except Exception as e:
            logger.error(f"Tesseract处理失败: {e}")
        
        return addresses
    
    def _generate_id(self, pdf_path: str, page_num: int, text: str, bbox: Tuple[int, int, int, int]) -> str:
        """生成唯一ID"""
        content = f"{pdf_path}_{page_num}_{text}_{bbox}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_sub_district(self, text: str) -> str:
        """提取具体小区域信息"""
        components = self.normalizer.extract_address_components(text)
        parts = []
        
        if 'chome' in components:
            parts.append(f"{components['chome']}丁目")
        if 'banchi' in components:
            parts.append(f"{components['banchi']}番")
        if 'go' in components:
            parts.append(f"{components['go']}号")
        if 'full_number' in components:
            parts.append(components['full_number'])
        
        return ''.join(parts) if parts else text

# ========================= 数据库管理器 =========================

class AddressDatabase:
    """地址数据库管理器"""
    
    def __init__(self, db_path: str = "address_index.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS addresses (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    center_x INTEGER,
                    center_y INTEGER,
                    confidence REAL,
                    pdf_path TEXT NOT NULL,
                    page_num INTEGER,
                    prefecture TEXT,
                    city TEXT,
                    district TEXT,
                    sub_district TEXT,
                    ocr_method TEXT,
                    created_at TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_normalized_text ON addresses(normalized_text)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prefecture_city_district ON addresses(prefecture, city, district)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pdf_path ON addresses(pdf_path)")
            
            conn.commit()
    
    def insert_addresses(self, addresses: List[AddressEntry]):
        """批量插入地址"""
        with sqlite3.connect(self.db_path) as conn:
            for addr in addresses:
                conn.execute("""
                    INSERT OR REPLACE INTO addresses VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    addr.id, addr.text, addr.normalized_text,
                    addr.bbox[0], addr.bbox[1], addr.bbox[2], addr.bbox[3],
                    addr.center_point[0], addr.center_point[1],
                    addr.confidence, addr.pdf_path, addr.page_num,
                    addr.prefecture, addr.city, addr.district, addr.sub_district,
                    addr.ocr_method, addr.created_at
                ))
            conn.commit()
    
    def search_addresses(self, query: str, prefecture: str = None, city: str = None, 
                        district: str = None, limit: int = 10) -> List[Dict]:
        """搜索地址"""
        normalizer = JapaneseAddressNormalizer()
        normalized_query = normalizer.normalize_text(query)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 构建查询条件
            conditions = ["normalized_text LIKE ?"]
            params = [f"%{normalized_query}%"]
            
            if prefecture:
                conditions.append("prefecture = ?")
                params.append(prefecture)
            if city:
                conditions.append("city = ?")
                params.append(city)
            if district:
                conditions.append("district = ?")
                params.append(district)
            
            query_sql = f"""
                SELECT * FROM addresses 
                WHERE {' AND '.join(conditions)}
                ORDER BY confidence DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor = conn.execute(query_sql, params)
            results = []
            
            for row in cursor:
                result = {
                    'id': row['id'],
                    'text': row['text'],
                    'normalized_text': row['normalized_text'],
                    'bbox': (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']),
                    'center_point': (row['center_x'], row['center_y']),
                    'confidence': row['confidence'],
                    'pdf_path': row['pdf_path'],
                    'page_num': row['page_num'],
                    'prefecture': row['prefecture'],
                    'city': row['city'],
                    'district': row['district'],
                    'sub_district': row['sub_district'],
                    'ocr_method': row['ocr_method'],
                    'created_at': row['created_at']
                }
                results.append(result)
            
            return results
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM addresses")
            total_addresses = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT pdf_path) FROM addresses")
            total_pdfs = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT prefecture, city, district, COUNT(*) FROM addresses GROUP BY prefecture, city, district")
            area_stats = cursor.fetchall()
            
            return {
                'total_addresses': total_addresses,
                'total_pdfs': total_pdfs,
                'area_statistics': area_stats
            }

# ========================= 主预处理器 =========================

class PDFPreprocessor:
    """PDF预处理器主类"""
    
    def __init__(self, data_dir: str = "rosenka_data", use_gpu: bool = True, max_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.ocr_processor = BatchOCRProcessor(use_gpu=use_gpu)
        self.database = AddressDatabase()
        
    def process_directory(self, target_dir: str = None) -> Dict:
        """处理指定目录下的所有PDF文件"""
        if target_dir:
            process_path = self.data_dir / target_dir
        else:
            process_path = self.data_dir
        
        if not process_path.exists():
            raise ValueError(f"目录不存在: {process_path}")
        
        # 查找所有PDF文件
        pdf_files = list(process_path.rglob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        results = []
        total_addresses = 0
        start_time = time.time()
        
        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for pdf_path in pdf_files:
                # 解析路径获取地理信息
                relative_path = pdf_path.relative_to(self.data_dir)
                path_parts = relative_path.parts
                
                if len(path_parts) >= 3:
                    prefecture = path_parts[0]
                    city = path_parts[1] 
                    district = path_parts[2] if len(path_parts) > 2 else ""
                    
                    future = executor.submit(
                        self._process_single_pdf, 
                        str(pdf_path), prefecture, city, district
                    )
                    futures.append(future)
            
            # 收集结果
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        total_addresses += result.total_addresses
                        logger.info(f"✅ {result.pdf_path}: {result.total_addresses} 个地址")
                    else:
                        logger.error(f"❌ {result.pdf_path}: {result.error_message}")
                except Exception as e:
                    logger.error(f"处理失败: {e}")
        
        processing_time = time.time() - start_time
        
        # 生成处理报告
        report = {
            'total_pdfs': len(pdf_files),
            'successful_pdfs': sum(1 for r in results if r.success),
            'total_addresses': total_addresses,
            'processing_time': processing_time,
            'results': results,
            'database_stats': self.database.get_statistics()
        }
        
        logger.info(f"预处理完成: {report['successful_pdfs']}/{report['total_pdfs']} PDF, {total_addresses} 个地址, 用时 {processing_time:.2f}秒")
        
        return report
    
    def _process_single_pdf(self, pdf_path: str, prefecture: str, city: str, district: str) -> ProcessingResult:
        """处理单个PDF文件"""
        start_time = time.time()
        
        try:
            addresses = self.ocr_processor.process_pdf(pdf_path, prefecture, city, district)
            
            if addresses:
                self.database.insert_addresses(addresses)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                pdf_path=pdf_path,
                total_addresses=len(addresses),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                pdf_path=pdf_path,
                total_addresses=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def search(self, query: str, prefecture: str = None, city: str = None, 
               district: str = None, limit: int = 10) -> List[Dict]:
        """搜索地址"""
        return self.database.search_addresses(query, prefecture, city, district, limit)

# ========================= 命令行接口 =========================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF预处理系统")
    parser.add_argument("--data-dir", default="rosenka_data", help="数据目录")
    parser.add_argument("--target-dir", help="目标子目录")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--workers", type=int, default=4, help="并发工作线程数")
    parser.add_argument("--search", help="搜索地址")
    
    args = parser.parse_args()
    
    preprocessor = PDFPreprocessor(
        data_dir=args.data_dir,
        use_gpu=args.gpu,
        max_workers=args.workers
    )
    
    if args.search:
        # 搜索模式
        results = preprocessor.search(args.search, limit=10)
        print(f"搜索结果 ({len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']} (置信度: {result['confidence']:.2f})")
            print(f"   位置: {result['pdf_path']} 第{result['page_num']+1}页")
            print(f"   坐标: {result['center_point']}")
            print()
    else:
        # 预处理模式
        print("🚀 开始PDF预处理...")
        report = preprocessor.process_directory(args.target_dir)
        
        print("\n📊 处理报告:")
        print(f"总PDF文件: {report['total_pdfs']}")
        print(f"成功处理: {report['successful_pdfs']}")
        print(f"提取地址: {report['total_addresses']}")
        print(f"处理时间: {report['processing_time']:.2f}秒")
        
        # 保存报告
        report_file = f"preprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 报告已保存: {report_file}")

if __name__ == "__main__":
    main() 