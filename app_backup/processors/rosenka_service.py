#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosenka_service.py
路線価図查询服务 - 生产级别的完整服务架构
基于已安装的Python包构建高性能查询服务
"""

import os
import json
import asyncio
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from pathlib import Path
import glob
import sqlite3
from contextlib import contextmanager

# Web框架和API
import streamlit as st
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# 数据处理
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# OCR和文本处理
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

import pytesseract
from rapidfuzz import fuzz, process

# PDF处理
import fitz  # PyMuPDF

# 导入现有的处理器
from simple_processor import SimplePDFProcessor, AddressLocation

# ========================= 配置和常量 =========================

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 基础配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "rosenka_data"
CACHE_DIR = BASE_DIR / ".cache"
DB_PATH = CACHE_DIR / "rosenka_service.db"
INDEX_PATH = CACHE_DIR / "address_index.pkl"

# 创建必要目录
CACHE_DIR.mkdir(exist_ok=True)

# 服务配置
SERVICE_CONFIG = {
    "max_workers": 4,
    "cache_ttl": 3600,  # 1小时
    "max_results": 100,
    "similarity_threshold": 50,
    "ocr_dpi": 300,
    "enable_async": True,
    "enable_cache": True,
    "enable_index": True
}

# ========================= 数据模型 =========================

@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    prefecture: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    similarity_threshold: float = 50
    max_results: int = 50
    use_cache: bool = True

@dataclass
class SearchResult:
    """搜索结果"""
    text: str
    similarity: float
    confidence: float
    bbox: Tuple[int, int, int, int]
    pdf_path: str
    page_num: int
    prefecture: str
    city: str
    district: str
    method: str
    timestamp: datetime
    image_data: Optional[str] = None  # base64编码的图像

@dataclass
class ServiceStats:
    """服务统计"""
    total_pdfs: int
    total_addresses: int
    cache_hits: int
    cache_misses: int
    search_count: int
    avg_response_time: float
    last_updated: datetime

# Pydantic模型用于API
class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    prefecture: Optional[str] = Field(None, description="都道府县")
    city: Optional[str] = Field(None, description="市区町村")
    district: Optional[str] = Field(None, description="町丁目")
    similarity_threshold: float = Field(50, ge=0, le=100, description="相似度阈值")
    max_results: int = Field(50, ge=1, le=200, description="最大结果数")
    use_cache: bool = Field(True, description="是否使用缓存")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    cache_used: bool
    timestamp: datetime

# ========================= 数据库管理 =========================

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    prefecture TEXT NOT NULL,
                    city TEXT NOT NULL,
                    district TEXT NOT NULL,
                    file_size INTEGER,
                    last_modified TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS addresses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_id INTEGER,
                    text TEXT NOT NULL,
                    bbox TEXT,  -- JSON格式的边界框
                    confidence REAL,
                    page_num INTEGER,
                    method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_id) REFERENCES pdf_files (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    results TEXT,  -- JSON格式的结果
                    hit_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_name TEXT UNIQUE NOT NULL,
                    stat_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_addresses_text ON addresses(text)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pdf_location ON pdf_files(prefecture, city, district)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON search_cache(expires_at)")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_pdf_file(self, path: str, prefecture: str, city: str, district: str):
        """添加PDF文件记录"""
        file_path = Path(path)
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pdf_files 
                (path, prefecture, city, district, file_size, last_modified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                prefecture,
                city,
                district,
                file_path.stat().st_size if file_path.exists() else 0,
                datetime.fromtimestamp(file_path.stat().st_mtime) if file_path.exists() else None
            ))
    
    def get_pdf_files(self, prefecture: str = None, city: str = None, district: str = None) -> List[Dict]:
        """获取PDF文件列表"""
        with self.get_connection() as conn:
            query = "SELECT * FROM pdf_files WHERE 1=1"
            params = []
            
            if prefecture:
                query += " AND prefecture = ?"
                params.append(prefecture)
            if city:
                query += " AND city = ?"
                params.append(city)
            if district:
                query += " AND district = ?"
                params.append(district)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def cache_search_result(self, query_hash: str, query_text: str, results: List[Dict], ttl: int = 3600):
        """缓存搜索结果"""
        expires_at = datetime.now() + timedelta(seconds=ttl)
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache 
                (query_hash, query_text, results, expires_at)
                VALUES (?, ?, ?, ?)
            """, (query_hash, query_text, json.dumps(results), expires_at))
    
    def get_cached_result(self, query_hash: str) -> Optional[List[Dict]]:
        """获取缓存的搜索结果"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT results FROM search_cache 
                WHERE query_hash = ? AND expires_at > ?
            """, (query_hash, datetime.now()))
            
            row = cursor.fetchone()
            if row:
                # 更新命中次数
                conn.execute("""
                    UPDATE search_cache SET hit_count = hit_count + 1 
                    WHERE query_hash = ?
                """, (query_hash,))
                return json.loads(row['results'])
        return None
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM search_cache WHERE expires_at < ?", (datetime.now(),))

# ========================= 地址索引管理 =========================

class AddressIndexManager:
    """地址索引管理器"""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = {}
        self.load_index()
    
    def load_index(self):
        """加载索引"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    self.index = pickle.load(f)
                logger.info(f"加载地址索引: {len(self.index)} 条记录")
            except Exception as e:
                logger.error(f"加载索引失败: {e}")
                self.index = {}
    
    def save_index(self):
        """保存索引"""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
            logger.info(f"保存地址索引: {len(self.index)} 条记录")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def add_address(self, text: str, location_info: Dict):
        """添加地址到索引"""
        if text not in self.index:
            self.index[text] = []
        self.index[text].append(location_info)
    
    def search_addresses(self, query: str, threshold: float = 50) -> List[Tuple[str, float, Dict]]:
        """搜索地址"""
        if not self.index:
            return []
        
        # 使用rapidfuzz进行模糊匹配
        matches = process.extract(
            query, 
            self.index.keys(), 
            scorer=fuzz.partial_ratio,
            limit=100
        )
        
        results = []
        for match_text, score, _ in matches:
            if score >= threshold:
                for location_info in self.index[match_text]:
                    results.append((match_text, score, location_info))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

# ========================= OCR引擎管理 =========================

class OCREngineManager:
    """OCR引擎管理器"""
    
    def __init__(self):
        self.engines = {}
        self._init_engines()
    
    def _init_engines(self):
        """初始化OCR引擎"""
        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.engines['paddleocr'] = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan'
                )
                logger.info("PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.engines['easyocr'] = easyocr.Reader(['ja', 'en'])
                logger.info("EasyOCR 初始化成功")
            except Exception as e:
                logger.error(f"EasyOCR 初始化失败: {e}")
        
        # Tesseract
        try:
            # 测试Tesseract是否可用
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = 'available'
            logger.info("Tesseract 可用")
        except Exception as e:
            logger.error(f"Tesseract 不可用: {e}")
    
    def extract_text(self, image: np.ndarray, method: str = 'auto') -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """提取文本"""
        if method == 'auto':
            # 自动选择最佳引擎
            if 'paddleocr' in self.engines:
                method = 'paddleocr'
            elif 'easyocr' in self.engines:
                method = 'easyocr'
            elif 'tesseract' in self.engines:
                method = 'tesseract'
            else:
                raise RuntimeError("没有可用的OCR引擎")
        
        if method == 'paddleocr' and 'paddleocr' in self.engines:
            return self._extract_with_paddleocr(image)
        elif method == 'easyocr' and 'easyocr' in self.engines:
            return self._extract_with_easyocr(image)
        elif method == 'tesseract' and 'tesseract' in self.engines:
            return self._extract_with_tesseract(image)
        else:
            raise RuntimeError(f"OCR引擎不可用: {method}")
    
    def _extract_with_paddleocr(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """使用PaddleOCR提取文本"""
        results = []
        try:
            ocr_results = self.engines['paddleocr'].ocr(image, cls=True)
            
            for line in ocr_results[0] if ocr_results[0] else []:
                bbox_points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # 转换边界框格式
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords)), int(max(y_coords)))
                
                results.append((text, bbox, confidence))
        except Exception as e:
            logger.error(f"PaddleOCR提取失败: {e}")
        
        return results
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """使用EasyOCR提取文本"""
        results = []
        try:
            ocr_results = self.engines['easyocr'].readtext(image)
            
            for bbox_points, text, confidence in ocr_results:
                # 转换边界框格式
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords)), int(max(y_coords)))
                
                results.append((text, bbox, confidence))
        except Exception as e:
            logger.error(f"EasyOCR提取失败: {e}")
        
        return results
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """使用Tesseract提取文本"""
        results = []
        try:
            # 配置Tesseract
            config = r'--oem 3 --psm 6 -l jpn'
            
            # 获取详细信息
            data = pytesseract.image_to_data(
                image, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    confidence = data['conf'][i] / 100.0
                    results.append((text, bbox, confidence))
        except Exception as e:
            logger.error(f"Tesseract提取失败: {e}")
        
        return results

# ========================= 核心搜索服务 =========================

class RosenkaSearchService:
    """路線価図搜索服务"""
    
    def __init__(self):
        self.db_manager = DatabaseManager(DB_PATH)
        self.index_manager = AddressIndexManager(INDEX_PATH)
        self.ocr_manager = OCREngineManager()
        self.processor = SimplePDFProcessor(dpi=SERVICE_CONFIG['ocr_dpi'])
        self.executor = ThreadPoolExecutor(max_workers=SERVICE_CONFIG['max_workers'])
        self.stats = {
            'search_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_response_time': 0.0
        }
        
        # 初始化文件索引
        self._init_file_index()
    
    def _init_file_index(self):
        """初始化文件索引"""
        logger.info("初始化文件索引...")
        
        if not DATA_DIR.exists():
            logger.warning(f"数据目录不存在: {DATA_DIR}")
            return
        
        # 扫描所有PDF文件
        pdf_files = list(DATA_DIR.glob("**/*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_path in pdf_files:
            # 解析路径结构 (prefecture/city/district/file.pdf)
            parts = pdf_path.relative_to(DATA_DIR).parts
            if len(parts) >= 4:
                prefecture, city, district = parts[0], parts[1], parts[2]
                self.db_manager.add_pdf_file(str(pdf_path), prefecture, city, district)
    
    def _generate_query_hash(self, query: SearchQuery) -> str:
        """生成查询哈希"""
        query_str = f"{query.query}_{query.prefecture}_{query.city}_{query.district}_{query.similarity_threshold}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def search_async(self, query: SearchQuery) -> List[SearchResult]:
        """异步搜索"""
        start_time = time.time()
        
        # 检查缓存
        query_hash = self._generate_query_hash(query)
        if query.use_cache and SERVICE_CONFIG['enable_cache']:
            cached_results = self.db_manager.get_cached_result(query_hash)
            if cached_results:
                self.stats['cache_hits'] += 1
                logger.info(f"缓存命中: {query.query}")
                return [SearchResult(**result) for result in cached_results]
        
        self.stats['cache_misses'] += 1
        
        # 执行搜索
        results = await self._perform_search(query)
        
        # 缓存结果
        if query.use_cache and SERVICE_CONFIG['enable_cache']:
            results_dict = [asdict(result) for result in results]
            self.db_manager.cache_search_result(
                query_hash, 
                query.query, 
                results_dict, 
                SERVICE_CONFIG['cache_ttl']
            )
        
        # 更新统计
        response_time = time.time() - start_time
        self.stats['search_count'] += 1
        self.stats['total_response_time'] += response_time
        
        logger.info(f"搜索完成: {query.query}, 用时: {response_time:.2f}s, 结果: {len(results)}")
        
        return results
    
    async def _perform_search(self, query: SearchQuery) -> List[SearchResult]:
        """执行搜索"""
        results = []
        
        # 获取要搜索的PDF文件
        pdf_files = self.db_manager.get_pdf_files(
            query.prefecture, 
            query.city, 
            query.district
        )
        
        if not pdf_files:
            logger.warning(f"没有找到匹配的PDF文件: {query.prefecture}/{query.city}/{query.district}")
            return results
        
        # 限制搜索文件数量（避免过载）
        max_files = min(len(pdf_files), 50)
        pdf_files = pdf_files[:max_files]
        
        # 并发搜索
        tasks = []
        for pdf_info in pdf_files:
            task = asyncio.create_task(
                self._search_in_pdf_async(pdf_info, query)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        pdf_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        for pdf_result in pdf_results:
            if isinstance(pdf_result, list):
                results.extend(pdf_result)
            elif isinstance(pdf_result, Exception):
                logger.error(f"搜索任务失败: {pdf_result}")
        
        # 排序和限制结果
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:query.max_results]
    
    async def _search_in_pdf_async(self, pdf_info: Dict, query: SearchQuery) -> List[SearchResult]:
        """在单个PDF中异步搜索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._search_in_pdf_sync, 
            pdf_info, 
            query
        )
    
    def _search_in_pdf_sync(self, pdf_info: Dict, query: SearchQuery) -> List[SearchResult]:
        """在单个PDF中同步搜索"""
        results = []
        pdf_path = pdf_info['path']
        
        try:
            # 转换PDF为图像
            images = self.processor.pdf_to_images(pdf_path)
            
            for page_num, image in images.items():
                # 使用OCR提取文本
                ocr_results = self.ocr_manager.extract_text(image)
                
                for text, bbox, confidence in ocr_results:
                    # 计算相似度
                    similarity = fuzz.partial_ratio(query.query, text)
                    
                    if similarity >= query.similarity_threshold:
                        result = SearchResult(
                            text=text,
                            similarity=similarity,
                            confidence=confidence,
                            bbox=bbox,
                            pdf_path=pdf_path,
                            page_num=page_num,
                            prefecture=pdf_info['prefecture'],
                            city=pdf_info['city'],
                            district=pdf_info['district'],
                            method='ocr_auto',
                            timestamp=datetime.now()
                        )
                        results.append(result)
        
        except Exception as e:
            logger.error(f"处理PDF失败: {pdf_path} - {e}")
        
        return results
    
    def get_service_stats(self) -> ServiceStats:
        """获取服务统计"""
        pdf_files = self.db_manager.get_pdf_files()
        avg_response_time = (
            self.stats['total_response_time'] / self.stats['search_count']
            if self.stats['search_count'] > 0 else 0
        )
        
        return ServiceStats(
            total_pdfs=len(pdf_files),
            total_addresses=len(self.index_manager.index),
            cache_hits=self.stats['cache_hits'],
            cache_misses=self.stats['cache_misses'],
            search_count=self.stats['search_count'],
            avg_response_time=avg_response_time,
            last_updated=datetime.now()
        )

# ========================= FastAPI应用 =========================

# 创建FastAPI应用
app = FastAPI(
    title="路線価図查询API",
    description="高性能的路線価図地址查询服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
search_service = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    global search_service
    search_service = RosenkaSearchService()
    logger.info("路線価図查询服务启动完成")
    
    yield
    
    # 关闭事件
    if search_service:
        search_service.executor.shutdown(wait=True)
    logger.info("路線価図查询服务关闭完成")

# 更新FastAPI应用配置
app = FastAPI(
    title="路線価図查询API",
    description="高性能的路線価図地址查询服务",
    version="1.0.0",
    lifespan=lifespan
)

def get_search_service() -> RosenkaSearchService:
    """获取搜索服务实例"""
    return search_service

@app.get("/")
async def root():
    """根路径"""
    return {"message": "路線価図查询API服务运行中", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/search", response_model=SearchResponse)
async def search_address(
    request: SearchRequest,
    service: RosenkaSearchService = Depends(get_search_service)
):
    """搜索地址"""
    start_time = time.time()
    
    try:
        # 创建搜索查询
        query = SearchQuery(
            query=request.query,
            prefecture=request.prefecture,
            city=request.city,
            district=request.district,
            similarity_threshold=request.similarity_threshold,
            max_results=request.max_results,
            use_cache=request.use_cache
        )
        
        # 执行搜索
        results = await service.search_async(query)
        
        # 转换结果格式
        results_dict = []
        for result in results:
            result_dict = asdict(result)
            # 移除图像数据（API响应中不包含）
            result_dict.pop('image_data', None)
            results_dict.append(result_dict)
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=results_dict,
            total_count=len(results_dict),
            search_time=search_time,
            cache_used=request.use_cache,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(service: RosenkaSearchService = Depends(get_search_service)):
    """获取服务统计"""
    stats = service.get_service_stats()
    return asdict(stats)

@app.get("/prefectures")
async def get_prefectures(service: RosenkaSearchService = Depends(get_search_service)):
    """获取都道府县列表"""
    pdf_files = service.db_manager.get_pdf_files()
    prefectures = list(set(pdf['prefecture'] for pdf in pdf_files))
    return sorted(prefectures)

@app.get("/cities/{prefecture}")
async def get_cities(
    prefecture: str,
    service: RosenkaSearchService = Depends(get_search_service)
):
    """获取指定都道府县的市区町村列表"""
    pdf_files = service.db_manager.get_pdf_files(prefecture=prefecture)
    cities = list(set(pdf['city'] for pdf in pdf_files))
    return sorted(cities)

@app.get("/districts/{prefecture}/{city}")
async def get_districts(
    prefecture: str,
    city: str,
    service: RosenkaSearchService = Depends(get_search_service)
):
    """获取指定市区町村的町丁目列表"""
    pdf_files = service.db_manager.get_pdf_files(prefecture=prefecture, city=city)
    districts = list(set(pdf['district'] for pdf in pdf_files))
    return sorted(districts)

@app.delete("/cache")
async def clear_cache(service: RosenkaSearchService = Depends(get_search_service)):
    """清理缓存"""
    service.db_manager.cleanup_expired_cache()
    return {"message": "缓存清理完成"}

# ========================= 主函数 =========================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="路線価図查询服务")
    parser.add_argument("--mode", choices=["api", "web"], default="web", help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        # 启动API服务
        uvicorn.run(
            "rosenka_service:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=False
        )
    else:
        # 启动Web界面
        from rosenka_web import run_web_app
        run_web_app(args.host, args.port)

if __name__ == "__main__":
    main() 