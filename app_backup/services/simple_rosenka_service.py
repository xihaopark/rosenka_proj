#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_rosenka_service.py
简化版路線価図查询API服务
避免复杂的依赖问题，专注于核心功能
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

# 基础依赖
import streamlit as st
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 数据处理
import numpy as np
import pandas as pd
from PIL import Image

# OCR引擎
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

import pytesseract
from rapidfuzz import fuzz

# PDF处理
import fitz  # PyMuPDF

# 导入现有的处理器
from simple_processor import SimplePDFProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基础配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "rosenka_data"
CACHE_DIR = BASE_DIR / ".cache"

# 创建必要目录
CACHE_DIR.mkdir(exist_ok=True)

# ========================= 数据模型 =========================

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    prefecture: Optional[str] = Field(None, description="都道府县")
    city: Optional[str] = Field(None, description="市区町村")
    district: Optional[str] = Field(None, description="町丁目")
    similarity_threshold: float = Field(50, ge=0, le=100, description="相似度阈值")
    max_results: int = Field(50, ge=1, le=200, description="最大结果数")
    use_cache: bool = Field(True, description="是否使用缓存")

class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    cache_used: bool
    timestamp: datetime

# ========================= 简化OCR引擎 =========================

class SimpleOCREngine:
    """简化OCR引擎"""
    
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
        
        # Tesseract
        try:
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = 'available'
            logger.info("Tesseract 可用")
        except Exception as e:
            logger.error(f"Tesseract 不可用: {e}")
    
    def extract_text(self, image: np.ndarray) -> List[tuple]:
        """提取文本"""
        results = []
        
        # 尝试PaddleOCR
        if 'paddleocr' in self.engines:
            try:
                ocr_results = self.engines['paddleocr'].ocr(image, cls=True)
                if ocr_results and ocr_results[0]:
                    for line in ocr_results[0]:
                        bbox_points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # 转换边界框格式
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        bbox = (int(min(x_coords)), int(min(y_coords)), 
                               int(max(x_coords)), int(max(y_coords)))
                        
                        results.append((text, bbox, confidence))
                return results
            except Exception as e:
                logger.error(f"PaddleOCR提取失败: {e}")
        
        # 尝试Tesseract
        if 'tesseract' in self.engines:
            try:
                config = r'--oem 3 --psm 6 -l jpn'
                data = pytesseract.image_to_data(
                    image, config=config, output_type=pytesseract.Output.DICT
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
                return results
            except Exception as e:
                logger.error(f"Tesseract提取失败: {e}")
        
        return results

# ========================= 简化搜索服务 =========================

class SimpleSearchService:
    """简化搜索服务"""
    
    def __init__(self):
        self.ocr_engine = SimpleOCREngine()
        self.processor = SimplePDFProcessor(dpi=300)
        self.cache = {}
        
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
        
        self.pdf_files = []
        for pdf_path in pdf_files:
            # 解析路径结构: rosenka_data/prefecture/city/district/file.pdf
            relative_path = pdf_path.relative_to(DATA_DIR)
            parts = relative_path.parts
            
            if len(parts) >= 4:  # 至少有4级: 都道府県/市区町村/町丁目/文件.pdf
                prefecture = parts[0]  # 都道府県
                city = parts[1]        # 市区町村
                district = parts[2]    # 町丁目
                filename = parts[-1]   # PDF文件名
                
                self.pdf_files.append({
                    'path': str(pdf_path),
                    'prefecture': prefecture,
                    'city': city,
                    'district': district,
                    'filename': filename,
                    'full_address': f"{prefecture}{city}{district}",
                    'relative_path': str(relative_path)
                })
        
        logger.info(f"索引了 {len(self.pdf_files)} 个PDF文件")
    
    def search(self, query: str, **kwargs) -> List[Dict]:
        """搜索地址 - 逐层精确匹配"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{query}_{hashlib.md5(query.encode()).hexdigest()}"
        if kwargs.get('use_cache', True) and cache_key in self.cache:
            logger.info(f"缓存命中: {query}")
            return self.cache[cache_key]
        
        results = []
        prefecture_filter = kwargs.get('prefecture')
        city_filter = kwargs.get('city')
        district_filter = kwargs.get('district')
        max_results = kwargs.get('max_results', 50)
        
        # 解析查询地址
        parsed_address = self._parse_address(query)
        logger.info(f"解析地址: {parsed_address}")
        
        # 过滤文件
        filtered_files = self.pdf_files
        
        if prefecture_filter:
            filtered_files = [f for f in filtered_files if f['prefecture'] == prefecture_filter]
        
        if city_filter:
            filtered_files = [f for f in filtered_files if f['city'] == city_filter]
        
        if district_filter:
            filtered_files = [f for f in filtered_files if f['district'] == district_filter]
        
        logger.info(f"搜索查询: {query}, 过滤后文件数: {len(filtered_files)}")
        
        # 逐层精确匹配
        for pdf_info in filtered_files:
            match_score = self._calculate_match_score(pdf_info, parsed_address)
            
            if match_score > 0:
                result = {
                    'pdf_path': pdf_info['path'],
                    'prefecture': pdf_info['prefecture'],
                    'city': pdf_info['city'],
                    'district': pdf_info['district'],
                    'filename': pdf_info['filename'],
                    'full_address': pdf_info['full_address'],
                    'similarity': match_score,
                    'method': 'hierarchical_match',
                    'match_type': 'hierarchical',
                    'relative_path': pdf_info['relative_path'],
                    'match_details': self._get_match_details(pdf_info, parsed_address)
                }
                results.append(result)
        
        # 按匹配分数排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:max_results]
        
        # 缓存结果
        if kwargs.get('use_cache', True):
            self.cache[cache_key] = results
        
        search_time = time.time() - start_time
        
        return {
            'results': results,
            'total_count': len(results),
            'search_time': search_time,
            'cache_used': cache_key in self.cache,
            'timestamp': datetime.now()
        }
    
    def _parse_address(self, query: str) -> Dict[str, str]:
        """解析地址查询，提取都道府县、市区町村、町丁目"""
        query = query.strip()
        parsed = {'prefecture': '', 'city': '', 'district': ''}
        
        # 都道府县关键词
        prefecture_keywords = [
            '北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県',
            '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県',
            '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県',
            '岐阜県', '静岡県', '愛知県', '三重県',
            '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県',
            '鳥取県', '島根県', '岡山県', '広島県', '山口県',
            '徳島県', '香川県', '愛媛県', '高知県',
            '福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県'
        ]
        
        # 查找都道府县
        for pref in prefecture_keywords:
            if pref in query:
                parsed['prefecture'] = pref
                break
        
        # 改进的市区町村识别
        # 首先尝试从现有数据中查找匹配的市区町村
        if parsed['prefecture']:
            available_cities = self._get_available_cities(parsed['prefecture'])
            
            # 在查询中查找匹配的市区町村
            for city in available_cities:
                if city in query:
                    parsed['city'] = city
                    break
        
        # 如果没有找到，使用模式匹配
        if not parsed['city']:
            city_patterns = ['市', '区', '町', '村']
            for pattern in city_patterns:
                # 查找包含该模式的词
                words = query.split()
                for word in words:
                    if pattern in word and word != parsed['prefecture']:
                        parsed['city'] = word
                        break
                if parsed['city']:
                    break
        
        # 查找町丁目 (通常包含数字或特定模式)
        district_patterns = ['丁目', '番地', '条', '段']
        for pattern in district_patterns:
            if pattern in query:
                # 提取包含该模式的完整词
                words = query.split()
                for word in words:
                    if pattern in word:
                        parsed['district'] = word
                        break
                if parsed['district']:
                    break
        
        # 如果没有找到明确的町丁目，尝试提取剩余部分
        if not parsed['district']:
            remaining = query
            if parsed['prefecture']:
                remaining = remaining.replace(parsed['prefecture'], '').strip()
            if parsed['city']:
                remaining = remaining.replace(parsed['city'], '').strip()
            if remaining:
                parsed['district'] = remaining
        
        return parsed
    
    def _get_available_cities(self, prefecture: str) -> List[str]:
        """获取指定都道府县的可用市区町村列表"""
        cities = set()
        for pdf_info in self.pdf_files:
            if pdf_info['prefecture'] == prefecture:
                cities.add(pdf_info['city'])
        return sorted(list(cities))
    
    def _calculate_match_score(self, pdf_info: Dict, parsed_address: Dict) -> int:
        """计算匹配分数 (0-100)"""
        score = 0
        
        # 都道府县匹配 (权重40)
        if parsed_address['prefecture'] and pdf_info['prefecture'] == parsed_address['prefecture']:
            score += 40
        elif parsed_address['prefecture'] and parsed_address['prefecture'] in pdf_info['prefecture']:
            score += 30
        
        # 市区町村匹配 (权重35)
        if parsed_address['city'] and pdf_info['city'] == parsed_address['city']:
            score += 35
        elif parsed_address['city'] and parsed_address['city'] in pdf_info['city']:
            score += 25
        
        # 町丁目匹配 (权重25)
        if parsed_address['district'] and pdf_info['district'] == parsed_address['district']:
            score += 25
        elif parsed_address['district'] and parsed_address['district'] in pdf_info['district']:
            score += 15
        
        # 部分匹配奖励
        if parsed_address['prefecture'] and pdf_info['prefecture'].startswith(parsed_address['prefecture']):
            score += 5
        if parsed_address['city'] and pdf_info['city'].startswith(parsed_address['city']):
            score += 5
        if parsed_address['district'] and pdf_info['district'].startswith(parsed_address['district']):
            score += 5
        
        return score
    
    def _get_match_details(self, pdf_info: Dict, parsed_address: Dict) -> Dict:
        """获取匹配详情"""
        details = {
            'prefecture_match': False,
            'city_match': False,
            'district_match': False,
            'exact_matches': 0,
            'partial_matches': 0
        }
        
        exact_matches = 0
        partial_matches = 0
        
        # 检查都道府县
        if parsed_address['prefecture']:
            if pdf_info['prefecture'] == parsed_address['prefecture']:
                details['prefecture_match'] = True
                exact_matches += 1
            elif parsed_address['prefecture'] in pdf_info['prefecture']:
                partial_matches += 1
        
        # 检查市区町村
        if parsed_address['city']:
            if pdf_info['city'] == parsed_address['city']:
                details['city_match'] = True
                exact_matches += 1
            elif parsed_address['city'] in pdf_info['city']:
                partial_matches += 1
        
        # 检查町丁目
        if parsed_address['district']:
            if pdf_info['district'] == parsed_address['district']:
                details['district_match'] = True
                exact_matches += 1
            elif parsed_address['district'] in pdf_info['district']:
                partial_matches += 1
        
        details['exact_matches'] = exact_matches
        details['partial_matches'] = partial_matches
        
        return details
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_pdfs': len(self.pdf_files),
            'total_addresses': 0,  # 简化版本不统计
            'cache_hits': 0,
            'cache_misses': 0,
            'search_count': 0,
            'avg_response_time': 0,
            'last_updated': datetime.now()
        }
    
    def get_prefectures(self) -> List[str]:
        """获取都道府县列表"""
        prefectures = set()
        for pdf_info in self.pdf_files:
            prefectures.add(pdf_info['prefecture'])
        return sorted(list(prefectures))
    
    def get_cities(self, prefecture: str) -> List[str]:
        """获取市区町村列表"""
        cities = set()
        for pdf_info in self.pdf_files:
            if pdf_info['prefecture'] == prefecture:
                cities.add(pdf_info['city'])
        return sorted(list(cities))
    
    def get_districts(self, prefecture: str, city: str) -> List[str]:
        """获取町丁目列表"""
        districts = set()
        for pdf_info in self.pdf_files:
            if pdf_info['prefecture'] == prefecture and pdf_info['city'] == city:
                districts.add(pdf_info['district'])
        return sorted(list(districts))

# ========================= FastAPI应用 =========================

# 创建FastAPI应用
app = FastAPI(
    title="路線価図查询API",
    description="简化版高性能路線価図地址查询服务",
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

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global search_service
    search_service = SimpleSearchService()
    logger.info("路線価図查询服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("路線価図查询服务关闭完成")

def get_search_service() -> SimpleSearchService:
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
async def search_address(request: SearchRequest):
    """搜索地址"""
    try:
        result = search_service.search(
            request.query,
            prefecture=request.prefecture,
            city=request.city,
            district=request.district,
            similarity_threshold=request.similarity_threshold,
            max_results=request.max_results,
            use_cache=request.use_cache
        )
        return SearchResponse(**result)
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return search_service.get_stats()

@app.get("/prefectures")
async def get_prefectures():
    """获取都道府县列表"""
    return search_service.get_prefectures()

@app.get("/cities/{prefecture}")
async def get_cities(prefecture: str):
    """获取市区町村列表"""
    return search_service.get_cities(prefecture)

@app.get("/districts/{prefecture}/{city}")
async def get_districts(prefecture: str, city: str):
    """获取町丁目列表"""
    return search_service.get_districts(prefecture, city)

@app.delete("/cache")
async def clear_cache():
    """清理缓存"""
    search_service.cache.clear()
    return {"message": "缓存清理完成"}

# ========================= 主函数 =========================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="路線価図查询服务")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    
    args = parser.parse_args()
    
    print("🗾 路線価図查询API服务启动中...")
    print(f"📡 服务地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    print("💡 按 Ctrl+C 停止服务")
    
    uvicorn.run(
        "simple_rosenka_service:app",
        host=args.host,
        port=args.port,
        reload=True
    )

if __name__ == "__main__":
    main() 