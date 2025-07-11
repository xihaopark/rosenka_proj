#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_local_app.py
极简版本地PDF查询应用
只提供地址输入和本地PDF搜索功能
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import json
import base64
import io
import pickle
import hashlib
from PIL import Image
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
import time

# 导入CV模型处理逻辑
from simple_processor import SimplePDFProcessor, AddressLocation

# ========================= 配置 =========================

st.set_page_config(
    page_title="🗾 路線価図本地查询",
    page_icon="🔍",
    layout="wide"
)

# 基础目录设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rosenka_data")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# 创建必要的目录
os.makedirs(CACHE_DIR, exist_ok=True)

# ========================= 本地文件管理 =========================

class LocalFileManager:
    """本地PDF文件管理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._scan_files()
    
    def _scan_files(self):
        """扫描本地PDF文件"""
        self.available_files = {}
        
        if not os.path.exists(self.data_dir):
            return
        
        # 扫描所有都道府县
        for prefecture in os.listdir(self.data_dir):
            prefecture_path = os.path.join(self.data_dir, prefecture)
            if not os.path.isdir(prefecture_path):
                continue
            
            self.available_files[prefecture] = {}
            
            # 扫描所有市区町村
            for city in os.listdir(prefecture_path):
                city_path = os.path.join(prefecture_path, city)
                if not os.path.isdir(city_path):
                    continue
                
                self.available_files[prefecture][city] = {}
                
                # 扫描所有町丁目
                for district in os.listdir(city_path):
                    district_path = os.path.join(city_path, district)
                    if not os.path.isdir(district_path):
                        continue
                    
                    # 查找PDF文件
                    pdf_files = glob.glob(os.path.join(district_path, "*.pdf"))
                    if pdf_files:
                        self.available_files[prefecture][city][district] = pdf_files
    
    def get_prefectures(self) -> List[str]:
        """获取可用的都道府县列表"""
        return list(self.available_files.keys())
    
    def get_cities(self, prefecture: str) -> List[str]:
        """获取指定都道府县的市区町村列表"""
        return list(self.available_files.get(prefecture, {}).keys())
    
    def get_districts(self, prefecture: str, city: str) -> List[str]:
        """获取指定市区町村的町丁目列表"""
        return list(self.available_files.get(prefecture, {}).get(city, {}).keys())
    
    def get_pdf_files(self, prefecture: str, city: str, district: str) -> List[str]:
        """获取指定区域的PDF文件列表"""
        return self.available_files.get(prefecture, {}).get(city, {}).get(district, [])
    
    def get_all_pdf_files(self) -> List[Tuple[str, str, str, str]]:
        """获取所有PDF文件的信息 (prefecture, city, district, pdf_path)"""
        all_files = []
        for prefecture, cities in self.available_files.items():
            for city, districts in cities.items():
                for district, pdf_files in districts.items():
                    for pdf_file in pdf_files:
                        all_files.append((prefecture, city, district, pdf_file))
        return all_files

# ========================= 地址搜索引擎 =========================

class SimpleAddressSearchEngine:
    """简化版地址搜索引擎"""
    
    def __init__(self):
        self.processor = SimplePDFProcessor(dpi=300)
        self.cache = {}
    
    def search_address(self, query: str, search_scope: str = "all") -> List[Dict]:
        """搜索地址"""
        file_manager = LocalFileManager(DATA_DIR)
        results = []
        
        # 根据搜索范围确定要搜索的文件
        if search_scope == "all":
            pdf_files = file_manager.get_all_pdf_files()
        else:
            # 可以扩展为特定区域搜索
            pdf_files = file_manager.get_all_pdf_files()
        
        # 搜索每个PDF文件
        for prefecture, city, district, pdf_path in pdf_files:
            pdf_results = self._search_in_pdf(
                pdf_path, query, prefecture, city, district
            )
            results.extend(pdf_results)
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:50]  # 返回前50个结果
    
    def _search_in_pdf(self, pdf_path: str, query: str, 
                      prefecture: str, city: str, district: str) -> List[Dict]:
        """在单个PDF中搜索"""
        # 检查缓存
        cache_key = f"{pdf_path}_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 转换PDF为图像
            images = self.processor.pdf_to_images(pdf_path)
            
            # 提取所有地址
            all_addresses = []
            for page_num, image in images.items():
                addresses = self.processor.extract_addresses(
                    image, os.path.basename(pdf_path), page_num,
                    prefecture, city, district
                )
                all_addresses.extend(addresses)
            
            # 搜索匹配的地址
            matches = []
            for addr in all_addresses:
                similarity = self._calculate_similarity(query, addr.text)
                if similarity > 50:  # 相似度阈值
                    matches.append({
                        'text': addr.text,
                        'similarity': similarity,
                        'bbox': addr.bbox,
                        'confidence': addr.confidence,
                        'pdf_path': pdf_path,
                        'page_num': addr.page_num,
                        'prefecture': prefecture,
                        'city': city,
                        'district': district,
                        'method': addr.method,
                        'image': images[addr.page_num]  # 包含图像用于显示
                    })
            
            # 缓存结果
            self.cache[cache_key] = matches
            return matches
            
        except Exception as e:
            st.error(f"处理PDF文件时出错: {pdf_path} - {e}")
            return []
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """计算文本相似度"""
        from rapidfuzz import fuzz
        
        # 精确匹配
        if query in text or text in query:
            return 100
        
        # 模糊匹配
        return fuzz.partial_ratio(query, text)

# ========================= 结果显示组件 =========================

def create_result_map(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     text: str, similarity: float) -> str:
    """创建结果地图"""
    pil_img = Image.fromarray(image)
    
    # 在图像上绘制边界框
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_img)
    
    # 绘制红色边界框
    draw.rectangle(bbox, outline='red', width=3)
    
    # 添加文本标签
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    label = f"{text} ({similarity:.0f}%)"
    draw.text((bbox[0], bbox[1] - 25), label, fill='red', font=font)
    
    # 转换为base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"

# ========================= 主应用 =========================

def main():
    st.title("🗾 路線価図本地查询")
    st.markdown("**极简版本** - 输入地址，即时查找本地PDF文件")
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        st.error(f"❌ 数据目录不存在: {DATA_DIR}")
        st.info("请确保已下载PDF文件到 rosenka_data 目录")
        return
    
    file_manager = LocalFileManager(DATA_DIR)
    
    # 显示可用数据统计
    prefectures = file_manager.get_prefectures()
    if not prefectures:
        st.warning("⚠️ 没有找到任何PDF文件")
        st.info("请先运行下载脚本获取PDF文件")
        return
    
    # 统计信息
    total_files = len(file_manager.get_all_pdf_files())
    st.success(f"✅ 找到 {len(prefectures)} 个都道府县，共 {total_files} 个PDF文件")
    
    # 主要界面
    st.header("🔍 地址搜索")
    
    # 搜索表单
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "输入要搜索的地址或关键词",
            placeholder="例如: 88-7, 川合３丁目, 120E, 六本木",
            help="支持部分匹配，输入地址、番地、价格等任意关键词"
        )
    
    with col2:
        search_scope = st.selectbox(
            "搜索范围",
            ["all", "recent"],
            format_func=lambda x: "全部文件" if x == "all" else "最近使用"
        )
    
    # 搜索按钮和结果
    if query:
        with st.spinner("🔍 搜索中..."):
            search_engine = SimpleAddressSearchEngine()
            results = search_engine.search_address(query, search_scope)
        
        if results:
            st.success(f"🎯 找到 {len(results)} 个匹配结果")
            
            # 按都道府县分组显示结果
            results_by_prefecture = {}
            for result in results:
                prefecture = result['prefecture']
                if prefecture not in results_by_prefecture:
                    results_by_prefecture[prefecture] = []
                results_by_prefecture[prefecture].append(result)
            
            # 显示结果
            for prefecture, prefecture_results in results_by_prefecture.items():
                st.subheader(f"📍 {prefecture} ({len(prefecture_results)}个结果)")
                
                # 显示前5个最佳结果
                top_results = prefecture_results[:5]
                
                for i, result in enumerate(top_results, 1):
                    with st.expander(
                        f"#{i} {result['text']} (相似度: {result['similarity']:.0f}%)",
                        expanded=i == 1  # 只展开第一个结果
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 显示地图
                            img_data = create_result_map(
                                result['image'], 
                                result['bbox'], 
                                result['text'], 
                                result['similarity']
                            )
                            st.image(img_data, use_column_width=True)
                        
                        with col2:
                            # 显示详细信息
                            st.markdown("**📋 详细信息**")
                            st.write(f"**文本:** {result['text']}")
                            st.write(f"**相似度:** {result['similarity']:.1f}%")
                            st.write(f"**置信度:** {result['confidence']:.1f}%")
                            st.write(f"**位置:** {result['prefecture']} > {result['city']} > {result['district']}")
                            st.write(f"**页码:** {result['page_num'] + 1}")
                            st.write(f"**方法:** {result['method']}")
                            st.write(f"**文件:** {os.path.basename(result['pdf_path'])}")
                
                # 如果有更多结果，显示提示
                if len(prefecture_results) > 5:
                    st.info(f"还有 {len(prefecture_results) - 5} 个结果未显示")
        
        else:
            st.warning("😔 没有找到匹配的结果")
            st.info("尝试使用不同的关键词或检查拼写")
    
    # 侧边栏 - 数据概览
    with st.sidebar:
        st.header("📊 数据概览")
        
        # 显示可用的都道府县
        st.subheader("可用都道府县")
        for prefecture in prefectures:
            cities = file_manager.get_cities(prefecture)
            total_districts = sum(
                len(file_manager.get_districts(prefecture, city)) 
                for city in cities
            )
            st.write(f"**{prefecture}** ({len(cities)}市区, {total_districts}町丁目)")
        
        st.divider()
        
        # 使用说明
        st.subheader("💡 使用说明")
        st.markdown("""
        1. **输入关键词**: 地址、番地、价格等
        2. **选择范围**: 全部文件或最近使用
        3. **查看结果**: 点击展开查看详细信息
        
        **支持的搜索类型:**
        - 地址: 六本木１丁目
        - 番地: 88-7
        - 价格: 120E
        - 部分匹配: 川合
        """)

if __name__ == "__main__":
    main() 