#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paddleocr_enhanced_search.py
PaddleOCR增强搜索应用 - 路線価図検索システム
使用PaddleOCR进行更准确的日语文字识别
"""

import streamlit as st
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
import re
import time
import logging

# 设置页面配置
st.set_page_config(
    page_title="路線価図検索システム - PaddleOCR增强版",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入PaddleOCR引擎
from core.ocr.paddle_ocr_engine import PaddleOCREngine
from core.database.database_manager import DatabaseManager
from core.pdf.pdf_processor import PDFProcessor
from core.detection.circle_detector import CircleDetector
from core.utils.image_utils import enhance_image_for_ocr
from core.utils.logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        font-family: 'Noto Sans JP', sans-serif;
        font-weight: bold;
    }
    .sub-header {
        color: #1B4F72;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e3f2fd;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    .paddleocr-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化组件
@st.cache_resource
def initialize_components():
    """初始化系统组件"""
    try:
        # 初始化PaddleOCR引擎
        ocr_engine = PaddleOCREngine(
            use_gpu=True,
            lang='japan',
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            rec_score_thresh=0.5
        )
        
        # 初始化其他组件
        db_manager = DatabaseManager("route_price_maps.db")
        pdf_processor = PDFProcessor()
        circle_detector = CircleDetector()
        
        return {
            'ocr_engine': ocr_engine,
            'db_manager': db_manager,
            'pdf_processor': pdf_processor,
            'circle_detector': circle_detector
        }
    except Exception as e:
        st.error(f"组件初始化失败: {e}")
        return None

def load_folder_structure():
    """加载文件夹结构"""
    base_path = Path("rosenka_data")
    folders = []
    
    if base_path.exists():
        for prefecture in base_path.iterdir():
            if prefecture.is_dir():
                for city in prefecture.iterdir():
                    if city.is_dir():
                        for district in city.iterdir():
                            if district.is_dir():
                                folders.append({
                                    'path': str(district),
                                    'display_name': f"{prefecture.name} {city.name} {district.name}",
                                    'prefecture': prefecture.name,
                                    'city': city.name,
                                    'district': district.name
                                })
    
    return folders

def search_in_pdf_with_paddleocr(pdf_path: str, query: str, components: Dict) -> List[Dict]:
    """使用PaddleOCR在PDF中搜索"""
    results = []
    
    try:
        # 提取PDF页面图像
        images = components['pdf_processor'].extract_images_from_pdf(pdf_path)
        
        for page_num, image in enumerate(images):
            # 增强图像预处理
            enhanced_image = enhance_image_for_ocr(image)
            
            # 使用PaddleOCR检测文本
            text_regions = components['ocr_engine'].detect_text_regions(enhanced_image)
            
            # 搜索匹配的文本
            for region in text_regions:
                text = region['text']
                confidence = region['confidence']
                
                # 计算相似度
                similarity = SequenceMatcher(None, query.lower(), text.lower()).ratio()
                
                if similarity > 0.3:  # 相似度阈值
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'similarity': similarity,
                        'bbox': region['bbox'],
                        'page': page_num + 1,
                        'pdf_path': pdf_path,
                        'engine': 'PaddleOCR'
                    })
        
    except Exception as e:
        logger.error(f"PDF搜索失败 {pdf_path}: {e}")
    
    return results

def display_search_results(results: List[Dict], query: str):
    """显示搜索结果"""
    if not results:
        st.warning("🔍 未找到匹配的结果")
        return
    
    # 按相似度排序
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    st.subheader(f"📋 搜索结果 ({len(results)} 个)")
    
    # 统计信息
    total_confidence = sum(r['confidence'] for r in results)
    avg_confidence = total_confidence / len(results) if results else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总结果数", len(results))
    with col2:
        st.metric("平均置信度", f"{avg_confidence:.2f}")
    with col3:
        st.metric("最高相似度", f"{results[0]['similarity']:.2f}")
    
    # 显示结果
    for i, result in enumerate(results[:20]):  # 显示前20个结果
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <h4>结果 {i+1}</h4>
                    <p><strong>文本:</strong> {result['text']}</p>
                    <p><strong>PDF:</strong> {Path(result['pdf_path']).name}</p>
                    <p><strong>页面:</strong> {result['page']}</p>
                    <p><strong>相似度:</strong> {result['similarity']:.3f}</p>
                    <span class="paddleocr-badge">PaddleOCR</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # 置信度指示器
                conf = result['confidence']
                if conf >= 0.8:
                    conf_class = "confidence-high"
                elif conf >= 0.6:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(f"""
                <div class="stats-container">
                    <p class="{conf_class}">置信度: {conf:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """主函数"""
    st.markdown('<h1 class="main-header">🗺️ 路線価図検索システム</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🚀 PaddleOCR增强版 - 更准确的日语文字识别</p>', unsafe_allow_html=True)
    
    # 初始化组件
    with st.spinner("🔄 初始化系统组件..."):
        components = initialize_components()
    
    if not components:
        st.error("❌ 系统初始化失败，请检查环境配置")
        return
    
    # 侧边栏配置
    st.sidebar.title("⚙️ 搜索配置")
    
    # 搜索模式选择
    search_mode = st.sidebar.selectbox(
        "搜索模式",
        ["实时搜索", "批量搜索", "高级搜索"],
        help="选择不同的搜索模式"
    )
    
    # OCR引擎信息
    engine_info = components['ocr_engine'].get_engine_info()
    st.sidebar.markdown("### 🔧 OCR引擎信息")
    st.sidebar.json(engine_info)
    
    # 主搜索界面
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # 搜索输入
    query = st.text_input(
        "🔍 输入搜索关键词",
        placeholder="例如: 藤白台, 43012, 大阪府...",
        help="支持地址、编号、地区名称等搜索"
    )
    
    # 搜索选项
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="OCR识别置信度过滤"
        )
    
    with col2:
        similarity_threshold = st.slider(
            "相似度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="文本匹配相似度过滤"
        )
    
    with col3:
        max_results = st.number_input(
            "最大结果数",
            min_value=10,
            max_value=100,
            value=20,
            step=10,
            help="限制显示结果数量"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 搜索按钮
    if st.button("🚀 开始搜索", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("⚠️ 请输入搜索关键词")
            return
        
        # 显示搜索进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔍 正在搜索...")
            
            # 加载文件夹结构
            folders = load_folder_structure()
            
            if not folders:
                st.warning("⚠️ 未找到数据文件夹，请检查rosenka_data目录")
                return
            
            all_results = []
            total_folders = len(folders)
            
            # 搜索每个文件夹
            for i, folder in enumerate(folders):
                progress = (i + 1) / total_folders
                progress_bar.progress(progress)
                status_text.text(f"🔍 搜索中... ({i+1}/{total_folders}) - {folder['display_name']}")
                
                # 搜索PDF文件
                pdf_files = list(Path(folder['path']).glob("*.pdf"))
                
                for pdf_file in pdf_files:
                    results = search_in_pdf_with_paddleocr(
                        str(pdf_file), 
                        query, 
                        components
                    )
                    
                    # 过滤结果
                    filtered_results = [
                        r for r in results 
                        if r['confidence'] >= confidence_threshold 
                        and r['similarity'] >= similarity_threshold
                    ]
                    
                    all_results.extend(filtered_results)
            
            progress_bar.progress(1.0)
            status_text.text("✅ 搜索完成!")
            
            # 显示结果
            if all_results:
                display_search_results(all_results[:max_results], query)
            else:
                st.warning("🔍 未找到匹配的结果")
                
        except Exception as e:
            st.error(f"❌ 搜索过程中发生错误: {e}")
            logger.error(f"搜索错误: {e}")
    
    # 显示系统信息
    with st.expander("📊 系统信息"):
        st.json({
            "OCR引擎": "PaddleOCR v4",
            "语言支持": "日语 (japan)",
            "GPU加速": engine_info.get('gpu_enabled', False),
            "检测阈值": engine_info.get('detection_threshold', 0.3),
            "识别阈值": engine_info.get('recognition_threshold', 0.5)
        })

if __name__ == "__main__":
    main() 