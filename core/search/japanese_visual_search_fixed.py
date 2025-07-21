#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japanese_visual_search_fixed.py
日本語可視化検索インターフェース - 修复版
使用Tesseract作为后备OCR，避免NumPy兼容性问题
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
import sys
import logging

# 添加核心模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入Tesseract OCR处理器
try:
    from preprocessing.tesseract_only_ocr import TesseractOnlyOCR, OCRResult
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="路線価図検索システム (修复版)",
    page_icon="🗺️",
    layout="wide"
)

# 日本語対応CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.2rem;
        margin-bottom: 1rem;
        font-family: 'Noto Sans JP', sans-serif;
    }
    .search-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .patch-item {
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 2px solid #ddd;
        text-align: center;
        max-width: 300px;
        margin: 0.5rem;
    }
    .patch-item.circle-priority {
        border-color: #ff6b6b;
        background-color: #fff5f5;
    }
    .patch-item.circle-priority .patch-title {
        color: #ff6b6b;
        font-weight: bold;
    }
    .patch-title {
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .patch-confidence {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .step-indicator {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-weight: bold;
        color: #1976d2;
    }
    .circle-indicator {
        display: inline-block;
        background-color: #ff6b6b;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    .ocr-status {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class OCREngineManager:
    """OCR引擎管理器"""
    
    def __init__(self):
        self.available_engines = []
        self.current_engine = None
        self.ocr_instance = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化可用的OCR引擎"""
        # 检查PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.available_engines.append('paddleocr')
            logger.info("✅ PaddleOCR 可用")
        except Exception as e:
            logger.warning(f"PaddleOCR 不可用: {e}")
        
        # 检查EasyOCR
        try:
            import easyocr
            self.available_engines.append('easyocr')
            logger.info("✅ EasyOCR 可用")
        except Exception as e:
            logger.warning(f"EasyOCR 不可用: {e}")
        
        # 检查Tesseract
        if TESSERACT_AVAILABLE:
            try:
                self.ocr_instance = TesseractOnlyOCR()
                self.available_engines.append('tesseract')
                self.current_engine = 'tesseract'
                logger.info("✅ Tesseract 可用")
            except Exception as e:
                logger.warning(f"Tesseract 不可用: {e}")
        
        if not self.available_engines:
            logger.error("❌ 没有可用的OCR引擎")
    
    def get_status(self) -> Dict:
        """获取OCR引擎状态"""
        return {
            'available_engines': self.available_engines,
            'current_engine': self.current_engine,
            'total_engines': len(self.available_engines)
        }
    
    def extract_text_from_image(self, image: np.ndarray) -> List[OCRResult]:
        """从图像提取文本"""
        if self.current_engine == 'tesseract' and self.ocr_instance:
            return self.ocr_instance.extract_text_detailed(image)
        else:
            # 如果其他引擎可用，可以在这里添加相应的处理
            return []

def calculate_header_height(image_height: int) -> int:
    """計算表頭の高さ（画像の上部15%を表頭として除外）"""
    return int(image_height * 0.15)

def detect_circles_in_image(image: np.ndarray) -> List[Dict]:
    """画像内の円を検出"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 画像の上部15%を除外（表頭部分）
    header_height = calculate_header_height(image.shape[0])
    roi = gray[header_height:, :]
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    
    # HoughCircles で円を検出
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=8,
        maxRadius=40
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Y座標を元の画像座標系に戻す
            actual_y = y + header_height
            detected_circles.append({
                'center': (x, actual_y),
                'radius': r,
                'confidence': 0.8
            })
    
    return detected_circles

def extract_patches_from_image(image: np.ndarray, patch_size: int = 200) -> List[Dict]:
    """画像からパッチを抽出"""
    height, width = image.shape[:2]
    patches = []
    
    # 画像の上部15%を除外
    header_height = calculate_header_height(height)
    effective_height = height - header_height
    
    # パッチのグリッドを計算
    cols = max(1, width // patch_size)
    rows = max(1, effective_height // patch_size)
    
    for row in range(rows):
        for col in range(cols):
            x = col * patch_size
            y = header_height + row * patch_size
            
            # パッチの境界を調整
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            
            patch = image[y:y_end, x:x_end]
            
            patches.append({
                'patch': patch,
                'position': (x, y),
                'size': (x_end - x, y_end - y),
                'grid_pos': (row, col)
            })
    
    return patches

def process_image_with_ocr(image: np.ndarray, ocr_manager: OCREngineManager) -> List[Dict]:
    """OCRを使用して画像を処理"""
    results = []
    
    # 円を検出
    circles = detect_circles_in_image(image)
    
    # パッチを抽出
    patches = extract_patches_from_image(image)
    
    # 各パッチでOCR処理
    for i, patch_info in enumerate(patches):
        patch = patch_info['patch']
        
        # OCRで文字を抽出
        ocr_results = ocr_manager.extract_text_from_image(patch)
        
        # 円との関連性をチェック
        has_circle = False
        for circle in circles:
            circle_x, circle_y = circle['center']
            patch_x, patch_y = patch_info['position']
            patch_w, patch_h = patch_info['size']
            
            # 円がパッチ内にあるかチェック
            if (patch_x <= circle_x <= patch_x + patch_w and 
                patch_y <= circle_y <= patch_y + patch_h):
                has_circle = True
                break
        
        # 結果を格納
        patch_result = {
            'patch_id': i,
            'position': patch_info['position'],
            'size': patch_info['size'],
            'grid_pos': patch_info['grid_pos'],
            'ocr_results': ocr_results,
            'has_circle': has_circle,
            'text_content': ' '.join([result.text for result in ocr_results])
        }
        
        results.append(patch_result)
    
    return results

def search_in_results(results: List[Dict], query: str) -> List[Dict]:
    """検索結果をフィルタリング"""
    if not query:
        return results
    
    matched_results = []
    
    for result in results:
        text_content = result['text_content']
        
        # 完全一致
        if query in text_content:
            result['match_score'] = 1.0
            result['match_type'] = 'exact'
            matched_results.append(result)
            continue
        
        # 部分一致
        similarity = SequenceMatcher(None, query, text_content).ratio()
        if similarity > 0.3:
            result['match_score'] = similarity
            result['match_type'] = 'partial'
            matched_results.append(result)
    
    # スコア順にソート
    matched_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return matched_results

def display_ocr_status(ocr_manager: OCREngineManager):
    """OCR引擎状态显示"""
    status = ocr_manager.get_status()
    
    if status['total_engines'] == 0:
        st.error("❌ 没有可用的OCR引擎")
        st.markdown("""
        **解决方案：**
        1. 运行 `python fix_numpy_compatibility.py` 修复NumPy兼容性
        2. 或者安装Tesseract: `pip install pytesseract`
        """)
        return False
    
    # 显示状态
    status_text = f"🔍 OCR引擎状态: {status['current_engine']} "
    status_text += f"(可用: {', '.join(status['available_engines'])})"
    
    st.markdown(f'<div class="ocr-status">{status_text}</div>', unsafe_allow_html=True)
    
    if status['current_engine'] == 'tesseract':
        st.info("💡 当前使用Tesseract OCR。如需更好的日文识别效果，请修复PaddleOCR/EasyOCR兼容性。")
    
    return True

def main():
    """メイン関数"""
    st.markdown('<h1 class="main-header">🗺️ 路線価図検索システム (修复版)</h1>', unsafe_allow_html=True)
    
    # 初始化OCR管理器
    ocr_manager = OCREngineManager()
    
    # 显示OCR状态
    if not display_ocr_status(ocr_manager):
        return
    
    # 搜索界面
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "📁 路線価図をアップロード",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="PNG、JPG、またはPDFファイルを選択してください"
    )
    
    # 搜索查询
    query = st.text_input(
        "🔍 検索クエリ",
        placeholder="地址或关键词を入力してください...",
        help="例: 東京都渋谷区"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # 显示处理步骤
            st.markdown('<div class="step-indicator">📷 画像を読み込み中...</div>', unsafe_allow_html=True)
            
            # 读取图像
            if uploaded_file.type == "application/pdf":
                # PDF处理
                pdf_bytes = uploaded_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = doc[0]  # 第一页
                
                # 渲染为图像
                mat = fitz.Matrix(2, 2)  # 2x缩放
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # 转换为numpy数组
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                doc.close()
            else:
                # 图像文件
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 显示原始图像
            st.image(image, caption="アップロードされた画像", use_column_width=True)
            
            # 处理图像
            st.markdown('<div class="step-indicator">🔍 OCR処理中...</div>', unsafe_allow_html=True)
            
            with st.spinner("画像を解析しています..."):
                results = process_image_with_ocr(image, ocr_manager)
            
            # 搜索结果
            if query:
                st.markdown('<div class="step-indicator">🎯 検索結果をフィルタリング中...</div>', unsafe_allow_html=True)
                filtered_results = search_in_results(results, query)
            else:
                filtered_results = results
            
            # 显示结果
            st.subheader(f"📋 検索結果 ({len(filtered_results)} 件)")
            
            if filtered_results:
                # 创建网格显示
                cols = st.columns(3)
                
                for i, result in enumerate(filtered_results[:12]):  # 最多显示12个结果
                    col = cols[i % 3]
                    
                    with col:
                        # 提取patch图像
                        patch_x, patch_y = result['position']
                        patch_w, patch_h = result['size']
                        patch_img = image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                        
                        # 创建显示容器
                        circle_class = "circle-priority" if result['has_circle'] else ""
                        
                        st.markdown(f'<div class="patch-item {circle_class}">', unsafe_allow_html=True)
                        
                        # 显示patch图像
                        st.image(patch_img, use_column_width=True)
                        
                        # 显示文本内容
                        text_content = result['text_content']
                        if text_content:
                            st.markdown(f'<div class="patch-title">{text_content}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="patch-title">テキストが検出されませんでした</div>', unsafe_allow_html=True)
                        
                        # 显示匹配信息
                        if 'match_score' in result:
                            score = result['match_score']
                            match_type = result['match_type']
                            st.markdown(f'<div class="patch-confidence">マッチ度: {score:.2f} ({match_type})</div>', unsafe_allow_html=True)
                        
                        # 显示圆形标识
                        if result['has_circle']:
                            st.markdown('<span class="circle-indicator">🔴 円検出</span>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("検索結果が見つかりませんでした。")
                
        except Exception as e:
            st.error(f"処理中にエラーが発生しました: {str(e)}")
            logger.error(f"処理エラー: {e}")
    
    # 使用说明
    with st.expander("📖 使用方法"):
        st.markdown("""
        ### 使用手順
        1. **画像アップロード**: PNG、JPG、またはPDFファイルを選択
        2. **検索クエリ入力**: 探したい地址や关键词を入力
        3. **結果確認**: マッチした部分が表示されます
        
        ### 機能説明
        - **円検出**: 🔴マークが付いた結果は円形の数字が検出されています
        - **OCR処理**: 現在使用中のOCR引擎で文字を認識
        - **マッチ度**: 検索クエリとの一致度を表示
        
        ### 注意事項
        - 現在はTesseract OCRを使用しています
        - より良い日本語認識のためには、PaddleOCRの修復をお勧めします
        """)

if __name__ == "__main__":
    main() 