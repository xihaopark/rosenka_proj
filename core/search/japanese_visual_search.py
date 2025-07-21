#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japanese_visual_search.py
日本語可視化検索インターフェース - 最適化版
"""

import streamlit as st
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
import re

# ページ設定
st.set_page_config(
    page_title="路線価図検索システム",
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
</style>
""", unsafe_allow_html=True)

def calculate_header_height(image_height: int) -> int:
    """
    計算表頭の高さ（画像の上部15%を表頭として除外）
    
    Args:
        image_height: 画像の高さ
        
    Returns:
        表頭の高さ（ピクセル）
    """
    return int(image_height * 0.15)

def is_numeric_query(query: str) -> bool:
    """
    クエリが数字かどうかを判定
    
    Args:
        query: 検索クエリ
        
    Returns:
        数字クエリの場合True
    """
    # 数字のみ、または数字+記号の組み合わせかチェック
    return bool(re.match(r'^[\d①②③④⑤⑥⑦⑧⑨⑩]+$', query.strip()))

@st.cache_data
def load_folder_structure():
    """フォルダ構造を読み込み"""
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

@st.cache_data
def load_ocr_data():
    """OCRデータを読み込み（最適化版）"""
    data = []
    
    # 複数のJSONファイルから読み込み
    json_files = [
        "fixed_ocr_results.json",
        "circle_detection_results.json",
        "enhanced_circle_detection_results.json"
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # 空ファイルまたは不完全なJSONファイルをスキップ
                    if not content or content == '[' or content == '{':
                        st.warning(f"⚠️ {json_file} は空または不完全です - スキップします")
                        continue
                    
                    file_data = json.loads(content)
                    
                    # リストでない場合はスキップ
                    if not isinstance(file_data, list):
                        st.warning(f"⚠️ {json_file} の形式が正しくありません - スキップします")
                        continue
                    
                # データを標準化
                for item in file_data:
                    if 'text' in item or 'inner_text' in item:
                        text = item.get('text', item.get('inner_text', ''))
                        if text and text.strip():
                            data.append({
                                'text': text,
                                'bbox': item.get('bbox', [0, 0, 0, 0]),
                                'confidence': item.get('confidence', item.get('ocr_confidence', 0.8)),
                                'detection_type': item.get('type', item.get('detection_type', 'text')),
                                'pdf_path': item.get('pdf_path', 'rosenka_data/大阪府/吹田市/藤白台１/43012.pdf'),
                                'source_file': json_file
                            })
                            
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ {json_file} のJSONパースエラー: {str(e)} - スキップします")
                continue
            except Exception as e:
                st.warning(f"⚠️ {json_file} 読み込みエラー: {str(e)} - スキップします")
                continue
    
    return data

def find_matching_folders(query: str, folders: List[Dict]) -> List[Dict]:
    """マッチするフォルダを検索（改良版モジュラー検索）"""
    if not query.strip():
        return []
    
    matches = []
    query_lower = query.lower()
    
    for folder in folders:
        similarity_scores = []
        
        # 1. 完全一致チェック
        if query_lower == folder['display_name'].lower():
            similarity_scores.append(1.0)
        
        # 2. 部分一致チェック
        if query_lower in folder['display_name'].lower():
            # 部分一致の位置による重み付け
            position = folder['display_name'].lower().find(query_lower)
            length_ratio = len(query_lower) / len(folder['display_name'])
            position_bonus = 1.0 - (position / len(folder['display_name']))
            similarity_scores.append(0.7 + 0.3 * position_bonus + 0.2 * length_ratio)
        
        # 3. 各構成要素での一致チェック
        parts = [folder['prefecture'], folder['city'], folder['district']]
        for part in parts:
            part_lower = part.lower()
            
            # 完全一致
            if query_lower == part_lower:
                similarity_scores.append(0.9)
            
            # 部分一致
            elif query_lower in part_lower:
                length_ratio = len(query_lower) / len(part)
                similarity_scores.append(0.6 + 0.3 * length_ratio)
            
            # 前方一致（特に重要）
            elif part_lower.startswith(query_lower):
                length_ratio = len(query_lower) / len(part)
                similarity_scores.append(0.8 + 0.2 * length_ratio)
        
        # 4. 文字単位での類似度チェック（日本語対応）
        display_similarity = SequenceMatcher(None, query_lower, folder['display_name'].lower()).ratio()
        if display_similarity > 0.3:
            similarity_scores.append(display_similarity * 0.5)
        
        # 5. 個別の部分での類似度チェック
        for part in parts:
            part_similarity = SequenceMatcher(None, query_lower, part.lower()).ratio()
            if part_similarity > 0.5:
                similarity_scores.append(part_similarity * 0.6)
        
        # 6. 数字を除外した比較（"藤白台" vs "藤白台１"）
        import re
        query_no_num = re.sub(r'[0-9０-９一二三四五六七八九十①②③④⑤⑥⑦⑧⑨⑩]', '', query_lower)
        if query_no_num:
            for part in parts:
                part_no_num = re.sub(r'[0-9０-９一二三四五六七八九十①②③④⑤⑥⑦⑧⑨⑩]', '', part.lower())
                if query_no_num == part_no_num:
                    similarity_scores.append(0.85)  # 数字を除いて完全一致
                elif query_no_num in part_no_num:
                    similarity_scores.append(0.7)   # 数字を除いて部分一致
        
        # 最高スコアを採用
        if similarity_scores:
            max_similarity = max(similarity_scores)
            if max_similarity > 0.3:  # 閾値を下げて、より多くの結果を含める
                matches.append({**folder, 'similarity': max_similarity})
    
    # 重複を除去し、類似度でソート
    unique_matches = {}
    for match in matches:
        path = match['path']
        if path not in unique_matches or match['similarity'] > unique_matches[path]['similarity']:
            unique_matches[path] = match
    
    return sorted(unique_matches.values(), key=lambda x: x['similarity'], reverse=True)[:5]  # 最大5件

def is_strict_circle_detection(item: Dict) -> bool:
    """严格判断是否为真正的圆圈检测"""
    # 检查detection_type字段
    if item.get('detection_type') != 'circle' and item.get('type') != 'circle':
        return False
    
    # 检查文本内容：真正的圆圈内通常是简单的数字或符号
    text = item.get('text', '').strip()
    if not text:
        return False
    
    # 严格标准：
    # 1. 纯数字（1-4位）
    if text.isdigit() and 1 <= len(text) <= 4:
        return True
    
    # 2. 带圆圈的数字符号
    circle_numbers = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
    if text in circle_numbers:
        return True
    
    # 3. 简单的日文字符（1-2个字符）
    if len(text) <= 2 and any(ord(c) > 127 for c in text):
        return True
    
    # 4. 检查bbox尺寸：真正的圆圈通常比较小
    bbox = item.get('bbox', [])
    if len(bbox) == 4:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # 圆圈通常是小区域（面积 < 3000像素）
        if area > 3000:
            return False
        
        # 圆圈通常接近正方形
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3:  # 宽高比不应该太极端
                return False
    
    return False

def create_large_patch_with_context(pdf_path: str, center_bbox: List[int], expand_factor: float = 4.0) -> Tuple[np.ndarray | None, Dict]:
    """创建带上下文的大patch，包含周围区域"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # 高解像度変換
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)  # type: ignore
        img_data = pix.tobytes("png")
        
        # OpenCV形式に変換
        nparr = np.frombuffer(img_data, np.uint8)
        full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        # 表頭部分を除去
        header_height = calculate_header_height(full_image.shape[0])
        image = full_image[header_height:, :]
        
        if len(center_bbox) == 4:
            x1, y1, x2, y2 = center_bbox
            
            # 座標を表頭除去後の画像に調整
            y1 = max(0, y1 - header_height)
            y2 = max(0, y2 - header_height)
            
            # 中心点计算
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 大幅扩展区域
            width, height = x2 - x1, y2 - y1
            expanded_w = int(max(width * expand_factor, 400))
            expanded_h = int(max(height * expand_factor, 400))
            
            # 计算新的边界
            new_x1 = max(0, center_x - expanded_w // 2)
            new_y1 = max(0, center_y - expanded_h // 2)
            new_x2 = min(image.shape[1], center_x + expanded_w // 2)
            new_y2 = min(image.shape[0], center_y + expanded_h // 2)
            
            # 提取大patch
            large_patch = image[new_y1:new_y2, new_x1:new_x2]
            
            # 在大patch上标记原始检测框
            relative_x1 = max(0, x1 - new_x1)
            relative_y1 = max(0, y1 - new_y1)
            relative_x2 = min(large_patch.shape[1], x2 - new_x1)
            relative_y2 = min(large_patch.shape[0], y2 - new_y1)
            
            # 绘制红色检测框
            cv2.rectangle(large_patch, (relative_x1, relative_y1), (relative_x2, relative_y2), (0, 0, 255), 3)
            
            # 添加PDF信息
            pdf_info = {
                'pdf_name': os.path.basename(pdf_path),
                'page_number': 1,
                'original_position': f"({x1}, {y1}) - ({x2}, {y2})",
                'patch_size': f"{large_patch.shape[1]} x {large_patch.shape[0]}",
                'expansion_factor': expand_factor
            }
            
            return large_patch, pdf_info
        
        return None, {}
        
    except Exception as e:
        st.error(f"大patch作成に失敗しました: {e}")
        return None, {}

def search_in_ocr_data(query: str, data: List[Dict], circle_only: bool = False) -> List[Dict]:
    """OCRデータから検索（严格圆圈判定版）"""
    if not query.strip():
        return []
    
    results = []
    query_lower = query.lower()
    is_numeric = is_numeric_query(query)
    
    # 使用严格的圆圈判定标准
    strict_circle_results = []
    other_results = []
    
    for item in data:
        if query_lower in item['text'].lower():
            if is_strict_circle_detection(item):
                strict_circle_results.append(item)
            else:
                other_results.append(item)
    
    # 圆圈专用模式：只返回严格的圆圈检测结果
    if circle_only:
        results = sorted(strict_circle_results, key=lambda x: x['confidence'], reverse=True)
        return results[:4]
    
    # 如果是数字查询，优先显示严格的圆圈内数字
    if is_numeric and strict_circle_results:
        results = sorted(strict_circle_results, key=lambda x: x['confidence'], reverse=True)[:2]
        # 补充一些其他结果
        other_sorted = sorted(other_results, key=lambda x: x['confidence'], reverse=True)[:2]
        results.extend(other_sorted)
    else:
        # 普通查询，严格圆圈结果优先
        all_results = strict_circle_results + other_results
        results = sorted(all_results, key=lambda x: (
            is_strict_circle_detection(x),  # 严格圆圈结果优先
            x['confidence']
        ), reverse=True)
    
    return results[:4]

def extract_patch_from_pdf(pdf_path: str, bbox: List[int], expand_factor: float = 2.0) -> np.ndarray | None:
    """PDFから局部パッチを抽出（表頭除去版）"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # 高解像度変換
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)  # type: ignore
        img_data = pix.tobytes("png")
        
        # OpenCV形式に変換
        nparr = np.frombuffer(img_data, np.uint8)
        full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        # 表頭部分を除去
        header_height = calculate_header_height(full_image.shape[0])
        image = full_image[header_height:, :]  # 表頭を除去した画像
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 座標を表頭除去後の画像に調整
            y1 = max(0, y1 - header_height)
            y2 = max(0, y2 - header_height)
            
            # 領域を拡張
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = x2 - x1, y2 - y1
            
            expanded_w = int(max(width * expand_factor, 200))
            expanded_h = int(max(height * expand_factor, 200))
            
            new_x1 = max(0, center_x - expanded_w // 2)
            new_y1 = max(0, center_y - expanded_h // 2)
            new_x2 = min(image.shape[1], center_x + expanded_w // 2)
            new_y2 = min(image.shape[0], center_y + expanded_h // 2)
            
            # パッチを抽出
            patch = image[new_y1:new_y2, new_x1:new_x2]
            
            # 検出ボックスを描画
            relative_x1 = max(0, x1 - new_x1)
            relative_y1 = max(0, y1 - new_y1)
            relative_x2 = min(patch.shape[1], x2 - new_x1)
            relative_y2 = min(patch.shape[0], y2 - new_y1)
            
            # 緑色の検出ボックス
            cv2.rectangle(patch, (relative_x1, relative_y1), (relative_x2, relative_y2), (0, 255, 0), 3)
            
            return patch
        
        return None
        
    except Exception as e:
        st.error(f"パッチ抽出に失敗しました: {e}")
        return None

def image_to_base64(image: np.ndarray) -> str:
    """画像をBase64エンコード"""
    if image is None:
        return ""
    
    try:
        # OpenCV画像をPIL画像に変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        st.error(f"画像エンコードエラー: {e}")
        return ""

def main():
    """メイン関数（強化版ステップ管理）"""
    st.markdown('<h1 class="main-header">🗺️ 路線価図検索システム</h1>', unsafe_allow_html=True)
    
    # セッション状態の強化初期化
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    if 'address_query' not in st.session_state:
        st.session_state.address_query = ""
    if 'matching_folders' not in st.session_state:
        st.session_state.matching_folders = []
    if 'step_transition_flag' not in st.session_state:
        st.session_state.step_transition_flag = False
    if 'circle_only_mode' not in st.session_state:
        st.session_state.circle_only_mode = False
    
    # データの読み込み
    folders = load_folder_structure()
    ocr_data = load_ocr_data()
    
    # デバッグ情報表示
    st.sidebar.write(f"🔍 現在のステップ: {st.session_state.current_step}")
    st.sidebar.write(f"📁 選択されたフォルダ: {st.session_state.selected_folder is not None}")
    if st.session_state.selected_folder:
        st.sidebar.write(f"📂 フォルダ名: {st.session_state.selected_folder.get('display_name', 'Unknown')}")
    st.sidebar.write(f"🔴 圆圈专用モード: {'ON' if st.session_state.circle_only_mode else 'OFF'}")
    
    # 強制的にステップ2に移行するボタン（デバッグ用）
    if st.sidebar.button("🚨 強制的にステップ2へ（デバッグ用）"):
        if not st.session_state.selected_folder:
            # デフォルトフォルダを設定
            st.session_state.selected_folder = {
                'display_name': '大阪府 吹田市 藤白台１',
                'path': 'rosenka_data/大阪府/吹田市/藤白台１'
            }
        st.session_state.current_step = 2
        st.rerun()
    
    # ステップ1: 住所入力
    if st.session_state.current_step == 1:
        st.markdown('<div class="step-indicator">ステップ1: 住所を入力してください</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            
            address_input = st.text_input(
                "住所を入力:",
                value=st.session_state.address_query,
                placeholder="例: 大阪府吹田市藤白台一丁目",
                key="address_input"
            )
            
            # 検索ボタン
            if st.button("📁 フォルダ検索", type="primary", key="search_folders"):
                if address_input.strip():
                    st.session_state.address_query = address_input
                    matching_folders = find_matching_folders(address_input, folders)
                    st.session_state.matching_folders = matching_folders
                    
                    if matching_folders:
                        st.success(f"✅ {len(matching_folders)} 件のフォルダが見つかりました")
                    else:
                        st.warning("⚠️ 該当するフォルダが見つかりませんでした")
                else:
                    st.warning("⚠️ 住所を入力してください")
            
            # フォルダ選択表示
            if st.session_state.matching_folders:
                st.markdown("### 📂 マッチしたフォルダを選択してください:")
                
                for i, folder in enumerate(st.session_state.matching_folders):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📂 {folder['display_name']}")
                        st.write(f"   パス: {folder['path']}")
                    with col2:
                        # 各フォルダに対して一意のキーを使用
                        if st.button(f"選択", key=f"select_folder_{i}_{folder['display_name']}"):
                            st.session_state.selected_folder = folder
                            st.session_state.current_step = 2
                            st.session_state.step_transition_flag = True
                            st.success(f"✅ フォルダ '{folder['display_name']}' を選択しました")
                            st.info("🔄 ステップ2に移行します...")
                            # 即座にリロード
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ステップ2: キーワード検索
    elif st.session_state.current_step == 2:
        st.markdown('<div class="step-indicator">ステップ2: キーワードで検索</div>', unsafe_allow_html=True)
        
        # 選択されたフォルダが存在するかチェック
        if not st.session_state.selected_folder:
            st.error("❌ フォルダが選択されていません。ステップ1に戻ります。")
            st.session_state.current_step = 1
            st.rerun()
            return
        
        # 選択されたフォルダ情報を表示
        st.info(f"📂 選択されたフォルダ: {st.session_state.selected_folder['display_name']}")
        
        # 戻るボタン
        if st.button("⬅️ 住所選択に戻る", key="back_to_step1"):
            st.session_state.current_step = 1
            st.session_state.matching_folders = []
            st.rerun()
        
        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            
            keyword_input = st.text_input(
                "キーワードを入力:",
                placeholder="例: 4301, 道路, 藤白, 1",
                key="keyword_input"
            )
            
            # 圆圈专用模式的checkbox
            circle_only = st.checkbox(
                "🔴 圆圈内数字のみ検索 (小さな円の中の数字だけを検索)",
                value=st.session_state.circle_only_mode,
                key="circle_only_checkbox",
                help="チェックすると、小さな円の中の数字のみを検索対象とします"
            )
            st.session_state.circle_only_mode = circle_only
            
            if st.button("🔍 検索実行", type="primary", key="execute_search"):
                if keyword_input.strip():
                    is_numeric = is_numeric_query(keyword_input)
                    
                    # 検索結果の表示（圆圈专用模式を適用）
                    results = search_in_ocr_data(keyword_input, ocr_data, circle_only=circle_only)
                    
                    if results:
                        if circle_only:
                            st.success(f"🔴 圆圈专用モード '{keyword_input}' の検索結果 (圆圈内数字のみ):")
                        elif is_numeric:
                            st.success(f"🔢 数字クエリ '{keyword_input}' の検索結果 (圆圈内数字优先):")
                        else:
                            st.success(f"📝 キーワード '{keyword_input}' の検索結果:")
                        
                        # 結果を2列で表示
                        cols = st.columns(2)
                        
                        for i, result in enumerate(results):
                            with cols[i % 2]:
                                is_strict_circle = is_strict_circle_detection(result)
                                
                                # 创建大patch（包含上下文）
                                large_patch, pdf_info = create_large_patch_with_context(
                                    result['pdf_path'], 
                                    result['bbox'], 
                                    expand_factor=4.0
                                )
                                
                                if large_patch is not None:
                                    img_base64 = image_to_base64(large_patch)
                                    
                                    # 严格圆圈检测结果的特殊样式
                                    if is_strict_circle:
                                        st.markdown(f'''
                                        <div class="patch-item circle-priority">
                                            <div class="patch-title">
                                                🔴 {result['text']}
                                                <span class="circle-indicator">严格圆圈</span>
                                            </div>
                                            <img src="{img_base64}" style="width: 100%; max-width: 400px; border-radius: 4px;">
                                            <div class="patch-info">
                                                📄 PDF: {pdf_info.get('pdf_name', 'Unknown')}<br>
                                                📍 位置: {pdf_info.get('original_position', 'Unknown')}<br>
                                                📏 Patch尺寸: {pdf_info.get('patch_size', 'Unknown')}<br>
                                                🎯 信頼度: {result['confidence']:.3f}
                                            </div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'''
                                        <div class="patch-item">
                                            <div class="patch-title">📝 {result['text']}</div>
                                            <img src="{img_base64}" style="width: 100%; max-width: 400px; border-radius: 4px;">
                                            <div class="patch-info">
                                                📄 PDF: {pdf_info.get('pdf_name', 'Unknown')}<br>
                                                📍 位置: {pdf_info.get('original_position', 'Unknown')}<br>
                                                📏 Patch尺寸: {pdf_info.get('patch_size', 'Unknown')}<br>
                                                🎯 信頼度: {result['confidence']:.3f}
                                            </div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 該当する結果が見つかりませんでした")
                else:
                    st.warning("⚠️ キーワードを入力してください")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 異常状態の処理
    else:
        st.error(f"❌ 無効なステップ: {st.session_state.current_step}")
        st.info("🔄 ステップ1にリセットします...")
        st.session_state.current_step = 1
        st.session_state.selected_folder = None
        st.rerun()

if __name__ == "__main__":
    main() 