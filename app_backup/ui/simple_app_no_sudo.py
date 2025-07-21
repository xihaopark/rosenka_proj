#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_app_no_sudo.py
无需sudo权限的极简版本地PDF查询应用
"""

import streamlit as st
import os
import io
import base64
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
import time

# 配置Streamlit
st.set_page_config(
    page_title="🗾 路線価図查询（无权限版）",
    page_icon="🔍",
    layout="wide"
)

# 基础目录设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rosenka_data")

# ========================= 无权限OCR处理器 =========================

class NoPermissionOCR:
    """无权限OCR处理器"""
    
    def __init__(self):
        self.available_methods = []
        self.setup_ocr_methods()
    
    def setup_ocr_methods(self):
        """设置可用的OCR方法"""
        
        # 尝试PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            self.available_methods.append('paddle')
            st.sidebar.success("✅ PaddleOCR可用")
        except:
            st.sidebar.info("❌ PaddleOCR不可用")
        
        # 尝试EasyOCR
        try:
            import easyocr
            self.easy_ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            self.available_methods.append('easy')
            st.sidebar.success("✅ EasyOCR可用")
        except:
            st.sidebar.info("❌ EasyOCR不可用")
        
        # 检查系统Tesseract
        try:
            import pytesseract
            pytesseract.get_languages()
            self.available_methods.append('tesseract')
            st.sidebar.success("✅ Tesseract可用")
        except:
            st.sidebar.info("❌ Tesseract不可用")
        
        if not self.available_methods:
            st.sidebar.warning("⚠️ 没有可用的OCR，将使用模拟数据")
    
    def extract_text_from_image(self, image: np.ndarray) -> List[Dict]:
        """从图像中提取文字"""
        results = []
        
        # 尝试使用可用的OCR方法
        for method in self.available_methods:
            try:
                if method == 'paddle' and hasattr(self, 'paddle_ocr'):
                    paddle_results = self.paddle_ocr.ocr(image, cls=True)
                    for line in paddle_results[0] or []:
                        if len(line) >= 2:
                            bbox_points, (text, confidence) = line
                            if confidence > 0.5:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                bbox = (int(min(x_coords)), int(min(y_coords)), 
                                       int(max(x_coords)), int(max(y_coords)))
                                results.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'method': 'PaddleOCR'
                                })
                
                elif method == 'easy' and hasattr(self, 'easy_ocr'):
                    easy_results = self.easy_ocr.readtext(image)
                    for bbox_points, text, confidence in easy_results:
                        if confidence > 0.5:
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            bbox = (int(min(x_coords)), int(min(y_coords)), 
                                   int(max(x_coords)), int(max(y_coords)))
                            results.append({
                                'text': text,
                                'bbox': bbox,
                                'confidence': confidence,
                                'method': 'EasyOCR'
                            })
                
                elif method == 'tesseract':
                    import pytesseract
                    import cv2
                    
                    # 图像预处理
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR
                    data = pytesseract.image_to_data(
                        binary,
                        lang='jpn+eng',
                        output_type=pytesseract.Output.DICT,
                        config='--psm 6'
                    )
                    
                    n = len(data['text'])
                    for i in range(n):
                        text = data['text'][i].strip()
                        if text:
                            conf = float(data['conf'][i]) if str(data['conf'][i]).isdigit() else 0
                            if conf > 30:
                                bbox = (
                                    data['left'][i],
                                    data['top'][i],
                                    data['left'][i] + data['width'][i],
                                    data['top'][i] + data['height'][i]
                                )
                                results.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'confidence': conf / 100.0,
                                    'method': 'Tesseract'
                                })
                
                # 如果找到结果就停止
                if results:
                    break
                    
            except Exception as e:
                st.sidebar.error(f"{method} OCR错误: {e}")
                continue
        
        # 如果没有OCR可用，生成模拟数据
        if not results and not self.available_methods:
            results = self._generate_mock_data(image)
        
        return results
    
    def _generate_mock_data(self, image: np.ndarray) -> List[Dict]:
        """生成模拟数据"""
        h, w = image.shape[:2]
        mock_texts = [
            "六本木１丁目88-7",
            "川合３丁目120E", 
            "東京都港区",
            "示例地址123"
        ]
        
        results = []
        for i, text in enumerate(mock_texts):
            x = (i % 2) * (w // 2) + 50
            y = (i // 2) * (h // 2) + 50
            results.append({
                'text': text,
                'bbox': (x, y, x + 200, y + 30),
                'confidence': 0.8,
                'method': 'Mock'
            })
        
        return results

# ========================= 本地文件管理器 =========================

@st.cache_data
def scan_local_files():
    """扫描本地PDF文件"""
    files_info = {}
    
    if not os.path.exists(DATA_DIR):
        return files_info
    
    for prefecture in os.listdir(DATA_DIR):
        prefecture_path = os.path.join(DATA_DIR, prefecture)
        if not os.path.isdir(prefecture_path):
            continue
        
        files_info[prefecture] = {}
        
        for city in os.listdir(prefecture_path):
            city_path = os.path.join(prefecture_path, city)
            if not os.path.isdir(city_path):
                continue
            
            files_info[prefecture][city] = {}
            
            for district in os.listdir(city_path):
                district_path = os.path.join(city_path, district)
                if not os.path.isdir(district_path):
                    continue
                
                pdf_files = glob.glob(os.path.join(district_path, "*.pdf"))
                if pdf_files:
                    files_info[prefecture][city][district] = pdf_files
    
    return files_info

# ========================= PDF处理 =========================

@st.cache_data
def pdf_to_images(pdf_path: str, max_pages: int = 3):
    """将PDF转换为图像（限制页数以提高性能）"""
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        images = {}
        
        # 限制处理的页数
        num_pages = min(len(doc), max_pages)
        
        zoom = 200 / 72.0  # 降低DPI以提高性能
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            images[page_num] = np.array(img)
        
        doc.close()
        return images
        
    except Exception as e:
        st.error(f"PDF处理失败: {e}")
        return {}

def search_in_pdf(pdf_path: str, query: str, ocr_processor: NoPermissionOCR) -> List[Dict]:
    """在PDF中搜索"""
    results = []
    
    # 转换PDF为图像
    images = pdf_to_images(pdf_path, max_pages=2)  # 只处理前2页
    
    for page_num, image in images.items():
        # OCR提取文字
        ocr_results = ocr_processor.extract_text_from_image(image)
        
        # 搜索匹配
        for ocr_result in ocr_results:
            text = ocr_result['text']
            similarity = calculate_similarity(query, text)
            
            if similarity > 50:  # 相似度阈值
                results.append({
                    'text': text,
                    'similarity': similarity,
                    'bbox': ocr_result['bbox'],
                    'confidence': ocr_result['confidence'],
                    'method': ocr_result['method'],
                    'page_num': page_num,
                    'image': image,
                    'pdf_path': pdf_path
                })
    
    return results

def calculate_similarity(query: str, text: str) -> float:
    """计算文本相似度"""
    from rapidfuzz import fuzz
    
    if query in text or text in query:
        return 100.0
    
    return fuzz.partial_ratio(query, text)

def create_result_image(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       text: str, similarity: float) -> str:
    """创建结果图像"""
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    # 绘制边界框
    draw.rectangle(bbox, outline='red', width=2)
    
    # 添加标签
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    label = f"{text} ({similarity:.0f}%)"
    draw.text((bbox[0], bbox[1] - 20), label, fill='red', font=font)
    
    # 转换为base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"

# ========================= 主应用 =========================

def main():
    st.title("🗾 路線価図本地查询（无权限版）")
    st.markdown("**极简版本** - 无需sudo权限，支持多种OCR方案")
    
    # 初始化OCR处理器
    if 'ocr_processor' not in st.session_state:
        st.session_state.ocr_processor = NoPermissionOCR()
    
    ocr_processor = st.session_state.ocr_processor
    
    # 扫描本地文件
    with st.spinner("扫描本地PDF文件..."):
        files_info = scan_local_files()
    
    if not files_info:
        st.error(f"❌ 数据目录不存在或为空: {DATA_DIR}")
        st.info("请确保PDF文件已下载到 rosenka_data 目录")
        return
    
    # 统计信息
    total_files = 0
    for prefecture_data in files_info.values():
        for city_data in prefecture_data.values():
            for district_files in city_data.values():
                total_files += len(district_files)
    
    st.success(f"✅ 找到 {len(files_info)} 个都道府县，共 {total_files} 个PDF文件")
    
    # 搜索界面
    st.header("🔍 地址搜索")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "输入要搜索的地址或关键词",
            placeholder="例如: 88-7, 川合３丁目, 120E, 六本木",
            help="支持部分匹配，输入地址、番地、价格等关键词"
        )
    
    with col2:
        max_results = st.number_input(
            "最大结果数",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
    
    # 搜索执行
    if query:
        with st.spinner("🔍 搜索中..."):
            all_results = []
            progress_bar = st.progress(0)
            
            # 限制搜索的文件数量以提高性能
            file_count = 0
            max_files = 100  # 最多搜索100个文件
            
            for prefecture, prefecture_data in files_info.items():
                for city, city_data in prefecture_data.items():
                    for district, district_files in city_data.items():
                        for pdf_file in district_files[:2]:  # 每个区域最多2个文件
                            if file_count >= max_files:
                                break
                            
                            try:
                                pdf_results = search_in_pdf(pdf_file, query, ocr_processor)
                                for result in pdf_results:
                                    result['prefecture'] = prefecture
                                    result['city'] = city
                                    result['district'] = district
                                all_results.extend(pdf_results)
                                
                                file_count += 1
                                progress_bar.progress(min(file_count / max_files, 1.0))
                                
                            except Exception as e:
                                st.sidebar.error(f"处理文件失败: {os.path.basename(pdf_file)} - {e}")
                                continue
                        
                        if file_count >= max_files:
                            break
                    if file_count >= max_files:
                        break
                if file_count >= max_files:
                    break
            
            progress_bar.empty()
        
        # 显示结果
        if all_results:
            # 按相似度排序
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = all_results[:max_results]
            
            st.success(f"🎯 找到 {len(all_results)} 个匹配结果，显示前 {len(top_results)} 个")
            
            # 按都道府县分组显示
            results_by_prefecture = {}
            for result in top_results:
                prefecture = result['prefecture']
                if prefecture not in results_by_prefecture:
                    results_by_prefecture[prefecture] = []
                results_by_prefecture[prefecture].append(result)
            
            for prefecture, prefecture_results in results_by_prefecture.items():
                st.subheader(f"📍 {prefecture} ({len(prefecture_results)}个结果)")
                
                for i, result in enumerate(prefecture_results[:3], 1):  # 每个都道府县最多显示3个
                    with st.expander(
                        f"#{i} {result['text']} (相似度: {result['similarity']:.0f}%)",
                        expanded=i == 1
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 显示图像
                            img_data = create_result_image(
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
        else:
            st.warning("😔 没有找到匹配的结果")
            st.info("尝试使用不同的关键词或检查拼写")
    
    # 侧边栏信息
    with st.sidebar:
        st.header("📊 系统状态")
        
        # OCR状态
        st.subheader("🔧 OCR状态")
        if ocr_processor.available_methods:
            for method in ocr_processor.available_methods:
                st.success(f"✅ {method.upper()}")
        else:
            st.warning("⚠️ 使用模拟数据")
        
        # 数据统计
        st.subheader("📁 数据统计")
        st.write(f"都道府県: {len(files_info)}")
        st.write(f"总文件数: {total_files}")
        
        # 性能说明
        st.subheader("⚡ 性能优化")
        st.info("""
        为提高性能，本版本：
        - 限制搜索文件数量
        - 只处理PDF前2页
        - 降低图像分辨率
        - 禁用文件监控
        """)
        
        # 使用说明
        st.subheader("💡 使用说明")
        st.markdown("""
        **无权限版特点:**
        - 无需sudo权限
        - 支持多种OCR方案
        - 自动降级到可用方法
        - 性能优化配置
        
        **搜索类型:**
        - 地址: 六本木１丁目
        - 番地: 88-7
        - 价格: 120E
        - 部分匹配: 川合
        """)

if __name__ == "__main__":
    main() 