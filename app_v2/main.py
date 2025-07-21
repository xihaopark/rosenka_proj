#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_rosenka_app.py
极简版路線価図查询系统 - 基于文件夹结构的精确查找
"""

import streamlit as st
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

# 配置
st.set_page_config(
    page_title="路線価図查询",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 极简样式
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 20px;
        padding: 10px;
    }
    .search-button {
        font-size: 18px;
        padding: 10px 30px;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class TextRegion:
    """检测到的文本区域"""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    page_num: int
    pdf_path: str
    text: str = ""  # OCR后填充
    
class AddressParser:
    """地址解析器 - 将输入地址解析为文件路径"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.structure = self._build_structure()
    
    def _build_structure(self) -> Dict:
        """构建文件夹结构索引"""
        structure = {}
        
        for prefecture_path in self.data_dir.iterdir():
            if not prefecture_path.is_dir():
                continue
                
            prefecture = prefecture_path.name
            structure[prefecture] = {}
            
            for city_path in prefecture_path.iterdir():
                if not city_path.is_dir():
                    continue
                    
                city = city_path.name
                structure[prefecture][city] = {}
                
                for district_path in city_path.iterdir():
                    if not district_path.is_dir():
                        continue
                        
                    district = district_path.name
                    pdf_files = list(district_path.glob("*.pdf"))
                    if pdf_files:
                        structure[prefecture][city][district] = pdf_files
        
        return structure
    
    def parse_address(self, address: str) -> List[Path]:
        """解析地址并返回对应的PDF文件路径列表"""
        # 智能解析地址组成部分
        parts = self._extract_address_parts(address)
        
        # 根据解析结果查找PDF文件
        pdf_files = []
        
        if parts['prefecture'] and parts['city'] and parts['district']:
            # 精确匹配
            files = self.structure.get(parts['prefecture'], {}).get(parts['city'], {}).get(parts['district'], [])
            pdf_files.extend(files)
        elif parts['prefecture'] and parts['city']:
            # 匹配市级所有地区
            city_data = self.structure.get(parts['prefecture'], {}).get(parts['city'], {})
            for district_files in city_data.values():
                pdf_files.extend(district_files)
        elif parts['prefecture']:
            # 匹配县级所有地区
            prefecture_data = self.structure.get(parts['prefecture'], {})
            for city_data in prefecture_data.values():
                for district_files in city_data.values():
                    pdf_files.extend(district_files)
        
        return pdf_files
    
    def _extract_address_parts(self, address: str) -> Dict[str, str]:
        """智能提取地址组成部分"""
        # 这里可以使用更复杂的NLP方法
        # 简单实现：基于关键词匹配
        
        parts = {
            'prefecture': '',
            'city': '',
            'district': ''
        }
        
        # 都道府県关键词
        prefectures = ['北海道', '東京都', '大阪府', '京都府'] + [f'{p}県' for p in [
            '青森', '岩手', '宮城', '秋田', '山形', '福島',
            '茨城', '栃木', '群馬', '埼玉', '千葉', '神奈川',
            # ... 其他县
        ]]
        
        for pref in prefectures:
            if pref in address:
                parts['prefecture'] = pref
                address = address.replace(pref, '')
                break
        
        # 提取市区町村
        if '市' in address:
            idx = address.find('市')
            parts['city'] = address[:idx+1]
            address = address[idx+1:]
        
        # 剩余部分作为地区
        if address:
            parts['district'] = address.strip()
        
        return parts

class TextDetectionModel:
    """文本检测模型 - 使用预训练的CV模型"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        """加载预训练的文本检测模型"""
        # 使用DBNet或CRAFT等模型
        # 这里以DBNet为例
        try:
            # 尝试加载本地模型
            model = torch.hub.load('pytorch/vision', 'dbnet_r50_fpn', pretrained=True)
            model = model.to(self.device)
            model.eval()
            return model
        except:
            # 备用方案：使用更简单的边缘检测
            st.warning("使用备用文本检测方案")
            return None
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """检测图像中的文本区域"""
        if self.model is not None:
            return self._detect_with_model(image)
        else:
            return self._detect_with_traditional(image)
    
    def _detect_with_model(self, image: np.ndarray) -> List[TextRegion]:
        """使用深度学习模型检测"""
        # 预处理
        img_tensor = self._preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # 后处理
        regions = self._postprocess_predictions(predictions, image)
        return regions
    
    def _detect_with_traditional(self, image: np.ndarray) -> List[TextRegion]:
        """使用传统CV方法检测文本区域"""
        regions = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤和提取文本区域
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤条件
            if w > 30 and h > 10 and w/h < 20:
                bbox = (x, y, x+w, y+h)
                region_img = image[y:y+h, x:x+w]
                
                regions.append(TextRegion(
                    image=region_img,
                    bbox=bbox,
                    confidence=0.8,
                    page_num=0,
                    pdf_path=""
                ))
        
        return regions
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        # 标准化等处理
        img = Image.fromarray(image)
        # ... 预处理步骤
        return torch.tensor(image).unsqueeze(0).to(self.device)

class MinimalRosenkaApp:
    """极简路線価図查询应用"""
    
    def __init__(self):
        self.data_dir = Path("rosenka_data")
        self.parser = AddressParser(self.data_dir)
        self.detector = TextDetectionModel()
        
    def run(self):
        """运行应用"""
        # 标题
        st.markdown("<h1 style='text-align: center;'>🗾 路線価図查询</h1>", unsafe_allow_html=True)
        
        # 搜索框
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            address = st.text_input(
                "",
                placeholder="输入地址（例：大阪府大阪市北区）",
                key="address_input"
            )
            
            search_button = st.button("🔍 查询", use_container_width=True)
        
        # 执行搜索
        if search_button and address:
            self._perform_search(address)
    
    def _perform_search(self, address: str):
        """执行搜索"""
        with st.spinner("查找中..."):
            # 1. 解析地址并找到PDF文件
            pdf_files = self.parser.parse_address(address)
            
            if not pdf_files:
                st.error("未找到相关文件")
                return
            
            st.success(f"找到 {len(pdf_files)} 个相关文件")
            
            # 2. 处理每个PDF文件
            all_regions = []
            
            for pdf_path in pdf_files[:5]:  # 限制处理数量
                regions = self._process_pdf(pdf_path)
                all_regions.extend(regions)
            
            # 3. 显示结果
            self._display_results(all_regions)
    
    def _process_pdf(self, pdf_path: Path) -> List[TextRegion]:
        """处理单个PDF文件"""
        import fitz
        
        regions = []
        doc = fitz.open(str(pdf_path))
        
        # 只处理前几页
        for page_num in range(min(len(doc), 3)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            
            # 检测文本区域
            page_regions = self.detector.detect_text_regions(img_array)
            
            # 设置PDF路径和页码
            for region in page_regions:
                region.pdf_path = str(pdf_path)
                region.page_num = page_num
            
            regions.extend(page_regions)
        
        doc.close()
        return regions
    
    def _display_results(self, regions: List[TextRegion]):
        """显示检测结果"""
        st.subheader(f"检测到 {len(regions)} 个文本区域")
        
        # 按页分组显示
        pages = {}
        for region in regions:
            key = f"{Path(region.pdf_path).name} - 第{region.page_num + 1}页"
            if key not in pages:
                pages[key] = []
            pages[key].append(region)
        
        for page_key, page_regions in pages.items():
            with st.expander(page_key):
                # 显示检测到的区域
                cols = st.columns(3)
                for i, region in enumerate(page_regions[:9]):  # 最多显示9个
                    with cols[i % 3]:
                        # 显示区域图像
                        st.image(region.image, use_column_width=True)
                        st.caption(f"区域 {i+1} (置信度: {region.confidence:.2f})")

def main():
    """主函数"""
    app = MinimalRosenkaApp()
    app.run()

if __name__ == "__main__":
    main()