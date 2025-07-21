#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing_app.py
PDF预处理应用程序
专门用于批量处理PDF文件，提取地址和坐标，建立索引数据库
"""

# 环境设置
import os
import sys
sys.path.insert(0, '/venv/main/lib/python3.10/site-packages')
os.environ['PATH'] = '/venv/main/bin:' + os.environ.get('PATH', '')

import streamlit as st
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

# 设置页面配置
st.set_page_config(
    page_title="PDF预处理系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 基础导入
try:
    import numpy as np
    import cv2
    from PIL import Image
    import fitz  # PyMuPDF
    BASIC_IMPORTS_OK = True
except ImportError as e:
    st.error(f"基础库导入失败: {e}")
    BASIC_IMPORTS_OK = False

# OCR导入
OCR_ENGINES = {}
try:
    from paddleocr import PaddleOCR
    OCR_ENGINES['PaddleOCR'] = True
except ImportError:
    OCR_ENGINES['PaddleOCR'] = False

try:
    import easyocr
    OCR_ENGINES['EasyOCR'] = True
except ImportError:
    OCR_ENGINES['EasyOCR'] = False

try:
    import pytesseract
    OCR_ENGINES['Tesseract'] = True
except ImportError:
    OCR_ENGINES['Tesseract'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义CSS
def load_preprocessing_css():
    """加载预处理应用的CSS样式"""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .status-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .success-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .error-card {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .processing-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 数据库管理
class PreprocessingDatabase:
    """预处理数据库管理"""
    
    def __init__(self, db_path: str = "rosenka_addresses.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    addresses_count INTEGER,
                    processing_time REAL,
                    error_message TEXT,
                    created_at TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_normalized_text ON addresses(normalized_text)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_location ON addresses(prefecture, city, district)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pdf_path ON addresses(pdf_path)")
            
            conn.commit()
    
    def get_stats(self):
        """获取统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM addresses")
            total_addresses = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT pdf_path) FROM addresses")
            processed_pdfs = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM processing_log WHERE status = 'success'")
            successful_pdfs = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM processing_log WHERE status = 'error'")
            failed_pdfs = cursor.fetchone()[0]
            
            return {
                'total_addresses': total_addresses,
                'processed_pdfs': processed_pdfs,
                'successful_pdfs': successful_pdfs,
                'failed_pdfs': failed_pdfs
            }
    
    def log_processing(self, pdf_path: str, status: str, addresses_count: int = 0, 
                      processing_time: float = 0, error_message: str = ""):
        """记录处理日志"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processing_log 
                (pdf_path, status, addresses_count, processing_time, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pdf_path, status, addresses_count, processing_time, error_message, 
                  datetime.now().isoformat()))
            conn.commit()

# 简化的OCR处理器
class SimpleOCRProcessor:
    """简化的OCR处理器"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.ocr_engine = None
        self.init_ocr()
    
    def init_ocr(self):
        """初始化OCR引擎"""
        if OCR_ENGINES['PaddleOCR']:
            try:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan',
                    use_gpu=self.use_gpu,
                    show_log=False
                )
                self.engine_name = 'PaddleOCR'
                return True
            except Exception as e:
                st.error(f"PaddleOCR初始化失败: {e}")
        
        if OCR_ENGINES['EasyOCR']:
            try:
                self.ocr_engine = easyocr.Reader(['ja', 'en'], gpu=self.use_gpu, verbose=False)
                self.engine_name = 'EasyOCR'
                return True
            except Exception as e:
                st.error(f"EasyOCR初始化失败: {e}")
        
        if OCR_ENGINES['Tesseract']:
            try:
                pytesseract.get_tesseract_version()
                self.ocr_engine = 'tesseract'
                self.engine_name = 'Tesseract'
                return True
            except Exception as e:
                st.error(f"Tesseract初始化失败: {e}")
        
        return False
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """处理单个PDF文件"""
        if not self.ocr_engine:
            return []
        
        addresses = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 转换为图像
                zoom = 1.5
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # 转换为numpy数组
                img_data = pix.tobytes("png")
                from io import BytesIO
                img = Image.open(BytesIO(img_data)).convert('RGB')
                image = np.array(img)
                
                # OCR识别
                page_addresses = self.extract_addresses_from_image(image, pdf_path, page_num)
                addresses.extend(page_addresses)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"处理PDF失败 {pdf_path}: {e}")
        
        return addresses
    
    def extract_addresses_from_image(self, image: np.ndarray, pdf_path: str, page_num: int) -> List[Dict]:
        """从图像中提取地址"""
        addresses = []
        
        if self.engine_name == 'PaddleOCR':
            addresses = self.extract_with_paddleocr(image, pdf_path, page_num)
        elif self.engine_name == 'EasyOCR':
            addresses = self.extract_with_easyocr(image, pdf_path, page_num)
        elif self.engine_name == 'Tesseract':
            addresses = self.extract_with_tesseract(image, pdf_path, page_num)
        
        return addresses
    
    def extract_with_paddleocr(self, image: np.ndarray, pdf_path: str, page_num: int) -> List[Dict]:
        """使用PaddleOCR提取地址"""
        addresses = []
        
        try:
            results = self.ocr_engine.ocr(image, cls=True)
            
            if results and results[0]:
                for line in results[0]:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    if self.is_valid_address(text):
                        address = self.create_address_entry(
                            text, bbox_points, confidence, pdf_path, page_num
                        )
                        addresses.append(address)
        
        except Exception as e:
            logger.error(f"PaddleOCR处理失败: {e}")
        
        return addresses
    
    def extract_with_easyocr(self, image: np.ndarray, pdf_path: str, page_num: int) -> List[Dict]:
        """使用EasyOCR提取地址"""
        addresses = []
        
        try:
            results = self.ocr_engine.readtext(image)
            
            for bbox_points, text, confidence in results:
                if self.is_valid_address(text):
                    address = self.create_address_entry(
                        text, bbox_points, confidence, pdf_path, page_num
                    )
                    addresses.append(address)
        
        except Exception as e:
            logger.error(f"EasyOCR处理失败: {e}")
        
        return addresses
    
    def extract_with_tesseract(self, image: np.ndarray, pdf_path: str, page_num: int) -> List[Dict]:
        """使用Tesseract提取地址"""
        addresses = []
        
        try:
            config = r'--oem 3 --psm 6 -l jpn+eng'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and data['conf'][i] > 30 and self.is_valid_address(text):
                    bbox_points = [
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ]
                    confidence = data['conf'][i] / 100.0
                    
                    address = self.create_address_entry(
                        text, bbox_points, confidence, pdf_path, page_num
                    )
                    addresses.append(address)
        
        except Exception as e:
            logger.error(f"Tesseract处理失败: {e}")
        
        return addresses
    
    def is_valid_address(self, text: str) -> bool:
        """验证是否为有效地址"""
        if not text or len(text.strip()) < 2:
            return False
        
        # 简单的地址关键词检查
        address_keywords = ['丁目', '番地', '番', '号', '町', '区', '市', '府', '県', '丁', '地']
        
        for keyword in address_keywords:
            if keyword in text:
                return True
        
        # 检查是否包含数字
        if any(char.isdigit() for char in text):
            return True
        
        return False
    
    def create_address_entry(self, text: str, bbox_points: List, confidence: float, 
                           pdf_path: str, page_num: int) -> Dict:
        """创建地址条目"""
        # 计算边界框
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        bbox = (int(min(x_coords)), int(min(y_coords)), 
               int(max(x_coords)), int(max(y_coords)))
        
        # 计算中心点
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # 解析路径信息
        path_parts = Path(pdf_path).parts
        prefecture = path_parts[-4] if len(path_parts) >= 4 else ""
        city = path_parts[-3] if len(path_parts) >= 3 else ""
        district = path_parts[-2] if len(path_parts) >= 2 else ""
        
        return {
            'id': f"{pdf_path}_{page_num}_{hash(text)}_{bbox}",
            'text': text,
            'normalized_text': text.strip(),
            'bbox': bbox,
            'center_point': (center_x, center_y),
            'confidence': confidence,
            'pdf_path': pdf_path,
            'page_num': page_num,
            'prefecture': prefecture,
            'city': city,
            'district': district,
            'sub_district': text,
            'ocr_method': self.engine_name,
            'created_at': datetime.now().isoformat()
        }

# 初始化session state
def init_session_state():
    """初始化会话状态"""
    if 'database' not in st.session_state:
        st.session_state.database = PreprocessingDatabase()
    
    if 'ocr_processor' not in st.session_state:
        st.session_state.ocr_processor = None
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []

def render_header():
    """渲染标题"""
    st.markdown('<h1 class="main-header">🔧 PDF预处理系统</h1>', unsafe_allow_html=True)
    st.markdown("---")

def render_system_status():
    """渲染系统状态"""
    st.subheader("📊 系统状态")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基础环境:**")
        if BASIC_IMPORTS_OK:
            st.success("✅ 基础库已加载")
        else:
            st.error("❌ 基础库加载失败")
    
    with col2:
        st.markdown("**OCR引擎:**")
        available_engines = [name for name, status in OCR_ENGINES.items() if status]
        if available_engines:
            st.success(f"✅ 可用引擎: {', '.join(available_engines)}")
        else:
            st.error("❌ 没有可用的OCR引擎")

def render_database_stats():
    """渲染数据库统计"""
    st.subheader("📈 数据库统计")
    
    stats = st.session_state.database.get_stats()
    
    st.markdown('<div class="processing-stats">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['total_addresses']:,}</div>
            <div class="stat-label">总地址数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['processed_pdfs']:,}</div>
            <div class="stat-label">已处理PDF</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['successful_pdfs']:,}</div>
            <div class="stat-label">成功处理</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['failed_pdfs']:,}</div>
            <div class="stat-label">处理失败</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_processing_interface():
    """渲染处理界面"""
    st.subheader("⚙️ 批量处理")
    
    # 目录选择
    col1, col2 = st.columns(2)
    
    with col1:
        target_directory = st.text_input(
            "目标目录",
            value="大阪府/吹田市/藤白台１",
            help="相对于rosenka_data的路径"
        )
    
    with col2:
        use_gpu = st.checkbox("使用GPU加速", value=True)
    
    # 检查目录
    data_dir = Path("rosenka_data")
    target_path = data_dir / target_directory
    
    if target_path.exists():
        pdf_files = list(target_path.rglob("*.pdf"))
        st.info(f"📁 找到 {len(pdf_files)} 个PDF文件")
        
        if pdf_files:
            # 显示文件列表
            with st.expander("📄 文件列表"):
                for pdf in pdf_files[:10]:  # 只显示前10个
                    size_mb = pdf.stat().st_size / (1024 * 1024)
                    st.write(f"• {pdf.name} ({size_mb:.1f}MB)")
                if len(pdf_files) > 10:
                    st.write(f"... 还有 {len(pdf_files) - 10} 个文件")
            
            # 处理按钮
            if st.button("🚀 开始批量处理", type="primary", disabled=(st.session_state.processing_status == "running")):
                start_batch_processing(target_directory, use_gpu)
        else:
            st.warning("目录中没有PDF文件")
    else:
        st.error(f"目录不存在: {target_path}")

def start_batch_processing(target_directory: str, use_gpu: bool):
    """开始批量处理"""
    st.session_state.processing_status = "running"
    
    # 初始化OCR处理器
    if not st.session_state.ocr_processor:
        st.session_state.ocr_processor = SimpleOCRProcessor(use_gpu=use_gpu)
    
    if not st.session_state.ocr_processor.ocr_engine:
        st.error("OCR引擎初始化失败")
        st.session_state.processing_status = "idle"
        return
    
    # 获取PDF文件列表
    data_dir = Path("rosenka_data")
    target_path = data_dir / target_directory
    pdf_files = list(target_path.rglob("*.pdf"))
    
    if not pdf_files:
        st.error("没有找到PDF文件")
        st.session_state.processing_status = "idle"
        return
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    total_files = len(pdf_files)
    processed_files = 0
    total_addresses = 0
    
    # 处理每个PDF文件
    for i, pdf_path in enumerate(pdf_files):
        status_text.text(f"🔄 处理文件 {i+1}/{total_files}: {pdf_path.name}")
        
        start_time = time.time()
        
        try:
            # 处理PDF
            addresses = st.session_state.ocr_processor.process_pdf(str(pdf_path))
            processing_time = time.time() - start_time
            
            if addresses:
                # 保存到数据库
                save_addresses_to_db(addresses)
                total_addresses += len(addresses)
                
                # 记录成功日志
                st.session_state.database.log_processing(
                    str(pdf_path), "success", len(addresses), processing_time
                )
                
                status_text.text(f"✅ 完成: {pdf_path.name} - {len(addresses)} 个地址")
            else:
                # 记录失败日志
                st.session_state.database.log_processing(
                    str(pdf_path), "no_addresses", 0, processing_time
                )
                
                status_text.text(f"⚠️ 无地址: {pdf_path.name}")
            
            processed_files += 1
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # 记录错误日志
            st.session_state.database.log_processing(
                str(pdf_path), "error", 0, processing_time, error_msg
            )
            
            status_text.text(f"❌ 错误: {pdf_path.name} - {error_msg}")
        
        # 更新进度
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        
        # 显示实时统计
        with results_container.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("已处理", processed_files)
            with col2:
                st.metric("总地址", total_addresses)
            with col3:
                st.metric("进度", f"{progress:.1%}")
    
    # 完成处理
    st.session_state.processing_status = "completed"
    progress_bar.progress(1.0)
    status_text.text("🎉 批量处理完成！")
    
    # 显示最终结果
    st.success(f"✅ 处理完成！共处理 {processed_files} 个PDF文件，提取 {total_addresses} 个地址")

def save_addresses_to_db(addresses: List[Dict]):
    """保存地址到数据库"""
    with sqlite3.connect(st.session_state.database.db_path) as conn:
        for addr in addresses:
            conn.execute("""
                INSERT OR REPLACE INTO addresses VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                addr['id'], addr['text'], addr['normalized_text'],
                addr['bbox'][0], addr['bbox'][1], addr['bbox'][2], addr['bbox'][3],
                addr['center_point'][0], addr['center_point'][1],
                addr['confidence'], addr['pdf_path'], addr['page_num'],
                addr['prefecture'], addr['city'], addr['district'], addr['sub_district'],
                addr['ocr_method'], addr['created_at']
            ))
        conn.commit()

def render_processing_status():
    """渲染处理状态"""
    if st.session_state.processing_status == "running":
        st.markdown('<div class="status-card">🔄 正在处理中...</div>', unsafe_allow_html=True)
    elif st.session_state.processing_status == "completed":
        st.markdown('<div class="success-card">✅ 处理完成</div>', unsafe_allow_html=True)
    elif st.session_state.processing_status == "error":
        st.markdown('<div class="error-card">❌ 处理出错</div>', unsafe_allow_html=True)

def main():
    """主函数"""
    # 加载样式
    load_preprocessing_css()
    
    # 初始化
    init_session_state()
    
    # 渲染界面
    render_header()
    render_system_status()
    render_database_stats()
    render_processing_interface()
    render_processing_status()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 🔧 预处理系统")
        st.markdown("专门用于批量处理PDF文件")
        
        st.markdown("---")
        st.markdown("### 📋 功能")
        st.markdown("""
        - 批量扫描PDF文件
        - OCR文字识别
        - 地址提取和标准化
        - 坐标位置记录
        - 数据库索引建立
        """)
        
        st.markdown("---")
        st.markdown("### 📊 使用流程")
        st.markdown("""
        1. 选择目标目录
        2. 配置处理参数
        3. 开始批量处理
        4. 查看处理结果
        5. 数据库自动建立索引
        """)
        
        if st.button("🗑️ 清空数据库"):
            if st.confirm("确定要清空所有数据吗？"):
                with sqlite3.connect(st.session_state.database.db_path) as conn:
                    conn.execute("DELETE FROM addresses")
                    conn.execute("DELETE FROM processing_log")
                    conn.commit()
                st.success("数据库已清空")
                st.rerun()

if __name__ == "__main__":
    main() 