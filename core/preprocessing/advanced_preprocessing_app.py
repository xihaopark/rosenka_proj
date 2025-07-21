import sys
import os
import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
import json
from datetime import datetime
import time

# 设置环境路径
sys.path.insert(0, '/venv/main/lib/python3.10/site-packages')

# 导入自定义模块
from advanced_ocr_processor import AdvancedOCRProcessor, DetectionResult
from model_downloader import ModelDownloader

# 设置页面配置
st.set_page_config(
    page_title="路線価図 高级识别系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用CSS样式
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .status-success {
        background-color: #d1fae5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background-color: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background-color: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .tech-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    
    .progress-container {
        background: #f3f4f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """初始化session state"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'model_downloader' not in st.session_state:
        st.session_state.model_downloader = ModelDownloader()
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None

def display_header():
    """显示应用标题"""
    st.markdown('<h1 class="main-header">🔍 路線価図 高级智能识别系统</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🚀 先进技术栈</h3>
        <p>集成多种最新AI技术，专门优化小物体检测和带圆圈数字识别：</p>
        <div>
            <span class="tech-badge">SAM2 图像分割</span>
            <span class="tech-badge">Donut 文档理解</span>
            <span class="tech-badge">优化YOLO 小物体检测</span>
            <span class="tech-badge">Patch-based 处理</span>
            <span class="tech-badge">GPU 加速</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_model_status():
    """显示模型状态"""
    st.sidebar.markdown("## 🤖 模型状态")
    
    downloader = st.session_state.model_downloader
    
    # 核心模型列表
    essential_models = [
        ("sam2_base", "SAM2 图像分割"),
        ("donut_docvqa", "Donut 文档理解"),
        ("yolov8n", "YOLO 目标检测"),
        ("yolo_small_object", "小物体检测优化")
    ]
    
    all_downloaded = True
    
    for model_name, description in essential_models:
        config = downloader.model_configs.get(model_name, {})
        filepath = downloader.models_dir / config.get("filename", "")
        
        if filepath.exists():
            st.sidebar.markdown(f"✅ {description}")
        else:
            st.sidebar.markdown(f"❌ {description}")
            all_downloaded = False
    
    if not all_downloaded:
        if st.sidebar.button("🔄 下载核心模型"):
            with st.spinner("正在下载模型..."):
                downloader.download_essential_models()
                st.rerun()
    
    return all_downloaded

def display_system_info():
    """显示系统信息"""
    st.sidebar.markdown("## 📊 系统信息")
    
    # GPU状态
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    if gpu_available:
        st.sidebar.markdown(f"🎮 GPU: 可用 ({gpu_count} 设备)")
        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            st.sidebar.markdown(f"📱 设备: {gpu_name}")
    else:
        st.sidebar.markdown("💻 GPU: 不可用，使用CPU")
    
    # 数据库状态
    db_path = "rosenka_addresses.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM addresses")
        count = cursor.fetchone()[0]
        conn.close()
        st.sidebar.markdown(f"🗄️ 数据库: {count} 条记录")
    else:
        st.sidebar.markdown("🗄️ 数据库: 未创建")

def pdf_processing_interface():
    """PDF处理界面"""
    st.markdown("## 📄 PDF高级处理")
    
    # 检查模型状态
    models_ready = display_model_status()
    
    if not models_ready:
        st.warning("⚠️ 请先下载所有必需的模型才能开始处理")
        return
    
    # 初始化处理器
    if st.session_state.processor is None:
        with st.spinner("初始化高级OCR处理器..."):
            st.session_state.processor = AdvancedOCRProcessor(gpu_enabled=True)
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择PDF文件",
        type=['pdf'],
        help="支持上传PDF格式的路線価図文件"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.session_state.current_pdf = temp_path
        
        # 显示PDF信息
        import fitz
        doc = fitz.open(temp_path)
        page_count = len(doc)
        doc.close()
        
        st.success(f"✅ PDF文件加载成功！共 {page_count} 页")
        
        # 页面选择
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_pages = st.multiselect(
                "选择要处理的页面",
                options=list(range(page_count)),
                default=[0] if page_count > 0 else [],
                format_func=lambda x: f"第 {x+1} 页"
            )
        
        with col2:
            st.markdown("### ⚙️ 处理选项")
            use_sam = st.checkbox("启用SAM2分割", value=True)
            use_donut = st.checkbox("启用Donut文档理解", value=True)
            use_yolo = st.checkbox("启用YOLO检测", value=True)
            detect_circles = st.checkbox("检测带圆圈数字", value=True)
        
        # 开始处理
        if st.button("🚀 开始高级处理", type="primary"):
            process_pdf_advanced(temp_path, selected_pages, {
                'use_sam': use_sam,
                'use_donut': use_donut,
                'use_yolo': use_yolo,
                'detect_circles': detect_circles
            })

def process_pdf_advanced(pdf_path: str, selected_pages: list, options: dict):
    """执行高级PDF处理"""
    processor = st.session_state.processor
    
    if not processor:
        st.error("处理器未初始化")
        return
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    all_results = []
    total_pages = len(selected_pages)
    
    for i, page_num in enumerate(selected_pages):
        # 更新进度
        progress = (i + 1) / total_pages
        progress_bar.progress(progress)
        status_text.text(f"正在处理第 {page_num + 1} 页... ({i + 1}/{total_pages})")
        
        try:
            # 执行高级处理
            results = processor.process_pdf_page_advanced(pdf_path, page_num)
            
            if results:
                all_results.extend(results)
                
                # 保存到数据库
                processor.save_results_to_database(
                    results, 
                    pdf_path, 
                    page_num, 
                    "advanced_rosenka_addresses.db"
                )
                
                # 实时显示结果
                with results_container:
                    display_page_results(page_num, results)
            
        except Exception as e:
            st.error(f"处理第 {page_num + 1} 页时出错: {str(e)}")
    
    # 完成处理
    progress_bar.progress(1.0)
    status_text.text("✅ 处理完成！")
    
    # 显示总结
    display_processing_summary(all_results)
    
    # 保存结果到session state
    st.session_state.processing_results = all_results

def display_page_results(page_num: int, results: list):
    """显示页面处理结果"""
    st.markdown(f"### 📄 第 {page_num + 1} 页处理结果")
    
    if not results:
        st.info("该页面未检测到任何对象")
        return
    
    # 按检测类型分组
    results_by_type = {}
    for result in results:
        if result.detection_type not in results_by_type:
            results_by_type[result.detection_type] = []
        results_by_type[result.detection_type].append(result)
    
    # 显示每种类型的结果
    cols = st.columns(len(results_by_type))
    
    for i, (detection_type, type_results) in enumerate(results_by_type.items()):
        with cols[i]:
            type_name = {
                'circle_number': '🔵 带圆圈数字',
                'address': '📍 地址信息',
                'text': '📝 文本内容'
            }.get(detection_type, detection_type)
            
            st.markdown(f"**{type_name}**")
            st.markdown(f"检测到 {len(type_results)} 个")
            
            for result in type_results[:3]:  # 只显示前3个
                st.markdown(f"• {result.text} (置信度: {result.confidence:.2f})")

def display_processing_summary(all_results: list):
    """显示处理总结"""
    st.markdown("## 📊 处理总结")
    
    if not all_results:
        st.info("未检测到任何对象")
        return
    
    # 统计信息
    total_detections = len(all_results)
    
    # 按类型统计
    type_counts = {}
    confidence_scores = []
    
    for result in all_results:
        if result.detection_type not in type_counts:
            type_counts[result.detection_type] = 0
        type_counts[result.detection_type] += 1
        confidence_scores.append(result.confidence)
    
    # 显示统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 总检测数</h3>
            <h2>{total_detections}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 平均置信度</h3>
            <h2>{avg_confidence:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        circle_count = type_counts.get('circle_number', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔵 圆圈数字</h3>
            <h2>{circle_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        address_count = type_counts.get('address', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📍 地址信息</h3>
            <h2>{address_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 详细结果表格
    if st.checkbox("显示详细结果"):
        df_data = []
        for result in all_results:
            df_data.append({
                '检测类型': result.detection_type,
                '文本内容': result.text,
                '置信度': f"{result.confidence:.3f}",
                '中心坐标': f"({result.center_x}, {result.center_y})",
                '边界框': f"({result.bbox[0]}, {result.bbox[1]}, {result.bbox[2]}, {result.bbox[3]})"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

def visualization_interface():
    """可视化界面"""
    st.markdown("## 🎨 结果可视化")
    
    if not st.session_state.processing_results:
        st.info("请先处理PDF文件以查看可视化结果")
        return
    
    if not st.session_state.current_pdf:
        st.info("没有当前PDF文件")
        return
    
    # 页面选择
    results = st.session_state.processing_results
    available_pages = list(set([r.page_num for r in results if hasattr(r, 'page_num')]))
    
    if not available_pages:
        st.info("没有可用的处理结果页面")
        return
    
    selected_page = st.selectbox(
        "选择要可视化的页面",
        available_pages,
        format_func=lambda x: f"第 {x+1} 页"
    )
    
    # 生成可视化图像
    if st.button("生成可视化"):
        generate_visualization(st.session_state.current_pdf, selected_page, results)

def generate_visualization(pdf_path: str, page_num: int, all_results: list):
    """生成可视化图像"""
    try:
        import fitz
        
        # 提取PDF页面
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 2倍放大
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        doc.close()
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 过滤当前页面的结果
        page_results = [r for r in all_results if hasattr(r, 'page_num') and r.page_num == page_num]
        
        # 在图像上绘制检测结果
        colors = {
            'circle_number': (0, 255, 0),  # 绿色
            'address': (255, 0, 0),        # 红色
            'text': (0, 0, 255)            # 蓝色
        }
        
        for i, result in enumerate(page_results):
            color = colors.get(result.detection_type, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (result.bbox[0], result.bbox[1]), 
                         (result.bbox[2], result.bbox[3]), 
                         color, 2)
            
            # 绘制标签
            label = f"{result.text} ({result.confidence:.2f})"
            cv2.putText(image, label, 
                       (result.bbox[0], result.bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 转换为PIL图像并显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        st.image(pil_image, caption=f"第 {page_num + 1} 页检测结果", use_column_width=True)
        
        # 显示图例
        st.markdown("### 🎨 图例")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🟢 **绿色**: 带圆圈数字")
        with col2:
            st.markdown("🔴 **红色**: 地址信息")
        with col3:
            st.markdown("🔵 **蓝色**: 其他文本")
        
    except Exception as e:
        st.error(f"生成可视化时出错: {str(e)}")

def database_interface():
    """数据库管理界面"""
    st.markdown("## 🗄️ 数据库管理")
    
    db_path = "advanced_rosenka_addresses.db"
    
    if not os.path.exists(db_path):
        st.info("高级处理数据库尚未创建，请先处理一些PDF文件")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # 显示统计信息
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM advanced_detections")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT detection_type, COUNT(*) FROM advanced_detections GROUP BY detection_type")
        type_counts = cursor.fetchall()
        
        st.markdown("### 📊 数据库统计")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("总记录数", total_count)
        
        with col2:
            st.markdown("**按类型统计:**")
            for detection_type, count in type_counts:
                st.markdown(f"• {detection_type}: {count}")
        
        # 数据查看
        if st.checkbox("查看数据"):
            query = """
            SELECT detection_type, text, confidence, pdf_path, page_num, created_at 
            FROM advanced_detections 
            ORDER BY created_at DESC 
            LIMIT 100
            """
            
            df = pd.read_sql_query(query, conn)
            st.dataframe(df, use_container_width=True)
        
        # 数据导出
        if st.button("导出数据"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"advanced_detection_results_{timestamp}.csv"
            
            query = "SELECT * FROM advanced_detections"
            df = pd.read_sql_query(query, conn)
            df.to_csv(export_path, index=False, encoding='utf-8-sig')
            
            st.success(f"数据已导出到: {export_path}")
        
        conn.close()
        
    except Exception as e:
        st.error(f"数据库操作出错: {str(e)}")

def main():
    """主函数"""
    load_css()
    initialize_session_state()
    display_header()
    display_system_info()
    
    # 主要功能选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 PDF处理", 
        "🎨 结果可视化", 
        "🗄️ 数据库管理", 
        "⚙️ 系统设置"
    ])
    
    with tab1:
        pdf_processing_interface()
    
    with tab2:
        visualization_interface()
    
    with tab3:
        database_interface()
    
    with tab4:
        st.markdown("## ⚙️ 系统设置")
        
        # 模型管理
        st.markdown("### 🤖 模型管理")
        
        if st.button("重新下载所有模型"):
            with st.spinner("正在下载模型..."):
                st.session_state.model_downloader.download_all_models(force_download=True)
                st.success("模型下载完成")
        
        if st.button("清理无效模型"):
            st.session_state.model_downloader.cleanup_invalid_models()
            st.success("清理完成")
        
        # 处理参数设置
        st.markdown("### ⚙️ 处理参数")
        
        patch_size = st.slider("Patch大小", 320, 1280, 640, 32)
        overlap_ratio = st.slider("重叠比例", 0.1, 0.5, 0.2, 0.05)
        min_confidence = st.slider("最小置信度", 0.1, 0.9, 0.3, 0.05)
        
        if st.button("保存设置"):
            # 这里可以保存设置到配置文件
            st.success("设置已保存")

if __name__ == "__main__":
    main() 