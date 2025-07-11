"""
SAM路線価図处理主程序
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app_sam.core.sam_rosenka_pipeline import SAMRosenkaPipeline
from app_sam.config import MODEL_CONFIG, DATA_DIR, OUTPUT_DIR

st.set_page_config(
    page_title="SAM路線価図分析",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    """加载处理管道（缓存）"""
    return SAMRosenkaPipeline(
        sam_checkpoint=str(MODEL_CONFIG['sam_checkpoint'])
    )

def main():
    st.title("🔍 SAM路線価図智能分析系统")
    
    # 加载模型
    with st.spinner("加载SAM模型..."):
        pipeline = load_pipeline()
    
    st.success("✅ 模型加载完成")
    
    # 文件选择
    st.subheader("选择PDF文件")
    
    # 扫描可用文件
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    if not pdf_files:
        st.error("未找到PDF文件")
        return
    
    # 创建选择框
    selected_pdf = st.selectbox(
        "选择要处理的PDF",
        pdf_files,
        format_func=lambda x: str(x.relative_to(DATA_DIR))
    )
    
    # 处理按钮
    if st.button("🚀 开始处理", type="primary"):
        with st.spinner("处理中..."):
            results = pipeline.process_pdf(selected_pdf)
            
        st.success("处理完成！")
        
        # 显示结果
        if results['pages']:
            page_data = results['pages'][0]
            st.metric("检测到的文本区域", page_data['num_regions'])
            
            # 显示部分文本
            if page_data['text_map']['segments']:
                st.subheader("检测到的文本示例")
                for i, segment in enumerate(page_data['text_map']['segments'][:5]):
                    st.write(f"{i+1}. {segment['text']}")

if __name__ == "__main__":
    main()