"""
可视化工具
"""
import cv2
import numpy as np
from PIL import Image
import streamlit as st

def visualize_segments(image: np.ndarray, segments: list) -> np.ndarray:
    """可视化分割结果"""
    vis_img = image.copy()
    
    # 生成随机颜色
    colors = np.random.randint(0, 255, (len(segments), 3))
    
    for idx, segment in enumerate(segments):
        if 'bbox' in segment:
            x, y, w, h = segment['bbox']
            color = colors[idx].tolist()
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            
            # 如果有mask，叠加显示
            if 'mask' in segment:
                mask = segment['mask']
                overlay = vis_img.copy()
                overlay[mask > 0] = color
                vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
    
    return vis_img

def display_results_in_streamlit(results: dict):
    """在Streamlit中展示结果"""
    for page_idx, page_data in enumerate(results['pages']):
        st.subheader(f"页 {page_idx + 1}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("文本区域数", page_data['num_regions'])
        with col2:
            st.metric("页面尺寸", f"{page_data['page_size'][0]}x{page_data['page_size'][1]}")
        with col3:
            st.metric("处理状态", "✅ 完成")
        
        # 显示检测到的文本
        if page_data['text_map']['segments']:
            st.write("**检测到的文本：**")
            text_df = []
            for seg in page_data['text_map']['segments'][:10]:  # 显示前10个
                text_df.append({
                    '文本': seg['text'][:50] + '...' if len(seg['text']) > 50 else seg['text'],
                    'OCR置信度': f"{seg['ocr_confidence']:.2%}",
                    '位置': f"({seg['bbox'][0]}, {seg['bbox'][1]})"
                })
            
            if text_df:
                st.dataframe(text_df)