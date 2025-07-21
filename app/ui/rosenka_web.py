#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosenka_web.py
路線価図查询Web界面
基于Streamlit的现代化Web应用
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time
import base64
import io
from datetime import datetime
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="🗾 路線価図查询系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .similarity-high { color: #28a745; font-weight: bold; }
    .similarity-medium { color: #ffc107; font-weight: bold; }
    .similarity-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API配置
API_BASE_URL = "http://localhost:8000"

# ========================= API客户端 =========================

class APIClient:
    """API客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def search_address(self, query: str, **kwargs) -> Dict:
        """搜索地址"""
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API请求失败: {e}")
            return {"results": [], "total_count": 0, "search_time": 0, "cache_used": False}
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_prefectures(self) -> List[str]:
        """获取都道府县列表"""
        try:
            response = self.session.get(f"{self.base_url}/prefectures", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取都道府县列表失败: {e}")
            return []
    
    def get_cities(self, prefecture: str) -> List[str]:
        """获取市区町村列表"""
        try:
            response = self.session.get(f"{self.base_url}/cities/{prefecture}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取市区町村列表失败: {e}")
            return []
    
    def get_districts(self, prefecture: str, city: str) -> List[str]:
        """获取町丁目列表"""
        try:
            response = self.session.get(f"{self.base_url}/districts/{prefecture}/{city}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取町丁目列表失败: {e}")
            return []
    
    def clear_cache(self) -> bool:
        """清理缓存"""
        try:
            response = self.session.delete(f"{self.base_url}/cache", timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"清理缓存失败: {e}")
            return False

# ========================= 辅助函数 =========================

def format_similarity(similarity: float) -> str:
    """格式化相似度显示"""
    if similarity >= 80:
        return f'<span class="similarity-high">{similarity:.1f}%</span>'
    elif similarity >= 60:
        return f'<span class="similarity-medium">{similarity:.1f}%</span>'
    else:
        return f'<span class="similarity-low">{similarity:.1f}%</span>'

def create_result_visualization(results: List[Dict]) -> go.Figure:
    """创建结果可视化图表"""
    if not results:
        return go.Figure()
    
    # 按都道府县分组
    prefecture_counts = {}
    for result in results:
        prefecture = result.get('prefecture', 'Unknown')
        prefecture_counts[prefecture] = prefecture_counts.get(prefecture, 0) + 1
    
    # 创建饼图
    fig = px.pie(
        values=list(prefecture_counts.values()),
        names=list(prefecture_counts.keys()),
        title="搜索结果按都道府县分布"
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig

def create_similarity_histogram(results: List[Dict]) -> go.Figure:
    """创建相似度分布直方图"""
    if not results:
        return go.Figure()
    
    similarities = [result.get('similarity', 0) for result in results]
    
    fig = px.histogram(
        x=similarities,
        nbins=20,
        title="相似度分布",
        labels={'x': '相似度 (%)', 'y': '数量'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

# ========================= 主应用 =========================

def main():
    """主应用"""
    # 初始化API客户端
    api_client = APIClient(API_BASE_URL)
    
    # 检查API连接
    try:
        stats = api_client.get_stats()
        api_connected = True
    except:
        api_connected = False
    
    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🗾 路線価図查询系统</h1>
        <p>高性能地址查询 • 实时OCR识别 • 智能匹配</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API连接状态
    if api_connected:
        st.success("✅ API服务连接正常")
    else:
        st.error("❌ API服务未连接，请启动后端服务")
        st.info("运行命令: `python rosenka_service.py --mode api`")
        return
    
    # 侧边栏
    with st.sidebar:
        st.header("🔧 搜索设置")
        
        # 获取可用的都道府县
        prefectures = api_client.get_prefectures()
        
        # 地区选择
        prefecture = st.selectbox(
            "都道府县",
            ["全部"] + prefectures,
            help="选择特定都道府县或搜索全部"
        )
        
        city = None
        district = None
        
        if prefecture != "全部":
            cities = api_client.get_cities(prefecture)
            city = st.selectbox(
                "市区町村",
                ["全部"] + cities,
                help="选择特定市区町村或搜索全部"
            )
            
            if city != "全部":
                districts = api_client.get_districts(prefecture, city)
                district = st.selectbox(
                    "町丁目",
                    ["全部"] + districts,
                    help="选择特定町丁目或搜索全部"
                )
        
        st.divider()
        
        # 搜索参数
        similarity_threshold = st.slider(
            "相似度阈值",
            min_value=0,
            max_value=100,
            value=50,
            help="设置最低相似度阈值"
        )
        
        max_results = st.slider(
            "最大结果数",
            min_value=10,
            max_value=200,
            value=50,
            help="限制返回的最大结果数"
        )
        
        use_cache = st.checkbox(
            "使用缓存",
            value=True,
            help="启用缓存可以提高重复查询的速度"
        )
        
        st.divider()
        
        # 系统信息
        st.subheader("📊 系统状态")
        if stats:
            st.metric("PDF文件数", stats.get('total_pdfs', 0))
            st.metric("地址索引数", stats.get('total_addresses', 0))
            st.metric("搜索次数", stats.get('search_count', 0))
            st.metric("缓存命中率", f"{stats.get('cache_hits', 0) / max(stats.get('search_count', 1), 1) * 100:.1f}%")
            st.metric("平均响应时间", f"{stats.get('avg_response_time', 0):.2f}s")
        
        # 管理操作
        st.divider()
        st.subheader("🛠️ 管理操作")
        
        if st.button("🗑️ 清理缓存"):
            if api_client.clear_cache():
                st.success("缓存清理成功")
            else:
                st.error("缓存清理失败")
    
    # 主内容区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 搜索界面
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.subheader("🔍 地址搜索")
        
        # 搜索输入
        query = st.text_input(
            "输入要搜索的地址或关键词",
            placeholder="例如: 88-7, 川合３丁目, 120E, 六本木",
            help="支持部分匹配，可以输入地址、番地、价格等任意关键词"
        )
        
        # 搜索按钮
        search_button = st.button("🚀 开始搜索", type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 执行搜索
        if search_button and query:
            with st.spinner("🔍 搜索中..."):
                # 构建搜索参数
                search_params = {
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results,
                    "use_cache": use_cache
                }
                
                if prefecture != "全部":
                    search_params["prefecture"] = prefecture
                if city and city != "全部":
                    search_params["city"] = city
                if district and district != "全部":
                    search_params["district"] = district
                
                # 执行搜索
                response = api_client.search_address(query, **search_params)
                
                # 保存搜索结果到session state
                st.session_state.search_results = response
                st.session_state.search_query = query
        
        # 显示搜索结果
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            results = st.session_state.search_results
            
            st.success(f"🎯 找到 {results['total_count']} 个匹配结果 (用时: {results['search_time']:.2f}s)")
            
            if results['cache_used']:
                st.info("⚡ 使用了缓存结果")
            
            # 结果列表
            if results['results']:
                st.subheader("📋 搜索结果")
                
                # 按都道府县分组
                results_by_prefecture = {}
                for result in results['results']:
                    prefecture_name = result.get('prefecture', 'Unknown')
                    if prefecture_name not in results_by_prefecture:
                        results_by_prefecture[prefecture_name] = []
                    results_by_prefecture[prefecture_name].append(result)
                
                # 显示每个都道府县的结果
                for prefecture_name, prefecture_results in results_by_prefecture.items():
                    st.markdown(f"### 📍 {prefecture_name} ({len(prefecture_results)}个结果)")
                    
                    # 显示前10个结果
                    display_results = prefecture_results[:10]
                    
                    for i, result in enumerate(display_results, 1):
                        # 使用full_address作为显示文本
                        display_text = result.get('full_address', result.get('text', 'Unknown'))
                        
                        with st.expander(
                            f"#{i} {display_text} - {format_similarity(result['similarity'])}",
                            expanded=i == 1
                        ):
                            col_info, col_details = st.columns([1, 1])
                            
                            with col_info:
                                st.markdown("**📋 基本信息**")
                                st.write(f"**地址:** {display_text}")
                                st.markdown(f"**相似度:** {format_similarity(result['similarity'])}", unsafe_allow_html=True)
                                st.write(f"**方法:** {result.get('method', 'path_match')}")
                                st.write(f"**匹配类型:** {result.get('match_type', 'path')}")
                            
                            with col_details:
                                st.markdown("**📍 位置信息**")
                                st.write(f"**都道府县:** {result['prefecture']}")
                                st.write(f"**市区町村:** {result['city']}")
                                st.write(f"**町丁目:** {result['district']}")
                                st.write(f"**文件名:** {result.get('filename', 'Unknown')}")
                                st.write(f"**相对路径:** {result.get('relative_path', 'Unknown')}")
                            
                            # 文件路径信息
                            pdf_path = result.get('pdf_path', '')
                            if pdf_path:
                                st.write(f"**完整路径:** {pdf_path}")
                    
                    if len(prefecture_results) > 10:
                        st.info(f"还有 {len(prefecture_results) - 10} 个结果未显示")
            
            else:
                st.warning("😔 没有找到匹配的结果")
                st.info("尝试使用不同的关键词或降低相似度阈值")
    
    with col2:
        # 结果统计和可视化
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            results = st.session_state.search_results
            
            if results['results']:
                st.subheader("📊 结果分析")
                
                # 基本统计
                st.markdown("**📈 统计信息**")
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.metric("总结果数", results['total_count'])
                    st.metric("搜索用时", f"{results['search_time']:.2f}s")
                
                with col_stat2:
                    avg_similarity = sum(r['similarity'] for r in results['results']) / len(results['results'])
                    st.metric("平均相似度", f"{avg_similarity:.1f}%")
                    st.metric("最高相似度", f"{max(r['similarity'] for r in results['results']):.1f}%")
                
                # 都道府县分布图
                st.plotly_chart(
                    create_result_visualization(results['results']),
                    use_container_width=True
                )
                
                # 相似度分布图
                st.plotly_chart(
                    create_similarity_histogram(results['results']),
                    use_container_width=True
                )
                
                # 导出功能
                st.subheader("💾 导出结果")
                
                # 转换为DataFrame
                df = pd.DataFrame(results['results'])
                
                # CSV下载
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 下载CSV",
                    data=csv,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON下载
                json_data = json.dumps(results, indent=2, ensure_ascii=False).encode('utf-8')
                st.download_button(
                    label="📄 下载JSON",
                    data=json_data,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            # 使用说明
            st.subheader("💡 使用说明")
            st.markdown("""
            **🔍 搜索功能:**
            - 输入地址、番地、价格等关键词
            - 支持部分匹配和模糊搜索
            - 可以按地区筛选搜索范围
            
            **⚙️ 参数设置:**
            - **相似度阈值**: 控制匹配精度
            - **最大结果数**: 限制返回数量
            - **使用缓存**: 提高重复查询速度
            
            **📊 结果分析:**
            - 查看搜索结果统计
            - 按都道府县分布图
            - 相似度分布分析
            
            **💾 导出功能:**
            - 支持CSV和JSON格式
            - 包含完整的搜索结果
            """)
            
            # 示例查询
            st.subheader("🎯 示例查询")
            example_queries = [
                "88-7",
                "川合３丁目",
                "120E",
                "六本木",
                "新宿",
                "渋谷"
            ]
            
            for example in example_queries:
                if st.button(f"🔍 {example}", key=f"example_{example}"):
                    st.session_state.example_query = example
                    st.rerun()

def run_web_app(host: str = "127.0.0.1", port: int = 8501):
    """运行Web应用"""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        __file__, 
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 