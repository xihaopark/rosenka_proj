#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosenka_downloader.py
下载全日本47都道府県所有市区町村的路線価図PDF，自动跳过本地已有文件，采用稳妥并发。
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import time
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import re
import pickle

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rosenka_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 下载根目录
DATA_DIR = Path('rosenka_data')
DATA_DIR.mkdir(exist_ok=True)

# 缓存文件
CACHE_FILE = 'download_cache.pkl'

# 并发线程数（稳妥）
MAX_WORKERS = 20

# 读取县市URL映射
with open('prefecture_url_mapping.json', 'r', encoding='utf-8') as f:
    PREFECTURE_URL_MAPPING = json.load(f)

class RosenkaDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 政令指定都市及其区
        self.major_cities = {
            '札幌市': ['厚別区', '北区', '清田区', '白石区', '中央区', '手稲区', '豊平区', '西区', '東区', '南区'],
            '仙台市': ['青葉区', '泉区', '太白区', '宮城野区', '若林区'],
            'さいたま市': ['西区', '北区', '大宮区', '見沼区', '中央区', '桜区', '浦和区', '南区', '緑区', '岩槻区'],
            '千葉市': ['稲毛区', '中央区', '花見川区', '緑区', '美浜区', '若葉区'],
            '横浜市': ['鶴見区', '神奈川区', '西区', '中区', '南区', '保土ケ谷区', '磯子区', '金沢区', '港北区', '戸塚区', '港南区', '旭区', '緑区', '瀬谷区', '栄区', '泉区', '青葉区', '都筑区'],
            '川崎市': ['川崎区', '幸区', '中原区', '高津区', '多摩区', '宮前区', '麻生区'],
            '相模原市': ['緑区', '中央区', '南区'],
            '新潟市': ['北区', '東区', '中央区', '江南区', '秋葉区', '南区', '西区', '西蒲区'],
            '静岡市': ['葵区', '駿河区', '清水区'],
            '浜松市': ['中区', '東区', '西区', '南区', '北区', '浜北区', '天竜区'],
            '名古屋市': ['千種区', '東区', '北区', '西区', '中村区', '中区', '昭和区', '瑞穂区', '熱田区', '中川区', '港区', '南区', '守山区', '緑区', '名東区', '天白区'],
            '京都市': ['北区', '上京区', '左京区', '中京区', '東山区', '下京区', '南区', '右京区', '伏見区', '山科区', '西京区'],
            '大阪市': ['都島区', '福島区', '此花区', '西区', '港区', '大正区', '天王寺区', '浪速区', '西淀川区', '東淀川区', '東成区', '生野区', '旭区', '城東区', '阿倍野区', '住吉区', '東住吉区', '西成区', '淀川区', '鶴見区', '住之江区', '平野区', '北区', '中央区'],
            '堺市': ['堺区', '美原区', '北区', '西区', '中区', '東区', '南区'],
            '神戸市': ['東灘区', '灘区', '兵庫区', '長田区', '須磨区', '垂水区', '北区', '中央区', '西区'],
            '岡山市': ['北区', '中区', '東区', '南区'],
            '広島市': ['中区', '東区', '南区', '西区', '安佐南区', '安佐北区', '安芸区', '佐伯区'],
            '北九州市': ['門司区', '若松区', '戸畑区', '小倉北区', '八幡東区', '小倉南区', '八幡西区'],
            '福岡市': ['東区', '博多区', '中央区', '南区', '西区', '城南区', '早良区'],
            '熊本市': ['中央区', '東区', '西区', '南区', '北区'],
        }
        
        # 加载缓存
        self.cache = self.load_cache()
    
    def load_cache(self):
        """加载下载缓存"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {'downloaded_files': set(), 'prefecture_data': {}}
    
    def save_cache(self):
        """保存下载缓存"""
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_page_content(self, url, encoding=None):
        """获取页面内容，尝试多种编码"""
        try:
            response = self.session.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            # 尝试多种编码
            encodings = ['shift_jis', 'utf-8', 'euc-jp', 'iso-8859-1', 'cp932', 'windows-1252']
            if encoding:
                encodings.insert(0, encoding)
            
            for enc in encodings:
                try:
                    content = response.content.decode(enc)
                    return content
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用errors='ignore'
            return response.content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"获取页面失败 {url}: {e}")
            return None
    
    def extract_hierarchy_from_html(self, soup, html_url, pref_name):
        """从HTML页面提取层级信息"""
        hierarchy = {
            'prefecture': pref_name,
            'city': '',
            'district': ''
        }
        
        # 方法1: 从导航层级提取（最准确）
        navi_level = soup.find('div', class_='navi__level')
        if navi_level:
            navi_text = navi_level.get_text()
            logger.debug(f"导航层级文本: {navi_text}")
            
            # 提取市区町村名
            city_pattern = r'([^市区町村]+[市区町村])'
            city_matches = re.findall(city_pattern, navi_text)
            
            if city_matches:
                city_name = city_matches[0]
                
                # 检查是否是政令指定都市
                if city_name in self.major_cities:
                    # 查找具体的区名
                    for ward in self.major_cities[city_name]:
                        if ward in navi_text:
                            hierarchy['city'] = city_name
                            hierarchy['district'] = ward
                            logger.info(f"政令指定都市: {city_name} -> {ward}")
                            return hierarchy
                    
                    # 如果没有找到具体区名，可能是市级别页面
                    hierarchy['city'] = city_name
                    logger.info(f"政令指定都市市级别: {city_name}")
                else:
                    # 普通市区町村
                    hierarchy['city'] = city_name
                    logger.info(f"普通市区町村: {city_name}")
        
        # 方法2: 从页面标题提取
        title = soup.find('title')
        if title and not hierarchy['city']:
            title_text = title.get_text()
            city_pattern = r'([^市区町村]+[市区町村])'
            city_matches = re.findall(city_pattern, title_text)
            if city_matches:
                hierarchy['city'] = city_matches[0]
                logger.info(f"从标题提取市区町村名: {hierarchy['city']}")
        
        # 方法3: 从H1标题提取
        h1 = soup.find('h1')
        if h1 and not hierarchy['city']:
            h1_text = h1.get_text()
            city_pattern = r'([^市区町村]+[市区町村])'
            city_matches = re.findall(city_pattern, h1_text)
            if city_matches:
                hierarchy['city'] = city_matches[0]
                logger.info(f"从H1提取市区町村名: {hierarchy['city']}")
        
        # 验证层级信息
        if not hierarchy['city']:
            logger.warning(f"无法提取市区町村名: {html_url}")
            return None
        
        logger.info(f"最终层级信息: {hierarchy}")
        return hierarchy
    
    def build_save_path(self, hierarchy, pdf_name):
        """构建保存路径"""
        if not hierarchy or not hierarchy['city']:
            logger.error(f"层级信息不完整: {hierarchy}")
            return None
        
        # 构建保存路径
        save_path_parts = [DATA_DIR]
        
        # 添加县名
        save_path_parts.append(Path(hierarchy['prefecture']))
        
        # 添加市区町村名
        save_path_parts.append(Path(hierarchy['city']))
        
        # 添加区名（如果有的话）
        if hierarchy['district']:
            save_path_parts.append(Path(hierarchy['district']))
        
        # 创建目录
        save_dir = Path(*save_path_parts)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建完整路径
        save_path = save_dir / pdf_name
        
        return save_path
    
    def download_pdf(self, pdf_url, save_path):
        """下载PDF文件"""
        try:
            # 检查是否已存在
            if save_path.exists():
                logger.info(f"已存在，跳过: {save_path}")
                return True
            
            # 下载文件
            response = self.session.get(pdf_url, timeout=30, verify=False)
            response.raise_for_status()
            
            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"下载: {pdf_url} -> {save_path}")
            
            # 更新缓存
            self.cache['downloaded_files'].add(str(save_path))
            
            return True
            
        except Exception as e:
            logger.error(f"下载失败 {pdf_url}: {e}")
            return False
    
    def process_pdf_page(self, html_url, pref_name):
        """处理PDF页面"""
        try:
            # 获取页面内容
            content = self.get_page_content(html_url)
            if not content:
                return []
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # 提取层级信息
            hierarchy = self.extract_hierarchy_from_html(soup, html_url, pref_name)
            if not hierarchy:
                return []
            
            # 查找PDF链接
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    pdf_url = urljoin(html_url, href)
                    pdf_name = os.path.basename(href)
                    
                    # 构建保存路径
                    save_path = self.build_save_path(hierarchy, pdf_name)
                    if save_path:
                        pdf_links.append((pdf_url, save_path))
            
            return pdf_links
            
        except Exception as e:
            logger.error(f"处理PDF页面失败 {html_url}: {e}")
            return []
    
    def process_city_page(self, city_url, pref_name):
        """处理市区町村页面"""
        try:
            # 获取页面内容
            content = self.get_page_content(city_url)
            if not content:
                return []
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # 查找PDF页面链接 - 只处理包含PDF链接的页面
            pdf_pages = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                # 只处理以.htm结尾且不包含年份信息的链接
                if (href.endswith('.htm') and 
                    not href.startswith('http') and
                    not any(year in href for year in ['r07', 'r06', 'r05', 'r04', 'r03', 'r02', 'r01', 'h30']) and
                    'fr.htm' in href):  # 确保是PDF页面
                    pdf_page_url = urljoin(city_url, href)
                    pdf_pages.append(pdf_page_url)
            
            return pdf_pages
            
        except Exception as e:
            logger.error(f"处理市区町村页面失败 {city_url}: {e}")
            return []
    
    def process_prefecture(self, pref_name, pref_url):
        """处理县页面"""
        try:
            logger.info(f"处理: {pref_name} {pref_url}")
            
            # 获取页面内容
            content = self.get_page_content(pref_url)
            if not content:
                return []
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # 查找市区町村链接 - 只处理当前年份的链接
            city_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                # 只处理包含city_frm.htm的链接，这些是市区町村选择页面
                if (href.endswith('.htm') and 
                    not href.startswith('http') and
                    'city_frm.htm' in href):
                    city_url = urljoin(pref_url, href)
                    city_links.append(city_url)
            
            return city_links
            
        except Exception as e:
            logger.error(f"处理县页面失败 {pref_url}: {e}")
            return []
    
    def download_all(self):
        """下载所有数据"""
        all_tasks = []
        
        # 处理所有县
        for pref_name, pref_url in PREFECTURE_URL_MAPPING['main_r07'].items():
            full_url = f"https://www.rosenka.nta.go.jp/{pref_url}"
            
            # 获取市区町村链接
            city_links = self.process_prefecture(pref_name, full_url)
            
            # 处理每个市区町村
            for city_url in city_links:
                pdf_pages = self.process_city_page(city_url, pref_name)
                
                # 处理每个PDF页面
                for pdf_page_url in pdf_pages:
                    pdf_links = self.process_pdf_page(pdf_page_url, pref_name)
                    all_tasks.extend(pdf_links)
        
        logger.info(f"共发现 {len(all_tasks)} 个PDF下载任务")
        
        # 并发下载
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for pdf_url, save_path in all_tasks:
                future = executor.submit(self.download_pdf, pdf_url, save_path)
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"下载任务失败: {e}")
        
        # 保存缓存
        self.save_cache()
        logger.info("下载完成")

if __name__ == "__main__":
    downloader = RosenkaDownloader()
    downloader.download_all() 