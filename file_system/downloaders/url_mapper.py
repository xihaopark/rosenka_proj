#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
url_mapper.py
自动抓取国税庁路線価図官网的真实URL映射
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import logging
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class URLMapper:
    """URL映射器 - 自动抓取真实URL映射"""
    
    def __init__(self):
        self.base_url = "https://www.rosenka.nta.go.jp"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def decode_content(self, content: bytes) -> str:
        """内容解码 - 专门处理日文网站的编码"""
        encodings = ['shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp', 'utf-8']
        
        for encoding in encodings:
            try:
                decoded = content.decode(encoding)
                if (re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', decoded) or 
                    '<html' in decoded.lower()):
                    logger.info(f"成功使用 {encoding} 编码解码内容")
                    return decoded
            except (UnicodeDecodeError, LookupError):
                continue
        
        logger.warning("所有编码尝试失败，使用shift_jis强制解码")
        return content.decode('shift_jis', errors='ignore')
    
    def fetch_prefecture_mapping(self, year: str = "main_r07") -> Dict[str, str]:
        """抓取都道府県的真实URL映射"""
        logger.info(f"开始抓取 {year} 年的都道府県URL映射...")
        
        url = f"{self.base_url}/{year}/index.htm"
        logger.info(f"访问URL: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            html = self.decode_content(response.content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # 查找所有都道府県链接
            prefecture_links = soup.find_all('a', href=re.compile(r'.*pref_frm\.htm$'))
            
            mapping = {}
            for link in prefecture_links:
                href = link['href']
                text = link.get_text(strip=True)
                
                # 清理URL路径
                if href.startswith('./'):
                    href = href[2:]
                elif href.startswith('/'):
                    href = href[1:]
                
                # 构建完整URL路径
                full_path = f"{year}/{href}"
                
                mapping[text] = full_path
                logger.info(f"映射: {text} -> {full_path}")
            
            logger.info(f"成功抓取 {len(mapping)} 个都道府県的URL映射")
            return mapping
            
        except Exception as e:
            logger.error(f"抓取都道府県映射失败: {e}")
            return {}
    
    def test_prefecture_url(self, prefecture: str, url_path: str) -> bool:
        """测试都道府県URL是否可访问"""
        test_url = f"{self.base_url}/{url_path}"
        try:
            response = self.session.get(test_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def generate_complete_mapping(self) -> Dict[str, Dict[str, str]]:
        """生成完整的URL映射表"""
        logger.info("开始生成完整的URL映射表...")
        
        # 抓取最新年份的映射
        current_mapping = self.fetch_prefecture_mapping("main_r07")
        
        # 测试每个URL
        valid_mapping = {}
        for prefecture, url_path in current_mapping.items():
            if self.test_prefecture_url(prefecture, url_path):
                valid_mapping[prefecture] = url_path
                logger.info(f"✅ {prefecture}: {url_path}")
            else:
                logger.warning(f"❌ {prefecture}: {url_path} - 无法访问")
        
        # 生成完整的映射表
        complete_mapping = {
            "main_r07": valid_mapping,  # 令和7年 (2025年)
            "main_r06": {},  # 令和6年 (2024年) - 如果需要可以单独抓取
            "main_r05": {},  # 令和5年 (2023年) - 如果需要可以单独抓取
        }
        
        return complete_mapping
    
    def save_mapping(self, mapping: Dict, filename: str = "prefecture_url_mapping.json"):
        """保存映射表到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"映射表已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存映射表失败: {e}")
    
    def load_mapping(self, filename: str = "prefecture_url_mapping.json") -> Dict:
        """从文件加载映射表"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            logger.info(f"映射表已从 {filename} 加载")
            return mapping
        except Exception as e:
            logger.error(f"加载映射表失败: {e}")
            return {}

def main():
    """主函数"""
    print("🗾 国税庁路線価図URL映射抓取器")
    print("=" * 50)
    
    mapper = URLMapper()
    
    # 生成完整映射表
    complete_mapping = mapper.generate_complete_mapping()
    
    # 保存映射表
    mapper.save_mapping(complete_mapping)
    
    # 显示统计信息
    current_mapping = complete_mapping.get("main_r07", {})
    print(f"\n📊 统计信息:")
    print(f"   总都道府県数: {len(current_mapping)}")
    print(f"   有效URL数: {len([v for v in current_mapping.values() if v])}")
    
    print(f"\n📋 当前映射表 (令和7年):")
    for prefecture, url_path in current_mapping.items():
        print(f"   {prefecture}: {url_path}")
    
    print(f"\n✅ 映射表已保存到: prefecture_url_mapping.json")
    print(f"💡 现在可以更新下载器使用这些真实URL映射")

if __name__ == "__main__":
    main() 