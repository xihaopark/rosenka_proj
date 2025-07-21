#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleanup_file_structure.py
清理和重新组织路線価図文件结构
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileStructureCleanup:
    def __init__(self, data_dir: str = "./rosenka_data"):
        self.data_dir = Path(data_dir)
        
        # 政令指定都市的区映射
        self.city_ward_mapping = {
            # 大阪府
            "大阪府": {
                "大阪市": [
                    "阿倍野区", "旭区", "港区", "此花区", "住吉区", "住之江区", 
                    "城東区", "生野区", "西区", "西成区", "西淀川区", "大正区",
                    "中央区", "鶴見区", "天王寺区", "都島区", "東住吉区", 
                    "東成区", "東淀川区", "福島区", "平野区", "北区", "淀川区", "浪速区"
                ],
                "堺市": [
                    "堺区", "中区", "東区", "西区", "南区", "北区", "美原区"
                ]
            },
            # 東京都 (特別区)
            "東京都": {
                "東京都": [  # 特別区は東京都直下でも可だが、統一のため
                    "千代田区", "中央区", "港区", "新宿区", "文京区", "台東区",
                    "墨田区", "江東区", "品川区", "目黒区", "大田区", "世田谷区",
                    "渋谷区", "中野区", "杉並区", "豊島区", "北区", "荒川区",
                    "板橋区", "練馬区", "足立区", "葛飾区", "江戸川区"
                ]
            },
            # 神奈川県
            "神奈川県": {
                "横浜市": [
                    "鶴見区", "神奈川区", "西区", "中区", "南区", "保土ケ谷区",
                    "磯子区", "金沢区", "港北区", "戸塚区", "港南区", "旭区",
                    "緑区", "瀬谷区", "栄区", "泉区", "青葉区", "都筑区"
                ],
                "川崎市": [
                    "川崎区", "幸区", "中原区", "高津区", "多摩区", "宮前区", "麻生区"
                ],
                "相模原市": [
                    "緑区", "中央区", "南区"
                ]
            },
            # 愛知県
            "愛知県": {
                "名古屋市": [
                    "千種区", "東区", "北区", "西区", "中村区", "中区", "昭和区",
                    "瑞穂区", "熱田区", "中川区", "港区", "南区", "守山区",
                    "緑区", "名東区", "天白区"
                ]
            },
            # 京都府
            "京都府": {
                "京都市": [
                    "北区", "上京区", "左京区", "中京区", "東山区", "下京区",
                    "南区", "右京区", "伏見区", "山科区", "西京区"
                ]
            },
            # 兵庫県
            "兵庫県": {
                "神戸市": [
                    "東灘区", "灘区", "兵庫区", "長田区", "須磨区", "垂水区",
                    "北区", "中央区", "西区"
                ]
            },
            # 福岡県
            "福岡県": {
                "福岡市": [
                    "東区", "博多区", "中央区", "南区", "西区", "城南区", "早良区"
                ],
                "北九州市": [
                    "門司区", "若松区", "戸畑区", "小倉北区", "小倉南区", "八幡東区", "八幡西区"
                ]
            },
            # 北海道
            "北海道": {
                "札幌市": [
                    "中央区", "北区", "東区", "白石区", "豊平区", "南区",
                    "西区", "厚別区", "手稲区", "清田区"
                ]
            },
            # 宮城県
            "宮城県": {
                "仙台市": [
                    "青葉区", "宮城野区", "若林区", "太白区", "泉区"
                ]
            },
            # 埼玉県
            "埼玉県": {
                "さいたま市": [
                    "西区", "北区", "大宮区", "見沼区", "中央区", "桜区",
                    "浦和区", "南区", "緑区", "岩槻区"
                ]
            },
            # 千葉県
            "千葉県": {
                "千葉市": [
                    "中央区", "花見川区", "稲毛区", "若葉区", "緑区", "美浜区"
                ]
            },
            # 静岡県
            "静岡県": {
                "静岡市": [
                    "葵区", "駿河区", "清水区"
                ],
                "浜松市": [
                    "中区", "東区", "西区", "南区", "北区", "浜北区", "天竜区"
                ]
            },
            # 新潟県
            "新潟県": {
                "新潟市": [
                    "北区", "東区", "中央区", "江南区", "秋葉区", "南区", "西区", "西蒲区"
                ]
            },
            # 熊本県
            "熊本県": {
                "熊本市": [
                    "中央区", "東区", "西区", "南区", "北区"
                ]
            }
        }
    
    def scan_misplaced_files(self) -> Dict[str, List[str]]:
        """扫描错误放置的文件"""
        misplaced = {}
        
        for prefecture in self.data_dir.iterdir():
            if not prefecture.is_dir():
                continue
                
            prefecture_name = prefecture.name
            if prefecture_name not in self.city_ward_mapping:
                continue
            
            logger.info(f"扫描 {prefecture_name}...")
            misplaced[prefecture_name] = []
            
            # 获取该都道府県下的所有目录
            for item in prefecture.iterdir():
                if not item.is_dir():
                    continue
                
                item_name = item.name
                
                # 检查是否是应该归属于某个市的区
                for city, wards in self.city_ward_mapping[prefecture_name].items():
                    if item_name in wards:
                        # 检查是否已经在正确位置
                        correct_path = prefecture / city / item_name
                        if item != correct_path:
                            misplaced[prefecture_name].append({
                                "ward": item_name,
                                "current_path": str(item),
                                "correct_city": city,
                                "correct_path": str(correct_path)
                            })
                            logger.warning(f"发现错误位置: {item} -> 应该在 {correct_path}")
                        break
        
        return misplaced
    
    def move_files(self, misplaced: Dict[str, List[str]], dry_run: bool = True):
        """移动文件到正确位置"""
        total_moved = 0
        
        for prefecture_name, items in misplaced.items():
            if not items:
                logger.info(f"{prefecture_name}: 无需移动")
                continue
            
            logger.info(f"处理 {prefecture_name}: {len(items)} 个区需要移动")
            
            for item in items:
                current_path = Path(item["current_path"])
                correct_path = Path(item["correct_path"])
                ward_name = item["ward"]
                city_name = item["correct_city"]
                
                if not current_path.exists():
                    logger.warning(f"源路径不存在: {current_path}")
                    continue
                
                if dry_run:
                    logger.info(f"[DRY RUN] 移动: {current_path} -> {correct_path}")
                    # 统计文件数量
                    file_count = sum(1 for _ in current_path.rglob("*.pdf"))
                    logger.info(f"[DRY RUN] 包含 {file_count} 个PDF文件")
                else:
                    try:
                        # 创建目标目录
                        correct_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 移动目录
                        logger.info(f"移动: {current_path} -> {correct_path}")
                        shutil.move(str(current_path), str(correct_path))
                        
                        # 统计文件数量
                        file_count = sum(1 for _ in correct_path.rglob("*.pdf"))
                        logger.info(f"成功移动 {ward_name} 到 {city_name}，包含 {file_count} 个PDF文件")
                        total_moved += 1
                        
                    except Exception as e:
                        logger.error(f"移动失败 {current_path} -> {correct_path}: {e}")
        
        if not dry_run:
            logger.info(f"总共成功移动 {total_moved} 个区")
        
        return total_moved
    
    def cleanup_empty_directories(self, dry_run: bool = True):
        """清理空目录"""
        removed_count = 0
        
        for prefecture in self.data_dir.iterdir():
            if not prefecture.is_dir():
                continue
            
            # 递归查找空目录
            for root, dirs, files in os.walk(prefecture, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        # 检查目录是否为空
                        if not any(dir_path.iterdir()):
                            if dry_run:
                                logger.info(f"[DRY RUN] 删除空目录: {dir_path}")
                            else:
                                logger.info(f"删除空目录: {dir_path}")
                                dir_path.rmdir()
                                removed_count += 1
                    except OSError:
                        # 目录不为空或其他错误
                        pass
        
        if not dry_run:
            logger.info(f"总共删除 {removed_count} 个空目录")
        
        return removed_count
    
    def generate_report(self, misplaced: Dict[str, List[str]]):
        """生成清理报告"""
        report_file = "cleanup_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("路線価図文件结构清理报告\n")
            f.write("=" * 50 + "\n\n")
            
            total_issues = sum(len(items) for items in misplaced.values())
            f.write(f"总计发现 {total_issues} 个位置错误的区\n\n")
            
            for prefecture_name, items in misplaced.items():
                if not items:
                    continue
                
                f.write(f"{prefecture_name}: {len(items)} 个区需要移动\n")
                f.write("-" * 30 + "\n")
                
                # 按城市分组
                by_city = {}
                for item in items:
                    city = item["correct_city"]
                    if city not in by_city:
                        by_city[city] = []
                    by_city[city].append(item["ward"])
                
                for city, wards in by_city.items():
                    f.write(f"  {city}: {', '.join(wards)}\n")
                
                f.write("\n")
        
        logger.info(f"清理报告已保存到: {report_file}")

def main():
    print("🧹 路線価図文件结构清理工具")
    print("=" * 50)
    
    cleanup = FileStructureCleanup()
    
    # 1. 扫描错误放置的文件
    print("\n📊 扫描错误放置的文件...")
    misplaced = cleanup.scan_misplaced_files()
    
    # 2. 生成报告
    cleanup.generate_report(misplaced)
    
    # 3. 显示统计信息
    total_issues = sum(len(items) for items in misplaced.values())
    print(f"\n📈 扫描结果:")
    print(f"  发现 {total_issues} 个位置错误的区")
    
    for prefecture_name, items in misplaced.items():
        if items:
            print(f"  {prefecture_name}: {len(items)} 个区")
    
    if total_issues == 0:
        print("🎉 所有文件都在正确位置！")
        return
    
    # 4. 询问是否执行移动
    print(f"\n🤔 操作选择:")
    print("1. 预览模式 (显示将要执行的操作)")
    print("2. 执行移动 (实际移动文件)")
    print("3. 取消")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n👀 预览模式:")
        cleanup.move_files(misplaced, dry_run=True)
        cleanup.cleanup_empty_directories(dry_run=True)
    elif choice == "2":
        print("\n🚀 执行移动:")
        moved = cleanup.move_files(misplaced, dry_run=False)
        removed = cleanup.cleanup_empty_directories(dry_run=False)
        print(f"\n✅ 清理完成!")
        print(f"  移动了 {moved} 个区")
        print(f"  删除了 {removed} 个空目录")
    else:
        print("❌ 操作取消")

if __name__ == "__main__":
    main() 