#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database_manager.py
数据库管理器 - 路線価図検索システム
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "route_price_maps.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 创建PDF文件表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        filepath TEXT UNIQUE NOT NULL,
                        total_pages INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建页面表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pdf_id INTEGER NOT NULL,
                        page_number INTEGER NOT NULL,
                        image_width INTEGER NOT NULL,
                        image_height INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pdf_id) REFERENCES pdf_files (id)
                    )
                """)
                
                # 创建文本区域表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS text_regions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        page_id INTEGER NOT NULL,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        engine TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (page_id) REFERENCES pages (id)
                    )
                """)
                
                # 创建圆形检测表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS circles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        page_id INTEGER NOT NULL,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        radius INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (page_id) REFERENCES pages (id)
                    )
                """)
                
                # 创建索引
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pdf_files_path ON pdf_files(filepath)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_pdf ON pages(pdf_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_text_regions_page ON text_regions(page_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_circles_page ON circles(page_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_text_regions_text ON text_regions(text)")
                
                conn.commit()
                logger.info("数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def add_pdf_file(self, filename: str, filepath: str, total_pages: int) -> int:
        """添加PDF文件记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO pdf_files 
                    (filename, filepath, total_pages)
                    VALUES (?, ?, ?)
                """, (filename, filepath, total_pages))
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"添加PDF文件失败: {e}")
            raise
    
    def add_page(self, pdf_id: int, page_number: int, image_width: int, image_height: int) -> int:
        """添加页面记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO pages 
                    (pdf_id, page_number, image_width, image_height)
                    VALUES (?, ?, ?, ?)
                """, (pdf_id, page_number, image_width, image_height))
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"添加页面失败: {e}")
            raise
    
    def add_text_region(self, page_id: int, x: int, y: int, width: int, height: int, 
                       text: str, confidence: float, engine: str):
        """添加文本区域"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO text_regions 
                    (page_id, x, y, width, height, text, confidence, engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (page_id, x, y, width, height, text, confidence, engine))
                
                conn.commit()
        except Exception as e:
            logger.error(f"添加文本区域失败: {e}")
            raise
    
    def add_circle(self, page_id: int, x: int, y: int, radius: int, confidence: float):
        """添加圆形检测"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO circles 
                    (page_id, x, y, radius, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (page_id, x, y, radius, confidence))
                
                conn.commit()
        except Exception as e:
            logger.error(f"添加圆形检测失败: {e}")
            raise
    
    def get_pdf_files(self) -> List[Dict]:
        """获取所有PDF文件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM pdf_files ORDER BY created_at DESC")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取PDF文件失败: {e}")
            return []
    
    def get_text_regions(self, pdf_id: Optional[int] = None, page_id: Optional[int] = None) -> List[Dict]:
        """获取文本区域"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if pdf_id:
                    cursor = conn.execute("""
                        SELECT tr.*, p.page_number, pf.filename 
                        FROM text_regions tr
                        JOIN pages p ON tr.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        WHERE p.pdf_id = ?
                        ORDER BY p.page_number, tr.y, tr.x
                    """, (pdf_id,))
                elif page_id:
                    cursor = conn.execute("""
                        SELECT tr.*, p.page_number, pf.filename 
                        FROM text_regions tr
                        JOIN pages p ON tr.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        WHERE tr.page_id = ?
                        ORDER BY tr.y, tr.x
                    """, (page_id,))
                else:
                    cursor = conn.execute("""
                        SELECT tr.*, p.page_number, pf.filename 
                        FROM text_regions tr
                        JOIN pages p ON tr.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        ORDER BY pf.filename, p.page_number, tr.y, tr.x
                    """)
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取文本区域失败: {e}")
            return []
    
    def get_circles(self, pdf_id: Optional[int] = None, page_id: Optional[int] = None) -> List[Dict]:
        """获取圆形检测"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if pdf_id:
                    cursor = conn.execute("""
                        SELECT c.*, p.page_number, pf.filename 
                        FROM circles c
                        JOIN pages p ON c.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        WHERE p.pdf_id = ?
                        ORDER BY p.page_number, c.y, c.x
                    """, (pdf_id,))
                elif page_id:
                    cursor = conn.execute("""
                        SELECT c.*, p.page_number, pf.filename 
                        FROM circles c
                        JOIN pages p ON c.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        WHERE c.page_id = ?
                        ORDER BY c.y, c.x
                    """, (page_id,))
                else:
                    cursor = conn.execute("""
                        SELECT c.*, p.page_number, pf.filename 
                        FROM circles c
                        JOIN pages p ON c.page_id = p.id
                        JOIN pdf_files pf ON p.pdf_id = pf.id
                        ORDER BY pf.filename, p.page_number, c.y, c.x
                    """)
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取圆形检测失败: {e}")
            return []
    
    def search_text(self, query: str, limit: int = 100) -> List[Dict]:
        """搜索文本"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT tr.*, p.page_number, pf.filename 
                    FROM text_regions tr
                    JOIN pages p ON tr.page_id = p.id
                    JOIN pdf_files pf ON p.pdf_id = pf.id
                    WHERE tr.text LIKE ?
                    ORDER BY tr.confidence DESC
                    LIMIT ?
                """, (f"%{query}%", limit))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"搜索文本失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # PDF文件数量
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_files")
                pdf_count = cursor.fetchone()[0]
                
                # 页面数量
                cursor = conn.execute("SELECT COUNT(*) FROM pages")
                page_count = cursor.fetchone()[0]
                
                # 文本区域数量
                cursor = conn.execute("SELECT COUNT(*) FROM text_regions")
                text_region_count = cursor.fetchone()[0]
                
                # 圆形检测数量
                cursor = conn.execute("SELECT COUNT(*) FROM circles")
                circle_count = cursor.fetchone()[0]
                
                # 平均置信度
                cursor = conn.execute("SELECT AVG(confidence) FROM text_regions")
                avg_confidence = cursor.fetchone()[0] or 0
                
                return {
                    'pdf_files': pdf_count,
                    'pages': page_count,
                    'text_regions': text_region_count,
                    'circles': circle_count,
                    'avg_confidence': round(avg_confidence, 3)
                }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def clear_database(self):
        """清空数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM circles")
                conn.execute("DELETE FROM text_regions")
                conn.execute("DELETE FROM pages")
                conn.execute("DELETE FROM pdf_files")
                conn.commit()
                logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            raise 