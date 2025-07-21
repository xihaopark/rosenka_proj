#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_circle_detection_images.py
生成小圆圈检测结果图片，用于肉眼检查
"""

import os
import json
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from japanese_visual_search import is_strict_circle_detection, load_ocr_data, calculate_header_height
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_circle_detection_images():
    """为藤白台1文件夹中的所有PDF生成圆圈检测结果图片"""
    
    # 目标文件夹
    target_folder = "rosenka_data/大阪府/吹田市/藤白台１"
    output_folder = "circle_detection_results"
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"🔍 开始处理文件夹: {target_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    
    # 加载OCR数据
    try:
        ocr_data = load_ocr_data()
        print(f"📊 加载了 {len(ocr_data)} 条OCR数据")
    except Exception as e:
        print(f"❌ OCR数据加载失败: {e}")
        return
    
    # 按PDF文件分组OCR数据
    pdf_ocr_data = {}
    for item in ocr_data:
        pdf_path = item.get('pdf_path', '')
        if pdf_path:
            if pdf_path not in pdf_ocr_data:
                pdf_ocr_data[pdf_path] = []
            pdf_ocr_data[pdf_path].append(item)
    
    print(f"📄 找到 {len(pdf_ocr_data)} 个PDF文件的OCR数据")
    
    # 处理藤白台1文件夹中的PDF
    processed_count = 0
    for pdf_path, detections in pdf_ocr_data.items():
        if "藤白台１" in pdf_path:
            print(f"\n🔄 处理PDF: {pdf_path}")
            
            # 过滤出严格圆圈检测结果
            strict_circles = [d for d in detections if is_strict_circle_detection(d)]
            all_circles = [d for d in detections if d.get('detection_type') == 'circle' or d.get('type') == 'circle']
            
            print(f"   📝 总检测结果: {len(detections)}")
            print(f"   🔴 标记为圆圈: {len(all_circles)}")
            print(f"   ✅ 严格圆圈: {len(strict_circles)}")
            
            # 生成可视化图片
            try:
                generate_visualization_image(pdf_path, all_circles, strict_circles, output_folder)
                processed_count += 1
            except Exception as e:
                print(f"   ❌ 生成图片失败: {e}")
    
    print(f"\n🎉 处理完成！共处理了 {processed_count} 个PDF文件")
    print(f"📁 检查结果图片请查看: {output_folder}/")

def generate_visualization_image(pdf_path: str, all_circles: list, strict_circles: list, output_folder: str):
    """为单个PDF生成圆圈检测可视化图片"""
    
    if not os.path.exists(pdf_path):
        print(f"   ⚠️ PDF文件不存在: {pdf_path}")
        return
    
    # 打开PDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    
    # 高分辨率转换
    mat = fitz.Matrix(2.0, 2.0)  # 2倍放大便于查看
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # 转换为OpenCV格式
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    doc.close()
    
    # 去除表头
    header_height = calculate_header_height(image.shape[0])
    image = image[header_height:, :]
    
    # 创建两个副本：一个显示所有圆圈，一个显示严格圆圈
    image_all = image.copy()
    image_strict = image.copy()
    
    # 绘制所有标记为圆圈的检测结果（蓝色）
    for detection in all_circles:
        bbox = detection.get('bbox', [])
        text = detection.get('text', '')
        confidence = detection.get('confidence', 0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # 调整坐标（去除表头）
            y1 = max(0, y1 - header_height)
            y2 = max(0, y2 - header_height)
            
            # 绘制蓝色边框
            cv2.rectangle(image_all, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 添加文本标签
            label = f"{text} ({confidence:.2f})"
            cv2.putText(image_all, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 绘制严格圆圈检测结果（红色）
    for detection in strict_circles:
        bbox = detection.get('bbox', [])
        text = detection.get('text', '')
        confidence = detection.get('confidence', 0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # 调整坐标（去除表头）
            y1 = max(0, y1 - header_height)
            y2 = max(0, y2 - header_height)
            
            # 绘制红色边框
            cv2.rectangle(image_strict, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 添加文本标签
            label = f"{text} ({confidence:.2f})"
            cv2.putText(image_strict, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 生成输出文件名
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    
    # 保存图片
    all_circles_path = os.path.join(output_folder, f"{pdf_name}_all_circles.jpg")
    strict_circles_path = os.path.join(output_folder, f"{pdf_name}_strict_circles.jpg")
    
    cv2.imwrite(all_circles_path, image_all)
    cv2.imwrite(strict_circles_path, image_strict)
    
    print(f"   ✅ 保存图片:")
    print(f"      📄 所有圆圈: {all_circles_path}")
    print(f"      🔴 严格圆圈: {strict_circles_path}")
    
    # 生成统计信息
    stats_path = os.path.join(output_folder, f"{pdf_name}_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"PDF文件: {pdf_path}\n")
        f.write(f"图像尺寸: {image.shape[1]} x {image.shape[0]}\n")
        f.write(f"表头高度: {header_height}px\n")
        f.write(f"总检测结果: {len(all_circles + [d for d in all_circles if d not in all_circles])}\n")
        f.write(f"标记为圆圈: {len(all_circles)}\n")
        f.write(f"严格圆圈: {len(strict_circles)}\n")
        f.write(f"过滤比例: {((len(all_circles) - len(strict_circles)) / len(all_circles) * 100):.1f}%\n\n")
        
        f.write("严格圆圈检测结果:\n")
        for i, detection in enumerate(strict_circles):
            text = detection.get('text', '')
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0)
            if len(bbox) == 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                f.write(f"  {i+1}. 文本: '{text}' | 尺寸: {width}x{height} | 面积: {area} | 信頼度: {confidence:.3f}\n")

if __name__ == "__main__":
    generate_circle_detection_images() 