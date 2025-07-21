#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_scene_text_detection.py
测试PaddleOCR场景文字检测效果
"""
import fitz
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from app.processors.rosenka_text_detection_system import RosenkaTextDetectionSystem

def visualize_detections(image, detections, title="Text Detection Results"):
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    ax = plt.gca()
    color_map = {
        'numbers': 'red',
        'japanese': 'blue',
        'mixed': 'yellow',
        'english': 'green'
    }
    for text_type, dets in detections.items():
        for box in dets:
            x1, y1, x2, y2 = box['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color_map.get(text_type, 'white'), facecolor='none')
            ax.add_patch(rect)
    plt.axis('off')
    plt.savefig('test_scene_text_detection_result.jpg')
    plt.close()

def main():
    pdf_path = Path('rosenka_data/大阪府/吹田市/藤白台１/43009.pdf')
    if not pdf_path.exists():
        print(f"PDF文件不存在: {pdf_path}")
        return
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    doc.close()
    print(f"图像尺寸: {img.shape}")
    detector = RosenkaTextDetectionSystem()
    detections = detector.detect_all_text(img)
    for k, v in detections.items():
        print(f"{k}: {len(v)} 区域")
    visualize_detections(img, detections, "PaddleOCR场景文字检测结果")
    print("可视化结果已保存为 test_scene_text_detection_result.jpg")

if __name__ == "__main__":
    main() 