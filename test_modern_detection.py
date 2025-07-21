#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modern_detection.py
测试现代化文字检测方案
"""
import sys
from pathlib import Path
import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append(str(Path(__file__).parent))
from app.processors.lightweight_detector import LightweightTextDetector
from app.processors.modern_text_detector import ModernTextDetector

def visualize_detections(image, detections, title="Text Detection Results", save_path="modern_detection_result.jpg"):
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    ax = plt.gca()
    for det in detections:
        x1, y1, x2, y2 = det.bbox if hasattr(det, 'bbox') else det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        if hasattr(det, 'text') and det.text:
            plt.text(x1, y1-5, det.text, color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        elif 'text' in det and det['text']:
            plt.text(x1, y1-5, det['text'], color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    plt.axis('off')
    plt.savefig(save_path)
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
    # 方法2: 现代检测器（CRAFT+Tesseract）
    detector2 = ModernTextDetector(use_trocr=True)
    results2 = detector2.detect_and_recognize(img)
    print(f"[现代检测] 检测到 {len(results2)} 个文字区域")
    for det in results2[:5]:
        print(f"  {det.text} @ {det.bbox}")
    visualize_detections(img, results2, "现代检测结果", "modern_detection_result.jpg")
    print("可视化结果已保存为 modern_detection_result.jpg")

if __name__ == "__main__":
    main() 