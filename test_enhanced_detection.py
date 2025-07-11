#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_enhanced_detection.py
测试增强版检测效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from app.processors.enhanced_text_detector import EnhancedTextDetector
from app.processors.image_preprocessor import ImagePreprocessor
from pdf2image import convert_from_path

def visualize_enhanced_results(image, results, output_path="enhanced_detection_result.png"):
    """可视化增强检测结果"""
    # Enlarge figure size for better readability
    fig, axes = plt.subplots(2, 2, figsize=(24, 24))
    
    # Use a Japanese font if available to render characters correctly
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Different types of results
    titles = ['Horizontal Text (红色)', 'Vertical Text (蓝色)', 'Circled Numbers (绿色)', 'Small Text (黄色)']
    result_keys = ['horizontal_text', 'vertical_text', 'circled_numbers', 'small_text']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Create a copy of the image to draw on, to avoid modifying the original
    vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for idx, (ax, title, key, color) in enumerate(zip(axes.flat, titles, result_keys, colors)):
        ax.imshow(vis_image)
        ax.set_title(f'{title}: {len(results.get(key, []))} detections', fontsize=16)
        
        # Draw detection boxes
        for det in results.get(key, []):
            if 'polygon' in det:
                poly = np.array(det['polygon'])
                rect = patches.Polygon(poly, linewidth=2, edgecolor=color, facecolor='none')
            elif 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
            else:
                continue
            ax.add_patch(rect)
            
            # Add text
            if 'text' in det:
                x1 = det.get('x1', det.get('bbox', [0,0])[0])
                y1 = det.get('y1', det.get('bbox', [0,0])[1])
                ax.text(x1, y1 - 10, det['text'], 
                       color='white', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"\n✅ Visualization saved to {output_path}")

def test_on_pdf(pdf_path="43009.pdf"):
    """测试您的PDF图像"""
    print("=" * 50)
    print("     Enhanced Text Detection Test")
    print("=" * 50)
    
    # Convert PDF to image
    print(f"\n1. Converting '{pdf_path}' to image (DPI=300)...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
        image = images[0]
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error converting PDF to image: {e}")
        print("   Please ensure you have poppler installed (`conda install -c conda-forge poppler`).")
        return
    
    print(f"   Original image dimensions: {image.shape}")
    
    # Preprocess
    print("\n2. Preprocessing image...")
    preprocessor = ImagePreprocessor()
    enhanced_image = preprocessor.enhance_low_resolution(image)
    print(f"   Enhanced image dimensions: {enhanced_image.shape}")
    
    # Detect
    print("\n3. Running enhanced detection (this may take a while)...")
    detector = EnhancedTextDetector(use_trocr=True)
    results = detector.detect_all_enhanced(enhanced_image)
    
    # Statistics
    print("\n4. Detection statistics:")
    for key, detections in results.items():
        print(f"   - Found {len(detections):>4} regions for '{key}'")
    
    # Visualize
    print("\n5. Visualizing results...")
    visualize_enhanced_results(enhanced_image, results)

if __name__ == "__main__":
    # Test your PDF file
    test_on_pdf("43009.pdf") 