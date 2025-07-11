"""
路線価図专门测试
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from app_sam.models.rosenka_specific_detector import RosenkaSpecificDetector
from app_sam.models.sam_text_segmentation import SAMTextSegmenter
from app_sam.config import MODEL_CONFIG

def test_rosenka_detection():
    """测试路線価図检测"""
    # 1. 加载测试图像
    test_pdf = Path("/mnt/data1/park/AI_park/rosenka_proj/rosenka_data/静岡県/富士宮市/山宮/31007.pdf")
    
    # 转换第一页为图像
    import fitz
    doc = fitz.open(str(test_pdf))
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    doc.close()
    
    print(f"图像尺寸: {img.shape}")
    
    # 2. 传统方法检测
    print("\n=== 传统方法检测 ===")
    detector = RosenkaSpecificDetector()
    candidates = detector.detect_text_candidates(img)
    print(f"检测到候选区域: {len(candidates)}")
    
    # 显示部分结果
    for i, cand in enumerate(candidates[:5]):
        print(f"  区域{i+1}: {cand['type']} at {cand['bbox']}")
    
    # 3. SAM检测（使用新参数）
    print("\n=== SAM检测 ===")
    sam = SAMTextSegmenter(
        model_type=MODEL_CONFIG['model_type'],
        checkpoint_path=str(MODEL_CONFIG['sam_checkpoint'])
    )
    
    # 使用混合方法
    regions = sam.segment_page_hybrid(img)
    print(f"SAM检测到区域: {len(regions)}")
    
    # 4. 保存可视化结果
    vis_img = img.copy()
    
    # 绘制检测结果
    for region in regions[:50]:  # 只画前50个
        if 'bbox' in region:
            x, y, w, h = region['bbox']
            color = (0, 255, 0) if 'traditional' in region.get('region_type', '') else (255, 0, 0)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
    
    # 保存结果
    output_path = Path("test_rosenka_detection_result.jpg")
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"\n可视化结果保存到: {output_path}")
    
    return len(regions) > 0

if __name__ == "__main__":
    success = test_rosenka_detection()
    print(f"\n测试{'成功' if success else '失败'}")