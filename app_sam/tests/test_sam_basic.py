"""
基础SAM测试
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app_sam.models.sam_text_segmentation import SAMTextSegmenter
from app_sam.config import MODEL_CONFIG
import numpy as np

def test_sam_loading():
    """测试SAM模型加载"""
    print("测试SAM模型加载...")
    
    try:
        segmenter = SAMTextSegmenter(
            model_type=MODEL_CONFIG['model_type'],
            checkpoint_path=str(MODEL_CONFIG['sam_checkpoint'])
        )
        print("✅ SAM模型加载成功")
        
        # 测试处理一个空白图像
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        regions = segmenter.segment_page(test_image)
        print(f"✅ 测试分割完成，检测到 {len(regions)} 个区域")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_sam_loading() 