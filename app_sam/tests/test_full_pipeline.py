"""
完整流程测试
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app_sam.core.sam_rosenka_pipeline import SAMRosenkaPipeline
from app_sam.config import MODEL_CONFIG, DATA_DIR

def test_full_pipeline():
    """测试完整处理流程"""
    print("初始化处理管道...")
    
    pipeline = SAMRosenkaPipeline(
        sam_checkpoint=str(MODEL_CONFIG['sam_checkpoint'])
    )
    
    # 查找一个测试PDF
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    if not pdf_files:
        print("❌ 未找到测试PDF文件")
        return False
    
    test_pdf = pdf_files[0]
    print(f"使用测试文件: {test_pdf}")
    
    try:
        # 处理PDF（只处理第一页）
        results = pipeline.process_pdf(test_pdf)
        print(f"✅ 处理成功")
        print(f"   - 页数: {len(results['pages'])}")
        if results['pages']:
            print(f"   - 第一页检测到的区域: {results['pages'][0]['num_regions']}")
        return True
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_pipeline()