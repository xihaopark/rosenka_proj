"""
集成测试 - 验证完整系统
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app_sam.core.sam_rosenka_pipeline import SAMRosenkaPipeline
from app_sam.config import MODEL_CONFIG, DATA_DIR
from app_sam.utils.performance_monitor import monitor_performance

@monitor_performance
def test_complete_workflow():
    """测试完整工作流"""
    print("=== 路線価図SAM处理系统集成测试 ===\n")
    
    # 1. 初始化
    print("1. 初始化系统...")
    pipeline = SAMRosenkaPipeline(
        sam_checkpoint=str(MODEL_CONFIG['sam_checkpoint'])
    )
    print("✅ Pipeline初始化成功")
    
    # 2. 查找测试文件
    print("\n2. 扫描PDF文件...")
    pdf_files = list(DATA_DIR.rglob("*.pdf"))[:3]  # 最多测试3个
    print(f"✅ 找到 {len(pdf_files)} 个测试文件")
    
    # 3. 处理每个文件
    success_count = 0
    for idx, pdf_file in enumerate(pdf_files):
        print(f"\n3.{idx+1} 处理: {pdf_file.name}")
        try:
            results = pipeline.process_pdf(pdf_file)
            success_count += 1
            
            # 显示结果统计
            total_regions = sum(p['num_regions'] for p in results['pages'])
            print(f"✅ 成功 - 总区域数: {total_regions}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    # 4. 总结
    print(f"\n=== 测试完成 ===")
    print(f"成功率: {success_count}/{len(pdf_files)}")
    
    return success_count == len(pdf_files)

if __name__ == "__main__":
    success = test_complete_workflow()
    exit(0 if success else 1)