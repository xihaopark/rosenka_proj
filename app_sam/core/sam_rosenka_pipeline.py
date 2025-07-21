"""
sam_rosenka_pipeline.py
基于SAM的路線価図处理完整流程
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import fitz  # PyMuPDF
import json
import numpy as np
import cv2
from typing import List
from app_sam.utils.performance_monitor import monitor_performance
#from utils.logger import logger
from app_sam.core.text_region_processor import TextSegment, TextRegionProcessor
from app_sam.models.sam_text_segmentation import SAMTextSegmenter

class SAMRosenkaPipeline:
    """SAM路線価図处理管道"""
    
    def __init__(self, sam_checkpoint='sam_vit_h_4b8939.pth'):
        self.sam_segmenter = SAMTextSegmenter(checkpoint_path=sam_checkpoint)
        self.text_processor = TextRegionProcessor()
        self.results_cache = {}
        
    def process_pdf(self, pdf_path: Path, output_dir: Path = None):
        """处理整个PDF文件"""
        print(f"处理PDF: {pdf_path}")
        
        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_processed"
        output_dir.mkdir(exist_ok=True)
        
        # 打开PDF
        doc = fitz.open(str(pdf_path))
        
        all_results = {
            'pdf_path': str(pdf_path),
            'pages': []
        }
        
        # 处理每一页
        for page_num in range(len(doc)):
            print(f"处理第 {page_num + 1} 页...")
            
            # 转换为图像
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # SAM分割
            #sam_regions = self.sam_segmenter.segment_page(img)
            sam_regions = self.sam_segmenter.segment_page_hybrid(img) 
            
            # 细化区域
            refined_regions = self.sam_segmenter.refine_text_regions(img, sam_regions)
            
            # OCR处理
            text_segments = self.text_processor.process_sam_regions(img, refined_regions)
            
            # 创建文本映射
            text_map = self.text_processor.create_text_map(
                text_segments, 
                (img.shape[1], img.shape[0])
            )
            
            # 保存页面结果
            page_result = {
                'page_num': page_num,
                'page_size': (img.shape[1], img.shape[0]),
                'text_map': text_map,
                'num_regions': len(text_segments)
            }
            
            all_results['pages'].append(page_result)
            
            # 保存可视化结果
            self._save_visualization(img, text_segments, output_dir / f"page_{page_num}.jpg")
            
            # 保存分割结果
            self._save_segments(text_segments, output_dir / f"page_{page_num}_segments")
        
        # 保存完整结果
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        doc.close()
        print(f"处理完成，结果保存在: {output_dir}")
        
        return all_results
    
    def _save_visualization(self, image: np.ndarray, segments: List[TextSegment], output_path: Path):
        """保存可视化结果"""
        vis_img = image.copy()
        
        # 绘制所有检测到的区域
        for segment in segments:
            x, y, w, h = segment.bbox
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 添加文本标签
            if segment.text:
                label = f"{segment.text[:20]}..."
                cv2.putText(vis_img, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 保存图像
        cv2.imwrite(str(output_path), vis_img)
    
    def _save_segments(self, segments: List[TextSegment], output_dir: Path):
        """保存分割的文本片段"""
        output_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(segments):
            # 保存片段图像
            segment_path = output_dir / f"segment_{i:04d}.png"
            cv2.imwrite(str(segment_path), segment.image)
            
            # 保存元数据
            metadata = {
                'index': i,
                'text': segment.text,
                'bbox': segment.bbox,
                'confidence': segment.confidence,
                'ocr_confidence': segment.ocr_confidence
            }
            
            meta_path = output_dir / f"segment_{i:04d}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)