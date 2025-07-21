"""
coordinate_mapper.py
文本区域与坐标映射系统
"""

@dataclass
class MappedTextRegion:
    """映射后的文本区域"""
    text: str
    bbox: Tuple[int, int, int, int]  # 原始坐标
    normalized_bbox: Tuple[float, float, float, float]  # 归一化坐标
    page_num: int
    pdf_path: str
    confidence: float
    detection_score: float
    ocr_score: float
    
class CoordinateMapper:
    """坐标映射器"""
    
    def __init__(self):
        self.mappings = {}
        
    def add_mapping(self, pdf_path: str, page_num: int, 
                   text_region: TextRegion, page_size: Tuple[int, int]):
        """添加映射"""
        # 计算归一化坐标
        w, h = page_size
        x1, y1, x2, y2 = text_region.bbox
        
        normalized_bbox = (
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h
        )
        
        mapped_region = MappedTextRegion(
            text=text_region.text,
            bbox=text_region.bbox,
            normalized_bbox=normalized_bbox,
            page_num=page_num,
            pdf_path=pdf_path,
            confidence=text_region.confidence,
            detection_score=text_region.confidence,
            ocr_score=0.0  # OCR后更新
        )
        
        # 存储映射
        key = f"{pdf_path}:{page_num}"
        if key not in self.mappings:
            self.mappings[key] = []
        self.mappings[key].append(mapped_region)
        
        return mapped_region
    
    def get_regions_at_location(self, pdf_path: str, page_num: int, 
                               point: Tuple[float, float]) -> List[MappedTextRegion]:
        """获取指定位置的文本区域"""
        key = f"{pdf_path}:{page_num}"
        regions = self.mappings.get(key, [])
        
        # 查找包含该点的区域
        matching_regions = []
        for region in regions:
            x1, y1, x2, y2 = region.normalized_bbox
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                matching_regions.append(region)
        
        return matching_regions
    
    def save_mappings(self, output_path: str):
        """保存映射到文件"""
        import json
        
        # 转换为可序列化格式
        serializable = {}
        for key, regions in self.mappings.items():
            serializable[key] = [
                {
                    'text': r.text,
                    'bbox': r.bbox,
                    'normalized_bbox': r.normalized_bbox,
                    'page_num': r.page_num,
                    'confidence': r.confidence,
                    'detection_score': r.detection_score,
                    'ocr_score': r.ocr_score
                }
                for r in regions
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)