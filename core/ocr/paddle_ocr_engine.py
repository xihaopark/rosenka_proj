import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from paddleocr import PaddleOCR
from .base_ocr_engine import BaseOCREngine

logger = logging.getLogger(__name__)

class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR engine for Japanese text detection and recognition.
    Optimized for route price maps with enhanced preprocessing.
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 lang: str = 'japan',
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.6,
                 rec_score_thresh: float = 0.5):
        """
        Initialize PaddleOCR engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language code ('japan' for Japanese)
            det_db_thresh: Detection threshold
            det_db_box_thresh: Detection box threshold
            rec_score_thresh: Recognition score threshold
        """
        super().__init__()
        
        self.use_gpu = use_gpu
        self.lang = lang
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.rec_score_thresh = rec_score_thresh
        
        # Initialize PaddleOCR
        try:
            self.ocr = PaddleOCR(
                use_gpu=use_gpu,
                lang=lang,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                rec_char_dict_path=None,  # Use default Japanese dict
                use_angle_cls=False  # Disable angle classification for speed
            )
            logger.info(f"PaddleOCR initialized successfully (GPU: {use_gpu}, Lang: {lang})")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR performance.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB (PaddleOCR expects RGB)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected text regions with coordinates and text
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run OCR
            results = self.ocr.ocr(processed_image, cls=False)
            
            if not results or not results[0]:
                return []
            
            detected_regions = []
            
            for line in results[0]:
                if len(line) >= 2:
                    # Extract coordinates and text
                    coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence)
                    
                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # Filter by confidence threshold
                        if confidence >= self.rec_score_thresh:
                            # Convert coordinates to integers
                            coords_int = [[int(x), int(y)] for x, y in coords]
                            
                            # Calculate bounding box
                            x_coords = [coord[0] for coord in coords_int]
                            y_coords = [coord[1] for coord in coords_int]
                            
                            bbox = {
                                'x': min(x_coords),
                                'y': min(y_coords),
                                'width': max(x_coords) - min(x_coords),
                                'height': max(y_coords) - min(y_coords)
                            }
                            
                            region = {
                                'bbox': bbox,
                                'text': text,
                                'confidence': confidence,
                                'coordinates': coords_int,
                                'engine': 'paddleocr'
                            }
                            
                            detected_regions.append(region)
            
            logger.info(f"PaddleOCR detected {len(detected_regions)} text regions")
            return detected_regions
            
        except Exception as e:
            logger.error(f"Error in PaddleOCR text detection: {e}")
            return []
    
    def recognize_text(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        Recognize text in specific regions.
        
        Args:
            image: Input image
            regions: List of regions to recognize text in
            
        Returns:
            List of regions with recognized text
        """
        # PaddleOCR already performs both detection and recognition
        # This method is kept for compatibility but returns the input regions
        return regions
    
    def get_engine_info(self) -> Dict:
        """Get information about the OCR engine."""
        return {
            'name': 'PaddleOCR',
            'version': 'PP-OCRv4',
            'language': self.lang,
            'gpu_enabled': self.use_gpu,
            'detection_threshold': self.det_db_thresh,
            'recognition_threshold': self.rec_score_thresh
        } 