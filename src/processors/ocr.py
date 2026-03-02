"""
OCR processor for extracting text from images.
Uses EasyOCR for Hindi/English support.
"""

import asyncio
from pathlib import Path
from typing import Optional
import logging

from PIL import Image
import easyocr

from ..config import settings

logger = logging.getLogger(__name__)

# Global reader instance - loaded once at startup (mirrors whisper model pattern)
_ocr_reader: Optional[easyocr.Reader] = None


def load_ocr_reader(languages: Optional[list[str]] = None) -> easyocr.Reader:
    """Load the EasyOCR reader (called at startup for warm first-request)."""
    global _ocr_reader
    if _ocr_reader is None:
        langs = languages or settings.ocr_language_list
        logger.info(f"Loading EasyOCR reader with languages: {langs}")
        _ocr_reader = easyocr.Reader(
            langs,
            gpu=True,  # Falls back to CPU automatically if GPU unavailable
            verbose=False,
        )
        logger.info("EasyOCR reader loaded successfully")
    return _ocr_reader


class OCRProcessor:
    """
    OCR processor using EasyOCR.
    Supports Hindi, English, and mixed text (Hinglish).
    """
    
    def __init__(self, languages: Optional[list[str]] = None):
        """
        Initialize the OCR processor.
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
        """
        self.languages = languages or settings.ocr_language_list
    
    @property
    def reader(self) -> easyocr.Reader:
        """Get the shared EasyOCR reader (loaded at startup)."""
        global _ocr_reader
        if _ocr_reader is None:
            load_ocr_reader(self.languages)
        return _ocr_reader
    
    async def extract_text(self, image_path: Path) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
        """
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return ""
        
        try:
            # Run OCR in thread pool (EasyOCR is synchronous)
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._extract_text_sync, str(image_path)
            )
            return result
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""
    
    def _extract_text_sync(self, image_path: str) -> str:
        """Synchronous text extraction."""
        # Preprocess image if needed
        processed_path = self._preprocess_image(image_path)
        
        # Run OCR
        results = self.reader.readtext(
            processed_path,
            detail=0,  # Return only text, not bounding boxes
            paragraph=True  # Merge nearby text into paragraphs
        )
        
        # Clean up temp file if preprocessing created one
        if processed_path != image_path:
            try:
                Path(processed_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        # Join all detected text
        text = "\n".join(results)
        
        logger.info(f"Extracted {len(text)} characters from {image_path}")
        return text.strip()
    
    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR accuracy.
        Returns path to processed image (or original if no processing needed).
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if too large (prevents memory issues)
                max_dimension = 4000
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save processed image
                    processed_path = image_path.replace('.', '_processed.')
                    img.save(processed_path)
                    return processed_path
                
                return image_path
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path
    
    async def extract_text_from_multiple(
        self, 
        image_paths: list[Path]
    ) -> dict[Path, str]:
        """
        Extract text from multiple images concurrently.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping paths to extracted text
        """
        tasks = [self.extract_text(path) for path in image_paths]
        results = await asyncio.gather(*tasks)
        
        return dict(zip(image_paths, results))
