"""
Stage 2: OCR Engine
Full implementation when easyocr / pytesseract are installed.
Returns stub results in text-only mode.
"""

import numpy as np
import logging
from dataclasses import dataclass, field

try:
    import mlflow
    _trace = mlflow.trace
except ImportError:
    _trace = lambda name=None, **kw: (lambda f: f)

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    words: list = field(default_factory=list)
    engine_used: str = "stub"
    page_number: int = 0


class OCREngine:

    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, languages: list = None, use_gpu: bool = False):
        self.languages = languages or ["en"]
        self.use_gpu   = use_gpu
        self._easyocr  = None
        self._tesseract = False
        self._init()

    def _init(self):
        try:
            import easyocr
            self._easyocr = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR unavailable: {e}")

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract = True
            logger.info("Tesseract initialized")
        except Exception:
            pass

    @_trace(name="run_ocr")
    def extract(self, image: np.ndarray, page_number: int = 0) -> OCRResult:
        if self._easyocr:
            result = self._easyocr_extract(image, page_number)
            if result.confidence >= self.CONFIDENCE_THRESHOLD:
                return result

        if self._tesseract:
            return self._tesseract_extract(image, page_number)

        # No OCR available — return empty (text-only mode handles this)
        return OCRResult(text="", confidence=0.0, engine_used="unavailable")

    def _easyocr_extract(self, image, page_number):
        raw = self._easyocr.readtext(image)
        words, parts = [], []
        total = 0.0
        for bbox, text, conf in raw:
            words.append({"word": text, "confidence": conf, "bbox": bbox})
            parts.append(text)
            total += conf
        avg = total / len(raw) if raw else 0.0
        return OCRResult(
            text=" ".join(parts), confidence=avg,
            words=words, engine_used="easyocr", page_number=page_number
        )

    def _tesseract_extract(self, image, page_number):
        import pytesseract
        from PIL import Image as PILImage
        data = pytesseract.image_to_data(
            PILImage.fromarray(image),
            output_type=pytesseract.Output.DICT
        )
        words, parts, total, count = [], [], 0.0, 0
        for i, word in enumerate(data["text"]):
            conf = data["conf"][i]
            if conf > 0 and word.strip():
                words.append({"word": word, "confidence": conf / 100.0})
                parts.append(word)
                total += conf / 100.0
                count += 1
        avg = total / count if count > 0 else 0.0
        return OCRResult(
            text=" ".join(parts), confidence=avg,
            words=words, engine_used="tesseract", page_number=page_number
        )

    def extract_document(self, pages: list) -> list:
        return [self.extract(p, i) for i, p in enumerate(pages)]
