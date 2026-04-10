"""
Stage 1: Image Preprocessing
Full implementation when opencv-python is installed.
Graceful stub when running in text-only mode.
"""

import logging
import numpy as np

try:
    import mlflow
    _trace = mlflow.trace
except ImportError:
    _trace = lambda name=None, **kw: (lambda f: f)

logger = logging.getLogger(__name__)

_CV2_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python not installed. Preprocessing will be skipped.")


class ImagePreprocessor:

    def __init__(self, config: dict = None):
        self.config = config or {
            "target_dpi": 300,
            "denoise_strength": 10,
            "binarize_block_size": 11,
            "binarize_c": 2,
            "deskew": True,
            "remove_borders": True,
        }

    @_trace(name="load_document")
    def load(self, source) -> list:
        from pathlib import Path
        if isinstance(source, bytes):
            if _CV2_AVAILABLE:
                arr = np.frombuffer(source, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return [img]
            return [np.zeros((100, 100, 3), dtype=np.uint8)]

        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.suffix.lower() == ".pdf":
                return self._load_pdf(p)
            if _CV2_AVAILABLE:
                img = cv2.imread(str(p))
                return [img] if img is not None else [np.zeros((100,100,3),dtype=np.uint8)]
            return [np.zeros((100, 100, 3), dtype=np.uint8)]

        return [np.zeros((100, 100, 3), dtype=np.uint8)]

    def _load_pdf(self, pdf_path) -> list:
        try:
            from pdf2image import convert_from_path
            pil_images = convert_from_path(str(pdf_path), dpi=self.config["target_dpi"])
            if _CV2_AVAILABLE:
                return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
            return [np.array(img) for img in pil_images]
        except Exception as e:
            logger.warning(f"PDF load failed: {e}")
            return [np.zeros((100, 100, 3), dtype=np.uint8)]

    @_trace(name="preprocess_image")
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if not _CV2_AVAILABLE:
            return image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.fastNlMeansDenoising(image, h=self.config["denoise_strength"])
        image = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config["binarize_block_size"],
            self.config["binarize_c"],
        )
        return image

    def process_document(self, source) -> list:
        pages = self.load(source)
        return [self.preprocess(p) for p in pages]
