"""
Stage 3: Text Cleaning & Structuring
Normalizes OCR artifacts, whitespace, punctuation.
"""

import re
import logging

try:
    import mlflow
    _trace = mlflow.trace
except ImportError:
    _trace = lambda name=None, **kw: (lambda f: f)

logger = logging.getLogger(__name__)


class TextCleaner:

    OCR_FIXES = {
        r"(?<=[a-z])(?=[A-Z])": " ",
    }

    @_trace(name="clean_text")
    def clean(self, raw_text: str) -> str:
        text = self._fix_encoding(raw_text)
        text = self._fix_ocr_artifacts(text)
        text = self._normalize_whitespace(text)
        text = self._fix_punctuation(text)
        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        replacements = {
            "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u00a0": " ",
            "\ufb01": "fi", "\ufb02": "fl",
        }
        for char, rep in replacements.items():
            text = text.replace(char, rep)
        return text

    def _fix_ocr_artifacts(self, text: str) -> str:
        for pattern, rep in self.OCR_FIXES.items():
            text = re.sub(pattern, rep, text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        text = re.sub(r" +([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?]){2,}", r"\1", text)
        return text

    def segment(self, text: str) -> list:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def clean_and_segment(self, raw_text: str) -> dict:
        cleaned = self.clean(raw_text)
        return {
            "cleaned_text": cleaned,
            "sentences":    self.segment(cleaned),
            "word_count":   len(cleaned.split()),
            "char_count":   len(cleaned),
        }
