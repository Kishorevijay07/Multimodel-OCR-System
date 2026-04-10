"""
Unit tests for all pipeline stages.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ─── Preprocessing Tests ───────────────────────────────────────────────────

class TestPreprocessing:
    def setup_method(self):
        from src.preprocessing import ImagePreprocessor
        self.pp = ImagePreprocessor()

    def test_grayscale_conversion(self):
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.pp.preprocess(color_img)
        assert len(result.shape) == 2

    def test_already_grayscale(self):
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = self.pp.preprocess(gray_img)
        assert len(result.shape) == 2

    def test_binarize_output(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        binary = self.pp.preprocess(gray)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})


# ─── Text Cleaner Tests ────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from src.text_cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_encoding_fix(self):
        text = "It\u2019s a \u201chello\u201d world"
        result = self.cleaner._fix_encoding(text)
        assert "'" in result
        assert '"' in result

    def test_whitespace_normalization(self):
        text = "hello    world\n\n\n\nbye"
        result = self.cleaner._normalize_whitespace(text)
        assert "    " not in result
        assert "\n\n\n" not in result

    def test_clean_returns_string(self):
        result = self.cleaner.clean("Some  OCR   text with artifacts")
        assert isinstance(result, str)

    def test_segment_splits_sentences(self):
        text = "First sentence. Second sentence. Third one."
        segments = self.cleaner.segment(text)
        assert len(segments) >= 1


# ─── Classifier Tests ──────────────────────────────────────────────────────

class TestClassifier:
    def setup_method(self):
        from src.classifier import DocumentClassifier
        self.clf = DocumentClassifier(use_finetuned=False, use_zero_shot=False)

    def test_prescription_classification(self):
        text = (
            "Patient: John Doe. Rx: Amoxicillin 500mg twice daily. "
            "Dispense 14 capsules. Prescribed by Dr. Smith. Refills: 0."
        )
        result = self.clf.classify(text)
        assert result.label == "medical_prescription"
        assert result.confidence > 0

    def test_legal_classification(self):
        text = (
            "This agreement is entered between the parties hereinafter. "
            "Governed by the laws of California. Indemnification clause applies."
        )
        result = self.clf.classify(text)
        assert result.label == "legal_contract"

    def test_unknown_returns_low_confidence(self):
        result = self.clf.classify("random gibberish text abc xyz 123")
        assert result.confidence < 0.5

    def test_all_scores_present(self):
        result = self.clf.classify("some text")
        assert isinstance(result.all_scores, dict)


# ─── NER Tests ────────────────────────────────────────────────────────────

class TestNERExtractor:
    def setup_method(self):
        from src.ner_extractor import NERExtractor
        self.ner = NERExtractor()

    def test_drug_extraction(self):
        text = "Patient should take Amoxicillin 500mg twice daily."
        result = self.ner.extract(text, "medical_prescription")
        assert "DRUG" in result.by_type or "DOSAGE" in result.by_type

    def test_dosage_extraction(self):
        text = "Take 500mg of the medicine once daily for 7 days."
        result = self.ner.extract(text, "medical_prescription")
        assert "DOSAGE" in result.by_type

    def test_monetary_value_legal(self):
        text = "The client agrees to pay USD 15,000 per month."
        result = self.ner.extract(text, "legal_contract")
        assert "MONETARY_VALUE" in result.by_type

    def test_date_extraction(self):
        text = "Signed on 14/03/2024 in California."
        result = self.ner.extract(text, "legal_contract")
        assert "DATE" in result.by_type

    def test_unknown_doc_type(self):
        result = self.ner.extract("some text", "unknown")
        assert result.entities == []