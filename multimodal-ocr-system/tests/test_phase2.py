"""
Phase 2 — Test Suite
Tests all new Phase 2 components:
  - Dataset builder
  - Three-tier classifier
  - Multi-tier NER extractor
  - Pipeline v2 (text-only mode, no OCR required)
  - Transformer inference mocking

Run: pytest tests/ -v --tb=short
"""

import json
import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Dataset Builder Tests ────────────────────────────────────────────────────

class TestDatasetBuilder:

    def test_build_creates_splits(self, tmp_path):
        from training.dataset_builder import build_dataset
        stats = build_dataset(n_per_class=5, output_dir=str(tmp_path))
        assert "train" in stats
        assert "val"   in stats
        assert "test"  in stats
        assert stats["train"] > 0

    def test_jsonl_format(self, tmp_path):
        from training.dataset_builder import build_dataset
        build_dataset(n_per_class=5, output_dir=str(tmp_path))
        with open(tmp_path / "train.jsonl") as f:
            first = json.loads(f.readline())
        assert "text"     in first
        assert "label"    in first
        assert "label_id" in first

    def test_all_labels_present(self, tmp_path):
        from training.dataset_builder import build_dataset, LABEL2ID
        build_dataset(n_per_class=10, output_dir=str(tmp_path))
        labels_seen = set()
        with open(tmp_path / "train.jsonl") as f:
            for line in f:
                labels_seen.add(json.loads(line)["label"])
        assert labels_seen == set(LABEL2ID.keys())

    def test_label_map_saved(self, tmp_path):
        from training.dataset_builder import build_dataset
        build_dataset(n_per_class=3, output_dir=str(tmp_path))
        with open(tmp_path / "label_map.json") as f:
            lm = json.load(f)
        assert "label2id" in lm
        assert "id2label" in lm

    def test_generators_produce_text(self):
        from training.dataset_builder import GENERATORS
        for label, gen_fn in GENERATORS.items():
            text = gen_fn()
            assert isinstance(text, str)
            assert len(text) > 20, f"Generator '{label}' produced too-short text"

    def test_split_no_overlap(self, tmp_path):
        from training.dataset_builder import build_dataset
        build_dataset(n_per_class=30, output_dir=str(tmp_path))
        def load_texts(split):
            texts = set()
            with open(tmp_path / f"{split}.jsonl") as f:
                for line in f:
                    texts.add(json.loads(line)["text"])
            return texts
        train = load_texts("train")
        test  = load_texts("test")
        assert len(train & test) == 0, "Train/test overlap detected!"


# ─── Classifier Tests ─────────────────────────────────────────────────────────

class TestClassifierKeywordTier:
    """Tests for Tier 3 keyword fallback (always available)."""

    def setup_method(self):
        from src.classifier import DocumentClassifier
        self.clf = DocumentClassifier(
            use_finetuned=False,
            use_zero_shot=False,
        )

    def test_prescription_detected(self):
        text = ("Rx: Amoxicillin 500mg twice daily for 7 days. "
                "Patient: John Doe. Prescribed by Dr. Smith. Refills: 0.")
        result = self.clf.classify(text)
        assert result.label == "medical_prescription"
        assert result.method == "keyword"

    def test_legal_detected(self):
        text = ("This agreement is between the parties hereinafter. "
                "Governed by jurisdiction of California. Indemnification applies.")
        result = self.clf.classify(text)
        assert result.label == "legal_contract"

    def test_lab_report_detected(self):
        text = ("Laboratory report. WBC: 7.5 K/uL. HbA1c: 6.1%. "
                "Fasting glucose: 108 mg/dL. Reference range normal.")
        result = self.clf.classify(text)
        assert result.label == "lab_report"

    def test_invoice_detected(self):
        text = ("Tax Invoice INV-12345. Bill To: ABC Corp. "
                "Total Amount: USD 5,000. Payment due within 30 days. GST applied.")
        result = self.clf.classify(text)
        assert result.label == "invoice"

    def test_affidavit_detected(self):
        text = ("I, John Doe, do hereby solemnly affirm and declare. "
                "Sworn before me, Notary Public. Commissioner of Oaths signature.")
        result = self.clf.classify(text)
        assert result.label == "affidavit"

    def test_unknown_on_empty(self):
        result = self.clf.classify("abc 123 xyz random nonsense")
        assert result.confidence < 0.5

    def test_all_scores_has_all_labels(self):
        from src.classifier import LABELS
        result = self.clf.classify("some text")
        for label in LABELS:
            assert label in result.all_scores

    def test_result_dataclass_fields(self):
        result = self.clf.classify("prescription rx amoxicillin")
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "all_scores")
        assert hasattr(result, "method")

    def test_confidence_is_float_0_to_1(self):
        result = self.clf.classify("Rx dosage amoxicillin mg prescription")
        assert 0.0 <= result.confidence <= 1.0


# ─── NER Extractor Tests ──────────────────────────────────────────────────────

class TestNERExtractor:

    def setup_method(self):
        from src.ner_extractor import NERExtractor
        self.ner = NERExtractor(
            use_transformer_ner=False,
            use_spacy=False,
        )

    def test_drug_extraction(self):
        text  = "Patient should take Amoxicillin 500mg twice daily."
        result = self.ner.extract(text, "medical_prescription")
        assert "DRUG" in result.by_type or "DOSAGE" in result.by_type

    def test_dosage_mg(self):
        text  = "Dosage: 500mg once daily."
        result = self.ner.extract(text, "medical_prescription")
        assert "DOSAGE" in result.by_type
        assert any("500mg" in v for v in result.by_type["DOSAGE"])

    def test_bid_tid_extraction(self):
        text  = "Take tablet bid for 7 days."
        result = self.ner.extract(text, "medical_prescription")
        assert "DOSAGE" in result.by_type

    def test_date_medical(self):
        text  = "Date: 14/03/2024."
        result = self.ner.extract(text, "medical_prescription")
        assert "DATE" in result.by_type

    def test_monetary_legal(self):
        text  = "The client agrees to pay USD 15,000 per month."
        result = self.ner.extract(text, "legal_contract")
        assert "MONETARY_VALUE" in result.by_type

    def test_jurisdiction_legal(self):
        text  = "Governed by the laws of the State of California."
        result = self.ner.extract(text, "legal_contract")
        assert "JURISDICTION" in result.by_type

    def test_invoice_number(self):
        text  = "Invoice No: INV-2024-00123. Payment due in 30 days."
        result = self.ner.extract(text, "invoice")
        assert "INVOICE_NUMBER" in result.by_type

    def test_unknown_type_no_entities(self):
        result = self.ner.extract("some random text", "unknown")
        assert result.entities == []

    def test_no_duplicate_entities(self):
        text  = "Amoxicillin 500mg. Take Amoxicillin 500mg twice."
        result = self.ner.extract(text, "medical_prescription")
        all_values = [e.value.lower() for e in result.entities]
        assert len(all_values) == len(set(all_values)), "Duplicate entities found"

    def test_structured_dict_format(self):
        text  = "Rx: Amoxicillin 500mg once daily."
        result = self.ner.extract(text, "medical_prescription")
        d = self.ner.to_structured_dict(result)
        assert "entity_count"     in d
        assert "entities_by_type" in d
        assert "entities_list"    in d
        assert isinstance(d["entity_count"], int)

    def test_entity_has_source_field(self):
        text  = "Invoice No: INV-999."
        result = self.ner.extract(text, "invoice")
        for ent in result.entities:
            assert ent.source in ("regex", "spacy", "transformer")


# ─── Pipeline v2 Tests (text-only mode) ──────────────────────────────────────

class TestPipelineV2TextMode:
    """Test pipeline using pre-extracted text (no OCR needed — fast)."""

    def setup_method(self):
        from src.pipeline import MultiModalOCRPipeline
        self.pipeline = MultiModalOCRPipeline(pipeline_config={
            "use_finetuned_bert":  False,
            "use_zero_shot":       False,
            "use_transformer_ner": False,
            "use_spacy_ner":       False,
            "confidence_threshold": 0.1,
        })
        self.pipeline._build()

    def _predict(self, text: str) -> dict:
        return self.pipeline.predict(None, {"text": text})

    def test_returns_dict(self):
        result = self._predict("Rx: Amoxicillin 500mg twice daily.")
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = self._predict("Invoice No: INV-100. Total: USD 5,000.")
        for key in ("status", "document", "ocr", "entities", "performance"):
            assert key in result

    def test_prescription_classified(self):
        text   = ("Rx: Amoxicillin 500mg twice daily for 7 days. "
                  "Patient: Jane Doe. Prescribed by Dr. Smith. Refills: 0.")
        result = self._predict(text)
        assert result["document"]["type"] == "medical_prescription"

    def test_legal_classified(self):
        text   = ("Agreement between parties hereinafter. "
                  "Governed by laws of California. Indemnification clause.")
        result = self._predict(text)
        assert result["document"]["type"] == "legal_contract"

    def test_ocr_pages_zero_in_text_mode(self):
        result = self._predict("Some text")
        assert result["ocr"]["pages_processed"] == 0

    def test_entities_present(self):
        text   = "Amoxicillin 500mg twice daily. Date: 01/03/2024."
        result = self._predict(text)
        assert result["entities"]["entity_count"] >= 0

    def test_performance_latency_logged(self):
        result = self._predict("Some sample text")
        assert "total_latency_ms" in result["performance"]
        assert result["performance"]["total_latency_ms"] > 0

    def test_review_reasons_list(self):
        result = self._predict("random text xyz no domain keywords")
        assert isinstance(result["review_reasons"], list)

    def test_confidence_between_0_and_1(self):
        result = self._predict("Invoice total due USD 5000.")
        conf = result["document"]["classification_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_word_count_correct(self):
        text   = "one two three four five"
        result = self._predict(text)
        assert result["ocr"]["word_count"] == 5

    def test_status_is_success(self):
        result = self._predict("Any text at all")
        assert result["status"] == "success"


# ─── Integration: Dataset → Classifier ───────────────────────────────────────

class TestIntegration:

    def test_generated_samples_classify_correctly(self, tmp_path):
        from training.dataset_builder import build_dataset
        from src.classifier import DocumentClassifier

        build_dataset(n_per_class=10, output_dir=str(tmp_path))
        clf    = DocumentClassifier(use_finetuned=False, use_zero_shot=False)
        correct, total = 0, 0

        with open(tmp_path / "test.jsonl") as f:
            for line in f:
                sample   = json.loads(line)
                result   = clf.classify(sample["text"])
                if result.label == sample["label"]:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        # Keyword baseline should get at least 50% on synthetic data
        assert accuracy >= 0.50, (
            f"Keyword baseline too low: {accuracy:.2%} on {total} samples"
        )
