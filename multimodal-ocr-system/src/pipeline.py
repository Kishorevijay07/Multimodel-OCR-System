"""
Phase 2 — Updated MLflow PyFunc Pipeline
Integrates fine-tuned BERT classifier + enhanced NER into the
unified deployable pipeline unit.

All 5 stages now support transformer inference.
MLflow Tracing covers every sub-step.
"""

from __future__ import annotations
import time
import logging
import os

import mlflow
import mlflow.pyfunc

logger = logging.getLogger(__name__)


class MultiModalOCRPipeline(mlflow.pyfunc.PythonModel):
    """
    Unified MLflow PyFunc pipeline — Phase 2 edition.

    Config keys (pass via pipeline_config dict):
      preprocessor        : dict   — ImagePreprocessor config
      ocr_languages       : list   — e.g. ["en"]
      ocr_use_gpu         : bool
      use_finetuned_bert  : bool   — use fine-tuned BERT classifier
      finetuned_model_dir : str    — path to saved BERT checkpoint
      finetuned_mlflow_uri: str    — MLflow model URI e.g. models:/Name/1
      use_zero_shot       : bool   — fall back to zero-shot BART
      use_transformer_ner : bool   — use transformer NER
      transformer_ner_model: str
      use_spacy_ner       : bool
      confidence_threshold: float  — below this → flag for human review
    """

    def __init__(self, pipeline_config: dict = None):
        self.pipeline_config = pipeline_config or {}

    # ── Component construction ───────────────────────────────────────────────

    def _build(self):
        cfg = self.pipeline_config

        # Stage 1: Preprocessing
        from src.preprocessing import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(
            config=cfg.get("preprocessor", {})
        )

        # Stage 2: OCR
        from src.ocr_engine import OCREngine
        self.ocr = OCREngine(
            languages=cfg.get("ocr_languages", ["en"]),
            use_gpu=cfg.get("ocr_use_gpu", False),
        )

        # Stage 3: Cleaner
        from src.text_cleaner import TextCleaner
        self.cleaner = TextCleaner()

        # Stage 4a: Classifier (Phase 2 — multi-tier)
        from src.classifier import DocumentClassifier
        self.classifier = DocumentClassifier(
            use_finetuned=cfg.get("use_finetuned_bert", False),
            finetuned_model_dir=cfg.get("finetuned_model_dir"),
            finetuned_mlflow_uri=cfg.get("finetuned_mlflow_uri"),
            use_zero_shot=cfg.get("use_zero_shot", False),
            max_length=cfg.get("max_length", 256),
        )

        # Stage 4b: NER (Phase 2 — multi-tier)
        from src.ner_extractor import NERExtractor
        self.ner = NERExtractor(
            use_transformer_ner=cfg.get("use_transformer_ner", False),
            transformer_ner_model=cfg.get("transformer_ner_model", "dslim/bert-base-NER"),
            use_spacy=cfg.get("use_spacy_ner", False),
            spacy_model=cfg.get("spacy_model", "en_core_web_sm"),
        )

        self._built = True
        logger.info("[Pipeline] All components initialized (Phase 2)")

    def load_context(self, context):
        """Called once when loaded for serving."""
        self._build()

    # ── Inference ────────────────────────────────────────────────────────────

    @mlflow.trace(name="full_pipeline_v2")
    def predict(self, context, model_input) -> dict:
        """
        Accepts:
          {"source": "/path/to/doc.pdf"}         ← file path
          {"source": <bytes>}                    ← raw bytes
          {"text": "already extracted text"}     ← skip OCR
        """
        if not getattr(self, "_built", False):
            self._build()

        t_total = time.time()
        cfg     = self.pipeline_config
        source  = model_input if not isinstance(model_input, dict) else None
        text_only = False

        if isinstance(model_input, dict):
            source    = model_input.get("source")
            raw_text  = model_input.get("text")
            text_only = raw_text is not None and source is None

        # ── Stage 1 + 2: Preprocessing & OCR ─────────────────────────────────
        if text_only:
            full_text = raw_text
            ocr_meta  = {
                "engine_used": "none",
                "avg_confidence": 1.0,
                "pages_processed": 0,
            }
        else:
            with mlflow.start_span(name="stage1_preprocessing") as span:
                t0 = time.time()
                pages = self.preprocessor.process_document(source)
                span.set_attribute("page_count", len(pages))
                span.set_attribute("latency_ms", round((time.time()-t0)*1000, 1))

            with mlflow.start_span(name="stage2_ocr") as span:
                t0 = time.time()
                ocr_results  = self.ocr.extract_document(pages)
                full_text    = "\n\n".join(r.text for r in ocr_results)
                avg_conf     = (
                    sum(r.confidence for r in ocr_results) / len(ocr_results)
                    if ocr_results else 0.0
                )
                ocr_meta = {
                    "engine_used":     ocr_results[0].engine_used if ocr_results else "none",
                    "avg_confidence":  round(avg_conf, 4),
                    "pages_processed": len(pages),
                }
                span.set_attribute("avg_ocr_confidence", avg_conf)
                span.set_attribute("total_chars", len(full_text))
                span.set_attribute("latency_ms", round((time.time()-t0)*1000, 1))

        # ── Stage 3: Text Cleaning ────────────────────────────────────────────
        with mlflow.start_span(name="stage3_cleaning") as span:
            t0 = time.time()
            clean = self.cleaner.clean_and_segment(full_text)
            span.set_attribute("word_count", clean["word_count"])
            span.set_attribute("latency_ms", round((time.time()-t0)*1000, 1))

        # ── Stage 4a: Classification ──────────────────────────────────────────
        with mlflow.start_span(name="stage4a_classification") as span:
            t0 = time.time()
            cls = self.classifier.classify(clean["cleaned_text"])
            span.set_attribute("doc_type",   cls.label)
            span.set_attribute("confidence", cls.confidence)
            span.set_attribute("method",     cls.method)
            span.set_attribute("latency_ms", round((time.time()-t0)*1000, 1))

        # ── Stage 4b: NER ─────────────────────────────────────────────────────
        with mlflow.start_span(name="stage4b_ner") as span:
            t0 = time.time()
            ner_result  = self.ner.extract(clean["cleaned_text"], cls.label)
            structured  = self.ner.to_structured_dict(ner_result)
            span.set_attribute("entity_count", structured["entity_count"])
            span.set_attribute("latency_ms",  round((time.time()-t0)*1000, 1))

        # ── Final output ──────────────────────────────────────────────────────
        total_ms = round((time.time() - t_total) * 1000, 2)
        mlflow.log_metric("total_pipeline_latency_ms", total_ms)

        threshold  = cfg.get("confidence_threshold", 0.3)
        low_ocr    = ocr_meta["avg_confidence"] < threshold and not text_only
        low_cls    = cls.confidence < threshold
        is_unknown = cls.label == "unknown"
        needs_review = low_ocr or low_cls or is_unknown

        return {
            "status":                "success",
            "requires_human_review": needs_review,
            "review_reasons": _review_reasons(low_ocr, low_cls, is_unknown),
            "document": {
                "type":                       cls.label,
                "classification_confidence":  cls.confidence,
                "classification_method":      cls.method,
                "classifier_model":           cls.model_name,
                "all_type_scores": {
                    k: round(v, 4)
                    for k, v in cls.all_scores.items()
                },
            },
            "ocr": {
                **ocr_meta,
                "extracted_text": clean["cleaned_text"],
                "word_count":     clean["word_count"],
                "char_count":     clean["char_count"],
            },
            "entities": structured,
            "performance": {
                "total_latency_ms": total_ms,
            },
        }


def _review_reasons(low_ocr, low_cls, is_unknown) -> list:
    reasons = []
    if low_ocr:     reasons.append("low_ocr_confidence")
    if low_cls:     reasons.append("low_classification_confidence")
    if is_unknown:  reasons.append("unknown_document_type")
    return reasons


# ─── Registration helper ──────────────────────────────────────────────────────

def build_and_register(
    config: dict = None,
    experiment_name: str = "multimodal-ocr-system",
    run_name: str        = "pipeline-v2-transformer",
    register_as: str     = "MultiModalOCRPipeline",
) -> str:
    config = config or {}
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "pipeline_version":      "2.0",
            "use_finetuned_bert":    config.get("use_finetuned_bert", False),
            "use_zero_shot":         config.get("use_zero_shot", False),
            "use_transformer_ner":   config.get("use_transformer_ner", False),
            "use_spacy_ner":         config.get("use_spacy_ner", False),
            "confidence_threshold":  config.get("confidence_threshold", 0.3),
            "ocr_languages":         str(config.get("ocr_languages", ["en"])),
        })

        pipeline = MultiModalOCRPipeline(pipeline_config=config)

        model_info = mlflow.pyfunc.log_model(
            artifact_path="ocr_pipeline_v2",
            python_model=pipeline,
            registered_model_name=register_as,
        )

        logger.info(f"Pipeline v2 registered: {model_info.model_uri}")
        return model_info.model_uri
