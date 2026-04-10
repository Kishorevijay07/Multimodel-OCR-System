"""
Phase 2 — Upgraded Document Classifier
Replaces the Phase 1 keyword classifier with a three-tier inference stack:

  Tier 1: Fine-tuned BERT         (best accuracy, needs GPU or decent CPU)
  Tier 2: Zero-shot BART          (no fine-tuning, good accuracy, slower)
  Tier 3: Keyword fallback        (instant, always available — Phase 1)

The classifier auto-selects the best available tier at startup.
All tiers return the same ClassificationResult dataclass.
"""

from __future__ import annotations
import logging
import os
import json
from dataclasses import dataclass
from typing import Optional

import mlflow
import numpy as np

logger = logging.getLogger(__name__)

# ─── Label registry ──────────────────────────────────────────────────────────
LABELS = [
    "medical_prescription",
    "lab_report",
    "legal_contract",
    "affidavit",
    "invoice",
    "unknown",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# ─── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    label: str
    confidence: float
    all_scores: dict
    method: str           # "finetuned_bert" | "zero_shot" | "keyword"
    model_name: str = ""


# ─── Tier 3: Keyword baseline (identical to Phase 1) ─────────────────────────
DOMAIN_KEYWORDS = {
    "medical_prescription": [
        "rx", "prescription", "dosage", "mg", "tablet", "capsule",
        "twice daily", "once daily", "patient name", "refill", "pharmacy",
        "prescribed by", "drug", "medicine", "sig:", "dispense",
    ],
    "lab_report": [
        "laboratory", "lab report", "specimen", "result", "reference range",
        "hba1c", "glucose", "hemoglobin", "platelet", "wbc", "rbc",
        "urine", "blood test", "serum", "normal range", "abnormal",
    ],
    "legal_contract": [
        "agreement", "contract", "parties", "whereas", "hereinafter",
        "indemnification", "liability", "jurisdiction", "governed by",
        "in witness whereof", "terms and conditions", "obligations",
        "breach", "termination", "arbitration",
    ],
    "affidavit": [
        "affidavit", "sworn", "deponent", "notary", "before me",
        "solemnly affirm", "subscribed and sworn", "commissioner of oaths",
        "do hereby declare", "witness my hand",
    ],
    "invoice": [
        "invoice", "bill to", "payment due", "total amount", "gst",
        "tax invoice", "invoice number", "due date", "payable",
        "quantity", "unit price", "subtotal",
    ],
    "unknown": [],
}


def _keyword_classify(text: str) -> ClassificationResult:
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in DOMAIN_KEYWORDS.items():
        if not keywords:
            scores[doc_type] = 0.0
            continue
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = hits / len(keywords)

    if max(scores.values()) == 0:
        return ClassificationResult(
            label="unknown", confidence=0.0,
            all_scores=scores, method="keyword",
        )

    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    normalized = {k: round(v / total, 4) for k, v in scores.items()}
    return ClassificationResult(
        label=best,
        confidence=round(scores[best], 4),
        all_scores=normalized,
        method="keyword",
    )


# ─── Tier 2: Zero-shot BART ───────────────────────────────────────────────────

class ZeroShotClassifier:
    """
    Uses facebook/bart-large-mnli for zero-shot classification.
    No fine-tuning required. Downloads ~1.6GB on first use.
    """
    MODEL_NAME = "facebook/bart-large-mnli"

    def __init__(self):
        self._pipe = None

    def load(self):
        from transformers import pipeline
        logger.info(f"Loading zero-shot model: {self.MODEL_NAME}")
        self._pipe = pipeline(
            "zero-shot-classification",
            model=self.MODEL_NAME,
            device=-1,   # CPU; set to 0 for GPU
        )
        logger.info("Zero-shot classifier ready")

    @mlflow.trace(name="zero_shot_classify")
    def classify(self, text: str) -> ClassificationResult:
        if self._pipe is None:
            self.load()
        # Truncate to ~512 tokens (approx 1500 chars)
        result = self._pipe(text[:1500], LABELS, multi_label=False)
        scores = dict(zip(result["labels"], result["scores"]))
        best   = result["labels"][0]
        return ClassificationResult(
            label=best,
            confidence=round(result["scores"][0], 4),
            all_scores={k: round(v, 4) for k, v in scores.items()},
            method="zero_shot",
            model_name=self.MODEL_NAME,
        )


# ─── Tier 1: Fine-tuned BERT ──────────────────────────────────────────────────

class FineTunedBERTClassifier:
    """
    Loads a fine-tuned BERT model from:
      a) MLflow Model Registry  (if mlflow_model_uri provided)
      b) Local checkpoint dir   (if local_model_dir provided)
    """

    def __init__(
        self,
        mlflow_model_uri: Optional[str]  = None,
        local_model_dir:  Optional[str]  = None,
        max_length: int                  = 256,
    ):
        self.mlflow_model_uri = mlflow_model_uri
        self.local_model_dir  = local_model_dir
        self.max_length       = max_length
        self._model           = None
        self._tokenizer       = None
        self._device          = None

    def load(self):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")

        if self.mlflow_model_uri:
            logger.info(f"Loading BERT from MLflow: {self.mlflow_model_uri}")
            import mlflow.pytorch
            self._model = mlflow.pytorch.load_model(self.mlflow_model_uri)
            # Tokenizer must be in a sibling artifact dir
            tokenizer_path = self._resolve_mlflow_tokenizer()
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        elif self.local_model_dir:
            # Resolve to latest checkpoint subdir if base dir has no config.json
            model_path = self.local_model_dir
            if not os.path.isfile(os.path.join(model_path, "config.json")):
                checkpoints = sorted(
                    [d for d in os.listdir(model_path)
                     if d.startswith("checkpoint-")
                     and os.path.isdir(os.path.join(model_path, d))],
                    key=lambda x: int(x.split("-")[-1]),
                )
                if checkpoints:
                    model_path = os.path.join(model_path, checkpoints[-1])
                    logger.info(f"Resolved to latest checkpoint: {model_path}")

            logger.info(f"Loading BERT from local dir: {model_path}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            tok_path = os.path.join(self.local_model_dir, "tokenizer")
            if not os.path.isdir(tok_path):
                tok_path = model_path   # tokenizer saved alongside checkpoint
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            raise ValueError("Must provide mlflow_model_uri or local_model_dir")

        self._model.to(self._device)
        self._model.eval()
        logger.info("Fine-tuned BERT classifier ready")

    def _resolve_mlflow_tokenizer(self) -> str:
        """Download tokenizer artifact from MLflow run."""
        import mlflow
        client = mlflow.tracking.MlflowClient()
        # Parse model URI like models:/DocumentBERTClassifier/1
        parts = self.mlflow_model_uri.replace("models:/", "").split("/")
        model_name, version = parts[0], parts[1] if len(parts) > 1 else "latest"
        mv = client.get_model_version(model_name, version)
        run_id = mv.run_id
        artifact_path = client.download_artifacts(run_id, "tokenizer")
        return artifact_path

    @mlflow.trace(name="finetuned_bert_classify")
    def classify(self, text: str) -> ClassificationResult:
        import torch

        if self._model is None:
            self.load()

        encoding = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)
            probs   = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        all_scores = {
            ID2LABEL[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return ClassificationResult(
            label=ID2LABEL[pred_id],
            confidence=round(float(probs[pred_id]), 4),
            all_scores=all_scores,
            method="finetuned_bert",
            model_name=getattr(self._model.config, "_name_or_path", "bert"),
        )


# ─── Main Classifier — auto-selects best tier ────────────────────────────────

class DocumentClassifier:
    """
    Smart classifier that automatically uses the best available tier:
      Tier 1 → Fine-tuned BERT  (if model path/URI given)
      Tier 2 → Zero-shot BART   (if use_zero_shot=True)
      Tier 3 → Keyword baseline (always available)

    Public API matches Phase 1 classifier exactly for drop-in compatibility.
    """

    def __init__(
        self,
        # Tier 1 options
        use_finetuned: bool             = False,
        finetuned_model_dir: str        = None,
        finetuned_mlflow_uri: str       = None,
        # Tier 2 options
        use_zero_shot: bool             = False,
        # Fallback
        keyword_confidence_threshold: float = 0.15,
        # Misc
        max_length: int = 256,
    ):
        self.use_finetuned = use_finetuned
        self.use_zero_shot = use_zero_shot
        self.keyword_threshold = keyword_confidence_threshold
        self._bert  = None
        self._zs    = None

        if use_finetuned:
            self._bert = FineTunedBERTClassifier(
                mlflow_model_uri=finetuned_mlflow_uri,
                local_model_dir=finetuned_model_dir,
                max_length=max_length,
            )
            try:
                self._bert.load()
                logger.info("[Classifier] Using Tier 1: Fine-tuned BERT")
            except Exception as e:
                logger.warning(f"[Classifier] BERT load failed: {e}. Falling back.")
                self._bert = None

        if use_zero_shot and self._bert is None:
            self._zs = ZeroShotClassifier()
            try:
                self._zs.load()
                logger.info("[Classifier] Using Tier 2: Zero-shot BART")
            except Exception as e:
                logger.warning(f"[Classifier] Zero-shot load failed: {e}. Falling back.")
                self._zs = None

        if self._bert is None and self._zs is None:
            logger.info("[Classifier] Using Tier 3: Keyword baseline")

    @mlflow.trace(name="classify_document")
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify document text. Cascades through tiers.
        """
        # Tier 1: Fine-tuned BERT
        if self._bert is not None:
            try:
                result = self._bert.classify(text)
                mlflow.log_metric("classifier_tier", 1)
                mlflow.log_metric("classifier_confidence", result.confidence)
                return result
            except Exception as e:
                logger.warning(f"BERT classify failed: {e}")

        # Tier 2: Zero-shot
        if self._zs is not None:
            try:
                result = self._zs.classify(text)
                mlflow.log_metric("classifier_tier", 2)
                mlflow.log_metric("classifier_confidence", result.confidence)
                return result
            except Exception as e:
                logger.warning(f"Zero-shot classify failed: {e}")

        # Tier 3: Keyword
        result = _keyword_classify(text)
        mlflow.log_metric("classifier_tier", 3)
        mlflow.log_metric("classifier_confidence", result.confidence)
        return result

    def warmup(self):
        """Pre-load models so first inference is fast."""
        dummy = "patient prescription medicine doctor rx dosage"
        self.classify(dummy)
        logger.info("[Classifier] Warmup complete")
