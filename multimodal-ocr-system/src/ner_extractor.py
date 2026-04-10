"""
Phase 2 — Enhanced NER Extractor
Adds transformer-based NER on top of Phase 1 regex patterns.

Three-tier approach:
  Tier 1: Domain-tuned transformer NER (BioBERT / LegalBERT)
  Tier 2: spaCy en_core_web_sm / en_core_web_trf
  Tier 3: Regex patterns (Phase 1, always available)
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Entity:
    label: str
    value: str
    start: int  = -1
    end:   int  = -1
    confidence: float = 1.0
    source: str = "regex"   # "regex" | "spacy" | "transformer"


@dataclass
class NERResult:
    entities:  list = field(default_factory=list)
    by_type:   dict = field(default_factory=dict)
    doc_type:  str  = "unknown"


# ─── Regex pattern library (Phase 1 + extended) ───────────────────────────────
MEDICAL_PATTERNS = {
    "DATE": [
        r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    ],
    "DOSAGE": [
        r"\b\d+\.?\d*\s*(?:mg|mcg|ml|g|units?|iu|mmol|mEq)\b",
        r"\b(?:once|twice|thrice|\d+\s+times?)\s+(?:a\s+)?(?:day|daily|week|month)\b",
        r"\bq\.?(?:\d+)?h\b",            # q8h, qd, etc.
        r"\b(?:bid|tid|qid|qd|prn)\b",   # Latin abbreviations
    ],
    "DRUG": [
        r"\b(?:amoxicillin|ibuprofen|metformin|lisinopril|atorvastatin|"
        r"paracetamol|acetaminophen|aspirin|omeprazole|amlodipine|metoprolol|"
        r"insulin|warfarin|prednisone|levothyroxine|albuterol|sertraline|"
        r"escitalopram|gabapentin|hydrochlorothiazide|losartan|pantoprazole)\b",
    ],
    "LAB_VALUE": [
        r"\b(?:hba1c|a1c|glucose|hemoglobin|hgb|wbc|rbc|platelet|plt|"
        r"creatinine|bilirubin|cholesterol|triglyceride|sodium|potassium|"
        r"albumin|ast|alt|tsh|psa)[:\s]+[\d\.]+\s*"
        r"(?:mg/dl|g/dl|mmol/l|%|k/ul|iu/l|ng/ml|meq/l)?\b",
    ],
    "PATIENT_AGE": [
        r"\b(?:age[d]?|patient\s+age)[:\s]*(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:yr|yrs|years?)\s*(?:old|\/[mf])?\b",
    ],
    "DOCTOR": [
        r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
        r"\b(?:prescribed|referring|consulting)\s+(?:physician|doctor)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b",
    ],
    "LICENSE_NUMBER": [
        r"\b(?:lic(?:ense)?|reg(?:istration)?|npi)[.#:\s]+([A-Z0-9\-]+)\b",
    ],
}

LEGAL_PATTERNS = {
    "DATE": [
        r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
        r"\bthis\s+\d+(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s+\d{4}\b",
        r"\b(?:effective|execution)\s+date[:\s]+[^\n.]{5,30}\b",
    ],
    "MONETARY_VALUE": [
        r"\b(?:USD|INR|EUR|GBP|\$|₹|€|£)\s*[\d,]+(?:\.\d{1,2})?\b",
        r"\b[\d,]+(?:\.\d{1,2})?\s*(?:dollars?|rupees?|euros?|pounds?)\b",
    ],
    "JURISDICTION": [
        r"\b(?:state|court|district|county|province)\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
        r"\b(?:governed\s+by|laws?\s+of|under\s+the\s+laws?\s+of)\s+"
        r"(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
    ],
    "OBLIGATION": [
        r"\b(?:shall|must|agrees?\s+to|is\s+obligated\s+to|undertakes?\s+to|"
        r"covenants?\s+to)\s+[^.]{5,80}\b",
    ],
    "PARTY": [
        r"\b(?:hereinafter\s+(?:referred\s+to\s+as|called))\s*[\"']?([A-Z][A-Za-z\s]+)[\"']?\b",
        r"\b(?:the\s+)?(?:company|client|vendor|contractor|licensor|licensee|"
        r"lessor|lessee|employer|employee|buyer|seller)\b",
    ],
    "CLAUSE_REFERENCE": [
        r"\b(?:section|clause|article|schedule|exhibit|annexure)\s+\d+(?:\.\d+)*\b",
    ],
    "DURATION": [
        r"\b\d+\s*(?:calendar\s+)?(?:days?|weeks?|months?|years?)\b",
    ],
}

INVOICE_PATTERNS = {
    "DATE":            LEGAL_PATTERNS["DATE"],
    "MONETARY_VALUE":  LEGAL_PATTERNS["MONETARY_VALUE"],
    "INVOICE_NUMBER":  [r"\b(?:invoice|inv|bill)\s*(?:no|num|number|#)[.:\s]+([A-Z0-9\-]+)\b"],
    "TAX_ID":          [r"\b(?:gst|vat|pan|tin|ein)[:\s]+([A-Z0-9]+)\b"],
    "DUE_DATE":        [r"\b(?:due\s+date|payment\s+due|payable\s+by)[:\s]+[^\n]{5,30}\b"],
}

PATTERN_MAP = {
    "medical_prescription": MEDICAL_PATTERNS,
    "lab_report":           MEDICAL_PATTERNS,
    "legal_contract":       LEGAL_PATTERNS,
    "affidavit":            LEGAL_PATTERNS,
    "invoice":              INVOICE_PATTERNS,
    "unknown":              {},
}


def _regex_extract(text: str, document_type: str) -> list:
    patterns = PATTERN_MAP.get(document_type, {})
    entities = []
    for entity_label, regex_list in patterns.items():
        for pattern in regex_list:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group().strip()
                if len(val) > 1:
                    entities.append(Entity(
                        label=entity_label,
                        value=val,
                        start=m.start(),
                        end=m.end(),
                        source="regex",
                    ))
    return entities


# ─── spaCy NER ───────────────────────────────────────────────────────────────
# Maps spaCy entity types to our schema
SPACY_LABEL_MAP = {
    "PERSON":  "PERSON",
    "ORG":     "ORGANIZATION",
    "GPE":     "LOCATION",
    "LOC":     "LOCATION",
    "DATE":    "DATE",
    "MONEY":   "MONETARY_VALUE",
    "CARDINAL": None,   # ignore
    "ORDINAL":  None,
}


def _spacy_extract(text: str, nlp) -> list:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        mapped = SPACY_LABEL_MAP.get(ent.label_)
        if mapped is None:
            continue
        entities.append(Entity(
            label=mapped,
            value=ent.text,
            start=ent.start_char,
            end=ent.end_char,
            confidence=0.9,
            source="spacy",
        ))
    return entities


# ─── Transformer NER ─────────────────────────────────────────────────────────
# Domain model recommendations:
#   Medical: "d4data/biomedical-ner-all"  or  "allenai/scibert_scivocab_uncased"
#   Legal:   "nlpaueb/legal-bert-base-uncased" (fine-tune for NER)
#   General: "dslim/bert-base-NER"

TRANSFORMER_NER_LABEL_MAP = {
    "B-PER": "PERSON",   "I-PER": "PERSON",
    "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "B-LOC": "LOCATION",  "I-LOC": "LOCATION",
    "B-MISC": None,       "I-MISC": None,
    # Biomedical
    "B-Disease": "CONDITION",   "I-Disease": "CONDITION",
    "B-Chemical": "DRUG",       "I-Chemical": "DRUG",
    "B-Gene": None,
}


def _transformer_ner_extract(text: str, ner_pipe) -> list:
    """Run HuggingFace NER pipeline and group B-/I- tokens."""
    raw = ner_pipe(text[:512])   # hard token limit
    entities = []
    current = None

    for tok in raw:
        label = TRANSFORMER_NER_LABEL_MAP.get(tok["entity"], None)
        if label is None:
            current = None
            continue

        if tok["entity"].startswith("B-") or current is None:
            if current:
                entities.append(current)
            current = Entity(
                label=label,
                value=tok["word"].lstrip("##"),
                start=tok["start"],
                end=tok["end"],
                confidence=round(float(tok["score"]), 4),
                source="transformer",
            )
        elif tok["entity"].startswith("I-") and current:
            word = tok["word"].lstrip("##")
            current.value += ("" if word.startswith("##") else " ") + word
            current.end    = tok["end"]
            current.confidence = min(current.confidence, round(float(tok["score"]), 4))

    if current:
        entities.append(current)

    return entities


# ─── Main NER Extractor ───────────────────────────────────────────────────────
class NERExtractor:
    """
    Multi-tier NER extractor. Auto-selects best available tier.
    """

    def __init__(
        self,
        use_transformer_ner: bool         = False,
        transformer_ner_model: str        = "dslim/bert-base-NER",
        use_spacy: bool                   = False,
        spacy_model: str                  = "en_core_web_sm",
    ):
        self.use_transformer_ner = use_transformer_ner
        self.use_spacy           = use_spacy
        self._ner_pipe           = None
        self._spacy_nlp          = None

        if use_transformer_ner:
            try:
                from transformers import pipeline as hf_pipeline
                logger.info(f"Loading transformer NER: {transformer_ner_model}")
                self._ner_pipe = hf_pipeline(
                    "ner",
                    model=transformer_ner_model,
                    aggregation_strategy=None,  # manual grouping
                    device=-1,
                )
                logger.info("Transformer NER ready")
            except Exception as e:
                logger.warning(f"Transformer NER load failed: {e}")

        if use_spacy:
            try:
                import spacy
                self._spacy_nlp = spacy.load(spacy_model)
                logger.info(f"spaCy model '{spacy_model}' loaded")
            except Exception as e:
                logger.warning(f"spaCy load failed: {e}. "
                               f"Run: python -m spacy download {spacy_model}")

    @mlflow.trace(name="extract_entities")
    def extract(self, text: str, document_type: str) -> NERResult:
        """
        Extract entities. Merges results from all available tiers,
        deduplicating by (label, normalized_value).
        """
        all_entities = []

        # Tier 1: Transformer NER
        if self._ner_pipe is not None:
            try:
                all_entities += _transformer_ner_extract(text, self._ner_pipe)
            except Exception as e:
                logger.warning(f"Transformer NER failed: {e}")

        # Tier 2: spaCy
        if self._spacy_nlp is not None:
            try:
                all_entities += _spacy_extract(text, self._spacy_nlp)
            except Exception as e:
                logger.warning(f"spaCy NER failed: {e}")

        # Tier 3: Regex (always runs — catches domain-specific patterns
        #          that generic models miss, e.g. dosage, invoice numbers)
        all_entities += _regex_extract(text, document_type)

        # Deduplicate: keep highest-confidence entity per (label, value)
        seen = {}
        for ent in all_entities:
            key = (ent.label, ent.value.lower().strip())
            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent

        unique_entities = list(seen.values())

        # Group by type
        by_type: dict = {}
        for ent in unique_entities:
            by_type.setdefault(ent.label, [])
            if ent.value not in by_type[ent.label]:
                by_type[ent.label].append(ent.value)

        mlflow.log_metric("ner_entity_count", len(unique_entities))

        return NERResult(
            entities=unique_entities,
            by_type=by_type,
            doc_type=document_type,
        )

    def to_structured_dict(self, ner_result: NERResult) -> dict:
        return {
            "entity_count":    len(ner_result.entities),
            "entities_by_type": ner_result.by_type,
            "entities_list": [
                {
                    "label":      e.label,
                    "value":      e.value,
                    "confidence": e.confidence,
                    "source":     e.source,
                }
                for e in ner_result.entities
            ],
        }
