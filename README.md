<div align="center">

```
╔═══════════════════════════════════════════════════════════╗
║          MULTI-MODAL OCR SYSTEM  ·  v2.0                  ║
║     Legal & Medical Document Intelligence Pipeline        ║
╚═══════════════════════════════════════════════════════════╝
```

**A production-grade, five-stage AI pipeline that reads scanned legal and medical documents — extracting text, classifying document type, and pulling structured entities — packaged as a single deployable MLflow model.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Live Demo](#) · [API Docs](http://localhost:8000/docs) · [MLflow UI](http://localhost:5000) · [Report Bug](#) · [Request Feature](#)

</div>

---

## The Problem This Solves

Hospitals, law firms, and courts process **thousands of scanned documents daily** — prescriptions, lab reports, contracts, affidavits, invoices. These exist as image files or PDFs that no standard software can read, understand, or route automatically.

**This system gives any institution a single REST call that returns a structured, auditable JSON answer instead of raw unreadable image bytes.** No hallucinations, no free-text generation — every output is a pre-defined schema with confidence scores and human-review flags.

---

## Architecture Overview

```
 INPUT: Image / PDF
        │
        ▼
┌──────────────────────┐
│  Stage 1             │  OpenCV: deskew · denoise · binarize · border crop
│  Preprocessing       │  Fixes skewed scans, noise, illumination variance
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 2             │  EasyOCR (primary) → Tesseract (confidence fallback)
│  OCR Engine          │  Per-word confidence scores · 80+ languages
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 3             │  Unicode fixes · OCR artifact substitution
│  Text Cleaning       │  Whitespace normalization · sentence segmentation
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 4a            │  Tier 1: Fine-tuned BERT  (best accuracy)
│  Classification      │  Tier 2: Zero-shot BART   (no training needed)
│                      │  Tier 3: Keyword baseline  (always available)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stage 4b            │  Transformer NER + spaCy + domain regex patterns
│  Entity Extraction   │  Medical: DRUG, DOSAGE, DOCTOR, LAB_VALUE, DATE
│                      │  Legal:   PARTY, JURISDICTION, MONETARY_VALUE, CLAUSE
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  MLflow PyFunc       │  Single versioned artifact · traced · registered
│  Unified Pipeline    │  Human-review flag · structured JSON output
└──────────────────────┘
```

All five stages are wrapped inside a single **MLflow Custom PyFunc** model — versioned, deployable, and rollback-capable as one unit.

---

## Key Features

| Feature | Description |
|---|---|
| **Dual-engine OCR** | EasyOCR primary with automatic Tesseract fallback based on confidence threshold |
| **3-tier classifier** | Fine-tuned BERT → Zero-shot BART → Keyword baseline, auto-cascades |
| **Domain NER** | 15+ entity types across medical and legal domains; regex + transformer fusion |
| **MLflow PyFunc** | Entire 5-stage pipeline as one versioned, deployable model artifact |
| **MLflow Tracing** | Per-stage latency spans; identify bottlenecks in production |
| **Synthetic training data** | 1,200 labelled samples generated without using any real patient/client data |
| **Hyperparameter search** | Automated grid search with champion model promotion |
| **Human-review flag** | Low-confidence outputs flagged before reaching clinical/legal records |
| **Text-only mode** | Skip OCR entirely when text is already extracted |
| **REST API** | FastAPI with file upload, background training, health endpoint |
| **React frontend** | Dark sci-fi UI with particle canvas, animated pipeline steps, confidence ring |
| **Zero generative output** | All outputs are structured schema — no hallucination risk |

---

## Supported Document Types

| Type | Key Entities Extracted |
|---|---|
| `medical_prescription` | DRUG, DOSAGE, DOCTOR, DATE, PATIENT_AGE, LICENSE_NUMBER |
| `lab_report` | LAB_VALUE, DATE, DOCTOR, PATIENT_AGE |
| `legal_contract` | PARTY, MONETARY_VALUE, JURISDICTION, OBLIGATION, CLAUSE_REFERENCE, DURATION |
| `affidavit` | PARTY, DATE, JURISDICTION |
| `invoice` | INVOICE_NUMBER, MONETARY_VALUE, TAX_ID, DUE_DATE, DATE |
| `unknown` | Flagged for human review |

---

## Project Structure

```
multimodal-ocr-system/
│
├── src/                          # Core pipeline modules
│   ├── preprocessing.py          # Stage 1: OpenCV image pipeline
│   ├── ocr_engine.py             # Stage 2: EasyOCR + Tesseract
│   ├── text_cleaner.py           # Stage 3: Normalisation & segmentation
│   ├── classifier.py             # Stage 4a: 3-tier document classifier
│   ├── ner_extractor.py          # Stage 4b: Multi-tier NER
│   └── pipeline.py               # MLflow PyFunc unified model
│
├── training/                     # BERT fine-tuning system
│   ├── dataset_builder.py        # Synthetic data generator (1,200 samples)
│   └── bert_trainer.py           # HuggingFace Trainer + MLflow logging
│
├── mlflow_setup/                 # Experiment management
│   ├── train_classifier.py       # Register keyword pipeline (Phase 1)
│   ├── hparam_search.py          # Grid search + champion promotion
│   └── evaluate_models.py        # Side-by-side tier comparison
│
├── api/
│   └── serve.py                  # FastAPI: /analyze /classify /train /health
│
├── frontend/
│   ├── index.html                # Standalone (zero build tools required)
│   └── src/                      # Vite + React component project
│       ├── components/           # Header, Tabs, UploadZone, ResultPanel...
│       ├── hooks/useHealth.js    # Live API polling
│       └── utils/api.js          # All endpoint wrappers
│
├── tests/
│   ├── test_pipeline.py          # Phase 1 unit tests
│   └── test_phase2.py            # Phase 2 unit + integration tests (40+)
│
├── data/samples/                 # Auto-generated JSONL training splits
├── run.py                        # Master CLI entry point
└── requirements.txt
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Tesseract OCR engine
# Ubuntu/Debian:
sudo apt install tesseract-ocr

# macOS:
brew install tesseract

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/multimodal-ocr-system.git
cd multimodal-ocr-system

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download spaCy language model
python -m spacy download en_core_web_sm
```

### Run the Demo (no GPU, no OCR needed)

```bash
python run.py demo
```

This runs the full text-only pipeline on three sample documents and prints structured output to terminal. No API server, no GPU, no database — just the core inference chain.

### Full System Startup

```bash
# Terminal 1 — MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2 — FastAPI backend
python run.py serve --port 8000

# Terminal 3 — Open the frontend
open frontend/index.html
# or serve it:
python -m http.server 3000 --directory frontend
```

Visit:
- **Frontend** → `http://localhost:3000`
- **API docs** → `http://localhost:8000/docs`
- **MLflow UI** → `http://localhost:5000`

---

## CLI Reference

```bash
python run.py <command> [options]
```

| Command | What it does |
|---|---|
| `demo` | Runs text-only pipeline on built-in samples. Zero dependencies. |
| `generate` | Generates 1,200 synthetic labelled training samples |
| `train` | Fine-tunes BERT classifier and logs to MLflow |
| `hparam` | Grid search over LR, max_length, dropout, base model |
| `evaluate` | Side-by-side comparison of all 3 classifier tiers |
| `serve` | Starts FastAPI REST server |
| `test` | Runs full test suite (40+ tests, no GPU needed) |
| `all` | generate → demo → test in sequence |

**Options:**

```bash
python run.py train \
  --base-model bert-base-uncased \   # or nlpaueb/legal-bert-base-uncased
  --epochs 5 \
  --lr 2e-5 \
  --batch-size 16 \
  --max-length 256 \
  --data-dir data/samples \
  --output-dir models/bert_classifier

python run.py hparam --max-runs 6
python run.py serve --port 8000
```

---

## API Reference

### `GET /health`

Returns system status, pipeline config, and version.

```json
{
  "status": "ok",
  "version": "2.0.0",
  "pipeline_loaded": true,
  "config": {
    "use_finetuned_bert": true,
    "use_zero_shot": false,
    "confidence_threshold": 0.3
  }
}
```

---

### `POST /analyze`

Upload a document image or PDF. Returns full 5-stage pipeline output.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@prescription.png"
```

```json
{
  "status": "success",
  "requires_human_review": false,
  "review_reasons": [],
  "document": {
    "type": "medical_prescription",
    "classification_confidence": 0.9412,
    "classification_method": "finetuned_bert",
    "all_type_scores": {
      "medical_prescription": 0.9412,
      "lab_report": 0.0321,
      "legal_contract": 0.0108,
      "affidavit": 0.0072,
      "invoice": 0.0063,
      "unknown": 0.0024
    }
  },
  "ocr": {
    "engine_used": "easyocr",
    "avg_confidence": 0.9134,
    "pages_processed": 1,
    "extracted_text": "MEDICAL PRESCRIPTION\nPatient: John Doe...",
    "word_count": 84
  },
  "entities": {
    "entity_count": 7,
    "entities_by_type": {
      "DRUG":   ["Amoxicillin", "Ibuprofen"],
      "DOSAGE": ["500mg", "400mg", "twice daily", "thrice daily"],
      "DATE":   ["14/03/2024"],
      "DOCTOR": ["Dr. Sarah Williams"]
    }
  },
  "performance": {
    "total_latency_ms": 342.5
  }
}
```

---

### `POST /classify`

Classify pre-extracted text without OCR.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This agreement is entered between the parties hereinafter..."}'
```

---

### `POST /train`

Trigger BERT fine-tuning in the background.

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "bert-base-uncased",
    "num_epochs": 5,
    "learning_rate": 0.00002,
    "n_per_class": 200
  }'
```

```json
{
  "status": "training_started",
  "message": "Fine-tuning started in background. Monitor at MLflow UI."
}
```

---

## Training Your Own Classifier

### Step 1 — Generate synthetic data

```bash
python run.py generate --n-per-class 200
# Creates: data/samples/train.jsonl (900 samples)
#          data/samples/val.jsonl   (180 samples)
#          data/samples/test.jsonl  (120 samples)
```

### Step 2 — Fine-tune BERT

```bash
python run.py train --base-model bert-base-uncased --epochs 5
```

**Domain-specific model options:**

| Model | Best for | HuggingFace ID |
|---|---|---|
| General | Both domains | `bert-base-uncased` |
| Legal | Contracts, affidavits | `nlpaueb/legal-bert-base-uncased` |
| Medical | Prescriptions, lab reports | `dmis-lab/biobert-v1.1` |

### Step 3 — Compare against baseline

```bash
python run.py evaluate
```

Example output:
```
Method              Accuracy   F1 Macro  F1 Weighted  Avg Conf
──────────────────────────────────────────────────────────────
keyword_baseline      0.7140     0.7082       0.7195    0.3420
zero_shot_bart        0.8630     0.8591       0.8647    0.7810
finetuned_bert        0.9480     0.9441       0.9472    0.9134
```

### Step 4 — Run hyperparameter search

```bash
python run.py hparam --max-runs 6
# Best model automatically promoted to 'champion' alias in MLflow registry
```

Then load the champion in production:
```python
model = mlflow.pyfunc.load_model("models:/DocumentBERTClassifier@champion")
```

---

## MLflow Integration

Every training run logs:

| Logged Item | Type |
|---|---|
| `test_accuracy`, `test_f1_macro`, `test_f1_weighted` | Metrics |
| `base_model`, `learning_rate`, `max_length`, `dropout` | Parameters |
| Per-class precision / recall / F1 | Metrics |
| Confusion matrix | JSON artifact |
| Full classification report | JSON artifact |
| Class weights | JSON artifact |
| Tokenizer files | Artifact directory |
| PyFunc model | Registered model |

The pipeline uses `@mlflow.trace` on every stage function — open the MLflow Tracing UI to see per-stage latency for any inference call.

---

## Frontend

The React frontend is available in two forms:

### Standalone (zero build tools)

```bash
open frontend/index.html
# or
python -m http.server 3000 --directory frontend
```

Works by loading React from CDN. No `npm install` required.

### Vite Project

```bash
cd frontend
npm install
npm run dev        # development server at localhost:5173
npm run build      # production build in dist/
```

**Environment variable:**

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

**UI features:**
- Animated particle field canvas background
- Drag-and-drop file upload with image preview
- Step-by-step animated pipeline progress (5 stages)
- Animated SVG confidence ring per classification
- Entity chips with per-type colour coding
- Score bars for all 6 document classes
- Live API health polling (30-second interval)
- Human-review flag with reason codes
- Sample document loader (5 built-in examples)
- Trigger BERT training from the browser

---

## Design Decisions

### Why no generative output?

Medical and legal domains have zero tolerance for hallucination. Every entity this system reports was literally present in the source document. Every classification comes with a probability. Low-confidence results surface a `requires_human_review` flag rather than silently passing uncertain output downstream. This makes the system **auditable** — you can always trace any output back to a specific character in the source image.

### Why MLflow PyFunc?

Traditional ML deployments separate OCR and NLP into different services, creating version mismatch risk. By extending `mlflow.pyfunc.PythonModel`, all five stages become one registered artifact. You can deploy version 3, compare it against version 2 on the same document, and roll back in one command — all standard MLflow operations.

### Why synthetic training data?

Real medical prescriptions and legal contracts are confidential, legally restricted, and expensive to annotate. The synthetic generator creates statistically plausible documents that share vocabulary and structure with real ones — without containing any real patient or client data. Because it's code, you can generate any volume and adjust class balance freely.

### Why three classifier tiers?

No single approach is best for all deployment conditions. The cascade means the system never crashes in production: if BERT weights are missing, it falls back to zero-shot; if that fails, keywords always work. All three return the identical `ClassificationResult` dataclass, so downstream code never knows which tier ran.

---

## Running Tests

```bash
# Full test suite (no GPU, no OCR library required)
pytest tests/ -v

# Phase 1 only
pytest tests/test_pipeline.py -v

# Phase 2 only (40+ tests)
pytest tests/test_phase2.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite stubs MLflow and OCR engines so all 40+ tests run in under 10 seconds on any machine. The integration test verifies that generated samples classify at ≥50% accuracy end-to-end.

---

## Configuration

### Environment Variables (API server)

```bash
# Which classifier tier to use
USE_FINETUNED_BERT=true
FINETUNED_MODEL_DIR=models/bert_classifier
FINETUNED_MLFLOW_URI=models:/DocumentBERTClassifier@champion

# Fallback tiers
USE_ZERO_SHOT=false

# NER
USE_TRANSFORMER_NER=false
USE_SPACY_NER=true

# Review threshold
CONFIDENCE_THRESHOLD=0.3
```

### Pipeline Config (Python)

```python
from src.pipeline import MultiModalOCRPipeline

pipeline = MultiModalOCRPipeline(pipeline_config={
    "ocr_languages":         ["en"],
    "use_finetuned_bert":    True,
    "finetuned_model_dir":   "models/bert_classifier",
    "use_zero_shot":         False,
    "use_transformer_ner":   False,
    "use_spacy_ner":         True,
    "confidence_threshold":  0.3,
})
pipeline._build()

result = pipeline.predict(None, {"source": "path/to/document.pdf"})
# or text-only (skip OCR):
result = pipeline.predict(None, {"text": "Rx: Amoxicillin 500mg..."})
```

---

## What the Output Looks Like — Professor Explanation

Five stages, one answer. A scanned prescription enters the system as pixels. Stage 1 straightens and cleans the image. Stage 2 reads the pixels into words. Stage 3 fixes the garbled characters OCR always produces. Stage 4a asks "what kind of document is this?" and answers with a confidence score. Stage 4b asks "what specific facts are in it?" and extracts drug names, dosages, dates, and doctor names. The result is a JSON dictionary you can store in a database, route to the right department, or surface in an EMR system — not an image file that a human has to read manually.

---

## Roadmap

- [ ] Multi-language OCR support (Hindi, Arabic, French)
- [ ] LayoutLM integration for table-aware extraction
- [ ] Active learning loop — flag uncertain samples for annotation
- [ ] Docker + Kubernetes deployment manifests
- [ ] PostgreSQL storage for document audit trail
- [ ] Webhook support for async document processing
- [ ] Streamlit evaluation dashboard
- [ ] Support for `dmis-lab/biobert-v1.1` NER fine-tuning

---

## Requirements

```
Python          3.10+
opencv-python   4.8+
easyocr         1.7+
pytesseract     0.3+
transformers    4.35+
torch           2.0+
accelerate      0.24+
mlflow          2.9+
fastapi         0.104+
spacy           3.7+
scikit-learn    1.3+
```

Full list: [`requirements.txt`](requirements.txt)

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — multi-language deep learning OCR
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — BERT, BART, BioBERT
- [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) — Chalkidis et al., 2020
- [MLflow](https://mlflow.org) — ML lifecycle management
- [spaCy](https://spacy.io) — industrial NLP

---

<div align="center">

Built as a final-year project demonstrating end-to-end ML system design — from raw scanned images to structured, auditable, production-deployable document intelligence.

**⭐ Star this repo if it helped you understand how real ML pipelines are built.**

</div>
