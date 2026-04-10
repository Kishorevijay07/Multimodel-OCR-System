"""
Phase 2 — FastAPI REST API
Serves the full pipeline including fine-tuned BERT classifier.
Adds:
  - /train    POST  — trigger training from API
  - /evaluate POST  — run evaluation and return comparison
  - /classify POST  — text-only classification
  - /analyze  POST  — full pipeline (image/PDF)
  - /health   GET   — health + loaded model info
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global pipeline instance ─────────────────────────────────────────────────
PIPELINE = None
PIPELINE_CONFIG = {}


def _load_pipeline():
    global PIPELINE, PIPELINE_CONFIG
    from src.pipeline import MultiModalOCRPipeline

    config = {
        "ocr_languages":         ["en"],
        "use_finetuned_bert":    os.getenv("USE_FINETUNED_BERT", "false").lower() == "true",
        "finetuned_model_dir":   os.getenv("FINETUNED_MODEL_DIR"),
        "finetuned_mlflow_uri":  os.getenv("FINETUNED_MLFLOW_URI"),
        "use_zero_shot":         os.getenv("USE_ZERO_SHOT", "false").lower() == "true",
        "use_transformer_ner":   os.getenv("USE_TRANSFORMER_NER", "false").lower() == "true",
        "use_spacy_ner":         os.getenv("USE_SPACY_NER", "false").lower() == "true",
        "confidence_threshold":  float(os.getenv("CONFIDENCE_THRESHOLD", "0.3")),
    }
    PIPELINE_CONFIG = config
    PIPELINE = MultiModalOCRPipeline(pipeline_config=config)
    PIPELINE._build()
    logger.info(f"Pipeline loaded with config: {config}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pipeline()
    yield


app = FastAPI(
    title="Multi-Modal OCR API — Phase 2",
    description=(
        "Legal & Medical document OCR with fine-tuned BERT classification "
        "and transformer-enhanced NER."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────

class TextClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Document text to classify")
    document_type: str = Field("auto", description="Force document type or 'auto'")


class TrainRequest(BaseModel):
    base_model: str     = "bert-base-uncased"
    num_epochs: int     = 5
    learning_rate: float = 2e-5
    n_per_class: int    = 200
    mlflow_uri: str     = "http://localhost:5000"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "version":     "2.0.0",
        "pipeline_loaded": PIPELINE is not None,
        "config": {
            k: v for k, v in PIPELINE_CONFIG.items()
            if "uri" not in k and "dir" not in k
        },
    }


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload image or PDF → full OCR + classification + NER pipeline.
    Returns structured JSON with document type, entities, and confidence.
    """
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    if ext not in allowed:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}"
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(413, "File too large. Maximum 50 MB.")

    try:
        result = PIPELINE.predict(None, {"source": content})
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline error: {str(e)}")


@app.post("/classify")
def classify_text(req: TextClassifyRequest):
    """
    Classify pre-extracted text without OCR.
    Faster; useful when text is already available.
    """
    try:
        result = PIPELINE.predict(None, {"text": req.text})
        if req.document_type != "auto":
            # Override classifier with forced type
            result["document"]["type"] = req.document_type
            result["document"]["classification_method"] = "forced"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train")
def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start fine-tuning BERT in the background.
    Returns immediately; check /health for completion.
    """
    def _train():
        try:
            import mlflow
            mlflow.set_tracking_uri(req.mlflow_uri)

            from training.dataset_builder import build_dataset
            build_dataset(n_per_class=req.n_per_class)

            from training.bert_trainer import BERTDocumentTrainer, TrainConfig
            config = TrainConfig(
                base_model=req.base_model,
                num_epochs=req.num_epochs,
                learning_rate=req.learning_rate,
                run_name=f"api-triggered-{req.base_model.split('/')[-1]}",
            )
            trainer = BERTDocumentTrainer(config)
            summary, run_id = trainer.run()
            logger.info(f"Background training complete. Run ID: {run_id}")
        except Exception as e:
            logger.error(f"Background training failed: {e}", exc_info=True)

    background_tasks.add_task(_train)
    return {
        "status":  "training_started",
        "message": (
            "Fine-tuning started in background. "
            "Monitor progress at your MLflow UI."
        ),
        "config": req.dict(),
    }


@app.post("/evaluate")
def evaluate(
    data_dir: str = "data/samples",
    model_dir: str = None,
):
    """
    Run model comparison evaluation and return summary.
    """
    try:
        from mlflow_setup.evaluate_models import run_comparison
        results = run_comparison(
            data_dir=data_dir,
            finetuned_model_dir=model_dir,
        )
        return {
            k: {
                "accuracy":   v["accuracy"],
                "f1_macro":   v["f1_macro"],
                "f1_weighted": v["f1_weighted"],
            }
            for k, v in results.items()
        }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )