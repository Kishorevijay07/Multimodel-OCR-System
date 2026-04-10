"""
Train & log the full pipeline with MLflow.
Run this to register the model for the first time.
"""

import mlflow
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import build_and_log_pipeline, MultiModalOCRPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(pipeline: MultiModalOCRPipeline, sample_dir: str):
    """Run the pipeline on sample documents and log metrics."""
    import os, json

    sample_files = [
        ("sample_prescription.png", "medical_prescription"),
        ("sample_lab_report.png", "lab_report"),
        ("sample_legal_contract.png", "legal_contract"),
    ]

    results = []
    correct = 0

    for filename, expected_type in sample_files:
        path = os.path.join(sample_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Sample not found: {path}")
            continue

        output = pipeline.predict(None, {"source": path})
        predicted = output["document"]["type"]
        is_correct = predicted == expected_type
        if is_correct:
            correct += 1

        results.append({
            "file": filename,
            "expected": expected_type,
            "predicted": predicted,
            "correct": is_correct,
            "ocr_confidence": output["ocr"]["avg_confidence"],
            "classification_confidence": output["document"]["classification_confidence"],
        })

        logger.info(
            f"{filename}: expected={expected_type}, "
            f"predicted={predicted}, correct={is_correct}"
        )

    accuracy = correct / len(results) if results else 0.0
    mlflow.log_metric("classification_accuracy", accuracy)
    mlflow.log_metric("total_samples_evaluated", len(results))
    mlflow.log_dict({"evaluation_results": results}, "evaluation_results.json")

    logger.info(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(results)})")
    return accuracy


if __name__ == "__main__":
    SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")

    # Generate samples if they don't exist
    if not os.path.exists(os.path.join(SAMPLE_DIR, "sample_prescription.png")):
        logger.info("Generating sample documents...")
        os.chdir(SAMPLE_DIR)
        from data.samples.generate_samples import (
            create_text_image, MEDICAL_PRESCRIPTION, LAB_REPORT, LEGAL_CONTRACT
        )
        create_text_image(MEDICAL_PRESCRIPTION, "sample_prescription.png")
        create_text_image(LAB_REPORT, "sample_lab_report.png")
        create_text_image(LEGAL_CONTRACT, "sample_legal_contract.png")

    # Build and evaluate pipeline
    mlflow.set_tracking_uri("file:./mlruns")  # or http://localhost:5000 if running the server

    config = {
        "ocr_languages": ["en"],
        "use_transformer": False,      # set True to use BERT zero-shot
        "confidence_threshold": 0.3,
        "preprocessor": {
            "target_dpi": 300,
            "denoise_strength": 10,
            "deskew": True,
            "remove_borders": True,
        },
    }

    model_uri = build_and_log_pipeline(
        experiment_name="multimodal-ocr-system",
        run_name="pipeline-v1-keyword",
        register_as="MultiModalOCRPipeline",
        config=config,
    )

    logger.info(f"\nModel registered at: {model_uri}")
    logger.info("Run 'mlflow ui' to view results in the browser.")