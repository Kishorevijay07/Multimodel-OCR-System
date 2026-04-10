#!/usr/bin/env python3
"""
Phase 2 — Master Run Script
Single entry point to orchestrate all Phase 2 operations.

Usage:
  python run.py generate          # Generate synthetic training data
  python run.py train             # Fine-tune BERT
  python run.py hparam            # Hyperparameter search
  python run.py evaluate          # Compare all models
  python run.py serve             # Start API server
  python run.py test              # Run test suite
  python run.py demo              # Quick text-only demo
  python run.py all               # Full pipeline end-to-end
"""

import sys
import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_generate(args):
    """Generate synthetic training data."""
    from training.dataset_builder import build_dataset
    stats = build_dataset(
        n_per_class=args.n_per_class,
        output_dir=args.data_dir,
    )
    logger.info(f"Dataset ready: {stats}")


def cmd_train(args):
    """Fine-tune BERT document classifier."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    from training.bert_trainer import BERTDocumentTrainer, TrainConfig
    config = TrainConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        train_batch_size=args.batch_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=f"bert-{args.base_model.split('/')[-1]}-finetune",
    )

    # Ensure data exists
    if not os.path.exists(os.path.join(args.data_dir, "train.jsonl")):
        logger.info("Data not found — generating first...")
        cmd_generate(args)

    trainer = BERTDocumentTrainer(config)
    summary, run_id = trainer.run()
    logger.info(f"\nTraining complete. Run ID: {run_id}")
    logger.info(f"F1 Macro: {summary['test_f1_macro']:.4f}")
    logger.info(f"Accuracy: {summary['test_accuracy']:.4f}")
    return run_id


def cmd_hparam(args):
    """Run hyperparameter search."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    from mlflow_setup.hparam_search import run_search
    best = run_search(max_runs=args.max_runs, mlflow_uri=MLFLOW_URI)
    if best:
        logger.info(f"Best: {best['run_name']} — F1: {best['test_f1_macro']:.4f}")


def cmd_evaluate(args):
    """Evaluate and compare all model tiers."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    from mlflow_setup.evaluate_models import run_comparison
    run_comparison(
        data_dir=args.data_dir,
        finetuned_model_dir=args.output_dir if os.path.exists(args.output_dir) else None,
        mlflow_uri=MLFLOW_URI,
    )


def cmd_serve(args):
    """Start FastAPI server."""
    import uvicorn
    env = {
        "USE_FINETUNED_BERT":  str(os.path.exists(args.output_dir)).lower(),
        "FINETUNED_MODEL_DIR": args.output_dir,
        "USE_ZERO_SHOT":       "false",
        "CONFIDENCE_THRESHOLD": "0.3",
    }
    for k, v in env.items():
        os.environ[k] = v

    logger.info(f"Starting API server on http://0.0.0.0:{args.port}")
    uvicorn.run("api.serve:app", host="0.0.0.0", port=args.port, reload=False)


def cmd_test(args):
    """Run test suite."""
    import pytest
    exit_code = pytest.main([
        "tests/test_phase2.py",
        "-v",
        "--tb=short",
        "-x" if args.fail_fast else "",
    ])
    sys.exit(exit_code)


def cmd_demo(args):
    """Run a quick text-only demo without GPU or file I/O."""
    from src.pipeline import MultiModalOCRPipeline

    pipeline = MultiModalOCRPipeline(pipeline_config={
        "use_finetuned_bert":  False,
        "use_zero_shot":       False,
        "use_transformer_ner": False,
        "use_spacy_ner":       False,
        "confidence_threshold": 0.2,
    })
    pipeline._build()

    demo_docs = {
        "Medical Prescription": (
            "MEDICAL PRESCRIPTION\n"
            "Patient: John Doe, 45M\n"
            "Date: 14/03/2024\n"
            "Rx:\n"
            "1. Amoxicillin 500mg — twice daily for 7 days\n"
            "2. Ibuprofen 400mg — thrice daily after meals\n"
            "Prescribed by: Dr. Sarah Williams\n"
            "Refills: 0"
        ),
        "Legal Contract": (
            "SERVICE AGREEMENT\n"
            "This agreement is entered on 10/03/2024 between:\n"
            "XYZ Technology Solutions Pvt. Ltd. (hereinafter 'Service Provider')\n"
            "AND ABC Retail Corporation (hereinafter 'Client')\n\n"
            "1. PAYMENT: Client shall pay USD 15,000 per month.\n"
            "2. TERMINATION: 30 days written notice required.\n"
            "3. JURISDICTION: Governed by laws of State of California."
        ),
        "Lab Report": (
            "PATHCARE DIAGNOSTICS — LABORATORY REPORT\n"
            "Patient: Jane Smith    Sample: LAB-2024-421\n"
            "HbA1c: 6.1%  (Normal < 5.7%)  — ELEVATED\n"
            "Fasting Glucose: 108 mg/dL  (Normal < 100)  — BORDERLINE\n"
            "Hemoglobin: 11.2 g/dL  — LOW\n"
            "WBC: 7.5 K/uL  — NORMAL"
        ),
    }

    print("\n" + "="*70)
    print("  Multi-Modal OCR System - Phase 2 Demo")
    print("="*70)

    for doc_name, text in demo_docs.items():
        print(f"\n{'-'*70}")
        print(f"  Input: {doc_name}")
        print(f"{'-'*70}")
        result = pipeline.predict(None, {"text": text})

        doc = result["document"]
        ent = result["entities"]
        ocr = result["ocr"]

        print(f"  [OK] Type:        {doc['type']}")
        print(f"  [..] Confidence:  {doc['classification_confidence']:.2%}")
        print(f"  [>>] Method:      {doc['classification_method']}")
        print(f"  [##] Words:       {ocr['word_count']}")
        print(f"  [@@] Entities:    {ent['entity_count']} found")

        if ent["entities_by_type"]:
            for label, values in ent["entities_by_type"].items():
                print(f"      {label}: {', '.join(values[:3])}")

        if result["requires_human_review"]:
            print(f"  [!!] Review:     {result['review_reasons']}")

        print(f"  [ms] Latency:    {result['performance']['total_latency_ms']:.1f} ms")

    print(f"\n{'='*70}")
    print("  Demo complete. Start the API: python run.py serve")
    print("="*70 + "\n")


def cmd_all(args):
    """Run the full Phase 2 pipeline end-to-end."""
    logger.info("Phase 2 — Full Pipeline Run")
    logger.info("Step 1/4: Generate data")
    cmd_generate(args)
    logger.info("Step 2/4: Demo (no GPU needed)")
    cmd_demo(args)
    logger.info("Step 3/4: Run tests")
    cmd_test(args)
    logger.info("Step 4/4: To train BERT, run: python run.py train")
    logger.info("          To start API, run:  python run.py serve")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Multi-Modal OCR System — Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("command", choices=[
        "generate", "train", "hparam", "evaluate",
        "serve", "test", "demo", "all",
    ])

    # Shared options
    p.add_argument("--data-dir",    default="data/samples")
    p.add_argument("--output-dir",  default="models/bert_classifier")
    p.add_argument("--n-per-class", type=int,   default=200)

    # Training options
    p.add_argument("--base-model",  default="bert-base-uncased")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max-length",  type=int,   default=256)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--max-runs",    type=int,   default=6)

    # Serve options
    p.add_argument("--port",        type=int,   default=8000)

    # Test options
    p.add_argument("--fail-fast",   action="store_true")

    args = p.parse_args()

    commands = {
        "generate": cmd_generate,
        "train":    cmd_train,
        "hparam":   cmd_hparam,
        "evaluate": cmd_evaluate,
        "serve":    cmd_serve,
        "test":     cmd_test,
        "demo":     cmd_demo,
        "all":      cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
