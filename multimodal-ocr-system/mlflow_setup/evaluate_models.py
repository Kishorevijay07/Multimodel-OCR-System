"""
Phase 2 — Model Evaluation & Comparison
Compares keyword baseline vs fine-tuned BERT on the test set.
Generates:
  - Per-class precision / recall / F1 table
  - Confusion matrix (text-based for terminal)
  - Confidence calibration summary
  - MLflow comparison table
All results logged to MLflow as artifacts.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from typing import Dict, List

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

LABELS = [
    "medical_prescription",
    "lab_report",
    "legal_contract",
    "affidavit",
    "invoice",
    "unknown",
]


# ─── Load test data ───────────────────────────────────────────────────────────

def load_test_samples(data_dir: str = "data/samples") -> List[Dict]:
    path = os.path.join(data_dir, "test.jsonl")
    if not os.path.exists(path):
        # Generate if missing
        from training.dataset_builder import build_dataset
        build_dataset(n_per_class=200, output_dir=data_dir)
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


# ─── Evaluate a classifier on test samples ───────────────────────────────────

def evaluate_classifier(classifier, samples: List[Dict]) -> Dict:
    y_true, y_pred, confidences = [], [], []

    for s in samples:
        result   = classifier.classify(s["text"])
        y_true.append(s["label"])
        y_pred.append(result.label)
        confidences.append(result.confidence)

    # Metrics
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score,
        classification_report, confusion_matrix,
    )

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    avg_conf = sum(confidences) / len(confidences)

    return {
        "accuracy":         round(acc, 4),
        "f1_macro":         round(f1m, 4),
        "f1_weighted":      round(f1w, 4),
        "avg_confidence":   round(avg_conf, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true":           y_true,
        "y_pred":           y_pred,
        "confidences":      confidences,
    }


# ─── Pretty-print helpers ─────────────────────────────────────────────────────

def print_metrics_table(results: Dict[str, Dict]):
    """Print a comparison table across models."""
    col_w = 22
    header = f"{'Method':<{col_w}} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'Avg Conf':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method, m in results.items():
        print(
            f"{method:<{col_w}} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f} "
            f"{m['f1_weighted']:>12.4f} "
            f"{m['avg_confidence']:>10.4f}"
        )
    print("=" * len(header))


def print_per_class_table(report: Dict, method_name: str):
    print(f"\nPer-class breakdown — {method_name}")
    print(f"{'Label':<28} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 65)
    for label in LABELS:
        if label in report:
            m = report[label]
            print(
                f"{label:<28} "
                f"{m['precision']:>10.3f} "
                f"{m['recall']:>8.3f} "
                f"{m['f1-score']:>8.3f} "
                f"{int(m['support']):>9}"
            )


def print_confusion_matrix(cm: List, labels: List[str]):
    short = [l[:8] for l in labels]
    header = " " * 12 + "  ".join(f"{s:>8}" for s in short)
    print(f"\nConfusion Matrix (rows=True, cols=Pred):\n{header}")
    for i, row in enumerate(cm):
        print(f"{short[i]:>12}  " + "  ".join(f"{v:>8}" for v in row))


# ─── Main comparison run ──────────────────────────────────────────────────────

def run_comparison(
    data_dir: str           = "data/samples",
    finetuned_model_dir: str = None,
    finetuned_mlflow_uri: str = None,
    mlflow_uri: str         = "http://localhost:5000",
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("multimodal-ocr-system")

    logger.info("Loading test samples...")
    samples = load_test_samples(data_dir)
    logger.info(f"Test samples: {len(samples)}")

    all_results = {}

    with mlflow.start_run(run_name="model-comparison-v2"):

        # ── Baseline: Keyword ──────────────────────────────────────────────
        logger.info("\n[1/3] Evaluating keyword baseline...")
        from src.classifier import DocumentClassifier
        kw_clf = DocumentClassifier(
            use_finetuned=False, use_zero_shot=False
        )
        kw_result = evaluate_classifier(kw_clf, samples)
        all_results["keyword_baseline"] = kw_result

        mlflow.log_metrics({
            "keyword_accuracy":   kw_result["accuracy"],
            "keyword_f1_macro":   kw_result["f1_macro"],
            "keyword_f1_weighted": kw_result["f1_weighted"],
        })

        # ── Fine-tuned BERT ────────────────────────────────────────────────
        if finetuned_model_dir or finetuned_mlflow_uri:
            logger.info("\n[2/3] Evaluating fine-tuned BERT...")
            try:
                bert_clf = DocumentClassifier(
                    use_finetuned=True,
                    finetuned_model_dir=finetuned_model_dir,
                    finetuned_mlflow_uri=finetuned_mlflow_uri,
                )
                bert_result = evaluate_classifier(bert_clf, samples)
                all_results["finetuned_bert"] = bert_result

                improvement = bert_result["f1_macro"] - kw_result["f1_macro"]
                mlflow.log_metrics({
                    "bert_accuracy":   bert_result["accuracy"],
                    "bert_f1_macro":   bert_result["f1_macro"],
                    "bert_f1_weighted": bert_result["f1_weighted"],
                    "bert_vs_keyword_f1_improvement": round(improvement, 4),
                })
            except Exception as e:
                logger.warning(f"Fine-tuned BERT eval failed: {e}")
        else:
            logger.info("\n[2/3] Skipping BERT eval (no model path provided)")
            logger.info("      Provide --model-dir or --mlflow-uri to compare")

        # ── Zero-shot BART ─────────────────────────────────────────────────
        logger.info("\n[3/3] Evaluating zero-shot BART (may download ~1.6GB)...")
        try:
            zs_clf = DocumentClassifier(
                use_finetuned=False, use_zero_shot=True
            )
            zs_result = evaluate_classifier(zs_clf, samples)
            all_results["zero_shot_bart"] = zs_result

            mlflow.log_metrics({
                "zeroshot_accuracy":   zs_result["accuracy"],
                "zeroshot_f1_macro":   zs_result["f1_macro"],
            })
        except Exception as e:
            logger.warning(f"Zero-shot eval failed: {e}")

        # ── Print comparison ───────────────────────────────────────────────
        print_metrics_table({
            k: {
                "accuracy":       v["accuracy"],
                "f1_macro":       v["f1_macro"],
                "f1_weighted":    v["f1_weighted"],
                "avg_confidence": v["avg_confidence"],
            }
            for k, v in all_results.items()
        })

        for method, result in all_results.items():
            print_per_class_table(result["classification_report"], method)
            print_confusion_matrix(result["confusion_matrix"], LABELS)

        # ── Log comparison artifact ────────────────────────────────────────
        summary = {
            method: {
                "accuracy":      r["accuracy"],
                "f1_macro":      r["f1_macro"],
                "f1_weighted":   r["f1_weighted"],
                "avg_confidence": r["avg_confidence"],
            }
            for method, r in all_results.items()
        }
        mlflow.log_dict(summary, "comparison_summary.json")

        # Best method
        best_method = max(all_results, key=lambda k: all_results[k]["f1_macro"])
        best_f1     = all_results[best_method]["f1_macro"]
        mlflow.log_param("best_method",   best_method)
        mlflow.log_metric("best_f1_macro", best_f1)

        print(f"\n✅ Best method: {best_method} (F1 Macro: {best_f1:.4f})")

    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     default="data/samples")
    p.add_argument("--model-dir",    default=None,
                   help="Local path to fine-tuned BERT checkpoint")
    p.add_argument("--mlflow-model", default=None,
                   help="MLflow model URI, e.g. models:/DocumentBERTClassifier/1")
    p.add_argument("--mlflow-uri",   default="http://localhost:5000")
    args = p.parse_args()

    run_comparison(
        data_dir=args.data_dir,
        finetuned_model_dir=args.model_dir,
        finetuned_mlflow_uri=args.mlflow_model,
        mlflow_uri=args.mlflow_uri,
    )
