"""
Phase 2 — Hyperparameter Search & Model Comparison
Runs a grid of experiments comparing:
  - Base model variants (bert-base, legal-bert, biobert)
  - Learning rates
  - Sequence lengths
  - Dropout values

All results logged to MLflow for side-by-side comparison.
Best model auto-promoted to 'Champion' alias in registry.
"""

from __future__ import annotations
import itertools
import json
import logging
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ─── Search space ─────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    # Start with bert-base-uncased (always available).
    # Uncomment domain-specific models once base is confirmed working:
    "base_model": [
        "bert-base-uncased",
        # "nlpaueb/legal-bert-base-uncased",   # legal domain
        # "dmis-lab/biobert-v1.1",             # medical domain
    ],
    "learning_rate":    [3e-5, 2e-5, 1e-5],
    "max_length":       [128, 256],
    "dropout":          [0.1, 0.2],
    "num_epochs":       [5],          # fix epochs, vary LR/dropout
}

# ─── Fixed settings ───────────────────────────────────────────────────────────
FIXED = {
    "train_batch_size":      16,
    "grad_accumulation":     2,
    "weight_decay":          0.01,
    "warmup_ratio":          0.1,
    "early_stopping_patience": 3,
    "data_dir":              "data/samples",
    "output_dir":            "models/hparam_search",
    "experiment_name":       "multimodal-ocr-hparam-search",
    "register_as":           "DocumentBERTClassifier",
}


def run_search(max_runs: int = 6, mlflow_uri: str = "http://localhost:5000"):
    """
    Run grid search up to max_runs experiments.
    Returns the best run_id and its metrics.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(FIXED["experiment_name"])

    # Generate all combinations
    keys   = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*SEARCH_SPACE.values()))

    logger.info(f"Total combinations: {len(combos)} | Running up to: {max_runs}")

    results = []

    for i, values in enumerate(combos[:max_runs]):
        params = dict(zip(keys, values))
        run_name = (
            f"{params['base_model'].split('/')[-1]}"
            f"_lr{params['learning_rate']:.0e}"
            f"_len{params['max_length']}"
            f"_drop{params['dropout']}"
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{min(max_runs, len(combos))}: {run_name}")
        logger.info(f"Params: {params}")

        try:
            from training.bert_trainer import BERTDocumentTrainer, TrainConfig

            config = TrainConfig(
                **params,
                **{k: v for k, v in FIXED.items()
                   if k in TrainConfig.__dataclass_fields__},
                run_name=run_name,
            )
            config.output_dir = os.path.join(FIXED["output_dir"], run_name)
            os.makedirs(config.output_dir, exist_ok=True)

            trainer = BERTDocumentTrainer(config)
            summary, run_id = trainer.run()

            results.append({
                "run_id":        run_id,
                "run_name":      run_name,
                "params":        params,
                "test_accuracy": summary["test_accuracy"],
                "test_f1_macro": summary["test_f1_macro"],
            })
            logger.info(
                f"Run complete — F1 Macro: {summary['test_f1_macro']:.4f} | "
                f"Accuracy: {summary['test_accuracy']:.4f}"
            )

        except Exception as e:
            logger.error(f"Run {run_name} FAILED: {e}", exc_info=True)
            continue

    if not results:
        logger.warning("No successful runs.")
        return None

    # ── Select best ──────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["test_f1_macro"])
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST RUN: {best['run_name']}")
    logger.info(f"  F1 Macro:  {best['test_f1_macro']:.4f}")
    logger.info(f"  Accuracy:  {best['test_accuracy']:.4f}")
    logger.info(f"  Params:    {best['params']}")

    # ── Promote best to Champion alias ────────────────────────────────────────
    try:
        client = MlflowClient()
        versions = client.search_model_versions(
            f"run_id='{best['run_id']}'"
        )
        if versions:
            mv = versions[0]
            client.set_registered_model_alias(
                name=FIXED["register_as"],
                alias="champion",
                version=mv.version,
            )
            logger.info(
                f"Promoted model version {mv.version} to 'champion' alias."
            )
    except Exception as e:
        logger.warning(f"Could not promote champion: {e}")

    # Save leaderboard
    results.sort(key=lambda r: r["test_f1_macro"], reverse=True)
    leaderboard_path = os.path.join(FIXED["output_dir"], "leaderboard.json")
    os.makedirs(FIXED["output_dir"], exist_ok=True)
    with open(leaderboard_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Leaderboard saved: {leaderboard_path}")

    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs",    type=int, default=6)
    parser.add_argument("--mlflow-uri",  default="http://localhost:5000")
    args = parser.parse_args()

    run_search(max_runs=args.max_runs, mlflow_uri=args.mlflow_uri)
