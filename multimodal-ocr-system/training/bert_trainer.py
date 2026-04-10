"""
Phase 2 — BERT Fine-Tuning Trainer
Fine-tunes bert-base-uncased (or legal-bert / bio-bert) for
multi-class document classification with full MLflow tracking.

Training Features:
  - HuggingFace Trainer API with custom callbacks
  - MLflow autolog + manual metric logging
  - Early stopping
  - Weighted loss for class imbalance
  - Checkpoint saving + model registration
  - Per-class evaluation report
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# ── Lazy HuggingFace imports (only needed at runtime) ──────────────────────
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
        confusion_matrix,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    base_model: str          = "bert-base-uncased"
    # Choices: "bert-base-uncased"       → general
    #          "nlpaueb/legal-bert-base-uncased" → legal domain
    #          "dmis-lab/biobert-v1.1"   → medical domain

    # Data
    data_dir: str            = "data/samples"
    output_dir: str          = "models/bert_classifier"
    max_length: int          = 256       # BERT max tokens per sample

    # Training
    num_epochs: int          = 5
    train_batch_size: int    = 16
    eval_batch_size: int     = 32
    learning_rate: float     = 2e-5
    weight_decay: float      = 0.01
    warmup_ratio: float      = 0.1
    grad_accumulation: int   = 2         # effective batch = 16 * 2 = 32

    # Regularisation
    dropout: float           = 0.1
    early_stopping_patience: int = 3

    # MLflow
    experiment_name: str     = "multimodal-ocr-system"
    run_name: str            = "bert-finetune-v1"
    register_as: str         = "DocumentBERTClassifier"

    # Labels
    label2id: Dict[str, int] = field(default_factory=lambda: {
        "medical_prescription": 0,
        "lab_report":           1,
        "legal_contract":       2,
        "affidavit":            3,
        "invoice":              4,
        "unknown":              5,
    })

    @property
    def id2label(self):
        return {v: k for k, v in self.label2id.items()}

    @property
    def num_labels(self):
        return len(self.label2id)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DocumentDataset(Dataset):
    """
    Tokenized HuggingFace-compatible dataset from JSONL file.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                self.samples.append({
                    "text":     obj["text"],
                    "label":    obj["label_id"],
                })

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding=False,          # handled by DataCollator
            max_length=self.max_length,
            return_tensors=None,
        )
        encoding["labels"] = item["label"]
        return encoding


# ─── Custom Trainer with Weighted Loss ────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Overrides compute_loss to support per-class weights.
    Useful when class distribution is imbalanced.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─── Metrics ─────────────────────────────────────────────────────────────────

def make_compute_metrics(id2label: dict):
    """Factory that returns a compute_metrics function for HF Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        return {
            "accuracy":  accuracy_score(labels, preds),
            "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        }

    return compute_metrics


# ─── MLflow Callback ─────────────────────────────────────────────────────────

class MLflowMetricsCallback:
    """
    Simple callback to manually log per-step metrics to MLflow
    alongside HuggingFace Trainer's built-in logging.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


# ─── Main Trainer ─────────────────────────────────────────────────────────────

class BERTDocumentTrainer:
    """
    Orchestrates the full fine-tuning workflow:
    1. Load & tokenize data
    2. Build model
    3. Train with MLflow tracking
    4. Evaluate & log reports
    5. Register model
    """

    def __init__(self, config: TrainConfig):
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers not installed.\n"
                "Run: pip install transformers torch datasets"
            )
        self.config = config
        self.tokenizer  = None
        self.model      = None
        self.trainer    = None

    # ── 1. Data ──────────────────────────────────────────────────────────────

    def load_data(self):
        logger.info(f"Loading tokenizer: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        self.train_dataset = DocumentDataset(
            os.path.join(self.config.data_dir, "train.jsonl"),
            self.tokenizer,
            self.config.max_length,
        )
        self.val_dataset = DocumentDataset(
            os.path.join(self.config.data_dir, "val.jsonl"),
            self.tokenizer,
            self.config.max_length,
        )
        self.test_dataset = DocumentDataset(
            os.path.join(self.config.data_dir, "test.jsonl"),
            self.tokenizer,
            self.config.max_length,
        )

    # ── 2. Model ──────────────────────────────────────────────────────────────

    def build_model(self):
        logger.info(f"Loading model: {self.config.base_model}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
            hidden_dropout_prob=self.config.dropout,
            attention_probs_dropout_prob=self.config.dropout,
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable    = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total_params:,} total, {trainable:,} trainable")
        return total_params, trainable

    # ── 3. Class Weights ──────────────────────────────────────────────────────

    def _compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights to handle class imbalance."""
        counts = np.zeros(self.config.num_labels)
        for sample in self.train_dataset.samples:
            counts[sample["label"]] += 1
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * self.config.num_labels
        return torch.tensor(weights, dtype=torch.float)

    # ── 4. Training ───────────────────────────────────────────────────────────

    def build_trainer(self, class_weights: Optional[torch.Tensor] = None):
        n_train = len(self.train_dataset)
        steps_per_epoch = n_train // (
            self.config.train_batch_size * self.config.grad_accumulation
        )
        total_steps   = steps_per_epoch * self.config.num_epochs
        warmup_steps  = int(total_steps * self.config.warmup_ratio)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=self.config.grad_accumulation,
            # Evaluation & saving
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            # Logging
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=50,
            report_to="none",       # We handle MLflow manually
            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            # Reproducibility
            seed=42,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
        )

        self.trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(self.config.id2label),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                ),
            ],
        )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────

    def evaluate_on_test(self) -> dict:
        """Full evaluation on the held-out test set."""
        logger.info("Running evaluation on test set...")
        preds_output = self.trainer.predict(self.test_dataset)

        logits  = preds_output.predictions
        labels  = preds_output.label_ids
        preds   = np.argmax(logits, axis=-1)

        label_names = [self.config.id2label[i] for i in range(self.config.num_labels)]

        report = classification_report(
            labels, preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(labels, preds)

        per_class_metrics = {}
        for label_name in label_names:
            if label_name in report:
                per_class_metrics[label_name] = {
                    "precision": report[label_name]["precision"],
                    "recall":    report[label_name]["recall"],
                    "f1":        report[label_name]["f1-score"],
                }

        summary = {
            "test_accuracy":  report["accuracy"],
            "test_f1_macro":  report["macro avg"]["f1-score"],
            "test_f1_weighted": report["weighted avg"]["f1-score"],
            "per_class":      per_class_metrics,
            "confusion_matrix": cm.tolist(),
        }
        return summary, report

    # ── 6. Full Run ───────────────────────────────────────────────────────────

    def run(self):
        """Execute the complete training pipeline with MLflow tracking."""
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=self.config.run_name) as run:

            # Log all config parameters
            mlflow.log_params({
                "base_model":             self.config.base_model,
                "max_length":             self.config.max_length,
                "num_epochs":             self.config.num_epochs,
                "train_batch_size":       self.config.train_batch_size,
                "learning_rate":          self.config.learning_rate,
                "weight_decay":           self.config.weight_decay,
                "warmup_ratio":           self.config.warmup_ratio,
                "grad_accumulation":      self.config.grad_accumulation,
                "dropout":                self.config.dropout,
                "early_stopping_patience": self.config.early_stopping_patience,
                "num_labels":             self.config.num_labels,
            })

            # 1. Data
            self.load_data()
            mlflow.log_params({
                "train_samples": len(self.train_dataset),
                "val_samples":   len(self.val_dataset),
                "test_samples":  len(self.test_dataset),
            })

            # 2. Model
            total_params, trainable = self.build_model()
            mlflow.log_params({
                "total_parameters":     total_params,
                "trainable_parameters": trainable,
            })

            # 3. Class weights
            class_weights = self._compute_class_weights()
            weight_dict = {
                self.config.id2label[i]: round(float(w), 4)
                for i, w in enumerate(class_weights)
            }
            logger.info(f"Class weights: {weight_dict}")
            mlflow.log_dict(weight_dict, "class_weights.json")

            # 4. Build trainer
            self.build_trainer(class_weights=class_weights)

            # 5. Train
            logger.info("Starting fine-tuning...")
            train_result = self.trainer.train()

            mlflow.log_metrics({
                "train_runtime_s":         train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "train_loss":              train_result.metrics.get("train_loss", 0),
            })

            # 6. Test evaluation
            test_summary, full_report = self.evaluate_on_test()

            mlflow.log_metrics({
                "test_accuracy":    test_summary["test_accuracy"],
                "test_f1_macro":    test_summary["test_f1_macro"],
                "test_f1_weighted": test_summary["test_f1_weighted"],
            })

            # Per-class metrics
            for label_name, metrics in test_summary["per_class"].items():
                safe = label_name.replace("/", "_")
                mlflow.log_metrics({
                    f"{safe}_precision": metrics["precision"],
                    f"{safe}_recall":    metrics["recall"],
                    f"{safe}_f1":        metrics["f1"],
                })

            # Log confusion matrix and full report
            mlflow.log_dict(test_summary["confusion_matrix"], "confusion_matrix.json")
            mlflow.log_dict(full_report, "classification_report.json")

            logger.info(
                f"\n{'='*50}\n"
                f"Test Accuracy:  {test_summary['test_accuracy']:.4f}\n"
                f"Test F1 Macro:  {test_summary['test_f1_macro']:.4f}\n"
                f"Test F1 Weighted: {test_summary['test_f1_weighted']:.4f}\n"
                f"{'='*50}"
            )

            # 7. Save & register model
            self._save_and_register(run)

            return test_summary, run.info.run_id

    def _save_and_register(self, run):
        """Save tokenizer + model and register in MLflow model registry."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save tokenizer alongside model
        self.tokenizer.save_pretrained(
            os.path.join(self.config.output_dir, "tokenizer")
        )

        # Create dummy input for signature inference
        dummy_text = "Sample prescription with Amoxicillin 500mg twice daily."
        dummy_encoding = self.tokenizer(
            dummy_text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Log model with MLflow
        with torch.no_grad():
            dummy_output = self.model(**dummy_encoding)

        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="bert_classifier",
            registered_model_name=self.config.register_as,
            extra_files=[
                os.path.join(self.config.output_dir, "tokenizer"),
            ],
        )

        # Also log tokenizer config as artifact
        mlflow.log_artifact(
            os.path.join(self.config.output_dir, "tokenizer"),
            artifact_path="tokenizer",
        )

        logger.info(
            f"Model registered as '{self.config.register_as}' "
            f"in MLflow registry."
        )


# ─── Entry Point ─────────────────────────────────────────────────────────────

def train(config: TrainConfig = None):
    config = config or TrainConfig()

    # Build dataset if not present
    train_path = os.path.join(config.data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        logger.info("Dataset not found. Generating synthetic training data...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from training.dataset_builder import build_dataset
        build_dataset(n_per_class=200, output_dir=config.data_dir)

    trainer = BERTDocumentTrainer(config)
    summary, run_id = trainer.run()

    logger.info(f"\nTraining complete. MLflow run ID: {run_id}")
    logger.info("View results: mlflow ui --port 5000")
    return summary, run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BERT document classifier")
    parser.add_argument("--base-model", default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--max-length", type=int,   default=256)
    parser.add_argument("--data-dir",   default="data/samples")
    parser.add_argument("--output-dir", default="models/bert_classifier")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)

    config = TrainConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    train(config)
