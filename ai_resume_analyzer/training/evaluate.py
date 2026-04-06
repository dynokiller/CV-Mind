"""
training/evaluate.py

Loads the fine-tuned RoBERTa model and runs a full evaluation pass:
  - Accuracy
  - Weighted F1
  - Per-class precision / recall / F1
  - Confusion matrix (text table)

Usage:
    python -m training.evaluate
"""

import os
import pickle
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from training.data_loader import load_and_prepare_data

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR          = "models/roberta-domain"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
RESULTS_FILE       = "evaluation_results.txt"


def load_label_encoder():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        return pickle.load(f)


def evaluate(dataset_path: str = "data/resume_dataset.csv"):
    """Full evaluation of the saved RoBERTa model on the validation split."""

    print("=" * 60)
    print("  RoBERTa Domain Classifier — Evaluation")
    print("=" * 60)

    # ── Load saved model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_DIR}'. Run training/train.py first."
        )

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    le        = load_label_encoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"[Evaluate] Device: {device}")

    # ── Build validation dataset ──────────────────────────────────────────────
    _, val_df, _ = load_and_prepare_data(dataset_path)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    hf_val = Dataset.from_pandas(val_df.rename(columns={"Resume_str": "text"}))
    hf_val = hf_val.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ── Run inference with dummy TrainingArguments (no training) ──────────────
    dummy_args = TrainingArguments(
        output_dir                  = "/tmp/eval_dummy",
        per_device_eval_batch_size  = 16,
        report_to                   = "none",
        dataloader_num_workers      = 0,
    )
    trainer = Trainer(model=model, args=dummy_args)

    print("[Evaluate] Running predictions...")
    predictions_output = trainer.predict(hf_val)
    logits   = predictions_output.predictions   # (N, num_classes)
    true_ids = predictions_output.label_ids      # (N,)

    pred_ids = np.argmax(logits, axis=-1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc        = accuracy_score(true_ids, pred_ids)
    f1_weighted = f1_score(true_ids, pred_ids, average="weighted", zero_division=0)
    f1_macro    = f1_score(true_ids, pred_ids, average="macro",    zero_division=0)

    class_names  = list(le.classes_)
    cls_report   = classification_report(true_ids, pred_ids, target_names=class_names, zero_division=0)
    conf_matrix  = confusion_matrix(true_ids, pred_ids)

    # ── Print & save ──────────────────────────────────────────────────────────
    summary = (
        f"\n{'=' * 60}\n"
        f"  EVALUATION RESULTS\n"
        f"{'=' * 60}\n"
        f"  Accuracy           : {acc:.4f}\n"
        f"  F1 Weighted        : {f1_weighted:.4f}\n"
        f"  F1 Macro           : {f1_macro:.4f}\n"
        f"\n--- Per-Class Report ---\n"
        f"{cls_report}\n"
        f"--- Confusion Matrix (indices = class order above) ---\n"
        f"{conf_matrix}\n"
        f"{'=' * 60}\n"
    )

    print(summary)

    with open(RESULTS_FILE, "w") as f:
        f.write(summary)

    print(f"[Evaluate] Results saved → {RESULTS_FILE}")
    return {
        "accuracy":     round(acc, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "f1_macro":     round(f1_macro, 4),
    }


if __name__ == "__main__":
    evaluate()
