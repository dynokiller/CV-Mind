"""
training/train.py

Fine-tunes RoBERTa for multi-class resume domain classification using
the HuggingFace Trainer API.

Key optimisations (v3 — clean & stable):
  - MAX_LENGTH 512 tokens (see data_loader.py)
  - lr 2e-5 with warmup_ratio=0.1
  - 8 epochs + early stopping patience=3
  - gradient_accumulation_steps=4 → effective batch size 32
  - label_smoothing_factor=0.1 (via TrainingArguments — native HuggingFace)
  - fp16 on GPU for speed
  - NO custom Trainer (the WeightedTrainer approach caused loss explosion)

Usage:
    python -m training.train
    # or with custom args:
    python -m training.train --epochs 8 --batch_size 8 --lr 2e-5
"""

import os
import argparse
import numpy as np
import torch
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

from training.data_loader import get_tokenized_datasets

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        class_weights = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "roberta-base"
OUTPUT_MODEL_DIR = "models/roberta-domain"
CHECKPOINTS_DIR  = "models/checkpoints"


def compute_metrics(eval_pred):
    """Called by Trainer after every evaluation step."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def train(
    epochs: int         = 8,
    batch_size: int     = 8,
    lr: float           = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    dataset_path: str   = "data/resume_dataset.csv",
):
    """
    Full training loop:
      1. Load & tokenize dataset
      2. Build RoBERTa classifier
      3. Configure Trainer with proven hyperparameters
      4. Train + evaluate
      5. Save final model + tokenizer
    """
    print("=" * 60)
    print("  RoBERTa Domain Classifier — Training (v3 clean)")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_ds, val_ds, le, num_classes = get_tokenized_datasets(dataset_path)
    print(f"\n[Train] Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    labels_list = train_ds["label"]
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_list),
        y=labels_list
    )
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
    print("[Train] Computed class weights for imbalanced domains.")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    id2label = {i: cls for i, cls in enumerate(le.classes_)}
    label2id = {cls: i for i, cls in id2label.items()}

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    model.to(device)

    print("[Train] Full fine-tuning enabled. All RoBERTa layers will be updated.")

    # ── 3. Training Arguments ─────────────────────────────────────────────────
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = CHECKPOINTS_DIR,
        num_train_epochs            = epochs,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        per_device_eval_batch_size  = 16,
        learning_rate               = lr,
        warmup_ratio                = warmup_ratio,
        weight_decay                = weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_weighted",
        greater_is_better           = True,
        logging_dir                 = "logs/train",
        logging_steps               = 50,
        label_smoothing_factor      = 0.1,
        bf16                        = False,
        fp16                        = False,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        save_total_limit            = 3,
    )

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    trainer = CustomTrainer(
        class_weights   = class_weights_tensor,
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n[Train] Starting training (v3)...\n")
    train_result = trainer.train()

    # ── 5. Final Evaluate ─────────────────────────────────────────────────────
    print("\n[Train] Final evaluation on validation set:")
    metrics = trainer.evaluate()
    print(metrics)

    # ── 6. Save model + tokenizer ─────────────────────────────────────────────
    print(f"\n[Train] Saving model -> {OUTPUT_MODEL_DIR}")
    trainer.save_model(OUTPUT_MODEL_DIR)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    print("[Train] Training complete.")
    print(f"  Train loss    : {train_result.training_loss:.4f}")
    print(f"  Val accuracy  : {metrics.get('eval_accuracy')}")
    print(f"  Val F1        : {metrics.get('eval_f1_weighted')}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for resume domain classification")
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--dataset",    type=str,   default="data/resume_dataset.csv")
    args = parser.parse_args()

    train(
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        dataset_path = args.dataset,
    )
