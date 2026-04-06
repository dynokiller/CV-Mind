"""
training/data_loader.py

Loads the Kaggle resume CSV, cleans text, encodes labels, and
returns HuggingFace-compatible tokenized datasets for RoBERTa.

Dataset columns expected:
  - Resume_str  (raw resume text)
  - Category    (domain label string)
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import RobertaTokenizerFast
import pickle

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH     = "data/resume_dataset.csv"
MODEL_CHECKPOINT = "roberta-base"
MAX_LENGTH       = 512   # Increased from 256 → 512 for better resume coverage
TEST_SIZE        = 0.2
RANDOM_SEED      = 42
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


def clean_text(text: str) -> str:
    """Clean and smart-truncate resume text.

    Resumes avg 6000+ chars but RoBERTa max is 512 tokens (~350-400 words).
    Strategy: take first 2000 chars (summary/skills) + last 500 chars (education)
    to capture the most domain-relevant signals before tokenization.
    """
    text = str(text)
    # Remove HTML tags aggressively (handles both <tag> and malformed HTML)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove HTML entities (&amp; &nbsp; etc.)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove CSS-like content (id="...", style="...", class="...")
    text = re.sub(r'(?:id|style|class|itemprop|itemscope|itemtype)=["\'][^"\']*["\']', " ", text)
    # Keep alphanumeric + key punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,\-+#/]", " ", text)
    # Collapse whitespace + lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Smart truncation: first 2000 chars (skills/summary/experience) + last 500 chars (education)
    # This covers ~400 tokens at 512 limit — enough signal for RoBERTa
    if len(text) > 2500:
        text = text[:2000] + " " + text[-500:]
    return text


def load_and_prepare_data(dataset_path: str = DATASET_PATH):
    """
    Reads CSV, cleans text, encodes labels, splits into train/val.

    Returns:
        train_df (pd.DataFrame), val_df (pd.DataFrame), label_encoder (LabelEncoder)
    """
    print(f"[DataLoader] Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            f"Please place your resume CSV at that path."
        )

    df = pd.read_csv(dataset_path)

    # ── Normalise column names ────────────────────────────────────────────────
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("resume_str", "resume", "resume_text"):
            col_map[col] = "Resume_str"
        elif lower in ("category", "label", "domain"):
            col_map[col] = "Category"
    df.rename(columns=col_map, inplace=True)

    if "Resume_str" not in df.columns or "Category" not in df.columns:
        raise ValueError(
            "CSV must contain text column (Resume_str/resume/resume_text) "
            "and label column (Category/label/domain)."
        )

    print(f"[DataLoader] Raw shape: {df.shape}")
    df.dropna(subset=["Resume_str", "Category"], inplace=True)
    df["Resume_str"] = df["Resume_str"].apply(clean_text)

    # ── Encode string labels → integers ──────────────────────────────────────
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])
    num_classes = len(le.classes_)
    print(f"[DataLoader] Classes ({num_classes}): {list(le.classes_)}")

    # ── Print class distribution for awareness ────────────────────────────────
    dist = df["Category"].value_counts()
    print(f"[DataLoader] Class distribution (min={dist.min()}, max={dist.max()}):")
    print(dist.to_string())

    # ── Persist label encoder so inference can decode predictions ─────────────
    os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[DataLoader] Label encoder saved → {LABEL_ENCODER_PATH}")

    # ── Train / validation split ──────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df[["Resume_str", "label"]],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    print(f"[DataLoader] Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), le


def tokenize_dataset(df: pd.DataFrame, tokenizer: RobertaTokenizerFast, max_length: int = MAX_LENGTH) -> Dataset:
    """Converts a pandas DataFrame to a tokenized HuggingFace Dataset."""

    hf_dataset = Dataset.from_pandas(df.rename(columns={"Resume_str": "text"}))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized


def get_tokenized_datasets(dataset_path: str = DATASET_PATH):
    """
    Full pipeline: load CSV → clean → split → tokenize.

    Returns:
        train_dataset, val_dataset, label_encoder, num_classes
    """
    train_df, val_df, le = load_and_prepare_data(dataset_path)
    num_classes = len(le.classes_)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds   = tokenize_dataset(val_df, tokenizer)

    return train_ds, val_ds, le, num_classes
