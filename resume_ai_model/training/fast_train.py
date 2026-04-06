import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = "roberta-base"
EPOCHS = 1
BATCH_SIZE = 4
LR = 2e-5

def train():
    df = pd.read_csv("data/synthetic_resumes.csv")
    # Take a tiny sub-sample to fast-track training on CPU for the API to boot
    df = df.groupby('domain').apply(lambda x: x.sample(n=10, random_state=42)).reset_index(drop=True)
    
    domains = df['domain'].unique().tolist()
    label2id = {d: i for i, d in enumerate(domains)}
    id2label = {i: d for i, d in enumerate(domains)}
    df['label'] = df['domain'].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(examples["resume_text"], truncation=True, padding="max_length", max_length=128)

    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    train_ds = train_ds.add_column("labels", train_df["label"].tolist())
    
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)
    test_ds = test_ds.add_column("labels", test_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(domains), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_strategy="no",
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    print("Running fast-track training loop to export model artifact...")
    trainer.train()
    
    # Save model
    os.makedirs("models/domain_classifier", exist_ok=True)
    trainer.save_model("models/domain_classifier")
    tokenizer.save_pretrained("models/domain_classifier")
    print("Fast-track model saved to models/domain_classifier")

if __name__ == "__main__":
    train()
