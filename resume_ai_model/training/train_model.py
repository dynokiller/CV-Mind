import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

MODEL_NAME = "roberta-base"
EPOCHS = 8
BATCH_SIZE = 16
LR = 2e-5

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

def train():
    df = pd.read_csv("data/synthetic_resumes.csv")
    domains = df['domain'].unique().tolist()
    label2id = {d: i for i, d in enumerate(domains)}
    id2label = {i: d for i, d in enumerate(domains)}
    df['label'] = df['domain'].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(examples["resume_text"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(domains), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training pipeline...")
    trainer.train()
    
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)
    
    # Save model
    os.makedirs("models/domain_classifier", exist_ok=True)
    trainer.save_model("models/domain_classifier")
    tokenizer.save_pretrained("models/domain_classifier")
    print("Model saved to models/domain_classifier")

if __name__ == "__main__":
    train()
