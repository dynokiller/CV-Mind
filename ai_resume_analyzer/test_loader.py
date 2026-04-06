import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
from training.data_loader import tokenize_dataset, load_and_prepare_data
from transformers import RobertaTokenizerFast

tr, vl, lc = load_and_prepare_data('data/resume_dataset.csv')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
ds = tokenize_dataset(tr.head(1), tokenizer)

with open('dump.txt', 'w', encoding='utf-8') as f:
    f.write(f"Input IDs: {ds['input_ids'][0][:20]}\n")
    f.write(f"Attention Mask: {ds['attention_mask'][0][:20]}\n")
    f.write(f"Label: {ds['label'][0]}\n")
