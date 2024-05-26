from models.text_models import TextToSql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.utils import Metrics, CustomDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


DATASET = "/Users/diegolopes/Repositories/ms-usp-text-to-sql/data/spider_dataset.csv"
DEVICE = "cpu"

df = pd.read_csv(DATASET)
df.dropna(inplace=True)

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(DEVICE)

X_train, X_test, y_train, y_test = train_test_split(df['question'], df['query'], test_size=0.2, random_state=42)

task_prefix = "translate portuguese to sql: "
max_source_length = 512
max_target_length = 128

x_train_encoding = tokenizer(
    list(X_train),
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
y_train_encoding = tokenizer(
    list(y_train),
    padding="longest",
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)
x_test_encoding = tokenizer(
    list(X_test),
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
y_test_encoding = tokenizer(
    list(y_test),
    padding="longest",
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

train_dataset = CustomDataset(
    input_ids=x_train_encoding.input_ids,
    attention_mask=x_train_encoding.attention_mask,
    labels=y_train_encoding.input_ids
)
test_dataset = CustomDataset(
    input_ids=x_test_encoding.input_ids,
    attention_mask=x_test_encoding.attention_mask,
    labels=y_test_encoding.input_ids
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=Metrics.compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
