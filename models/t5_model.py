import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from utils.models_utils import CustomDataset, Metrics

DATASET = "/Users/diegolopes/Repositories/ms-usp-text-to-sql/data/geo_dataset.csv"
DEVICE = "mps"

df = pd.read_csv(DATASET)
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df['question'], df['query'], test_size=0.2)

max_source_length = len(max(list(X_train)))
max_target_length = len(max(list(X_test)))

tokenizer_finetuned = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model_finetuned = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL").to(DEVICE)

x_train_encoding = tokenizer_finetuned(
    [f"<s>translate English to SQL: {sentence}</s>" for sentence in X_train],
    padding=True,
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)
y_train_encoding = tokenizer_finetuned(
    list(y_train),
    padding=True,
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)

x_test_encoding = tokenizer_finetuned(
    [f"<s>translate English to SQL: {sentence}</s>" for sentence in X_test],
    padding=True,
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)
y_test_encoding = tokenizer_finetuned(
    list(y_test),
    padding=True,
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)

train_input_ids, train_attention_mask = x_train_encoding.input_ids, x_train_encoding.attention_mask
train_labels = y_train_encoding.input_ids

test_input_ids, test_attention_mask = x_test_encoding.input_ids, x_test_encoding.attention_mask
test_labels = y_test_encoding.input_ids

train_dataset = CustomDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = CustomDataset(test_input_ids, test_attention_mask, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=1,
    weight_decay=0.01,
)

metrics = Metrics
test_sentence = "<s>translate English to SQL: How many cities are there in Sao Paulo state?</s>"
features = tokenizer_finetuned(test_sentence, return_tensors="pt").to(DEVICE)
output = model_finetuned.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])

query_1 = tokenizer_finetuned.decode(output[0])

print(f"Antes do Treinamento\n{query_1}")

trainer = Trainer(
    model=model_finetuned,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

features = tokenizer_finetuned(test_sentence, return_tensors="pt").to(DEVICE)
output = model_finetuned.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])

query_2 = tokenizer_finetuned.decode(output[0])

print(f"Após o Treinamento\n{query_2}")
