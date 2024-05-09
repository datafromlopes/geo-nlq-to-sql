from models.text_models import TextToSql
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils.utils import MakeTorchData


DATASET = "/Users/diegolopes/Repositories/ms-usp-text-to-sql/data/spider_dataset.csv"
MODEL_BASE = "bert-base-cased"
DEVICE = "cpu"

df = pd.read_csv(DATASET)
df.dropna(inplace=True)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['query'])

X_train, X_test, y_train, y_test = train_test_split(df['question'], encoded_labels, test_size=0.4, random_state=42)

text_to_sql = TextToSql(model_name=MODEL_BASE, device=DEVICE)

train_tokenized = text_to_sql.tokenize(texts=X_train)
test_tokenized = text_to_sql.tokenize(texts=X_test)

train_dataset = MakeTorchData(train_tokenized, y_train.ravel())
valid_dataset = MakeTorchData(test_tokenized, y_test.ravel())

trainer = text_to_sql.trainer(train_dataset=train_dataset, test_dataset=valid_dataset)

trainer.train()
trainer.evaluate()
