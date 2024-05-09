from models.text_models import TextToSql
import pandas as pd

DATA_SOURCE_FOLDER = "/Users/diegolopes/Repositories/texttosql/data"

train_dataset = pd.read_csv(f"{DATA_SOURCE_FOLDER}/training_data.csv")
validation_dataset = pd.read_csv(f"{DATA_SOURCE_FOLDER}/validation_data.csv")
test_dataset = pd.read_csv(f"{DATA_SOURCE_FOLDER}/test_data.csv")

x_train = train_dataset['question'].values
y_train = train_dataset['query'].values

x_test = test_dataset['question'].values
y_test = test_dataset['query'].values

x_validation = validation_dataset['question'].values
y_validation = validation_dataset['query'].values

MAX_SEQUENCE_LENGTH = 500
MAX_FEATURES = len(x_train)
MODEL_BASE = "bert-base-cased"
DEVICE = "mps"

text_to_sql = TextToSql(model_name=MODEL_BASE, device=DEVICE)

encoder_train = text_to_sql.encode(text=x_train, max_length=MAX_SEQUENCE_LENGTH)
encoder_test = text_to_sql.encode(text=x_test, max_length=MAX_SEQUENCE_LENGTH)
encoder_validation = text_to_sql.encode(text=x_validation, max_length=MAX_SEQUENCE_LENGTH)

trainer = text_to_sql(train_dataset=encoder_train, valid_dataset=encoder_validation)

trainer.train()
trainer.evaluate()
