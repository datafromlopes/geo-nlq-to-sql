from transformers import T5Tokenizer, T5ForConditionalGeneration  # T5 Model
from transformers import LlamaTokenizer, LlamaForCausalLM         # LLAMA Model
from transformers import AutoModelForCausalLM                     # GPT2 Model
from transformers import BertModel, BertTokenizer                 # BERT Model
from transformers import AutoTokenizer                            # Tokenizer for GPT2 and BERT Models
from transformers import TrainingArguments
from transformers import Trainer
from utils.utils import Metrics
import torch


class TextToSql:
    def __init__(self, model_name, device="mps", num_labels=1):
        match model_name:
            case 't5-base':
                model = f"google-t5/{model_name}"
                self.__tokenizer = T5Tokenizer.from_pretrained(model)
                self.__model = T5ForConditionalGeneration.from_pretrained(model, num_labels=num_labels).to(device)
            case 'bert-base-cased':
                model = f"google-bert/{model_name}"
                self.__tokenizer = BertTokenizer.from_pretrained(model)
                self.__model = BertModel.from_pretrained(model, num_labels=num_labels).to(device)
            case 'bert-base-portuguese-cased':
                model_path = f"neuralmind/{model_name}"
                self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.__model = BertModel.from_pretrained(model_path)
            case 'Llama-2-7b-hf':
                model_path = f"meta-llama/{model_name}"
                self.__tokenizer = LlamaTokenizer.from_pretrained(model_path)
                self.__model = LlamaForCausalLM.from_pretrained(model_path)
            case 'gpt2':
                self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.__model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize(self, texts, return_tensors="pt"):
        encodings = self.__tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            return_tensors=return_tensors,
            add_special_tokens=True
        )
        return encodings

    def trainer(self, train_dataset, test_dataset):
        metrics = Metrics()
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            save_total_limit=10,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=self.__model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=metrics.compute_metrics,
        )

        return trainer
