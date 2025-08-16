import os
import argparse
import pandas as pd
import chardet
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import csv
from datetime import datetime
from transformers import TrainerCallback
import torch
from typing import List


class TokenizerTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.texts = []
        self.labels = []

    def load_data(self):
        """Load and preprocess data from CSV"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File {self.csv_path} does not exist")

        # Detect encoding
        with open(self.csv_path, 'rb') as f:
            rawdata = f.read(10000)
            detected = chardet.detect(rawdata)

        encodings_to_try = [
            detected['encoding'],
            'utf-8',
            'gb18030',
            'iso-8859-1'
        ]

        df = None
        for encoding in set(encodings_to_try):
            try:
                df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"Successfully read with {encoding} encoding")
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if 'abstract' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'abstract' and 'label' columns")

        clean_texts, clean_labels = [], []
        for text, label in zip(df['abstract'], df['label']):
            if pd.notnull(text):
                clean_text = str(text).encode('utf-8', 'ignore').decode('utf-8').strip()
                if clean_text:
                    clean_texts.append(clean_text)
                    clean_labels.append(str(label).lower())

        self.texts = clean_texts
        label_map = {'battery': 0, 'non-battery': 1}
        self.labels = [label_map.get(label, 0) for label in clean_labels]

        if len(self.texts) == 0:
            raise ValueError("No valid text data after cleaning")

        print(f"Dataset size after cleaning: {len(self.texts)} samples")
        return self

    def train_tokenizer(self, output_dir):
        """Train and save custom tokenizer"""
        if not self.texts:
            raise RuntimeError("Please call load_data() first")

        os.makedirs(output_dir, exist_ok=True)
        train_text_path = os.path.join(output_dir, "train_text.txt")
        with open(train_text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.texts))

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=300000,
            min_frequency=1,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        tokenizer.train(files=[train_text_path], trainer=trainer)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

        print(f"Tokenizer saved to {output_dir}")
        return self


class ModelTrainer:
    def __init__(self, tokenizer_dir, log_path, data_path):
        try:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        except Exception as e:
            print(f"Direct tokenizer load failed, retraining: {e}")
            token_trainer = TokenizerTrainer(data_path)
            token_trainer.load_data().train_tokenizer(tokenizer_dir)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

        self.tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'eos_token': '[SEP]',
            'bos_token': '[CLS]'
        })

        self.log_path = log_path
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'train_loss', 'eval_loss', 'learning_rate'])

    class LogCallback(TrainerCallback):
        def __init__(self, log_path):
            self.log_path = log_path
            self.current_logs = {}

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            step = state.global_step
            if step not in self.current_logs:
                self.current_logs[step] = {}

            self.current_logs[step].update({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'epoch': logs.get('epoch', ''),
                'step': step,
                'train_loss': logs.get('loss', ''),
                'eval_loss': logs.get('eval_loss', ''),
                'learning_rate': logs.get('learning_rate', '')
            })

            current = self.current_logs[step]
            if ('train_loss' in current and current['train_loss']) or ('eval_loss' in current and current['eval_loss']):
                with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current['timestamp'],
                        current['epoch'],
                        current['step'],
                        current['train_loss'],
                        current.get('eval_loss', ''),
                        current['learning_rate']
                    ])
                del self.current_logs[step]

    def train(self, texts, labels, output_dir):
        if not texts or not labels:
            raise ValueError("Training data is empty")

        dataset = Dataset.from_dict({'text': texts})

        def preprocess_function(examples):
            texts = [str(text) if text is not None else "" for text in examples['text']]
            encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=512)
            encodings['labels'] = encodings['input_ids'].copy()
            return encodings

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset"
        )

        total_tokens = sum(len(input_ids) for input_ids in processed_dataset['input_ids'])
        print(f"Total tokens in dataset: {total_tokens}")

        train_test_split = processed_dataset.train_test_split(test_size=0.1)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="steps",
            eval_steps=200,
            logging_steps=100,
            save_steps=200,
            max_steps=2400,
            num_train_epochs=3,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=True if torch.cuda.is_available() else False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            remove_unused_columns=True,
            overwrite_output_dir=True,
            save_total_limit=1
        )

        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(self.tokenizer))

        def data_collator(features):
            return {
                "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
                "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long)
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test_split['train'],
            eval_dataset=train_test_split['test'],
            data_collator=data_collator,
            callbacks=[ModelTrainer.LogCallback(self.log_path)]
        )

        trainer.train()

        final_model_path = os.path.join(output_dir, 'final_model')
        if os.path.exists(final_model_path):
            import shutil
            shutil.rmtree(final_model_path)
        trainer.save_model(final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT2 with custom tokenizer")
    parser.add_argument("--data_path", required=True, help="Path to training CSV data")
    parser.add_argument("--tokenizer_dir", required=True, help="Directory to save tokenizer")
    parser.add_argument("--model_dir", required=True, help="Directory to save trained model")
    parser.add_argument("--log_path", required=True, help="Path to save training log CSV")
    args = parser.parse_args()

    try:
        print("Loading dataset...")
        token_trainer = TokenizerTrainer(args.data_path).load_data()

        print("Training tokenizer...")
        token_trainer.train_tokenizer(args.tokenizer_dir)

        print("\nInitializing model trainer...")
        model_trainer = ModelTrainer(args.tokenizer_dir, args.log_path, args.data_path)

        print("Starting model training...")
        model_trainer.train(token_trainer.texts, token_trainer.labels, args.model_dir)

        print(f"\nTraining finished! Log file: {model_trainer.log_path}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
