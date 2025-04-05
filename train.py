import os
import pandas as pd
import chardet
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import csv
from datetime import datetime
from transformers import TrainerCallback
import torch
from typing import List, Dict


class TokenizerTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.texts = []
        self.labels = []

    def load_data(self):
        """数据加载与预处理"""
        # 文件存在性检查
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"文件 {self.csv_path} 不存在")

        # 编码检测与读取
        with open(self.csv_path, 'rb') as f:
            rawdata = f.read(10000)
            detected = chardet.detect(rawdata)

        # 尝试多种编码方式
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
                print(f"成功使用 {encoding} 编码读取数据")
                break
            except (UnicodeDecodeError, LookupError):
                continue

        # 数据验证
        if 'abstract' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV必须包含 'abstract' 和 'label' 列")

        # 取前1000条数据（同时处理文本和标签）
        #df = df.head(1000)

        # 数据清洗
        clean_texts = []
        clean_labels = []
        
        # 同步处理文本和标签
        for text, label in zip(df['abstract'], df['label']):
            if pd.notnull(text):
                clean_text = str(text).encode('utf-8', 'ignore').decode('utf-8').strip()
                if clean_text:  # 确保清理后的文本不为空
                    clean_texts.append(clean_text)
                    clean_labels.append(str(label).lower())

        # 将清理后的数据赋值给实例变量
        self.texts = clean_texts
        
        # 将字符串标签转换为数值
        label_map = {'battery': 0, 'non-battery': 1}  # 定义标签映射
        self.labels = [label_map.get(label, 0) for label in clean_labels]

        if len(self.texts) == 0:
            raise ValueError("清洗后无有效文本数据")

        print(f"处理后的数据集大小：{len(self.texts)} 条")
        return self

    def train_tokenizer(self, output_dir):
        """训练并保存分词器"""
        if not self.texts:
            raise RuntimeError("请先调用 load_data() 加载数据")

        # 准备训练文本
        os.makedirs(output_dir, exist_ok=True)
        train_text_path = os.path.join(output_dir, "train_text.txt")
        with open(train_text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.texts))

        # 初始化BPE分词器
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        # 训练配置
        trainer = trainers.BpeTrainer(
            vocab_size=300000,
            min_frequency=1,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # 执行训练
        tokenizer.train(files=[train_text_path], trainer=trainer)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

        print(f"分词器已保存至 {output_dir}")
        return self


class ModelTrainer:
    def __init__(self, tokenizer_dir):
        # 修改加载分词器的方式
        try:
            # 首先尝试直接加载
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        except Exception as e:
            print(f"直接加载分词器失败，尝试重新训练: {e}")
            # 如果加载失败，重新训练分词器
            print("训练分词器中...")
            token_trainer = TokenizerTrainer(DATA_PATH)
            token_trainer.load_data()
            token_trainer.train_tokenizer(tokenizer_dir)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

        # 添加特殊token
        self.tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'eos_token': '[SEP]',
            'bos_token': '[CLS]'
        })
        
        # 初始化日志系统
        self.log_path = "C:\\Users\\Shawn\\Desktop\\BatteryGPT2\\dataandlog\\training_log_3.csv"
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件头"""
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'epoch',
                'step',
                'train_loss',
                'eval_loss',
                'learning_rate'
            ])

    def _log_metrics(self, metrics, step):
        """记录训练指标"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                metrics.get('epoch', 0),
                step,
                metrics.get('loss', ''),
                metrics.get('eval_loss', ''),
                metrics.get('learning_rate', '')
            ])

    def preprocess_function(self, examples):
        """
        对输入数据进行预处理，包括分词、填充、截断和标签处理
        :param examples: 包含文本和标签的字典
        :return: 预处理后的字典，包含input_ids, attention_mask, labels
        """
        tokenizer = self.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        # 确保输入文本不为空
        texts = [str(text) if text is not None else "" for text in examples['text']]
        
        # 添加特殊标记
        texts = [f"[CLS] {text} [SEP]" for text in texts]
        
        # 使用tokenizer处理文本
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # 确保labels已经是数值类型
        labels = [int(label) for label in examples['labels']]  # 确保标签是整数
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

    def create_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """
        将文本和标签数据整理成适合模型训练的Dataset格式
        :param texts: 文本列表
        :param labels: 标签列表
        :return: 自定义的Dataset
        """
        dataset = [{"text": text, "label": label} for text, label in zip(texts, labels)]
        encoded_dataset = self.preprocess_function({"text": [item["text"] for item in dataset], "label": [item["label"] for item in dataset]})
        return Dataset.from_dict({
            "input_ids": encoded_dataset["input_ids"],
            "attention_mask": encoded_dataset["attention_mask"],
            "labels": encoded_dataset["labels"]
        })

    # 添加自定义回调类
    class LogCallback(TrainerCallback):
        def __init__(self, log_path):
            self.log_path = log_path
            self.current_logs = {}  # 用于临时存储当前step的日志

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            
            step = state.global_step
            if step not in self.current_logs:
                self.current_logs[step] = {}
            
            # 更新当前step的日志
            self.current_logs[step].update({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'epoch': logs.get('epoch', ''),
                'step': step,
                'train_loss': logs.get('loss', ''),
                'eval_loss': logs.get('eval_loss', ''),
                'learning_rate': logs.get('learning_rate', '')
            })
            
            # 如果同时有训练损失和验证损失，或者是纯训练损失，就写入文件
            current = self.current_logs[step]
            if ('train_loss' in current and current['train_loss']) or \
               ('eval_loss' in current and current['eval_loss']):
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
                # 清除已写入的日志
                del self.current_logs[step]

    def train(self, texts, labels, output_dir):
        # 确保数据不为空
        if not texts or not labels:
            raise ValueError("训练数据为空")
        
        # 创建数据集
        dataset_dict = {
            'text': texts
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # 数据预处理函数
        def preprocess_function(examples):
            # 确保输入文本不为空
            texts = [str(text) if text is not None else "" for text in examples['text']]
            
            # 使用tokenizer处理文本
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors=None
            )
            
            # 对于问答系统，我们使用同样的文本作为标签
            encodings['labels'] = encodings['input_ids'].copy()
            
            return encodings
        
        # 应用预处理
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="处理数据集"
        )
        
        # 在预处理后统计总token数
        total_tokens = sum(
            len(input_ids) 
            for input_ids in processed_dataset['input_ids']
        )
        print(f"数据集的Token总数: {total_tokens}")

        # 分割数据集
        train_test_split = processed_dataset.train_test_split(test_size=0.1)
        
        # 设置训练参数
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
        
        # 初始化语言模型
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(self.tokenizer))
        
        # 定义数据整理函数
        def data_collator(features):
            batch = {}
            
            # 将列表转换为张量
            batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
            batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            
            return batch
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test_split['train'],
            eval_dataset=train_test_split['test'],
            data_collator=data_collator,
            callbacks=[ModelTrainer.LogCallback(self.log_path)]  # 添加回调
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model')
        if os.path.exists(final_model_path):
            import shutil
            shutil.rmtree(final_model_path)  # 删除已存在的模型目录
        trainer.save_model(final_model_path)


if __name__ == "__main__":
    try:
        # 第一步：训练分词器
        TOKENIZER_DIR = "C:/Users/Shawn/Desktop/BatteryGPT2/custom_tokenizer"
        DATA_PATH = "C:/Users/Shawn/Desktop/BatteryGPT2/data/training_data.csv"

        print("正在加载数据...")
        token_trainer = TokenizerTrainer(DATA_PATH).load_data()

        print("训练分词器中...")
        token_trainer.train_tokenizer(TOKENIZER_DIR)

        # 第二步：训练模型
        MODEL_DIR = "C:/Users/Shawn/Desktop/BatteryGPT2/model_output_3"

        print("\n初始化模型训练器...")
        model_trainer = ModelTrainer(TOKENIZER_DIR)

        print("开始模型训练...")
        model_trainer.train(token_trainer.texts, token_trainer.labels, MODEL_DIR)

        print(f"\n训练完成！日志文件：{model_trainer.log_path}")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")