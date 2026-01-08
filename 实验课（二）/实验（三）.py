import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 禁止 Tokenizers 并行，防止死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# ==========================================
# 1. 文件路径配置
# ==========================================
TRAIN_FILE_1 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_1.csv"
TRAIN_FILE_2 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_2.csv"
DEV_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\dev.csv"
TEST_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\test.csv"

# 模型名称
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# ==========================================
# 2. 数据处理类
# ==========================================
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def load_data(file_paths, tokenizer, max_length=128, is_test=False):
    """
    读取无表头 CSV 文件
    约定：第1列(索引0)为标签，第3列(索引2)为文本
    """
    data_frames = []
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for path in file_paths:
        # header=None: 防止第一行数据被误认为是表头
        df = pd.read_csv(path, header=None)
        data_frames.append(df)
    
    df = pd.concat(data_frames, ignore_index=True)
    
    # 打印形状以便确认加载情况
    if not is_test:
        print(f"Loaded training/dev data shape: {df.shape}")
    else:
        print(f"Loaded test data shape: {df.shape}")

    # === 关键索引 ===
    # 0 = Label (1或2), 2 = Content (评论文本)
    label_idx = 0
    text_idx = 2

    texts = df[text_idx].astype(str).tolist()
    
    # Tokenization
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    labels = None
    if not is_test:
        # 标签转换: 原始 1(负)->0, 2(正)->1
        labels = df[label_idx].apply(lambda x: int(x) - 1).tolist()
    
    return SentimentDataset(encodings, labels)

def compute_metrics(pred):
    """
    计算评估指标：Accuracy, F1, Precision, Recall, AUC
    """
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    
    # 计算概率 (用于 AUC)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    prob_positive = probs[:, 1] # 取正类的概率
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    # 计算 AUC (防止测试集只有单一类别时报错)
    try:
        auc = roc_auc_score(labels, prob_positive)
    except:
        auc = 0.5
        
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Qwen 需要手动指定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    # 加载数据
    train_dataset = load_data([TRAIN_FILE_1, TRAIN_FILE_2], tokenizer)
    eval_dataset = load_data(DEV_FILE, tokenizer)
    test_dataset = load_data(TEST_FILE, tokenizer, is_test=False) 

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,   # 显存够大可调大，例如 8 或 16
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,                # 每50步打印一次日志
        
        # === 关键设置 ===
        eval_strategy="epoch",           # 每个epoch评估一次
        save_strategy="epoch",           # 每个epoch保存一次
        report_to="none",                # 禁用 TensorBoard 绘图，防止报错
        # =============
        
        load_best_model_at_end=True,
        learning_rate=2e-5,
        fp16=True,                       # 开启混合精度加速
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 在测试集上评估
    print("\nEvaluating on Test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # 保存最终模型
    trainer.save_model("./qwen_sentiment_model")
    print("Model saved to ./qwen_sentiment_model")

    # ==========================================
    # 新增：生成并打印混淆矩阵
    # ==========================================
    print("\n=== Generating Confusion Matrix ===")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[详细解读]")
    print(f"真负类 (True Negative): {tn} (预测正确：负面)")
    print(f"假正类 (False Positive): {fp} (预测错误：实际负面->预测正面)")
    print(f"假负类 (False Negative): {fn} (预测错误：实际正面->预测负面)")
    print(f"真正类 (True Positive): {tp} (预测正确：正面)")

if __name__ == "__main__":
    main()