import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
# 修正1: 保持使用 torch.optim 的 AdamW
from torch.optim import AdamW 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import re
import numpy as np

# ==========================================
# 0. 配置路径和参数
# ==========================================
TRAIN_FILE_1 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_1.csv"
TRAIN_FILE_2 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_2.csv"
DEV_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\dev.csv"
TEST_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\test.csv"

# 超参数配置
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 1. 数据预处理 & 准备
# ==========================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class AmazonDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['reviewText'].values 
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_process_data(file_paths):
    dfs = []
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for path in file_paths:
        try:
            # 修正重点：header=None 表示文件没有表头
            df = pd.read_csv(path, header=None)
            
            # 根据列数自动判断格式并命名列名
            if len(df.columns) == 3:
                # 典型的Amazon格式: [Label, Title, Review]
                df.columns = ['label', 'title', 'reviewText']
            elif len(df.columns) == 2:
                # 可能是: [Label, Review]
                df.columns = ['label', 'reviewText']
            elif len(df.columns) >= 4:
                 # 可能是带ID的数据，尝试取最后两列或特定列，这里假设前两列有效
                 # 如果运行还有错，需要具体看数据，但针对你报错的文件，上面3列的情况应该能命中
                 df.columns = ['label', 'reviewText'] + [f'col_{i}' for i in range(len(df.columns)-2)]
            
            # 打印信息确认
            print(f"成功读取 {path}，列名已设置为: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None

    full_df = pd.concat(dfs, ignore_index=True)

    # 安全检查：如果有非数字的label（比如误读了文件头），将其过滤
    # 将label列转换为数字，无法转换的变为NaN
    full_df['label'] = pd.to_numeric(full_df['label'], errors='coerce')
    full_df = full_df.dropna(subset=['label', 'reviewText']) # 删除标签或文本为空的行

    full_df['reviewText'] = full_df['reviewText'].apply(clean_text)

    # 标签映射：1->0 (负面), 2->1 (正面)
    full_df['label'] = full_df['label'].astype(int)
    unique_labels = full_df['label'].unique()
    
    if set(unique_labels) == {1, 2}:
        full_df['label'] = full_df['label'].map({1: 0, 2: 1})
        print("标签处理：已将 1/2 映射为 0/1")
    elif set(unique_labels).issubset({0, 1}):
        print("标签处理：检测到标签已为 0/1，跳过映射")
    else:
        print(f"警告：检测到未知的标签类别 {unique_labels}，尝试默认映射 (1->0, 2->1)")
        full_df['label'] = full_df['label'].map({1: 0, 2: 1})
        full_df = full_df.dropna(subset=['label']) # 再次清洗可能映射失败的

    full_df['label'] = full_df['label'].astype(int)
    
    return full_df

# ==========================================
# 准备数据加载器
# ==========================================
print("Loading Data...")
train_df = load_and_process_data([TRAIN_FILE_1, TRAIN_FILE_2])
dev_df = load_and_process_data(DEV_FILE)
test_df = load_and_process_data(TEST_FILE)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_dataset = AmazonDataset(train_df, tokenizer, MAX_LEN)
dev_dataset = AmazonDataset(dev_df, tokenizer, MAX_LEN)
test_dataset = AmazonDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==========================================
# 2. 模型构建
# ==========================================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

# ==========================================
# 3. 模型训练
# ==========================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) 

def train_epoch(model, data_loader, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# ==========================================
# 4. 评估函数
# ==========================================
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    conf_matrix = confusion_matrix(all_targets, all_preds)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": conf_matrix,
        "loss": np.mean(losses)
    }

# ==========================================
# 主循环
# ==========================================
print("Starting Training...")

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        DEVICE,
        len(train_df)
    )

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    reports = eval_model(
        model,
        dev_loader,
        DEVICE,
        len(dev_df)
    )
    
    print(f"Val   loss {reports['loss']:.4f}")
    print(f"Val   Accuracy: {reports['accuracy']:.4f}")
    print(f"Val   F1 Score: {reports['f1_score']:.4f}")
    
print("Training Complete. Testing on TEST set...")

test_reports = eval_model(
    model,
    test_loader,
    DEVICE,
    len(test_df)
)

print("-" * 30)
print("FINAL TEST RESULTS (实验指标):")
print("-" * 30)
print(f"准确率 (Accuracy): {test_reports['accuracy']:.4f}")
print(f"F1-score:         {test_reports['f1_score']:.4f}")
print(f"AUC-ROC:          {test_reports['auc_roc']:.4f}")
print("混淆矩阵 (Confusion Matrix):")
print(test_reports['confusion_matrix'])
print("-" * 30)