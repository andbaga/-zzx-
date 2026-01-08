import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
import re
from tqdm import tqdm
import os
import itertools

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # 路径设置 (请确认路径无误)
    TRAIN_FILE_1 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_1.csv"
    TRAIN_FILE_2 = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（一）\Amazon reviews\train_part_2.csv"
    DEV_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\dev.csv"
    TEST_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（二）\dataset\test.csv"
    
    SAVE_PATH = "best_model.pth"  # 最优模型保存路径
    
    # 模型参数
    EMBED_DIM = 128         
    FILTER_SIZES = [3, 4, 5] 
    NUM_FILTERS = 100       
    DROPOUT = 0.5           
    NUM_CLASSES = 2         
    
    # 训练参数
    BATCH_SIZE = 128        
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5          
    MAX_VOCAB_SIZE = 50000  
    MAX_SEQ_LEN = 200       
    
    DEBUG = False 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {Config.DEVICE}")

# ==========================================
# 2. 数据预处理
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    text = re.sub(r"\s+", " ", text)          
    return text.lower().strip()

def build_vocab(texts, max_size):
    print("Building Vocabulary...")
    word_freq = {}
    for text in tqdm(texts, desc="Tokenizing"):
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_size-2]:
        vocab[word] = len(vocab)
    print(f"Vocab size: {len(vocab)}")
    return vocab

class AmazonDataset(Dataset):
    def __init__(self, data_paths, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len
        dfs = []
        paths = data_paths if isinstance(data_paths, list) else [data_paths]
        
        for path in paths:
            if not os.path.exists(path):
                continue
            print(f"Loading {path}...")
            df = pd.read_csv(path, header=None, names=['label', 'title', 'content'])
            if Config.DEBUG: df = df.head(1000) 
            dfs.append(df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        # 标签修正：确保是 0 和 1
        if full_df['label'].min() == 1:
            full_df['label'] = full_df['label'] - 1
            
        self.labels = full_df['label'].values
        self.texts = (full_df['title'].fillna("") + " " + full_df['content'].fillna("")).apply(clean_text).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. TextCNN 模型
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        xm = [F.max_pool1d(F.relu(conv(x)).squeeze(3), F.relu(conv(x)).squeeze(3).size(2)).squeeze(2) for conv in self.convs]
        x = self.dropout(torch.cat(xm, 1))
        return self.fc(x)

# ==========================================
# 4. 训练与评估 (含最优模型保存)
# ==========================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5 
    return avg_loss, acc, auc, f1, all_labels, all_preds

def train_model():
    print("Preparing Data...")
    # 快速构建词表
    df_sample = pd.read_csv(Config.TRAIN_FILE_1, header=None, names=['l','t','c'])
    if Config.DEBUG: df_sample = df_sample.head(1000)
    else: df_sample = df_sample.head(100000)
    vocab = build_vocab((df_sample['t'].fillna("")+" "+df_sample['c'].fillna("")).apply(clean_text).values, Config.MAX_VOCAB_SIZE)
    
    train_ds = AmazonDataset([Config.TRAIN_FILE_1, Config.TRAIN_FILE_2], vocab, Config.MAX_SEQ_LEN)
    dev_ds = AmazonDataset(Config.DEV_FILE, vocab, Config.MAX_SEQ_LEN)
    test_ds = AmazonDataset(Config.TEST_FILE, vocab, Config.MAX_SEQ_LEN)
    
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = TextCNN(len(vocab), Config.EMBED_DIM, Config.NUM_CLASSES, Config.FILTER_SIZES, Config.NUM_FILTERS, Config.DROPOUT).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []}
    
    # ----------------------------------------
    # 最优模型保存机制
    # ----------------------------------------
    best_val_acc = 0.0
    
    print("Start Training...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        for texts, labels in loop:
            texts, labels = texts.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_dl)
        val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(model, dev_dl, criterion)
        
        print(f"\nSummary Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        
        # 核心修改：保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.SAVE_PATH)
            print(f"  >>> New Best Model Saved! (Acc: {best_val_acc:.4f})")
        
        print("-" * 50)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)

    # ----------------------------------------
    # 加载最优模型进行测试
    # ----------------------------------------
    print(f"\nTraining Finished. Loading Best Model from {Config.SAVE_PATH}...")
    model.load_state_dict(torch.load(Config.SAVE_PATH))
    
    print("Evaluating on Test Set with Best Model...")
    test_loss, test_acc, test_auc, test_f1, y_true, y_pred = evaluate(model, test_dl, criterion)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test AUC:      {test_auc:.4f}")
    print(f"Final Test F1-Score: {test_f1:.4f}")
    
    return history, y_true, y_pred

# ==========================================
# 5. 可视化
# ==========================================
def plot_results(history, y_true, y_pred):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    
    # Acc & AUC
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_acc'], 'b-', label='Val Acc')
    plt.plot(epochs, history['val_auc'], 'g-', label='Val AUC')
    plt.title('Validation Acc & AUC'); plt.xlabel('Epochs'); plt.legend()

    # F1
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_f1'], 'm-', label='Val F1')
    plt.title('Validation F1 Score'); plt.xlabel('Epochs'); plt.ylabel('F1'); plt.legend()
    
    # Confusion Matrix Heatmap
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    plt.colorbar()
    classes = ['Negative(0)', 'Positive(1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45); plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        history, y_true, y_pred = train_model()
        plot_results(history, y_true, y_pred)
        print("\nAll tasks completed successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()