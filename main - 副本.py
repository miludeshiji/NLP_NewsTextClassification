import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import List, Tuple, Optional, runtime_checkable  # 在文件开头添加

# 数据加载
train_df = pd.read_csv('train_set.csv', sep='\t')
test_df = pd.read_csv('test_a.csv', sep='\t')

# 预处理结果保存路径
PROCESSED_DATA_PATH = 'processed_data.pkl'
VOCAB_PATH = 'vocab.pkl'

# 尝试加载预处理结果
try:
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        X, y, X_test = pickle.load(f)
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print('成功加载预处理结果')
except FileNotFoundError:
    # 数据预处理
train_df['text'] = train_df['text'].apply(lambda x: list(map(int, x.split())))
test_df['text'] = test_df['text'].apply(lambda x: list(map(int, x.split())))

# 构建词汇表
all_tokens = []
for text in train_df['text']:
    all_tokens.extend(text)

# 重构词汇表逻辑
unique_tokens = sorted(set(all_tokens))
vocab = {token: idx+1 for idx, token in enumerate(unique_tokens)}
vocab_size = max(vocab.values()) + 1  # 正确包含0~max_index

# 修正模型初始化
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

# 完善设备同步
texts, labels = texts.to(device, non_blocking=True), labels.to(device, non_blocking=True)

# 最终边界验证
assert all(0 <= idx < vocab_size for seq in train_df['text'] for idx in seq), "存在非法索引值"

# 在模型类内初始化嵌入层
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

# 添加完整设备同步
model = model.to(device)
data_loader = DataLoader(..., pin_memory=True)

# 强化边界检查
train_df['text'] = train_df['text'].apply(
    lambda x: [min(vocab.get(t,0), vocab_size-1) for t in x]
)

# 添加预处理验证
assert all(0 <= t < vocab_size for text in train_df['text'] for t in text)

# 序列填充和截断
max_len = 200
train_df['text'] = train_df['text'].apply(lambda x: x[:max_len] + [0]*(max_len - len(x)) if len(x) < max_len else x[:max_len])
test_df['text'] = test_df['text'].apply(lambda x: x[:max_len] + [0]*(max_len - len(x)) if len(x) < max_len else x[:max_len])

# 转换为张量
X = torch.tensor(train_df['text'].tolist(), dtype=torch.long)
y = torch.tensor(train_df['label'].tolist(), dtype=torch.long)
X_test = torch.tensor(test_df['text'].tolist(), dtype=torch.long)

# 保存预处理结果
with open(PROCESSED_DATA_PATH, 'wb') as f:
    pickle.dump((X, y, X_test), f)
with open(VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab, f)
print(f'预处理结果已保存至{PROCESSED_DATA_PATH}和{VOCAB_PATH}')

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据集类

# 更新函数定义
class NewsDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx]
    def __len__(self):
        return len(self.texts)

train_dataset = NewsDataset(X_train, y_train)
val_dataset = NewsDataset(X_val, y_val)
test_dataset = NewsDataset(X_test)

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 模型定义
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.squeeze(0))
        return out

# 建议代码结构
# === 超参数配置 ===
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64

# === 数据处理函数 ===

num_classes = 14
epochs = 10
lr = 0.001

# 初始化模型、损失函数和优化器
model = TextClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
device = torch.device('cuda')
# 添加索引验证
print(f'Max token index: {max(vocab.values())}, Vocab size: {vocab_size}')
assert max(vocab.values()) < vocab_size, "Token index exceeds embedding dimensions"

# 统一设备管理
model = model.to(device)
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)

# 迁移设备转移至训练循环内部
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    
    val_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val F1: {val_f1}')

# 预测测试集
model.eval()
predictions = []
with torch.no_grad():
    for texts in test_loader:
        texts = texts.to(device)
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)

# 生成提交文件
submission = pd.DataFrame({'label': predictions})
submission.to_csv('test_a_sample_submit.csv', index=False)

# 添加调试信息
print('词汇表样本:', dict(list(vocab.items())[:5]))
print('输入数据样本:', train_df['text'].iloc[0][:10])

# 验证嵌入层尺寸
assert self.embedding.num_embeddings == vocab_size, \
    f"嵌入层尺寸{self.embedding.num_embeddings}与词汇表大小{vocab_size}不匹配"

# 可视化数据分布
import matplotlib.pyplot as plt
token_indices = [t for text in train_df['text'] for t in text]
plt.hist(token_indices, bins=50)
plt.savefig('token_distribution.png')