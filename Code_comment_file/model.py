import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. 定义TextCNN模型 ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 创建多个不同尺寸卷积核的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # -> [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)    # -> [batch_size, 1, seq_len, embed_dim] (增加一个通道维度)
        
        # 将x通过每个卷积层，然后进行池化
        conved = [F.relu(conv(x)) for conv in self.convs] 
        # conved[i]: [batch_size, num_channels, seq_len - kernel_sizes[i] + 1, 1]

        pooled = [F.max_pool1d(conv.squeeze(3), conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[i]: [batch_size, num_channels]
        
        # 将所有池化结果拼接起来
        cat = torch.cat(pooled, dim=1) # -> [batch_size, num_channels * len(kernel_sizes)]
        
        # 应用dropout并送入全连接层
        cat = self.dropout(cat)
        output = self.fc(cat) # -> [batch_size, num_classes]
        return output

# --- 2. 训练函数 ---
def train_model(model, train_loader, epochs, learning_rate, device, model_save_path):
    """
    训练模型并保存

    :param model: 要训练的PyTorch模型
    :param train_loader: 训练数据的DataLoader
    :param epochs: 训练轮次
    :param learning_rate: 学习率
    :param device: 'cuda' 或 'cpu'
    :param model_save_path: 模型保存路径
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device) # 将模型移动到指定设备
    model.train() # 设置为训练模式

    print(f"开始在 {device}上进行训练...")
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}")

    # 保存模型
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")

# --- 3. 预测函数 ---
def predict(model, test_loader, device):
    """
    使用训练好的模型进行预测

    :param model: 已加载的PyTorch模型
    :param test_loader: 测试数据的DataLoader
    :param device: 'cuda' 或 'cpu'
    :return: 预测标签列表
    """
    model.to(device)
    model.eval() # 设置为评估模式
    
    all_preds = []
    print(f"开始在 {device}上进行预测...")
    with torch.no_grad(): # 在评估时不需要计算梯度
        for inputs in tqdm(test_loader):
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            # 获取概率最高的类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    print("预测完成。")
    return all_preds