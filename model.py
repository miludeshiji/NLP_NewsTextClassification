import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conved = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(conv.squeeze(3), conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        output = self.fc(cat)
        return output
def train_model(model, train_loader, epochs, learning_rate, device, model_save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    print(f"开始在 {device}上进行训练...")
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}")
    
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")
    
def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    print(f"开始在 {device}上进行预测...")
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
    print("预测完成。")
    return all_preds