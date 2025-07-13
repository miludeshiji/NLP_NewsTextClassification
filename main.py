import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# 从其他文件中导入核心函数
from feature import preprocess_data
from model import TextCNN, train_model, predict

def run():
    # --- 1. 定义路径和超参数 ---
    # 路径
    DATA_DIR = './text_data'
    PROCESSED_DIR = './processed_data'
    MODEL_DIR = './model'
    RESULT_DIR = './result'
    MODEL_PATH = os.path.join(MODEL_DIR, 'textcnn_model.pth')
    SUBMISSION_PATH = os.path.join(RESULT_DIR, 'submission.csv')
    
    # 确保输出目录存在
    for dir_path in [MODEL_DIR, RESULT_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 模型超参数 (VOCAB_SIZE 将被动态设置)
    EMBED_DIM = 128
    NUM_CLASSES = 14
    NUM_CHANNELS = 128
    KERNEL_SIZES = [3, 4, 5]
    MAX_LEN = 1024

    # 训练超参数
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # --- 2. 数据准备 ---
    print("步骤 1/4: 开始处理数据...")
    # 现在会接收第四个返回值：vocab_size
    train_data, labels, test_data, vocab_size = preprocess_data(DATA_DIR, PROCESSED_DIR, max_len=MAX_LEN)
    
    # 使用动态获取的 vocab_size
    VOCAB_SIZE = vocab_size

    # 将数据转换为Tensor
    train_tensor = torch.LongTensor(train_data)
    labels_tensor = torch.LongTensor(labels)
    test_tensor = torch.LongTensor(test_data)

    # 创建DataLoader
    train_dataset = TensorDataset(train_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 3. 模型训练 ---
    print("\n步骤 2/4: 初始化模型...")
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用动态计算的 VOCAB_SIZE 初始化模型
    model = TextCNN(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_classes=NUM_CLASSES, 
        num_channels=NUM_CHANNELS,
        kernel_sizes=KERNEL_SIZES
    )

    # 检查是否已有训练好的模型，若无则进行训练
    if not os.path.exists(MODEL_PATH):
        print(f"未找到已训练的模型，开始新的训练...")
        train_model(model, train_loader, EPOCHS, LEARNING_RATE, device, MODEL_PATH)
    else:
        print(f"找到已训练的模型: {MODEL_PATH}，将跳过训练。")

    # --- 4. 模型预测 ---
    print("\n步骤 3/4: 加载模型并进行预测...")
    # 加载已保存的模型权重
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    predictions = predict(model, test_loader, device)

    # --- 5. 生成提交文件 ---
    print("\n步骤 4/4: 生成提交文件...")
    # 读取提交样例文件以保持格式一致
    sample_submit_df = pd.read_csv(os.path.join(DATA_DIR, 'test_a_sample_submit.csv'))
    sample_submit_df['label'] = predictions
    sample_submit_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"任务完成！预测结果已保存至: {SUBMISSION_PATH}")


if __name__ == '__main__':
    run()