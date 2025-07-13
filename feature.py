import pandas as pd
import os
import pickle
from tqdm import tqdm

def preprocess_data(data_dir, processed_dir, max_len=1000):
    processed_file_path = os.path.join(processed_dir, 'processed_data.pkl')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if os.path.exists(processed_file_path):
        print("发现已处理的数据，直接加载...")
        with open(processed_file_path, 'rb') as f:
            data = pickle.load(f)
        print("数据加载完毕。")
        return data['train_padded'], data['labels'], data['test_padded'], data['vocab_size']
    print("未发现已处理的数据，开始进行预处理...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_set.csv'), sep='\t')
    test_df = pd.read_csv(os.path.join(data_dir, 'test_a.csv'), sep='\t')
    max_vocab_idx = 0
    def text_to_int_list(texts):
        nonlocal max_vocab_idx
        int_lists = []
        for text in tqdm(texts):
            int_list = list(map(int, str(text).split()))
            if int_list:
                current_max = max(int_list)
                if current_max > max_vocab_idx:
                    max_vocab_idx = current_max
            int_lists.append(int_list)
        return int_lists
    print("正在转换训练集文本并寻找最大词汇ID...")
    train_texts = text_to_int_list(train_df['text'])
    print("正在转换测试集文本并寻找最大词汇ID...")
    test_texts = text_to_int_list(test_df['text'])
    vocab_size = max_vocab_idx + 1
    print(f"检测到最大词汇ID为: {max_vocab_idx}，因此设置词典大小为: {vocab_size}")

    labels = train_df['label'].tolist()
    def pad_sequences(sequences, maxlen, padding_value=0):
        padded_sequences = []
        for seq in tqdm(sequences):
            if len(seq) > maxlen:
                padded_seq = seq[:maxlen]
            else:
                padded_seq = seq + [padding_value] * (maxlen - len(seq))
            padded_sequences.append(padded_seq)
        return padded_sequences
    print(f"正在将所有序列填充或截断至长度 {max_len}...")
    train_padded = pad_sequences(train_texts, max_len)
    test_padded = pad_sequences(test_texts, max_len)
    data_to_save = {
        'train_padded': train_padded,
        'labels': labels,
        'test_padded': test_padded,
        'vocab_size': vocab_size
    }
    print(f"正在将处理后的数据保存至 '{processed_file_path}'...")
    with open(processed_file_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print("数据预处理和保存完成。")
    return train_padded, labels, test_padded, vocab_size