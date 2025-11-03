import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """WikiText-2数据集 - 使用Qwen Tokenizer"""
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 读取文本
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 清理文本
        lines = [line.strip() for line in text.split('\n') 
                if line.strip() and not line.startswith('=')]
        self.text = ' '.join(lines)
        
        print(f"Processing {file_path}...")
        # 使用Qwen tokenizer编码
        self.tokens = tokenizer.encode(self.text, add_special_tokens=False)
        
        # 只保留完整的序列
        total_sequences = len(self.tokens) // (self.max_seq_len + 1)
        self.tokens = self.tokens[:total_sequences * (self.max_seq_len + 1)]
        
        print(f"  → {len(self.tokens)} tokens, {total_sequences} sequences")
        
    def __len__(self):
        return len(self.tokens) // (self.max_seq_len + 1)
    
    def __getitem__(self, idx):
        # 获取序列
        start = idx * (self.max_seq_len + 1)
        end = start + self.max_seq_len + 1
        
        chunk = self.tokens[start:end]
        
        # 输入和目标
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        
        return input_ids, targets


def load_wikitext2(data_dir, max_seq_len, batch_size):
    """加载WikiText-2数据集 - 使用Qwen Tokenizer"""
    
    # 文件路径 (支持 .txt 和 .tokens 两种后缀)
    train_files = ['wiki.train.tokens', 'wiki.train.txt']
    valid_files = ['wiki.valid.tokens', 'wiki.valid.txt']
    test_files = ['wiki.test.tokens', 'wiki.test.txt']
    
    # 查找实际存在的文件
    def find_file(file_list):
        for fname in file_list:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find any of {file_list} in {data_dir}")
    
    train_path = find_file(train_files)
    valid_path = find_file(valid_files)
    test_path = find_file(test_files)
    
    print("\n" + "="*60)
    print("Loading Qwen Tokenizer...")
    print("="*60)
    
    # 加载Qwen tokenizer
    # 使用Qwen2.5系列的tokenizer (最新且性能好)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("="*60 + "\n")
    
    # 创建数据集
    train_dataset = WikiTextDataset(train_path, tokenizer, max_seq_len)
    valid_dataset = WikiTextDataset(valid_path, tokenizer, max_seq_len)
    test_dataset = WikiTextDataset(test_path, tokenizer, max_seq_len)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader, len(tokenizer)