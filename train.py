import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from model import DecoderOnlyTransformer
from data import load_wikitext2


def get_lr(step, warmup_steps, d_model):
    """带warmup的学习率调度"""
    if step < warmup_steps:
        return (d_model ** -0.5) * step * (warmup_steps ** -1.5)
    else:
        return (d_model ** -0.5) * (step ** -0.5)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载数据
        print("\n" + "="*50)
        self.train_loader, self.valid_loader, self.test_loader, vocab_size = load_wikitext2(
            config.data_dir,
            config.max_seq_len,
            config.batch_size
        )
        
        # 更新词汇表大小
        config.vocab_size = vocab_size
        
        # 创建模型
        print("\n" + "="*50)
        print("Creating model...")
        self.model = DecoderOnlyTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        ).to(self.device)
        
        print(f"Model parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(config)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_amp else None
        
        # 记录
        self.train_losses = []
        self.valid_losses = []
        self.train_ppls = []
        self.valid_ppls = []
        self.global_step = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            with autocast(enabled=self.config.use_amp):
                logits, loss = self.model(input_ids, targets)
                loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 学习率调度
                lr = get_lr(self.global_step, self.config.warmup_steps, self.config.d_model)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # 统计
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_tokens += targets.numel()
            
            # 更新进度条
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                avg_loss = total_loss / ((batch_idx + 1) / self.config.gradient_accumulation_steps)
                ppl = math.exp(min(avg_loss, 20))
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{ppl:.2f}',
                    'lr': f'{lr:.6f}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_ppl = math.exp(min(avg_loss, 20))
        return avg_loss, avg_ppl
    
    @torch.no_grad()
    def evaluate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for input_ids, targets in tqdm(self.valid_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                logits, loss = self.model(input_ids, targets)
            
            total_loss += loss.item()
            total_tokens += targets.numel()
        
        avg_loss = total_loss / len(self.valid_loader)
        avg_ppl = math.exp(min(avg_loss, 20))
        return avg_loss, avg_ppl
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        best_valid_loss = float('inf')
        
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_ppl = self.train_epoch(epoch)
            
            # 验证
            valid_loss, valid_ppl = self.evaluate()
            
            # 记录
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_ppls.append(train_ppl)
            self.valid_ppls.append(valid_ppl)
            
            # 打印结果
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs} | Time: {elapsed:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print(f"✓ New best validation loss!")
            
            print("-" * 50)
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best Valid Loss: {best_valid_loss:.4f}")
        print(f"Best Valid PPL: {math.exp(min(best_valid_loss, 20)):.2f}")
        print("="*50 + "\n")
        
        # 绘制曲线
        self.plot_curves()
    
    def plot_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.valid_losses, 'r-', label='Valid Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Perplexity曲线
        ax2.plot(epochs, self.train_ppls, 'b-', label='Train PPL', linewidth=2)
        ax2.plot(epochs, self.valid_ppls, 'r-', label='Valid PPL', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Training curves saved to 'training_curves.png'")
        plt.show()


def main():
    # 创建配置
    config = Config()
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()