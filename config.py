class Config:
    # Model Architecture
    d_model = 512          # 模型维度
    n_heads = 4            # 注意力头数
    n_layers = 8           # Transformer层数
    d_ff = 2048            # FFN中间层维度 (通常是4倍d_model)
    dropout = 0.3          # Dropout率
    max_seq_len = 256      # 最大序列长度
    
    # Training
    batch_size = 6         # 单步batch size (词汇量更大,调小一点)
    gradient_accumulation_steps = 6  # 梯度累积步数
    effective_batch_size = batch_size * gradient_accumulation_steps  # 有效batch size = 36
    
    learning_rate = 1e-4   # 学习率
    weight_decay = 0.1    # 权重衰减
    max_epochs = 4        # 训练轮数
    warmup_steps = 500     # 学习率warmup步数


    # Early Stopping
    early_stopping = True  # 启用早停
    patience = 5           # 验证loss连续5个epoch不下降则停止
    
    # Learning Rate Scheduling
    use_scheduler = True   # 使用学习率调度器
    lr_decay_factor = 0.5  # 学习率衰减因子
    lr_patience = 3        # 验证loss连续3个epoch不下降则降低学习率
    
    # Gradient Clipping
    max_grad_norm = 0.5    # 🔧 梯度裁剪 (1.0→0.5, 更激进)
    
    # Device
    device = 'cuda'        # 使用GPU
    use_amp = True         # 使用混合精度训练
    
    # Data
    data_dir = './wikitext-2'  # 数据集目录
    vocab_size = None      # 将在数据加载后设置
    
    # Logging
    log_interval = 50      # 每多少步打印一次
    eval_interval = 500    # 每多少步验证一次
    
    def __repr__(self):
        return f"""
Config (Regularized):
  Model: d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}
  Regularization: dropout={self.dropout}, weight_decay={self.weight_decay}, grad_clip={self.max_grad_norm}
  Training: batch_size={self.batch_size}, grad_accum={self.gradient_accumulation_steps}, lr={self.learning_rate}
  Early Stopping: {self.early_stopping} (patience={self.patience})
  LR Scheduler: {self.use_scheduler} (factor={self.lr_decay_factor}, patience={self.lr_patience})
  Sequence Length: {self.max_seq_len}
  Epochs: {self.max_epochs}
  Device: {self.device}, AMP: {self.use_amp}
"""