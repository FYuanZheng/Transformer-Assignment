# Decoder-Only Transformer 语言模型

手写实现的 Decoder-only Transformer，使用旋转位置编码(RoPE)和GELU激活函数，在 WikiText-2 数据集上训练。

## 🌟 特性

- ✅ **手写实现 Transformer**：不使用 `torch.nn.Transformer`
- ✅ **旋转位置编码(RoPE)**：替代传统的绝对位置编码
- ✅ **GELU 激活函数**：现代 Transformer 标配
- ✅ **Qwen Tokenizer**：使用工业级分词器
- ✅ **4GB 显存优化**：混合精度训练 + 梯度累积
- ✅ **完整训练流程**：带进度条和可视化

## 📁 项目结构

```
.
├── config.py          # 配置文件（超参数）
├── model.py           # Transformer模型定义
├── data.py            # 数据加载和预处理
├── train.py           # 训练主程序
├── ablation.py        # 消融实验
├── requirements.txt   # 依赖包列表
└── wikitext-2/        # 数据集目录
    ├── wiki.train.tokens (或 .txt)
    ├── wiki.valid.tokens (或 .txt)
    └── wiki.test.tokens  (或 .txt)
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（可选）
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖
pip install -r requirements.txt

# 注意：PyTorch 需要根据你的CUDA版本安装
# 访问 https://pytorch.org/ 获取正确的安装命令
```

### 2. 准备数据集

下载 WikiText-2 数据集并放置到 `./wikitext-2/` 目录：
- [WikiText-2 下载链接](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

数据集也已包含在项目中上传，不用再次下载。

### 3. 开始训练

```bash
python train.py
```

## ⚙️ 配置说明

主要配置在 `config.py` 中：

```python
# 模型架构
d_model = 256          # 模型维度
n_heads = 4            # 注意力头数
n_layers = 6           # Transformer层数
max_seq_len = 128      # 序列长度

# 训练参数
batch_size = 2         # 单步batch
gradient_accumulation_steps = 16  # 梯度累积
max_epochs = 20        # 训练轮数
learning_rate = 3e-4   # 学习率
```

### 💾 显存优化配置

| 配置 | 4GB显存 | 8GB显存 | 16GB显存 |
|------|---------|---------|----------|
| d_model | 256 | 512 | 768 |
| n_layers | 6 | 8 | 12 |
| batch_size | 2 | 8 | 16 |
| max_seq_len | 128 | 256 | 512 |

## 📊 训练输出

训练过程中会显示：
- 实时进度条（每个batch）
- 每个epoch的 Loss 和 Perplexity
- 训练曲线图 `training_curves.png`

示例输出：
```
Epoch 1/20 | Time: 120.5s
Train Loss: 5.2341 | Train PPL: 187.23
Valid Loss: 4.8912 | Valid PPL: 132.67
✓ New best validation loss!
```

## 🎯 模型架构细节

### RoPE (旋转位置编码)
```python
# 相比绝对位置编码的优势：
- 更好的长度外推能力
- 相对位置信息编码
- 现代LLM标配（GPT-NeoX, LLaMA等）
```

### Pre-LN Transformer
```python
# Layer结构:
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

## 📈 预期效果

WikiText-2 数据集上的参考指标：

| Epochs | Valid PPL | 说明 |
|--------|-----------|------|
| 5 | 150-200 | 初步收敛 |
| 10 | 80-120 | 良好效果 |
| 20 | 50-80 | 较优效果 |

**注**: 小模型配置(d_model=256)的PPL会比大模型高，这是正常的。

## 🔧 故障排查

### OOM (显存不足)
```bash
# 方法1: 减小batch size
batch_size = 1

# 方法2: 减小序列长度
max_seq_len = 64

# 方法3: 减小模型
d_model = 128
n_layers = 4
```

### Tokenizer下载失败
```python
# 在 data.py 中切换到 GPT-2 tokenizer:
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### CUDA不可用
```python
# 在 config.py 中改为CPU训练:
device = 'cpu'
use_amp = False  # CPU不支持混合精度
```

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始Transformer论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE论文
- [WikiText-2 Dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
