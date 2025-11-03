import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import Config
from train import Trainer
import numpy as np


class AblationConfig(Config):
    """消融实验配置"""
    def __init__(self, n_layers, n_heads, exp_name):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.exp_name = exp_name
        
        # 减少训练轮数以加快实验速度
        self.max_epochs = 5  # 20→15 epochs
        
        # 确保 d_model 能被 n_heads 整除
        assert self.d_model % n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})"
        
    def __repr__(self):
        return f"""
Ablation Experiment: {self.exp_name}
  Layers: {self.n_layers}, Heads: {self.n_heads}
  d_model: {self.d_model}, batch_size: {self.batch_size}
  Epochs: {self.max_epochs}
"""


class AblationStudy:
    """消融实验管理器"""
    def __init__(self, base_config, save_dir='ablation_results'):
        self.base_config = base_config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 实验配置
        self.experiments = self.design_experiments()
        self.results = []
        
    def design_experiments(self):
        """设计消融实验
        
        实验设计：
        1. 固定头数，变化层数：验证深度的影响
        2. 固定层数，变化头数：验证多头注意力的影响
        """
        experiments = []
        
        # 基准实验
        experiments.append({
            'name': 'baseline',
            'n_layers': 8,
            'n_heads': 4,
            'description': 'Baseline (8 layers, 4 heads)'
        })
        
        # 实验组1：固定头数(4)，变化层数
        print("\n" + "="*60)
        print("Experiment Group 1: Varying Layers (Fixed Heads=4)")
        print("="*60)
        for n_layers in [4, 6, 10, 12]:
            experiments.append({
                'name': f'layers_{n_layers}_heads_4',
                'n_layers': n_layers,
                'n_heads': 4,
                'description': f'{n_layers} layers, 4 heads'
            })
            print(f"  - {n_layers} layers, 4 heads")
        
        # 实验组2：固定层数(8)，变化头数
        print("\n" + "="*60)
        print("Experiment Group 2: Varying Heads (Fixed Layers=8)")
        print("="*60)
        for n_heads in [2, 8, 16]:
            # 确保能整除
            if 512 % n_heads == 0:
                experiments.append({
                    'name': f'layers_8_heads_{n_heads}',
                    'n_layers': 8,
                    'n_heads': n_heads,
                    'description': f'8 layers, {n_heads} heads'
                })
                print(f"  - 8 layers, {n_heads} heads")
        
        print(f"\nTotal: {len(experiments)} experiments")
        print("="*60 + "\n")
        
        return experiments
    
    def run_experiment(self, exp_config):
        """运行单个实验"""
        print("\n" + "="*70)
        print(f"🔬 Running: {exp_config['description']}")
        print("="*70)
        
        # 创建配置
        config = AblationConfig(
            n_layers=exp_config['n_layers'],
            n_heads=exp_config['n_heads'],
            exp_name=exp_config['name']
        )
        print(config)
        
        # 训练
        trainer = Trainer(config)
        trainer.train()
        
        # 收集结果
        result = {
            'name': exp_config['name'],
            'description': exp_config['description'],
            'n_layers': exp_config['n_layers'],
            'n_heads': exp_config['n_heads'],
            'n_params': trainer.model.count_parameters(),
            'train_losses': trainer.train_losses,
            'valid_losses': trainer.valid_losses,
            'train_ppls': trainer.train_ppls,
            'valid_ppls': trainer.valid_ppls,
            'best_valid_loss': min(trainer.valid_losses),
            'best_valid_ppl': min(trainer.valid_ppls),
            'final_valid_loss': trainer.valid_losses[-1],
            'final_valid_ppl': trainer.valid_ppls[-1]
        }
        
        # 保存结果
        self.results.append(result)
        self.save_results()
        
        # 清理显存
        del trainer
        torch.cuda.empty_cache()
        
        return result
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "🚀"*35)
        print(f"Starting Ablation Study: {len(self.experiments)} experiments")
        print("🚀"*35 + "\n")
        
        for i, exp in enumerate(self.experiments, 1):
            print(f"\n{'='*70}")
            print(f"Progress: [{i}/{len(self.experiments)}]")
            print(f"{'='*70}")
            
            try:
                self.run_experiment(exp)
            except Exception as e:
                print(f"❌ Experiment {exp['name']} failed: {e}")
                continue
        
        print("\n" + "✅"*35)
        print("Ablation Study Completed!")
        print("✅"*35 + "\n")
        
        # 生成分析报告
        self.generate_report()
    
    def save_results(self):
        """保存实验结果"""
        results_file = os.path.join(self.save_dir, 'results.json')
        
        # 转换为可序列化的格式
        serializable_results = []
        for r in self.results:
            result_copy = r.copy()
            # 将numpy数组转为列表
            for key in ['train_losses', 'valid_losses', 'train_ppls', 'valid_ppls']:
                if key in result_copy:
                    result_copy[key] = [float(x) for x in result_copy[key]]
            serializable_results.append(result_copy)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def generate_report(self):
        """生成分析报告和可视化"""
        print("\n" + "="*70)
        print("📊 Generating Analysis Report...")
        print("="*70 + "\n")
        
        # 1. 创建结果表格
        self.create_results_table()
        
        # 2. 绘制训练曲线对比
        self.plot_training_curves()
        
        # 3. 绘制层数影响分析
        self.plot_layers_analysis()
        
        # 4. 绘制头数影响分析
        self.plot_heads_analysis()
        
        # 5. 参数量vs性能分析
        self.plot_params_vs_performance()
        
        print("\n✅ Report generated successfully!")
        print(f"   All results saved to: {self.save_dir}/")
    
    def create_results_table(self):
        """创建结果对比表"""
        print("Creating results table...")
        
        # 准备数据
        data = []
        for r in self.results:
            data.append({
                'Experiment': r['description'],
                'Layers': r['n_layers'],
                'Heads': r['n_heads'],
                'Parameters (M)': f"{r['n_params']/1e6:.2f}",
                'Best Valid Loss': f"{r['best_valid_loss']:.4f}",
                'Best Valid PPL': f"{r['best_valid_ppl']:.2f}",
                'Final Valid Loss': f"{r['final_valid_loss']:.4f}",
                'Final Valid PPL': f"{r['final_valid_ppl']:.2f}"
            })
        
        df = pd.DataFrame(data)
        
        # 保存为CSV
        csv_path = os.path.join(self.save_dir, 'results_table.csv')
        df.to_csv(csv_path, index=False)
        
        # 打印表格
        print("\n" + "="*100)
        print("ABLATION STUDY RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        # 保存为文本
        txt_path = os.path.join(self.save_dir, 'results_table.txt')
        with open(txt_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("ABLATION STUDY RESULTS\n")
            f.write("="*100 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n" + "="*100 + "\n")
    
    def plot_training_curves(self):
        """绘制所有实验的训练曲线"""
        print("Plotting training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for idx, r in enumerate(self.results):
            epochs = range(1, len(r['valid_losses']) + 1)
            label = r['description']
            color = colors[idx]
            
            # Train Loss
            axes[0, 0].plot(epochs, r['train_losses'], label=label, 
                           color=color, linewidth=2, alpha=0.8)
            
            # Valid Loss
            axes[0, 1].plot(epochs, r['valid_losses'], label=label, 
                           color=color, linewidth=2, alpha=0.8)
            
            # Train PPL
            axes[1, 0].plot(epochs, r['train_ppls'], label=label, 
                           color=color, linewidth=2, alpha=0.8)
            
            # Valid PPL
            axes[1, 1].plot(epochs, r['valid_ppls'], label=label, 
                           color=color, linewidth=2, alpha=0.8)
        
        # 设置标题和标签
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_title('Training Perplexity', fontsize=14, fontweight='bold')
        axes[1, 1].set_title('Validation Perplexity', fontsize=14, fontweight='bold')
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_ylabel('Perplexity', fontsize=12)
        axes[1, 1].set_ylabel('Perplexity', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'all_training_curves.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_layers_analysis(self):
        """分析层数的影响（固定头数=4）"""
        print("Plotting layers analysis...")
        
        # 筛选固定头数的实验
        layer_results = [r for r in self.results if r['n_heads'] == 4]
        layer_results.sort(key=lambda x: x['n_layers'])
        
        if len(layer_results) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        layers = [r['n_layers'] for r in layer_results]
        best_ppls = [r['best_valid_ppl'] for r in layer_results]
        final_ppls = [r['final_valid_ppl'] for r in layer_results]
        params = [r['n_params']/1e6 for r in layer_results]
        
        # Best Valid PPL vs Layers
        axes[0].plot(layers, best_ppls, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        axes[0].set_xlabel('Number of Layers', fontsize=12)
        axes[0].set_ylabel('Best Valid Perplexity', fontsize=12)
        axes[0].set_title('Best PPL vs Layers (Heads=4)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Final Valid PPL vs Layers
        axes[1].plot(layers, final_ppls, 's-', linewidth=2, markersize=10, color='#A23B72')
        axes[1].set_xlabel('Number of Layers', fontsize=12)
        axes[1].set_ylabel('Final Valid Perplexity', fontsize=12)
        axes[1].set_title('Final PPL vs Layers (Heads=4)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Parameters vs Layers
        axes[2].plot(layers, params, 'd-', linewidth=2, markersize=10, color='#F18F01')
        axes[2].set_xlabel('Number of Layers', fontsize=12)
        axes[2].set_ylabel('Parameters (M)', fontsize=12)
        axes[2].set_title('Model Size vs Layers', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'layers_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_heads_analysis(self):
        """分析头数的影响（固定层数=8）"""
        print("Plotting heads analysis...")
        
        # 筛选固定层数的实验
        head_results = [r for r in self.results if r['n_layers'] == 8]
        head_results.sort(key=lambda x: x['n_heads'])
        
        if len(head_results) < 2:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        heads = [r['n_heads'] for r in head_results]
        best_ppls = [r['best_valid_ppl'] for r in head_results]
        final_ppls = [r['final_valid_ppl'] for r in head_results]
        
        # Best Valid PPL vs Heads
        axes[0].plot(heads, best_ppls, 'o-', linewidth=2, markersize=10, color='#06A77D')
        axes[0].set_xlabel('Number of Heads', fontsize=12)
        axes[0].set_ylabel('Best Valid Perplexity', fontsize=12)
        axes[0].set_title('Best PPL vs Heads (Layers=8)', fontsize=14, fontweight='bold')
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True, alpha=0.3)
        
        # Final Valid PPL vs Heads
        axes[1].plot(heads, final_ppls, 's-', linewidth=2, markersize=10, color='#D62246')
        axes[1].set_xlabel('Number of Heads', fontsize=12)
        axes[1].set_ylabel('Final Valid Perplexity', fontsize=12)
        axes[1].set_title('Final PPL vs Heads (Layers=8)', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'heads_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_params_vs_performance(self):
        """绘制参数量vs性能的关系"""
        print("Plotting parameters vs performance...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        params = [r['n_params']/1e6 for r in self.results]
        best_ppls = [r['best_valid_ppl'] for r in self.results]
        labels = [r['description'] for r in self.results]
        
        # 散点图
        scatter = ax.scatter(params, best_ppls, s=200, alpha=0.6, 
                            c=range(len(self.results)), cmap='viridis')
        
        # 添加标签
        for i, label in enumerate(labels):
            ax.annotate(label, (params[i], best_ppls[i]), 
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Parameters (M)', fontsize=12)
        ax.set_ylabel('Best Valid Perplexity', fontsize=12)
        ax.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'params_vs_performance.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """运行消融实验"""
    # 创建基础配置
    base_config = Config()
    
    # 创建消融实验
    ablation = AblationStudy(base_config)
    
    # 运行所有实验
    ablation.run_all()
    
    print("\n" + "🎉"*35)
    print(f"All results saved to: {ablation.save_dir}/")
    print("🎉"*35 + "\n")


if __name__ == '__main__':
    main()