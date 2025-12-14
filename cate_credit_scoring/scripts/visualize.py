

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

sns.set_style('whitegrid')


def plot_training_curves(history_path='best_model.pt', save_path='training_curves.png'):
    """绘制训练曲线"""
    checkpoint = torch.load(history_path, map_location='cpu')
    history = checkpoint['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总损失
    axes[0, 0].plot(history['train_loss'], label='Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 分类损失
    axes[0, 1].plot(history['train_cls_loss'], label='Classification Loss', 
                    color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 对比损失
    axes[1, 0].plot(history['train_con_loss'], label='Contrastive Loss', 
                    color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Contrastive Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC曲线
    aucs = [m['auc'] for m in history['val_metrics']]
    axes[1, 1].plot(aucs, label='Validation AUC', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")


def plot_cv_results(results_path='results.json', save_path='cv_results.png'):
    """绘制交叉验证结果"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    metrics = ['recall', 'f_score', 'g_mean', 'accuracy', 
               'balanced_accuracy', 'mcc', 'auc']
    
    avg_values = [results['avg_metrics'][m] for m in metrics]
    std_values = [results['std_metrics'][m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, avg_values, yerr=std_values, capsize=5, 
                  color='skyblue', edgecolor='navy', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Results (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val, std) in enumerate(zip(bars, avg_values, std_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{val:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"CV results plot saved to: {save_path}")


if __name__ == '__main__':
    # 绘制训练曲线
    plot_training_curves()
    
    # 绘制交叉验证结果
    plot_cv_results()
