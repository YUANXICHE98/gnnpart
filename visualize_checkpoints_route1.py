#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Route1检查点可视化脚本 - 增强版
包含自适应稀疏分析、训练监控和过拟合检测
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_metrics(metrics_file):
    """加载训练指标"""
    if not os.path.exists(metrics_file):
        print(f"指标文件不存在: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        print(f"成功加载 {len(metrics)} 个epoch的指标")
        return metrics
    except Exception as e:
        print(f"加载指标文件失败: {e}")
        return None

def extract_metric_series(metrics, metric_name):
    """提取指标时间序列"""
    if not metrics:
        return [], []
    
    epochs = []
    values = []
    
    for metric in metrics:
        if metric_name in metric:
            epochs.append(metric['epoch'])
            values.append(metric[metric_name])
    
    return epochs, values

def plot_basic_metrics(metrics, save_dir):
    """绘制基础训练指标 - 9宫格布局"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 提取数据
    epochs, train_loss = extract_metric_series(metrics, 'train_loss')
    _, val_loss = extract_metric_series(metrics, 'val_loss')
    _, train_acc = extract_metric_series(metrics, 'train_acc')
    _, val_acc = extract_metric_series(metrics, 'val_acc')
    _, train_f1 = extract_metric_series(metrics, 'train_f1')
    _, val_f1 = extract_metric_series(metrics, 'val_f1')
    _, train_sparsity = extract_metric_series(metrics, 'train_sparsity')
    _, val_sparsity = extract_metric_series(metrics, 'val_sparsity')
    
    # 1. 损失变化 (0,0)
    axes[0, 0].plot(epochs, train_loss, 'b-', marker='o', linewidth=2, markersize=4, label='train_loss')
    axes[0, 0].plot(epochs, val_loss, 'r-', marker='s', linewidth=2, markersize=4, label='val_loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('loss change')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
        
    # 2. 训练准确率变化 (0,1)
    axes[0, 1].plot(epochs, train_acc, 'g-', marker='o', linewidth=2, markersize=4, label='train_acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Accuracy')
    axes[0, 1].set_title('train_acc change')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 验证准确率变化 (0,2)
    axes[0, 2].plot(epochs, val_acc, 'orange', marker='s', linewidth=2, markersize=4, label='val_acc')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Validation Accuracy')
    axes[0, 2].set_title('val_acc change')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 训练稀疏度变化 (1,0)
    axes[1, 0].plot(epochs, train_sparsity, 'purple', marker='o', linewidth=2, markersize=4, label='train_sparsity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Sparsity')
    axes[1, 0].set_title('train_sparsity change')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 验证稀疏度变化 (1,1)
    axes[1, 1].plot(epochs, val_sparsity, 'brown', marker='s', linewidth=2, markersize=4, label='val_sparsity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Sparsity')
    axes[1, 1].set_title('val_sparsity change')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 验证F1变化 (1,2)
    axes[1, 2].plot(epochs, val_f1, 'red', marker='o', linewidth=2, markersize=4, label='val_f1')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_title('val_f1 change')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. 准确率对比 (2,0)
    axes[2, 0].plot(epochs, train_acc, 'g-', marker='o', linewidth=2, markersize=4, label='train_acc')
    axes[2, 0].plot(epochs, val_acc, 'orange', marker='s', linewidth=2, markersize=4, label='val_acc')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('acc change')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. 稀疏度对比 (2,1)
    axes[2, 1].plot(epochs, train_sparsity, 'purple', marker='o', linewidth=2, markersize=4, label='train_sparsity')
    axes[2, 1].plot(epochs, val_sparsity, 'brown', marker='s', linewidth=2, markersize=4, label='val_sparsity')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Sparsity')
    axes[2, 1].set_title('sparsity change')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. F1 vs 稀疏度散点图 (2,2)
    # 创建颜色映射
    scatter = axes[2, 2].scatter(val_sparsity, val_f1, c=epochs, cmap='viridis', s=50, alpha=0.7)
    axes[2, 2].set_xlabel('Validation Sparsity')
    axes[2, 2].set_ylabel('Validation F1')
    axes[2, 2].set_title('f1 vs sparsity')
    axes[2, 2].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[2, 2])
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'route1_basic_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"基础指标图表已保存: {save_path}")
    
    return fig

def plot_sparsity_analysis(metrics, save_dir):
    """绘制稀疏度深度分析"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取数据
    epochs, train_sparsity = extract_metric_series(metrics, 'train_sparsity')
    _, val_sparsity = extract_metric_series(metrics, 'val_sparsity')
    _, train_f1 = extract_metric_series(metrics, 'train_f1')
    _, val_f1 = extract_metric_series(metrics, 'val_f1')
    
    # 1. 稀疏度vs F1分数关系
    if train_sparsity and val_f1:
        axes[0, 0].scatter(train_sparsity, train_f1, alpha=0.7, c=epochs, cmap='viridis', s=50, label='Training')
        axes[0, 0].scatter(val_sparsity, val_f1, alpha=0.7, c=epochs, cmap='plasma', s=50, label='Validation')
        axes[0, 0].axvspan(0.2, 0.3, alpha=0.2, color='green', label='Target Sparsity')
        axes[0, 0].set_xlabel('Sparsity Rate')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('Sparsity vs F1 Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 稀疏度变化趋势
    if len(epochs) > 1:
        # 计算稀疏度变化率
        sparsity_change = []
        for i in range(1, len(val_sparsity)):
            change = val_sparsity[i] - val_sparsity[i-1]
            sparsity_change.append(change)
        
        change_epochs = epochs[1:]
        axes[0, 1].plot(change_epochs, sparsity_change, 'g-', linewidth=2, marker='o')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Sparsity Change Rate')
        axes[0, 1].set_title('Sparsity Change Trend')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 双轴图：稀疏度和F1分数
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, val_sparsity, 'b-', linewidth=2, marker='o', label='Sparsity Rate')
    ax1.axhspan(0.2, 0.3, alpha=0.2, color='blue', label='Target Range')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sparsity Rate', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax2.plot(epochs, val_f1, 'r-', linewidth=2, marker='x', label='Validation F1')
    ax2.set_ylabel('F1 Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    axes[1, 0].set_title('Sparsity vs F1 Score')
    
    # 4. 稀疏度收敛分析
    if len(val_sparsity) >= 5:
        # 计算最近5个epoch的稀疏度标准差
        window_size = 5
        convergence_epochs = []
        convergence_std = []
        
        for i in range(window_size-1, len(val_sparsity)):
            window_data = val_sparsity[i-window_size+1:i+1]
            std_val = np.std(window_data)
            convergence_epochs.append(epochs[i])
            convergence_std.append(std_val)
        
        axes[1, 1].plot(convergence_epochs, convergence_std, 'purple', linewidth=2, marker='s')
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Convergence Threshold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sparsity Std (5-epoch window)')
        axes[1, 1].set_title('Sparsity Convergence Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'route1_sparsity_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"稀疏度分析图表已保存: {save_path}")
    
    return fig

def generate_training_report(metrics, save_dir):
    """生成训练报告"""
    if not metrics:
        return
    
    report_path = os.path.join(save_dir, 'route1_training_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Route1 GNN软掩码训练报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本信息
        f.write("【基本信息】\n")
        f.write(f"总训练轮数: {len(metrics)}\n")
        
        if metrics:
            last_metric = metrics[-1]
            f.write(f"最终训练损失: {last_metric.get('train_loss', 'N/A'):.4f}\n")
            f.write(f"最终验证损失: {last_metric.get('val_loss', 'N/A'):.4f}\n")
            f.write(f"最终训练准确率: {last_metric.get('train_acc', 'N/A'):.4f}\n")
            f.write(f"最终验证准确率: {last_metric.get('val_acc', 'N/A'):.4f}\n")
            f.write(f"最终训练F1: {last_metric.get('train_f1', 'N/A'):.4f}\n")
            f.write(f"最终验证F1: {last_metric.get('val_f1', 'N/A'):.4f}\n")
            f.write(f"最终稀疏度: {last_metric.get('val_sparsity', 'N/A'):.4f}\n")
            f.write(f"稀疏度状态: {last_metric.get('sparsity_status', 'N/A')}\n\n")
        
        # 最佳性能
        f.write("【最佳性能】\n")
        val_f1_scores = [m.get('val_f1', 0) for m in metrics]
        val_acc_scores = [m.get('val_acc', 0) for m in metrics]
        
        if val_f1_scores:
            best_f1_idx = np.argmax(val_f1_scores)
            best_f1_epoch = metrics[best_f1_idx]['epoch']
            best_f1_score = val_f1_scores[best_f1_idx]
            f.write(f"最佳F1分数: {best_f1_score:.4f} (Epoch {best_f1_epoch})\n")
        
        if val_acc_scores:
            best_acc_idx = np.argmax(val_acc_scores)
            best_acc_epoch = metrics[best_acc_idx]['epoch']
            best_acc_score = val_acc_scores[best_acc_idx]
            f.write(f"最佳准确率: {best_acc_score:.4f} (Epoch {best_acc_epoch})\n\n")
        
        # 稀疏度分析
        f.write("【稀疏度分析】\n")
        val_sparsity = [m.get('val_sparsity', 0) for m in metrics]
        if val_sparsity:
            f.write(f"平均稀疏度: {np.mean(val_sparsity):.4f}\n")
            f.write(f"稀疏度标准差: {np.std(val_sparsity):.4f}\n")
            f.write(f"最低稀疏度: {np.min(val_sparsity):.4f}\n")
            f.write(f"最高稀疏度: {np.max(val_sparsity):.4f}\n")
            
            # 目标范围内的epoch数
            in_range_count = sum(1 for s in val_sparsity if 0.2 <= s <= 0.3)
            f.write(f"稀疏度在目标范围内的epoch数: {in_range_count}/{len(val_sparsity)}\n\n")
    
    print(f"训练报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Route1检查点可视化')
    parser.add_argument('--checkpoint_dir', type=str, default='autodl-tmp/models/route1_gnn_softmask/checkpoints/checkpoints529', help='检查点目录')
    parser.add_argument('--output_dir', type=str, default='autodl-tmp/models/route1_gnn_softmask/checkpoints/checkpoints529/visualization', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找metrics.json文件
    metrics_file = os.path.join(args.checkpoint_dir, 'metrics.json')
    
    # 加载指标
    metrics = load_metrics(metrics_file)
    if not metrics:
        print("无法加载训练指标，退出...")
        return
    
    print(f"开始生成Route1可视化图表...")
    
    # 生成各种图表
    try:
        plot_basic_metrics(metrics, args.output_dir)
        plot_sparsity_analysis(metrics, args.output_dir)
        generate_training_report(metrics, args.output_dir)
        
        print(f"\n✅ 所有可视化图表已生成完成!")
        print(f"输出目录: {args.output_dir}")
        print(f"包含文件:")
        print(f"  - route1_basic_metrics.png: 基础训练指标")
        print(f"  - route1_sparsity_analysis.png: 稀疏度深度分析")
        print(f"  - route1_training_report.txt: 详细训练报告")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")

if __name__ == '__main__':
    main() 