#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Route0过拟合监控模块
监控训练验证差距、注意力权重分布、路径复杂度等指标
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from sklearn.metrics import entropy

class Route0OverfittingMonitor:
    def __init__(self, window_size=10, patience=5):
        self.window_size = window_size
        self.patience = patience
        
        # 历史指标记录
        self.train_loss_history = deque(maxlen=window_size)
        self.val_loss_history = deque(maxlen=window_size)
        self.train_acc_history = deque(maxlen=window_size)
        self.val_acc_history = deque(maxlen=window_size)
        self.train_auprc_history = deque(maxlen=window_size)
        self.val_auprc_history = deque(maxlen=window_size)
        self.train_recall_history = deque(maxlen=window_size)
        self.val_recall_history = deque(maxlen=window_size)
        
        # 注意力权重分析
        self.attention_entropy_history = deque(maxlen=window_size)
        self.attention_concentration_history = deque(maxlen=window_size)
        
        # 路径复杂度分析
        self.path_length_history = deque(maxlen=window_size)
        self.path_diversity_history = deque(maxlen=window_size)
        
        # 过拟合标志
        self.overfitting_signals = []
    
    def update_metrics(self, train_metrics, val_metrics):
        """更新训练和验证指标"""
        # 基础指标
        self.train_loss_history.append(train_metrics['loss'])
        self.val_loss_history.append(val_metrics['loss'])
        self.train_acc_history.append(train_metrics['accuracy'])
        self.val_acc_history.append(val_metrics['accuracy'])
        self.train_auprc_history.append(train_metrics['auprc'])
        self.val_auprc_history.append(val_metrics['auprc'])
        self.train_recall_history.append(train_metrics['recall'])
        self.val_recall_history.append(val_metrics['recall'])
    
    def analyze_attention_weights(self, model, data_loader, device, max_samples=100):
        """分析注意力权重分布"""
        model.eval()
        all_attention_weights = []
        
        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= max_samples:
                    break
                
                g = batch['graph'].to(device)
                node_feats = batch['node_feats'].to(device)
                edge_weights = batch['edge_weights'].to(device)
                
                # 获取注意力权重（需要修改模型forward方法返回注意力权重）
                try:
                    _, attention_weights = model.forward_with_attention(g, node_feats, edge_weights)
                    all_attention_weights.extend(attention_weights.cpu().numpy().flatten())
                except AttributeError:
                    # 如果模型没有forward_with_attention方法，跳过
                    break
                
                sample_count += 1
        
        if all_attention_weights:
            # 计算注意力熵（多样性指标）
            attention_entropy = entropy(np.histogram(all_attention_weights, bins=50)[0] + 1e-8)
            self.attention_entropy_history.append(attention_entropy)
            
            # 计算注意力集中度（过拟合指标）
            attention_concentration = np.std(all_attention_weights)
            self.attention_concentration_history.append(attention_concentration)
            
            return attention_entropy, attention_concentration
        
        return None, None
    
    def analyze_path_complexity(self, model, data_loader, device, max_samples=50):
        """分析路径复杂度"""
        model.eval()
        path_lengths = []
        unique_paths = set()
        
        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= max_samples:
                    break
                
                g = batch['graph'].to(device)
                node_feats = batch['node_feats'].to(device)
                edge_weights = batch['edge_weights'].to(device)
                question_idx = batch['question_idx'].to(device)
                
                # 获取批量大小
                batch_size = g.batch_size if hasattr(g, 'batch_size') else len(batch['question_idx'])
                
                for i in range(min(batch_size, max_samples - sample_count)):
                    try:
                        # 提取路径（需要模型支持路径提取）
                        if hasattr(model, '_extract_paths'):
                            # 创建边注意力分数的模拟数据
                            edge_attention_scores = {}
                            edge_src, edge_dst = g.edges()
                            for j in range(len(edge_src)):
                                src, dst = edge_src[j].item(), edge_dst[j].item()
                                edge_attention_scores[(src, dst)] = [np.random.random()]
                            
                            paths = model._extract_paths(g, edge_attention_scores, question_idx[i].item())
                            
                            for path, weight in paths:
                                path_lengths.append(len(path))
                                unique_paths.add(tuple(path))
                    except:
                        continue
                
                sample_count += batch_size
        
        if path_lengths:
            avg_path_length = np.mean(path_lengths)
            path_diversity = len(unique_paths) / len(path_lengths) if path_lengths else 0
            
            self.path_length_history.append(avg_path_length)
            self.path_diversity_history.append(path_diversity)
            
            return avg_path_length, path_diversity
        
        return None, None
    
    def detect_overfitting_signals(self):
        """检测过拟合信号"""
        signals = []
        
        if len(self.train_loss_history) >= self.patience and len(self.val_loss_history) >= self.patience:
            # 信号1: 训练损失持续下降，验证损失上升
            recent_train_loss = list(self.train_loss_history)[-self.patience:]
            recent_val_loss = list(self.val_loss_history)[-self.patience:]
            
            train_trend = np.polyfit(range(len(recent_train_loss)), recent_train_loss, 1)[0]
            val_trend = np.polyfit(range(len(recent_val_loss)), recent_val_loss, 1)[0]
            
            if train_trend < -0.001 and val_trend > 0.001:
                signals.append("训练验证损失趋势分离")
            
            # 信号2: 训练验证准确率差距过大
            if len(self.train_acc_history) >= self.patience:
                recent_train_acc = list(self.train_acc_history)[-self.patience:]
                recent_val_acc = list(self.val_acc_history)[-self.patience:]
                
                avg_gap = np.mean([t - v for t, v in zip(recent_train_acc, recent_val_acc)])
                if avg_gap > 0.15:  # 15%差距阈值
                    signals.append(f"训练验证准确率差距过大: {avg_gap:.3f}")
        
        # 信号3: 注意力权重过度集中
        if len(self.attention_concentration_history) >= 3:
            recent_concentration = list(self.attention_concentration_history)[-3:]
            if all(c > 0.8 for c in recent_concentration):  # 注意力过度集中
                signals.append("注意力权重过度集中")
        
        # 信号4: 路径复杂度异常
        if len(self.path_length_history) >= 3:
            recent_lengths = list(self.path_length_history)[-3:]
            if all(l > 5 for l in recent_lengths):  # 路径过长
                signals.append("推理路径过度复杂")
        
        if len(self.path_diversity_history) >= 3:
            recent_diversity = list(self.path_diversity_history)[-3:]
            if all(d < 0.3 for d in recent_diversity):  # 路径多样性过低
                signals.append("推理路径多样性不足")
        
        self.overfitting_signals = signals
        return signals
    
    def should_early_stop(self, min_epochs=10):
        """综合判断是否应该早停"""
        if len(self.val_loss_history) < min_epochs:
            return False, "训练轮数不足"
        
        # 检测过拟合信号
        signals = self.detect_overfitting_signals()
        
        # 如果有多个过拟合信号，建议早停
        if len(signals) >= 2:
            return True, f"检测到多个过拟合信号: {'; '.join(signals)}"
        
        # 检查验证指标是否停止改进
        if len(self.val_auprc_history) >= self.patience:
            recent_auprc = list(self.val_auprc_history)[-self.patience:]
            recent_recall = list(self.val_recall_history)[-self.patience:]
            
            auprc_improved = max(recent_auprc) - min(recent_auprc) > 0.01
            recall_improved = max(recent_recall) - min(recent_recall) > 0.01
            
            if not auprc_improved and not recall_improved:
                return True, "验证指标停止改进"
        
        # 检查是否达到目标性能
        if (len(self.val_auprc_history) > 0 and len(self.val_recall_history) > 0 and
            self.val_auprc_history[-1] >= 0.35 and self.val_recall_history[-1] >= 0.20):
            return True, "达到目标性能"
        
        return False, "继续训练"
    
    def generate_overfitting_report(self, epoch):
        """生成过拟合分析报告"""
        report = f"\n=== Route0 过拟合分析报告 (Epoch {epoch}) ===\n"
        
        # 基础指标分析
        if len(self.train_loss_history) > 0 and len(self.val_loss_history) > 0:
            train_loss = self.train_loss_history[-1]
            val_loss = self.val_loss_history[-1]
            loss_gap = val_loss - train_loss
            
            report += f"损失分析:\n"
            report += f"  训练损失: {train_loss:.4f}\n"
            report += f"  验证损失: {val_loss:.4f}\n"
            report += f"  损失差距: {loss_gap:.4f} {'⚠️' if loss_gap > 0.1 else '✅'}\n"
        
        if len(self.train_acc_history) > 0 and len(self.val_acc_history) > 0:
            train_acc = self.train_acc_history[-1]
            val_acc = self.val_acc_history[-1]
            acc_gap = train_acc - val_acc
            
            report += f"\n准确率分析:\n"
            report += f"  训练准确率: {train_acc:.4f}\n"
            report += f"  验证准确率: {val_acc:.4f}\n"
            report += f"  准确率差距: {acc_gap:.4f} {'⚠️' if acc_gap > 0.15 else '✅'}\n"
        
        # 注意力权重分析
        if len(self.attention_entropy_history) > 0:
            attention_entropy = self.attention_entropy_history[-1]
            report += f"\n注意力分析:\n"
            report += f"  注意力熵: {attention_entropy:.4f} {'⚠️' if attention_entropy < 2.0 else '✅'}\n"
        
        if len(self.attention_concentration_history) > 0:
            attention_concentration = self.attention_concentration_history[-1]
            report += f"  注意力集中度: {attention_concentration:.4f} {'⚠️' if attention_concentration > 0.8 else '✅'}\n"
        
        # 路径复杂度分析
        if len(self.path_length_history) > 0:
            avg_path_length = self.path_length_history[-1]
            report += f"\n路径分析:\n"
            report += f"  平均路径长度: {avg_path_length:.2f} {'⚠️' if avg_path_length > 5 else '✅'}\n"
        
        if len(self.path_diversity_history) > 0:
            path_diversity = self.path_diversity_history[-1]
            report += f"  路径多样性: {path_diversity:.4f} {'⚠️' if path_diversity < 0.3 else '✅'}\n"
        
        # 过拟合信号
        signals = self.detect_overfitting_signals()
        if signals:
            report += f"\n⚠️ 过拟合信号:\n"
            for signal in signals:
                report += f"  - {signal}\n"
        else:
            report += f"\n✅ 未检测到明显过拟合信号\n"
        
        # 早停建议
        should_stop, reason = self.should_early_stop()
        report += f"\n早停建议: {'是' if should_stop else '否'} ({reason})\n"
        
        return report
    
    def plot_overfitting_analysis(self, save_path=None):
        """绘制过拟合分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 损失曲线
        if len(self.train_loss_history) > 0:
            epochs = range(len(self.train_loss_history))
            axes[0, 0].plot(epochs, list(self.train_loss_history), 'b-', label='训练损失')
            axes[0, 0].plot(epochs, list(self.val_loss_history), 'r-', label='验证损失')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 准确率曲线
        if len(self.train_acc_history) > 0:
            epochs = range(len(self.train_acc_history))
            axes[0, 1].plot(epochs, list(self.train_acc_history), 'b-', label='训练准确率')
            axes[0, 1].plot(epochs, list(self.val_acc_history), 'r-', label='验证准确率')
            axes[0, 1].set_title('准确率曲线')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # AUPRC曲线
        if len(self.train_auprc_history) > 0:
            epochs = range(len(self.train_auprc_history))
            axes[0, 2].plot(epochs, list(self.train_auprc_history), 'b-', label='训练AUPRC')
            axes[0, 2].plot(epochs, list(self.val_auprc_history), 'r-', label='验证AUPRC')
            axes[0, 2].set_title('AUPRC曲线')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # 注意力熵
        if len(self.attention_entropy_history) > 0:
            epochs = range(len(self.attention_entropy_history))
            axes[1, 0].plot(epochs, list(self.attention_entropy_history), 'g-')
            axes[1, 0].set_title('注意力熵')
            axes[1, 0].axhline(y=2.0, color='r', linestyle='--', label='过拟合阈值')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 路径长度
        if len(self.path_length_history) > 0:
            epochs = range(len(self.path_length_history))
            axes[1, 1].plot(epochs, list(self.path_length_history), 'purple')
            axes[1, 1].set_title('平均路径长度')
            axes[1, 1].axhline(y=5.0, color='r', linestyle='--', label='复杂度阈值')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # 路径多样性
        if len(self.path_diversity_history) > 0:
            epochs = range(len(self.path_diversity_history))
            axes[1, 2].plot(epochs, list(self.path_diversity_history), 'orange')
            axes[1, 2].set_title('路径多样性')
            axes[1, 2].axhline(y=0.3, color='r', linestyle='--', label='多样性阈值')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 