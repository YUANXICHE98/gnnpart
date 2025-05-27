#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线0：路径感知多头注意力GNN - 训练脚本
使用多头注意力机制和路径记忆的GNN在子图上训练，用于多跳推理
集成了节点角色特定变换、边类型特定处理和路径记忆模块
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter  # 可视化工具
from collections import deque
from tqdm import tqdm
import networkx as nx

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import load_graph_data

# Route0过拟合监控类
class Route0OverfittingMonitor:
    """Route0过拟合监控器 - 多维度检测过拟合信号"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.attention_entropies = []
        self.path_complexities = []
        
        # 过拟合信号
        self.overfitting_signals = {
            'loss_gap': False,
            'acc_gap': False,
            'attention_collapse': False,
            'path_complexity': False
        }
        
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
    
    def calculate_attention_entropy(self, attention_weights):
        """计算注意力权重的熵 - 低熵表示注意力过度集中"""
        if len(attention_weights) == 0:
            return 0.0
        
        # 归一化注意力权重
        weights = np.array(list(attention_weights.values()))
        if len(weights) == 0:
            return 0.0
        
        # 计算每条边的平均注意力
        avg_weights = []
        for edge_weights in weights:
            if len(edge_weights) > 0:
                avg_weights.append(np.mean(edge_weights))
        
        if len(avg_weights) == 0:
            return 0.0
        
        # 归一化
        avg_weights = np.array(avg_weights)
        avg_weights = avg_weights / (np.sum(avg_weights) + 1e-8)
        
        # 计算熵
        entropy = -np.sum(avg_weights * np.log(avg_weights + 1e-8))
        return entropy
    
    def calculate_path_complexity(self, model, g, node_feats, edge_weights, question_idx):
        """计算路径复杂度 - 过拟合时路径可能过于复杂"""
        try:
            # 获取注意力分数（需要模型支持）
            if hasattr(model, '_extract_paths'):
                # 模拟注意力分数
                edge_attention_scores = {}
                edge_src, edge_dst = g.edges()
                for i in range(len(edge_src)):
                    src, dst = edge_src[i].item(), edge_dst[i].item()
                    edge_key = (src, dst)
                    edge_attention_scores[edge_key] = [np.random.random()]  # 占位符
                
                paths = model._extract_paths(g, edge_attention_scores, question_idx)
                
                if len(paths) > 0:
                    # 计算平均路径长度
                    avg_path_length = np.mean([len(path[0]) for path in paths])
                    # 计算路径权重方差
                    path_weights = [path[1] for path in paths]
                    weight_variance = np.var(path_weights) if len(path_weights) > 1 else 0
                    
                    return avg_path_length + weight_variance
                else:
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def update(self, train_loss, val_loss, train_acc, val_acc, 
               attention_weights=None, model=None, g=None, node_feats=None, 
               edge_weights=None, question_idx=None):
        """更新监控指标"""
        
        # 更新历史记录
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # 计算注意力熵
        if attention_weights is not None:
            entropy = self.calculate_attention_entropy(attention_weights)
            self.attention_entropies.append(entropy)
        
        # 计算路径复杂度
        if model is not None and g is not None:
            complexity = self.calculate_path_complexity(
                model, g, node_feats, edge_weights, question_idx
            )
            self.path_complexities.append(complexity)
        
        # 检测过拟合信号
        self._detect_overfitting()
        
        # 早停检查
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
    
    def _detect_overfitting(self):
        """检测各种过拟合信号"""
        if len(self.train_losses) < 3:
            return
        
        # 1. 损失差距检测
        recent_train_loss = np.mean(self.train_losses[-3:])
        recent_val_loss = np.mean(self.val_losses[-3:])
        loss_gap = recent_val_loss - recent_train_loss
        self.overfitting_signals['loss_gap'] = loss_gap > 0.1
        
        # 2. 准确率差距检测
        recent_train_acc = np.mean(self.train_accs[-3:])
        recent_val_acc = np.mean(self.val_accs[-3:])
        acc_gap = recent_train_acc - recent_val_acc
        self.overfitting_signals['acc_gap'] = acc_gap > 0.1
        
        # 3. 注意力坍塌检测
        if len(self.attention_entropies) >= 3:
            recent_entropy = np.mean(self.attention_entropies[-3:])
            self.overfitting_signals['attention_collapse'] = recent_entropy < 0.5
        
        # 4. 路径复杂度检测
        if len(self.path_complexities) >= 3:
            recent_complexity = np.mean(self.path_complexities[-3:])
            self.overfitting_signals['path_complexity'] = recent_complexity > 5.0
    
    def should_early_stop(self):
        """判断是否应该早停"""
        # 如果多个信号同时触发，建议早停
        signal_count = sum(self.overfitting_signals.values())
        return signal_count >= 2 or self.early_stop_counter >= self.patience
    
    def get_report(self):
        """生成过拟合分析报告"""
        if len(self.train_losses) == 0:
            return "暂无数据"
        
        report = []
        report.append("=== Route0过拟合分析报告 ===")
        
        # 当前状态
        if len(self.train_losses) > 0:
            report.append(f"当前训练损失: {self.train_losses[-1]:.4f}")
            report.append(f"当前验证损失: {self.val_losses[-1]:.4f}")
            report.append(f"损失差距: {self.val_losses[-1] - self.train_losses[-1]:.4f}")
        
        if len(self.train_accs) > 0:
            report.append(f"当前训练准确率: {self.train_accs[-1]:.4f}")
            report.append(f"当前验证准确率: {self.val_accs[-1]:.4f}")
            report.append(f"准确率差距: {self.train_accs[-1] - self.val_accs[-1]:.4f}")
        
        # 过拟合信号
        report.append("\n过拟合信号检测:")
        for signal, triggered in self.overfitting_signals.items():
            status = "🔴 触发" if triggered else "🟢 正常"
            report.append(f"  {signal}: {status}")
        
        # 建议
        signal_count = sum(self.overfitting_signals.values())
        if signal_count == 0:
            report.append("\n建议: 训练状态良好，继续训练")
        elif signal_count == 1:
            report.append("\n建议: 出现轻微过拟合信号，密切监控")
        else:
            report.append("\n建议: 多个过拟合信号触发，考虑早停或调整超参数")
        
        return "\n".join(report)
    
    def plot_analysis(self, save_path=None):
        """绘制过拟合分析图表"""
        if len(self.train_losses) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Route0过拟合分析', fontsize=16)
        
        epochs = range(len(self.train_losses))
        
        # 1. 损失对比
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_title('损失对比')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 准确率对比
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_title('准确率对比')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失差距
        if len(self.train_losses) > 0:
            loss_gaps = [v - t for v, t in zip(self.val_losses, self.train_losses)]
            axes[0, 2].plot(epochs, loss_gaps, 'g-', linewidth=2)
            axes[0, 2].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='过拟合阈值')
            axes[0, 2].set_title('验证-训练损失差距')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss Gap')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 准确率差距
        if len(self.train_accs) > 0:
            acc_gaps = [t - v for t, v in zip(self.train_accs, self.val_accs)]
            axes[1, 0].plot(epochs, acc_gaps, 'orange', linewidth=2)
            axes[1, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='过拟合阈值')
            axes[1, 0].set_title('训练-验证准确率差距')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy Gap')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 注意力熵
        if len(self.attention_entropies) > 0:
            axes[1, 1].plot(range(len(self.attention_entropies)), self.attention_entropies, 'purple', linewidth=2)
            axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='坍塌阈值')
            axes[1, 1].set_title('注意力熵')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Attention Entropy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '暂无注意力数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('注意力熵')
        
        # 6. 路径复杂度
        if len(self.path_complexities) > 0:
            axes[1, 2].plot(range(len(self.path_complexities)), self.path_complexities, 'brown', linewidth=2)
            axes[1, 2].axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='复杂度阈值')
            axes[1, 2].set_title('路径复杂度')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Path Complexity')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, '暂无路径数据', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('路径复杂度')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"过拟合分析图表已保存到: {save_path}")
        
        return fig

# 路径感知GNN模型
class PathAttentionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, num_heads=4, dropout=0.2):
        super(PathAttentionGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 节点类型特定的变换
        self.question_transform = nn.Linear(in_dim, hidden_dim)
        self.entity_transform = nn.Linear(in_dim, hidden_dim)
        self.context_transform = nn.Linear(in_dim, hidden_dim)
        
        # 多头注意力层
        self.attention_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim // num_heads),  # 查询变换
                nn.Linear(hidden_dim, hidden_dim // num_heads),  # 键变换
                nn.Linear(hidden_dim, hidden_dim // num_heads)   # 值变换
            ]) for _ in range(num_heads)
        ])
        
        # 边类型特定的变换 - 修改为适应多头注意力的维度
        head_dim = hidden_dim // num_heads
        self.edge_transforms = nn.ModuleDict({
            'answers': nn.Linear(head_dim, head_dim),
            'evidencedBy': nn.Linear(head_dim, head_dim),
            'supportsAnswer': nn.Linear(head_dim, head_dim),
            'relatedTo': nn.Linear(head_dim, head_dim),
            'default': nn.Linear(head_dim, head_dim)
        })
        
        # 注意力输出转换
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        
        # 路径记忆模块
        self.path_memory = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 路径重要性评分
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 层间残差连接
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, 1)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, g, node_feats, edge_weights=None):
        # 根据节点类型应用不同的转换
        h = torch.zeros(node_feats.size(0), self.hidden_dim, device=node_feats.device)
        
        # 根据节点角色应用不同的变换
        for i, ntype in enumerate(g.ndata['role']):
            if ntype == 'question':
                h[i] = self.question_transform(node_feats[i])
            elif ntype in ['evidence', 'answer']:
                h[i] = self.entity_transform(node_feats[i])
            else:  # context
                h[i] = self.context_transform(node_feats[i])
        
        # 初始化特征
        h = F.relu(h)
        
        # 保存所有节点的路径记忆 - 修复：使用detach()避免就地操作
        path_memories = h.detach().clone()
        
        # 保存每层的节点表示用于残差连接
        previous_h = h
        
        # 边缘注意力分数
        edge_attention_scores = {}
        
        # 消息传递
        for layer_idx in range(self.num_layers):
            # 多头注意力机制
            multi_head_out = []
            
            for head_idx in range(self.num_heads):
                q_transform, k_transform, v_transform = self.attention_heads[head_idx]
                
                # 计算查询、键、值
                queries = q_transform(h)
                keys = k_transform(h) 
                values = v_transform(h)
                
                # 注意力分数矩阵 (使用src-dst边信息)
                edge_src, edge_dst = g.edges()
                
                # 初始化消息 - 修改为适应多头注意力的维度
                head_messages = torch.zeros(h.size(0), self.hidden_dim // self.num_heads, device=h.device)
                
                # 对每条边计算注意力
                for i in range(len(edge_src)):
                    src, dst = edge_src[i], edge_dst[i]
                    
                    # 计算注意力分数
                    attn_score = torch.sum(queries[dst] * keys[src]) / np.sqrt(self.hidden_dim // self.num_heads)
                    
                    # 获取边类型并应用特定变换
                    edge_type = g.edata['rel'][i] if 'rel' in g.edata else 'default'
                    if isinstance(edge_type, torch.Tensor):
                        edge_type = 'default'  # 处理张量情况
                    
                    # 获取对应的边变换
                    if edge_type in self.edge_transforms:
                        transform = self.edge_transforms[edge_type]
                    else:
                        transform = self.edge_transforms['default']
                    
                    # 应用注意力和边权重
                    edge_weight = edge_weights[i] if edge_weights is not None else 1.0
                    attn_weight = F.softmax(attn_score, dim=0) * edge_weight
                    
                    # 保存注意力分数用于路径分析
                    edge_key = (src.item(), dst.item())
                    if edge_key not in edge_attention_scores:
                        edge_attention_scores[edge_key] = []
                    edge_attention_scores[edge_key].append(attn_weight.item())
                    
                    # 计算消息
                    message = transform(values[src]) * attn_weight
                    head_messages[dst] += message
                
                multi_head_out.append(head_messages)
            
            # 合并多头输出
            if len(multi_head_out) > 0:
                combined_messages = torch.cat(multi_head_out, dim=-1)
            else:
                combined_messages = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
            
            # 修复：避免就地操作，创建新的路径记忆张量
            new_path_memories = torch.zeros_like(path_memories)
            for i in range(len(h)):
                # 确保输入维度匹配
                if combined_messages.size(-1) == self.hidden_dim:
                    new_path_memories[i] = self.path_memory(combined_messages[i], path_memories[i])
                else:
                    # 如果维度不匹配，使用线性变换调整
                    adjusted_input = self.attention_output(combined_messages[i])
                    new_path_memories[i] = self.path_memory(adjusted_input, path_memories[i])
            
            # 更新路径记忆
            path_memories = new_path_memories
            
            # 最终的节点更新 (使用残差连接)
            h = self.layer_norms[layer_idx](previous_h + self.attention_output(combined_messages))
            h = self.dropout_layer(h)
            previous_h = h
        
        # 合并节点特征和路径记忆
        final_repr = h + path_memories
        
        return final_repr
    
    def forward_with_attention(self, g, node_feats, edge_weights=None):
        """前向传播并返回注意力权重，用于过拟合监控"""
        # 根据节点类型应用不同的转换
        h = torch.zeros(node_feats.size(0), self.hidden_dim, device=node_feats.device)
        
        # 根据节点角色应用不同的变换
        for i, ntype in enumerate(g.ndata['role']):
            if ntype == 'question':
                h[i] = self.question_transform(node_feats[i])
            elif ntype in ['evidence', 'answer']:
                h[i] = self.entity_transform(node_feats[i])
            else:  # context
                h[i] = self.context_transform(node_feats[i])
        
        # 初始化特征
        h = F.relu(h)
        
        # 保存所有节点的路径记忆
        path_memories = h.detach().clone()
        
        # 保存每层的节点表示用于残差连接
        previous_h = h
        
        # 边缘注意力分数 - 用于监控
        edge_attention_scores = {}
        
        # 消息传递
        for layer_idx in range(self.num_layers):
            # 多头注意力机制
            multi_head_out = []
            
            for head_idx in range(self.num_heads):
                q_transform, k_transform, v_transform = self.attention_heads[head_idx]
                
                # 计算查询、键、值
                queries = q_transform(h)
                keys = k_transform(h) 
                values = v_transform(h)
                
                # 注意力分数矩阵 (使用src-dst边信息)
                edge_src, edge_dst = g.edges()
                
                # 初始化消息
                head_messages = torch.zeros(h.size(0), self.hidden_dim // self.num_heads, device=h.device)
                
                # 对每条边计算注意力
                for i in range(len(edge_src)):
                    src, dst = edge_src[i], edge_dst[i]
                    
                    # 计算注意力分数
                    attn_score = torch.sum(queries[dst] * keys[src]) / np.sqrt(self.hidden_dim // self.num_heads)
                    
                    # 获取边类型并应用特定变换
                    edge_type = g.edata['rel'][i] if 'rel' in g.edata else 'default'
                    if isinstance(edge_type, torch.Tensor):
                        edge_type = 'default'
                    
                    # 获取对应的边变换
                    if edge_type in self.edge_transforms:
                        transform = self.edge_transforms[edge_type]
                    else:
                        transform = self.edge_transforms['default']
                    
                    # 应用注意力和边权重
                    edge_weight = edge_weights[i] if edge_weights is not None else 1.0
                    attn_weight = F.softmax(attn_score, dim=0) * edge_weight
                    
                    # 保存注意力分数用于监控
                    edge_key = (src.item(), dst.item())
                    if edge_key not in edge_attention_scores:
                        edge_attention_scores[edge_key] = []
                    edge_attention_scores[edge_key].append(attn_weight.item())
                    
                    # 计算消息
                    message = transform(values[src]) * attn_weight
                    head_messages[dst] += message
                
                multi_head_out.append(head_messages)
            
            # 合并多头输出
            if len(multi_head_out) > 0:
                combined_messages = torch.cat(multi_head_out, dim=-1)
            else:
                combined_messages = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
            
            # 更新路径记忆
            new_path_memories = torch.zeros_like(path_memories)
            for i in range(len(h)):
                if combined_messages.size(-1) == self.hidden_dim:
                    new_path_memories[i] = self.path_memory(combined_messages[i], path_memories[i])
                else:
                    adjusted_input = self.attention_output(combined_messages[i])
                    new_path_memories[i] = self.path_memory(adjusted_input, path_memories[i])
            
            path_memories = new_path_memories
            
            # 最终的节点更新 (使用残差连接)
            h = self.layer_norms[layer_idx](previous_h + self.attention_output(combined_messages))
            h = self.dropout_layer(h)
            previous_h = h
        
        # 合并节点特征和路径记忆
        final_repr = h + path_memories
        
        return final_repr, edge_attention_scores
    
    def _extract_paths(self, g, edge_attention_scores, question_idx, max_paths=5, max_length=3):
        """提取最可能的路径"""
        # 建立有向图
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(g.number_of_nodes()):
            G.add_node(i)
        
        # 添加带权重的边
        edge_src, edge_dst = g.edges()
        for i in range(len(edge_src)):
            src, dst = edge_src[i].item(), edge_dst[i].item()
            edge_key = (src, dst)
            # 使用平均注意力分数作为边权重
            weight = np.mean(edge_attention_scores.get(edge_key, [0.01]))
            G.add_edge(src, dst, weight=weight)
        
        # 寻找从问题节点到其他节点的最可能路径
        if question_idx == -1 and g.number_of_nodes() > 0:
            question_idx = 0
            
        # 路径提取
        paths = []
        for target in range(g.number_of_nodes()):
            if target != question_idx:
                try:
                    # 找到最短路径
                    shortest_path = nx.shortest_path(G, question_idx, target, weight='weight')
                    if len(shortest_path) <= max_length:
                        # 计算路径总权重
                        path_weight = sum(G[shortest_path[i]][shortest_path[i+1]]['weight'] 
                                         for i in range(len(shortest_path)-1))
                        paths.append((shortest_path, path_weight))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        # 按照路径权重排序
        paths.sort(key=lambda x: x[1], reverse=True)
        return paths[:max_paths]
    
    def compute_answer_scores(self, g, node_feats, edge_weights, question_idx, candidate_idxs):
        """计算所有候选答案的分数"""
        # 获取所有节点的表示
        node_embeddings = self.forward(g, node_feats, edge_weights)
        
        # 获取问题节点的表示
        q_embedding = node_embeddings[question_idx]
        
        # 计算每个候选答案的分数
        scores = torch.zeros(len(candidate_idxs), device=node_feats.device)
        for i, ans_idx in enumerate(candidate_idxs):
            # 计算问题和答案表示之间的余弦相似度
            ans_embedding = node_embeddings[ans_idx]
            sim = F.cosine_similarity(q_embedding.unsqueeze(0), ans_embedding.unsqueeze(0))
            scores[i] = sim
        
        return scores
    
    def get_edge_predictions(self, g, node_feats, edge_weights):
        """预测边的重要性，用于计算AUPRC"""
        # 获取所有节点的表示
        node_embeddings = self.forward(g, node_feats, edge_weights)
        
        # 获取边的源和目标
        edge_src, edge_dst = g.edges()
        
        # 计算预测的边重要性
        pred_edge_weights = torch.zeros(len(edge_src), device=node_feats.device)
        for i in range(len(edge_src)):
            src, dst = edge_src[i], edge_dst[i]
            
            # 计算源节点和目标节点之间的相似度
            src_embed = node_embeddings[src]
            dst_embed = node_embeddings[dst]
            
            # 使用点积作为相似度
            sim = torch.dot(src_embed, dst_embed)
            pred_edge_weights[i] = sim
        
        return pred_edge_weights

# 图数据集
class GraphDataset(Dataset):
    def __init__(self, graph_dir, graph_files=None, max_nodes=100):
        self.graph_dir = graph_dir
        self.max_nodes = max_nodes
        
        # 加载图文件列表
        if graph_files is None:
            self.graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.json')]
        else:
            self.graph_files = graph_files
        
        print(f"加载了 {len(self.graph_files)} 个图文件")
    
    def __len__(self):
        return len(self.graph_files)
    
    def __getitem__(self, idx):
        graph_file = os.path.join(self.graph_dir, self.graph_files[idx])
        
        # 加载图数据
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 构建DGL图
        g, node_feats, edge_weights, meta_info = self.build_graph(graph_data)
        
        return {
            'graph': g,
            'node_feats': node_feats,
            'edge_weights': edge_weights,
            'question_idx': meta_info['question_idx'],
            'answer_idx': meta_info['answer_idx'],
            'candidate_idxs': meta_info['candidate_idxs'],
            'graph_id': graph_data.get('id', self.graph_files[idx])
        }
    
    def build_graph(self, graph_data):
        """将JSON图转换为DGL图"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # 限制节点数量
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
        
        # 创建节点特征
        node_feats = []
        node_roles = []
        
        # 记录问题和答案节点的索引
        question_idx = -1
        answer_idx = -1
        candidate_idxs = []
        
        # 处理节点
        for i, node in enumerate(nodes):
            # 节点特征
            if 'feat' in node and node['feat'] != 'PLACEHOLDER':
                try:
                    feat = torch.tensor(node['feat'], dtype=torch.float)
                    if feat.shape[0] < 768:
                        feat = torch.cat([feat, torch.zeros(768-feat.shape[0])])
                    elif feat.shape[0] > 768:
                        feat = feat[:768]
                except:
                    feat = torch.randn(768)
            else:
                feat = torch.randn(768)
            
            node_feats.append(feat)
            
            # 节点角色
            role = node.get('role', 'context')
            node_roles.append(role)
            
            # 记录特殊节点
            if role == 'question':
                question_idx = i
            elif role == 'answer':
                answer_idx = i
                candidate_idxs.append(i)
            elif role == 'evidence':
                # 证据节点也可能是候选答案
                candidate_idxs.append(i)
        
        # 如果没有找到问题节点，使用第一个节点
        if question_idx == -1 and len(nodes) > 0:
            question_idx = 0
        
        # 如果没有找到答案节点，使用最后一个节点
        if answer_idx == -1 and len(nodes) > 0:
            answer_idx = len(nodes) - 1
            candidate_idxs.append(answer_idx)
        
        # 如果没有候选答案，使用所有非问题节点
        if not candidate_idxs:
            candidate_idxs = [i for i in range(len(nodes)) if i != question_idx]
        
        # 转换为张量
        node_feats = torch.stack(node_feats)
        
        # 处理边
        src_ids = []
        dst_ids = []
        edge_types = []
        edge_weights = []
        ground_truth = []  # 用于边的分类，标记重要边
        
        for edge in edges:
            src = edge.get('src', '').replace('n', '')
            dst = edge.get('dst', '').replace('n', '')
            
            # 检查边的有效性
            if not src.isdigit() or not dst.isdigit():
                continue
            
            src_id, dst_id = int(src), int(dst)
            
            # 确保节点索引有效
            if src_id >= len(nodes) or dst_id >= len(nodes):
                continue
            
            # 添加边
            src_ids.append(src_id)
            dst_ids.append(dst_id)
            
            # 边类型
            rel = edge.get('rel', 'default')
            edge_types.append(rel)
            
            # 边权重（如果有）
            weight = edge.get('weight', 1.0)
            edge_weights.append(weight)
            
            # 标记重要边（连接答案节点或证据节点的边）
            is_important = (src_id == answer_idx) or (dst_id == answer_idx) or \
                           (nodes[src_id].get('role') == 'evidence') or \
                           (nodes[dst_id].get('role') == 'evidence')
            ground_truth.append(float(is_important))
        
        # 创建DGL图
        g = dgl.graph((src_ids, dst_ids), num_nodes=len(nodes))
        
        # 根据实际添加到图中的节点过滤node_roles
        valid_node_indices = set(range(g.number_of_nodes()))
        filtered_node_roles = [node_roles[i] for i in valid_node_indices]
        
        # 将角色映射为数值
        role_map = {'question': 0, 'context': 1, 'answer': 2, 'evidence': 3, 'distractor': 4}
        numeric_roles = [role_map.get(role, 0) for role in filtered_node_roles] 
        g.ndata['role'] = torch.tensor(numeric_roles)
        
        # 边类型转为数字ID
        if edge_types:
            edge_type_set = list(sorted(set(edge_types)))
            edge_type_map = {etype: idx for idx, etype in enumerate(edge_type_set)}
            numeric_edge_types = [edge_type_map[etype] for etype in edge_types]
            g.edata['rel'] = torch.tensor(numeric_edge_types, dtype=torch.long)
        
        if edge_weights:
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_weights = torch.ones(g.number_of_edges(), dtype=torch.float)
        
        # 添加边的ground truth (用于计算AUPRC)
        g.edata['ground_truth'] = torch.tensor(ground_truth, dtype=torch.float)
        
        # 元信息
        meta_info = {
            'question_idx': question_idx,
            'answer_idx': answer_idx,
            'candidate_idxs': candidate_idxs
        }
        
        return g, node_feats, edge_weights, meta_info

def collate_fn(samples):
    graphs = [s['graph'] for s in samples]
    
    # 检查图是否为异构图
    is_hetero = False
    if graphs and hasattr(graphs[0], 'ntypes') and len(graphs[0].ntypes) > 1:
        is_hetero = True
    
    if is_hetero:
        # 对于异构图，不使用batch操作，直接保存图列表
        batched_graphs = graphs
    else:
        # 对于同构图，使用dgl.batch
        batched_graphs = dgl.batch(graphs)
        
    node_feats = torch.cat([s['node_feats'] for s in samples], dim=0)
    edge_weights = torch.cat([s['edge_weights'] for s in samples], dim=0)
    question_idx = torch.tensor([s['question_idx'] for s in samples], dtype=torch.long)
    answer_idx = torch.tensor([s['answer_idx'] for s in samples], dtype=torch.long)
    candidate_idxs = [s['candidate_idxs'] for s in samples]
    graph_id = [s['graph_id'] for s in samples]
    
    return {
        'graph': batched_graphs,
        'node_feats': node_feats,
        'edge_weights': edge_weights,
        'question_idx': question_idx,
        'answer_idx': answer_idx,
        'candidate_idxs': candidate_idxs,
        'graph_id': graph_id
    }

def calculate_node_recall_at_k(model, data_loader, device, k=20):
    """计算Node Recall@k指标"""
    model.eval()
    total_recall = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算Node Recall@k"):
            g = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            edge_weights = batch['edge_weights'].to(device)
            question_idx = batch['question_idx'].to(device)
            answer_idx = batch['answer_idx'].to(device)
            
            # 获取批量大小 - 避免使用len(g)
            batch_size = g.batch_size if hasattr(g, 'batch_size') else len(batch['question_idx'])
            
            # 检查是否为异构图
            is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
            
            # 处理每个样本
            for i in range(batch_size):
                # 获取当前图和特征
                if is_hetero:
                    current_g = g[i] if isinstance(g, list) else g
                    current_feats = node_feats[i]
                    current_weights = edge_weights[i] if edge_weights.dim() > 1 else edge_weights
                    q_idx = question_idx[i]
                    a_idx = answer_idx[i]
                else:
                    # 对于同构图，使用批处理索引
                    current_g = g
                    current_feats = node_feats
                    current_weights = edge_weights
                    q_idx = question_idx[i]
                    a_idx = answer_idx[i]
                
                # 获取节点嵌入
                node_embeddings = model.forward(current_g, current_feats, current_weights)
                
                # 获取问题节点嵌入
                q_embedding = node_embeddings[q_idx]
                
                # 计算问题节点与所有节点的相似度
                similarities = []
                q_idx_scalar = q_idx.item()  # 转为标量用于比较
                for j in range(len(node_embeddings)):
                    if j != q_idx_scalar:  # 排除问题节点自身
                        sim = F.cosine_similarity(q_embedding.unsqueeze(0), node_embeddings[j].unsqueeze(0))
                        similarities.append((j, sim))
                
                # 按相似度排序并取top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_k_nodes = [item[0] for item in similarities[:k]]
                
                # 检查答案节点是否在top-k中
                a_idx_scalar = a_idx.item()  # 转为标量用于比较
                if a_idx_scalar in top_k_nodes:
                    total_recall += 1
                
                count += 1
    
    return total_recall / count if count > 0 else 0


def calculate_auprc(model, data_loader, device):
    """计算边分类的AUPRC"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            g = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            edge_weights = batch['edge_weights'].to(device)
            
            # 获取边的预测权重
            pred_weights = model.get_edge_predictions(g, node_feats, edge_weights)
            
            # 获取边的真实标签
            true_labels = g.edata['ground_truth']
            
            # 收集预测和标签
            all_preds.extend(pred_weights.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    # 计算PR曲线和AUPRC
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auprc = auc(recall, precision)
    
    return auprc, precision, recall

def train(model, train_loader, optimizer, device, epoch, writer):
    """训练模型一个epoch并返回损失和指标"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f"Epoch {epoch+1}", leave=True)
    
    for batch_idx, batch in progress_bar:
        g = batch['graph'].to(device)
        node_feats = batch['node_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device)
        question_idx = batch['question_idx'].to(device)
        answer_idx = batch['answer_idx'].to(device)
        candidate_idxs = batch['candidate_idxs']
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 构建候选答案的一维张量
        candidate_tensor = []
        for candidates in candidate_idxs:
            candidate_tensor.extend(candidates)
        candidate_tensor = torch.tensor(candidate_tensor, device=device)
        
        # 计算候选答案的分数
        batch_scores = []
        batch_labels = []
        
        # 获取批量大小 - 避免使用len(g)
        batch_size = g.batch_size if hasattr(g, 'batch_size') else len(candidate_idxs)
        
        # 检查是否为异构图
        is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
        
        for i in range(batch_size):
            # 获取当前图的候选答案索引
            candidates = candidate_idxs[i]
            
            # 计算每个候选答案的分数
            if is_hetero:
                # 对于异构图，我们需要单独处理每个图
                current_g = g[i] if isinstance(g, list) else g
                current_feats = node_feats[i]
                current_weights = edge_weights[i] if edge_weights.dim() > 1 else edge_weights
                current_q_idx = question_idx[i]
            else:
                # 对于同构图，使用批处理索引
                current_g = g
                current_feats = node_feats
                current_weights = edge_weights
                current_q_idx = question_idx[i]
            
            scores = model.compute_answer_scores(
                current_g, current_feats, current_weights, 
                current_q_idx, candidates
            )
            
            # 创建标签，正确的答案标记为1，其他标记为0
            labels = torch.zeros_like(scores)
            for j, c_idx in enumerate(candidates):
                if c_idx == answer_idx[i].item():
                    labels[j] = 1
                    break
            
            batch_scores.append(scores)
            batch_labels.append(labels)
        
        # 计算损失
        scores = torch.cat(batch_scores)
        labels = torch.cat(batch_labels)
        
        # 使用BCE loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率
        pred = (scores > 0.5).float()
        current_correct = (pred == labels).sum().item()
        correct += current_correct
        current_total = len(scores)
        total += current_total
        
        # 记录Loss到TensorBoard
        writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # 更新进度条信息
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{current_correct/current_total:.4f}" if current_total > 0 else "N/A"
        })
    
    # 计算边分类的AUPRC
    print("计算训练集AUPRC...")
    auprc, precision, recall = calculate_auprc(model, train_loader, device)
    
    # 计算Node Recall@20
    print("计算训练集Node Recall@20...")
    node_recall = calculate_node_recall_at_k(model, train_loader, device, k=20)
    
    # 绘制PR曲线并保存
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Epoch {epoch})')
    plt.legend()
    
    # 保存PR曲线到TensorBoard
    writer.add_figure('Training/PR_Curve', plt.gcf(), epoch)
    
    # 记录AUPRC和Node Recall到TensorBoard
    writer.add_scalar('Training/AUPRC', auprc, epoch)
    writer.add_scalar('Training/NodeRecall@20', node_recall, epoch)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    writer.add_scalar('Training/Accuracy', accuracy, epoch)
    
    print(f"训练: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, AUPRC: {auprc:.4f}, Recall@20: {node_recall:.4f}")
    
    return avg_loss, accuracy, auprc, node_recall

def validate(model, val_loader, device, epoch, writer):
    """验证模型并返回损失和指标"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证", leave=False)
        for batch in progress_bar:
            g = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            edge_weights = batch['edge_weights'].to(device)
            question_idx = batch['question_idx'].to(device)
            answer_idx = batch['answer_idx'].to(device)
            candidate_idxs = batch['candidate_idxs']
            
            # 计算候选答案的分数
            batch_scores = []
            batch_labels = []
            
            # 获取批量大小 - 避免使用len(g)
            batch_size = g.batch_size if hasattr(g, 'batch_size') else len(candidate_idxs)
            
            # 检查是否为异构图
            is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
            
            for i in range(batch_size):
                # 获取当前图的候选答案索引
                candidates = candidate_idxs[i]
                
                # 计算每个候选答案的分数
                if is_hetero:
                    # 对于异构图，我们需要单独处理每个图
                    current_g = g[i] if isinstance(g, list) else g
                    current_feats = node_feats[i]
                    current_weights = edge_weights[i] if edge_weights.dim() > 1 else edge_weights
                    current_q_idx = question_idx[i]
                else:
                    # 对于同构图，使用批处理索引
                    current_g = g
                    current_feats = node_feats
                    current_weights = edge_weights
                    current_q_idx = question_idx[i]
                
                scores = model.compute_answer_scores(
                    current_g, current_feats, current_weights, 
                    current_q_idx, candidates
                )
                
                # 创建标签，正确的答案标记为1，其他标记为0
                labels = torch.zeros_like(scores)
                for j, c_idx in enumerate(candidates):
                    if c_idx == answer_idx[i].item():
                        labels[j] = 1
                        break
                
                batch_scores.append(scores)
                batch_labels.append(labels)
            
            # 计算损失
            scores = torch.cat(batch_scores)
            labels = torch.cat(batch_labels)
            
            # 使用BCE loss
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            pred = (scores > 0.5).float()
            correct += (pred == labels).sum().item()
            total += len(scores)
            
            # 收集预测和标签
            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # 计算边分类的AUPRC和PR曲线
    print("计算验证集AUPRC...")
    auprc, precision, recall = calculate_auprc(model, val_loader, device)
    
    # 计算Node Recall@20
    print("计算验证集Node Recall@20...")
    node_recall = calculate_node_recall_at_k(model, val_loader, device, k=20)
    
    # 绘制验证集的PR曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Validation Precision-Recall Curve (Epoch {epoch})')
    plt.legend()
    
    # 保存PR曲线到TensorBoard
    writer.add_figure('Validation/PR_Curve', plt.gcf(), epoch)
    
    # 记录验证集指标到TensorBoard
    writer.add_scalar('Validation/Loss', total_loss / len(val_loader), epoch)
    writer.add_scalar('Validation/AUPRC', auprc, epoch)
    writer.add_scalar('Validation/NodeRecall@20', node_recall, epoch)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    writer.add_scalar('Validation/F1', 2 * (auprc * node_recall) / (auprc + node_recall) if auprc + node_recall > 0 else 0, epoch)
    
    print(f"验证: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, AUPRC: {auprc:.4f}, Recall@20: {node_recall:.4f}")
    
    return avg_loss, accuracy, auprc, node_recall

def check_early_stopping(auprc_history, recall_history, em_f1_history, patience=5, threshold_auprc=0.30, threshold_recall=0.90):
    """检查是否应该早停"""
    # 如果历史记录不足，不能早停
    if len(auprc_history) < patience or len(recall_history) < patience or len(em_f1_history) < patience:
        return False
    
    # 条件1: AUPRC >= 0.30 且 Recall >= 0.90
    if auprc_history[-1] >= threshold_auprc and recall_history[-1] >= threshold_recall:
        return True
    
    # 条件2: 最近5个epoch三个指标均增幅 < 0.1pp
    recent_auprc = list(auprc_history)[-patience:]
    recent_recall = list(recall_history)[-patience:]
    recent_em_f1 = list(em_f1_history)[-patience:]
    
    auprc_improved = max(recent_auprc) - min(recent_auprc) < 0.001
    recall_improved = max(recent_recall) - min(recent_recall) < 0.001
    em_f1_improved = max(recent_em_f1) - min(recent_em_f1) < 0.001
    
    return auprc_improved and recall_improved and em_f1_improved

def main():
    parser = argparse.ArgumentParser(description='纯GNN多跳推理训练')
    parser.add_argument('--graph_dir', type=str, required=True, help='图数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='GNN层数 (6-8)')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载检查点')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录和检查点目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置TensorBoard
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    all_files = [f for f in os.listdir(args.graph_dir) if f.endswith('.json')]
    random.shuffle(all_files)
    
    # 划分训练集和验证集
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]
    
    train_dataset = GraphDataset(args.graph_dir, train_files)
    val_dataset = GraphDataset(args.graph_dir, val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = PathAttentionGNN(in_dim=768, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=4, dropout=args.dropout)
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建过拟合监控器
    overfitting_monitor = Route0OverfittingMonitor(patience=5, min_delta=0.001)
    
    # 加载检查点（如果有）
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从epoch {start_epoch}恢复训练...")
    
    # 训练循环
    print("开始训练...")
    best_val_auprc = 0
    patience_counter = 0
    
    # 历史指标记录用于早停
    auprc_history = deque(maxlen=10)
    recall_history = deque(maxlen=10)
    em_f1_history = deque(maxlen=10)
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc, train_auprc, train_recall = train(model, train_loader, optimizer, device, epoch, writer)
        
        # 验证
        val_loss, val_acc, val_auprc, val_recall = validate(model, val_loader, device, epoch, writer)
        
        # 计算EM/F1 (简化版，这里用val_acc代替)
        val_em_f1 = val_acc
        
        # 获取注意力权重用于过拟合监控
        attention_weights = None
        sample_g = None
        sample_feats = None
        sample_weights = None
        sample_q_idx = None
        
        try:
            # 从验证集获取一个样本用于注意力分析
            for batch in val_loader:
                sample_g = batch['graph'].to(device)
                sample_feats = batch['node_feats'].to(device)
                sample_weights = batch['edge_weights'].to(device)
                sample_q_idx = batch['question_idx'][0].item() if len(batch['question_idx']) > 0 else 0
                
                # 获取注意力权重
                with torch.no_grad():
                    _, attention_weights = model.forward_with_attention(sample_g, sample_feats, sample_weights)
                break
        except Exception as e:
            print(f"获取注意力权重时出错: {e}")
            attention_weights = None
        
        # 更新过拟合监控器
        overfitting_monitor.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            attention_weights=attention_weights,
            model=model,
            g=sample_g,
            node_feats=sample_feats,
            edge_weights=sample_weights,
            question_idx=sample_q_idx
        )
        
        # 生成过拟合分析报告
        print("\n" + overfitting_monitor.get_report())
        
        # 保存过拟合分析图表
        analysis_plot_path = os.path.join(args.output_dir, f'overfitting_analysis_epoch_{epoch}.png')
        overfitting_monitor.plot_analysis(save_path=analysis_plot_path)
        
        # 更新历史指标
        auprc_history.append(val_auprc)
        recall_history.append(val_recall)
        em_f1_history.append(val_em_f1)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUPRC: {train_auprc:.4f}, Recall@20: {train_recall:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUPRC: {val_auprc:.4f}, Recall@20: {val_recall:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auprc': val_auprc,
            'val_recall': val_recall,
        }, checkpoint_path)
        
        # 如果是最佳模型，保存为best_model.pt
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存最佳模型，AUPRC: {val_auprc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 检查过拟合监控器的早停建议
        if overfitting_monitor.should_early_stop():
            print(f"过拟合监控器建议早停，停止训练。")
            break
        
        # 检查早停条件
        if check_early_stopping(auprc_history, recall_history, em_f1_history):
            print(f"触发早停条件，停止训练。")
            break
        
        # 超过10轮没有改进，进行早停
        if patience_counter >= 10:
            print(f"超过10轮无改进，停止训练。")
            break
    
    # 生成最终的过拟合分析报告
    final_analysis_path = os.path.join(args.output_dir, 'final_overfitting_analysis.png')
    overfitting_monitor.plot_analysis(save_path=final_analysis_path)
    
    # 保存最终分析报告
    final_report_path = os.path.join(args.output_dir, 'final_overfitting_report.txt')
    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write(overfitting_monitor.get_report())
    
    print(f"\n最终过拟合分析报告已保存到: {final_report_path}")
    print(f"最终过拟合分析图表已保存到: {final_analysis_path}")
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("训练完成！")

if __name__ == "__main__":
    main() 