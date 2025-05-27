#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线3：差分解释器 - 训练脚本
通过对子图进行梯度分析，识别关键路径与关系
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
import collections
from sklearn.metrics import jaccard_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import load_graph_data

# 差分解释器模型
class DiffExplainer(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(DiffExplainer, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN层，包含注意力机制
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GNNLayer(hidden_dim, hidden_dim, dropout)
            )
        
        # 读出层，用于产生图表示
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 边重要性生成器
        self.edge_importance = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 目标函数历史记录
        self.objective_history = []
        self.edge_importance_history = []
        self.jaccard_history = []
    
    def forward(self, g, node_feats, calc_edge_importance=True):
        """
        前向传播
        
        参数:
        - g: DGL图
        - node_feats: 节点特征 [num_nodes, node_dim]
        - calc_edge_importance: 是否计算边重要性
        
        返回:
        - score: 图的分数（问题与答案的匹配分数）
        - node_embeds: 节点嵌入 [num_nodes, hidden_dim]
        - edge_importance: 边重要性 [num_edges]
        """
        # 编码节点特征
        h = self.node_encoder(node_feats)
        
        # 消息传递
        edge_attentions = []
        for layer in self.gnn_layers:
            h, attention = layer(g, h)
            edge_attentions.append(attention)
        
        # 读出层，产生图表示
        graph_score = self.readout(h.mean(dim=0, keepdim=True))
        
        # 计算边重要性（可选）
        edge_importance = None
        if calc_edge_importance:
            edge_importance = self._compute_edge_importance(g, h, edge_attentions)
        
        return graph_score.squeeze(0), h, edge_importance
    
    def _compute_edge_importance(self, g, node_embeds, edge_attentions):
        """计算边的重要性"""
        # 获取边的源和目标节点
        src, dst = g.edges()
        
        # 为每条边计算重要性
        edge_importance = []
        for i in range(len(src)):
            # 拼接源节点和目标节点的嵌入
            edge_feat = torch.cat([node_embeds[src[i]], node_embeds[dst[i]]], dim=0)
            
            # 计算边的重要性
            importance = self.edge_importance(edge_feat)
            edge_importance.append(importance)
        
        # 累加所有GNN层的注意力权重
        if edge_attentions:
            edge_attentions_sum = sum(edge_attentions) / len(edge_attentions)
            edge_importance = torch.cat(edge_importance) * edge_attentions_sum
        else:
            edge_importance = torch.cat(edge_importance)
        
        return edge_importance
    
    def explain_gradient_descent(self, g, node_feats, question_idx, answer_idx, 
                              lr=0.01, max_iter=1000, delta_threshold=1e-4, 
                              sparsity_weight=0.1, device='cuda'):
        """
        使用梯度下降方法优化边重要性，直到目标函数变化小于阈值
        
        参数:
        - g: DGL图
        - node_feats: 节点特征
        - question_idx: 问题节点索引
        - answer_idx: 答案节点索引
        - lr: 学习率
        - max_iter: 最大迭代次数
        - delta_threshold: 目标函数变化阈值
        - sparsity_weight: 稀疏性权重
        
        返回:
        - edge_mask: 最终的边掩码
        - objective_history: 目标函数历史
        - jaccard_history: 边掩码Jaccard相似度历史
        """
        # 初始化边掩码，全1
        src, dst = g.edges()
        edge_mask = torch.ones(len(src), requires_grad=True, device=device)
        
        # 优化器
        optimizer = optim.Adam([edge_mask], lr=lr)
        
        # 目标函数历史
        objective_history = []
        edge_masks_history = []
        
        # 梯度下降迭代
        prev_objective = float('inf')
        stable_count = 0  # 计数器，记录稳定的迭代次数
        
        for iter_idx in range(max_iter):
            # 前向传播，但使用边掩码调整图
            masked_g = g.clone()
            masked_g.edata['mask'] = edge_mask
            
            # 计算模型输出
            score, node_embeds, _ = self.forward(masked_g, node_feats, calc_edge_importance=False)
            
            # 提取问题和答案节点的嵌入
            q_embed = node_embeds[question_idx]
            a_embed = node_embeds[answer_idx]
            
            # 计算相似度
            sim = F.cosine_similarity(q_embed.unsqueeze(0), a_embed.unsqueeze(0))
            
            # 计算稀疏性惩罚
            sparsity = torch.mean(edge_mask)
            
            # 总目标函数: 最大化相似度，最小化边掩码数量
            # 注意：因为我们是最大化，所以前面的相似度项是正的
            objective = sim - sparsity_weight * sparsity
            
            # 反向传播
            optimizer.zero_grad()
            (-objective).backward()  # 最大化目标函数等于最小化其负值
            optimizer.step()
            
            # 限制边掩码在[0, 1]范围内
            with torch.no_grad():
                edge_mask.clamp_(0, 1)
            
            # 记录目标函数和边掩码
            current_objective = objective.item()
            objective_history.append(current_objective)
            edge_masks_history.append(edge_mask.detach().clone())
            
            # 计算目标函数变化
            delta_objective = abs(current_objective - prev_objective)
            prev_objective = current_objective
            
            # 检查收敛条件1: |ΔJ| < delta_threshold
            if delta_objective < delta_threshold:
                stable_count += 1
            else:
                stable_count = 0
            
            # 检查收敛条件2: 边掩码稳定（Jaccard > 0.95）
            if iter_idx > 0 and len(edge_masks_history) >= 2:
                # 二值化边掩码用于计算Jaccard
                current_binary = (edge_mask > 0.5).float()
                prev_binary = (edge_masks_history[-2] > 0.5).float()
                jaccard = torch.sum(current_binary * prev_binary) / torch.sum(torch.maximum(current_binary, prev_binary))
                self.jaccard_history.append(jaccard.item())
                
                # 如果Jaccard相似度高且目标函数稳定，提前终止
                if jaccard > 0.95 and stable_count >= 3:
                    print(f"收敛于迭代 {iter_idx+1}: Jaccard = {jaccard.item():.4f}, |ΔJ| = {delta_objective:.6f}")
                    break
            
            # 打印进度
            if (iter_idx + 1) % 10 == 0:
                print(f"迭代 {iter_idx+1}/{max_iter}, 目标函数: {current_objective:.4f}, |ΔJ|: {delta_objective:.6f}")
        
        # 保存历史记录
        self.objective_history = objective_history
        self.edge_importance_history = edge_masks_history
        
        # 返回最终边掩码和历史数据
        return edge_mask.detach(), objective_history, self.jaccard_history

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(GNNLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(in_dim * 2, 1),
            nn.LeakyReLU(0.2)
        )
        
        # 变换函数
        self.transform = nn.Linear(in_dim, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = nn.ReLU()
    
    def forward(self, g, h):
        """
        前向传播
        
        参数:
        - g: DGL图
        - h: 节点特征 [num_nodes, in_dim]
        
        返回:
        - h_new: 更新后的节点特征 [num_nodes, out_dim]
        - attention_weights: 注意力权重 [num_edges]
        """
        src, dst = g.edges()
        
        # 计算注意力权重
        attention_weights = []
        for i in range(len(src)):
            # 拼接源节点和目标节点的嵌入
            edge_feat = torch.cat([h[src[i]], h[dst[i]]], dim=0)
            
            # 计算注意力权重
            weight = self.attention(edge_feat)
            attention_weights.append(weight)
        
        # 对注意力权重进行softmax
        attention_weights = torch.cat(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # 存储节点的新特征
        h_new = torch.zeros_like(h)
        
        # 根据注意力权重聚合邻居信息
        for i, (s, d) in enumerate(zip(src, dst)):
            # 目标节点接收加权的源节点消息
            h_new[d] += h[s] * attention_weights[i]
        
        # 应用变换和激活函数
        h_new = self.transform(h_new)
        h_new = self.activation(h_new)
        h_new = self.dropout(h_new)
        
        # 残差连接
        h_new = h_new + h
        
        return h_new, attention_weights

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
        g, node_feats, question_idx, answer_idx, metadata = self.build_graph(graph_data)
        
        return {
            'graph': g,
            'node_feats': node_feats,
            'question_idx': question_idx,
            'answer_idx': answer_idx,
            'metadata': metadata,
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
        
        # 记录问题和答案节点的索引
        question_idx = -1
        answer_idx = -1
        
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
            
            # 记录特殊节点
            role = node.get('role', '')
            if role == 'question':
                question_idx = i
            elif role == 'answer':
                answer_idx = i
        
        # 如果没有找到问题节点，使用第一个节点
        if question_idx == -1 and len(nodes) > 0:
            question_idx = 0
        
        # 如果没有找到答案节点，使用最后一个节点
        if answer_idx == -1 and len(nodes) > 0:
            answer_idx = len(nodes) - 1
        
        # 转换为张量
        node_feats = torch.stack(node_feats)
        
        # 处理边
        src_ids = []
        dst_ids = []
        edge_types = []
        
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
        
        # 创建DGL图
        g = dgl.graph((src_ids, dst_ids), num_nodes=len(nodes))
        
        # 图级别元数据
        metadata = {
            'node_labels': [node.get('label', f'Node {i}') for i, node in enumerate(nodes)],
            'edge_types': edge_types,
            'edge_pairs': list(zip(src_ids, dst_ids))
        }
        
        return g, node_feats, question_idx, answer_idx, metadata

def collate_fn(samples):
    """收集函数，处理批次数据"""
    # 获取数据
    graphs = [s['graph'] for s in samples]
    
    # 检查图是否为异构图
    is_hetero = False
    if graphs and hasattr(graphs[0], 'ntypes') and len(graphs[0].ntypes) > 1:
        is_hetero = True
    
    if is_hetero:
        # 对于异构图，不使用batch操作，直接保存图列表
        batched_graph = graphs
    else:
        # 对于同构图，使用dgl.batch
        batched_graph = dgl.batch(graphs)
        
    node_feats = torch.cat([s['node_feats'] for s in samples], dim=0)
    question_idxs = torch.tensor([s['question_idx'] for s in samples], dtype=torch.long)
    answer_idxs = torch.tensor([s['answer_idx'] for s in samples], dtype=torch.long)
    metadata = [s['metadata'] for s in samples]
    graph_id = [s['graph_id'] for s in samples]
    return {
        'graph': batched_graph,
        'node_feats': node_feats,
        'question_idx': question_idxs,
        'answer_idx': answer_idxs,
        'metadata': metadata,
        'graph_id': graph_id
    }

# 训练函数
def train(model, train_loader, optimizer, device, epochs, grad_clip=1.0, output_dir="./output"):
    """训练模型"""
    model.train()
    losses = []
    accuracies = []
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # 创建可视化目录
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # 获取数据
            graphs = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            question_idxs = batch['question_idx'].to(device)
            answer_idxs = batch['answer_idx'].to(device)
            
            # 获取批量大小 - 避免使用len(graphs)
            # 使用question_idxs的长度作为批量大小，这个应该总是可用的
            batch_size = len(question_idxs)
            
            # 检查是否为异构图列表
            is_hetero = hasattr(graphs, 'ntypes') and len(graphs.ntypes) > 1
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 损失和正确预测计数
            batch_loss = 0
            batch_correct = 0
            
            # 每个图单独处理
            for i in range(batch_size):
                # 获取当前图
                if is_hetero:
                    # 对于异构图，可能需要单独处理每个图
                    g = graphs
                else:
                    # 对于批处理的同构图，获取第i个图
                    if isinstance(graphs, list):
                        g = graphs[i]
                    else:
                        g = graphs[i:i+1]  # 使用切片获取子图
                
                feats = node_feats[i]
                q_idx = question_idxs[i]
                a_idx = answer_idxs[i]
                
                # 前向传播
                score, node_embeds, _ = model(g, feats)
                
                # 计算对比损失：问题-答案对比其他节点
                q_embed = node_embeds[q_idx]
                a_embed = node_embeds[a_idx]
                
                # 问题和答案的余弦相似度
                pos_score = F.cosine_similarity(q_embed.unsqueeze(0), a_embed.unsqueeze(0))
                
                # 负样本：所有非答案节点
                neg_indices = [j for j in range(feats.size(0)) if j != a_idx.item()]
                
                if neg_indices:  # 确保有负样本
                    neg_embeds = node_embeds[neg_indices]
                    
                    # 问题和所有负样本节点的余弦相似度
                    neg_scores = F.cosine_similarity(q_embed.unsqueeze(0), neg_embeds)
                    
                    # 计算对比损失 (hinge loss)
                    margin = 0.5
                    hinge_loss = torch.mean(torch.clamp(neg_scores - pos_score + margin, min=0))
                    
                    # 累加到批损失
                    batch_loss += hinge_loss
                    
                    # 检查预测是否正确 (问题-答案相似度是否最高)
                    if pos_score > torch.max(neg_scores):
                        batch_correct += 1
                else:
                    # 没有负样本时认为预测正确
                    batch_correct += 1
            
            # 平均批损失
            batch_loss = batch_loss / batch_size
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # 更新权重
            optimizer.step()
            
            # 记录损失和准确率
            epoch_loss += batch_loss.item()
            correct += batch_correct
            total += batch_size
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{batch_loss.item():.4f}",
                'accuracy': f"{batch_correct/batch_size:.4f}"
            })
        
        # 计算整个epoch的平均损失和准确率
        epoch_loss /= len(train_loader)
        epoch_accuracy = correct / total
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        
        # 记录到TensorBoard
        writer.add_scalar('Training/Loss', epoch_loss, epoch)
        writer.add_scalar('Training/Accuracy', epoch_accuracy, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        # 每隔几个epoch进行一次详细样本解释和可视化
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # 从训练集中选择一个样本进行解释
            sample_batch = next(iter(train_loader))
            sample_graph = sample_batch['graph'][0].to(device)
            sample_feats = sample_batch['node_feats'][0].to(device)
            sample_q_idx = sample_batch['question_idx'][0].to(device)
            sample_a_idx = sample_batch['answer_idx'][0].to(device)
            
            # 使用梯度下降解释
            edge_mask, objective_history, jaccard_history = model.explain_gradient_descent(
                sample_graph, sample_feats, sample_q_idx, sample_a_idx, 
                device=device
            )
            
            # 绘制目标函数变化曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(objective_history)
            plt.xlabel('迭代次数')
            plt.ylabel('目标函数 J')
            plt.title(f'目标函数变化 (Epoch {epoch+1})')
            
            # 如果有足够的数据点，计算目标函数的变化率 ΔJ
            if len(objective_history) > 1:
                delta_j = [abs(objective_history[i] - objective_history[i-1]) for i in range(1, len(objective_history))]
                ax2 = plt.twinx()
                ax2.plot(range(1, len(objective_history)), delta_j, 'r--', alpha=0.7)
                ax2.set_ylabel('|ΔJ|', color='r')
                ax2.tick_params(axis='y', colors='r')
                ax2.axhline(y=1e-4, color='g', linestyle='--', label='|ΔJ| = 1e-4')
                ax2.legend(loc='upper right')
            
            # 绘制Jaccard相似度曲线
            if jaccard_history:
                plt.subplot(1, 2, 2)
                plt.plot(jaccard_history)
                plt.axhline(y=0.95, color='r', linestyle='--', label='Jaccard = 0.95')
                plt.xlabel('迭代次数')
                plt.ylabel('Jaccard相似度')
                plt.title('边掩码稳定性 (Jaccard)')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'explanation_epoch_{epoch+1}.png'))
            plt.close()
            
            # 添加到TensorBoard
            writer.add_figure('Explanations/ObjectiveJaccard', plt.gcf(), epoch)
    
    # 关闭TensorBoard记录器
    writer.close()
    
    return losses, accuracies

# 评估函数
def evaluate(model, val_loader, device, output_dir="./output"):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 保存解释结果
    explanations = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 获取数据
            graphs = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            question_idxs = batch['question_idx'].to(device)
            answer_idxs = batch['answer_idx'].to(device)
            metadata = batch['metadata']
            
            # 获取批量大小 - 避免使用len(graphs)
            # 使用question_idxs的长度作为批量大小，这个应该总是可用的
            batch_size = len(question_idxs)
            
            # 检查是否为异构图列表
            is_hetero = hasattr(graphs, 'ntypes') and len(graphs.ntypes) > 1
            
            # 每个图单独处理
            for i in range(batch_size):
                # 获取当前图
                if is_hetero:
                    # 对于异构图，可能需要单独处理每个图
                    g = graphs
                else:
                    # 对于批处理的同构图，获取第i个图
                    if isinstance(graphs, list):
                        g = graphs[i]
                    else:
                        g = graphs[i:i+1]  # 使用切片获取子图
                
                feats = node_feats[i]
                q_idx = question_idxs[i]
                a_idx = answer_idxs[i]
                meta = metadata[i]
                
                # 前向传播
                score, node_embeds, edge_importance = model(g, feats)
                
                # 计算对比损失
                q_embed = node_embeds[q_idx]
                a_embed = node_embeds[a_idx]
                
                # 问题和答案的余弦相似度
                pos_score = F.cosine_similarity(q_embed.unsqueeze(0), a_embed.unsqueeze(0))
                
                # 负样本：所有非答案节点
                neg_indices = [j for j in range(feats.size(0)) if j != a_idx.item()]
                
                if neg_indices:  # 确保有负样本
                    neg_embeds = node_embeds[neg_indices]
                    
                    # 问题和所有负样本节点的余弦相似度
                    neg_scores = F.cosine_similarity(q_embed.unsqueeze(0), neg_embeds)
                    
                    # 计算对比损失
                    margin = 0.5
                    hinge_loss = torch.mean(torch.clamp(neg_scores - pos_score + margin, min=0))
                    
                    # 累加到总损失
                    total_loss += hinge_loss.item()
                    
                    # 检查预测是否正确
                    if pos_score > torch.max(neg_scores):
                        correct += 1
                else:
                    # 没有负样本时认为预测正确
                    correct += 1
                
                # 收集解释结果
                if edge_importance is not None:
                    # 获取边的信息
                    edge_src, edge_dst = g.edges()
                    edge_types = meta['edge_types'] if 'edge_types' in meta else []
                    node_labels = meta['node_labels'] if 'node_labels' in meta else []
                    
                    # 确保edge_importance的长度与边数量相同
                    if len(edge_importance) != len(edge_src):
                        print(f"警告：边重要性长度 ({len(edge_importance)}) 与边数量 ({len(edge_src)}) 不匹配")
                        if len(edge_importance) > len(edge_src):
                            edge_importance = edge_importance[:len(edge_src)]
                        else:
                            # 扩展edge_importance
                            padding = torch.zeros(len(edge_src) - len(edge_importance), device=edge_importance.device)
                            edge_importance = torch.cat([edge_importance, padding])
                    
                    # 整理边的重要性信息
                    edges_info = []
                    for j in range(len(edge_src)):
                        src_idx = edge_src[j].item()
                        dst_idx = edge_dst[j].item()
                        edge_type = edge_types[j] if j < len(edge_types) else 'default'
                        imp = edge_importance[j].item()
                        
                        src_label = node_labels[src_idx] if src_idx < len(node_labels) else f"Node {src_idx}"
                        dst_label = node_labels[dst_idx] if dst_idx < len(node_labels) else f"Node {dst_idx}"
                        
                        edges_info.append({
                            'src_idx': src_idx,
                            'dst_idx': dst_idx,
                            'src_label': src_label,
                            'dst_label': dst_label,
                            'edge_type': edge_type,
                            'importance': imp
                        })
                    
                    # 排序并记录
                    edges_info.sort(key=lambda x: x['importance'], reverse=True)
                    explanations.append({
                        'graph_id': batch['graph_id'][i] if 'graph_id' in batch else f"graph_{i}",
                        'edges_info': edges_info,
                        'score': score.item()
                    })
            
            total += batch_size
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    print(f"Validation, Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 保存解释结果到文件
    if explanations:
        result_file = os.path.join(output_dir, 'explanations.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2)
        print(f"解释结果已保存到 {result_file}")
    
    return avg_loss, accuracy, explanations

# 分析边的重要性
def analyze_edge_importance(model, graph_data, device, output_dir="./output"):
    """分析图中边的重要性并可视化"""
    # 准备数据
    dataset = GraphDataset("", [])
    g, node_feats, question_idx, answer_idx, metadata = dataset.build_graph(graph_data)
    
    # 移动到设备
    g = g.to(device)
    node_feats = node_feats.to(device)
    
    # 使用梯度下降解释
    model.eval()
    edge_mask, objective_history, jaccard_history = model.explain_gradient_descent(
        g, node_feats, question_idx, answer_idx, device=device
    )
    
    # 绘制目标函数和Jaccard曲线
    plt.figure(figsize=(12, 10))
    
    # 目标函数曲线
    plt.subplot(2, 2, 1)
    plt.plot(objective_history)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数 J')
    plt.title('目标函数变化')
    
    # 目标函数变化率曲线
    plt.subplot(2, 2, 2)
    if len(objective_history) > 1:
        delta_j = [abs(objective_history[i] - objective_history[i-1]) for i in range(1, len(objective_history))]
        plt.plot(range(1, len(objective_history)), delta_j)
        plt.axhline(y=1e-4, color='r', linestyle='--', label='ΔJ = 1e-4')
        plt.yscale('log')  # 使用对数尺度更好地显示变化
        plt.xlabel('迭代次数')
        plt.ylabel('|ΔJ|')
        plt.title('目标函数变化率')
        plt.legend()
    
    # Jaccard相似度曲线
    plt.subplot(2, 2, 3)
    if jaccard_history:
        plt.plot(jaccard_history)
        plt.axhline(y=0.95, color='r', linestyle='--', label='Jaccard = 0.95')
        plt.xlabel('迭代次数')
        plt.ylabel('Jaccard相似度')
        plt.title('边掩码稳定性 (Jaccard)')
        plt.legend()
    
    # 边重要性分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(edge_mask.cpu().numpy(), bins=20, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='阈值 = 0.5')
    plt.xlabel('边重要性')
    plt.ylabel('频率')
    plt.title('边重要性分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_importance_analysis.png'))
    
    # 获取排序后的边重要性
    edge_info = []
    src, dst = g.edges()
    for i in range(len(src)):
        src_idx = src[i].item()
        dst_idx = dst[i].item()
        
        edge_info.append({
            'src': src_idx,
            'dst': dst_idx,
            'importance': edge_mask[i].item()
        })
    
    # 按重要性排序
    edge_info.sort(key=lambda x: x['importance'], reverse=True)
    
    # 转为DataFrame用于更好的展示
    edge_df = pd.DataFrame(edge_info)
    
    return edge_df, objective_history, jaccard_history

def main():
    parser = argparse.ArgumentParser(description='差分解释器训练')
    parser.add_argument('--graph_dir', type=str, required=True, help='子图目录')
    parser.add_argument('--model_dir', type=str, default='models/route3_diff_explainer/checkpoints', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='models/route3_diff_explainer/output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA加速训练')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 选择设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 加载数据集
    graph_files = [f for f in os.listdir(args.graph_dir) if f.endswith('.json')]
    np.random.shuffle(graph_files)
    
    # 划分训练集和验证集
    val_size = int(0.2 * len(graph_files))
    train_files = graph_files[val_size:]
    val_files = graph_files[:val_size]
    
    print(f'训练集: {len(train_files)}个图, 验证集: {len(val_files)}个图')
    
    # 创建数据加载器
    train_dataset = GraphDataset(args.graph_dir, train_files)
    val_dataset = GraphDataset(args.graph_dir, val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # 创建模型
    model = DiffExplainer(
        node_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练模型
    print("开始训练...")
    losses, accuracies = train(model, train_loader, optimizer, device, args.epochs)
    
    # 评估模型
    val_loss, val_accuracy, explanations = evaluate(model, val_loader, device)
    print(f"验证集 Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'training_losses': losses,
        'training_accuracies': accuracies
    }, model_path)
    
    print(f"训练完成，模型保存到 {model_path}")
    
    # 分析示例图的边重要性
    if len(val_dataset) > 0:
        print("\n分析边的重要性:")
        # 获取第一个样本
        sample_idx = 0
        graph_file = os.path.join(args.graph_dir, val_dataset.graph_files[sample_idx])
        
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 分析边的重要性
        edge_df, objective_history, jaccard_history = analyze_edge_importance(model, graph_data, device)
        
        # 打印最重要的边
        print(f"Graph Score: {objective_history[-1]:.4f}")
        for i, edge in enumerate(edge_df[:5]):
            print(f"Edge {i+1}: {edge['src_label']} -> {edge['dst_label']} ({edge['edge_type']}), Importance: {edge['importance']:.4f}")

if __name__ == '__main__':
    main() 