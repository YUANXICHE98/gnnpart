#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线4：多链思维 - 训练脚本
使用多链思维方法训练知识图谱推理
"""

import os
import sys
import json
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboardX import SummaryWriter
from collections import Counter
import math
import streamlit.components.v1 as components
import dgl

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import load_graph_data

# 多链思维模型
class MultiChainReasoner(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_reasoning_chains=20, chain_length=3):
        super(MultiChainReasoner, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_reasoning_chains = num_reasoning_chains
        self.chain_length = chain_length
        
        # 节点特征变换
        self.node_transform = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 链初始化网络（创建多个思维链的起点）
        self.chain_init = nn.Linear(hidden_dim, hidden_dim * num_reasoning_chains)
        
        # 链迭代器（每个链的推理步骤）
        self.chain_iterators = nn.ModuleList([
            ChainIterator(hidden_dim) for _ in range(num_reasoning_chains)
        ])
        
        # 注意力聚合器（结合多个链的结果）
        self.attention_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * num_reasoning_chains, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 最终预测层
        self.predictor = nn.Linear(hidden_dim, 1)
        
        # RankPrompt网络
        self.rank_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_feats, edges=None):
        """
        前向传播
        
        参数:
        - node_feats: 节点特征，形状为 [num_nodes, node_dim]
        - edges: 边信息，形状为 [num_edges, 2]，每行是 (src, dst)
        
        返回:
        - chain_outputs: 每个推理链的输出
        - final_output: 最终的聚合输出
        - attention_weights: 注意力权重
        """
        # 转换节点特征
        node_embeds = self.node_transform(node_feats)
        
        # 初始化思维链
        batch_size = node_feats.size(0)
        chain_states = self.chain_init(node_embeds).view(batch_size, self.num_reasoning_chains, self.hidden_dim)
        
        # 存储中间结果
        chain_outputs = []
        
        # 迭代每个链
        for i in range(self.num_reasoning_chains):
            chain_iterator = self.chain_iterators[i]
            chain_state = chain_states[:, i, :]
            
            # 在链中执行多步推理
            for _ in range(self.chain_length):
                chain_state = chain_iterator(chain_state, node_embeds, edges)
            
            chain_outputs.append(chain_state)
        
        # 将所有链的输出拼接
        concatenated = torch.cat(chain_outputs, dim=1)
        
        # 注意力聚合
        final_output = self.attention_aggregator(concatenated)
        
        # 计算链的注意力权重（可解释性）
        attention_weights = F.softmax(torch.matmul(final_output, torch.cat([co.unsqueeze(1) for co in chain_outputs], dim=1).squeeze(2)), dim=1)
        
        # 最终预测
        prediction = self.predictor(final_output)
        
        return chain_outputs, final_output, attention_weights, prediction
    
    def compute_chain_answers(self, node_feats, candidate_idxs, edges=None):
        """
        计算每个推理链对候选答案的预测
        
        参数:
        - node_feats: 节点特征 [num_nodes, node_dim]
        - candidate_idxs: 候选答案节点索引
        - edges: 边信息
        
        返回:
        - chain_predictions: 每个链对每个候选的预测分数
        - ensemble_prediction: 集成后的预测
        - entropy: 答案分布的熵
        - majority_ratio: 多数投票占比
        """
        # 前向传播，获取各链的输出
        chain_outputs, final_output, attention_weights, _ = self.forward(node_feats, edges)
        
        # 计算每个链对每个候选答案的相似度得分
        chain_predictions = []
        
        for chain_output in chain_outputs:
            # 计算链输出与每个候选答案的相似度
            candidate_scores = []
            for candidate_idx in candidate_idxs:
                candidate_embed = node_feats[candidate_idx]
                similarity = F.cosine_similarity(chain_output.unsqueeze(0), candidate_embed.unsqueeze(0))
                candidate_scores.append(similarity.item())
            
            # 归一化分数（可选）
            if candidate_scores:
                max_score = max(candidate_scores)
                min_score = min(candidate_scores)
                if max_score > min_score:
                    normalized_scores = [(s - min_score) / (max_score - min_score) for s in candidate_scores]
                    candidate_scores = normalized_scores
            
            chain_predictions.append(candidate_scores)
        
        # 计算整体预测（加权平均）
        ensemble_prediction = None
        if chain_predictions:
            # 使用注意力权重加权
            weights = attention_weights.squeeze(0).cpu().numpy()
            weighted_predictions = np.zeros_like(chain_predictions[0])
            
            for i, pred in enumerate(chain_predictions):
                weighted_predictions += np.array(pred) * weights[i]
            
            ensemble_prediction = weighted_predictions.tolist()
        
        # 计算最终答案的熵和多数投票比例
        entropy = 0
        majority_ratio = 0
        
        if chain_predictions:
            # 找出每个链预测的最高分答案
            best_answers = [np.argmax(pred) for pred in chain_predictions]
            
            # 统计答案频率
            answer_counts = Counter(best_answers)
            total_chains = len(best_answers)
            
            # 计算多数投票占比
            majority_answer = answer_counts.most_common(1)[0][0]
            majority_ratio = answer_counts[majority_answer] / total_chains
            
            # 计算熵：H(X) = -sum(p(x) * log(p(x)))
            entropy = 0
            for count in answer_counts.values():
                prob = count / total_chains
                entropy -= prob * math.log2(prob)
            
            # 归一化熵（除以最大可能熵）
            max_entropy = math.log2(len(candidate_idxs)) if len(candidate_idxs) > 0 else 0
            if max_entropy > 0:
                entropy /= max_entropy
        
        return chain_predictions, ensemble_prediction, entropy, majority_ratio
    
    def rank_prompt_compare(self, node_feats_a, node_feats_b, question_idx):
        """
        使用RankPrompt机制比较两个答案的质量
        
        参数:
        - node_feats_a: 第一个图的节点特征
        - node_feats_b: 第二个图的节点特征
        - question_idx: 问题节点索引
        
        返回:
        - win_rate: 第一个答案优于第二个的概率
        """
        # 获取问题特征
        q_embed_a = node_feats_a[question_idx]
        q_embed_b = node_feats_b[question_idx]
        
        # 获取图表示
        _, final_output_a, _, _ = self.forward(node_feats_a)
        _, final_output_b, _, _ = self.forward(node_feats_b)
        
        # 组合特征
        combined_a = torch.cat([q_embed_a, final_output_a], dim=0)
        combined_b = torch.cat([q_embed_b, final_output_b], dim=0)
        
        # 计算排名分数
        score_a = self.rank_network(combined_a)
        score_b = self.rank_network(combined_b)
        
        # 计算胜率
        diff = score_a - score_b
        win_rate = torch.sigmoid(diff).item()
        
        return win_rate

class ChainIterator(nn.Module):
    def __init__(self, hidden_dim):
        super(ChainIterator, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 注意力机制，关注相关节点
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 状态更新网络
        self.state_update = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, state, node_embeds, edges=None):
        """
        执行一步推理
        
        参数:
        - state: 当前链状态，形状为 [batch_size, hidden_dim]
        - node_embeds: 节点嵌入，形状为 [num_nodes, hidden_dim]
        - edges: 边信息，限制注意力的计算
        
        返回:
        - new_state: 更新后的链状态
        """
        batch_size = state.size(0)
        num_nodes = node_embeds.size(0)
        
        # 计算注意力分数
        # 将state扩展为 [batch_size, 1, hidden_dim]
        expanded_state = state.unsqueeze(1)
        # 将node_embeds扩展为 [1, num_nodes, hidden_dim]
        expanded_nodes = node_embeds.unsqueeze(0)
        
        # 计算节点与状态的交互特征 [batch_size, num_nodes, hidden_dim*2]
        interaction_feats = torch.cat([
            expanded_state.expand(-1, num_nodes, -1),
            expanded_nodes.expand(batch_size, -1, -1)
        ], dim=2)
        
        # 计算注意力分数 [batch_size, num_nodes, 1]
        attention_scores = self.attention(interaction_feats)
        
        # 如果提供了边信息，则限制注意力范围
        if edges is not None:
            # 创建掩码，只允许有边连接的节点之间计算注意力
            mask = torch.zeros(batch_size, num_nodes, device=state.device)
            for src, dst in edges:
                mask[:, dst] = 1  # 只关注目标节点
            
            # 应用掩码
            attention_scores = attention_scores * mask.unsqueeze(2)
        
        # Softmax得到注意力权重 [batch_size, num_nodes, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权聚合节点特征 [batch_size, hidden_dim]
        context = torch.sum(attention_weights * expanded_nodes, dim=1)
        
        # 更新状态
        new_state = self.state_update(context, state)
        
        return new_state

# 数据集
class GraphReasoningDataset(Dataset):
    def __init__(self, graph_dir, graph_files=None):
        self.graph_dir = graph_dir
        
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
        
        # 解析节点和边
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # 提取节点特征
        node_feats = []
        question_idx = -1
        answer_idx = -1
        
        for i, node in enumerate(nodes):
            # 节点特征
            if 'feat' in node and node['feat'] != 'PLACEHOLDER':
                try:
                    feat = np.array(node['feat'], dtype=np.float32)
                    if feat.shape[0] < 768:
                        feat = np.pad(feat, (0, 768-feat.shape[0]))
                    elif feat.shape[0] > 768:
                        feat = feat[:768]
                except:
                    feat = np.random.rand(768).astype(np.float32)
            else:
                feat = np.random.rand(768).astype(np.float32)
            
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
        
        # 提取边信息
        edge_pairs = []
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
            
            edge_pairs.append((src_id, dst_id))
        
        # 转为numpy数组
        node_feats = np.array(node_feats, dtype=np.float32)
        edge_pairs = np.array(edge_pairs, dtype=np.int64) if edge_pairs else np.zeros((0, 2), dtype=np.int64)
        
        return {
            'node_feats': node_feats,
            'edge_pairs': edge_pairs,
            'question_idx': question_idx,
            'answer_idx': answer_idx,
            'graph_id': graph_data.get('id', self.graph_files[idx]),
            'num_nodes': len(nodes)
        }

def collate_fn(samples):
    """收集函数，处理批次数据"""
    # 必须保证数据项的一致性
    collated_data = {
        'node_feats': torch.stack([s['node_feats'] for s in samples]),
        'question_idx': torch.tensor([s['question_idx'] for s in samples], dtype=torch.long),
        'answer_idx': torch.tensor([s['answer_idx'] for s in samples], dtype=torch.long),
        'candidate_idxs': [s['candidate_idxs'] for s in samples],
        'graph_id': [s['graph_id'] for s in samples]
    }
    
    # 如果有边信息，也整合
    if 'edges' in samples[0]:
        collated_data['edges'] = [s['edges'] for s in samples]
    
    # 如果有完整图对象，也保留
    if 'graph' in samples[0]:
        # 注意：对于DGLGraph，需要使用批处理函数
        try:
            import dgl
            graphs = [s['graph'] for s in samples]
            # 判断是否为异构图
            if hasattr(graphs[0], 'ntypes') and len(graphs[0].ntypes) > 1:
                # 对于异构图，直接保存图列表，不进行批处理
                collated_data['graph'] = graphs
            else:
                # 对于同构图，使用dgl.batch批处理
                collated_data['graph'] = dgl.batch(graphs)
        except:
            # 如果没有dgl或出错，直接保存图列表
            collated_data['graph'] = [s['graph'] for s in samples]
    
    return collated_data

# 训练函数
def train(model, train_loader, optimizer, device, epoch, output_dir="./output", log_interval=10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # 创建可视化目录
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 存储熵和多数投票比例
    entropy_values = []
    majority_ratios = []
    
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # 获取数据
        node_feats = data['node_feats'].to(device)
        edge_pairs = data['edge_pairs'].to(device) if data['edge_pairs'] is not None else None
        question_idxs = data['question_idx'].to(device)
        answer_idxs = data['answer_idx'].to(device)
        
        batch_size = node_feats.size(0)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        chain_outputs, final_output, attention_weights, predictions = model(node_feats, edge_pairs)
        
        # 计算损失
        # 多链对比损失
        chain_loss = 0
        for i, chain_output in enumerate(chain_outputs):
            # 计算链与问题和答案的相似度
            q_embeds = torch.stack([node_feats[b, question_idxs[b]] for b in range(batch_size)])
            a_embeds = torch.stack([node_feats[b, answer_idxs[b]] for b in range(batch_size)])
            
            # 计算链与问题-答案对的对比损失
            positive_sim = F.cosine_similarity(chain_output, a_embeds)
            negative_sim = F.cosine_similarity(chain_output, q_embeds)
            
            chain_loss += torch.mean(torch.clamp(negative_sim - positive_sim + 0.5, min=0))
        
        chain_loss = chain_loss / len(chain_outputs)
        
        # 最终预测损失（标签为1表示正确答案）
        final_loss = F.binary_cross_entropy_with_logits(
            predictions, 
            torch.ones_like(predictions)
        )
        
        # 合并损失
        loss = final_loss + 0.5 * chain_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率（简化版，实际应用中需更复杂的评估）
        pred = (predictions > 0).float()
        correct += pred.sum().item()
        total += batch_size
        
        # 计算每个样本的熵和多数投票比例
        for b in range(batch_size):
            # 构建候选列表（包括正确答案和其他节点）
            all_nodes = list(range(node_feats[b].size(0)))
            candidates = [answer_idxs[b].item()]  # 正确答案
            
            # 添加其他候选节点（简化为随机选择）
            other_nodes = [n for n in all_nodes if n != answer_idxs[b].item()]
            if other_nodes:
                # 随机选择几个作为候选
                num_candidates = min(5, len(other_nodes))
                candidates.extend(random.sample(other_nodes, num_candidates))
            
            # 计算每个链的答案以及熵和多数投票比例
            _, _, entropy, majority_ratio = model.compute_chain_answers(
                node_feats[b], candidates, edge_pairs[b] if edge_pairs is not None else None
            )
            
            entropy_values.append(entropy)
            majority_ratios.append(majority_ratio)
            
            # 记录到TensorBoard
            global_step = epoch * len(train_loader) * batch_size + batch_idx * batch_size + b
            writer.add_scalar('Training/Entropy', entropy, global_step)
            writer.add_scalar('Training/MajorityRatio', majority_ratio, global_step)
        
        # 打印进度
        if (batch_idx + 1) % log_interval == 0:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # 计算平均熵和多数投票比例
    avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0
    avg_majority = sum(majority_ratios) / len(majority_ratios) if majority_ratios else 0
    
    # 绘制熵与多数投票比例的关系图
    plt.figure(figsize=(10, 6))
    plt.scatter(entropy_values, majority_ratios, alpha=0.5)
    plt.xlabel('答案熵')
    plt.ylabel('多数投票比例')
    plt.title(f'答案熵 vs 多数投票比例 (Epoch {epoch})')
    plt.grid(True, alpha=0.3)
    
    # 添加收敛区域标记
    plt.axhline(y=2/3, color='r', linestyle='--', label='多数投票阈值 (2/3)')
    plt.axvline(x=0.5, color='g', linestyle='--', label='熵阈值 (0.5)')
    
    # 标记区域
    plt.axvspan(0, 0.5, ymin=2/3/1.0, ymax=1.0, alpha=0.2, color='green', label='收敛区域')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', f'entropy_majority_ep{epoch}.png'))
    plt.close()
    
    # 添加到TensorBoard
    writer.add_scalar('Training/AvgLoss', avg_loss, epoch)
    writer.add_scalar('Training/Accuracy', accuracy, epoch)
    writer.add_scalar('Training/AvgEntropy', avg_entropy, epoch)
    writer.add_scalar('Training/AvgMajority', avg_majority, epoch)
    writer.add_figure('Training/EntropyMajorityPlot', plt.gcf(), epoch)
    
    print(f'Train Epoch: {epoch}, Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, '
          f'Avg Entropy: {avg_entropy:.4f}, Avg Majority: {avg_majority:.4f}')
    
    # 检查是否达到收敛条件
    is_converged = False
    if avg_entropy < 0.5 and avg_majority >= 2/3:
        print(f"达到熵和多数投票收敛条件: E={avg_entropy:.4f} < 0.5, m={avg_majority:.4f} >= 2/3")
        is_converged = True
    
    return avg_loss, accuracy, avg_entropy, avg_majority, is_converged

# 验证函数
def validate(model, val_loader, device, epoch, output_dir="./output"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 收集评估指标
    rank_win_rates = []
    entropy_values = []
    majority_ratios = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating"):
            # 获取数据
            node_feats = data['node_feats'].to(device)
            edge_pairs = data['edge_pairs'].to(device) if data['edge_pairs'] is not None else None
            question_idxs = data['question_idx'].to(device)
            answer_idxs = data['answer_idx'].to(device)
            
            batch_size = node_feats.size(0)
            
            # 前向传播
            chain_outputs, final_output, attention_weights, predictions = model(node_feats, edge_pairs)
            
            # 计算损失与训练时相同
            chain_loss = 0
            for i, chain_output in enumerate(chain_outputs):
                q_embeds = torch.stack([node_feats[b, question_idxs[b]] for b in range(batch_size)])
                a_embeds = torch.stack([node_feats[b, answer_idxs[b]] for b in range(batch_size)])
                
                positive_sim = F.cosine_similarity(chain_output, a_embeds)
                negative_sim = F.cosine_similarity(chain_output, q_embeds)
                
                chain_loss += torch.mean(torch.clamp(negative_sim - positive_sim + 0.5, min=0))
            
            chain_loss = chain_loss / len(chain_outputs)
            
            final_loss = F.binary_cross_entropy_with_logits(
                predictions, 
                torch.ones_like(predictions)
            )
            
            loss = final_loss + 0.5 * chain_loss
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            pred = (predictions > 0).float()
            correct += pred.sum().item()
            total += batch_size
            
            # 计算RankPrompt胜率和熵/多数票比例
            for b in range(batch_size):
                # RankPrompt评估（比较当前样本与随机样本）
                if batch_size > 1:
                    # 选择另一个样本进行比较
                    other_idx = (b + 1) % batch_size
                    win_rate = model.rank_prompt_compare(
                        node_feats[b], node_feats[other_idx], question_idxs[b]
                    )
                    rank_win_rates.append(win_rate)
                
                # 计算熵和多数投票比例
                # 构建候选列表
                all_nodes = list(range(node_feats[b].size(0)))
                candidates = [answer_idxs[b].item()]  # 正确答案
                
                # 添加其他候选节点
                other_nodes = [n for n in all_nodes if n != answer_idxs[b].item()]
                if other_nodes:
                    num_candidates = min(5, len(other_nodes))
                    candidates.extend(random.sample(other_nodes, num_candidates))
                
                # 计算各指标
                _, _, entropy, majority_ratio = model.compute_chain_answers(
                    node_feats[b], candidates, edge_pairs[b] if edge_pairs is not None else None
                )
                
                entropy_values.append(entropy)
                majority_ratios.append(majority_ratio)
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # 计算平均RankPrompt胜率
    avg_rank_win_rate = sum(rank_win_rates) / len(rank_win_rates) if rank_win_rates else 0
    
    # 计算平均熵和多数投票比例
    avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0
    avg_majority = sum(majority_ratios) / len(majority_ratios) if majority_ratios else 0
    
    # 检查是否达到RankPrompt收敛条件
    rank_converged = avg_rank_win_rate >= 0.6
    entropy_majority_converged = (avg_entropy < 0.5 and avg_majority >= 2/3)
    is_converged = rank_converged or entropy_majority_converged
    
    # 创建验证集的熵-多数投票散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(entropy_values, majority_ratios, alpha=0.5)
    plt.xlabel('答案熵')
    plt.ylabel('多数投票比例')
    plt.title(f'验证集: 答案熵 vs 多数投票比例 (Epoch {epoch})')
    plt.grid(True, alpha=0.3)
    
    # 添加收敛区域标记
    plt.axhline(y=2/3, color='r', linestyle='--', label='多数投票阈值 (2/3)')
    plt.axvline(x=0.5, color='g', linestyle='--', label='熵阈值 (0.5)')
    plt.axvspan(0, 0.5, ymin=2/3/1.0, ymax=1.0, alpha=0.2, color='green', label='收敛区域')
    plt.legend()
    
    # 添加收敛指标文本
    plt.text(0.05, 0.05, 
             f'RankPrompt Win-rate: {avg_rank_win_rate:.4f}\n'
             f'Entropy: {avg_entropy:.4f}\n'
             f'Majority: {avg_majority:.4f}\n'
             f'Converged: {"Yes" if is_converged else "No"}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', f'val_entropy_majority_ep{epoch}.png'))
    
    # 创建交互式HTML版本用于Streamlit
    html_path = os.path.join(output_dir, 'visualizations', f'val_entropy_majority_ep{epoch}.html')
    try:
        fig = plt.gcf()
        import plotly.express as px
        import plotly.graph_objects as go
        
        # 创建plotly交互式图表
        fig_plotly = px.scatter(
            x=entropy_values, 
            y=majority_ratios,
            labels={'x': '答案熵', 'y': '多数投票比例'},
            title=f'验证集: 答案熵 vs 多数投票比例 (Epoch {epoch})'
        )
        
        # 添加阈值线
        fig_plotly.add_hline(y=2/3, line_dash="dash", line_color="red", annotation_text="多数投票阈值 (2/3)")
        fig_plotly.add_vline(x=0.5, line_dash="dash", line_color="green", annotation_text="熵阈值 (0.5)")
        
        # 添加收敛区域
        fig_plotly.add_shape(
            type="rect",
            x0=0, y0=2/3,
            x1=0.5, y1=1,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0
        )
        
        # 添加指标注释
        fig_plotly.add_annotation(
            x=0.1, y=0.1,
            text=f'RankPrompt: {avg_rank_win_rate:.4f}<br>Entropy: {avg_entropy:.4f}<br>Majority: {avg_majority:.4f}',
            showarrow=False,
            bgcolor="white", opacity=0.8
        )
        
        # 保存为HTML
        fig_plotly.write_html(html_path)
    except ImportError:
        print("未安装plotly，跳过生成交互式图表")
    
    plt.close()
    
    print(f'Validation, Avg. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, '
          f'RankPrompt Win-rate: {avg_rank_win_rate:.4f}, '
          f'Avg Entropy: {avg_entropy:.4f}, Avg Majority: {avg_majority:.4f}, '
          f'Converged: {"Yes" if is_converged else "No"}')
    
    return avg_loss, accuracy, avg_rank_win_rate, avg_entropy, avg_majority, is_converged

# 获取推理链可视化
def get_reasoning_chains(model, data, device, top_k=5, visualize=True, output_dir="./output"):
    """分析模型生成的推理链，并可视化结果"""
    # 转换为张量，并移至设备
    node_feats = data['node_feats'].to(device)
    candidate_idxs = data['candidate_idxs']
    question_idx = data['question_idx'].to(device)
    answer_idx = data['answer_idx'].item()
    
    # 如果有图结构，也添加
    g = None
    if 'graph' in data:
        g = data['graph'].to(device)
    
    # 获取每个推理链对候选答案的预测
    chain_predictions, ensemble_prediction, entropy, majority_ratio = model.compute_chain_answers(
        node_feats, candidate_idxs, g
    )
    
    # 解析注意力权重
    attention_weights = attention_weights.squeeze(0).cpu().numpy()
    
    # 获取每个链最关注的节点
    chains = []
    for i in range(model.num_reasoning_chains):
        # 计算链i和所有节点的相似度
        chain_output = chain_outputs[i].squeeze(0)
        node_embeds = node_feats.squeeze(0)
        
        similarities = F.cosine_similarity(chain_output.unsqueeze(0), node_embeds)
        similarities = similarities.cpu().numpy()
        
        # 获取top-k节点索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 检查是否包含答案节点
        contains_answer = answer_idx in top_indices
        
        chains.append({
            'chain_idx': i,
            'top_nodes': top_indices.tolist(),
            'similarities': similarities[top_indices].tolist(),
            'weight': attention_weights[i],
            'contains_answer': contains_answer
        })
    
    # 计算多链一致性指标
    # 构建候选列表
    candidates = [answer_idx]  # 正确答案
    
    # 添加其他候选节点
    other_nodes = [n for n in range(node_feats.size(0)) if n != answer_idx]
    if other_nodes:
        num_candidates = min(5, len(other_nodes))
        candidates.extend(random.sample(other_nodes, num_candidates))
    
    # 计算各指标
    chain_predictions, ensemble_prediction, entropy, majority_ratio = model.compute_chain_answers(
        node_feats.squeeze(0), candidates, g.squeeze(0) if g is not None else None
    )
    
    # 如果需要可视化
    if visualize:
        # 创建胜率矩阵可视化
        if len(chain_predictions) > 1:
            # 计算每条链的胜率矩阵
            num_chains = len(chain_predictions)
            win_matrix = np.zeros((num_chains, num_chains))
            
            for i in range(num_chains):
                for j in range(num_chains):
                    if i != j:
                        # 计算链i和链j的胜率
                        win_rate = np.mean([
                            1 if chain_predictions[i][k] > chain_predictions[j][k] else 0
                            for k in range(len(candidates))
                        ])
                        win_matrix[i, j] = win_rate
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(win_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('推理链胜率矩阵')
            plt.xlabel('链 j')
            plt.ylabel('链 i')
            
            # 添加说明注解
            plt.figtext(0.5, 0.01, '值表示链i优于链j的概率', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', 'chain_win_matrix.png'))
            plt.close()
        
        # 创建采样数量与熵/多数率的关系曲线
        plt.figure(figsize=(12, 6))
        
        # 模拟不同采样数下的熵和多数投票比例
        x_samples = list(range(1, model.num_reasoning_chains + 1))
        entropy_by_samples = []
        majority_by_samples = []
        
        for n_samples in x_samples:
            # 随机选择n_samples条链
            sample_indices = random.sample(range(model.num_reasoning_chains), n_samples)
            
            # 获取各链对候选答案的预测
            sample_predictions = [chain_predictions[i] for i in sample_indices]
            
            # 找出每个链预测的最高分答案
            best_answers = [np.argmax(pred) for pred in sample_predictions]
            
            # 统计答案频率
            answer_counts = Counter(best_answers)
            
            # 计算多数投票占比
            majority_answer = answer_counts.most_common(1)[0][0]
            majority_ratio = answer_counts[majority_answer] / n_samples
            majority_by_samples.append(majority_ratio)
            
            # 计算熵
            sample_entropy = 0
            for count in answer_counts.values():
                prob = count / n_samples
                sample_entropy -= prob * math.log2(prob)
            
            # 归一化熵
            max_entropy = math.log2(len(candidates)) if len(candidates) > 0 else 0
            if max_entropy > 0:
                sample_entropy /= max_entropy
            
            entropy_by_samples.append(sample_entropy)
        
        # 绘制熵随采样数变化曲线
        plt.subplot(1, 2, 1)
        plt.plot(x_samples, entropy_by_samples, 'o-', label='熵')
        plt.axhline(y=0.5, color='r', linestyle='--', label='阈值 = 0.5')
        plt.xlabel('推理链采样数')
        plt.ylabel('答案熵 (归一化)')
        plt.title('熵随采样数的变化')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 绘制多数投票比例随采样数变化曲线
        plt.subplot(1, 2, 2)
        plt.plot(x_samples, majority_by_samples, 'o-', label='多数投票比例')
        plt.axhline(y=2/3, color='r', linestyle='--', label='阈值 = 2/3')
        plt.xlabel('推理链采样数')
        plt.ylabel('多数投票比例')
        plt.title('多数投票比例随采样数的变化')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', 'sampling_curves.png'))
        plt.close()
        
        # 尝试创建交互式版本
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=('熵随采样数的变化', '多数投票比例随采样数的变化'))
            
            # 熵曲线
            fig.add_trace(
                go.Scatter(x=x_samples, y=entropy_by_samples, mode='lines+markers', name='熵'),
                row=1, col=1
            )
            fig.add_shape(
                type='line',
                x0=1, y0=0.5,
                x1=model.num_reasoning_chains, y1=0.5,
                line=dict(color='red', dash='dash'),
                row=1, col=1
            )
            
            # 多数投票比例曲线
            fig.add_trace(
                go.Scatter(x=x_samples, y=majority_by_samples, mode='lines+markers', name='多数投票比例'),
                row=1, col=2
            )
            fig.add_shape(
                type='line',
                x0=1, y0=2/3,
                x1=model.num_reasoning_chains, y1=2/3,
                line=dict(color='red', dash='dash'),
                row=1, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title_text=f"推理链采样分析 (熵: {entropy:.4f}, 多数投票: {majority_ratio:.4f})",
                height=500, width=1000
            )
            
            fig.update_xaxes(title_text='推理链采样数', row=1, col=1)
            fig.update_xaxes(title_text='推理链采样数', row=1, col=2)
            fig.update_yaxes(title_text='答案熵 (归一化)', row=1, col=1)
            fig.update_yaxes(title_text='多数投票比例', row=1, col=2)
            
            # 保存为交互式HTML
            fig.write_html(os.path.join(output_dir, 'visualizations', 'sampling_curves_interactive.html'))
        except ImportError:
            print("未安装plotly，跳过生成交互式图表")
    
    return chains, attention_weights, {
        'entropy': entropy,
        'majority_ratio': majority_ratio,
        'chain_predictions': chain_predictions,
        'ensemble_prediction': ensemble_prediction
    }

def main():
    parser = argparse.ArgumentParser(description='多链思维训练')
    parser.add_argument('--graph_dir', type=str, required=True, help='子图目录')
    parser.add_argument('--model_dir', type=str, default='models/route4_multi_chain/checkpoints', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='models/route4_multi_chain/output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_chains', type=int, default=20, help='推理链数量')
    parser.add_argument('--chain_length', type=int, default=3, help='推理链长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--visualize', action='store_true', help='生成更详细的可视化')
    parser.add_argument('--entropy_threshold', type=float, default=0.5, help='答案熵收敛阈值')
    parser.add_argument('--majority_threshold', type=float, default=2/3, help='多数投票比例阈值')
    parser.add_argument('--rankprompt_threshold', type=float, default=0.6, help='RankPrompt胜率阈值')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA加速训练')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 选择设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型保存目录和输出目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 加载数据集
    graph_files = [f for f in os.listdir(args.graph_dir) if f.endswith('.json')]
    np.random.shuffle(graph_files)
    
    # 划分训练集和验证集
    val_size = int(0.2 * len(graph_files))
    train_files = graph_files[val_size:]
    val_files = graph_files[:val_size]
    
    print(f'训练集: {len(train_files)}个图, 验证集: {len(val_files)}个图')
    
    # 创建数据加载器
    train_dataset = GraphReasoningDataset(args.graph_dir, train_files)
    val_dataset = GraphReasoningDataset(args.graph_dir, val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = MultiChainReasoner(
        node_dim=768,  # 假设使用768维向量
        hidden_dim=args.hidden_dim,
        num_reasoning_chains=args.num_chains,
        chain_length=args.chain_length
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练
    best_val_loss = float('inf')
    
    # 跟踪熵、多数投票比例和RankPrompt胜率
    train_entropies = []
    train_majorities = []
    val_entropies = []
    val_majorities = []
    val_rankprompt_rates = []
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc, train_entropy, train_majority, train_converged = train(
            model, train_loader, optimizer, device, epoch, output_dir=args.output_dir
        )
        
        # 验证
        val_loss, val_acc, val_rankprompt, val_entropy, val_majority, val_converged = validate(
            model, val_loader, device, epoch, output_dir=args.output_dir
        )
        
        # 记录指标
        train_entropies.append(train_entropy)
        train_majorities.append(train_majority)
        val_entropies.append(val_entropy)
        val_majorities.append(val_majority)
        val_rankprompt_rates.append(val_rankprompt)
        
        # 保存到TensorBoard
        writer.add_scalar('Summary/TrainLoss', train_loss, epoch)
        writer.add_scalar('Summary/TrainAccuracy', train_acc, epoch)
        writer.add_scalar('Summary/ValLoss', val_loss, epoch)
        writer.add_scalar('Summary/ValAccuracy', val_acc, epoch)
        writer.add_scalar('Metrics/TrainEntropy', train_entropy, epoch)
        writer.add_scalar('Metrics/TrainMajority', train_majority, epoch)
        writer.add_scalar('Metrics/ValEntropy', val_entropy, epoch)
        writer.add_scalar('Metrics/ValMajority', val_majority, epoch)
        writer.add_scalar('Metrics/ValRankPrompt', val_rankprompt, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.model_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型到 {model_path}')
        
        # 保存检查点
        checkpoint_path = os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_rankprompt': val_rankprompt,
            'train_entropy': train_entropy,
            'train_majority': train_majority,
            'val_entropy': val_entropy,
            'val_majority': val_majority
        }, checkpoint_path)
        
        # 检查是否达到收敛条件
        if val_converged:
            print(f"验证集已达到收敛条件, 提前停止训练。Epoch: {epoch}")
            break
    
    # 关闭TensorBoard
    writer.close()
    
    print(f'训练完成. 最佳验证损失: {best_val_loss:.4f}')
    
    # 生成最终的收敛指标图表
    plt.figure(figsize=(12, 10))
    
    # 熵随epoch变化
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_entropies) + 1), train_entropies, label='训练')
    plt.plot(range(1, len(val_entropies) + 1), val_entropies, label='验证')
    plt.axhline(y=args.entropy_threshold, color='r', linestyle='--', label=f'阈值 = {args.entropy_threshold}')
    plt.title('答案熵随训练进度变化')
    plt.xlabel('Epoch')
    plt.ylabel('熵')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 多数投票比例随epoch变化
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_majorities) + 1), train_majorities, label='训练')
    plt.plot(range(1, len(val_majorities) + 1), val_majorities, label='验证')
    plt.axhline(y=args.majority_threshold, color='r', linestyle='--', label=f'阈值 = {args.majority_threshold}')
    plt.title('多数投票比例随训练进度变化')
    plt.xlabel('Epoch')
    plt.ylabel('多数投票比例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RankPrompt胜率随epoch变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(val_rankprompt_rates) + 1), val_rankprompt_rates)
    plt.axhline(y=args.rankprompt_threshold, color='r', linestyle='--', label=f'阈值 = {args.rankprompt_threshold}')
    plt.title('RankPrompt胜率随训练进度变化')
    plt.xlabel('Epoch')
    plt.ylabel('RankPrompt胜率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 收敛情况图
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(val_entropies, val_majorities, c=val_rankprompt_rates, cmap='viridis', s=100)
    plt.axhline(y=args.majority_threshold, color='r', linestyle='--')
    plt.axvline(x=args.entropy_threshold, color='g', linestyle='--')
    plt.axvspan(0, args.entropy_threshold, ymin=0, ymax=args.majority_threshold/1.0, alpha=0.2, color='green')
    plt.title('收敛指标关系图')
    plt.xlabel('熵')
    plt.ylabel('多数投票比例')
    plt.colorbar(scatter, label='RankPrompt胜率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'final_metrics.png'))
    plt.close()
    
    # 分析一个样本的推理链
    if len(val_dataset) > 0:
        print("\n分析一个样本的多链推理结果:")
        sample_idx = 0
        sample = val_dataset[sample_idx]
        
        chains, weights, metrics = get_reasoning_chains(
            model, sample, device, visualize=args.visualize, output_dir=args.output_dir
        )
        
        print(f"样本 ID: {sample['graph_id']}")
        print(f"熵: {metrics['entropy']:.4f}, 多数投票比例: {metrics['majority_ratio']:.4f}")
        
        print("\n各推理链的表现:")
        for i, chain in enumerate(chains):
            contains_answer = "包含答案" if chain['contains_answer'] else "不包含答案"
            print(f"链 {i+1} (权重: {chain['weight']:.4f}, {contains_answer}):")
            for j, node_idx in enumerate(chain['top_nodes']):
                print(f"  节点 {node_idx}: 相似度 {chain['similarities'][j]:.4f}")
        
        # 为每条链生成人类可理解的理由（简化版，实际中应该使用LLM）
        print("\n生成的推理链解释:")
        for i, chain in enumerate(chains[:3]):  # 仅显示前3条链
            if chain['contains_answer']:
                explanation = f"该链关注节点 {chain['top_nodes'][:3]} 并正确识别了答案节点。"
            else:
                explanation = f"该链关注节点 {chain['top_nodes'][:3]} 但未识别答案节点。"
            print(f"链 {i+1}: {explanation}")
        
        # 整体结论
        if metrics['majority_ratio'] >= args.majority_threshold:
            conclusion = "大多数推理链一致认可答案，结果可信。"
        elif metrics['entropy'] < args.entropy_threshold:
            conclusion = "推理链答案分布明确，结果较可信。"
        else:
            conclusion = "推理链观点分散，结果的确定性较低。"
        
        print(f"\n最终判断: {conclusion}")

if __name__ == '__main__':
    main() 