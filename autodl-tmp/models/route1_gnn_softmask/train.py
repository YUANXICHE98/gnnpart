#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线1：GNN+软掩码 - 训练脚本
使用软掩码GNN在子图上训练，动态学习边的重要性
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
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
from collections import deque
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from tqdm import tqdm  # 导入tqdm用于进度显示

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import load_graph_data

# 动态自适应稀疏控制器
class AdaptiveSparsityController:
    def __init__(self, target_sparsity=0.25, window_size=10):
        self.target_sparsity = target_sparsity
        self.window_size = window_size
        
        # 性能监控
        self.performance_history = deque(maxlen=window_size)
        self.sparsity_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        
        # 最佳性能记录
        self.best_performance = 0.0
        self.total_training_steps = 0
        
        # 修改：更保守的权重设置，防止过度稀疏
        self.base_l1_weight = 0.001   # 降低基础权重
        self.max_l1_weight = 0.01     # 降低最大权重
        self.min_l1_weight = 0.0001   # 降低最小权重
        
        # 新增：温度参数控制
        self.temperature_history = deque(maxlen=5)
        self.target_temperature_range = (0.5, 2.0)  # 温度安全范围
        self.temperature_adjustment_rate = 0.1  # 温度调整速率
        
    def update_history(self, performance, sparsity, loss, temperature=None):
        """更新历史记录"""
        self.performance_history.append(performance)
        self.sparsity_history.append(sparsity)
        self.loss_history.append(loss)
        
        if temperature is not None:
            self.temperature_history.append(temperature)
        
        if performance > self.best_performance:
            self.best_performance = performance
    
    def set_total_steps(self, total_steps):
        """设置总训练步数"""
        self.total_training_steps = total_steps
    
    def get_training_stage_factor(self, current_step):
        """根据训练阶段返回不同的调整因子 - 改进版"""
        if self.total_training_steps == 0:
            return 1.0
            
        progress = current_step / self.total_training_steps
        
        if progress < 0.3:
            # 前30%：非常温和的稀疏化，让模型先学会基本特征
            return 0.2
        elif progress < 0.6:
            # 中30%：逐渐增加稀疏化
            return 0.5 + (progress - 0.3) * 1.67  # 从0.5渐增到1.0
        else:
            # 后40%：稳定的稀疏化
            return 1.0
    
    def detect_performance_drop(self, threshold=0.03):
        """检测性能下降 - 降低阈值，提高敏感度"""
        if len(self.performance_history) < 3:
            return False
        
        recent_avg = np.mean(list(self.performance_history)[-2:])
        earlier_avg = np.mean(list(self.performance_history)[-4:-2])
        
        return (earlier_avg - recent_avg) > threshold
    
    def detect_sparsity_explosion(self, threshold=0.3):
        """检测稀疏度爆炸 - 新增功能"""
        if len(self.sparsity_history) < 2:
            return False
        
        current = self.sparsity_history[-1]
        previous = self.sparsity_history[-2]
        
        # 如果稀疏度在一个epoch内增长超过30%，认为是爆炸
        return (current - previous) > threshold
    
    def get_sparsity_trend(self):
        """获取稀疏度趋势"""
        if len(self.sparsity_history) < 3:
            return 0
        
        recent = list(self.sparsity_history)
        return recent[-1] - recent[0]
    
    def compute_temperature_adjustment(self, current_sparsity, current_temperature):
        """计算温度参数调整 - 新增核心功能"""
        target_temp = current_temperature
        
        # 稀疏度过高：降低温度，使掩码更温和
        if current_sparsity > self.target_sparsity + 0.15:
            adjustment = -self.temperature_adjustment_rate * min(2.0, current_sparsity / self.target_sparsity)
            target_temp = max(self.target_temperature_range[0], current_temperature + adjustment)
            
        # 稀疏度过低：适度提高温度
        elif current_sparsity < self.target_sparsity - 0.1:
            adjustment = self.temperature_adjustment_rate * 0.5  # 更保守的增长
            target_temp = min(self.target_temperature_range[1], current_temperature + adjustment)
        
        # 检测稀疏度爆炸，强制降低温度
        if self.detect_sparsity_explosion():
            target_temp = max(0.3, current_temperature * 0.7)  # 强制降温
            print(f"⚠️ 检测到稀疏度爆炸，强制降温到 {target_temp:.3f}")
        
        return target_temp
    
    def compute_adaptive_weight(self, current_sparsity, current_performance, training_step):
        """计算自适应稀疏权重 - 改进版，更保守"""
        # 1. 稀疏度偏差项
        sparsity_deviation = abs(current_sparsity - self.target_sparsity)
        
        # 2. 训练阶段系数
        stage_factor = self.get_training_stage_factor(training_step)
        
        # 3. 更保守的动态权重计算
        if current_sparsity > self.target_sparsity + 0.2:
            # 稀疏度严重过高，几乎停止正则化
            weight = self.min_l1_weight
        elif current_sparsity > self.target_sparsity + 0.1:
            # 稀疏度过高，大幅降低正则化
            weight = self.min_l1_weight * 2
        elif current_sparsity < self.target_sparsity - 0.15:
            # 稀疏度过低，适度增加正则化
            weight = self.base_l1_weight + (self.max_l1_weight - self.base_l1_weight) * 0.5
        else:
            # 在合理范围内，使用基础权重
            weight = self.base_l1_weight * stage_factor
        
        # 性能下降保护
        if self.detect_performance_drop():
            weight *= 0.5  # 性能下降时减半权重
            print(f"⚠️ 检测到性能下降，降低稀疏权重到 {weight:.6f}")
        
        # 确保权重在合理范围内
        weight = max(self.min_l1_weight, min(self.max_l1_weight, weight))
        
        return weight
    
    def compute_entropy_regularization(self, edge_masks):
        """熵正则化：鼓励掩码分布的多样性 - 改进版"""
        if len(edge_masks) == 0:
            return torch.tensor(0.0, device=edge_masks.device if hasattr(edge_masks, 'device') else 'cpu')
        
        # 避免所有掩码都趋向同一个值
        p = torch.clamp(edge_masks, 1e-8, 1-1e-8)
        entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p)).mean()
        
        # 根据当前稀疏度调整熵权重
        current_sparsity = (edge_masks < 0.5).float().mean()
        if current_sparsity > 0.8:
            # 过度稀疏时，强化熵正则化，鼓励多样性
            return -entropy * 2.0
        else:
            return -entropy * 0.5  # 正常情况下使用较小权重

# 软掩码GNN模型
class SoftMaskGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.2, sparsity_target=0.25):
        super(SoftMaskGNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparsity_target = sparsity_target
        
        # 初始化自适应稀疏控制器
        self.sparsity_controller = AdaptiveSparsityController(target_sparsity=sparsity_target)
        
        # 节点类型特定的变换
        self.question_transform = nn.Linear(in_dim, hidden_dim)
        self.entity_transform = nn.Linear(in_dim, hidden_dim)
        self.context_transform = nn.Linear(in_dim, hidden_dim)
        # 添加：训练步数追踪
        self.register_buffer('training_steps', torch.tensor(0.0))
        
        # 修改：更保守的初始温度和参数设置
        self.mask_temperature = nn.Parameter(torch.tensor(0.8))  # 降低初始温度
        
        # 新增：温度控制相关参数
        self.temperature_bounds = (0.3, 2.5)  # 温度边界
        self.register_buffer('last_sparsity', torch.tensor(0.0))  # 记录上次稀疏度
        
        # 边类型特定的变换
        self.edge_transforms = nn.ModuleDict({
            'answers': nn.Linear(hidden_dim, hidden_dim),
            'evidencedBy': nn.Linear(hidden_dim, hidden_dim),
            'supportsAnswer': nn.Linear(hidden_dim, hidden_dim),
            'relatedTo': nn.Linear(hidden_dim, hidden_dim),
            'default': nn.Linear(hidden_dim, hidden_dim)
        })
        
        # 边重要性预测网络 (使用Sigmoid确保值在0-1之间)
        self.edge_importance = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 边掩码温度参数（可训练）
        self.mask_temperature = nn.Parameter(torch.tensor(1.0))
        
        # 消息传递层
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, 1)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, g, node_feats, edge_weights=None):
        """
        前向传播来获取节点表示和边掩码
        
        参数:
        - g: DGL图
        - node_feats: 节点特征 [num_nodes, in_dim]
        - edge_weights: 可选的预定义边权重
        
        返回:
        - h: 节点嵌入 [num_nodes, hidden_dim]
        - edge_masks: 边掩码 [num_edges, 1]
        - sparsity: 掩码稀疏率
        """
        # 初始化节点嵌入
        h = torch.zeros(node_feats.size(0), self.hidden_dim, device=node_feats.device)
        
        # 根据节点角色应用不同的变换
        if hasattr(g, 'ndata') and 'role' in g.ndata:
            roles = g.ndata['role']
            # 确保角色数据是合适的尺寸
            if len(roles) != node_feats.size(0):
                # 如果角色和节点特征数量不匹配，为所有节点使用默认变换
                h = self.context_transform(node_feats)
            else:
                for i in range(node_feats.size(0)):
                    role = roles[i].item() if isinstance(roles[i], torch.Tensor) else roles[i]
                    
                    # 基于角色选择变换
                    if role == 0:  # 问题节点
                        h[i] = self.question_transform(node_feats[i])
                    elif role in [2, 3]:  # 答案/证据节点
                        h[i] = self.entity_transform(node_feats[i])
                    else:  # 默认为上下文节点
                        h[i] = self.context_transform(node_feats[i])
        else:
            # 如果没有角色信息，对所有节点应用相同的变换
            h = self.context_transform(node_feats)
        
        # 初始化特征
        h = F.relu(h)
        
        # 获取边的源和目标节点索引
        edge_src, edge_dst = g.edges()
        num_edges = len(edge_src)
        
        # 创建边特征，拼接边的源节点和目标节点表示
        edge_feats = []
        for i in range(num_edges):
            src, dst = edge_src[i], edge_dst[i]
            # 确保索引有效
            if src < h.size(0) and dst < h.size(0):
                edge_feat = torch.cat([h[src], h[dst]], dim=0)
                edge_feats.append(edge_feat)
        
        if len(edge_feats) > 0:
            edge_feats = torch.stack(edge_feats)
            # 学习边掩码（每条边一个掩码值）
            edge_importances = self.edge_importance(edge_feats).squeeze()
            
            # 修改：使用更合理的阈值和温度
            edge_masks = torch.sigmoid((edge_importances - 0.5) * torch.exp(self.mask_temperature))
            
            # 修改：添加"结构化噪声"以提高掩码多样性
            if self.training:
                # 在训练期间添加结构化噪声
                noise = torch.randn_like(edge_importances) * 0.02  # 进一步降低噪声强度
                
                # 安全获取is_gold_path
                is_gold_path = None
                if hasattr(g, 'edata'):
                    if 'is_gold_path' in g.edata:
                        is_gold_path = g.edata['is_gold_path']
                        # 确保张量在正确的设备上
                        if is_gold_path.device != edge_importances.device:
                            is_gold_path = is_gold_path.to(edge_importances.device)
                            
                        # 确保尺寸匹配
                        if len(is_gold_path) == len(edge_importances):
                            # 使用布尔掩码进行索引
                            gold_edges = (is_gold_path > 0)
                            if gold_edges.any():
                                noise[gold_edges] *= 0.1  # 大幅减少黄金路径边的噪声
                edge_masks = torch.sigmoid((edge_importances + noise - 0.5) * torch.exp(self.mask_temperature))
            
            # 如果提供了先验权重，则与学习到的重要性相结合
            if edge_weights is not None and len(edge_weights) > 0:
                edge_weights = edge_weights.view(-1)
                if len(edge_weights) != len(edge_masks):
                    # 如果长度不匹配，使用广播或者扩展
                    if len(edge_weights) == 1:
                        # 单一权重，直接广播
                        edge_masks = edge_masks * edge_weights
                    else:
                        # 尺寸不匹配，取最小长度
                        min_len = min(len(edge_masks), len(edge_weights))
                        edge_masks = edge_masks[:min_len] * edge_weights[:min_len]
                else:
                    edge_masks = edge_masks * edge_weights
        else:
            # 如果没有边，创建空掩码
            edge_masks = torch.tensor([], device=node_feats.device)
        
        # 对每一层进行消息传递
        for layer_idx, layer in enumerate(self.layers):
            # 创建新的节点表示
            new_h = torch.zeros_like(h)
            
            # 对每条边进行消息传递
            for i in range(num_edges):
                src, dst = edge_src[i], edge_dst[i]
                # 确保索引有效
                if src >= h.size(0) or dst >= h.size(0):
                    continue
                    
                # 获取边类型
                edge_type = 'default'
                if hasattr(g, 'edata') and 'rel' in g.edata and i < len(g.edata['rel']):
                    edge_type = g.edata['rel'][i].item() if isinstance(g.edata['rel'][i], torch.Tensor) else g.edata['rel'][i]
                
                # 获取对应的边变换
                if edge_type in self.edge_transforms:
                    transform = self.edge_transforms[edge_type]
                else:
                    transform = self.edge_transforms['default']
                
                # 获取该边的掩码
                mask_val = edge_masks[i] if i < len(edge_masks) else torch.tensor(1.0, device=h.device)
                
                # 先应用转换，再应用掩码
                src_transformed = transform(h[src])
                if isinstance(mask_val, torch.Tensor) and mask_val.dim() > 0:
                    if mask_val.size(0) != 1:
                        mask_val = mask_val.mean()  # 转为标量
                
                # 掩码后的消息
                message = src_transformed * mask_val
                
                # 累积消息到目标节点
                new_h[dst] += message
            
            # 更新节点表示（包括残差连接）
            h = F.relu(layer(new_h + h))
            h = self.dropout_layer(h)
        
        # 修改：使用更合理的稀疏度阈值
        sparsity = (edge_masks < 0.5).float().mean().item() if len(edge_masks) > 0 else 0.0
        
        # 返回节点表示、边掩码和稀疏率
        return h, edge_masks, sparsity
    
    def compute_answer_scores(self, g, node_feats, edge_weights, question_idx, candidate_idxs):
        """计算所有候选答案的分数"""
        # 获取所有节点的表示和边重要性
        node_embeddings, edge_masks, sparsity = self.forward(g, node_feats, edge_weights)
        
        # 获取问题节点的表示
        q_embedding = node_embeddings[question_idx]
        
        # 计算每个候选答案的分数
        scores = torch.zeros(len(candidate_idxs), device=node_feats.device)
        
        # 确保维度匹配
        q_embed_reshaped = q_embedding.unsqueeze(0)  # [1, hidden_dim]
        
        for i, ans_idx in enumerate(candidate_idxs):
            # 安全检查避免索引越界
            if ans_idx < node_embeddings.size(0):
                # 计算余弦相似度
                ans_embedding = node_embeddings[ans_idx].unsqueeze(0)  # [1, hidden_dim]
                scores[i] = F.cosine_similarity(q_embed_reshaped, ans_embedding, dim=1)[0]
            else:
                # 对于无效索引，设置低分
                scores[i] = torch.tensor(-1.0, device=scores.device)
        
        return scores, edge_masks, sparsity
    
    def compute_adaptive_sparsity_loss(self, edge_masks, current_performance, training_step):
        """计算自适应稀疏性损失"""
        if len(edge_masks) == 0:
            return torch.tensor(0.0, device=edge_masks.device if hasattr(edge_masks, 'device') else 'cpu'), 0.0
        
        # 当前稀疏度
        current_sparsity = (edge_masks < 0.5).float().mean()
        
        # 动态权重
        adaptive_weight = self.sparsity_controller.compute_adaptive_weight(
            current_sparsity.item(), current_performance, training_step
        )
        
        # 基础稀疏损失
        sparsity_distance = F.mse_loss(
            current_sparsity, 
            torch.tensor(self.sparsity_target, device=edge_masks.device)
        )
        
        # 多层次正则化
        l1_reg = edge_masks.mean()  # 修改：使用mean而不是abs().mean()
        entropy_reg = self.sparsity_controller.compute_entropy_regularization(edge_masks)
        
        # 组合损失
        total_sparsity_loss = (
            sparsity_distance + 
            adaptive_weight * l1_reg + 
            0.005 * entropy_reg  # 降低熵正则化权重
        )
        
        return total_sparsity_loss, adaptive_weight

    def compute_node_embeddings(self, g, node_feats, edge_weights=None):
        """计算节点嵌入，不计算预测分数"""
        # 调用forward但只返回节点嵌入
        h, _, _ = self.forward(g, node_feats, edge_weights)
        return h

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
            
            # 边权重
            weight = edge.get('weight', 1.0)
            edge_weights.append(weight)
        
        # 创建DGL图
        g = dgl.graph((src_ids, dst_ids), num_nodes=len(nodes))
        
        # 确保tensor长度与图中节点数量匹配
        role_map = {'question': 0, 'context': 1, 'answer': 2, 'evidence': 3, 'distractor': 4}
        numeric_roles = [role_map.get(role, 0) for role in node_roles]
        g.ndata['role'] = torch.tensor(numeric_roles[:g.number_of_nodes()])
        
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
        batched_graph = graphs
    else:
        # 对于同构图，使用dgl.batch
        batched_graph = dgl.batch(graphs)
    
    node_feats = torch.cat([s['node_feats'] for s in samples], dim=0)
    edge_weights = torch.cat([s['edge_weights'] for s in samples], dim=0)
    question_idx = torch.tensor([s['question_idx'] for s in samples], dtype=torch.long)
    answer_idx = torch.tensor([s['answer_idx'] for s in samples], dtype=torch.long)
    candidate_idxs = [s['candidate_idxs'] for s in samples]
    graph_id = [s['graph_id'] for s in samples]
    return {
        'graph': batched_graph,
        'node_feats': node_feats,
        'edge_weights': edge_weights,
        'question_idx': question_idx,
        'answer_idx': answer_idx,
        'candidate_idxs': candidate_idxs,
        'graph_id': graph_id
    }

def contrastive_loss(node_embeds, question_idx, answer_idxs, negative_idxs, temperature=0.1):
    """添加对比学习损失，拉近问题和正确答案，推远问题和错误答案"""
    # 确保问题索引是单个值，如果是批次则取第一个
    if isinstance(question_idx, torch.Tensor) and question_idx.dim() > 0:
        q_idx = question_idx[0].item()
    else:
        q_idx = question_idx
        
    # 获取问题嵌入
    q_embed = node_embeds[q_idx]
    
    # 确保答案索引是一维张量
    if isinstance(answer_idxs, torch.Tensor) and answer_idxs.dim() > 1:
        ans_idxs = answer_idxs.flatten()
    else:
        ans_idxs = answer_idxs
    
    # 正例：正确答案
    pos_embeds = node_embeds[ans_idxs]
    
    # 确保形状正确，保持维度一致性
    q_embed_reshaped = q_embed.view(1, -1)  # 将问题嵌入重塑为 [1, dim]
    pos_sim = F.cosine_similarity(q_embed_reshaped, pos_embeds, dim=1) / temperature
    
    # 负例：错误答案或随机节点
    neg_embeds = node_embeds[negative_idxs]
    
    # 确保问题嵌入与负例嵌入的维度匹配
    # 重塑问题嵌入以匹配负例的第二维度
    if neg_embeds.size(1) != q_embed.size(0):
        # 如果维度不匹配，可能需要调整负例嵌入
        neg_embeds = neg_embeds[:, :q_embed.size(0)]
    
    neg_sim = F.cosine_similarity(q_embed_reshaped, neg_embeds, dim=1) / temperature
    
    # 正规化相似度
    logits = torch.cat([pos_sim, neg_sim])
    labels = torch.zeros(len(logits), device=node_embeds.device)
    labels[:len(pos_sim)] = 1.0
    
    # 计算对比损失
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def train(model, train_loader, optimizer, device, epoch, writer, sparsity_weight=0.01):
    """训练一个epoch并返回指标"""
    model.train()
    total_loss = 0
    total_sparsity_loss = 0
    total_task_loss = 0
    correct = 0
    total = 0
    
    # 跟踪所有图的掩码稀疏率
    all_sparsities = []
    # 添加F1分数跟踪
    all_f1_scores = []
    # 跟踪自适应权重
    all_adaptive_weights = []
    
    # 预训练阶段（前几个epoch）
    if epoch < 3:
        print(f"执行预训练 (Epoch {epoch+1}/3)...")
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取批次数据并移至设备
            g = batch['graph'].to(device)
            node_feats = batch['node_feats'].to(device)
            edge_weights = batch['edge_weights'].to(device)
            question_idx = batch['question_idx'].to(device)
            answer_idx = batch['answer_idx'].to(device)
            candidate_idxs = batch['candidate_idxs']
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 修改：使用正常的前向传播而不是只获取边掩码
            batch_scores = []
            batch_labels = []
            batch_masks = []
            
            # 获取批量大小
            batch_size = g.batch_size if hasattr(g, 'batch_size') else len(candidate_idxs)
            is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
            
            for i in range(batch_size):
                candidates = candidate_idxs[i]
                
                if is_hetero:
                    current_g = g[i] if isinstance(g, list) else g
                    current_feats = node_feats[i]
                    current_weights = edge_weights[i] if edge_weights.dim() > 1 else edge_weights
                    current_q_idx = question_idx[i]
                else:
                    current_g = g
                    current_feats = node_feats
                    current_weights = edge_weights
                    current_q_idx = question_idx[i]
                
                scores, edge_masks, _ = model.compute_answer_scores(
                    current_g, current_feats, current_weights, 
                    current_q_idx, candidates
                )
                
                # 创建标签
                labels = torch.zeros_like(scores)
                for j, c_idx in enumerate(candidates):
                    if c_idx == answer_idx[i].item():
                        labels[j] = 1
                        break
                
                batch_scores.append(scores)
                batch_labels.append(labels)
                batch_masks.append(edge_masks)
            
            # 计算主任务损失
            scores = torch.cat(batch_scores)
            labels = torch.cat(batch_labels)
            
            # 预训练使用较小的稀疏正则化
            task_loss = F.binary_cross_entropy_with_logits(scores, labels)
            
            # 轻微的稀疏正则化（预训练阶段）
            all_masks = torch.cat(batch_masks)
            sparsity_loss = 0.001 * all_masks.mean()  # 很小的L1正则化
            
            pretrain_loss = task_loss + sparsity_loss
            pretrain_loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += pretrain_loss.item()
            pred = (scores > 0.5).float()
            current_correct = (pred == labels).sum().item()
            epoch_correct += current_correct
            epoch_total += len(scores)
            
            # 更新训练步数
            with torch.no_grad():
                model.training_steps += 1.0
                
            # 记录到TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Pretrain/Loss', pretrain_loss.item(), global_step)
            writer.add_scalar('Pretrain/TaskLoss', task_loss.item(), global_step)
            writer.add_scalar('Pretrain/SparsityLoss', sparsity_loss.item(), global_step)
        
        # 计算预训练epoch平均值
        avg_pretrain_loss = epoch_loss / len(train_loader)
        pretrain_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        print(f"预训练 Epoch {epoch}: Loss: {avg_pretrain_loss:.4f}, Acc: {pretrain_accuracy:.4f}")
        
        # 返回预训练结果
        return avg_pretrain_loss, pretrain_accuracy, 0.0, 0.0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch+1}", leave=True)
    
    for batch_idx, batch in progress_bar:
        g = batch['graph'].to(device)
        node_feats = batch['node_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device)
        question_idx = batch['question_idx'].to(device)
        answer_idx = batch['answer_idx'].to(device)
        candidate_idxs = batch['candidate_idxs']
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 在计算负例索引之前，首先获取node_embeds
        node_embeds = None
        # 尝试从模型输出获取节点嵌入
        try:
            # 对整个批次图执行一次前向传播，获取所有节点嵌入
            node_embeds, _, _ = model(g, node_feats, edge_weights)
        except Exception as e:
            print(f"获取节点嵌入时出错: {e}")
            # 如果批处理前向传播失败，尝试对第一个图单独处理
            if batch_size > 0:
                try:
                    if is_hetero:
                        current_g = g[0] if isinstance(g, list) else g
                        current_feats = node_feats[0] if node_feats.dim() > 1 else node_feats
                        current_weights = edge_weights[0] if edge_weights.dim() > 1 else edge_weights
                    else:
                        # 对于同构图，使用完整图但只提取第一个图的嵌入
                        current_g = g
                        current_feats = node_feats
                        current_weights = edge_weights
                    
                    node_embeds, _, _ = model(current_g, current_feats, current_weights)
                except Exception as e:
                    print(f"尝试处理第一个图也失败: {e}")
                    node_embeds = None
        
        # 计算候选答案的分数
        batch_scores = []
        batch_labels = []
        batch_masks = []
        batch_sparsities = []
        
        # 获取批量大小 - 避免使用len(g)
        batch_size = g.batch_size if hasattr(g, 'batch_size') else len(candidate_idxs)
        
        # 检查是否为异构图
        is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
        
        for i in range(batch_size):
            # 获取当前图的候选答案索引
            candidates = candidate_idxs[i]
            
            # 计算每个候选答案的分数并获取边掩码
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
            
            scores, edge_masks, sparsity = model.compute_answer_scores(
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
            batch_masks.append(edge_masks)
            batch_sparsities.append(sparsity)
        
        # 计算主任务损失
        scores = torch.cat(batch_scores)
        labels = torch.cat(batch_labels)
        
        # 使用BCE loss
        task_loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        # 计算准确率和F1分数
        pred = (scores > 0.5).float()
        current_correct = (pred == labels).sum().item()
        current_total = len(scores)
        current_accuracy = current_correct / current_total if current_total > 0 else 0
        
        # 计算F1分数
        true_positives = ((pred == 1) & (labels == 1)).sum().item()
        pred_positives = (pred == 1).sum().item()
        actual_positives = (labels == 1).sum().item()
        
        precision = true_positives / pred_positives if pred_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        current_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算自适应稀疏性损失
        all_masks = torch.cat(batch_masks)
        sparsity_loss, adaptive_weight = model.compute_adaptive_sparsity_loss(
            all_masks, current_f1, model.training_steps.item()
        )
        
        # 更新稀疏控制器历史
        avg_sparsity = np.mean(batch_sparsities)
        current_temperature = model.mask_temperature.item()
        model.sparsity_controller.update_history(current_f1, avg_sparsity, task_loss.item(), current_temperature)
        
        # 新增：温度自动调整机制
        with torch.no_grad():
            target_temperature = model.sparsity_controller.compute_temperature_adjustment(
                avg_sparsity, current_temperature
            )
            
            # 平滑调整温度，避免剧烈变化
            adjustment_factor = 0.1  # 调整速度
            new_temperature = current_temperature + adjustment_factor * (target_temperature - current_temperature)
            
            # 限制温度在安全范围内
            new_temperature = max(model.temperature_bounds[0], min(model.temperature_bounds[1], new_temperature))
            
            # 更新温度参数
            model.mask_temperature.data.fill_(new_temperature)
            
            # 记录温度变化
            if abs(new_temperature - current_temperature) > 0.01:
                print(f"🌡️ 温度调整: {current_temperature:.3f} → {new_temperature:.3f} (稀疏度: {avg_sparsity:.3f})")
            
            # 更新上次稀疏度记录
            model.last_sparsity.data.fill_(avg_sparsity)
        
        # 获取负例索引
        negative_idxs = []
        for i, candidates in enumerate(candidate_idxs):
            # 选择非答案的候选项作为负例
            neg_candidates = [c for c in candidates if c != answer_idx[i].item()]
            if neg_candidates:
                negative_idxs.extend(neg_candidates)
        
        # 只有当有足够的负例时才计算对比损失
        if node_embeds is not None and len(negative_idxs) > 0:
            # 确保维度匹配
            # 如果我们只有一个batch的嵌入，传递batch中的第一个问题和答案
            # 使用负例创建合适维度的tensor
            try:
                contrast_loss = contrastive_loss(
                    node_embeds, 
                    question_idx[0] if question_idx.dim() > 0 else question_idx, 
                    answer_idx[0] if answer_idx.dim() > 0 else answer_idx,
                    torch.tensor(negative_idxs, device=device),
                    temperature=0.1
                )
                
                # 加入总损失 - 使用动态权重
                loss = task_loss + sparsity_loss + 0.1 * contrast_loss
            except Exception as e:
                print(f"计算对比损失时出错: {e}")
                # 跳过对比损失
                loss = task_loss + sparsity_loss
        else:
            # 如果没有足够的负例，只使用任务损失和稀疏性损失
            loss = task_loss + sparsity_loss
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 更新训练步数
        with torch.no_grad():
            model.training_steps += 1.0
        
        # 统计
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        
        # 跟踪掩码稀疏率
        for sparsity in batch_sparsities:
            all_sparsities.append(sparsity)
        
        correct += current_correct
        total += current_total
        
        # 跟踪F1分数和自适应权重
        all_f1_scores.append(current_f1)
        # 修复：为每个图添加相同的自适应权重，确保维度匹配
        for _ in batch_sparsities:
            all_adaptive_weights.append(adaptive_weight)
        
        # 记录到TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/TaskLoss', task_loss.item(), global_step)
        writer.add_scalar('Train/SparsityLoss', sparsity_loss.item(), global_step)
        writer.add_scalar('Train/SparsityRate', np.mean(batch_sparsities), global_step)
        writer.add_scalar('Train/Temperature', model.mask_temperature.item(), global_step)  # 新增温度记录
        writer.add_scalar('Train/AdaptiveWeight', adaptive_weight, global_step)
        
        # 更新进度条信息
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'task_loss': f"{task_loss.item():.4f}",
            'sparsity_loss': f"{sparsity_loss.item():.4f}",
            'sparsity': f"{np.mean(batch_sparsities):.3f}",
            'temp': f"{model.mask_temperature.item():.2f}",  # 添加温度显示
            'adaptive_w': f"{adaptive_weight:.5f}",
            'acc': f"{current_correct/current_total:.4f}" if current_total > 0 else "N/A"
        })
    
    # 计算平均值
    avg_loss = total_loss / len(train_loader)
    avg_task_loss = total_task_loss / len(train_loader)
    avg_sparsity_loss = total_sparsity_loss / len(train_loader)
    avg_sparsity = np.mean(all_sparsities)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    avg_adaptive_weight = np.mean(all_adaptive_weights) if all_adaptive_weights else 0
    accuracy = correct / total if total > 0 else 0
    
    # 打印epoch结果
    print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Task Loss: {avg_task_loss:.4f}, "
          f"Sparsity Loss: {avg_sparsity_loss:.4f}, Acc: {accuracy:.4f}, F1: {avg_f1:.4f}, "
          f"Sparsity: {avg_sparsity:.4f}, Adaptive Weight: {avg_adaptive_weight:.5f}")
    
    # 记录详细数据到TensorBoard（移除绘图，只记录数据）
    writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Train/EpochAccuracy', accuracy, epoch)
    writer.add_scalar('Train/EpochF1', avg_f1, epoch)
    writer.add_scalar('Train/EpochSparsityRate', avg_sparsity, epoch)
    writer.add_scalar('Train/EpochAdaptiveWeight', avg_adaptive_weight, epoch)
    
    # 记录分布数据（用于后续可视化）
    writer.add_histogram('Train/SparsityDistribution', torch.tensor(all_sparsities), epoch)
    writer.add_histogram('Train/AdaptiveWeightDistribution', torch.tensor(all_adaptive_weights), epoch)
    writer.add_histogram('Train/F1Distribution', torch.tensor(all_f1_scores), epoch)
    
    return avg_loss, accuracy, avg_f1, avg_sparsity

def validate(model, val_loader, device, epoch, writer):
    """验证模型并返回指标"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 跟踪所有图的掩码稀疏率和F1分数
    all_sparsities = []
    all_f1_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, total=len(val_loader), desc="Validation", leave=False)
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
            batch_sparsities = []
            
            # 获取批量大小 - 避免使用len(g)
            batch_size = g.batch_size if hasattr(g, 'batch_size') else len(candidate_idxs)
            
            # 检查是否为异构图
            is_hetero = hasattr(g, 'ntypes') and len(g.ntypes) > 1
            
            for i in range(batch_size):
                # 获取当前图的候选答案索引
                candidates = candidate_idxs[i]
                
                # 计算每个候选答案的分数并获取边掩码和稀疏率
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
                
                scores, _, sparsity = model.compute_answer_scores(
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
                batch_sparsities.append(sparsity)
            
            # 拼接结果
            scores = torch.cat(batch_scores)
            labels = torch.cat(batch_labels)
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss.item()
            
            # 计算准确率
            pred = (scores > 0.5).float()
            current_correct = (pred == labels).sum().item()
            correct += current_correct
            current_total = len(scores)
            total += current_total
            
            # 计算F1分数
            true_positives = ((pred == 1) & (labels == 1)).sum().item()
            pred_positives = (pred == 1).sum().item()
            actual_positives = (labels == 1).sum().item()
            
            precision = true_positives / pred_positives if pred_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 跟踪稀疏率和F1分数
            current_sparsity = np.mean(batch_sparsities)
            all_sparsities.extend(batch_sparsities)
            all_f1_scores.append(f1)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{current_correct/current_total:.4f}",
                'f1': f"{f1:.4f}", 
                'sparsity': f"{current_sparsity:.3f}"
            })
    
    # 计算平均值
    avg_loss = total_loss / len(val_loader)
    avg_sparsity = np.mean(all_sparsities)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    accuracy = correct / total if total > 0 else 0
    
    # 打印结果
    print(f"Validation: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {avg_f1:.4f}, Sparsity: {avg_sparsity:.4f}")
    
    # 记录到TensorBoard（移除绘图，只记录数据）
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    writer.add_scalar('Validation/F1', avg_f1, epoch)
    writer.add_scalar('Validation/SparsityRate', avg_sparsity, epoch)
    
    return avg_loss, accuracy, avg_f1, avg_sparsity

def analyze_edge_importance(model, graph_data, device):
    """分析边重要性并可视化热图"""
    model.eval()
    g = graph_data['graph'].to(device)
    node_feats = graph_data['node_feats'].to(device)
    edge_weights = graph_data['edge_weights'].to(device)
    
    with torch.no_grad():
        # 前向传播获取边掩码
        _, edge_masks, sparsity = model.forward(g, node_feats, edge_weights)
        
        # 获取边的源和目标节点
        edge_src, edge_dst = g.edges()
        
        # 分析重要边
        important_edges = []
        for i in range(len(edge_src)):
            src, dst = edge_src[i], edge_dst[i]
            mask = edge_masks[i].item()
            
            if mask > 0.5:  # 重要边阈值
                important_edges.append((src.item(), dst.item(), mask))
        
        # 按重要性排序
        important_edges.sort(key=lambda x: x[2], reverse=True)
        
        # 打印最重要的边
        print(f"掩码稀疏率: {sparsity:.4f}, 重要边数量: {len(important_edges)}")
        for src, dst, mask in important_edges[:10]:
            print(f"边 ({src} -> {dst}): 重要性 = {mask:.4f}")
        
        return edge_masks, sparsity, important_edges

def check_early_stopping(sparsity_history, f1_history, patience=8):
    """
    检查是否应该早停 - 修改为更宽松的条件
    
    条件: 当稀疏率ρ∈[0.15, 0.50]且dev F1 8个epoch无提升时才停止
    """
    if len(sparsity_history) < patience or len(f1_history) < patience:
        return False
    
    # 检查稀疏率是否在目标范围内（放宽范围）
    latest_sparsity = sparsity_history[-1]
    in_target_range = 0.05 <= latest_sparsity <= 0.60  # 更宽的范围
    
    # 检查F1是否有提升（更严格的无提升条件）
    recent_f1 = list(f1_history)[-patience:]
    best_recent_f1 = max(recent_f1)
    no_improvement = best_recent_f1 <= f1_history[-patience] + 1e-4
    
    # 额外检查：如果F1下降过多，也要早停
    recent_avg = np.mean(list(f1_history)[-3:])
    earlier_avg = np.mean(list(f1_history)[-6:-3]) if len(f1_history) >= 6 else recent_avg
    severe_drop = (earlier_avg - recent_avg) > 0.15  # F1下降超过15%
    
    return in_target_range and no_improvement and not severe_drop

def main():
    parser = argparse.ArgumentParser(description='软掩码GNN训练')
    parser.add_argument('--graph_dir', type=str, required=True, help='图数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='GNN层数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--sparsity_weight', type=float, default=0.05, help='稀疏性损失权重(已弃用，使用自适应权重)')
    parser.add_argument('--sparsity_target', type=float, default=0.25, help='目标稀疏率 (0.1-0.4)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    print(f"[Route1] 创建输出目录: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Route1] 输出目录创建状态: {os.path.exists(args.output_dir)}")
    
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    print(f"[Route1] 创建检查点目录: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[Route1] 检查点目录创建状态: {os.path.exists(checkpoint_dir)}")
    
    # 设置TensorBoard
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 选择设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据集
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
    model = SoftMaskGNN(
        in_dim=768, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sparsity_target=args.sparsity_target
    )
    model.to(device)
    
    # 设置总训练步数
    total_steps = args.epochs * len(train_loader)
    model.sparsity_controller.set_total_steps(total_steps)
    print(f"总训练步数: {total_steps}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    print("=" * 60)
    print("🚀 开始Route1 GNN软掩码训练")
    print("=" * 60)
    print(f"📊 训练配置:")
    print(f"   模型参数: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    print(f"   训练参数: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"   稀疏化目标: {args.sparsity_target:.2f}")
    print(f"   数据集: 训练{len(train_dataset)}个, 验证{len(val_dataset)}个")
    print(f"   设备: {device}")
    print(f"   输出目录: {args.output_dir}")
    print("=" * 60)
    
    best_val_f1 = 0
    patience_counter = 0
    
    # 清理旧的metrics.json文件（可选）
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print(f"发现现有的metrics.json文件，是否清理？(建议清理以避免数据混乱)")
        print(f"文件路径: {metrics_path}")
        # 在自动化环境中，直接清理旧文件
        try:
            os.remove(metrics_path)
            print("已清理旧的metrics.json文件")
        except Exception as e:
            print(f"清理文件失败: {e}")
    
    # 历史指标记录用于早停
    sparsity_history = deque(maxlen=10)
    f1_history = deque(maxlen=10)
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc, train_f1, train_sparsity = train(
            model, train_loader, optimizer, device, epoch, writer, 
            sparsity_weight=args.sparsity_weight
        )
        
        # 验证
        val_loss, val_acc, val_f1, val_sparsity = validate(model, val_loader, device, epoch, writer)
        
        # 更新历史指标
        sparsity_history.append(val_sparsity)
        f1_history.append(val_f1)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Sparsity: {train_sparsity:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Sparsity: {val_sparsity:.4f}")
        
        # 检查稀疏度是否在合理范围内
        sparsity_status = "正常"
        if val_sparsity > 0.8:
            sparsity_status = "过高"
        elif val_sparsity < 0.1:
            sparsity_status = "过低"
        print(f"  稀疏度状态: {sparsity_status} (目标: {args.sparsity_target:.2f})")
        
        # 保存检查点
        print(f"\n[Route1] 正在保存检查点到: {checkpoint_dir}")
        print(f"[Route1] 检查点目录存在: {os.path.exists(checkpoint_dir)}")

        # 先定义路径，再使用
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        print(f"[Route1] 具体文件路径: {checkpoint_path}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_sparsity': val_sparsity,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_sparsity': train_sparsity,
            'val_acc': val_acc,
            'sparsity_controller_state': {
                'performance_history': list(model.sparsity_controller.performance_history),
                'sparsity_history': list(model.sparsity_controller.sparsity_history),
                'best_performance': model.sparsity_controller.best_performance
            }
        }, checkpoint_path)
        
        print(f"[Route1] 检查点保存成功: {os.path.exists(checkpoint_path)}")
        
        # 修复：改进训练指标保存逻辑，避免重复epoch
        metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_f1': float(train_f1),
            'train_sparsity': float(train_sparsity),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'val_sparsity': float(val_sparsity),
            'sparsity_status': sparsity_status
        }
        
        # 检查是否已存在相同epoch的数据，如果存在则替换
        epoch_exists = False
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    all_metrics = []
        else:
            all_metrics = []
        
        for i, existing_metric in enumerate(all_metrics):
            if existing_metric.get('epoch') == epoch:
                all_metrics[i] = metrics  # 替换现有数据
                epoch_exists = True
                break
        
        # 如果不存在相同epoch，则追加
        if not epoch_exists:
            all_metrics.append(metrics)
        
        # 按epoch排序确保数据有序
        all_metrics.sort(key=lambda x: x.get('epoch', 0))
        
        # 保存更新后的数据
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"[Route1] 指标已保存到: {metrics_path} (共{len(all_metrics)}个epoch)")
        
        # 如果是最佳模型，保存为best_model.pt
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存最佳模型，F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 改进的早停条件
        should_stop = False
        
        # 条件1：稀疏度在合理范围内且F1分数连续5轮无提升
        if 0.15 <= val_sparsity <= 0.4 and patience_counter >= 5:
            print(f"触发早停：稀疏度在合理范围内({val_sparsity:.3f})且F1分数5轮无提升")
            should_stop = True
        
        # 条件2：稀疏度过高且持续3轮以上
        elif len(sparsity_history) >= 3 and all(s > 0.8 for s in list(sparsity_history)[-3:]):
            print(f"触发早停：稀疏度持续过高(>0.8)")
            should_stop = True
        
        # 条件3：超过15轮无改进
        elif patience_counter >= 15:
            print(f"触发早停：超过15轮无改进")
            should_stop = True
        
        if should_stop:
            break
    
    # 训练结束后，加载最佳模型并分析结果
    best_model = SoftMaskGNN(
        in_dim=768, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sparsity_target=args.sparsity_target
    )
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    best_model.to(device)
    
    # 分析几个样本的边重要性
    print("\n=== 最佳模型分析 ===")
    for i in range(min(3, len(val_dataset))):
        print(f"\n分析样本 {i}:")
        edge_masks, sparsity, important_edges = analyze_edge_importance(
            best_model, val_dataset[i], device
        )
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("训练完成！")
    print(f"最佳验证F1分数: {best_val_f1:.4f}")
    print(f"最终稀疏度状态: {sparsity_status}")

if __name__ == '__main__':
    main() 