#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线1：GNN软掩码 - 推理脚本
使用GNN结合软掩码进行知识图谱推理
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
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import dgl

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# GNN软掩码模型定义
class GNNSoftMask(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers=3, dropout=0.2):
        super(GNNSoftMask, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点嵌入层
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # 边掩码生成器
        self.edge_mask_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 添加：训练步数追踪
        self.register_buffer('training_steps', torch.tensor(0.0))
        
        # 修改：更高的初始温度
        self.mask_temperature = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, node_features, edge_index):
        """
        前向传播
        
        参数:
        - node_features: 节点特征，形状为 [num_nodes, node_dim]
        - edge_index: 边索引，形状为 [2, num_edges]
        
        返回:
        - prediction: 预测结果
        - edge_mask: 边掩码权重
        - node_embeds: 节点嵌入
        """
        # 初始节点嵌入
        node_embeds = self.node_embedding(node_features)
        
        # 确认边的数量
        num_edges = edge_index.size(1)
        
        # 计算初始边掩码
        if num_edges > 0:
            edge_feats = []
            for j in range(num_edges):
                src, dst = edge_index[0, j], edge_index[1, j]
                edge_feat = torch.cat([node_embeds[src], node_embeds[dst]], dim=0)
                edge_feats.append(edge_feat)
            
            edge_feats = torch.stack(edge_feats)
            # 学习边掩码（每条边一个掩码值）
            edge_importances = self.edge_mask_generator(edge_feats).squeeze()
            
            # 修改：使用更合理的温度和阈值设置
            if self.training:
                temp = torch.exp(self.mask_temperature)
                offset = 0.3  # 训练时的偏移
            else:
                # 推理时使用更平衡的参数
                temp = 2.0  # 降低温度，减少过度稀疏化
                offset = 0.4  # 降低阈值，允许更多边保持重要性
            
            edge_masks = torch.sigmoid((edge_importances - offset) * temp)
        else:
            # 如果没有边，创建空掩码
            edge_masks = torch.tensor([], device=node_features.device)
        
        # GNN消息传递
        for i in range(self.num_layers):
            # 消息传递
            messages = torch.zeros_like(node_embeds)
            
            # 只有在有边的情况下才进行消息传递
            if num_edges > 0:
                for j in range(num_edges):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    # 确保索引在有效范围内
                    mask_value = edge_masks[j] if j < len(edge_masks) else torch.tensor(1.0, device=node_features.device)
                    # 确保mask是标量
                    if isinstance(mask_value, torch.Tensor) and mask_value.numel() > 1:
                            mask_value = mask_value.mean()
                    
                    messages[dst] += node_embeds[src] * mask_value
            
            # 节点更新
            node_embeds = self.gnn_layers[i](node_embeds + messages)
            
            # 更新边掩码
            if i < self.num_layers - 1 and num_edges > 0:
                edge_feats = []
                for j in range(num_edges):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    edge_feat = torch.cat([node_embeds[src], node_embeds[dst]], dim=0)
                    edge_feats.append(edge_feat)
                
                edge_feats = torch.stack(edge_feats)
                # 学习边掩码（每条边一个掩码值）
                edge_importances = self.edge_mask_generator(edge_feats).squeeze()
                
                # 修改：使用与forward一致的温度和阈值设置
                if self.training:
                    temp = torch.exp(self.mask_temperature)
                    offset = 0.3  # 训练时的偏移
                else:
                    # 推理时使用更平衡的参数
                    temp = 2.0  # 降低温度，减少过度稀疏化
                    offset = 0.4  # 降低阈值，允许更多边保持重要性
                
                edge_masks = torch.sigmoid((edge_importances - offset) * temp)
        
        # 预测（以问题节点特征作为输入）
        question_idx = 0  # 假设第一个节点是问题节点
        prediction = self.predictor(node_embeds[question_idx])
        
        return prediction, edge_masks, node_embeds
    
    def compute_edge_mask(self, node_embeds, edge_index):
        """计算边掩码"""
        edge_features = []
        for j in range(edge_index.size(1)):
            src, dst = edge_index[0, j], edge_index[1, j]
            edge_feat = torch.cat([node_embeds[src], node_embeds[dst]], dim=0)
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features)
        edge_mask = self.edge_mask_generator(edge_features).squeeze(-1)
        
        return edge_mask

# 单图推理函数
def inference_single_graph(model, graph_data, device, visualize=False, threshold=0.5):
    """
    对单个图执行推理
    
    参数:
    - model: 训练好的GNN模型，如果为None则创建一个新的模型
    - graph_data: 图数据
    - device: 设备
    - visualize: 是否可视化
    - threshold: 掩码阈值，用于确定重要边
    
    返回:
    - result: 推理结果
    """
    # 从图数据中提取节点和边
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    question = graph_data.get('question', '')
    
    # 如果没有问题文本，从问题节点中提取
    if not question:
        for node in nodes:
            if node.get('role', '') == 'question':
                question = node.get('surface_text', '')
                break
    
    # 提取节点特征
    node_features = []
    node_labels = []
    node_roles = {}
    
    question_idx = -1
    answer_indices = []
    
    for i, node in enumerate(nodes):
        # 提取节点特征
        if 'feat' in node and node['feat'] != 'PLACEHOLDER':
            try:
                feat = np.array(node['feat'], dtype=np.float32)
            except:
                # 使用随机特征
                feat = np.random.rand(768).astype(np.float32)
        else:
            # 使用随机特征
            feat = np.random.rand(768).astype(np.float32)
        
        node_features.append(feat)
        
        # 提取节点标签 - 优先使用surface_text
        node_text = node.get('surface_text', node.get('value', f'Node_{i}'))
        node_labels.append(node_text)
        
        # 记录节点角色
        role = node.get('role', '')
        name = node.get('name', '')
        node_roles[i] = role
        
        if role == 'question':
            # 优先选择name为"question_entity"的节点作为主问题节点
            if question_idx == -1:  # 如果还没有找到问题节点
                question_idx = i
            elif node.get('name', '') == 'question_entity':  # 如果找到更合适的问题节点
                question_idx = i
        elif role == 'answer':
            answer_indices.append(i)
    
    # 如果没有找到问题节点，使用第一个节点
    if question_idx == -1 and len(nodes) > 0:
        question_idx = 0
    
    # 提取边
    edge_index = []
    edge_labels = []
    
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        # 确保边有效
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        
        # 确保节点索引有效
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        edge_index.append([src_id, dst_id])
        edge_labels.append(edge.get('value', ''))
    
    # 转换为PyTorch张量
    node_features = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long).to(device) if edge_index else torch.zeros((2, 0), dtype=torch.long).to(device)
    
    # 创建或使用模型
    if model is None:
        model = GNNSoftMask(node_dim=node_features.size(1), hidden_dim=256, num_layers=3).to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 执行推理
    with torch.no_grad():
        prediction, edge_mask, node_embeds = model(node_features, edge_index)
    
    # 处理边掩码
    edge_importance = {}
    if edge_index.size(1) > 0:
        edge_mask_np = edge_mask.cpu().numpy()
        # 确保edge_mask_np维度匹配
        if isinstance(edge_mask_np, np.ndarray):
            if edge_mask_np.ndim == 0:
                edge_mask_np = np.array([float(edge_mask_np)])
            
        for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
            if i < len(edge_mask_np):
                edge_importance[(int(src), int(dst))] = float(edge_mask_np[i])
            else:
                # 如果索引超出范围，使用默认值
                edge_importance[(int(src), int(dst))] = 0.5
                print(f"Warning: Edge index out of range: {i} < {len(edge_mask_np)}")
    
    # 计算节点相似度（与问题和答案节点的相似度）
    node_importance = {}
    node_embeds_np = node_embeds.cpu().numpy()
    question_embed = node_embeds_np[question_idx]
    
    for i, embed in enumerate(node_embeds_np):
        if i != question_idx:
            # 计算与问题节点的余弦相似度
            sim = np.dot(embed, question_embed) / (np.linalg.norm(embed) * np.linalg.norm(question_embed) + 1e-8)
            node_importance[i] = float(sim)
    
    # 初始化重要路径列表
    important_paths = []
    
    # 找出重要路径
    if question_idx != -1 and answer_indices:
        # 创建NetworkX图以找出路径
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(len(nodes)):
            G.add_node(i)
        
        # 计算边重要性的相对差异
        edge_importances_values = list(edge_importance.values())
        if edge_importances_values:
            min_importance = min(edge_importances_values)
            max_importance = max(edge_importances_values)
            importance_range = max_importance - min_importance
            
            # 如果所有边重要性都很接近，使用不同的策略
            if importance_range < 0.05:  # 重要性差异很小
                # 策略：直接使用无权重图
                G_unweighted = nx.DiGraph()
                for i in range(len(nodes)):
                    G_unweighted.add_node(i)
                for (src, dst) in edge_importance.keys():
                    G_unweighted.add_edge(src, dst)
                G = G_unweighted  # 主要使用无权重图
            else:
                # 原始策略：权重为边掩码的互补
                for (src, dst), importance in edge_importance.items():
                    weight = 1.0 - importance
                    G.add_edge(src, dst, weight=weight)
        
        # 查找从问题到答案的路径
        for answer_idx in answer_indices:
            paths_found = []
            
            try:
                # 尝试最短路径（考虑权重）
                if importance_range >= 0.05:
                    path = nx.shortest_path(G, source=question_idx, target=answer_idx, weight='weight')
                    paths_found.append(('weighted', path))
            except Exception as e:
                pass
                
            try:
                # 尝试最短路径（不考虑权重）
                path = nx.shortest_path(G, source=question_idx, target=answer_idx)
                paths_found.append(('shortest', path))
            except Exception as e:
                pass
                
            # 如果还是没找到，检查图的连通性
            if not paths_found:
                if G.has_node(question_idx) and G.has_node(answer_idx):
                    # 检查是否有出边和入边
                    q_out = list(G.successors(question_idx))
                    a_in = list(G.predecessors(answer_idx))
                    
                    # 尝试2跳路径
                    for intermediate in q_out:
                        if G.has_edge(intermediate, answer_idx):
                            path = [question_idx, intermediate, answer_idx]
                            paths_found.append(('two_hop', path))
                            break
                
            # 计算每条路径的重要性并添加到结果
            for path_type, path in paths_found:
                if len(path) > 1:  # 确保路径有意义
                    path_importance = 0.0
                    path_edges = []
                    
                    for i in range(len(path) - 1):
                        src, dst = path[i], path[i+1]
                        edge_key = (src, dst)
                        if edge_key in edge_importance:
                            edge_imp = edge_importance[edge_key]
                            path_importance += edge_imp
                            path_edges.append((src, dst, edge_imp))
                    
                    # 添加到重要路径列表
                    avg_importance = path_importance / len(path_edges) if path_edges else 0
                    important_paths.append({
                        'path': path,
                        'importance': avg_importance,
                        'edges': path_edges,
                        'type': path_type
                    })
    
    # 按重要性排序
    important_paths.sort(key=lambda x: x['importance'], reverse=True)
    
    # 获取预测结果
    prediction_score = torch.sigmoid(prediction).item()
    
    # 改进的预测逻辑：结合路径信息
    if important_paths:
        # 如果找到了重要路径，基于路径质量调整预测
        best_path_importance = important_paths[0]['importance']
        
        # 如果路径重要性高于阈值，或者找到了直接连接，认为预测为True
        if best_path_importance > 0.3 or len(important_paths[0]['path']) == 2:  # 直接连接
            predicted_label = True
            # 调整置信度，结合原始分数和路径重要性
            adjusted_score = max(prediction_score, 0.5 + best_path_importance * 0.5)
            prediction_score = min(adjusted_score, 0.95)  # 限制最高值
        else:
            predicted_label = prediction_score > 0.5
    else:
        # 没有找到路径，使用原始判断
        predicted_label = prediction_score > 0.5
    
    # 可视化
    if visualize:
        visualize_graph(nodes, edges, node_labels, edge_labels, node_importance, edge_importance, 
                       question_idx, answer_indices, important_paths[:1] if important_paths else [])
    
    # 生成解释文本
    explanation = generate_explanation(nodes, node_labels, edges, edge_labels, question, 
                                      edge_importance, important_paths)
    
    # 构建结果
    result = {
        'prediction': bool(predicted_label),
        'score': prediction_score,
        'edge_importance': {f"{k[0]}-{k[1]}": v for k, v in edge_importance.items()},
        'node_importance': {str(k): v for k, v in node_importance.items()},
        'important_paths': [{'path': p['path'], 'importance': p['importance']} for p in important_paths[:3]],
        'explanation': explanation
    }
    
    return result

def visualize_graph(nodes, edges, node_labels, edge_labels, node_importance, edge_importance, 
                   question_idx, answer_indices, important_paths):
    """
    可视化图及重要边
    
    参数:
    - nodes: 节点列表
    - edges: 边列表
    - node_labels: 节点标签
    - edge_labels: 边标签
    - node_importance: 节点重要性
    - edge_importance: 边重要性
    - question_idx: 问题节点索引
    - answer_indices: 答案节点索引列表
    - important_paths: 重要路径列表
    """
    plt.figure(figsize=(12, 10))
    
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(len(nodes)):
        G.add_node(i, label=node_labels[i] if i < len(node_labels) else f"Node_{i}")
    
    # 添加边
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        G.add_edge(src_id, dst_id)
    
    # 创建布局
    pos = nx.spring_layout(G, seed=42)
    
    # 节点颜色: 问题节点为绿色，答案节点为蓝色，其他节点基于重要性着色
    node_colors = []
    node_sizes = []
    
    for i in range(len(nodes)):
        if i == question_idx:
            node_colors.append('lightgreen')
            node_sizes.append(700)
        elif i in answer_indices:
            node_colors.append('lightblue')
            node_sizes.append(600)
        else:
            # 根据重要性着色
            importance = node_importance.get(i, 0.0)
            node_colors.append(plt.cm.Oranges(0.3 + 0.7 * importance))
            node_sizes.append(300 + 400 * importance)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # 绘制所有边（灰色）
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True)
    
    # 绘制重要边（根据边掩码着色）
    edge_colors = []
    edge_widths = []
    edge_list = []
    
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        edge_key = (src_id, dst_id)
        importance = edge_importance.get(edge_key, 0.0)
        
        if importance > 0.3:  # 只显示重要性超过阈值的边
            edge_list.append(edge_key)
            edge_colors.append(plt.cm.Blues(importance))
            edge_widths.append(1 + 5 * importance)
    
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    # 绘制最重要的路径（红色）
    if important_paths:
        best_path = important_paths[0]['path']
        best_path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, edge_color='red', width=3, alpha=0.8)
    
    # 绘制节点标签
    # 截断过长的标签
    short_labels = {}
    for i, label in enumerate(node_labels):
        if len(label) > 20:
            short_labels[i] = label[:17] + "..."
        else:
            short_labels[i] = label
    
    nx.draw_networkx_labels(G, pos, labels=short_labels, font_size=8)
    
    # 添加图例
    plt.legend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], color='red', linewidth=3),
        plt.Line2D([0], [0], color='blue', linewidth=2)
    ], ['问题节点', '答案节点', '最优路径', '重要边'], loc='upper right')
    
    plt.title("GNN软掩码推理结果可视化", fontsize=16)
    plt.axis('off')
    
    return plt.gcf()

def generate_explanation(nodes, node_labels, edges, edge_labels, question, edge_importance, important_paths):
    """
    生成解释文本
    
    参数:
    - nodes: 节点列表
    - node_labels: 节点标签
    - edges: 边列表
    - edge_labels: 边标签
    - question: 问题文本
    - edge_importance: 边重要性
    - important_paths: 重要路径列表
    
    返回:
    - 解释文本
    """
    explanation = []
    
    # 添加问题
    explanation.append(f"问题: {question}")
    explanation.append("")
    
    # 添加重要路径分析
    explanation.append("【关键推理路径】")
    if important_paths:
        for i, path_data in enumerate(important_paths[:3]):  # 最多显示前3条路径
            path = path_data['path']
            importance = path_data['importance']
            
            path_text = []
            for node_idx in path:
                if node_idx < len(node_labels):
                    path_text.append(f"「{node_labels[node_idx]}」")
            
            explanation.append(f"路径 {i+1} (重要性: {importance:.4f}):")
            explanation.append(" → ".join(path_text))
            explanation.append("")
    else:
        explanation.append("未找到重要路径")
        explanation.append("")
    
    # 添加重要边分析
    explanation.append("【关键关系】")
    important_edges = [(k, v) for k, v in edge_importance.items() if v > 0.3]
    important_edges.sort(key=lambda x: x[1], reverse=True)
    
    for (src, dst), importance in important_edges[:5]:  # 最多显示前5条重要边
        src_label = node_labels[src] if src < len(node_labels) else f"节点{src}"
        dst_label = node_labels[dst] if dst < len(node_labels) else f"节点{dst}"
        
        # 查找边标签
        edge_label = ""
        for edge in edges:
            edge_src = edge.get('src', '').replace('n', '')
            edge_dst = edge.get('dst', '').replace('n', '')
            
            if edge_src.isdigit() and edge_dst.isdigit():
                if int(edge_src) == src and int(edge_dst) == dst:
                    edge_label = edge.get('value', '')
                    break
        
        if edge_label:
            explanation.append(f"- 「{src_label}」 {edge_label} 「{dst_label}」 (重要性: {importance:.4f})")
        else:
            explanation.append(f"- 「{src_label}」 -> 「{dst_label}」 (重要性: {importance:.4f})")
    
    explanation.append("")
    
    # 添加总结
    explanation.append("【推理结论】")
    if important_paths:
        best_path = important_paths[0]['path']
        
        # 构建推理结论
        conclusion = "根据GNN软掩码分析，"
        
        source_node = node_labels[best_path[0]] if best_path[0] < len(node_labels) else f"节点{best_path[0]}"
        target_node = node_labels[best_path[-1]] if best_path[-1] < len(node_labels) else f"节点{best_path[-1]}"
        
        if len(best_path) <= 3:
            # 简单路径
            conclusion += f"从{source_node}可以直接推导出{target_node}。"
        else:
            # 复杂路径，包含中间节点
            middle_nodes = []
            for i in range(1, len(best_path)-1):
                node_idx = best_path[i]
                node_label = node_labels[node_idx] if node_idx < len(node_labels) else f"节点{node_idx}"
                middle_nodes.append(node_label)
            
            middle_text = "，".join(middle_nodes)
            conclusion += f"从{source_node}出发，经过{middle_text}，最终到达{target_node}。"
        
        explanation.append(conclusion)
        explanation.append(f"总体重要性评分: {important_paths[0]['importance']:.4f}")
    else:
        explanation.append("无法确定明确的推理路径。")
    
    return "\n".join(explanation)

def batch_inference(model, graph_dir, output_dir, device, visualize=False, batch_size=16):
    """
    批量执行推理
    
    参数:
    - model: 模型（如果为None则创建新模型）
    - graph_dir: 图目录
    - output_dir: 输出目录
    - device: 设备
    - visualize: 是否可视化
    - batch_size: 批大小
    
    返回:
    - 准确率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要可视化，创建可视化目录
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # 获取图文件列表
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.json')]
    
    # 统计信息
    correct = 0
    total = 0
    
    # 批量处理
    for i in tqdm(range(0, len(graph_files), batch_size), desc="Processing graphs"):
        batch_files = graph_files[i:i+batch_size]
        
        for graph_file in batch_files:
            # 加载图数据
            graph_path = os.path.join(graph_dir, graph_file)
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
            except Exception as e:
                print(f"加载 {graph_file} 失败: {e}")
                continue
            
            # 执行推理
            try:
                start_time = time.time()
                
                result = inference_single_graph(
                    model, 
                    graph_data, 
                    device,
                    visualize=visualize
                )
                
                # 添加时间信息
                result['inference_time'] = time.time() - start_time
                
                # 如果有标签，检查预测是否正确
                if 'label' in graph_data:
                    label = graph_data['label']
                    prediction = result['prediction']
                    if prediction == label:
                        correct += 1
                    total += 1
                    
                    # 添加标签信息
                    result['true_label'] = label
                    result['correct'] = prediction == label
                
                # 保存结果
                output_file = os.path.join(output_dir, f"result_{graph_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 如果需要可视化，保存图像
                if visualize:
                    plt.savefig(os.path.join(vis_dir, f"viz_{os.path.splitext(graph_file)[0]}.png"))
                    plt.close()
            
            except Exception as e:
                print(f"处理 {graph_file} 失败: {e}")
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 保存汇总结果
    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"批量推理完成。准确率: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

def path_reasoning(model, graph_data, device):
    """实现基于路径的推理解释"""
    # 准备图数据
    try:
        # 尝试导入全局build_graph函数
        from utils.graph_utils import build_graph
        g, node_feats, edge_weights, meta_info = build_graph(graph_data)
    except (ImportError, AttributeError):
        # 如果导入失败，使用本地定义的函数
        g, node_feats, edge_weights = prepare_graph_for_inference(graph_data, device)
        meta_info = {
            'question_idx': get_question_idx(graph_data),
            'candidate_idxs': get_candidate_idxs(graph_data)
        }
    
    # 将数据移至设备
    g = g.to(device)
    node_feats = node_feats.to(device)
    edge_weights = edge_weights.to(device)
    
    # 获取节点嵌入和边掩码
    node_embeds, edge_masks, _ = model(g, node_feats, edge_weights)
    
    # 获取问题和答案节点
    question_idx = meta_info['question_idx']
    answer_idxs = meta_info['candidate_idxs']
    
    # 构建带权重的图（权重为边掩码的互补）
    G = nx.DiGraph()
    for i in range(len(node_feats)):
        G.add_node(i)
    
    edge_src, edge_dst = g.edges()
    for i in range(len(edge_src)):
        src, dst = edge_src[i].item(), edge_dst[i].item()
        mask_val = edge_masks[i].item() if i < len(edge_masks) else 0.5
        # 权重为掩码的互补，使重要边权重低
        G.add_edge(src, dst, weight=1.0-mask_val)
    
    # 找出从问题到答案的最短路径
    paths = []
    explanations = []
    
    for ans_idx in answer_idxs:
        try:
            path = nx.shortest_path(G, question_idx, ans_idx, weight='weight')
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # 修复：安全地获取边掩码值
            path_importance = 0.0
            path_edge_count = 0
            
            for s, t in path_edges:
                try:
                    # 修复：使用正确的方式获取边索引
                    edge_id = g.edge_ids(s, t)
                    if edge_id < len(edge_masks):
                        path_importance += edge_masks[edge_id].item()
                        path_edge_count += 1
                except:
                    # 如果边索引获取失败，使用默认值
                    path_importance += 0.5
                    path_edge_count += 1
            
            avg_importance = path_importance / path_edge_count if path_edge_count > 0 else 0.0
            
            # 修复：确保generate_explanation函数定义
            explanation = generate_path_explanation(graph_data, path, avg_importance)
            
            paths.append(path)
            explanations.append(explanation)
        except nx.NetworkXNoPath:
            # 使用fallback方案，例如忽略权重的最短路径
            try:
                path = nx.shortest_path(G, source=question_idx, target=ans_idx)
            except:
                path = []
    
    return paths, explanations

# 添加缺失的函数
def generate_path_explanation(graph_data, path, importance):
    """生成路径解释文本"""
    nodes = graph_data.get('nodes', [])
    explanation = "推理路径:\n"
    
    for i, node_idx in enumerate(path):
        if 0 <= node_idx < len(nodes):
            node = nodes[node_idx]
            node_text = node.get('surface_text', f"节点{node_idx}")
            node_role = node.get('role', '')
            
            # 添加前缀标记节点角色
            prefix = ""
            if node_role == 'question':
                prefix = "[问题] "
            elif node_role == 'answer':
                prefix = "[答案] "
            elif node_role == 'evidence':
                prefix = "[证据] "
            
            # 截断长文本
            if len(node_text) > 50:
                node_text = node_text[:47] + "..."
            
            explanation += f"{i+1}. {prefix}{node_text}\n"
            
            # 如果不是最后一个节点，添加连接符
            if i < len(path) - 1:
                explanation += "   ↓\n"
    
    explanation += f"\n路径重要性评分: {importance:.4f}"
    return explanation

def get_important_edges(model, graph_data, device, threshold=0.5):
    """获取模型认为重要的边"""
    # 准备图数据
    try:
        from utils.graph_utils import build_graph
        g, node_feats, edge_weights, _ = build_graph(graph_data)
    except ImportError:
        g, node_feats, edge_weights = prepare_graph_for_inference(graph_data, device)
    
    g = g.to(device)
    node_feats = node_feats.to(device)
    edge_weights = edge_weights.to(device) if edge_weights is not None else None
    
    # 获取边掩码
    with torch.no_grad():
        _, edge_masks, _ = model(g, node_feats, edge_weights)
    
    # 获取重要边
    important_edges = []
    edge_src, edge_dst = g.edges()
    
    for i in range(len(edge_src)):
        if i < len(edge_masks) and edge_masks[i].item() > threshold:
            src, dst = edge_src[i].item(), edge_dst[i].item()
            importance = edge_masks[i].item()
            
            # 获取节点信息
            src_info = get_node_info(graph_data, src)
            dst_info = get_node_info(graph_data, dst)
            
            important_edges.append({
                'src': src,
                'dst': dst,
                'importance': importance,
                'src_info': src_info,
                'dst_info': dst_info
            })
    
    # 按重要性排序
    important_edges.sort(key=lambda x: x['importance'], reverse=True)
    return important_edges

def get_node_info(graph_data, node_idx):
    """获取节点信息"""
    nodes = graph_data.get('nodes', [])
    if 0 <= node_idx < len(nodes):
        node = nodes[node_idx]
        return {
            'text': node.get('surface_text', f"节点{node_idx}"),
            'role': node.get('role', 'unknown')
        }
    return {'text': f"节点{node_idx}", 'role': 'unknown'}

def prepare_graph_for_inference(graph_data, device):
    """从图数据构建DGL图"""
    # 提取节点和边
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # 提取节点特征
    node_features = []
    for node in nodes:
        if 'feat' in node and node['feat'] != 'PLACEHOLDER':
            try:
                feat = np.array(node['feat'], dtype=np.float32)
            except:
                feat = np.random.rand(768).astype(np.float32)
        else:
            feat = np.random.rand(768).astype(np.float32)
        node_features.append(feat)
    
    # 提取边
    src_ids = []
    dst_ids = []
    edge_weights = []
    
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        src_ids.append(src_id)
        dst_ids.append(dst_id)
        
        # 尝试获取edge_prior作为权重
        weight = edge.get('edge_prior', edge.get('weight', 1.0))
        edge_weights.append(weight)
    
    # 创建DGL图
    g = dgl.graph((src_ids, dst_ids), num_nodes=len(nodes))
    
    # 转换为张量
    node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32) if edge_weights else None
    
    return g, node_features, edge_weights

def ensure_tensor_on_device(tensor, device):
    """确保tensor在指定设备上"""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        return tensor.to(device)
    return tensor

def get_question_idx(graph_data):
    """获取问题节点索引"""
    nodes = graph_data.get('nodes', [])
    for i, node in enumerate(nodes):
        if node.get('role', '') == 'question':
            return i
    # 如果没找到问题节点，默认返回0
    return 0 if nodes else -1

def get_candidate_idxs(graph_data):
    """获取候选答案节点索引列表"""
    nodes = graph_data.get('nodes', [])
    candidates = []
    
    # 首先查找答案节点
    for i, node in enumerate(nodes):
        if node.get('role', '') == 'answer':
            candidates.append(i)
    
    # 如果没有明确的答案节点，查找证据节点
    if not candidates:
        for i, node in enumerate(nodes):
            if node.get('role', '') == 'evidence':
                candidates.append(i)
    
    # 如果还是没有候选，使用除问题节点外的所有节点
    if not candidates:
        question_idx = get_question_idx(graph_data)
        candidates = [i for i in range(len(nodes)) if i != question_idx]
    
    return candidates

def main():
    parser = argparse.ArgumentParser(description="路线1：GNN软掩码推理")
    parser.add_argument('--graph_dir', type=str, required=True, help='图数据目录')
    parser.add_argument('--model_path', type=str, default='', help='模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs/route1_results', help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--single_file', type=str, default='', help='单个文件推理')
    args = parser.parse_args()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型（如果提供了模型路径）
    model = None
    if args.model_path and os.path.exists(args.model_path):
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            node_dim = 768  # 默认节点特征维度
            model = GNNSoftMask(node_dim=node_dim, hidden_dim=args.hidden_dim).to(device)
            model.load_state_dict(state_dict)
            print(f"成功加载模型: {args.model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用初始化模型")
    
    # 单个文件推理或批量推理
    if args.single_file:
        graph_path = os.path.join(args.graph_dir, args.single_file) if not os.path.isabs(args.single_file) else args.single_file
        
        try:
            # 加载图数据
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 执行推理
            result = inference_single_graph(
                model,
                graph_data,
                device,
                visualize=args.visualize
            )
            
            # 保存结果
            output_file = os.path.join(args.output_dir, f"result_{os.path.basename(args.single_file)}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 打印结果
            print(f"推理结果: {result['prediction']}")
            print(f"置信度: {result['score']:.4f}")
            print(f"解释: \n{result['explanation']}")
            
            # 如果需要可视化，显示图像
            if args.visualize:
                plt.show()
        
        except Exception as e:
            print(f"处理文件失败: {e}")
    else:
        # 批量推理
        accuracy = batch_inference(
            model,
            args.graph_dir,
            args.output_dir,
            device,
            visualize=args.visualize,
            batch_size=args.batch_size
        )
        
        print(f"批量推理完成。准确率: {accuracy:.4f}")
        print(f"结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main() 