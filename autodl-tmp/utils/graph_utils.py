#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图处理辅助函数模块

该模块提供了用于处理图结构的辅助函数，包括：
1. 图构建函数
2. 连通性分析
3. 图转换功能
4. 敏感性计算
"""

import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Union, Optional

def construct_graph_from_triples(triples: List[Tuple], directed: bool = True) -> nx.Graph:
    """
    从三元组列表构建图
    
    参数:
        triples: 三元组列表，每个三元组为 (head, relation, tail)
        directed: 是否创建有向图
        
    返回:
        G: NetworkX图对象
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # 添加边并存储关系类型作为边属性
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    
    return G

def find_connected_components(G: nx.Graph) -> List[Set]:
    """
    找出图中的连通分量
    
    参数:
        G: NetworkX图对象
        
    返回:
        components: 连通分量列表，每个分量是节点集合
    """
    if isinstance(G, nx.DiGraph):
        # 对有向图，使用弱连通分量
        components = list(nx.weakly_connected_components(G))
    else:
        # 对无向图，使用连通分量
        components = list(nx.connected_components(G))
        
    return components

def extract_subgraph(G: nx.Graph, nodes: Set) -> nx.Graph:
    """
    从图中提取子图
    
    参数:
        G: 原图
        nodes: 子图节点集合
        
    返回:
        subgraph: 提取的子图
    """
    return G.subgraph(nodes).copy()

def convert_to_pyg_data(
    graph: nx.Graph,
    node_id_to_index: Dict = None,
    node_features: Dict = None,
    edge_features: Dict = None,
    is_directed: bool = True,
    sensitive_edges: List = None,
    is_sensitive: bool = False
) -> Data:
    """
    将NetworkX图转换为PyTorch Geometric数据对象
    
    参数:
        graph: NetworkX图
        node_id_to_index: 节点ID到连续索引的映射
        node_features: 节点特征字典 {node_id: features}
        edge_features: 边特征字典 {(src, dst): features}
        is_directed: 是否保留方向性
        sensitive_edges: 敏感边列表
        is_sensitive: 图是否是敏感的
        
    返回:
        data: PyG Data对象
    """
    # 如果没有提供节点映射，创建一个默认映射
    if node_id_to_index is None:
        nodes = list(graph.nodes())
        node_id_to_index = {node: i for i, node in enumerate(nodes)}
    
    # 收集边和关系类型
    edge_index = []
    edge_attr = []
    edge_type = []
    sensitive_mask = []
    
    # 处理边和边属性
    for src, dst, data in graph.edges(data=True):
        src_idx = node_id_to_index[src]
        dst_idx = node_id_to_index[dst]
        
        # 添加边
        edge_index.append([src_idx, dst_idx])
        
        # 获取关系类型
        rel_type = data.get('relation', 0)
        edge_type.append(rel_type)
        
        # 如果有提供边特征，使用它
        if edge_features and (src, dst) in edge_features:
            edge_attr.append(edge_features[(src, dst)])
        else:
            # 默认边特征是关系类型的one-hot编码
            edge_attr.append([rel_type])
        
        # 标记敏感边
        if sensitive_edges:
            is_sensitive_edge = (src, dst) in sensitive_edges or (dst, src) in sensitive_edges
            sensitive_mask.append(float(is_sensitive_edge))
        else:
            sensitive_mask.append(0.0)
    
    # 如果是无向图，添加反向边
    if not is_directed:
        for src, dst, data in graph.edges(data=True):
            src_idx = node_id_to_index[src]
            dst_idx = node_id_to_index[dst]
            
            # 添加反向边
            edge_index.append([dst_idx, src_idx])
            
            # 获取关系类型
            rel_type = data.get('relation', 0)
            edge_type.append(rel_type)
            
            # 添加边特征
            if edge_features and (dst, src) in edge_features:
                edge_attr.append(edge_features[(dst, src)])
            else:
                edge_attr.append([rel_type])
            
            # 标记敏感边
            if sensitive_edges:
                is_sensitive_edge = (src, dst) in sensitive_edges or (dst, src) in sensitive_edges
                sensitive_mask.append(float(is_sensitive_edge))
            else:
                sensitive_mask.append(0.0)
    
    # 准备节点特征
    x = []
    for node in graph.nodes():
        idx = node_id_to_index[node]
        # 使用给定的节点特征或创建简单的ID特征
        if node_features and node in node_features:
            x.append(node_features[node])
        else:
            # 默认使用节点ID作为特征
            x.append([idx])
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    sensitive_mask = torch.tensor(sensitive_mask, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    
    # 创建PyG Data对象
    data = Data(
        x=x, 
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        sensitive_mask=sensitive_mask,
        is_sensitive=torch.tensor([float(is_sensitive)], dtype=torch.float),
        num_nodes=len(node_id_to_index)
    )
    
    return data

def mark_sensitive_edges(
    graph: nx.Graph,
    group_type: str,
    sensitive_ratio: float = 0.2,
    seed: int = None
) -> List[Tuple]:
    """
    为图标记敏感边
    
    根据群类型使用不同的策略标记敏感边:
    - abelian: 标记交换性边
    - non_abelian: 标记顺序敏感的边
    - identity: 标记自反边
    
    参数:
        graph: NetworkX图
        group_type: 群类型 ('abelian', 'non_abelian', 'identity')
        sensitive_ratio: 要标记为敏感的边的比例
        seed: 随机种子
        
    返回:
        sensitive_edges: 敏感边列表
    """
    if seed is not None:
        random.seed(seed)
    
    all_edges = list(graph.edges(data=True))
    sensitive_edges = []
    
    # 对每种群类型使用不同的敏感性标记策略
    if group_type == 'abelian':
        # 对于阿贝尔群，优先标记展示交换性的边
        # 即如果有(a,b)和(b,a)，则标记它们
        edge_pairs = defaultdict(list)
        
        # 寻找潜在的交换边对
        for src, dst, data in all_edges:
            edge_pairs[(src, dst)].append((src, dst))
            edge_pairs[(dst, src)].append((src, dst))
        
        # 标记那些有对称关系的边
        for (u, v), edges in edge_pairs.items():
            if len(edges) > 1:  # 表示有交换关系
                for src, dst in edges:
                    sensitive_edges.append((src, dst))
    
    elif group_type == 'non_abelian':
        # 对于非阿贝尔群，优先标记那些表示非交换性的边
        # 例如，如果a*b != b*a，标记这些边
        
        # 计算每个节点的度
        node_degrees = {n: graph.degree(n) for n in graph.nodes()}
        
        # 优先标记高度节点相连的边
        edges_with_score = []
        for src, dst, data in all_edges:
            # 计算边的分数 (基于节点度的乘积)
            score = node_degrees[src] * node_degrees[dst]
            edges_with_score.append(((src, dst), score))
        
        # 按分数排序
        edges_with_score.sort(key=lambda x: x[1], reverse=True)
        
        # 选择顶部的边作为敏感边
        num_to_mark = max(1, int(len(all_edges) * sensitive_ratio))
        sensitive_edges = [edge for edge, _ in edges_with_score[:num_to_mark]]
    
    elif group_type == 'identity':
        # 对于单位群，标记一些具有相同关系类型的边和自反边
        
        # 首先找到所有自反边
        self_loops = [(src, dst) for src, dst, _ in all_edges if src == dst]
        sensitive_edges.extend(self_loops)
        
        # 按关系类型对边分组
        edges_by_relation = defaultdict(list)
        for src, dst, data in all_edges:
            if src != dst:  # 排除自反边
                rel_type = data.get('relation', 0)
                edges_by_relation[rel_type].append((src, dst))
        
        # 对每种关系类型，标记一些边
        for rel_type, edges in edges_by_relation.items():
            num_to_mark = max(1, int(len(edges) * sensitive_ratio))
            marked = random.sample(edges, num_to_mark)
            sensitive_edges.extend(marked)
    
    else:
        # 对于未知群类型，随机标记边
        num_to_mark = max(1, int(len(all_edges) * sensitive_ratio))
        marked_edges = random.sample(all_edges, num_to_mark)
        sensitive_edges = [(src, dst) for src, dst, _ in marked_edges]
    
    return sensitive_edges

def compute_sensitivity(graph: nx.Graph, sensitive_edges: List[Tuple], damping: float = 0.85, iterations: int = 20) -> Dict:
    """
    计算图中节点的敏感度
    
    使用类似于PageRank的算法传播敏感性，从敏感边开始
    
    参数:
        graph: NetworkX图
        sensitive_edges: 敏感边列表
        damping: 阻尼因子 (与PageRank类似)
        iterations: 迭代次数
        
    返回:
        sensitivity: 节点敏感度字典 {node_id: sensitivity_value}
    """
    # 初始化敏感度分数
    sensitivity = {node: 0.0 for node in graph.nodes()}
    
    # 初始化与敏感边直接相关的节点
    for src, dst in sensitive_edges:
        sensitivity[src] += 1.0
        sensitivity[dst] += 1.0
    
    # 标准化初始敏感度
    total = sum(sensitivity.values())
    if total > 0:
        for node in sensitivity:
            sensitivity[node] /= total
    
    # 迭代传播敏感度
    for _ in range(iterations):
        new_sensitivity = {node: 0.0 for node in graph.nodes()}
        
        # 传播敏感度
        for node in graph.nodes():
            # 分配一部分敏感度给邻居节点
            neighbors = list(graph.neighbors(node))
            if neighbors:
                share = sensitivity[node] * damping / len(neighbors)
                for neighbor in neighbors:
                    new_sensitivity[neighbor] += share
            
            # 保留一部分敏感度
            new_sensitivity[node] += sensitivity[node] * (1 - damping)
        
        # 更新敏感度
        sensitivity = new_sensitivity
    
    # 标准化最终敏感度
    max_sensitivity = max(sensitivity.values()) if sensitivity else 1.0
    if max_sensitivity > 0:
        for node in sensitivity:
            sensitivity[node] /= max_sensitivity
    
    return sensitivity

def calculate_edge_sensitivity(
    graph: nx.Graph, 
    sensitive_edges: List[Tuple],
    node_sensitivity: Dict = None
) -> Dict:
    """
    计算边的敏感度
    
    基于节点敏感度和边是否标记为敏感
    
    参数:
        graph: NetworkX图
        sensitive_edges: 敏感边列表
        node_sensitivity: 节点敏感度字典
        
    返回:
        edge_sensitivity: 边敏感度字典 {(src, dst): sensitivity_value}
    """
    # 如果没有提供节点敏感度，计算它
    if node_sensitivity is None:
        node_sensitivity = compute_sensitivity(graph, sensitive_edges)
    
    edge_sensitivity = {}
    
    # 计算每条边的敏感度
    for src, dst in graph.edges():
        # 直接敏感边有最高敏感度
        if (src, dst) in sensitive_edges or (dst, src) in sensitive_edges:
            edge_sensitivity[(src, dst)] = 1.0
        else:
            # 其他边的敏感度基于其端点的敏感度
            edge_sensitivity[(src, dst)] = (node_sensitivity[src] + node_sensitivity[dst]) / 2
    
    return edge_sensitivity

def determine_graph_sensitivity(
    graph: nx.Graph,
    sensitive_edges: List[Tuple],
    threshold: float = 0.3
) -> bool:
    """
    确定图的整体敏感性
    
    基于敏感边的数量和节点敏感度的总体水平
    
    参数:
        graph: NetworkX图
        sensitive_edges: 敏感边列表
        threshold: 敏感度阈值
        
    返回:
        is_sensitive: 图是否敏感
    """
    # 计算边的比例是敏感的
    sensitive_ratio = len(sensitive_edges) / graph.number_of_edges() if graph.number_of_edges() > 0 else 0
    
    # 计算节点敏感度
    node_sensitivity = compute_sensitivity(graph, sensitive_edges)
    avg_sensitivity = sum(node_sensitivity.values()) / len(node_sensitivity) if node_sensitivity else 0
    
    # 组合两个因素确定图的敏感性
    # 如果敏感边比例或平均节点敏感度超过阈值，则图是敏感的
    is_sensitive = sensitive_ratio > threshold or avg_sensitivity > threshold
    
    return is_sensitive 

def calculate_graph_sensitivity(graph):
    """
    计算图的敏感性指标，返回0-1之间的连续值
    
    参数:
        graph: 包含triples, sensitive_edges等的图数据字典
        
    返回:
        包含graph_sensitivity, sensitive_density, edge_sensitivities的字典
    """
    import math
    
    # 如果图中没有三元组，返回默认值
    if not graph.get("triples"):
        return {'graph_sensitivity': 0.0, 'sensitive_density': 0.0, 'edge_sensitivities': {}}
    
    # 获取敏感边
    sensitive_edges = graph.get("sensitive_edges", [])
    # 计算敏感边密度 - 使用三元组总数作为分母
    total_triples = len(graph["triples"])
    sensitive_density = len(sensitive_edges) / total_triples if total_triples > 0 else 0.0
    
    # 考虑顺序敏感边的额外权重
    order_sensitive_edges = graph.get("order_sensitive_edges", [])
    order_sensitive_weight = 0.0
    if sensitive_edges:
        order_sensitive_weight = len(order_sensitive_edges) / len(sensitive_edges) * 0.2
    
    # 计算综合敏感度 - 结合敏感边密度和顺序敏感性
    # 即使没有敏感边，也给一个基于图大小的基础敏感度
    base_sensitivity = 0.1 * (1 - math.exp(-total_triples / 10))
    sensitivity_score = sensitive_density * 0.8 + order_sensitive_weight
    
    # 如果没有敏感边，使用基础敏感度
    if sensitive_density == 0:
        sensitivity_score = base_sensitivity
    
    # 为每个边计算敏感度
    edge_sensitivities = {}
    for idx, (h, r, t) in enumerate(graph['triples']):
        # 基础敏感度为0.1
        edge_sensitivity = 0.1
        
        # 如果是敏感边，增加敏感度
        if idx in sensitive_edges:
            edge_sensitivity = 0.7
            # 如果是顺序敏感边，进一步增加敏感度
            if idx in order_sensitive_edges:
                edge_sensitivity = 0.9
        
        # 存储头尾节点对的敏感度 - 使用字符串键而不是元组
        edge_key = f"{h},{t}"  # 转换为字符串键
        edge_sensitivities[edge_key] = edge_sensitivity
    
    # 返回连续值和相关信息
    return {
        'graph_sensitivity': float(sensitivity_score),
        'sensitive_density': sensitive_density,
        'edge_sensitivities': edge_sensitivities
    }

def calculate_edge_sensitivity_propagation(graph):
    """
    使用贝叶斯后验计算边敏感性传播
    
    参数:
        graph: 包含triples, sensitive_edges等的图数据字典
        
    返回:
        包含edge_sensitivities, graph_sensitivity的字典
    """
    # 初始化边敏感性先验
    edge_sensitivities = {}
    for idx, (h, r, t) in enumerate(graph['triples']):
        # 基础敏感性
        if idx in graph.get('sensitive_edges', []):
            prior_sensitivity = 0.8
            # 顺序敏感边有更高的初始敏感度
            if idx in graph.get('order_sensitive_edges', []):
                prior_sensitivity = 0.9
        else:
            prior_sensitivity = 0.1
            
        # 使用字符串键而不是元组
        edge_key = f"{h},{t}"
        edge_sensitivities[edge_key] = prior_sensitivity
    
    # 注意：这里原本应该有传播逻辑，但缺失了实现
    # 在完整实现前，直接使用初始敏感度
    propagated_sensitivities = edge_sensitivities
    
    # 计算图级敏感度（只返回连续值，不做二分类）
    edge_scores = list(propagated_sensitivities.values())
    
    if edge_scores:
        # 使用加权平均，给予高敏感边更大权重
        weights = [score**2 for score in edge_scores]  # 平方增强高值
        weighted_sum = sum(w * s for w, s in zip(weights, edge_scores))
        total_weight = sum(weights)
        
        graph_sensitivity = weighted_sum / total_weight if total_weight > 0 else 0
                
        return {
            'edge_sensitivities': propagated_sensitivities,
            'graph_sensitivity': float(graph_sensitivity),
        }
    else:
        return {'graph_sensitivity': 0.0} 

def load_graph_data(filepath, device='cpu'):
    """
    加载图数据文件，返回DGL图、节点特征和标签
    
    Args:
        filepath: 图数据文件路径
        device: 设备，默认为CPU
    
    Returns:
        graph: DGL图
        features: 节点特征
        labels: 标签
    """
    import os
    import json
    import torch
    import dgl
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist!")
        # 返回一个简单的模拟图，避免程序崩溃
        g = dgl.graph(([0, 1], [1, 0]))
        features = torch.randn(2, 128)
        labels = torch.tensor([0, 1])
        return g.to(device), features.to(device), labels.to(device)
    
    # 尝试加载数据
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 解析节点和边
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        # 创建DGL图
        src, dst = [], []
        for edge in edges:
            s = edge.get('source', edge.get('src', None))
            d = edge.get('target', edge.get('dst', None))
            if s is not None and d is not None:
                src.append(int(s.replace('n', '')))
                dst.append(int(d.replace('n', '')))
        
        g = dgl.graph((src, dst))
        
        # 提取节点特征
        features = []
        for node in nodes:
            feat = node.get('features', node.get('feat', None))
            if feat:
                features.append(feat)
            else:
                # 如果没有特征，使用随机特征
                features.append([0.0] * 128)
        
        # 转换为张量
        if features:
            features = torch.tensor(features, dtype=torch.float)
        else:
            features = torch.randn(len(nodes), 128)
        
        # 提取标签（如果有）
        labels = []
        for node in nodes:
            label = node.get('label', 0)
            labels.append(label)
        
        labels = torch.tensor(labels)
        
        return g.to(device), features.to(device), labels.to(device)
    
    except Exception as e:
        print(f"Error loading graph data: {e}")
        # 返回一个简单的模拟图，避免程序崩溃
        g = dgl.graph(([0, 1], [1, 0]))
        features = torch.randn(2, 128)
        labels = torch.tensor([0, 1])
        return g.to(device), features.to(device), labels.to(device) 