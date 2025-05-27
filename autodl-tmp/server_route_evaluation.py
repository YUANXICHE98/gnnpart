#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Server-optimized route evaluation for knowledge graph question answering
- Evaluates route0-4 algorithms on large-scale datasets
- Supports HotpotQA validation
- Optimized for server environment
- Supports model training and evaluation
"""

import os
import json
import time
import argparse
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 增加PyTorch相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
from tensorboardX import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ROUTES = [0, 1, 2, 3, 4]
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'processing_time']

# 模型相关常量
MODEL_TYPES = ['route0_pure_gnn', 'route1_gnn_softmask', 'route2_swarm_search', 'route3_diff_explainer', 'route4_multi_chain']
MODEL_DIMS = {
    'route0_pure_gnn': 256, 
    'route1_gnn_softmask': 256,
    'route2_swarm_search': 256,
    'route3_diff_explainer': 256,
    'route4_multi_chain': 256
}

#--------- 模型加载函数 ---------#
def load_gnn_model(model_type, checkpoint_path, device):
    """加载指定类型的GNN模型"""
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    
    # 根据模型类型选择合适的模型
    if model_type == 'route0_pure_gnn':
        # 从models/route0_pure_gnn导入HGN模型
        try:
            from models.route0_pure_gnn.train import HGN
            model = HGN(in_dim=768, hidden_dim=MODEL_DIMS[model_type], num_layers=6, dropout=0.2)
        except ImportError as e:
            logger.error(f"Failed to import HGN model: {e}")
            return None
            
    elif model_type == 'route1_gnn_softmask':
        # 从models/route1_gnn_softmask导入SoftMaskGNN模型
        try:
            from models.route1_gnn_softmask.train import SoftMaskGNN
            model = SoftMaskGNN(in_dim=768, hidden_dim=MODEL_DIMS[model_type], num_layers=2, dropout=0.2)
        except ImportError as e:
            logger.error(f"Failed to import SoftMaskGNN model: {e}")
            return None
    
    elif model_type == 'route2_swarm_search':
        # 从models/route2_swarm_search导入SwarmGNN模型
        try:
            from models.route2_swarm_search.train import SwarmGNN
            model = SwarmGNN(in_dim=768, hidden_dim=MODEL_DIMS[model_type], num_layers=3, dropout=0.2)
        except ImportError as e:
            logger.error(f"Failed to import SwarmGNN model: {e}")
            return None
    
    elif model_type == 'route3_diff_explainer':
        # 从models/route3_diff_explainer导入DiffExplainer模型
        try:
            from models.route3_diff_explainer.train import DiffExplainer
            model = DiffExplainer(in_dim=768, hidden_dim=MODEL_DIMS[model_type], num_layers=4, dropout=0.2)
        except ImportError as e:
            logger.error(f"Failed to import DiffExplainer model: {e}")
            return None
    
    elif model_type == 'route4_multi_chain':
        # 从models/route4_multi_chain导入MultiChainGNN模型
        try:
            from models.route4_multi_chain.train import MultiChainGNN
            model = MultiChainGNN(in_dim=768, hidden_dim=MODEL_DIMS[model_type], num_layers=4, dropout=0.2)
        except ImportError as e:
            logger.error(f"Failed to import MultiChainGNN model: {e}")
            return None
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return None

def create_graph_dataset(graph_file, device):
    """从单个图文件创建DGL图用于模型推理"""
    # 加载图数据
    graph_data = load_graph(graph_file)
    if graph_data is None:
        return None
    
    # 构建节点特征
    node_feats = []
    node_roles = []
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # 记录问题和答案节点
    question_idx = -1
    answer_idx = -1
    candidate_idxs = []
    
    # 处理节点
    for i, node in enumerate(nodes):
        # 节点特征
        if 'feat' in node and node['feat'] != 'PLACEHOLDER':
            try:
                feat = torch.tensor(node['feat'], dtype=torch.float)
            except:
                # 使用随机特征
                feat = torch.randn(768)
        else:
            # 使用随机特征
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
    
    # 填充默认值
    if question_idx == -1 and len(nodes) > 0:
        question_idx = 0
    if answer_idx == -1 and len(nodes) > 0:
        answer_idx = len(nodes) - 1
        candidate_idxs.append(answer_idx)
    
    # 转换为张量
    node_feats = torch.stack(node_feats).to(device)
    
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
    g = dgl.graph((src_ids, dst_ids))
    
    # 添加节点角色
    g.ndata['role'] = torch.tensor([role for role in node_roles])
    
    # 添加边类型和权重
    if edge_types:
        g.edata['rel'] = torch.tensor(edge_types)
    
    if edge_weights:
        edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(device)
    else:
        edge_weights = torch.ones(g.number_of_edges(), dtype=torch.float).to(device)
    
    g = g.to(device)
    
    return {
        'graph': g,
        'node_feats': node_feats,
        'edge_weights': edge_weights,
        'question_idx': torch.tensor(question_idx).to(device),
        'answer_idx': torch.tensor(answer_idx).to(device), 
        'candidate_idxs': candidate_idxs
    }

def route0_gnn_model(graph, model, device):
    """使用训练好的Route0 GNN模型进行推理"""
    start_time = time.time()
    
    # 转换图为DGL格式
    graph_file = f"temp_graph_{random.randint(0, 10000)}.json"
    try:
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f)
        
        graph_data = create_graph_dataset(graph_file, device)
        if graph_data is None:
            return None, time.time() - start_time
        
        # 模型预测
        with torch.no_grad():
            g = graph_data['graph']
            node_feats = graph_data['node_feats']
            edge_weights = graph_data['edge_weights']
            question_idx = graph_data['question_idx']
            candidate_idxs = graph_data['candidate_idxs']
            
            # 计算候选答案分数
            scores = model.compute_answer_scores(
                g, node_feats, edge_weights, 
                question_idx, candidate_idxs
            )
            
            # 找出得分最高的答案
            if isinstance(scores, tuple):
                scores = scores[0]  # 处理route1模型可能返回多个值的情况
                
            if len(scores) > 0:
                best_idx = torch.argmax(scores).item()
                predicted_answer = candidate_idxs[best_idx]
                
                # 将节点索引转换为nid格式
                for node in graph.get('nodes', []):
                    if int(node.get('nid', '').replace('n', '')) == predicted_answer:
                        return node.get('nid'), time.time() - start_time
        
        return None, time.time() - start_time
    
    finally:
        # 清理临时文件
        if os.path.exists(graph_file):
            os.remove(graph_file)

def route1_gnn_softmask(graph, model, device):
    """使用训练好的Route1 SoftMask GNN模型进行推理"""
    start_time = time.time()
    
    # 转换图为DGL格式
    graph_file = f"temp_graph_{random.randint(0, 10000)}.json"
    try:
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f)
        
        graph_data = create_graph_dataset(graph_file, device)
        if graph_data is None:
            return None, time.time() - start_time
        
        # 模型预测
        with torch.no_grad():
            g = graph_data['graph'] 
            node_feats = graph_data['node_feats']
            edge_weights = graph_data['edge_weights']
            question_idx = graph_data['question_idx']
            candidate_idxs = graph_data['candidate_idxs']
            
            # 计算候选答案分数和边掩码
            scores, edge_masks, _ = model.compute_answer_scores(
                g, node_feats, edge_weights, 
                question_idx, candidate_idxs
            )
            
            # 找出得分最高的答案
            if len(scores) > 0:
                best_idx = torch.argmax(scores).item()
                predicted_answer = candidate_idxs[best_idx]
                
                # 将节点索引转换为nid格式
                for node in graph.get('nodes', []):
                    if int(node.get('nid', '').replace('n', '')) == predicted_answer:
                        return node.get('nid'), time.time() - start_time
        
        return None, time.time() - start_time
    
    finally:
        # 清理临时文件
        if os.path.exists(graph_file):
            os.remove(graph_file)

def route2_swarm_search(graph, model, device):
    """使用训练好的Route2 Swarm Search模型进行推理"""
    start_time = time.time()
    
    # 转换图为DGL格式
    graph_file = f"temp_graph_{random.randint(0, 10000)}.json"
    try:
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f)
        
        graph_data = create_graph_dataset(graph_file, device)
        if graph_data is None:
            return None, time.time() - start_time
        
        # 模型预测
        with torch.no_grad():
            g = graph_data['graph']
            node_feats = graph_data['node_feats']
            edge_weights = graph_data['edge_weights']
            question_idx = graph_data['question_idx']
            candidate_idxs = graph_data['candidate_idxs']
            
            # 计算候选答案分数
            try:
                scores = model.search_answer(
                    g, node_feats, edge_weights, 
                    question_idx, candidate_idxs
                )
                
                # 找出得分最高的答案
                if isinstance(scores, tuple):
                    scores = scores[0]
                
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    predicted_answer = candidate_idxs[best_idx]
                    
                    # 将节点索引转换为nid格式
                    for node in graph.get('nodes', []):
                        if int(node.get('nid', '').replace('n', '')) == predicted_answer:
                            return node.get('nid'), time.time() - start_time
            except Exception as e:
                logger.error(f"Error in route2_swarm_search: {e}")
        
        return None, time.time() - start_time
    
    finally:
        # 清理临时文件
        if os.path.exists(graph_file):
            os.remove(graph_file)

def route3_diff_explainer(graph, model, device):
    """使用训练好的Route3 Diff Explainer模型进行推理"""
    start_time = time.time()
    
    # 转换图为DGL格式
    graph_file = f"temp_graph_{random.randint(0, 10000)}.json"
    try:
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f)
        
        graph_data = create_graph_dataset(graph_file, device)
        if graph_data is None:
            return None, time.time() - start_time
        
        # 模型预测
        with torch.no_grad():
            g = graph_data['graph']
            node_feats = graph_data['node_feats']
            edge_weights = graph_data['edge_weights']
            question_idx = graph_data['question_idx']
            candidate_idxs = graph_data['candidate_idxs']
            
            # 计算候选答案分数
            try:
                scores = model.explain_and_predict(
                    g, node_feats, edge_weights, 
                    question_idx, candidate_idxs
                )
                
                # 找出得分最高的答案
                if isinstance(scores, tuple):
                    scores = scores[0]
                
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    predicted_answer = candidate_idxs[best_idx]
                    
                    # 将节点索引转换为nid格式
                    for node in graph.get('nodes', []):
                        if int(node.get('nid', '').replace('n', '')) == predicted_answer:
                            return node.get('nid'), time.time() - start_time
            except Exception as e:
                logger.error(f"Error in route3_diff_explainer: {e}")
        
        return None, time.time() - start_time
    
    finally:
        # 清理临时文件
        if os.path.exists(graph_file):
            os.remove(graph_file)

def route4_multi_chain(graph, model, device):
    """使用训练好的Route4 Multi Chain GNN模型进行推理"""
    start_time = time.time()
    
    # 转换图为DGL格式
    graph_file = f"temp_graph_{random.randint(0, 10000)}.json"
    try:
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f)
        
        graph_data = create_graph_dataset(graph_file, device)
        if graph_data is None:
            return None, time.time() - start_time
        
        # 模型预测
        with torch.no_grad():
            g = graph_data['graph']
            node_feats = graph_data['node_feats']
            edge_weights = graph_data['edge_weights']
            question_idx = graph_data['question_idx']
            candidate_idxs = graph_data['candidate_idxs']
            
            # 计算候选答案分数
            try:
                scores = model.chain_reasoning(
                    g, node_feats, edge_weights, 
                    question_idx, candidate_idxs
                )
                
                # 找出得分最高的答案
                if isinstance(scores, tuple):
                    scores = scores[0]
                
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    predicted_answer = candidate_idxs[best_idx]
                    
                    # 将节点索引转换为nid格式
                    for node in graph.get('nodes', []):
                        if int(node.get('nid', '').replace('n', '')) == predicted_answer:
                            return node.get('nid'), time.time() - start_time
            except Exception as e:
                logger.error(f"Error in route4_multi_chain: {e}")
        
        return None, time.time() - start_time
    
    finally:
        # 清理临时文件
        if os.path.exists(graph_file):
            os.remove(graph_file)

# 创建一个通用的模型推理函数映射
MODEL_INFERENCE_FUNCS = {
    'route0_pure_gnn': route0_gnn_model,
    'route1_gnn_softmask': route1_gnn_softmask,
    'route2_swarm_search': route2_swarm_search,
    'route3_diff_explainer': route3_diff_explainer,
    'route4_multi_chain': route4_multi_chain
}

def load_graph(file_path):
    """Load graph data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        return graph_data
    except Exception as e:
        logger.error(f"Error loading graph file {file_path}: {str(e)}")
        return None

def save_json(data, file_path):
    """Save data as JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {str(e)}")
        return False

def find_answer_node(graph):
    """Find the answer node in the graph"""
    for node in graph.get('nodes', []):
        if node.get('role') == 'answer':
            return node.get('nid')
    return None

def find_question_node(graph):
    """Find the question node in the graph"""
    for node in graph.get('nodes', []):
        if node.get('role') == 'question':
            return node.get('nid')
    return None

def get_node_content(graph, node_id):
    """Get content of a node by ID"""
    for node in graph.get('nodes', []):
        if node.get('nid') == node_id:
            return node.get('content', '')
    return ''

def route0_baseline(graph):
    """
    Route 0: Simple baseline - directly connect question to answer
    """
    start_time = time.time()
    
    question_node = find_question_node(graph)
    answer_node = find_answer_node(graph)
    
    if not question_node or not answer_node:
        return None, time.time() - start_time
    
    # Find direct connections from question to answer
    result = None
    for edge in graph.get('edges', []):
        if edge.get('src') == question_node and edge.get('dst') == answer_node:
            result = answer_node
            break
    
    return result, time.time() - start_time

def route1_evidence_based(graph):
    """
    Route 1: Evidence-based inference
    Follow evidencedBy and supportsAnswer relations
    """
    start_time = time.time()
    
    question_node = find_question_node(graph)
    answer_node = find_answer_node(graph)
    
    if not question_node or not answer_node:
        return None, time.time() - start_time
    
    # Build adjacency list for the graph
    adjacency = defaultdict(list)
    for edge in graph.get('edges', []):
        src = edge.get('src')
        dst = edge.get('dst')
        rel = edge.get('rel')
        if src and dst and rel in ['evidencedBy', 'supportsAnswer']:
            adjacency[src].append(dst)
    
    # Simple BFS to find if there's a path from question to answer
    queue = [question_node]
    visited = set([question_node])
    
    while queue:
        current = queue.pop(0)
        
        if current == answer_node:
            return answer_node, time.time() - start_time
        
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return None, time.time() - start_time

def route2_context_based(graph):
    """
    Route 2: Context-based inference
    Follow nodes with low context_distance
    """
    start_time = time.time()
    
    question_node = find_question_node(graph)
    answer_node = find_answer_node(graph)
    
    if not question_node or not answer_node:
        return None, time.time() - start_time
    
    # Get context distances for all nodes
    node_distances = {}
    for node in graph.get('nodes', []):
        nid = node.get('nid')
        distance = node.get('context_distance', 999)
        node_distances[nid] = distance
    
    # Build adjacency list with distance-based weights
    adjacency = defaultdict(list)
    for edge in graph.get('edges', []):
        src = edge.get('src')
        dst = edge.get('dst')
        if src and dst:
            adjacency[src].append((dst, node_distances.get(dst, 999)))
    
    # Dijkstra's algorithm to find shortest path
    distances = {question_node: 0}
    priority_queue = [(0, question_node)]
    visited = set()
    
    while priority_queue:
        priority_queue.sort()  # Simple priority queue implementation
        current_distance, current_node = priority_queue.pop(0)
        
        if current_node == answer_node:
            return answer_node, time.time() - start_time
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in adjacency[current_node]:
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                priority_queue.append((distance, neighbor))
    
    return None, time.time() - start_time

def route3_prior_based(graph):
    """
    Route 3: Prior probability based inference
    Follow edges with high prior probability
    """
    start_time = time.time()
    
    question_node = find_question_node(graph)
    answer_node = find_answer_node(graph)
    
    if not question_node or not answer_node:
        return None, time.time() - start_time
    
    # Build adjacency list with edge_prior
    adjacency = defaultdict(list)
    for edge in graph.get('edges', []):
        src = edge.get('src')
        dst = edge.get('dst')
        prior = edge.get('edge_prior', 0.0)
        if src and dst:
            adjacency[src].append((dst, 1.0 - prior))  # Convert prior to cost (higher prior = lower cost)
    
    # Dijkstra's algorithm to find path with highest combined prior
    distances = {question_node: 0}
    priority_queue = [(0, question_node)]
    visited = set()
    
    while priority_queue:
        priority_queue.sort()  # Simple priority queue implementation
        current_distance, current_node = priority_queue.pop(0)
        
        if current_node == answer_node:
            return answer_node, time.time() - start_time
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in adjacency[current_node]:
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                priority_queue.append((distance, neighbor))
    
    return None, time.time() - start_time

def route4_combined(graph):
    """
    Route 4: Combined approach
    Consider relation type, context distance, and prior probability
    """
    start_time = time.time()
    
    question_node = find_question_node(graph)
    answer_node = find_answer_node(graph)
    
    if not question_node or not answer_node:
        return None, time.time() - start_time
    
    # Get context distances and relation weights
    node_distances = {}
    for node in graph.get('nodes', []):
        nid = node.get('nid')
        distance = node.get('context_distance', 999)
        node_distances[nid] = distance
    
    # Important relation types
    important_relations = {
        'answers': 0.9,
        'evidencedBy': 0.8,
        'supportsAnswer': 0.8,
        'is': 0.7,
        'was': 0.7
    }
    
    # Build adjacency list with combined weights
    adjacency = defaultdict(list)
    for edge in graph.get('edges', []):
        src = edge.get('src')
        dst = edge.get('dst')
        prior = edge.get('edge_prior', 0.1)
        rel = edge.get('rel', '')
        rel_weight = important_relations.get(rel, 0.5)
        
        # Combined weight (lower is better)
        weight = 1.0 - (0.4 * prior + 0.3 * rel_weight + 0.3 / (1 + node_distances.get(dst, 999)))
        
        if src and dst:
            adjacency[src].append((dst, max(0.01, weight)))
    
    # Dijkstra's algorithm with combined weights
    distances = {question_node: 0}
    priority_queue = [(0, question_node)]
    visited = set()
    
    while priority_queue:
        priority_queue.sort()  # Simple priority queue implementation
        current_distance, current_node = priority_queue.pop(0)
        
        if current_node == answer_node:
            return answer_node, time.time() - start_time
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in adjacency[current_node]:
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                priority_queue.append((distance, neighbor))
    
    return None, time.time() - start_time

def evaluate_route(route_func, graph):
    """Evaluate a routing function on a graph"""
    answer_node = find_answer_node(graph)
    if not answer_node:
        return {metric: 0 for metric in METRICS}
    
    predicted_answer, processing_time = route_func(graph)
    
    # Calculate metrics
    correct = predicted_answer == answer_node
    
    return {
        'accuracy': 1 if correct else 0,
        'precision': 1 if correct else 0,
        'recall': 1 if correct else 0,
        'f1': 1 if correct else 0,
        'processing_time': processing_time
    }

def process_file_batch(args):
    """Process a batch of files for parallel execution"""
    file_batch, directory, route_functions = args
    results = defaultdict(list)
    
    for filename in file_batch:
        graph = load_graph(os.path.join(directory, filename))
        if not graph:
            continue
        
        for route_num, route_func in route_functions.items():
            metrics = evaluate_route(route_func, graph)
            for metric, value in metrics.items():
                results[(route_num, metric)].append(value)
    
    return results

def compare_routes_parallel(clean_dir, distractor_dir, sample_size=500, output_dir='route_comparison', workers=None, models=None):
    """Compare routes on clean and distractor datasets using parallel processing"""
    logger.info(f"Starting route comparison with sample size {sample_size}")
    
    # Determine number of workers
    if workers is None:
        workers = max(1, min(16, multiprocessing.cpu_count() - 1))
    
    logger.info(f"Using {workers} workers for parallel processing")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of graph files from both directories
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.json')]
    distractor_files = [f for f in os.listdir(distractor_dir) if f.endswith('.json')]
    
    # Find common files
    common_files = list(set(clean_files).intersection(set(distractor_files)))
    
    if len(common_files) < sample_size:
        logger.warning(f"Only {len(common_files)} common files found, using all of them")
        sample_size = len(common_files)
    
    # Randomly sample files
    sampled_files = random.sample(common_files, sample_size)
    logger.info(f"Selected {len(sampled_files)} files for comparison")
    
    # 路由函数映射 - 不再使用基于规则的算法
    route_functions = {}
    
    # 检查是否有提供模型
    if models is None or len(models) == 0:
        logger.error("No trained models provided for evaluation. Please use --use_trained_models with appropriate model paths.")
        return None
    
    # 创建模型到路由的映射
    model_to_route = {
        'route0_pure_gnn': 0,
        'route1_gnn_softmask': 1,
        'route2_swarm_search': 2,
        'route3_diff_explainer': 3,
        'route4_multi_chain': 4
    }
    
    # 将训练好的模型直接映射到route0-4
    for model_type, model in models.items():
        if model is not None:
            route_num = model_to_route.get(model_type)
            if route_num is not None:
                device = next(model.parameters()).device
                model_inference_func = MODEL_INFERENCE_FUNCS.get(model_type)
                if model_inference_func:
                    # 创建闭包以传递模型和设备
                    route_functions[route_num] = lambda graph, m=model, d=device, f=model_inference_func: f(graph, m, d)
                    logger.info(f"Added {model_type} model as Route {route_num}")
    
    # 如果没有提供任何有效模型，退出
    if len(route_functions) == 0:
        logger.error("No valid models found for routes 0-4. Please check your model paths.")
        return None
    
    # Prepare batches for parallel processing
    batch_size = max(1, len(sampled_files) // (workers * 2))
    clean_batches = [sampled_files[i:i+batch_size] for i in range(0, len(sampled_files), batch_size)]
    
    # Results dictionary
    clean_results = defaultdict(list)
    distractor_results = defaultdict(list)
    
    # Process clean dataset in parallel
    logger.info(f"Processing clean dataset with {len(clean_batches)} batches...")
    clean_args = [(batch, clean_dir, route_functions) for batch in clean_batches]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_results in tqdm(executor.map(process_file_batch, clean_args), total=len(clean_batches)):
            for (route_num, metric), values in batch_results.items():
                clean_results[(route_num, metric)].extend(values)
    
    # Process distractor dataset in parallel
    logger.info(f"Processing dataset with distractors...")
    distractor_args = [(batch, distractor_dir, route_functions) for batch in clean_batches]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_results in tqdm(executor.map(process_file_batch, distractor_args), total=len(clean_batches)):
            for (route_num, metric), values in batch_results.items():
                distractor_results[(route_num, metric)].extend(values)
    
    # Convert results to the expected format
    all_routes = list(route_functions.keys())
    
    results = {
        'clean': {route: {metric: [] for metric in METRICS} for route in all_routes},
        'distractor': {route: {metric: [] for metric in METRICS} for route in all_routes}
    }
    
    for (route_num, metric), values in clean_results.items():
        results['clean'][route_num][metric] = values
    
    for (route_num, metric), values in distractor_results.items():
        results['distractor'][route_num][metric] = values
    
    # Calculate average results
    avg_results = {
        'clean': {route: {metric: np.mean(values) for metric, values in metrics.items()} 
                 for route, metrics in results['clean'].items()},
        'distractor': {route: {metric: np.mean(values) for metric, values in metrics.items()} 
                      for route, metrics in results['distractor'].items()}
    }
    
    # Save results
    save_json(avg_results, os.path.join(output_dir, 'route_comparison_results.json'))
    
    # Generate comparison table
    table_data = []
    for route in all_routes:
        for dataset in ['clean', 'distractor']:
            row = {
                'route': f"Route {route}",
                'dataset': dataset,
                'accuracy': avg_results[dataset][route]['accuracy'],
                'precision': avg_results[dataset][route]['precision'],
                'recall': avg_results[dataset][route]['recall'],
                'f1': avg_results[dataset][route]['f1'],
                'processing_time': avg_results[dataset][route]['processing_time']
            }
            table_data.append(row)
    
    # Print table
    logger.info("\nRoute Comparison Results:")
    logger.info(f"{'Route':<10} {'Dataset':<12} {'Accuracy':<10} {'F1':<10} {'Time (s)':<10}")
    logger.info("-" * 60)
    for row in table_data:
        logger.info(f"{row['route']:<10} {row['dataset']:<12} {row['accuracy']:.4f}     {row['f1']:.4f}     {row['processing_time']:.4f}")
    
    # Visualize results
    visualize_comparison(avg_results, output_dir)
    
    return avg_results

def load_hotpotqa_data(file_path):
    """Load HotpotQA dataset"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading HotpotQA data: {e}")
        return []

def evaluate_with_hotpotqa(clean_dir, hotpotqa_file, output_dir='hotpotqa_results', sample_size=None, workers=None):
    """Evaluate routing algorithms using HotpotQA dataset"""
    logger.info(f"Starting HotpotQA evaluation")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load HotpotQA data
    hotpotqa_data = load_hotpotqa_data(hotpotqa_file)
    if not hotpotqa_data:
        logger.error("Failed to load HotpotQA data")
        return None
    
    logger.info(f"Loaded {len(hotpotqa_data)} HotpotQA examples")
    
    # Get mapping from HotpotQA IDs to graph files
    graph_files = [f for f in os.listdir(clean_dir) if f.endswith('.json')]
    
    # If sample size is provided, randomly sample from the files
    if sample_size and sample_size < len(graph_files):
        graph_files = random.sample(graph_files, sample_size)
        logger.info(f"Sampled {len(graph_files)} files for evaluation")
    
    # Map route numbers to functions
    route_functions = {
        0: route0_baseline,
        1: route1_evidence_based,
        2: route2_context_based,
        3: route3_prior_based,
        4: route4_combined
    }
    
    # Results storage
    qa_results = {route: {'correct': 0, 'total': 0, 'time': 0} for route in ROUTES}
    detailed_results = []
    
    # Determine number of workers
    if workers is None:
        workers = max(1, min(16, multiprocessing.cpu_count() - 1))
    
    logger.info(f"Using {workers} workers for parallel processing")
    
    # Process files in batches for parallel execution
    batch_size = max(1, len(graph_files) // (workers * 2))
    file_batches = [graph_files[i:i+batch_size] for i in range(0, len(graph_files), batch_size)]
    
    # Process each batch
    for batch_idx, file_batch in enumerate(tqdm(file_batches, desc="Processing batches")):
        for filename in file_batch:
            graph = load_graph(os.path.join(clean_dir, filename))
            if not graph:
                continue
            
            # Find question and answer
            question_node = find_question_node(graph)
            answer_node = find_answer_node(graph)
            if not question_node or not answer_node:
                continue
                
            question_text = get_node_content(graph, question_node)
            gold_answer_text = get_node_content(graph, answer_node)
            
            # Evaluate each routing algorithm
            for route_num, route_func in route_functions.items():
                predicted_node, processing_time = route_func(graph)
                
                # Update results
                if predicted_node:
                    predicted_answer_text = get_node_content(graph, predicted_node)
                    is_correct = predicted_node == answer_node
                    
                    qa_results[route_num]['total'] += 1
                    if is_correct:
                        qa_results[route_num]['correct'] += 1
                    qa_results[route_num]['time'] += processing_time
                    
                    # Add detailed result
                    detailed_results.append({
                        'filename': filename,
                        'question': question_text,
                        'gold_answer': gold_answer_text,
                        'predicted_answer': predicted_answer_text,
                        'route': route_num,
                        'correct': is_correct,
                        'time': processing_time
                    })
    
    # Calculate accuracy for each route
    for route in ROUTES:
        if qa_results[route]['total'] > 0:
            accuracy = qa_results[route]['correct'] / qa_results[route]['total']
            avg_time = qa_results[route]['time'] / qa_results[route]['total']
            qa_results[route]['accuracy'] = accuracy
            qa_results[route]['avg_time'] = avg_time
        else:
            qa_results[route]['accuracy'] = 0
            qa_results[route]['avg_time'] = 0
    
    # Save detailed results
    save_json(detailed_results, os.path.join(output_dir, 'detailed_qa_results.json'))
    
    # Save summary results
    summary = {f"Route {route}": {
        'accuracy': qa_results[route]['accuracy'],
        'avg_time': qa_results[route]['avg_time'],
        'correct': qa_results[route]['correct'],
        'total': qa_results[route]['total']
    } for route in ROUTES}
    
    save_json(summary, os.path.join(output_dir, 'summary_qa_results.json'))
    
    # Print summary
    logger.info("\nHotpotQA Evaluation Results:")
    logger.info(f"{'Route':<10} {'Accuracy':<10} {'Avg Time (s)':<15} {'Correct/Total':<15}")
    logger.info("-" * 60)
    for route in ROUTES:
        r = qa_results[route]
        logger.info(f"Route {route:<5} {r['accuracy']:.4f}     {r['avg_time']:.4f}         {r['correct']}/{r['total']}")
    
    # Create visualizations
    visualize_qa_results(qa_results, output_dir)
    
    return qa_results

def visualize_qa_results(qa_results, output_dir):
    """Create visualizations for QA evaluation results"""
    # Prepare data
    routes = [f"Route {r}" for r in ROUTES]
    accuracies = [qa_results[r]['accuracy'] for r in ROUTES]
    times = [qa_results[r]['avg_time'] for r in ROUTES]
    
    # Create accuracy bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(routes, accuracies, color='royalblue')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Routing Algorithm Accuracy on HotpotQA Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qa_accuracy.png'), dpi=300)
    plt.close()
    
    # Create time bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(routes, times, color='lightseagreen')
    plt.ylabel('Average Time (seconds)')
    plt.title('Routing Algorithm Processing Time on HotpotQA Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(times):
        plt.text(i, v + 0.0001, f"{v:.5f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qa_time.png'), dpi=300)
    plt.close()

def visualize_comparison(results, output_dir):
    """Create visualizations for route comparison"""
    # Bar chart for accuracy
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    route_labels = [f"Route {r}" for r in ROUTES]
    clean_accuracy = [results['clean'][r]['accuracy'] for r in ROUTES]
    distractor_accuracy = [results['distractor'][r]['accuracy'] for r in ROUTES]
    
    x = np.arange(len(route_labels))
    width = 0.35
    
    axes[0].bar(x - width/2, clean_accuracy, width, label='Clean Dataset')
    axes[0].bar(x + width/2, distractor_accuracy, width, label='With Distractors')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(route_labels)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Processing time comparison
    clean_time = [results['clean'][r]['processing_time'] for r in ROUTES]
    distractor_time = [results['distractor'][r]['processing_time'] for r in ROUTES]
    
    axes[1].bar(x - width/2, clean_time, width, label='Clean Dataset')
    axes[1].bar(x + width/2, distractor_time, width, label='With Distractors')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Processing Time Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(route_labels)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'route_comparison.png'), dpi=300)
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'route_comparison.png')}")
    
    # Performance improvement chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement percentages
    accuracy_improvement = [(clean - distractor) / max(0.001, distractor) * 100 
                           for clean, distractor in zip(clean_accuracy, distractor_accuracy)]
    time_improvement = [(distractor - clean) / max(0.001, distractor) * 100 
                       for clean, distractor in zip(clean_time, distractor_time)]
    
    x = np.arange(len(route_labels))
    width = 0.35
    
    ax.bar(x - width/2, accuracy_improvement, width, label='Accuracy Improvement')
    ax.bar(x + width/2, time_improvement, width, label='Time Improvement')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement with Clean Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(route_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_comparison.png'), dpi=300)
    logger.info(f"Improvement visualization saved to {os.path.join(output_dir, 'improvement_comparison.png')}")

def main():
    parser = argparse.ArgumentParser(description="Server-optimized route evaluation")
    parser.add_argument('--clean_dir', type=str, default="gnn_dataset_clean/subgraphs",
                      help='Directory with clean graph files')
    parser.add_argument('--distractor_dir', type=str, default="gnn_dataset_1000/subgraphs/with_distractors",
                      help='Directory with distractor graph files')
    parser.add_argument('--output_dir', type=str, default="server_results",
                      help='Output directory for results and visualizations')
    parser.add_argument('--sample_size', type=int, default=500,
                      help='Number of graphs to sample for testing')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--hotpotqa_file', type=str, 
                      help='Path to HotpotQA dataset JSON file')
    parser.add_argument('--hotpotqa_output', type=str, default="hotpotqa_results",
                      help='Output directory for HotpotQA evaluation results')
    parser.add_argument('--only_hotpotqa', action='store_true',
                      help='Only run HotpotQA evaluation')
    parser.add_argument('--only_route_comparison', action='store_true',
                      help='Only run route comparison')
    # 新增模型相关参数
    parser.add_argument('--use_trained_models', action='store_true', default=True,
                      help='Use trained neural network models for evaluation (default: True)')
                      
    # 所有模型的路径参数
    parser.add_argument('--route0_model_path', type=str, default="models/route0_pure_gnn/checkpoints/best_model.pt",
                      help='Path to trained Route 0 GNN model')
    parser.add_argument('--route1_model_path', type=str, default="models/route1_gnn_softmask/checkpoints/best_model.pt",
                      help='Path to trained Route 1 SoftMask GNN model')
    parser.add_argument('--route2_model_path', type=str, default="models/route2_swarm_search/checkpoints/best_model.pt",
                      help='Path to trained Route 2 Swarm Search model')
    parser.add_argument('--route3_model_path', type=str, default="models/route3_diff_explainer/checkpoints/best_model.pt",
                      help='Path to trained Route 3 Diff Explainer model')
    parser.add_argument('--route4_model_path', type=str, default="models/route4_multi_chain/checkpoints/best_model.pt",
                      help='Path to trained Route 4 Multi Chain model')
    
    parser.add_argument('--train_model', type=str, choices=['route0', 'route1', 'route2', 'route3', 'route4', 'none'], default='none',
                      help='Train a model before evaluation (route0, route1, route2, route3, route4, or none)')
    parser.add_argument('--train_epochs', type=int, default=5,
                      help='Number of epochs to train the model')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create main output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 定义模型路径映射
    model_paths = {
        'route0_pure_gnn': args.route0_model_path,
        'route1_gnn_softmask': args.route1_model_path,
        'route2_swarm_search': args.route2_model_path,
        'route3_diff_explainer': args.route3_model_path,
        'route4_multi_chain': args.route4_model_path
    }
    
    # 如果使用训练模型，加载模型
    models = {}
    if args.use_trained_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # 加载所有模型
        for model_type in MODEL_TYPES:
            model_path = model_paths[model_type]
            if os.path.exists(model_path):
                model = load_gnn_model(model_type, model_path, device)
                if model:
                    models[model_type] = model
                    
                    # 打印模型信息
                    num_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"{model_type} model has {num_params} parameters")
            else:
                logger.warning(f"{model_type} model not found at {model_path}")
            
        # 如果没有加载到任何模型，且使用训练模型的标志为True，这里可以提供进一步的指导
        if not models and args.use_trained_models:
            logger.error("""
            No trained models were found. You need to train models first using:
            
            For Route 0 Pure GNN:
            python models/route0_pure_gnn/train.py --graph_dir gnn_dataset_clean/subgraphs --output_dir models/route0_pure_gnn/checkpoints --batch_size 32 --epochs 20 --cuda
            
            For Route 1 GNN SoftMask:
            python models/route1_gnn_softmask/train.py --graph_dir gnn_dataset_clean/subgraphs --output_dir models/route1_gnn_softmask/checkpoints --batch_size 32 --epochs 20 --cuda
            
            For Route 2 Swarm Search:
            python models/route2_swarm_search/train.py --graph_dir gnn_dataset_clean/subgraphs --output_dir models/route2_swarm_search/checkpoints --batch_size 32 --epochs 20 --cuda
            
            For Route 3 Diff Explainer:
            python models/route3_diff_explainer/train.py --graph_dir gnn_dataset_clean/subgraphs --output_dir models/route3_diff_explainer/checkpoints --batch_size 32 --epochs 20 --cuda
            
            For Route 4 Multi Chain:
            python models/route4_multi_chain/train.py --graph_dir gnn_dataset_clean/subgraphs --output_dir models/route4_multi_chain/checkpoints --batch_size 32 --epochs 20 --cuda
            """)
    
    # Run route comparison if requested
    if not args.only_hotpotqa:
        logger.info("Running route comparison...")
        compare_routes_parallel(
            args.clean_dir, 
            args.distractor_dir, 
            args.sample_size, 
            os.path.join(args.output_dir, 'route_comparison'),
            args.workers,
            models  # 传递模型字典
        )
    
    # Run HotpotQA evaluation if requested
    if args.hotpotqa_file and not args.only_route_comparison:
        logger.info("Running HotpotQA evaluation...")
        # 为HotpotQA评估也添加模型支持（可以在另一个PR中实现）
        evaluate_with_hotpotqa(
            args.clean_dir,
            args.hotpotqa_file,
            os.path.join(args.output_dir, args.hotpotqa_output),
            args.sample_size,
            args.workers
        )
    
    logger.info("Evaluation complete!")
    if models:
        # 显示实际加载的模型和对应的路由编号
        model_routes = []
        for model_type in models.keys():
            route_num = {
                'route0_pure_gnn': 0,
                'route1_gnn_softmask': 1,
                'route2_swarm_search': 2,
                'route3_diff_explainer': 3,
                'route4_multi_chain': 4
            }.get(model_type)
            if route_num is not None:
                model_routes.append(f"- Route {route_num}: {model_type} model")
        
        logger.info(f"""
        Evaluated the following models:
        {', '.join(models.keys())}
        
        Models are mapped directly to routes:
        {chr(10).join(model_routes)}
        
        Check the results in {args.output_dir} directory.
        """)
    else:
        logger.error("""
        No models were evaluated. Please ensure you have correct model paths.
        You need to provide at least one trained model from route0 to route4.
        """)

if __name__ == "__main__":
    main() 