#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线0：GNN+相似度推理 - 推理脚本
使用GNN模型结合节点相似度分析和路径权重计算进行知识图谱推理
包含：基础GNN架构 + 节点相似度计算 + 重要路径分析 + 解释生成
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# GNN+相似度推理模型定义
class PureGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(PureGNN, self).__init__()
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
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features, edge_index):
        """
        前向传播
        
        参数:
        - node_features: 节点特征，形状为 [num_nodes, node_dim]
        - edge_index: 边索引，形状为 [2, num_edges]
        
        返回:
        - prediction: 预测结果
        - node_embeds: 节点嵌入
        """
        # 初始节点嵌入
        node_embeds = self.node_embedding(node_features)
        
        # GNN消息传递
        for i in range(self.num_layers):
            # 收集消息
            messages = torch.zeros_like(node_embeds)
            for j in range(edge_index.size(1)):
                src, dst = edge_index[0, j], edge_index[1, j]
                message = torch.cat([node_embeds[src], node_embeds[dst]], dim=0)
                messages[dst] += self.gnn_layers[i](message)
            
            # 更新节点表示
            node_embeds = node_embeds + messages
        
        # 预测（假设第一个节点是问题节点）
        question_idx = 0  # 问题节点索引
        prediction = self.predictor(torch.cat([node_embeds[question_idx], node_embeds.mean(dim=0)], dim=0))
        
        return prediction, node_embeds

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
        
        # 提取节点标签
        node_labels.append(node.get('value', f'Node_{i}'))
        
        # 记录节点角色
        role = node.get('role', '')
        node_roles[i] = role
        
        if role == 'question':
            question_idx = i
        elif role == 'answer':
            answer_indices.append(i)
    
    # 如果没有找到问题节点，使用第一个节点
    if question_idx == -1 and len(nodes) > 0:
        question_idx = 0
    
    # 提取边
    edge_src = []
    edge_dst = []
    edge_types = []
    
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        edge_src.append(src_id)
        edge_dst.append(dst_id)
        
        # 边类型
        edge_type = edge.get('rel', 'default')
        edge_types.append(edge_type)
    
    # 转换为PyTorch张量
    node_features = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device) if edge_src else torch.zeros((2, 0), dtype=torch.long).to(device)
    
    # 边类型编码（如果有）
    if edge_types:
        # 创建边类型映射
        type_map = {t: i for i, t in enumerate(set(edge_types))}
        edge_type = torch.tensor([type_map[t] for t in edge_types], dtype=torch.long).to(device)
    else:
        edge_type = None
    
    # 创建或使用模型
    if model is None:
        model = PureGNN(node_dim=node_features.size(1), hidden_dim=128).to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 执行推理
    with torch.no_grad():
        prediction, node_embeds = model(node_features, edge_index)
    
    # 将预测转换为概率
    prediction_score = torch.sigmoid(prediction).item()
    predicted_label = prediction_score > 0.5
    
    # 计算节点相似度
    node_embeds_np = node_embeds.cpu().numpy()
    question_embed = node_embeds_np[question_idx] if question_idx < len(node_embeds_np) else node_embeds_np[0]
    
    node_similarities = {}
    for i, embed in enumerate(node_embeds_np):
        if i != question_idx:
            sim = np.dot(embed, question_embed) / (np.linalg.norm(embed) * np.linalg.norm(question_embed) + 1e-8)
            node_similarities[i] = float(sim)
    
    # 计算与答案节点的相似度
    answer_sims = []
    for ans_idx in answer_indices:
        if ans_idx < len(node_embeds_np):
            answer_embed = node_embeds_np[ans_idx]
            sim = np.dot(question_embed, answer_embed) / (np.linalg.norm(question_embed) * np.linalg.norm(answer_embed) + 1e-8)
            answer_sims.append((ans_idx, sim))
    
    # 创建简单路径和注意力分数
    paths = []
    attention_scores = {}
    
    # 基于相似度构建简单路径
    for ans_idx, sim in answer_sims:
        if sim > 0.3:  # 只考虑相似度较高的
            path = [question_idx, ans_idx]
            paths.append((path, sim))
    
    # 分析重要路径
    important_paths = []
    for path, weight in paths:
        if weight > threshold:
            path_nodes = [node_labels[idx] for idx in path if idx < len(node_labels)]
            important_paths.append((path_nodes, weight))
    
    # 可视化
    if visualize:
        visualize_graph(nodes, edges, node_labels, question_idx, answer_indices, 
                       node_similarities, important_paths)
    
    # 生成解释文本
    explanation = generate_explanation(node_labels, question, question_idx, 
                                      answer_indices, node_similarities, important_paths, answer_sims)
    
    # 构建结果
    result = {
        'prediction': bool(predicted_label),
        'score': prediction_score,
        'node_similarities': {str(k): v for k, v in node_similarities.items()},
        'answer_similarities': [(int(idx), float(sim)) for idx, sim in answer_sims],
        'important_paths': [(p, float(w)) for p, w in important_paths],
        'explanation': explanation
    }
    
    return result

def visualize_graph(nodes, edges, node_labels, question_idx, answer_indices, node_similarities, important_paths=None):
    """
    可视化图及节点相似度
    
    参数:
    - nodes: 节点列表
    - edges: 边列表
    - node_labels: 节点标签
    - question_idx: 问题节点索引
    - answer_indices: 答案节点索引列表
    - node_similarities: 节点相似度
    - important_paths: 重要路径
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
    
    # 节点颜色和大小
    node_colors = []
    node_sizes = []
    
    for i in range(len(nodes)):
        if i == question_idx:
            node_colors.append('green')
            node_sizes.append(700)
        elif i in answer_indices:
            node_colors.append('blue')
            node_sizes.append(600)
        else:
            # 根据相似度着色
            similarity = node_similarities.get(i, 0.0)
            node_colors.append(plt.cm.Oranges(0.3 + 0.7 * max(0, similarity)))
            node_sizes.append(300 + 400 * max(0, similarity))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
    
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
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Oranges(0.8), markersize=10)
    ], ['问题节点', '答案节点', '相关节点'], loc='upper right')
    
    plt.title("GNN+相似度推理结果可视化", fontsize=16)
    plt.axis('off')
    
    return plt.gcf()

def generate_explanation(node_labels, question, question_idx, answer_indices, node_similarities, important_paths=None, answer_sims=None):
    """
    生成解释文本
    
    参数:
    - node_labels: 节点标签
    - question: 问题文本
    - question_idx: 问题节点索引
    - answer_indices: 答案节点索引列表
    - node_similarities: 节点相似度
    - important_paths: 重要路径
    - answer_sims: 答案节点相似度列表
    
    返回:
    - 解释文本
    """
    explanation = []
    
    # 添加问题
    explanation.append(f"问题: {question}")
    explanation.append("")
    
    # 添加最相关节点分析
    explanation.append("【关键相关节点】")
    sorted_nodes = sorted(node_similarities.items(), key=lambda x: x[1], reverse=True)
    for i, (node_idx, similarity) in enumerate(sorted_nodes[:5]):  # 最多显示前5个相关节点
        if node_idx < len(node_labels):
            explanation.append(f"- {node_labels[node_idx]} (相似度: {similarity:.4f})")
    
    explanation.append("")
    
    # 添加答案节点分析
    explanation.append("【答案分析】")
    if answer_sims:
        sorted_answers = sorted(answer_sims, key=lambda x: x[1], reverse=True)
    for ans_idx, sim in sorted_answers:
        if ans_idx < len(node_labels):
            explanation.append(f"- {node_labels[ans_idx]} (与问题相似度: {sim:.4f})")
    
    explanation.append("")
    
    # 添加重要路径分析
    explanation.append("【重要路径分析】")
    if important_paths:
        for path, weight in important_paths:
            explanation.append(f"- 路径: {' -> '.join(path)} (权重: {weight:.4f})")
    
    explanation.append("")
    
    # 添加总结
    explanation.append("【推理结论】")
    if answer_sims:
        sorted_answers = sorted(answer_sims, key=lambda x: x[1], reverse=True)
        top_answer_idx, top_sim = sorted_answers[0]
        if top_answer_idx < len(node_labels):
            # 构建推理结论
            question_label = node_labels[question_idx] if question_idx < len(node_labels) else "问题"
            answer_label = node_labels[top_answer_idx]
            
            explanation.append(f"根据GNN分析，{question_label}与{answer_label}之间存在紧密关联，相似度为{top_sim:.4f}。")
            
            # 如果有其他很相关的节点，也提一下
            related_nodes = []
            for node_idx, sim in sorted_nodes[:2]:  # 取前两个最相关的节点
                if node_idx not in answer_indices and sim > 0.5:  # 相似度阈值
                    related_nodes.append(node_labels[node_idx])
            
            if related_nodes:
                explanation.append(f"此外，{', '.join(related_nodes)}等节点对推理也有重要贡献。")
    else:
        explanation.append("无法找到与问题足够相关的答案节点。")
    
    return "\n".join(explanation)

def batch_inference(model, graph_dir, output_dir, device, visualize=False, batch_size=16):
    """
    批量执行推理
    
    参数:
    - model: 训练好的GNN模型（如果为None则创建新模型）
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
    for i in tqdm(range(0, len(graph_files), batch_size), desc="处理图数据"):
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
                
                # 如果有标签，计算准确性
                if 'label' in graph_data:
                    label = graph_data['label']
                    prediction = result['prediction']
                    
                    result['true_label'] = label
                    result['correct'] = prediction == label
                    
                    if prediction == label:
                        correct += 1
                    total += 1
                
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

def main():
    parser = argparse.ArgumentParser(description="路线0：GNN+相似度推理")
    parser.add_argument('--graph_dir', type=str, required=True, help='图数据目录')
    parser.add_argument('--model_path', type=str, default='', help='模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs/route0_results', help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
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
            model = PureGNN(node_dim=node_dim, hidden_dim=args.hidden_dim).to(device)
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