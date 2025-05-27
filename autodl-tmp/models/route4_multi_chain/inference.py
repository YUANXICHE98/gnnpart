#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线4：多链思维 - 推理脚本
使用多链思维方法进行知识图谱推理和路径解释
"""

import os
import sys
import json
import argparse
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入模型定义
from models.route4_multi_chain.train import MultiChainReasoner, get_reasoning_chains

# 单图推理
def inference_single_graph(model, graph_data, device, visualize=False, threshold=0.5):
    """
    对单个图执行推理
    
    参数:
    - model: 训练好的模型
    - graph_data: 图数据
    - device: 设备
    - visualize: 是否可视化
    - threshold: 阈值，用于确定重要边
    
    返回:
    - result: 推理结果
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 提取图数据
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    question = graph_data.get('question', '')
    
    # 找到问题和答案节点
    question_idx = -1
    answer_indices = []
    for i, node in enumerate(nodes):
        role = node.get('role', '')
        if role == 'question':
            question_idx = i
        elif role == 'answer':
            answer_indices.append(i)
    
    # 如果没有找到问题节点，使用第一个节点
    if question_idx == -1 and len(nodes) > 0:
        question_idx = 0
    
    # 准备节点特征
    node_feats = []
    for node in nodes:
        if 'feat' in node and node['feat'] != 'PLACEHOLDER':
            try:
                feat = np.array(node['feat'], dtype=np.float32)
            except:
                # 如果特征有问题，使用随机特征
                feat = np.random.rand(768).astype(np.float32)
        else:
            # 使用随机特征
            feat = np.random.rand(768).astype(np.float32)
        
        node_feats.append(feat)
    
    # 转为numpy数组
    node_feats = np.array(node_feats, dtype=np.float32)
    
    # 准备边信息
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
    
    # 转为tensor
    node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
    edge_pairs = torch.tensor(edge_pairs, dtype=torch.int64).to(device) if edge_pairs else None
    
    # 执行推理
    with torch.no_grad():
        chain_outputs, final_output, attention_weights, prediction = model(node_feats, edge_pairs)
    
    # 获取推理链
    chains_data, chain_weights = get_reasoning_chains(model, {
        'node_feats': node_feats.squeeze(0).cpu().numpy(),
        'edge_pairs': edge_pairs.cpu().numpy() if edge_pairs is not None else np.zeros((0, 2), dtype=np.int64),
        'num_nodes': len(nodes)
    }, device, top_k=min(5, len(nodes)))
    
    # 计算每条边的重要性分数
    edge_importance = {}
    if edge_pairs is not None:
        for i, (src, dst) in enumerate(edge_pairs.cpu().numpy()):
            # 对每个链
            for chain_idx, chain in enumerate(chains_data):
                # 如果源节点和目标节点都在这个链的重要节点中
                if src in chain['top_nodes'] and dst in chain['top_nodes']:
                    # 计算这条边在这个链中的重要性
                    src_idx = chain['top_nodes'].index(src)
                    dst_idx = chain['top_nodes'].index(dst)
                    
                    # 边重要性 = 源节点相似度 * 目标节点相似度 * 链权重
                    importance = chain['similarities'][src_idx] * chain['similarities'][dst_idx] * chain['weight']
                    
                    edge_key = (int(src), int(dst))
                    if edge_key in edge_importance:
                        edge_importance[edge_key] += importance
                    else:
                        edge_importance[edge_key] = importance
    
    # 找出最重要的路径（从问题到答案）
    important_paths = []
    if question_idx != -1 and answer_indices:
        # 创建一个图
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(len(nodes)):
            G.add_node(i)
        
        # 添加边（带权重）
        for (src, dst), importance in edge_importance.items():
            G.add_edge(src, dst, weight=1.0 - importance)  # 权重越小，边越重要
        
        # 对每个答案节点，找出最短路径
        for answer_idx in answer_indices:
            try:
                # 尝试找出从问题到答案的最短路径
                path = nx.shortest_path(G, source=question_idx, target=answer_idx, weight='weight')
                path_importance = 0
                path_edges = []
                
                # 计算路径重要性
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    edge_key = (src, dst)
                    if edge_key in edge_importance:
                        path_importance += edge_importance[edge_key]
                        path_edges.append((src, dst, edge_importance[edge_key]))
                
                important_paths.append({
                    'path': path,
                    'importance': path_importance,
                    'edges': path_edges
                })
            except nx.NetworkXNoPath:
                # 无路径可达
                pass
    
    # 根据重要性排序路径
    important_paths.sort(key=lambda x: x['importance'], reverse=True)
    
    # 预测结果
    prediction_score = prediction.item()
    predicted_answer = prediction_score > threshold
    
    # 可视化
    if visualize:
        visualize_graph_with_paths(nodes, edges, edge_importance, important_paths, question_idx, answer_indices)
    
    # 生成说明文本
    explanation = generate_explanation(nodes, question, chains_data, important_paths)
    
    return {
        'prediction': predicted_answer,
        'score': prediction_score,
        'chains': chains_data,
        'important_paths': important_paths,
        'edge_importance': {f"{k[0]}-{k[1]}": v for k, v in edge_importance.items()},
        'explanation': explanation
    }

# 批量推理
def batch_inference(model, graph_dir, output_dir, device, visualize=False, batch_size=32):
    """
    批量执行推理并保存结果
    
    参数:
    - model: 训练好的模型
    - graph_dir: 图目录
    - output_dir: 输出目录
    - device: 设备
    - visualize: 是否可视化
    - batch_size: 批大小
    
    返回:
    - 成功率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图文件列表
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.json')]
    
    # 处理结果统计
    correct = 0
    total = 0
    
    # 处理每个图
    for i in tqdm(range(0, len(graph_files), batch_size), desc="Processing graphs"):
        batch_files = graph_files[i:i + batch_size]
        
        for graph_file in batch_files:
            # 加载图数据
            graph_path = os.path.join(graph_dir, graph_file)
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
            except Exception as e:
                print(f"Error loading {graph_file}: {e}")
                continue
            
            # 执行推理
            try:
                result = inference_single_graph(model, graph_data, device, visualize=visualize)
                
                # 检查是否有标签
                if 'label' in graph_data:
                    true_label = graph_data['label']
                    if result['prediction'] == true_label:
                        correct += 1
                    total += 1
                
                # 保存结果
                output_file = os.path.join(output_dir, f"result_{graph_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 如果需要可视化，保存图片
                if visualize:
                    fig_path = os.path.join(output_dir, f"viz_{os.path.splitext(graph_file)[0]}.png")
                    plt.savefig(fig_path)
                    plt.close()
            
            except Exception as e:
                print(f"Error processing {graph_file}: {e}")
    
    # 计算成功率
    success_rate = correct / total if total > 0 else 0
    print(f"Success rate: {success_rate:.4f} ({correct}/{total})")
    
    # 保存汇总结果
    summary = {
        'success_rate': success_rate,
        'correct': correct,
        'total': total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return success_rate

# 可视化图和路径
def visualize_graph_with_paths(nodes, edges, edge_importance, important_paths, question_idx, answer_indices):
    """
    可视化图和重要路径
    
    参数:
    - nodes: 节点列表
    - edges: 边列表
    - edge_importance: 边重要性字典
    - important_paths: 重要路径列表
    - question_idx: 问题节点索引
    - answer_indices: 答案节点索引列表
    """
    plt.figure(figsize=(12, 8))
    
    G = nx.DiGraph()
    
    # 添加节点
    for i, node in enumerate(nodes):
        label = node.get('value', f"Node {i}")
        if len(label) > 20:
            label = label[:17] + "..."
        G.add_node(i, label=label)
    
    # 添加边
    max_importance = max(edge_importance.values()) if edge_importance else 1.0
    for edge in edges:
        src = int(edge.get('src', '').replace('n', ''))
        dst = int(edge.get('dst', '').replace('n', ''))
        
        # 检查边的有效性
        if src >= len(nodes) or dst >= len(nodes):
            continue
        
        edge_key = (src, dst)
        importance = edge_importance.get(edge_key, 0.0)
        width = 1 + 4 * (importance / max_importance) if max_importance > 0 else 1
        
        G.add_edge(src, dst, width=width, importance=importance)
    
    # 位置
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制普通边
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if (u, v) not in edge_importance]
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=0.5, alpha=0.3, arrows=True)
    
    # 绘制重要边
    important_edges = [(u, v) for u, v in edge_importance.keys()]
    important_widths = [1 + 4 * (G[u][v]['importance'] / max_importance) for u, v in important_edges]
    important_colors = [plt.cm.Blues(G[u][v]['importance'] / max_importance) for u, v in important_edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=important_edges, width=important_widths, 
                          edge_color=important_colors, arrows=True)
    
    # 绘制最重要路径
    if important_paths:
        top_path = important_paths[0]
        path_edges = [(top_path['path'][i], top_path['path'][i+1]) for i in range(len(top_path['path'])-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red', arrows=True)
    
    # 节点颜色
    node_colors = []
    for i in range(len(nodes)):
        if i == question_idx:
            node_colors.append('lightgreen')
        elif i in answer_indices:
            node_colors.append('lightblue')
        else:
            node_colors.append('white')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, edgecolors='black')
    
    # 绘制标签
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # 添加图例
    plt.plot([0], [0], '-', color='red', linewidth=3, label='最重要路径')
    plt.plot([0], [0], '-', color=plt.cm.Blues(0.8), linewidth=2, label='高重要性边')
    plt.scatter([0], [0], s=100, color='lightgreen', edgecolors='black', label='问题节点')
    plt.scatter([0], [0], s=100, color='lightblue', edgecolors='black', label='答案节点')
    
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.title('知识图谱推理多链可视化', fontsize=16)
    
    return plt.gcf()

# 生成解释文本
def generate_explanation(nodes, question, chains_data, important_paths):
    """
    生成解释文本
    
    参数:
    - nodes: 节点列表
    - question: 问题文本
    - chains_data: 推理链数据
    - important_paths: 重要路径
    
    返回:
    - explanation: 解释文本
    """
    explanation = []
    
    # 添加问题
    explanation.append(f"问题: {question}")
    explanation.append("")
    
    # 添加思维链分析
    explanation.append("【多链思维分析】")
    for i, chain in enumerate(chains_data):
        chain_weight = chain['weight']
        explanation.append(f"思维链 {i+1} (权重: {chain_weight:.4f}):")
        
        chain_nodes = []
        for idx, node_idx in enumerate(chain['top_nodes']):
            if node_idx < len(nodes):
                node_value = nodes[node_idx].get('value', f"节点 {node_idx}")
                similarity = chain['similarities'][idx]
                chain_nodes.append(f"「{node_value}」(相似度: {similarity:.4f})")
        
        explanation.append("  重要节点: " + " → ".join(chain_nodes))
        explanation.append("")
    
    # 添加路径分析
    if important_paths:
        explanation.append("【关键推理路径】")
        
        for i, path_data in enumerate(important_paths[:3]):  # 最多展示前3条路径
            path = path_data['path']
            importance = path_data['importance']
            
            path_text = []
            for node_idx in path:
                if node_idx < len(nodes):
                    node_value = nodes[node_idx].get('value', f"节点 {node_idx}")
                    path_text.append(f"「{node_value}」")
            
            explanation.append(f"路径 {i+1} (重要性: {importance:.4f}):")
            explanation.append("  " + " → ".join(path_text))
            explanation.append("")
    
    # 生成综合推理结论
    explanation.append("【推理结论】")
    if important_paths:
        top_path = important_paths[0]
        path = top_path['path']
        
        # 提取路径中的关系，构建一个连贯的句子
        conclusion = "根据多链思维分析，"
        
        path_text = []
        for node_idx in path:
            if node_idx < len(nodes):
                node_value = nodes[node_idx].get('value', f"节点 {node_idx}")
                path_text.append(f"{node_value}")
        
        conclusion += "，".join(path_text)
        conclusion += "。"
        
        explanation.append(conclusion)
    else:
        explanation.append("无法找到有效的推理路径。")
    
    return "\n".join(explanation)

def main():
    parser = argparse.ArgumentParser(description='多链思维推理')
    parser.add_argument('--graph_dir', type=str, required=True, help='子图目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs/route4_results', help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_chains', type=int, default=3, help='推理链数量')
    parser.add_argument('--chain_length', type=int, default=3, help='推理链长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    parser.add_argument('--single_file', type=str, default='', help='处理单个文件')
    parser.add_argument('--device', type=str, default='', help='设备（cuda或cpu）')
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = MultiChainReasoner(
        node_dim=768,  # 假设使用768维向量
        hidden_dim=args.hidden_dim,
        num_reasoning_chains=args.num_chains,
        chain_length=args.chain_length
    ).to(device)
    
    # 加载模型权重
    try:
        if os.path.isfile(args.model_path):
            # 尝试直接加载
            state_dict = torch.load(args.model_path, map_location=device)
            # 检查是否是checkpoint文件
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            print(f"成功加载模型: {args.model_path}")
        else:
            print(f"未找到模型文件: {args.model_path}")
            return
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 单个文件推理或批量推理
    if args.single_file:
        # 处理单个文件
        graph_path = args.single_file if os.path.isabs(args.single_file) else os.path.join(args.graph_dir, args.single_file)
        
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 执行推理
            result = inference_single_graph(model, graph_data, device, visualize=args.visualize)
            
            # 保存结果
            output_file = os.path.join(args.output_dir, f"result_{os.path.basename(args.single_file)}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 打印结果
            print(f"推理结果: {result['prediction']}")
            print(f"分数: {result['score']:.4f}")
            print(f"解释: {result['explanation']}")
            
            # 如果需要可视化，保存图片
            if args.visualize:
                fig_path = os.path.join(args.output_dir, f"viz_{os.path.splitext(os.path.basename(args.single_file))[0]}.png")
                plt.savefig(fig_path)
                plt.show()
        
        except Exception as e:
            print(f"处理文件失败: {e}")
    else:
        # 批量推理
        success_rate = batch_inference(
            model=model,
            graph_dir=args.graph_dir,
            output_dir=args.output_dir,
            device=device,
            visualize=args.visualize,
            batch_size=args.batch_size
        )
        
        print(f"批量推理完成。成功率: {success_rate:.4f}")
        print(f"结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main() 