#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线3：差分解释器 - 推理脚本
对子图进行推理，识别并可视化关键路径与重要关系
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.route3_diff_explainer.train import DiffExplainer, GraphDataset

def inference_single_graph(model, graph_data, device, importance_threshold=0.3):
    """
    对单个图执行推理，计算边的重要性
    
    参数:
    - model: 训练好的DiffExplainer模型
    - graph_data: 图数据
    - device: 设备
    - importance_threshold: 边重要性的阈值，高于此值的边被视为重要
    
    返回:
    - 分析结果，包括边的重要性和关键路径
    """
    # 准备数据
    dataset = GraphDataset("", [])
    g, node_feats, question_idx, answer_idx, metadata = dataset.build_graph(graph_data)
    
    # 移动到设备
    g = g.to(device)
    node_feats = node_feats.to(device)
    
    # 执行推理
    model.eval()
    with torch.no_grad():
        score, node_embeds, edge_importance = model(g, node_feats)
    
    # 边的信息
    edges = g.edges()
    src, dst = edges
    edge_types = metadata['edge_types']
    node_labels = metadata['node_labels']
    
    # 确保边重要性的长度与边数量相同
    if edge_importance is not None and len(edge_importance) != len(src):
        print(f"警告：边重要性长度 ({len(edge_importance)}) 与边数量 ({len(src)}) 不匹配")
        # 可能是因为异构图，调整边重要性的尺寸
        if len(edge_importance) > len(src):
            edge_importance = edge_importance[:len(src)]
        else:
            # 扩展边重要性
            padding = torch.zeros(len(src) - len(edge_importance), device=edge_importance.device)
            edge_importance = torch.cat([edge_importance, padding])
    
    # 整理边的重要性信息
    edge_info = []
    for i in range(len(src)):
        src_idx = src[i].item()
        dst_idx = dst[i].item()
        edge_type = edge_types[i] if i < len(edge_types) else 'default'
        importance = edge_importance[i].item() if i < len(edge_importance) else 0.0
        
        edge_info.append({
            'src_idx': src_idx,
            'dst_idx': dst_idx,
            'src_label': node_labels[src_idx],
            'dst_label': node_labels[dst_idx],
            'edge_type': edge_type,
            'importance': importance,
            'is_important': importance > importance_threshold
        })
    
    # 按重要性排序
    edge_info.sort(key=lambda x: x['importance'], reverse=True)
    
    # 查找问题到答案的最短路径
    G = nx.DiGraph()
    
    # 添加所有重要边到图中
    important_edges = [(e['src_idx'], e['dst_idx']) for e in edge_info if e['is_important']]
    G.add_edges_from(important_edges)
    
    # 尝试找出问题到答案的路径
    paths = []
    try:
        if question_idx in G.nodes() and answer_idx in G.nodes():
            # 尝试找到所有简单路径
            all_paths = list(nx.all_simple_paths(G, question_idx, answer_idx, cutoff=5))
            # 限制最多返回3条路径
            paths = all_paths[:3]
    except nx.NetworkXNoPath:
        # 如果没有路径，则为空列表
        paths = []
    
    # 找到路径上的边
    path_edges = []
    for path in paths:
        for i in range(len(path) - 1):
            path_edges.append((path[i], path[i+1]))
    
    # 整理结果
    result = {
        'graph_score': score.item(),
        'question_idx': question_idx,
        'answer_idx': answer_idx,
        'edge_info': edge_info,
        'paths': paths,
        'node_labels': node_labels,
        'important_edges': important_edges,
        'path_edges': path_edges
    }
    
    return result

def visualize_explanation(graph_data, result, output_path):
    """可视化解释结果"""
    plt.figure(figsize=(12, 10))
    
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    nodes = graph_data.get('nodes', [])
    for i, node in enumerate(nodes):
        role = node.get('role', 'context')
        label = node.get('label', f'Node {i}')
        G.add_node(i, role=role, label=label[:15])  # 限制标签长度
    
    # 添加所有边（包括不重要的边）
    edges = graph_data.get('edges', [])
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        rel = edge.get('rel', 'default')
        
        # 只有当源节点和目标节点都存在时才添加边
        if src_id < len(nodes) and dst_id < len(nodes):
            G.add_edge(src_id, dst_id, rel=rel)
    
    # 获取问题和答案节点
    question_idx = result['question_idx']
    answer_idx = result['answer_idx']
    
    # 设置节点颜色
    node_colors = []
    for i in range(len(nodes)):
        role = G.nodes[i]['role'] if 'role' in G.nodes[i] else 'context'
        if i == question_idx:
            node_colors.append('blue')
        elif i == answer_idx:
            node_colors.append('green')
        elif role == 'evidence':
            node_colors.append('orange')
        else:
            node_colors.append('lightgray')
    
    # 布局
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制所有边（灰色，半透明）
    nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=True, edge_color='gray')
    
    # 绘制重要边（根据重要性使用不同颜色）
    edge_info = result['edge_info']
    
    cmap = plt.cm.plasma
    for edge in edge_info:
        if edge['is_important']:
            src_idx = edge['src_idx']
            dst_idx = edge['dst_idx']
            importance = edge['importance']
            
            # 边的颜色根据重要性
            edge_color = cmap(importance)
            # 边的宽度根据重要性
            width = 1 + 3 * importance
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, edgelist=[(src_idx, dst_idx)], 
                                  width=width, alpha=0.8, arrows=True, 
                                  edge_color=[edge_color])
    
    # 绘制找到的路径
    paths = result['paths']
    path_colors = ['red', 'purple', 'cyan']
    
    for i, path in enumerate(paths):
        if i < len(path_colors):  # 限制最多显示3条路径
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                  width=2, alpha=0.9, arrows=True, 
                                  edge_color=path_colors[i], 
                                  label=f"Path {i+1} (length: {len(path)})")
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=500)
    
    # 添加节点标签
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # 添加图例
    plt.scatter([], [], c='blue', s=100, label='问题')
    plt.scatter([], [], c='green', s=100, label='答案')
    plt.scatter([], [], c='orange', s=100, label='证据')
    plt.scatter([], [], c='lightgray', s=100, label='上下文')
    
    # 为路径添加图例
    for i, path in enumerate(paths):
        if i < len(path_colors):
            plt.plot([], [], c=path_colors[i], label=f"路径 {i+1} (长度: {len(path)})")
    
    plt.legend(loc='upper right')
    plt.title('差分解释器分析结果')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_explanation_text(result, graph_data):
    """生成解释文本"""
    explanation = []
    
    # 获取问题和答案节点的标签
    question_idx = result['question_idx']
    answer_idx = result['answer_idx']
    
    nodes = graph_data.get('nodes', [])
    question_label = nodes[question_idx].get('label', f'节点 {question_idx}') if question_idx < len(nodes) else '未知问题'
    answer_label = nodes[answer_idx].get('label', f'节点 {answer_idx}') if answer_idx < len(nodes) else '未知答案'
    
    # 添加基本信息
    explanation.append(f"问题: {question_label}")
    explanation.append(f"答案: {answer_label}")
    explanation.append(f"图的整体匹配分数: {result['graph_score']:.4f}\n")
    
    # 添加发现的路径信息
    paths = result['paths']
    if paths:
        explanation.append("发现的重要路径:")
        for i, path in enumerate(paths):
            path_str = " -> ".join([result['node_labels'][idx] for idx in path])
            explanation.append(f"路径 {i+1} (长度: {len(path)}): {path_str}")
        explanation.append("")
    else:
        explanation.append("未找到从问题到答案的重要路径\n")
    
    # 添加重要边信息
    important_edges = [edge for edge in result['edge_info'] if edge['is_important']]
    explanation.append(f"发现 {len(important_edges)} 条重要关系:")
    
    for i, edge in enumerate(important_edges[:10]):  # 限制最多显示10条重要边
        explanation.append(f"{i+1}. {edge['src_label']} -> {edge['dst_label']} ({edge['edge_type']}), 重要性: {edge['importance']:.4f}")
    
    if len(important_edges) > 10:
        explanation.append(f"... 以及 {len(important_edges) - 10} 条其他重要关系")
    
    return "\n".join(explanation)

def batch_inference(model, graph_dir, output_dir, importance_threshold=0.3, device='cuda', batch_size=8, visualize=False):
    """批量执行推理"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要可视化，创建可视化目录
    vis_dir = os.path.join(output_dir, "visualizations")
    explain_dir = os.path.join(output_dir, "explanations")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(explain_dir, exist_ok=True)
    
    # 获取所有图文件
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.json')]
    print(f"找到 {len(graph_files)} 个图文件")
    
    # 批处理
    results = {}
    total_graphs = 0
    has_path_count = 0
    
    start_time = time.time()
    
    for i in range(0, len(graph_files), batch_size):
        batch_files = graph_files[i:i+batch_size]
        
        for graph_file in tqdm(batch_files, desc=f"处理批次 {i//batch_size + 1}/{(len(graph_files)-1)//batch_size + 1}"):
            graph_path = os.path.join(graph_dir, graph_file)
            
            # 加载图数据
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 执行推理
            try:
                # 执行推理
                result = inference_single_graph(
                    model, 
                    graph_data, 
                    device, 
                    importance_threshold
                )
                
                # 统计数据
                total_graphs += 1
                if result['paths']:
                    has_path_count += 1
                
                # 保存结果
                results[graph_file] = result
                
                # 单独保存该图的结果
                output_path = os.path.join(output_dir, f"result_{graph_file}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    # 将结果转换为可序列化格式
                    serializable_result = {
                        'graph_score': result['graph_score'],
                        'question_idx': result['question_idx'],
                        'answer_idx': result['answer_idx'],
                        'edge_info': result['edge_info'],
                        'paths': result['paths'],
                        'node_labels': result['node_labels'],
                        'important_edges': result['important_edges'],
                        'path_edges': result['path_edges']
                    }
                    json.dump(serializable_result, f, ensure_ascii=False, indent=2)
                
                # 可视化
                if visualize:
                    # 生成可视化图像
                    vis_path = os.path.join(vis_dir, f"vis_{graph_file.replace('.json', '.png')}")
                    visualize_explanation(graph_data, result, vis_path)
                    
                    # 生成解释文本
                    explanation_text = generate_explanation_text(result, graph_data)
                    explanation_path = os.path.join(explain_dir, f"explain_{graph_file.replace('.json', '.txt')}")
                    with open(explanation_path, 'w', encoding='utf-8') as f:
                        f.write(explanation_text)
            
            except Exception as e:
                print(f"处理 {graph_file} 时出错: {str(e)}")
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 计算总体统计数据
    path_rate = has_path_count / total_graphs if total_graphs > 0 else 0
    
    # 保存总结果
    summary = {
        'total_graphs': total_graphs,
        'graphs_with_path': has_path_count,
        'path_rate': path_rate,
        'total_time': total_time,
        'avg_time_per_graph': total_time / total_graphs if total_graphs > 0 else 0,
        'importance_threshold': importance_threshold
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"推理完成. 共 {total_graphs} 个图，找到路径的比例: {path_rate:.4f}, 总时间: {total_time:.2f}秒")
    
    return results, summary

def main():
    parser = argparse.ArgumentParser(description='差分解释器推理')
    parser.add_argument('--graph_dir', type=str, required=True, help='子图目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='GNN层数')
    parser.add_argument('--importance_threshold', type=float, default=0.3, help='边重要性阈值')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图像和解释')
    args = parser.parse_args()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = DiffExplainer(
        node_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'已加载模型: {args.model_path}')
    
    # 执行批量推理
    batch_inference(
        model, 
        args.graph_dir, 
        args.output_dir, 
        args.importance_threshold,
        device,
        args.batch_size,
        args.visualize
    )

if __name__ == '__main__':
    main() 