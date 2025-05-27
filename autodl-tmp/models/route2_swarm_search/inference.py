#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线2：群体搜索 - 推理脚本
使用多个智能体群体搜索图中的关键路径
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
from collections import Counter, defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 智能体类
class SearchAgent:
    def __init__(self, agent_id, start_node, graph, exploration_rate=0.2, learning_rate=0.1):
        """
        初始化搜索智能体
        
        参数:
        - agent_id: 智能体ID
        - start_node: 起始节点
        - graph: 图结构
        - exploration_rate: 探索率
        - learning_rate: 学习率
        """
        self.id = agent_id
        self.current_node = start_node
        self.graph = graph
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        
        # 记录已访问节点和路径
        self.visited = {start_node}
        self.path = [start_node]
        
        # 智能体的局部知识库
        self.node_values = {}  # 节点价值
        self.edge_weights = {}  # 边权重
        
        # 智能体状态
        self.found_target = False
        self.target_node = None
        self.accumulated_reward = 0.0
        self.steps_taken = 0
        self.max_steps = len(graph.nodes) * 2  # 最大步数
    
    def choose_next_node(self):
        """选择下一个要访问的节点"""
        # 获取相邻节点
        neighbors = list(self.graph.neighbors(self.current_node))
        
        # 过滤掉已访问的节点
        unvisited = [n for n in neighbors if n not in self.visited]
        
        # 如果没有未访问的节点，考虑回溯到之前的节点
        if not unvisited:
            if len(self.path) > 1:
                return self.path[-2]  # 回溯到上一个节点
            return random.choice(neighbors) if neighbors else self.current_node  # 没有路可走，随机或留在原地
        
        # 探索vs利用
        if random.random() < self.exploration_rate:
            # 探索：随机选择
            return random.choice(unvisited)
        else:
            # 利用：选择价值最高的节点
            node_scores = []
            for node in unvisited:
                edge_key = (self.current_node, node)
                # 使用节点价值和边权重的组合
                score = self.node_values.get(node, 0.0) + self.edge_weights.get(edge_key, 0.0)
                node_scores.append((node, score))
            
            # 选择得分最高的节点
            if node_scores:
                return max(node_scores, key=lambda x: x[1])[0]
            
            # 如果没有得分，随机选择
            return random.choice(unvisited)
    
    def move(self):
        """移动到下一个节点"""
        if self.found_target or self.steps_taken >= self.max_steps:
            return False  # 已找到目标或达到最大步数，停止
        
        next_node = self.choose_next_node()
        
        # 更新路径和访问记录
        self.path.append(next_node)
        self.visited.add(next_node)
        self.current_node = next_node
        self.steps_taken += 1
        
        return True  # 继续搜索
    
    def get_path(self):
        """获取智能体走过的路径"""
        return self.path
    
    def update_knowledge(self, reward, target_found=False):
        """更新智能体的知识"""
        self.accumulated_reward += reward
        
        if target_found:
            self.found_target = True
            self.target_node = self.current_node
        
        # 更新节点价值
        self.node_values[self.current_node] = self.node_values.get(self.current_node, 0.0) + \
            self.learning_rate * reward
        
        # 更新最后一条边的权重
        if len(self.path) > 1:
            edge_key = (self.path[-2], self.current_node)
            self.edge_weights[edge_key] = self.edge_weights.get(edge_key, 0.0) + \
                self.learning_rate * reward
    
    def share_knowledge(self, other_agent):
        """与其他智能体共享知识"""
        # 合并节点价值
        for node, value in other_agent.node_values.items():
            if node in self.node_values:
                self.node_values[node] = (self.node_values[node] + value) / 2
            else:
                self.node_values[node] = value * 0.8  # 略微折扣从其他智能体获得的知识
        
        # 合并边权重
        for edge, weight in other_agent.edge_weights.items():
            if edge in self.edge_weights:
                self.edge_weights[edge] = (self.edge_weights[edge] + weight) / 2
            else:
                self.edge_weights[edge] = weight * 0.8

# 单图推理函数
def inference_single_graph(model, graph_data, device, num_agents=5, iterations=50, visualize=False):
    """
    对单个图执行推理
    
    参数:
    - model: 传入的模型（此路线不使用，保持接口一致）
    - graph_data: 图数据
    - device: 设备
    - num_agents: 智能体数量
    - iterations: 迭代次数
    - visualize: 是否可视化
    
    返回:
    - result: 推理结果
    """
    # 从图数据中提取信息
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    question_text = graph_data.get('question', '')
    
    # 构建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    node_roles = {}  # 记录节点角色
    question_node = None
    answer_nodes = []
    
    for i, node in enumerate(nodes):
        node_id = i
        node_text = node.get('value', f'Node_{i}')
        node_role = node.get('role', '')
        
        G.add_node(node_id, text=node_text, role=node_role)
        node_roles[node_id] = node_role
        
        if node_role == 'question':
            question_node = node_id
        elif node_role == 'answer':
            answer_nodes.append(node_id)
    
    # 如果没有找到问题节点，使用第一个节点
    if question_node is None and len(nodes) > 0:
        question_node = 0
    
    # 添加边
    for edge in edges:
        src = edge.get('src', '').replace('n', '')
        dst = edge.get('dst', '').replace('n', '')
        
        if not src.isdigit() or not dst.isdigit():
            continue
        
        src_id, dst_id = int(src), int(dst)
        if src_id >= len(nodes) or dst_id >= len(nodes):
            continue
        
        # 提取边的类型和文本
        edge_type = edge.get('type', '')
        edge_text = edge.get('value', '')
        
        G.add_edge(src_id, dst_id, type=edge_type, text=edge_text)
    
    # 群体搜索
    successful_paths = []
    all_paths = []
    node_visit_counts = Counter()
    edge_visit_counts = Counter()
    
    # 执行多轮搜索
    for iteration in range(iterations):
        # 创建一组智能体
        agents = []
        for i in range(num_agents):
            agent = SearchAgent(
                agent_id=i,
                start_node=question_node,
                graph=G,
                exploration_rate=0.3 - (0.2 * iteration / iterations),  # 逐渐减少探索
                learning_rate=0.1
            )
            agents.append(agent)
        
        # 每个智能体搜索，直到找到答案或达到最大步数
        active_agents = list(range(num_agents))
        for step in range(len(G.nodes) * 2):  # 最大步数
            # 移动每个活跃的智能体
            for agent_idx in active_agents[:]:
                agent = agents[agent_idx]
                continued = agent.move()
                
                # 更新节点访问计数
                node_visit_counts[agent.current_node] += 1
                
                # 更新边访问计数
                if len(agent.path) > 1:
                    edge = (agent.path[-2], agent.path[-1])
                    edge_visit_counts[edge] += 1
                
                # 检查是否找到答案节点
                if agent.current_node in answer_nodes:
                    # 找到答案，给予奖励
                    agent.update_knowledge(reward=1.0, target_found=True)
                    successful_paths.append(agent.get_path())
                    active_agents.remove(agent_idx)
                elif not continued:
                    # 智能体停止搜索
                    active_agents.remove(agent_idx)
            
            # 如果没有活跃的智能体，跳出循环
            if not active_agents:
                break
            
            # 共享知识（每5步共享一次）
            if step > 0 and step % 5 == 0:
                for i in range(len(agents)):
                    for j in range(i+1, len(agents)):
                        if i in active_agents and j in active_agents:
                            agents[i].share_knowledge(agents[j])
                            agents[j].share_knowledge(agents[i])
        
        # 收集所有智能体的路径
        for agent in agents:
            all_paths.append(agent.get_path())
    
    # 分析结果
    # 1. 最常访问的节点
    most_common_nodes = node_visit_counts.most_common(5)
    
    # 2. 最常访问的边
    most_common_edges = edge_visit_counts.most_common(5)
    
    # 3. 找到的成功路径
    if successful_paths:
        # 按路径长度排序，优先考虑较短的路径
        successful_paths.sort(key=len)
        best_path = successful_paths[0]
    else:
        # 如果没有找到成功路径，使用所有路径中访问答案节点最多的路径
        answer_visit_counts = defaultdict(int)
        for path in all_paths:
            for node in path:
                if node in answer_nodes:
                    answer_visit_counts[tuple(path)] += 1
        
        if answer_visit_counts:
            best_path = list(max(answer_visit_counts.items(), key=lambda x: x[1])[0])
        else:
            # 如果没有路径访问过答案节点，选择最常访问节点组成的路径
            common_nodes = [node for node, _ in most_common_nodes]
            best_path = [question_node] + common_nodes
    
    # 计算路径重要性分数
    path_scores = {}
    if successful_paths:
        # 找出每个节点在成功路径中出现的频率
        node_success_freq = Counter()
        for path in successful_paths:
            for node in path:
                node_success_freq[node] += 1
        
        # 为每条成功路径计算得分
        for i, path in enumerate(successful_paths):
            path_score = sum(node_success_freq[node] for node in path) / len(path)
            # 较短的路径得分略高
            length_bonus = 1.0 + (1.0 / len(path))
            path_scores[i] = path_score * length_bonus
    
    # 制作推理结果
    node_importance = {node: count / sum(node_visit_counts.values()) 
                      for node, count in node_visit_counts.items()}
    
    edge_importance = {f"{src}-{dst}": count / sum(edge_visit_counts.values()) 
                      for (src, dst), count in edge_visit_counts.items()}
    
    # 计算推理结果的置信度
    confidence = 0.0
    if successful_paths:
        # 计算成功路径的比例
        success_ratio = len(successful_paths) / (num_agents * iterations)
        # 平均路径得分
        avg_path_score = sum(path_scores.values()) / len(path_scores) if path_scores else 0.0
        confidence = 0.7 * success_ratio + 0.3 * avg_path_score
    else:
        # 如果没有成功路径，置信度较低
        confidence = 0.3
    
    # 可视化
    if visualize:
        visualize_search_results(G, node_roles, best_path, node_importance, edge_importance)
    
    # 生成解释文本
    explanation = generate_explanation(G, question_text, best_path, successful_paths, 
                                      most_common_nodes, most_common_edges)
    
    return {
        'prediction': bool(successful_paths),  # 如果找到成功路径，则预测为真
        'score': confidence,
        'best_path': best_path,
        'successful_paths': successful_paths,
        'node_importance': node_importance,
        'edge_importance': edge_importance,
        'explanation': explanation
    }

def visualize_search_results(G, node_roles, best_path, node_importance, edge_importance):
    """可视化搜索结果"""
    plt.figure(figsize=(12, 8))
    
    # 创建布局
    pos = nx.spring_layout(G, seed=42)
    
    # 节点颜色映射
    node_colors = []
    for node in G.nodes():
        if node_roles.get(node) == 'question':
            node_colors.append('lightgreen')  # 问题节点为绿色
        elif node_roles.get(node) == 'answer':
            node_colors.append('lightblue')   # 答案节点为蓝色
        else:
            # 其他节点根据重要性着色
            importance = node_importance.get(node, 0.0)
            node_colors.append(plt.cm.Oranges(importance))
    
    # 节点大小映射（根据重要性）
    node_sizes = [300 + 700 * node_importance.get(node, 0.0) for node in G.nodes()]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, edgecolors='black')
    
    # 绘制边（默认灰色）
    nx.draw_networkx_edges(G, pos, edge_color='grey', width=1, alpha=0.3)
    
    # 绘制最佳路径边（红色）
    if best_path:
        best_path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, edge_color='red', width=2.5)
    
    # 绘制重要边（深蓝色）
    edge_list = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        edge_key = f"{u}-{v}"
        importance = float(edge_importance.get(edge_key, 0.0))
        if importance > 0.05:  # 只突出显示重要性较高的边
            edge_list.append((u, v))
            edge_widths.append(1 + 5 * importance)
    
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='blue', width=edge_widths, alpha=0.6)
    
    # 节点标签
    node_labels = {node: G.nodes[node].get('text', str(node))[:15] + '...' 
                  if len(G.nodes[node].get('text', str(node))) > 15 
                  else G.nodes[node].get('text', str(node)) 
                  for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # 添加图例
    plt.legend([
        plt.Line2D([0], [0], color='lightgreen', marker='o', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='lightblue', marker='o', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='red', lw=2),
        plt.Line2D([0], [0], color='blue', lw=2)
    ], ['问题节点', '答案节点', '最佳路径', '重要边'], loc='upper right')
    
    plt.title('群体智能搜索结果', fontsize=16)
    plt.axis('off')
    
    return plt.gcf()

def generate_explanation(G, question, best_path, successful_paths, common_nodes, common_edges):
    """生成解释文本"""
    explanation = []
    
    # 添加问题
    explanation.append(f"问题: {question}")
    explanation.append("")
    
    # 搜索统计
    explanation.append("【群体搜索统计】")
    explanation.append(f"- 成功路径数量: {len(successful_paths)}")
    explanation.append(f"- 最佳路径长度: {len(best_path)}")
    explanation.append("")
    
    # 最常访问的节点
    explanation.append("【关键知识点】")
    for node_id, count in common_nodes:
        node_text = G.nodes[node_id].get('text', f"节点{node_id}")
        explanation.append(f"- {node_text} (访问次数: {count})")
    explanation.append("")
    
    # 最佳路径分析
    explanation.append("【推理路径】")
    if best_path:
        path_nodes = []
        for node in best_path:
            node_text = G.nodes[node].get('text', f"节点{node}")
            path_nodes.append(node_text)
        
        explanation.append(" → ".join(path_nodes))
    else:
        explanation.append("未找到有效路径")
    explanation.append("")
    
    # 推理结论
    explanation.append("【推理结论】")
    if successful_paths:
        # 构建一个连贯的结论句子
        conclusion = "根据群体搜索，"
        if len(best_path) > 1:
            start_node_text = G.nodes[best_path[0]].get('text', f"节点{best_path[0]}")
            end_node_text = G.nodes[best_path[-1]].get('text', f"节点{best_path[-1]}")
            
            # 简化的中间节点文本
            if len(best_path) > 3:
                middle_texts = []
                for i in range(1, len(best_path)-1):
                    middle_text = G.nodes[best_path[i]].get('text', f"节点{best_path[i]}")
                    middle_texts.append(middle_text)
                middle_part = "，".join(middle_texts)
                conclusion += f"从{start_node_text}出发，经过{middle_part}，最终到达{end_node_text}。"
            else:
                conclusion += f"从{start_node_text}可以直接推导出{end_node_text}。"
        
        explanation.append(conclusion)
        explanation.append(f"搜索成功率: {len(successful_paths)/5:.1%}")
    else:
        explanation.append("搜索未能找到明确的推理路径，无法得出确定结论。")
    
    return "\n".join(explanation)

def batch_inference(graph_dir, output_dir, num_agents=5, iterations=50, visualize=False, batch_size=16):
    """批量推理函数"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图文件列表
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.json')]
    
    # 用于统计
    correct = 0
    total = 0
    
    # 处理每个图
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
            start_time = time.time()
            try:
                result = inference_single_graph(
                    model=None,  # 此路线不使用模型
                    graph_data=graph_data,
                    device=None,  # 此路线不使用设备
                    num_agents=num_agents,
                    iterations=iterations,
                    visualize=visualize
                )
                
                # 如果有标签，记录正确性
                if 'label' in graph_data:
                    true_label = graph_data['label']
                    prediction = result['prediction']
                    if prediction == true_label:
                        correct += 1
                    total += 1
                
                # 添加时间信息
                result['inference_time'] = time.time() - start_time
                
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
                print(f"处理 {graph_file} 失败: {e}")
    
    # 计算整体准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 保存汇总结果
    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_agents': num_agents,
        'iterations': iterations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"批量推理完成。准确率: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="路线2: 群体搜索推理")
    parser.add_argument('--graph_dir', type=str, required=True, help='图数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs/route2_results', help='输出目录')
    parser.add_argument('--num_agents', type=int, default=5, help='智能体数量')
    parser.add_argument('--iterations', type=int, default=50, help='迭代次数')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--single_file', type=str, default='', help='单个文件进行推理')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 单个文件推理或批量推理
    if args.single_file:
        graph_path = os.path.join(args.graph_dir, args.single_file) if not os.path.isabs(args.single_file) else args.single_file
        
        try:
            # 加载图数据
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 执行推理
            result = inference_single_graph(
                model=None,
                graph_data=graph_data,
                device=None,
                num_agents=args.num_agents,
                iterations=args.iterations,
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
            graph_dir=args.graph_dir,
            output_dir=args.output_dir,
            num_agents=args.num_agents,
            iterations=args.iterations,
            visualize=args.visualize,
            batch_size=args.batch_size
        )
        
        print(f"批量推理完成。准确率: {accuracy:.4f}")
        print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 