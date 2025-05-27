#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路线2：群体搜索 - 训练脚本
使用多Agent群体搜索来训练知识图谱上的路径选择
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
from collections import defaultdict, deque
import dgl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from tensorboardX import SummaryWriter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import load_graph_data

# Q-Network模型，用于智能体决策
class PathQNetwork(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_actions=10):
        super(PathQNetwork, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 路径历史编码器（LSTM）
        self.path_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 组合状态表示
        self.combine_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Q值预测（对每个可能的动作）
        self.q_predictor = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, current_node_feat, path_node_feats, available_actions=None):
        """
        前向传播计算Q值
        
        参数:
        - current_node_feat: 当前节点特征 [batch_size, node_dim]
        - path_node_feats: 路径历史节点特征 [batch_size, path_length, node_dim]
        - available_actions: 可用动作掩码 [batch_size, num_actions]
        
        返回:
        - q_values: 每个动作的Q值 [batch_size, num_actions]
        """
        # 编码当前节点
        current_encoded = self.node_encoder(current_node_feat)
        
        # 编码路径历史
        path_length = path_node_feats.size(1)
        path_reshaped = path_node_feats.view(-1, self.node_dim)
        path_encoded = self.node_encoder(path_reshaped)
        path_encoded = path_encoded.view(-1, path_length, self.hidden_dim)
        
        # LSTM处理路径
        path_output, _ = self.path_encoder(path_encoded)
        path_output = path_output[:, -1, :]  # 取最后一个时间步
        
        # 组合状态表示
        combined = torch.cat([current_encoded, path_output], dim=1)
        state_repr = self.combine_state(combined)
        
        # 预测Q值
        q_values = self.q_predictor(state_repr)
        
        # 如果提供了可用动作掩码，应用掩码
        if available_actions is not None:
            # 将不可用动作的Q值设为很小的值
            q_values = q_values.clone()
            invalid_mask = ~available_actions.bool()
            q_values[invalid_mask] = float('-inf')
        
        return q_values

# 使用离策略训练记忆库
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, avail_actions, next_avail_actions):
        self.buffer.append((state, action, reward, next_state, done, avail_actions, next_avail_actions))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones, avail_actions, next_avail_actions = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones, avail_actions, next_avail_actions
    
    def __len__(self):
        return len(self.buffer)

# 群体智能体环境
class SwarmEnvironment:
    def __init__(self, graph_data, max_path_length=5, num_agents=8):
        self.graph_data = graph_data
        self.max_path_length = max_path_length
        self.num_agents = num_agents
        
        # 解析图数据
        self.nodes = graph_data.get('nodes', [])
        self.edges = graph_data.get('edges', [])
        
        # 构建邻接表，用于快速查询节点的邻居
        self.adjacency = self._build_adjacency()
        
        # 查找问题节点和答案节点
        self.question_idx, self.answer_idx = self._find_special_nodes()
        
        # 初始化智能体状态
        self.agent_states = []
        
        # 信息素相关属性
        self.pheromone = {}  # 边的信息素浓度，以(src, dst)作为键
        self.pheromone_history = []  # 存储历史信息素变化
        self.path_scores = []  # 存储生成路径的LLM评分
        self.entropy_history = []  # 存储信息素熵的历史变化
        self._init_pheromone()  # 初始化信息素，使用先验概率
    
    def _build_adjacency(self):
        """构建邻接表，将边转换为字典格式以便快速查询"""
        adjacency = defaultdict(list)
        
        for edge in self.edges:
            src = edge.get('src', '').replace('n', '')
            dst = edge.get('dst', '').replace('n', '')
            
            if not src.isdigit() or not dst.isdigit():
                continue
            
            src_id, dst_id = int(src), int(dst)
            
            # 添加边信息
            adjacency[src_id].append({
                'node_id': dst_id,
                'rel': edge.get('rel', 'default'),
                'prior': edge.get('prior', 0.5)
            })
            
            # 如果是双向图，也添加反向边
            if not edge.get('directed', False):
                adjacency[dst_id].append({
                    'node_id': src_id,
                    'rel': edge.get('rel', 'default'),
                    'prior': edge.get('prior', 0.5)
                })
        
        return adjacency
    
    def _find_special_nodes(self):
        """找到问题节点和答案节点的索引"""
        question_idx = -1
        answer_idx = -1
        
        for i, node in enumerate(self.nodes):
            role = node.get('role', '')
            if role == 'question':
                question_idx = i
            elif role == 'answer':
                answer_idx = i
        
        # 如果没有找到问题节点，使用第一个节点
        if question_idx == -1 and len(self.nodes) > 0:
            question_idx = 0
        
        # 如果没有找到答案节点，使用最后一个节点
        if answer_idx == -1 and len(self.nodes) > 0:
            answer_idx = len(self.nodes) - 1
        
        return question_idx, answer_idx
    
    def reset(self):
        """重置环境，所有智能体从问题节点出发"""
        self.agent_states = []
        
        for _ in range(self.num_agents):
            # 每个智能体的状态包括：当前节点，路径历史，已访问节点集合
            agent_state = {
                'current_node': self.question_idx,
                'path': [self.question_idx],
                'visited': {self.question_idx},
                'done': False,
                'success': False,
                'steps': 0
            }
            self.agent_states.append(agent_state)
        
        return self._get_observation()
    
    def _get_node_features(self, node_idx):
        """获取节点特征"""
        if 0 <= node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            # 如果节点有特征，直接使用
            if 'feat' in node and node['feat'] != 'PLACEHOLDER':
                try:
                    # 假设你期望768维
                    expected_dim = 768
                    feat = np.array(node['feat'], dtype=np.float32)
                    if feat.shape[0] < expected_dim:
                        feat = np.pad(feat, (0, expected_dim - feat.shape[0]))
                    elif feat.shape[0] > expected_dim:
                        feat = feat[:expected_dim]
                    return feat
                except:
                    pass
        
        # 如果没有特征或出错，返回零向量
        return np.zeros(768, dtype=np.float32)
    
    def _get_observation(self):
        """获取所有智能体的观察"""
        observations = []
        avail_actions_list = []
        
        for agent in self.agent_states:
            if agent['done']:
                # 如果智能体已完成，提供空观察
                obs = {
                    'current_node_feat': np.zeros(768, dtype=np.float32),
                    'path_node_feats': np.zeros((self.max_path_length, 768), dtype=np.float32),
                    'current_node_idx': -1,
                    'path_length': 0
                }
                avail_actions = np.zeros(10, dtype=np.bool_)
            else:
                # 当前节点特征
                current_node_feat = self._get_node_features(agent['current_node'])
                
                # 路径历史节点特征
                path = agent['path'][-self.max_path_length:] if len(agent['path']) > self.max_path_length else agent['path']
                path_length = len(path)
                
                path_node_feats = np.zeros((self.max_path_length, 768), dtype=np.float32)
                for i, node_idx in enumerate(path):
                    path_node_feats[i] = self._get_node_features(node_idx)
                
                # 获取可用动作（邻居节点）
                neighbors = self.adjacency[agent['current_node']]
                avail_actions = np.zeros(10, dtype=np.bool_)
                
                # 最多考虑10个邻居节点作为动作
                for i, neighbor in enumerate(neighbors[:10]):
                    # 如果邻居节点没有被访问过或者是答案节点，则可选
                    if neighbor['node_id'] not in agent['visited'] or neighbor['node_id'] == self.answer_idx:
                        avail_actions[i] = True
                
                obs = {
                    'current_node_feat': current_node_feat,
                    'path_node_feats': path_node_feats,
                    'current_node_idx': agent['current_node'],
                    'path_length': path_length
                }
            
            observations.append(obs)
            avail_actions_list.append(avail_actions)
        
        return observations, avail_actions_list
    
    def _get_neighbors(self, node_idx):
        """获取节点的邻居"""
        return self.adjacency.get(node_idx, [])[:10]  # 最多返回10个邻居
    
    def step(self, actions):
        """执行智能体的动作，返回新的观察、奖励和是否结束"""
        rewards = []
        all_done = True
        
        for i, (agent, action) in enumerate(zip(self.agent_states, actions)):
            # 如果智能体已经完成任务，跳过
            if agent['done']:
                rewards.append(0)
                continue
            
            # 获取可用动作
            neighbors = self._get_neighbors(agent['current_node'])
            
            # 如果动作有效
            if 0 <= action < len(neighbors):
                # 执行动作，移动到选择的邻居节点
                next_node = neighbors[action]['node_id']
                prior = neighbors[action]['prior']
                
                # 更新状态
                agent['current_node'] = next_node
                agent['path'].append(next_node)
                agent['visited'].add(next_node)
                agent['steps'] += 1
                
                # 检查是否找到答案
                if next_node == self.answer_idx:
                    agent['done'] = True
                    agent['success'] = True
                    reward = 10.0  # 成功的奖励
                # 检查是否超过最大步数
                elif agent['steps'] >= self.max_path_length:
                    agent['done'] = True
                    agent['success'] = False
                    # 根据路径长度和邻近程度计算奖励
                    distance_to_answer = 1.0  # 简化距离
                    reward = -1.0 - distance_to_answer
                else:
                    # 中间步骤的奖励，根据边的先验概率加权
                    reward = prior - 0.1  # 稍微鼓励探索
            else:
                # 无效动作
                reward = -0.5
            
            rewards.append(reward)
            
            # 检查是否所有智能体都完成了
            if not agent['done']:
                all_done = False
        
        # 获取新的观察
        observations, avail_actions = self._get_observation()
        
        # 如果所有智能体都完成了或者达到最大步数，环境结束
        done = all_done
        
        # 计算额外信息
        info = {
            'success_rate': sum(1 for agent in self.agent_states if agent['success']) / self.num_agents,
            'avg_path_length': sum(len(agent['path']) for agent in self.agent_states) / self.num_agents
        }
        
        return observations, rewards, done, avail_actions, info
    
    def _init_pheromone(self):
        """初始化信息素，使用边的先验概率作为初值"""
        for node_id, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                dst = neighbor['node_id']
                # 使用先验概率作为初始信息素值
                prior = neighbor.get('prior', 0.5)  # 默认为0.5
                self.pheromone[(node_id, dst)] = prior
        
        # 计算并存储初始信息素熵
        initial_entropy = self.calculate_pheromone_entropy()
        self.entropy_history.append(initial_entropy)
    
    def update_pheromone(self, paths, rewards, evaporation=0.1):
        """
        更新信息素浓度
        
        参数:
        - paths: 每个智能体的路径 [(src, dst), ...]
        - rewards: 每个路径对应的奖励值
        - evaporation: 信息素蒸发率
        """
        # 信息素蒸发
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - evaporation)
        
        # 增加新的信息素
        for path, reward in zip(paths, rewards):
            if reward > 0:  # 只有奖励为正的路径才添加信息素
                for edge in path:
                    if edge in self.pheromone:
                        self.pheromone[edge] += reward
        
        # 保存当前信息素状态的副本
        current_pheromone = self.pheromone.copy()
        self.pheromone_history.append(current_pheromone)
        
        # 计算并存储当前信息素熵
        current_entropy = self.calculate_pheromone_entropy()
        self.entropy_history.append(current_entropy)
    
    def calculate_pheromone_entropy(self):
        """计算当前信息素的熵，用于判断收敛情况"""
        if not self.pheromone:
            return 0
        
        # 获取所有信息素值
        values = list(self.pheromone.values())
        total = sum(values)
        
        if total == 0:
            return 0
        
        # 计算归一化概率分布
        probs = [v / total for v in values]
        
        # 计算熵: H = -sum(p * log(p))
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
        
        return entropy
    
    def get_top_k_paths(self, k=10):
        """获取信息素浓度最高的前K条路径"""
        # 使用信息素构建有向图
        G = {}
        for (src, dst), pheromone in self.pheromone.items():
            if src not in G:
                G[src] = []
            G[src].append((dst, pheromone))
        
        # 对每个邻居按信息素浓度排序
        for node in G:
            G[node].sort(key=lambda x: x[1], reverse=True)
        
        # 使用DFS查找高信息素路径
        paths = []
        
        def dfs(node, path, visited):
            if node == self.answer_idx or len(path) >= self.max_path_length:
                if path and path[-1][0] == self.answer_idx:  # 路径到达答案节点
                    paths.append(path)
                return
            
            if node not in G:
                return
            
            # 选择信息素浓度最高的未访问邻居
            for next_node, _ in G[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    dfs(next_node, path + [((node, next_node), self.pheromone.get((node, next_node), 0))], visited)
                    visited.remove(next_node)
        
        # 从问题节点开始搜索
        dfs(self.question_idx, [], {self.question_idx})
        
        # 按路径总信息素排序并返回前k条
        paths.sort(key=lambda p: sum(edge[1] for edge in p), reverse=True)
        
        return paths[:k]
    
    def is_converged(self, min_entropy_ratio=0.8, no_improvement_rounds=10):
        """
        判断蚁群算法是否已收敛
        
        参数:
        - min_entropy_ratio: 当前熵与初始熵的比率阈值，小于此值视为已收敛
        - no_improvement_rounds: 连续多少轮没有更好路径出现时视为收敛
        
        返回:
        - 是否已收敛
        """
        if len(self.entropy_history) < 2:
            return False
        
        # 条件1: 信息素熵降低到初始值的min_entropy_ratio以下
        initial_entropy = self.entropy_history[0]
        current_entropy = self.entropy_history[-1]
        entropy_condition = current_entropy <= min_entropy_ratio * initial_entropy
        
        # 条件2: 连续no_improvement_rounds轮没有更好的路径
        improvement_condition = False
        if len(self.path_scores) >= no_improvement_rounds:
            recent_scores = self.path_scores[-no_improvement_rounds:]
            max_score = max(recent_scores)
            max_idx = recent_scores.index(max_score)
            # 如果最大分数不在最近几轮，则认为没有改进
            improvement_condition = max_idx == 0  # 最大分数在最早的一轮
        
        return entropy_condition and improvement_condition

    def update_path_score(self, score):
        """更新路径得分，用于LLM评分反馈"""
        self.path_scores.append(score)

# 图数据集
class GraphDataset(Dataset):
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
        
        return graph_data

# 训练函数
def train(q_network, target_network, optimizer, replay_buffer, graph_dataset, 
          batch_size=32, episodes=1000, gamma=0.99, target_update=10, 
          eps_start=0.9, eps_end=0.05, eps_decay=200, device='cuda', 
          output_dir='./output'):
    
    # 训练数据收集
    episode_rewards = []
    episode_success_rates = []
    entropy_history = []
    path_coverage_rates = []
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # 创建结果目录
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 模拟LLM评分函数（实际应用需要调用真实LLM API）
    def simulate_llm_scoring(path, g):
        """模拟LLM对路径的评分 (0-1之间)"""
        # 在实际应用中，这里应该调用LLM API评估路径质量
        # 这里使用简单启发式：如果路径包含答案节点，得分高
        path_nodes = [edge[0] for edge in path] + [path[-1][1]]
        if g.answer_idx in path_nodes:
            distance_to_answer = path_nodes.index(g.answer_idx) / len(path_nodes)
            return 1.0 - 0.5 * distance_to_answer  # 越早到达答案，分数越高
        else:
            # 计算路径终点与答案节点的相似度作为分数
            return 0.3  # 低分
    
    # 训练循环
    for episode in range(episodes):
        # 选择一个图
        graph_idx = random.randint(0, len(graph_dataset) - 1)
        graph_data = graph_dataset[graph_idx]
        
        # 创建环境
        env = SwarmEnvironment(graph_data)
        observations, avail_actions = env.reset()
        
        # 计算epsilon（随时间衰减的探索率）
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-1. * episode / eps_decay)
        
        # 存储episode数据
        total_reward = 0
        paths = []  # 存储所有智能体的路径
        
        # 环境交互
        done = False
        while not done:
            # 所有智能体选择动作
            actions = []
            for i, (obs, avail) in enumerate(zip(observations, avail_actions)):
                # 准备观察数据
                current_node_feat = torch.tensor(obs['current_node_feat'], dtype=torch.float32).unsqueeze(0).to(device)
                path_node_feats = torch.tensor(obs['path_node_feats'], dtype=torch.float32).unsqueeze(0).to(device)
                avail_tensor = torch.tensor(avail, dtype=torch.bool).unsqueeze(0).to(device)
                
                # epsilon-greedy策略
                if random.random() < epsilon:
                    # 随机选择一个可用动作
                    valid_actions = [i for i, v in enumerate(avail) if v]
                    if valid_actions:
                        action = random.choice(valid_actions)
                    else:
                        action = random.randint(0, 9)  # 随机选择任意动作
                else:
                    # 使用Q网络选择动作
                    with torch.no_grad():
                        q_values = q_network(current_node_feat, path_node_feats, avail_tensor)
                        action = q_values.argmax(dim=1).item()
                
                actions.append(action)
            
            # 执行动作
            next_observations, rewards, done, next_avail_actions, info = env.step(actions)
            
            # 存储到经验回放缓冲区
            for i in range(len(observations)):
                # 只存储未完成的智能体的经验
                if not (env.agent_states[i]['done'] and env.agent_states[i]['steps'] == 0):
                    state = (observations[i]['current_node_feat'], observations[i]['path_node_feats'])
                    next_state = (next_observations[i]['current_node_feat'], next_observations[i]['path_node_feats'])
                    
                    replay_buffer.push(
                        state=state,
                        action=actions[i],
                        reward=rewards[i],
                        next_state=next_state,
                        done=env.agent_states[i]['done'],
                        avail_actions=avail_actions[i],
                        next_avail_actions=next_avail_actions[i]
                    )
            
            # 更新当前观察
            observations = next_observations
            avail_actions = next_avail_actions
            
            # 累计奖励
            total_reward += sum(rewards)
            
            # 从经验回放缓冲区中学习
            if len(replay_buffer) > batch_size:
                # 采样
                states, actions, rewards, next_states, dones, avail_actions, next_avail_actions = replay_buffer.sample(batch_size)
                
                # 准备批量数据
                current_feats = torch.tensor(np.vstack([s[0] for s in states]), dtype=torch.float32).to(device)
                path_feats = torch.tensor(np.array([s[1] for s in states]), dtype=torch.float32).to(device)
                next_current_feats = torch.tensor(np.vstack([s[0] for s in next_states]), dtype=torch.float32).to(device)
                next_path_feats = torch.tensor(np.array([s[1] for s in next_states]), dtype=torch.float32).to(device)
                batch_actions = torch.tensor(actions, dtype=torch.long).to(device)
                batch_rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                batch_dones = torch.tensor(dones, dtype=torch.float32).to(device)
                batch_avail = torch.tensor(np.array(avail_actions), dtype=torch.bool).to(device)
                batch_next_avail = torch.tensor(np.array(next_avail_actions), dtype=torch.bool).to(device)
                
                # 计算当前Q值
                q_values = q_network(current_feats, path_feats, batch_avail)
                q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # 计算下一状态的最大Q值
                with torch.no_grad():
                    next_q_values = target_network(next_current_feats, next_path_feats, batch_next_avail)
                    next_q_values = next_q_values.max(1)[0]
                    expected_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)
                
                # 计算损失并更新
                loss = F.smooth_l1_loss(q_values, expected_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
                optimizer.step()
        
        # 收集路径数据
        for agent in env.agent_states:
            if agent['success']:
                # 收集路径中的边
                path_edges = []
                for i in range(len(agent['path']) - 1):
                    src = agent['path'][i]
                    dst = agent['path'][i + 1]
                    path_edges.append((src, dst))
                paths.append(path_edges)

        # 如果有成功的路径，更新信息素
        if paths:
            # 计算路径奖励（使用模拟LLM评分或其他指标）
            path_rewards = [simulate_llm_scoring(path, env) for path in paths]
            
            # 更新信息素
            env.update_pheromone(paths, path_rewards)
            
            # 记录LLM评分
            avg_score = sum(path_rewards) / len(path_rewards)
            env.update_path_score(avg_score)
        
        # 获取Top-K路径并计算覆盖率
        top_paths = env.get_top_k_paths(k=10)
        total_nodes = len(env.nodes)
        covered_nodes = set()
        for path in top_paths:
            for edge, _ in path:
                covered_nodes.add(edge[0])
                covered_nodes.add(edge[1])
        coverage_rate = len(covered_nodes) / total_nodes if total_nodes > 0 else 0
        path_coverage_rates.append(coverage_rate)
        
        # 记录信息素熵
        if env.entropy_history:
            entropy_history.append(env.entropy_history[-1])
        
        # 记录episode结果
        episode_rewards.append(total_reward)
        episode_success_rates.append(info['success_rate'])
        
        # 生成信息素熵可视化
        if (episode + 1) % 10 == 0 and entropy_history:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(entropy_history)), entropy_history)
            plt.axhline(y=0.8 * entropy_history[0] if entropy_history else 0, color='r', linestyle='--', 
                       label='收敛阈值 (80% 初始熵)')
            plt.xlabel('Episode')
            plt.ylabel('信息素熵')
            plt.title(f'信息素熵随训练进度变化 (Episode {episode+1})')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'entropy_ep{episode+1}.png'))
            plt.close()
            
            # 添加到TensorBoard
            writer.add_scalar('Metrics/PheromoneEntropy', entropy_history[-1], episode)
            writer.add_scalar('Metrics/PathCoverageRate', coverage_rate, episode)
            writer.add_figure('Plots/EntropyHistory', plt.gcf(), episode)
        
        # 检查是否已收敛
        if env.is_converged():
            print(f"蚁群算法已收敛，提前结束训练。Episode: {episode+1}")
            
            # 生成最终热力图可视化
            if env.pheromone:
                # 创建边的热力图
                edge_data = []
                for (src, dst), pheromone in env.pheromone.items():
                    edge_data.append({
                        'source': str(src),
                        'target': str(dst),
                        'pheromone': pheromone
                    })
                
                df = pd.DataFrame(edge_data)
                
                # 使用plotly创建热力图
                fig = go.Figure(data=go.Heatmap(
                    z=df['pheromone'],
                    x=df['target'],
                    y=df['source'],
                    colorscale='Viridis',
                    colorbar=dict(title='信息素浓度')
                ))
                
                fig.update_layout(
                    title='边信息素浓度热力图',
                    xaxis_title='目标节点',
                    yaxis_title='源节点'
                )
                
                fig.write_html(os.path.join(output_dir, 'visualizations', 'pheromone_heatmap.html'))
            
            break
        
        # 更新目标网络
        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_success = np.mean(episode_success_rates[-10:])
            avg_entropy = np.mean(list(entropy_history)[-10:]) if entropy_history else 0
            avg_coverage = np.mean(path_coverage_rates[-10:]) if path_coverage_rates else 0
            
            print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}, Success Rate: {avg_success:.2f}, "
                  f"Entropy: {avg_entropy:.4f}, Coverage: {avg_coverage:.2f}, Epsilon: {epsilon:.2f}")
            
            # 添加到TensorBoard
            writer.add_scalar('Metrics/AvgReward', avg_reward, episode)
            writer.add_scalar('Metrics/SuccessRate', avg_success, episode)
            writer.add_scalar('Metrics/PathCoverage', avg_coverage, episode)
    
    # 关闭TensorBoard
    writer.close()
    
    return episode_rewards, episode_success_rates, entropy_history, path_coverage_rates

def main():
    parser = argparse.ArgumentParser(description='群体搜索路径训练')
    parser.add_argument('--graph_dir', type=str, required=True, help='子图目录')
    parser.add_argument('--model_dir', type=str, default='models/route2_swarm_search/checkpoints', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='models/route2_swarm_search/output', help='输出目录')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--buffer_size', type=int, default=100000, help='经验回放缓冲区大小')
    parser.add_argument('--num_agents', type=int, default=8, help='智能体数量')
    parser.add_argument('--max_path_length', type=int, default=5, help='最大路径长度')
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
    graph_dataset = GraphDataset(args.graph_dir)
    
    # 创建Q网络和目标网络
    q_network = PathQNetwork(node_dim=768, hidden_dim=args.hidden_dim).to(device)
    target_network = PathQNetwork(node_dim=768, hidden_dim=args.hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # 创建优化器
    optimizer = optim.Adam(q_network.parameters(), lr=args.lr)
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练
    print("开始训练...")
    episode_rewards, episode_success_rates, entropy_history, coverage_rates = train(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        graph_dataset=graph_dataset,
        batch_size=args.batch_size,
        episodes=args.episodes,
        gamma=args.gamma,
        device=device,
        output_dir=args.output_dir
    )
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'final_model.pt')
    torch.save({
        'q_network': q_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_success_rates': episode_success_rates,
        'entropy_history': entropy_history,
        'coverage_rates': coverage_rates
    }, model_path)
    
    # 保存最终指标图表
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(episode_success_rates)
    plt.title('Success Rates')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.subplot(2, 2, 3)
    plt.plot(entropy_history)
    plt.axhline(y=0.8 * entropy_history[0] if entropy_history else 0, color='r', linestyle='--')
    plt.title('Pheromone Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    
    plt.subplot(2, 2, 4)
    plt.plot(coverage_rates)
    plt.axhline(y=0.9, color='g', linestyle='--', label='目标覆盖率 90%')
    plt.title('Path Coverage Rates')
    plt.xlabel('Episode')
    plt.ylabel('Coverage Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'final_metrics.png'))
    plt.close()
    
    print(f"训练完成. 模型保存到 {model_path}")
    
    # 保存最佳模型（根据成功率）
    best_idx = np.argmax(episode_success_rates)
    print(f"最佳成功率: {episode_success_rates[best_idx]:.4f}, Episode: {best_idx+1}")

if __name__ == '__main__':
    main() 