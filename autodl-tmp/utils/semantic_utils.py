#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语义处理工具模块

该模块提供了用于处理实体和关系语义的辅助函数，包括：
1. 语义模型加载
2. 语义相似度计算
3. 语义匹配功能
"""

import numpy as np

def normalize_name(name):
    """
    标准化名称以提高匹配率
    
    参数:
        name: 实体或关系名称
        
    返回:
        标准化后的名称
    """
    # 简化名称以提高匹配率
    return str(name).lower().replace(',', '').replace('.', '').strip()

def load_semantic_model():
    """
    加载适合中英混合文本的语义相似度模型
    
    返回:
        语义模型对象，如果加载失败则返回None
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("加载多语言语义相似度模型...")
        # 这个模型对中英文混合文本效果较好
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except ImportError:
        print("警告: 请先安装sentence-transformers库: pip install sentence-transformers")
        return None

def compute_semantic_similarity(entity1, entity2, model):
    """
    计算两个实体之间的语义相似度
    
    参数:
        entity1: 第一个实体名称
        entity2: 第二个实体名称
        model: 语义模型对象
        
    返回:
        相似度分数 (0-1)
    """
    if model is None:
        return 0.0
    
    # 编码实体名称
    embeddings = model.encode([str(entity1), str(entity2)])
    
    # 计算余弦相似度
    from numpy import dot
    from numpy.linalg import norm
    
    similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    return float(similarity)

def find_semantic_matches(query_entity, entity_list, model, threshold=0.7):
    """
    基于语义相似度查找匹配实体
    
    参数:
        query_entity: 查询实体名称
        entity_list: 候选实体列表
        model: 语义模型对象
        threshold: 相似度阈值
        
    返回:
        匹配实体列表，每项包含(实体名称, 相似度分数)
    """
    if model is None or not entity_list:
        return []
    
    # 过滤掉空字符串和默认节点标记
    valid_entities = [e for e in entity_list if e and not e.startswith("节点")]
    if not valid_entities:
        return []
    
    try:
        # 编码查询实体
        query_embedding = model.encode(str(query_entity))
        
        # 批量编码候选实体
        candidate_embeddings = model.encode([str(e) for e in valid_entities])
        
        # 计算相似度并筛选
        matches = []
        for i, embedding in enumerate(candidate_embeddings):
            from numpy import dot
            from numpy.linalg import norm
            
            # 避免除零错误
            norm_query = norm(query_embedding)
            norm_cand = norm(embedding)
            
            if norm_query > 0 and norm_cand > 0:
                similarity = dot(query_embedding, embedding) / (norm_query * norm_cand)
                if similarity >= threshold:
                    matches.append((valid_entities[i], float(similarity)))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    except Exception as e:
        print(f"计算语义相似度时出错: {e}")
        return [] 