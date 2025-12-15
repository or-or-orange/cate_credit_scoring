import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class PathFusionEmbedding(nn.Module):
    """决策路径信息融合模块"""
    
    def __init__(self, tree_structures: List[Dict], embedding_dim: int):
        """
        Args:
            tree_structures: 树结构信息列表
            embedding_dim: 嵌入维度
        """
        super().__init__()
        
        self.tree_structures = tree_structures
        self.embedding_dim = embedding_dim
        self.n_trees = len(tree_structures)
        
        # 计算总节点数
        self.n_total_nodes = sum(t['n_nodes'] for t in tree_structures)
        
        # 为每个节点创建嵌入向量
        self.node_embeddings = nn.Embedding(self.n_total_nodes, embedding_dim)
        
        # 初始化嵌入向量（使用节点样本均值）
        self._initialize_embeddings()
        
        # LSTM用于路径融合
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 构建叶节点到全局节点ID的映射
        self._build_leaf_to_global_mapping()

        # 新增：记录每棵树对应的全局叶节点ID范围（start, end）
        self.tree_leaf_ranges = []  # 存储 (tree_idx, leaf_start, leaf_end)
        current_leaf = 0
        for tree_idx, tree_info in enumerate(tree_structures):
            n_leaves = tree_info['n_leaves']
            self.tree_leaf_ranges.append((tree_idx, current_leaf, current_leaf + n_leaves))
            current_leaf += n_leaves
    
    def _initialize_embeddings(self):
        """使用节点样本均值初始化嵌入向量（优化版：用线性层替代平均分组）"""
        global_node_id = 0
        
        # 检查是否有节点样本数据
        if self.tree_structures and 'node_samples' in self.tree_structures[0]:
            first_node_samples = next(iter(self.tree_structures[0]['node_samples'].values()))
            n_features = len(first_node_samples)
            # 创建线性层用于降维
            projection = nn.Linear(n_features, self.embedding_dim)
            
            for tree_info in self.tree_structures:
                node_samples = tree_info['node_samples']
                for local_node_id in range(tree_info['n_nodes']):
                    if local_node_id in node_samples:
                        sample_mean = node_samples[local_node_id]
                        # 转换为张量并通过线性层降维
                        sample_tensor = torch.tensor(sample_mean, dtype=torch.float32).unsqueeze(0)
                        init_vector = projection(sample_tensor).squeeze(0)
                        self.node_embeddings.weight.data[global_node_id] = init_vector
                    global_node_id += 1
        else:
            # 若无样本数据，使用随机初始化
            # nn.init.xavier_uniform_(self.node_embeddings.weight.data)
            nn.init.normal_(self.node_embeddings.weight, mean=0.0, std=0.01) 
    
    def _build_leaf_to_global_mapping(self):
        """构建叶节点到全局节点ID的映射"""
        self.leaf_to_global = {}
        global_node_offset = 0
        global_leaf_id = 0
        
        for tree_idx, tree_info in enumerate(self.tree_structures):
            leaf_paths = tree_info['leaf_paths']
            
            for local_leaf_id, path in leaf_paths.items():
                # 将局部节点ID转换为全局节点ID
                global_path = [node_id + global_node_offset for node_id in path]
                self.leaf_to_global[global_leaf_id] = global_path
                global_leaf_id += 1
            
            global_node_offset += tree_info['n_nodes']
    
    def forward(self, cross_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            cross_features: multi-hot向量 (batch_size, n_leaves)
        
        Returns:
            path_fusion_matrix: (batch_size, n_trees, embedding_dim)
        """
        batch_size = cross_features.shape[0]
        device = cross_features.device
        
        # 获取每个样本激活的叶节点
        activated_leaves = (cross_features > 0).nonzero(as_tuple=False)  # (n_activated, 2)
        
        # 初始化输出
        path_fusion_matrix = torch.zeros(
            batch_size, self.n_trees, self.embedding_dim,
            device=device
        )
        
        # 对每个样本处理
        for sample_idx in range(batch_size):
            # 获取该样本激活的叶节点
            sample_leaves = activated_leaves[activated_leaves[:, 0] == sample_idx, 1]
            
            for leaf_id in sample_leaves.cpu().numpy():
                leaf_id = int(leaf_id)
                # 通过叶节点ID查找所属树索引
                tree_idx = None
                for t_idx, leaf_start, leaf_end in self.tree_leaf_ranges:
                    if leaf_start <= leaf_id < leaf_end:
                        tree_idx = t_idx
                        break
                if tree_idx is None:
                    raise ValueError(f"Leaf ID {leaf_id} not found in any tree")
                
                # 获取该叶节点的路径（全局节点ID）
                global_path = self.leaf_to_global[leaf_id]
                
                # 获取路径上所有节点的嵌入
                path_node_ids = torch.tensor(global_path, dtype=torch.long, device=device)
                path_embeddings = self.node_embeddings(path_node_ids)  # (path_length, embedding_dim)
                
                # LSTM融合路径信息
                path_embeddings = path_embeddings.unsqueeze(0)  # (1, path_length, embedding_dim)
                _, (h_n, _) = self.lstm(path_embeddings)
                path_vector = h_n[-1, 0, :]  # (embedding_dim,)
                
                # 存储到对应的树位置（使用正确的tree_idx）
                path_fusion_matrix[sample_idx, tree_idx] = path_vector
        
        return path_fusion_matrix
