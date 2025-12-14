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
    
    def _initialize_embeddings(self):
        """使用节点样本均值初始化嵌入向量"""
        global_node_id = 0
        
        for tree_info in self.tree_structures:
            node_samples = tree_info['node_samples']
            
            for local_node_id in range(tree_info['n_nodes']):
                if local_node_id in node_samples:
                    sample_mean = node_samples[local_node_id]
                    
                    # 降维到embedding_dim（简单平均分组）
                    n_features = len(sample_mean)
                    group_size = max(1, n_features // self.embedding_dim)
                    
                    init_vector = []
                    for i in range(self.embedding_dim):
                        start_idx = i * group_size
                        end_idx = min((i + 1) * group_size, n_features)
                        if start_idx < n_features:
                            init_vector.append(sample_mean[start_idx:end_idx].mean())
                        else:
                            init_vector.append(0.0)
                    
                    self.node_embeddings.weight.data[global_node_id] = torch.tensor(
                        init_vector, dtype=torch.float32
                    )
                
                global_node_id += 1
    
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
            
            tree_idx = 0
            for leaf_id in sample_leaves.cpu().numpy():
                # 获取该叶节点的路径（全局节点ID）
                global_path = self.leaf_to_global[int(leaf_id)]
                
                # 获取路径上所有节点的嵌入
                path_node_ids = torch.tensor(global_path, dtype=torch.long, device=device)
                path_embeddings = self.node_embeddings(path_node_ids)  # (path_length, embedding_dim)
                
                # LSTM融合路径信息
                path_embeddings = path_embeddings.unsqueeze(0)  # (1, path_length, embedding_dim)
                _, (h_n, _) = self.lstm(path_embeddings)
                path_vector = h_n[-1, 0, :]  # (embedding_dim,)
                
                # 存储到对应的树位置
                path_fusion_matrix[sample_idx, tree_idx] = path_vector
                tree_idx += 1
        
        return path_fusion_matrix
