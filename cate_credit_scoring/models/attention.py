import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """加性注意力机制"""
    
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: 原始特征维度
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 注意力网络（MLP）
        self.W = nn.Linear(input_dim + embedding_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, X_original: torch.Tensor, 
                path_fusion_matrix: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            X_original: 原始特征 (batch_size, input_dim)
            path_fusion_matrix: 路径融合矩阵 (batch_size, n_trees, embedding_dim)
        
        Returns:
            representation: 加权表示 (batch_size, embedding_dim)
            attention_weights: 注意力权重 (batch_size, n_trees)
        """
        batch_size, n_trees, embedding_dim = path_fusion_matrix.shape
        
        # 扩展原始特征以匹配树的数量
        X_expanded = X_original.unsqueeze(1).repeat(1, n_trees, 1)
        # shape: (batch_size, n_trees, input_dim)
        
        # 拼接原始特征和路径融合向量
        concat_features = torch.cat([X_expanded, path_fusion_matrix], dim=2)
        # shape: (batch_size, n_trees, input_dim + embedding_dim)
        
        # 计算注意力分数
        hidden = self.relu(self.W(concat_features))  # (batch_size, n_trees, hidden_dim)
        scores = self.v(hidden).squeeze(-1)  # (batch_size, n_trees)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=1)
        
        # 加权求和
        representation = torch.sum(
            attention_weights.unsqueeze(-1) * path_fusion_matrix,
            dim=1
        )
        # shape: (batch_size, embedding_dim)
        
        return representation, attention_weights
