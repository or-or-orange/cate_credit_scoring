import torch
import torch.nn as nn
from .embedding import PathFusionEmbedding
from .attention import AdditiveAttention


class CATE(nn.Module):
    """CATE主模型"""
    
    def __init__(self, config, tree_extractor_gbdt, tree_extractor_rf):
        """
        Args:
            config: 配置对象
            tree_extractor_gbdt: GBDT特征提取器
            tree_extractor_rf: RF特征提取器
        """
        super().__init__()
        
        self.config = config
        self.tree_extractor_gbdt = tree_extractor_gbdt
        self.tree_extractor_rf = tree_extractor_rf
        
        # 获取输入维度
        self.input_dim = None  # 将在第一次forward时设置
        self.embedding_dim = config.model.embedding_dim
        self.attention_hidden_dim = config.model.attention_hidden_dim
        
        # 路径融合嵌入
        self.path_fusion_gbdt = PathFusionEmbedding(
            tree_structures=tree_extractor_gbdt.tree_structures,
            embedding_dim=self.embedding_dim
        )
        self.path_fusion_rf = PathFusionEmbedding(
            tree_structures=tree_extractor_rf.tree_structures,
            embedding_dim=self.embedding_dim
        )
        
        # 注意力机制（延迟初始化）
        self.attention = None
        
        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # 分类层（加性模型）
        self.w_gbdt = nn.Linear(self.embedding_dim, 1, bias=False)
        self.w_rf = nn.Linear(self.embedding_dim, 1, bias=False)
        self.bias_gbdt = nn.Parameter(torch.zeros(1))
        self.bias_rf = nn.Parameter(torch.zeros(1))
        self.bias_global = nn.Parameter(torch.zeros(1))
    
    def _init_attention(self, input_dim: int):
        """延迟初始化注意力机制"""
        if self.attention is None:
            self.attention = AdditiveAttention(
                input_dim=input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.attention_hidden_dim
            )
            self.input_dim = input_dim
    
    def forward(self, X: torch.Tensor, return_attention: bool = False):
        """
        前向传播
        
        Args:
            X: 原始特征 (batch_size, input_dim)
            return_attention: 是否返回注意力权重
        
        Returns:
            prediction: 预测概率 (batch_size,)
            hidden_vectors: (hidden_gbdt, hidden_rf) 用于对比学习
            attention_weights: (可选) (attn_gbdt, attn_rf)
        """
        # 延迟初始化注意力机制
        if self.attention is None:
            self._init_attention(X.shape[1])
            self.attention = self.attention.to(X.device)
        
        # 1. 树模型提取交叉特征
        cross_features_gbdt, cross_features_rf = self._extract_cross_features(X)
        
        # 2. 路径融合
        path_matrix_gbdt = self.path_fusion_gbdt(cross_features_gbdt)
        path_matrix_rf = self.path_fusion_rf(cross_features_rf)
        
        # 3. 注意力机制
        repr_gbdt, attn_gbdt = self.attention(X, path_matrix_gbdt)
        repr_rf, attn_rf = self.attention(X, path_matrix_rf)
        
        # 4. 投影头（对比学习）
        hidden_gbdt = self.projection_head(repr_gbdt)
        hidden_rf = self.projection_head(repr_rf)
        
        # 5. 分类（加性模型）
        logit_gbdt = self.w_gbdt(repr_gbdt) + self.bias_gbdt
        logit_rf = self.w_rf(repr_rf) + self.bias_rf
        logit = logit_gbdt + logit_rf + self.bias_global
        prediction = torch.sigmoid(logit).squeeze(-1)
        
        if return_attention:
            return prediction, (hidden_gbdt, hidden_rf), (attn_gbdt, attn_rf)
        else:
            return prediction, (hidden_gbdt, hidden_rf)
    
    def _extract_cross_features(self, X: torch.Tensor) -> tuple:
        """
        提取交叉特征（数据增强）
        
        Args:
            X: 原始特征 (batch_size, input_dim)
        
        Returns:
            (cross_features_gbdt, cross_features_rf)
        """
        X_np = X.cpu().numpy()
        
        # GBDT交叉特征
        cross_features_gbdt = self.tree_extractor_gbdt.transform(X_np)
        cross_features_gbdt = torch.from_numpy(cross_features_gbdt).to(X.device)
        
        # RF交叉特征
        cross_features_rf = self.tree_extractor_rf.transform(X_np)
        cross_features_rf = torch.from_numpy(cross_features_rf).to(X.device)
        
        return cross_features_gbdt, cross_features_rf
