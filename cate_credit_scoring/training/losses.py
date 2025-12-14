import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """监督对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, hidden_vectors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算监督对比损失
        
        Args:
            hidden_vectors: 隐向量 (2*batch_size, embedding_dim)
            labels: 标签 (2*batch_size,)
        
        Returns:
            loss: 对比损失标量
        """
        # L2归一化
        hidden_vectors = F.normalize(hidden_vectors, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(hidden_vectors, hidden_vectors.T)
        # shape: (2*batch_size, 2*batch_size)
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = 1 - mask_positive
        
        # 移除自身
        mask_positive.fill_diagonal_(0)
        
        # 计算对比损失
        batch_size = hidden_vectors.shape[0]
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            # 正样本索引
            positive_indices = mask_positive[i].nonzero(as_tuple=True)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # 负样本索引
            negative_indices = mask_negative[i].nonzero(as_tuple=True)[0]
            
            # 计算分子：正样本相似度
            pos_sim = similarity_matrix[i, positive_indices] / self.temperature
            pos_exp = torch.exp(pos_sim)
            
            # 计算分母：所有负样本相似度
            neg_sim = similarity_matrix[i, negative_indices] / self.temperature
            neg_exp = torch.exp(neg_sim).sum()
            
            # 对每个正样本计算损失
            sample_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
            loss += sample_loss
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CombinedLoss(nn.Module):
    """组合损失：分类损失 + 对比损失 + 正则化"""
    
    def __init__(self, lambda_reg: float = 0.001, temperature: float = 0.07):
        """
        Args:
            lambda_reg: L2正则化系数
            temperature: 对比学习温度参数
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCELoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor,
                hidden_vectors: torch.Tensor, model_parameters) -> tuple:
        """
        计算总损失
        
        Args:
            predictions: 预测概率 (2*batch_size,)
            labels: 标签 (2*batch_size,)
            hidden_vectors: 隐向量 (2*batch_size, embedding_dim)
            model_parameters: 模型参数（用于正则化）
        
        Returns:
            (total_loss, cls_loss, con_loss, reg_loss)
        """
        # 分类损失
        cls_loss = self.bce_loss(predictions, labels)
        
        # 对比损失
        con_loss = self.contrastive_loss(hidden_vectors, labels)
        
        # L2正则化
        reg_loss = 0.0
        for param in model_parameters:
            reg_loss += torch.norm(param, p=2)
        reg_loss = self.lambda_reg * reg_loss
        
        # 总损失
        total_loss = cls_loss + con_loss + reg_loss
        
        return total_loss, cls_loss, con_loss, reg_loss
