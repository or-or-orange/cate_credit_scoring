import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """监督对比损失"""
    
    def __init__(self, temp_pos: float = 5.0, temp_neg: float = 10.0):
        """
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temp_pos = temp_pos  # 正样本温度
        self.temp_neg = temp_neg
        # self.temperature = temperature
    
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
            positive_indices = mask_positive[i].nonzero(as_tuple=True)[0]
            
            if len(positive_indices) == 0:
                continue
            
            negative_indices = mask_negative[i].nonzero(as_tuple=True)[0]

            if len(negative_indices) > 64:  # 从128进一步减少到64
                negative_indices = negative_indices[torch.randperm(len(negative_indices))[:64]]
            pos_sim = similarity_matrix[i, positive_indices] / self.temp_pos

            pos_exp = torch.exp(pos_sim)  # (n_pos,)
            
            # 负样本相似度（每个负样本单独计算）
            neg_sim = similarity_matrix[i, negative_indices] / self.temp_neg
            neg_exp = torch.exp(neg_sim).unsqueeze(0)  # (1, n_neg)

            sample_loss = -torch.log(
                pos_exp.unsqueeze(1) / (pos_exp.unsqueeze(1) + neg_exp)  # (n_pos, n_neg)
            ).mean()  # 平均所有正负对
            
            loss += sample_loss
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CombinedLoss(nn.Module):
    """组合损失：分类损失 + 对比损失 + 正则化"""
    
    def __init__(self, lambda_reg: float = 0.001, temp_pos: float = 5.0, temp_neg: float = 10.0):
        """
        Args:
            lambda_reg: L2正则化系数
            temperature: 对比学习温度参数
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCELoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temp_pos=temp_pos, temp_neg=temp_neg)
    
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
