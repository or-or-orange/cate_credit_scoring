import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from .losses import CombinedLoss
from utils.metrics import compute_metrics


class CATETrainer:
    """CATE训练器"""
    
    def __init__(self, model, config):
        super().__init__()
        """
        Args:
            model: CATE模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,  # 每5个epoch调整一次
            gamma=0.5     # 学习率乘以0.5
        )
        # 损失函数
        self.loss_fn = CombinedLoss(
            lambda_reg=config.training.lambda_reg,
            temp_pos=config.training.temp_pos,
            temp_neg=config.training.temp_neg
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_con_loss': [],
            'train_reg_loss': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            包含各项损失的字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_con_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            predictions, (hidden_gbdt, hidden_rf) = self.model(batch_X)
            
            # 拼接增强样本（GBDT和RF作为两个视角）
            predictions_aug = torch.cat([predictions, predictions])
            labels_aug = torch.cat([batch_y, batch_y])
            hidden_aug = torch.cat([hidden_gbdt, hidden_rf])
            
            # 计算损失
            loss, cls_loss, con_loss, reg_loss = self.loss_fn(
                predictions_aug,
                labels_aug,
                hidden_aug,
                self.model.parameters()
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_con_loss += con_loss.item()
            total_reg_loss += reg_loss.item()
            n_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}'
            })
        
        # 计算平均损失
        avg_losses = {
            'total_loss': total_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches,
            'con_loss': total_con_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches
        }
        
        return avg_losses
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                # 前向传播
                predictions, _ = self.model(batch_X)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_y.numpy())
        
        # 合并所有批次
        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_labels)
        
        # 计算指标
        metrics = compute_metrics(y_pred, y_true)
        
        return metrics
    
    def fit(self, train_loader, val_loader, n_epochs: int):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
        """
        print(f"\n{'='*60}")
        print(f"Training on device: {self.device}")
        print(f"{'='*60}\n")
        
        best_auc = 0.0
        best_epoch = 0
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("-" * 60)
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度器更新（每个epoch结束后）
            self.scheduler.step()
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            # 保存历史
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['train_cls_loss'].append(train_losses['cls_loss'])
            self.history['train_con_loss'].append(train_losses['con_loss'])
            self.history['train_reg_loss'].append(train_losses['reg_loss'])
            self.history['val_metrics'].append(val_metrics)
            
            # 打印结果
            print(f"\nTraining Losses:")
            print(f"  Total: {train_losses['total_loss']:.4f}")
            print(f"  Classification: {train_losses['cls_loss']:.4f}")
            print(f"  Contrastive: {train_losses['con_loss']:.4f}")
            print(f"  Regularization: {train_losses['reg_loss']:.4f}")
            
            print(f"\nValidation Metrics:")
            print(f"  AUC: {val_metrics['auc']:.4f}")
            print(f"  F-score: {val_metrics['f_score']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  G-mean: {val_metrics['g_mean']:.4f}")
            
            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
