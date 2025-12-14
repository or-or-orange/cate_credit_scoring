import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from utils.config import Config
from utils.metrics import print_metrics
from data.loader import DataLoader as CreditDataLoader
from data.preprocessor import DataPreprocessor
from models.tree_models import TreeFeatureExtractor
from models.cate import CATE
from training.trainer import CATETrainer


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    创建数据加载器
    
    Args:
        X: 特征
        y: 标签
        batch_size: 批次大小
        shuffle: 是否打乱
    
    Returns:
        DataLoader
    """
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"⏱️  {name}: {end - start:.2f}s")


def train_fold(config, X_train, y_train, X_val, y_val, fold: int):
    """训练单个fold（添加计时）"""
    print(f"\n{'='*70}")
    print(f"Training Fold {fold}")
    print(f"{'='*70}")
    
    # 1. 预训练树模型
    with timer("Tree models training"):
        tree_extractor_gbdt = TreeFeatureExtractor(
            model_type='gbdt',
            n_estimators=config.tree.n_estimators,
            max_depth=config.tree.max_depth,
            random_state=config.tree.random_state
        )
        tree_extractor_gbdt.fit(X_train, y_train)
        
        tree_extractor_rf = TreeFeatureExtractor(
            model_type='rf',
            n_estimators=config.tree.n_estimators,
            max_depth=config.tree.max_depth,
            random_state=config.tree.random_state
        )
        tree_extractor_rf.fit(X_train, y_train)
    
    # 2. 构建CATE模型
    with timer("CATE model building"):
        model = CATE(config, tree_extractor_gbdt, tree_extractor_rf)
    
    # 3. 创建数据加载器
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # 4. 训练模型
    with timer("CATE model training"):
        trainer = CATETrainer(model, config)
        trainer.fit(train_loader, val_loader, n_epochs=config.training.n_epochs)
    
    # 5. 评估
    trainer.load_checkpoint('best_model.pt')
    val_metrics = trainer.evaluate(val_loader)
    
    return val_metrics


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*20 + "CATE Credit Scoring")
    print("="*70)
    
    # 1. 加载配置
    print("\n--- Loading Configuration ---")
    config = Config('config.yaml')
    print(f"Config loaded from: config.yaml")
    
    # 2. 加载数据
    print("\n--- Loading Data ---")
    data_loader = CreditDataLoader(config.data.path)
    data = data_loader.load()
    
    # 假设最后一列是标签
    X_raw = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1]
    
    # 3. 数据预处理
    print("\n--- Data Preprocessing ---")
    preprocessor = DataPreprocessor(
        missing_threshold=config.data.missing_threshold,
        categorical_threshold=config.data.categorical_threshold
    )
    X, y = preprocessor.fit_transform(X_raw, y_raw)
    
    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    print(f"Positive rate: {y.mean():.2%}")
    
    # 4. K折交叉验证
    print(f"\n--- {config.cv.n_folds}-Fold Cross Validation ---")
    
    kfold = StratifiedKFold(
        n_splits=config.cv.n_folds,
        shuffle=config.cv.shuffle,
        random_state=config.cv.random_state
    )
    
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练单个fold
        val_metrics = train_fold(config, X_train, y_train, X_val, y_val, fold)
        all_metrics.append(val_metrics)
        
        # 打印当前fold结果
        print_metrics(val_metrics, prefix=f"Fold {fold} ")
    
    # 5. 汇总结果
    print("\n" + "="*70)
    print(" "*15 + "Final Results (Cross Validation)")
    print("="*70)
    
    # 计算平均指标
    avg_metrics = {}
    std_metrics = {}
    
    for metric_name in all_metrics[0].keys():
        if metric_name not in ['tp', 'fp', 'tn', 'fn']:  # 跳过混淆矩阵元素
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values)
    
    print("\nMetric              Mean ± Std")
    print("-" * 70)
    print(f"Recall:             {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Precision:          {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"F-score:            {avg_metrics['f_score']:.4f} ± {std_metrics['f_score']:.4f}")
    print(f"G-mean:             {avg_metrics['g_mean']:.4f} ± {std_metrics['g_mean']:.4f}")
    print(f"Accuracy:           {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {avg_metrics['balanced_accuracy']:.4f} ± {std_metrics['balanced_accuracy']:.4f}")
    print(f"MCC:                {avg_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")
    print(f"AUC:                {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print("="*70)
    
    # 6. 保存结果
    print("\n--- Saving Results ---")
    
    import json
    results = {
        'config': {
            'n_estimators': config.tree.n_estimators,
            'max_depth': config.tree.max_depth,
            'embedding_dim': config.model.embedding_dim,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'n_epochs': config.training.n_epochs
        },
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'fold_metrics': all_metrics
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: results.json")
    
    print("\n" + "="*70)
    print(" "*25 + "Training Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
