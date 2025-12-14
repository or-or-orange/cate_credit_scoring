import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, matthews_corrcoef, confusion_matrix
)
from typing import Dict


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        y_pred: 预测概率 (n_samples,)
        y_true: 真实标签 (n_samples,)
        threshold: 分类阈值
    
    Returns:
        包含所有指标的字典
    """
    # 转换为二分类
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # 基础指标
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 计算各项指标
    metrics = {
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'f_score': 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0,
        'g_mean': np.sqrt(recall * specificity),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'balanced_accuracy': (recall + specificity) / 2,
        'mcc': matthews_corrcoef(y_true, y_pred_binary),
        'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """打印指标"""
    print(f"\n{prefix}Evaluation Metrics:")
    print("=" * 50)
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"F-score:            {metrics['f_score']:.4f}")
    print(f"G-mean:             {metrics['g_mean']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"MCC:                {metrics['mcc']:.4f}")
    print(f"AUC:                {metrics['auc']:.4f}")
    print("=" * 50)
    print(f"Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, "
          f"TN={metrics['tn']}, FN={metrics['fn']}")
