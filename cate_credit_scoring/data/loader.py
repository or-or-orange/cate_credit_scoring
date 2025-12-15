import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # 根据文件扩展名选择加载方式
        if self.data_path.suffix == '.csv':
            data = pd.read_csv(self.data_path, encoding="latin-1")
        elif self.data_path.suffix in ['.xlsx', '.xls']:
            data = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        print(f"Loaded data with shape: {data.shape}")
        return data
    
    @staticmethod
    def split_features_labels(data: pd.DataFrame, label_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        分离特征和标签
        
        Args:
            data: 完整数据
            label_col: 标签列名
        
        Returns:
            (features, labels)
        """
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")
        
        X = data.drop(columns=[label_col])
        y = data[label_col]
        
        return X, y
