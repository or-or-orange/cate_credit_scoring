import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, missing_threshold: float = 0.5, categorical_threshold: int = 20):
        self.missing_threshold = missing_threshold
        self.categorical_threshold = categorical_threshold
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并转换训练数据
        
        Args:
            X: 特征数据
            y: 标签数据
        
        Returns:
            (X_transformed, y_transformed)
        """
        print("\n=== Data Preprocessing ===")
        
        # 1. 标准化标签
        y = self._standardize_labels(y)
        
        # 2. 移除高缺失率特征
        X = self._remove_high_missing_features(X)
        
        # 3. 移除含缺失值的样本
        X, y = self._remove_missing_samples(X, y)
        
        # 4. 识别特征类型
        self._identify_feature_types(X)
        
        # 5. 编码分类特征
        X = self._encode_categorical_features(X, fit=True)
        
        # 6. 保存特征名
        self.feature_names = X.columns.tolist()
        
        print(f"Final shape: X={X.shape}, y={y.shape}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        return X.values.astype(np.float32), y.values.astype(np.float32)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        转换测试数据
        
        Args:
            X: 特征数据
        
        Returns:
            X_transformed
        """
        # 使用相同的列
        X = X[self.feature_names[:len([c for c in self.feature_names if c in X.columns])]]
        
        # 编码分类特征
        X = self._encode_categorical_features(X, fit=False)
        
        return X.values.astype(np.float32)
    
    def _standardize_labels(self, y: pd.Series) -> pd.Series:
        """标准化标签为0/1"""
        unique_values = y.unique()
        
        if len(unique_values) != 2:
            raise ValueError(f"Expected binary labels, got {len(unique_values)} unique values")
        
        # 映射到0和1
        if set(unique_values) != {0, 1}:
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            y = y.map(mapping)
            print(f"Labels mapped: {mapping}")
        
        return y
    
    def _remove_high_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """移除高缺失率特征"""
        missing_ratios = X.isnull().sum() / len(X)
        high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        
        if high_missing_cols:
            print(f"Removing {len(high_missing_cols)} features with missing rate > {self.missing_threshold}")
            X = X.drop(columns=high_missing_cols)
        
        return X
    
    def _remove_missing_samples(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """移除含缺失值的样本"""
        missing_mask = X.isnull().any(axis=1)
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            print(f"Removing {n_missing} samples with missing values")
            X = X[~missing_mask]
            y = y[~missing_mask]
        
        return X, y
    
    def _identify_feature_types(self, X: pd.DataFrame):
        """识别数值和分类特征"""
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                self.categorical_features.append(col)
            elif X[col].dtype in ['int64', 'float64']:
                # 如果唯一值很少，也视为分类特征
                if X[col].nunique() < 10:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """编码分类特征"""
        X = X.copy()
        
        for col in self.categorical_features:
            if col not in X.columns:
                continue
            
            cardinality = X[col].nunique()
            
            if cardinality < self.categorical_threshold:
                # One-hot编码
                if fit:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                else:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                # 频率编码
                if fit:
                    freq_map = X[col].value_counts(normalize=True).to_dict()
                    self.label_encoders[col] = freq_map
                    X[col] = X[col].map(freq_map).fillna(0)
                else:
                    freq_map = self.label_encoders.get(col, {})
                    X[col] = X[col].map(freq_map).fillna(0)
        
        return X
