import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from typing import Tuple, List


class TreeFeatureExtractor:
    """树模型特征提取器"""
    
    def __init__(self, model_type: str = 'gbdt', n_estimators: int = 64, 
                 max_depth: int = 4, random_state: int = 42):
        """
        Args:
            model_type: 'gbdt' or 'rf'
            n_estimators: 树的数量
            max_depth: 树的最大深度
            random_state: 随机种子
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 初始化模型
        if model_type == 'gbdt':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                subsample=0.8
            )
        elif model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                max_features='sqrt'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.n_leaves = None
        self.leaf_counts = []  # 每棵树的叶节点数
        self.tree_structures = []  # 存储树结构信息
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练树模型
        
        Args:
            X: 特征 (n_samples, n_features)
            y: 标签 (n_samples,)
        """
        print(f"\nTraining {self.model_type.upper()} with {self.n_estimators} trees...")
        self.model.fit(X, y)
        
        # 计算叶节点总数
        self._count_leaves()
        
        # 提取树结构
        self._extract_tree_structures(X)
        
        print(f"Total leaves: {self.n_leaves}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换为交叉特征（multi-hot向量）
        
        Args:
            X: 特征 (n_samples, n_features)
        
        Returns:
            cross_features: multi-hot向量 (n_samples, n_leaves)
        """
        # 获取每棵树的叶节点索引
        leaf_indices = self.model.apply(X)  # shape: (n_samples, n_estimators)
        
        # 转换为multi-hot向量
        cross_features = self._to_multihot(leaf_indices)
        
        return cross_features
    
    def _count_leaves(self):
        """计算叶节点总数"""
        self.leaf_counts = []
        
        if self.model_type == 'gbdt':
            for tree in self.model.estimators_.flatten():
                n_leaves = tree.tree_.n_leaves
                self.leaf_counts.append(n_leaves)
        else:  # rf
            for tree in self.model.estimators_:
                n_leaves = tree.tree_.n_leaves
                self.leaf_counts.append(n_leaves)
        
        self.n_leaves = sum(self.leaf_counts)
    
    def _extract_tree_structures(self, X: np.ndarray):
        """提取树结构信息（用于路径融合）"""
        self.tree_structures = []
        
        if self.model_type == 'gbdt':
            trees = self.model.estimators_.flatten()
        else:
            trees = self.model.estimators_
        
        for tree_idx, tree in enumerate(trees):
            tree_info = {
                'tree': tree.tree_,
                'n_nodes': tree.tree_.node_count,
                'n_leaves': tree.tree_.n_leaves,
                'children_left': tree.tree_.children_left,
                'children_right': tree.tree_.children_right,
                'feature': tree.tree_.feature,
                'threshold': tree.tree_.threshold,
            }
            
            # 计算每个叶节点的路径
            leaf_paths = self._get_all_leaf_paths(tree_info)
            tree_info['leaf_paths'] = leaf_paths
            
            # 计算每个节点的样本均值（用于初始化嵌入）
            node_samples = self._get_node_samples(tree, X)
            tree_info['node_samples'] = node_samples
            
            self.tree_structures.append(tree_info)
    
    def _get_all_leaf_paths(self, tree_info: dict) -> dict:
        """获取所有叶节点的路径"""
        children_left = tree_info['children_left']
        children_right = tree_info['children_right']
        n_nodes = tree_info['n_nodes']
        
        # 找到所有叶节点
        leaf_ids = [i for i in range(n_nodes) if children_left[i] == children_right[i]]
        
        # 对每个叶节点，回溯到根节点
        leaf_paths = {}
        for leaf_id in leaf_ids:
            path = self._get_path_to_root(leaf_id, children_left, children_right, n_nodes)
            leaf_paths[leaf_id] = path
        
        return leaf_paths
    
    def _get_path_to_root(self, leaf_id: int, children_left: np.ndarray, 
                          children_right: np.ndarray, n_nodes: int) -> List[int]:
        """从叶节点回溯到根节点"""
        path = [leaf_id]
        
        # 构建父节点映射
        parent = np.full(n_nodes, -1, dtype=int)
        for i in range(n_nodes):
            if children_left[i] != -1:
                parent[children_left[i]] = i
            if children_right[i] != -1:
                parent[children_right[i]] = i
        
        # 回溯
        current = leaf_id
        while parent[current] != -1:
            current = parent[current]
            path.append(current)
        
        # 反转路径（从根到叶）
        path.reverse()
        
        return path
    
    def _get_node_samples(self, tree, X: np.ndarray) -> dict:
        """计算每个节点的样本均值"""
        decision_path = tree.decision_path(X)
        node_samples = {}
        
        for node_id in range(tree.tree_.node_count):
            # 获取到达该节点的样本
            samples_at_node = X[decision_path[:, node_id].toarray().flatten().astype(bool)]
            
            if len(samples_at_node) > 0:
                node_samples[node_id] = samples_at_node.mean(axis=0)
            else:
                node_samples[node_id] = np.zeros(X.shape[1])
        
        return node_samples
    
    def _to_multihot(self, leaf_indices: np.ndarray) -> np.ndarray:
        """
        将叶节点索引转换为multi-hot向量
        
        Args:
            leaf_indices: (n_samples, n_estimators)
        
        Returns:
            multihot: (n_samples, n_leaves)
        """
        n_samples, n_trees = leaf_indices.shape
        multihot = np.zeros((n_samples, self.n_leaves), dtype=np.float32)
        
        offset = 0
        for tree_idx in range(n_trees):
            for sample_idx in range(n_samples):
                leaf_id = leaf_indices[sample_idx, tree_idx]
                multihot[sample_idx, offset + leaf_id] = 1.0
            
            offset += self.leaf_counts[tree_idx]
        
        return multihot
