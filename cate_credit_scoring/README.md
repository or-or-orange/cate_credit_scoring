# CATE: Contrastive Augmentation and Tree-Enhanced Embedding for Credit Scoring

PyTorch implementation of the CATE model from the paper "CATE: Contrastive augmentation and tree-enhanced embedding for credit scoring" (Information Sciences, 2023).

## Features

- ✅ Tree-based feature extraction (GBDT & Random Forest)
- ✅ Decision path information fusion with LSTM
- ✅ Additive attention mechanism
- ✅ Supervised contrastive learning
- ✅ Dual-task learning (classification + contrastive)
- ✅ K-fold cross-validation
- ✅ Comprehensive evaluation metrics

## Installation

```bash
pip install -r requirements.txt
```




# 1. 安装依赖

pip install -r requirements.txt

# 2. 准备数据（CSV格式，最后一列为标签）

# 将数据放在 data/credit_data.csv

# 3. 修改配置

# 编辑 config.yaml 设置数据路径和参数

# 4. 运行训练

python main.py

# 5. 可视化结果（可选）

python scripts/visualize.py
