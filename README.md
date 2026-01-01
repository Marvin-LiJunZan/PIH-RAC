# PIH-RAC: Peak Identification for Recycled Aggregate Concrete

## 📖 项目简介

本项目专注于再生骨料混凝土（RAC）的峰值识别和本构关系研究，通过多种机器学习方法预测峰值应力和峰值应变，为再生混凝土的力学性能分析提供支持。

## 🎯 主要功能

- **峰值应力预测** - 使用XGBoost、LightGBM、CatBoost、RandomForest等算法
- **峰值应变预测** - 集成多种机器学习模型和深度学习方法
- **噪声分析** - Bootstrap和分位数回归方法生成预测区间
- **多目标优化** - NSGA-III算法优化混凝土配合比
- **弹性模量计算** - 基于实验数据的弹性模量预测
- **能量分析** - 混凝土破坏过程的能量演化分析

## 📁 项目结构

```
PIHRAC/
├── 📊 dataset/                    # 数据集文件
│   ├── dataset_final.xlsx         # 主数据集
│   ├── dataset_with_*.xlsx        # 不同模型的数据集版本
│   └── cluster_analysis/          # 聚类分析
├── 🧠 LSTM/                      # 深度学习模型
│   ├── Bidirectional_LSTM_Enhanced_cross_validation.py
│   └── trained_model_cross_validation.py
├── 📈 peak_stress/               # 峰值应力预测
│   ├── XGBoost/
│   ├── LightGBM/
│   ├── CatBoost/
│   └── RandomForest/
├── 📉 peak_strain/               # 峰值应变预测
│   ├── XGBoost/
│   ├── LightGBM/
│   ├── CatBoost/
│   ├── NGBoost/
│   └── PINN/                     # 物理信息神经网络
├── 🔧 elastic_modulus/           # 弹性模量分析
├── ⚡ energy_analysis/            # 能量分析
├── 🎯 multi_objective_optimization/ # 多目标优化
├── 📝 画图/                      # 可视化脚本（保持原名称）
├── 💻 软件/                      # 软件工具（保持原名称）
└── 📄 软著/                      # 软件申请材料（保持原名称）
```

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
```

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `scikit-learn` - 机器学习算法
- `xgboost` - XGBoost模型
- `lightgbm` - LightGBM模型
- `catboost` - CatBoost模型
- `tensorflow/pytorch` - 深度学习
- `optuna` - 超参数优化
- `shap` - 模型解释
- `pandas, numpy` - 数据处理
- `matplotlib, seaborn` - 数据可视化

## 📋 使用方法

### 1. 数据准备

```python
import pandas as pd
from dataset.dataloader import load_data

# 加载数据集
data = load_data('dataset/dataset_final.xlsx')
```

### 2. 模型训练

#### XGBoost峰值应力预测
```bash
cd peak_stress/XGBoost
python XGBoost_train.py
```

#### LSTM峰值应变预测
```bash
cd LSTM
python trained_model_cross_validation.py
```

### 3. 模型分析

```bash
# 噪声分析和预测区间
python peak_stress/XGBoost/XGBoost_noise_analysis.py

# SHAP模型解释
python peak_strain/CatBoost/CatBoost_noise_analysis.py
```

## 🎨 特性功能

### 🔍 模型解释性
- **SHAP分析** - 特征重要性可视化
- **PDP分析** - 部分依赖图
- **特征交互** - 2D交互效应分析

### 📊 预测区间
- **Bootstrap方法** - 重采样置信区间
- **分位数回归** - 不确定性量化
- **鲁棒性分析** - 模型稳定性评估

### 🎯 多目标优化
```python
from multi_objective_optimization import nsga3_optimization

# 优化混凝土配合比
results = nsga3_optimization(
    objectives=['peak_stress', 'peak_strain', 'cost'],
    constraints=['w_c_ratio', 'ca_content']
)
```

## 📈 模型性能

| 模型 | 峰值应力 R² | 峰值应变 R² | RMSE |
|------|-------------|-------------|------|
| XGBoost | 0.96+ | 0.94+ | < 0.05 |
| LightGBM | 0.95+ | 0.93+ | < 0.06 |
| CatBoost | 0.95+ | 0.92+ | < 0.07 |
| LSTM | - | 0.89+ | < 0.08 |

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📧 联系方式

- **作者**: Marvin Li Junzan
- **邮箱**: [your-email@example.com]
- **GitHub**: [Marvin-LiJunZan](https://github.com/Marvin-LiJunZan)

## 🙏 致谢

感谢所有为再生混凝土研究做出贡献的研究者和开源社区的支持。

## 📚 参考文献

1. 相关的混凝土力学研究论文
2. 机器学习在土木工程中的应用
3. 再生骨料混凝土本构关系研究

---

**注意**: 本项目仅供学术研究使用，商业应用请遵循相关许可协议。
