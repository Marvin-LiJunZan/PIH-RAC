# PIH-RAC: Peak Identification for Recycled Aggregate Concrete

## 📖 Project Overview

This project focuses on peak identification and constitutive relationship research for Recycled Aggregate Concrete (RAC). It predicts peak stress and peak strain using various machine learning methods, providing support for mechanical property analysis of recycled concrete.

## 🎯 Main Features

- **Peak Stress Prediction** - Using XGBoost, LightGBM, CatBoost, RandomForest algorithms
- **Peak Strain Prediction** - Integrating various ML models and deep learning methods
- **Noise Analysis** - Bootstrap and Quantile Regression methods for prediction intervals
- **Multi-objective Optimization** - NSGA-III algorithm for concrete mix proportion optimization
- **Elastic Modulus Calculation** - Elastic modulus prediction based on experimental data
- **Energy Analysis** - Energy evolution analysis of concrete failure process

## 📁 Project Structure

```
PIHRAC/
├── 📊 dataset/                    # Dataset files
│   ├── dataset_final.xlsx         # Main dataset
│   ├── dataset_with_*.xlsx        # Dataset versions for different models
│   └── cluster_analysis/          # Cluster analysis
├── 🧠 LSTM/                      # Deep learning models
│   ├── Bidirectional_LSTM_Enhanced_cross_validation.py
│   └── trained_model_cross_validation.py
├── 📈 peak_stress/               # Peak stress prediction
│   ├── XGBoost/
│   ├── LightGBM/
│   ├── CatBoost/
│   └── RandomForest/
├── 📉 peak_strain/               # Peak strain prediction
│   ├── XGBoost/
│   ├── LightGBM/
│   ├── CatBoost/
│   ├── NGBoost/
│   └── PINN/                     # Physics-Informed Neural Networks
├── 🔧 elastic_modulus/           # Elastic modulus analysis
├── ⚡ energy_analysis/            # Energy analysis
├── 🎯 multi_objective_optimization/ # Multi-objective optimization
├── 📝 画图/                      # Visualization scripts (original Chinese name)
├── 💻 软件/                      # Software tools (original Chinese name)
└── 📄 软著/                      # Software application materials (original Chinese name)
```

## 🚀 Quick Start

### Requirements

```bash
Python >= 3.8
```

### Installation

```bash
pip install -r requirements.txt
```

Main dependencies:
- `scikit-learn` - Machine learning algorithms
- `xgboost` - XGBoost model
- `lightgbm` - LightGBM model
- `catboost` - CatBoost model
- `tensorflow/pytorch` - Deep learning
- `optuna` - Hyperparameter optimization
- `shap` - Model interpretation
- `pandas, numpy` - Data processing
- `matplotlib, seaborn` - Data visualization

## 📋 Usage

### 1. Data Preparation

```python
import pandas as pd
from dataset.dataloader import load_data

# Load dataset
data = load_data('dataset/dataset_final.xlsx')
```

### 2. Model Training

#### XGBoost Peak Stress Prediction
```bash
cd peak_stress/XGBoost
python XGBoost_train.py
```

#### LSTM Peak Strain Prediction
```bash
cd LSTM
python trained_model_cross_validation.py
```

### 3. Model Analysis

```bash
# Noise analysis and prediction intervals
python peak_stress/XGBoost/XGBoost_noise_analysis.py

# SHAP model interpretation
python peak_strain/CatBoost/CatBoost_noise_analysis.py
```

## 🎨 Advanced Features

### 🔍 Model Interpretability
- **SHAP Analysis** - Feature importance visualization
- **PDP Analysis** - Partial dependence plots
- **Feature Interaction** - 2D interaction effect analysis

### 📊 Prediction Intervals
- **Bootstrap Method** - Resampling confidence intervals
- **Quantile Regression** - Uncertainty quantification
- **Robustness Analysis** - Model stability assessment

### 🎯 Multi-objective Optimization
```python
from multi_objective_optimization import nsga3_optimization

# Optimize concrete mix proportions
results = nsga3_optimization(
    objectives=['peak_stress', 'peak_strain', 'cost'],
    constraints=['w_c_ratio', 'ca_content']
)
```

## 📈 Model Performance

| Model | Peak Stress R² | Peak Strain R² | RMSE |
|-------|---------------|---------------|------|
| XGBoost | 0.96+ | 0.94+ | < 0.05 |
| LightGBM | 0.95+ | 0.93+ | < 0.06 |
| CatBoost | 0.95+ | 0.92+ | < 0.07 |
| LSTM | - | 0.89+ | < 0.08 |

## 🔬 Technical Details

### Input Features
- **Material Parameters**: 15 material properties
- **Compressive Strength** (fc)
- **Peak Strain** (Xiao_strain)
- **Total Features**: 17 dimensions

### Model Architecture
- **Ensemble Methods**: XGBoost, LightGBM, CatBoost, RandomForest
- **Deep Learning**: Bidirectional LSTM with attention mechanism
- **Physics-Informed**: PINN incorporating constitutive equations
- **Optimization**: NSGA-III for multi-objective problems

### Evaluation Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root mean square error
- **MAE**: Mean absolute error
- **Prediction Intervals**: 95% confidence bands

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📧 Contact

- **Author**: Junzan Li
- **Email**: [1283014568@qq.com]
- **GitHub**: [Marvin-LiJunZan](https://github.com/Marvin-LiJunZan)

## 🙏 Acknowledgments

Thanks to all researchers who have contributed to recycled concrete research and the open-source community for their support.

## 📚 References

1. Related concrete mechanics research papers
2. Machine learning applications in civil engineering
3. Constitutive relationship research for recycled aggregate concrete

---

**Note**: This project is for academic research purposes only. For commercial applications, please follow relevant licensing agreements.

## 🔗 Related Projects

- [Concrete Material Properties Database](https://github.com/your-repo/concrete-db)
- [ML in Civil Engineering](https://github.com/your-repo/ml-civil)
- [Recycled Aggregate Research](https://github.com/your-repo/rac-research)
