# PINN Peak Strain Prediction - Workflow Documentation

## Code Structure

This project consists of two independent scripts for **training** and **inference**, with clear responsibilities:

| Script | Filename | Main Function | Output Content |
|------|--------|---------|---------|
| **Training Script** | `peak_strain_model_minmax.py` | Model training, hyperparameter optimization, cross-validation | Model weights, configuration files, training logs |
| **Inference Script** | `peak_strain_inference.py` | Model inference, performance evaluation, comparison with Xiao formula | Prediction results, performance metrics, comparison analysis |

---

## 1. Training Script (`peak_strain_model_minmax.py`)

### Functionality
- ✅ Focused on model training
- ✅ Optuna hyperparameter optimization
- ✅ Subset cross-validation
- ✅ Save model and configuration

### Output Metrics

#### Training Process
```
[Epoch 020] train_loss=0.000123 | val_loss=0.000156 | val_R²=0.8654
[Epoch 040] train_loss=0.000098 | val_loss=0.000142 | val_R²=0.8723
...
Early stopping triggered (patience=80), stopped training at epoch 245
```

#### Optuna Optimization (Optional)
```
Trial 1/50 | hidden=256, layers=4, dropout=0.15, lr=1.2e-03
✓ Trial 1 completed: Average validation R² = 0.8456 ± 0.0123

Trial 2/50 | hidden=512, layers=6, dropout=0.20, lr=8.5e-04
✓ Trial 2 completed: Average validation R² = 0.8698 ± 0.0098
...

✓ Optuna optimization completed
Best validation R² = 0.8734
Best hyperparameters:
  - hidden_dim: 512
  - num_layers: 6
  - dropout: 0.15
  - learning_rate: 0.00185
```

#### Cross-Validation Results
```
Subset-based cross-validation (3 folds)

[Fold 1/3] Validation set: subset1
  ✓ Completed [Best epoch=234] - Validation R²=0.8712, Test R²=0.8645

[Fold 2/3] Validation set: subset2
  ✓ Completed [Best epoch=198] - Validation R²=0.8598, Test R²=0.8521

[Fold 3/3] Validation set: subset3
  ✓ Completed [Best epoch=256] - Validation R²=0.8834, Test R²=0.8756

✓ Best Fold: Fold 3 (Test R²=0.8756)
```

### Output Files
```
SAVE/
├── pinn_peak_strain.pt          # Model weights
├── training_summary.json        # Training configuration and metrics
├── scalers.pkl                  # Data preprocessors
├── model_architecture.json      # Model architecture
└── training.log                 # Complete training log
```

### Usage
```bash
# Complete training workflow (including hyperparameter optimization)
python peak_strain_model_minmax.py --tune-trials 50 --final-epochs 500

# Skip hyperparameter optimization, use default configuration
python peak_strain_model_minmax.py --skip-tuning --final-epochs 300

# Custom save directory
python peak_strain_model_minmax.py --save-dir my_model --final-epochs 400
```

---

## 2. Inference Script (`peak_strain_inference.py`)

### Functionality
- ✅ Load trained model
- ✅ Predict on dataset
- ✅ Calculate Xiao formula predictions
- ✅ Detailed performance comparison analysis
- ✅ Export prediction results

### Output Metrics

#### Model Loading Information
```
[1/5] Loading trained model...
✓ Loaded training configuration
  - Best validation R² = 0.8756
  - Best epoch = 256

✓ Loaded data preprocessors
  - Number of features = 16
  - Feature columns: water, cement, w/c, CS, sand, CA, r, ...

✓ Loaded model weights
  - Network structure: 6 layers × 512 dimensions (dropout=0.15)
```

#### Prediction Process
```
[2/5] Loading dataset...
✓ Loaded dataset: dataset_final.xlsx (shape: (100, 25))

✓ Data preprocessing completed
  - Number of samples = 100
  - Feature dimensions = 16

[3/5] PINN model prediction...
✓ PINN prediction completed (R² = 0.8756)

[4/5] Xiao formula prediction...
✓ Xiao formula prediction completed (R² = 0.4215)
```

#### Performance Comparison (Core Output)
```
====================================================================================================
PINN vs Xiao Formula - Performance Comparison
====================================================================================================
Model          | R²       | EVS      | MAE        | MSE        | RMSE       | MAPE(%)
----------------------------------------------------------------------------------------------------
PINN         | 0.8756   | 0.8798   | 1.642e-04  | 4.523e-08  | 2.127e-04  | 8.45
Xiao Formula | 0.4215   | 0.4198   | 4.987e-04  | 5.234e-07  | 7.235e-04  | 21.34
----------------------------------------------------------------------------------------------------
PINN Improvement: R² increase=107.8%, MAE reduction=67.1%, RMSE reduction=70.6%
====================================================================================================
```

### Output Files
```
SAVE/  (or custom output directory)
├── pinn_vs_xiao_comparison.xlsx        # Sample-by-sample comparison (predictions, true values, errors)
├── dataset_final_with_predictions.xlsx  # Original data + prediction columns
└── inference_metrics.json               # Detailed performance metrics JSON
```

#### `pinn_vs_xiao_comparison.xlsx` Column Description
| Column Name | Description |
|------|------|
| Sample ID | Sample identifier |
| True Strain | Experimentally measured peak strain |
| PINN Prediction | PINN model predicted value |
| Xiao Prediction | Xiao formula calculated value |
| PINN Error | PINN prediction - True strain |
| Xiao Error | Xiao prediction - True strain |
| PINN Absolute Error | \|PINN Error\| |
| Xiao Absolute Error | \|Xiao Error\| |
| PINN Relative Error (%) | (PINN Absolute Error / True strain) × 100 |
| Xiao Relative Error (%) | (Xiao Absolute Error / True strain) × 100 |

### Usage
```bash
# Basic usage
python peak_strain_inference.py --model-dir SAVE --dataset ../dataset/dataset_final.xlsx

# Specify output directory
python peak_strain_inference.py --model-dir SAVE --dataset data.xlsx --output-dir results

# Use GPU for inference
python peak_strain_inference.py --model-dir SAVE --dataset data.xlsx --device cuda
```

---

## Complete Workflow

### Step 1: Train Model
```bash
cd peak_strain
python peak_strain_model_minmax.py --tune-trials 50 --final-epochs 500
```

**Expected Output**:
- Model files generated in SAVE directory
- Console prints training process and cross-validation results
- Final prompt indicating training completion

### Step 2: Inference Evaluation
```bash
python peak_strain_inference.py \
    --model-dir SAVE \
    --dataset ../dataset/dataset_final.xlsx
```

**Expected Output**:
- Console prints performance comparison table
- Prediction result Excel files generated in SAVE directory
- Performance metrics JSON file generated

---

## Metric Descriptions

| Metric | Full Name | Description | Ideal Value |
|------|---------|------|-------|
| **R²** | Coefficient of Determination | Coefficient of determination, measures model goodness of fit | Closer to 1 is better |
| **EVS** | Explained Variance Score | Explained variance score | Closer to 1 is better |
| **MAE** | Mean Absolute Error | Mean absolute error | Smaller is better |
| **MSE** | Mean Squared Error | Mean squared error | Smaller is better |
| **RMSE** | Root Mean Squared Error | Root mean squared error | Smaller is better |
| **MedAE** | Median Absolute Error | Median absolute error | Smaller is better |
| **MAPE** | Mean Absolute Percentage Error | Mean absolute percentage error | Smaller is better |
| **Max Error** | Maximum Residual Error | Maximum residual error | Smaller is better |

---

## Notes

### Training Script
1. ⚠️ Hyperparameter optimization takes a long time (50 trials × 3 folds ≈ 1-2 hours)
2. ⚠️ Recommend using `--skip-tuning` for quick testing on first run
3. ⚠️ Ensure dataset contains `DataSlice` column (for subset division)
4. ⚠️ Ensure dataset contains `r` column (aggregate replacement rate, for physics loss)

### Inference Script
1. ⚠️ Must run training script first to generate model files
2. ⚠️ Inference dataset must contain all feature columns used during training
3. ⚠️ Data format must be consistent with training data (column names, data types)
4. ⚠️ `r` column must exist (for calculating Xiao formula)

---

## Frequently Asked Questions

### Q1: Training script reports error "Column does not exist"
**A**: Check if dataset contains required columns:
- Feature columns (water, cement, fc, r, etc.)
- Target column (peak_strain)
- Division column (DataSlice)

### Q2: Inference script reports error "Model file not found"
**A**: Ensure SAVE directory contains the following files after training:
- pinn_peak_strain.pt
- training_summary.json
- scalers.pkl
- model_architecture.json

### Q3: Xiao formula R² is negative
**A**: This is normal, indicating that the Xiao formula performs poorly on this dataset.
     The advantage of PINN models lies in data-driven learning, surpassing simple empirical formulas.

### Q4: How to improve model performance?
**A**: You can try:
1. Increase hyperparameter search iterations (--tune-trials 100)
2. Adjust loss function weights (--lambda-data, --lambda-physics)
3. Increase training epochs (--final-epochs 800)
4. Check data quality and feature engineering

---

## Version Information

- **Training Script Version**: v2.0 (2024-11-20)
- **Inference Script Version**: v2.0 (2024-11-20)
- **Major Improvements**:
  - Separated training and inference logic
  - Optimized output metric display
  - Improved documentation
  - Unified Xiao formula implementation

---

## License

This project is for academic research use only.
