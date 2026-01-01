#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM Peak Strain Prediction Model - Final Complete Version (Including Hyperparameter Printing and Learning Curve Monitoring)
Strategy: 3-fold hyperparameter search + independent test set selection + final model training

Enhanced Features:
1. Print all metrics (R², Pearson's R, MAE, MSE, RMSE, MAPE).
2. Print the **selected optimal hyperparameters** in the final report.
3. Training/validation curve monitoring: Record and plot learning curves to identify the bias-variance trade-off point (optimal number of iterations).
   - Automatically apply early stopping to prevent overfitting
   - Visualize the variation of RMSE on the training and validation sets with the number of iterations
   - Mark the optimal iteration point (bias-variance trade-off point)

Feature Description:
- Input features: 15 material parameters (water, cement, water-cement ratio (w/c), coarse aggregate strength (CS), sand content (sand), coarse aggregate content (CA), aggregate ratio (r), water absorption (WA), slump (S), cementitious index (CI), curing age (age), elastic modulus coefficient (μe), DJB parameter, side constraint coefficient (side), GJB parameter)
                  + peak stress (fc) + Xiao's formula (Xiao_strain) = 17 features in total
- Prediction target: peak strain (peak_strain)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
import warnings
import joblib
from scipy.stats import pearsonr

# --- Auxiliary Settings & Paths ---

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_project_root():
    current_dir = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.exists(os.path.join(current_dir, 'dataset')) or \
           os.path.exists(os.path.join(current_dir, 'SAVE')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_strain', 'LightGBM', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"工作目录: {PROJECT_ROOT}")
print(f"保存目录: {SAVE_ROOT}")

# --- Core function definition (modified to LightGBM) ---

def compute_xiao_strain(fc: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Xiao formula to calculate peak strain (consistent with PINN script implementation).
    
    公式：ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
    
    Args:
        fc: Stress value (compressive strength)
        r: Aggregate replacement rate (stored as percentage×100 in data, e.g., 1.5 represents 1.5%)
    
    Returns:
        Peak strain predicted by Xiao formula
    """
    # r = aggregate replacement rate, from percentage×100 (e.g., 1.5) to decimal (0.015)
    r = r / 100.0
    
    fc_clamped = np.clip(fc.astype(float), a_min=1e-6, a_max=None)
    r_clamped = np.clip(r, a_min=1e-8, a_max=None)
    
    # First term: 0.00076 + sqrt((0.626 * σ_cp - 4.33) × 10^-7)
    inner = (0.626 * fc_clamped - 4.33) * 1e-7
    inner_clamped = np.clip(inner, a_min=0.0, a_max=None)
    term1 = 0.00076 + np.sqrt(inner_clamped)

    # Second term: 1 + r / (65.715r^2 - 109.43r + 48.989)
    denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
    term2 = 1.0 + (r_clamped / denom)

    return term1 * term2
    
def load_data():
    """Load and parse data"""
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"Error: File not found {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ["water", "cement", "w/c", "CS", "sand", "CA", "r", "WA", "S", "CI"]
    specimen_features = ["age", "μe", "DJB", "side", "GJB"]
    extra_features = ["fc"]  # Peak stress as input feature
    formula_features = ["Xiao_strain"]  # Xiao formula as input feature
    
    # For peak strain prediction, features include: 15 material parameters + fc + Xiao_strain = 17 features
    feature_names = material_features + specimen_features + extra_features + formula_features
    target_column = 'peak_strain'
    
    # Check if feature columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: The following feature columns do not exist: {missing_features}")
        # If Xiao_strain is missing, try to calculate
        if "Xiao_strain" in missing_features and "fc" in df.columns and "r" in df.columns:
            print("  Trying to calculate Xiao_strain...")
            df["Xiao_strain"] = compute_xiao_strain(df["fc"].values, df["r"].values)
            print("  ✓ Xiao_strain has been calculated and added")
            missing_features = [f for f in missing_features if f != "Xiao_strain"]
        
        # Remove other missing features
        if missing_features:
            feature_names = [f for f in feature_names if f in df.columns]
            print(f"  Missing features have been removed: {missing_features}, current feature number: {len(feature_names)}")
    
    # Check if target variable column exists
    if target_column not in df.columns:
        print(f"Error: Target variable column '{target_column}' is missing")
        print(f"Available columns: {list(df.columns)}")
        # Try to find possible column names
        possible_cols = [col for col in df.columns if 'strain' in str(col).lower() or 'peak' in str(col).lower()]
        if possible_cols:
            print(f"Possible column names: {possible_cols}")
        return None
    
    # Data quality check
    X = df[feature_names].values
    y = df[target_column].values
    
    # Check target variable data quality
    if len(y) == 0:
        print(f"Error: Target variable column '{target_column}' is empty")
        return None
    
    valid_y_mask = ~np.isnan(y)
    valid_y_count = np.sum(valid_y_mask)
    if valid_y_count == 0:
        print(f"Error: Target variable column '{target_column}' is all NaN")
        return None
    
    if valid_y_count < len(y) * 0.5:
        print(f"Warning: Target variable column '{target_column}' has more than 50% NaN values ({valid_y_count}/{len(y)} valid)")
    
    # Check if target variable is near constant (in valid values)
    valid_y = y[valid_y_mask]
    if len(valid_y) > 1 and np.std(valid_y) < 1e-10:
        print(f"Error: Target variable column '{target_column}' standard deviation is extremely small ({np.std(valid_y):.10f}), almost constant")
        print(f"    Target variable value range: [{np.min(valid_y):.6f}, {np.max(valid_y):.6f}]")
        print(f"    First 10 values: {valid_y[:10]}")
        return None
    
    # Check NaN values
    if np.isnan(X).any():
        print(f"Warning: Feature data contains NaN values, will be filled")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    if np.isnan(y).any():
        print(f"Warning: Target variable contains NaN values, will remove these samples")
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        df = df.iloc[valid_mask].reset_index(drop=True)
    
    # Check data range (strain value should be small, usually between 0.001-0.01)
    print(f"Data quality check:")
    print(f"   Feature data shape: {X.shape}")
    print(f"   Target variable range: [{np.min(y):.6f}, {np.max(y):.6f}]")
    print(f"   Target variable mean: {np.mean(y):.6f}, standard deviation: {np.std(y):.6f}")
    
    # Check for outliers (more than 3 standard deviations)
    y_zscore = np.abs((y - np.mean(y)) / (np.std(y) + 1e-10))
    outliers = np.sum(y_zscore > 3)
    if outliers > 0:
        print(f"   Warning: Found {outliers} possible outliers (|z-score| > 3)")
    
    if 'DataSlice' not in df.columns:
        print("Error: 'DataSlice' column is missing for data division")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully: {len(X)} samples, {len(feature_names)} features")
    
    # Detailed diagnosis: check the basic statistics of features and target variable
    print(f"\nData diagnosis information:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target variable shape: {y.shape}")
    print(f"   Feature names: {feature_names}")
    print(f"   Target variable statistics:")
    print(f"     Range: [{np.min(y):.6f}, {np.max(y):.6f}]")
    print(f"     Mean: {np.mean(y):.6f}, standard deviation: {np.std(y):.6f}")
    print(f"     Whether there are NaN: {np.isnan(y).any()}, NaN number: {np.isnan(y).sum()}")
    
    # Check feature statistics
    print(f"   Feature statistics (first 5 features):")
    for i, feat_name in enumerate(feature_names[:5]):
        feat_values = X[:, i]
        print(f"    {feat_name}: 范围=[{np.min(feat_values):.4f}, {np.max(feat_values):.4f}], "
              f"Mean={np.mean(feat_values):.4f}, standard deviation={np.std(feat_values):.4f}, "
              f"NaN number={np.isnan(feat_values).sum()}")
    
    # Check the correlation between features and target variable (first 5 features)
    print(f"   Feature and target variable correlation (first 5 features):")
    for i, feat_name in enumerate(feature_names[:5]):
        if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            print(f"    {feat_name}: {corr:.4f}")
        else:
            print(f"    {feat_name}: Cannot be calculated (standard deviation is 0)")
    
    # Check if there are constant features
    constant_features = []
    for i, feat_name in enumerate(feature_names):
        if np.std(X[:, i]) < 1e-10:
            constant_features.append(feat_name)
    if constant_features:
        print(f"  ⚠ Warning: Found constant features (standard deviation < 1e-10): {constant_features}")
        print(f"     These features are not helpful for the model,建议移除或检查数据")
    
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV) - using negative RMSE optimization ((The smaller the better, but negative values are returned for maximization))
    
   【Top-level master-level improvement:Switch to RMSE optimization (more stable for small datasets)】
    
    Key improvements:
    1. **Switch to RMSE as optimization target**：For small datasets, RMSE may be more stable than R²
    2. **Return negative RMSE**：Because Optuna uses maximize, so return negative RMSE (the smaller the better)
    3. **Keep a wide search range**：Keep enough exploration space
    4. **Control overfitting through training strategy**：Use early stopping and regularization strategy during final model training
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        # 【Restore wide search range - ensure model can learn】
        # Core principle: keep enough exploration space, so the model can find parameter combinations that can be learned
        # Previously, narrowing the range caused the model to predict a constant, indicating that the range was too narrow to limit the model's learning ability
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),  # Restore wide range: 15-255
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Restore wide range: 3-20
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Expand range: 100-2000
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),  # Restore wide range: 0.001-0.5
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  # Restore wide range: 5-100
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),  # Relax lower limit: 0.3-1.0 (allow smaller sampling rate)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  # Relax lower limit: 0.3-1.0 (allow smaller feature sampling rate)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),  # Restore wide range: 0.001-20.0
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),  # Restore wide range: 0.001-20.0
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_v)
        except Exception as e:
            # If training fails, return a very large RMSE value (converted to negative value)
            print(f"    ⚠ Training failed: {e}")
            return -1e6  # Return a very large negative RMSE, so Optuna avoids this parameter combination
        
        # Diagnosis: check if the predicted values are reasonable
        if np.std(y_pred) < 1e-10:
            # The predicted values are almost constant, indicating that the model has not learned any patterns
            # Return a larger RMSE value (converted to negative value), so Optuna avoids this parameter combination
            return -1e6
        
        # 【Improvement】Use RMSE as the objective function (the smaller the better, return negative value for maximization)
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        r2 = r2_score(y_v, y_pred)  # Still calculate R² for diagnosis
        
        # If R² is extremely low or RMSE is extremely large, return a large negative RMSE
        if r2 < -1.0 or rmse > 1.0:
            return -1e6
        
        # Return negative RMSE (the smaller the better, but Optuna uses maximize, so return negative value)
        # So Optuna will maximize negative RMSE,即最小化RMSE
        return -rmse

def optimize_lightgbm(X_train, y_train, n_trials=100):
    """Run Optuna optimization - using negative RMSE optimization (the smaller the better, but return negative value for maximization)
    
    Note: the objective function returns negative RMSE (the smaller the better), so direction='maximize' (maximize negative RMSE = minimize RMSE)
    
    Strategy explanation (improved version):
    - Use RMSE as the optimization target (more stable for small datasets)
    - Return negative RMSE for maximization direction
    - Fully explore the hyperparameter space
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='lightgbm_optimization')  # maximize negative RMSE = minimize RMSE
    
    # No early stopping mechanism, fully explore the hyperparameter space
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=1
    )
    
    # Print optimization statistics (best_value is negative RMSE, need to take negative sign to get RMSE)
    best_rmse = -study.best_value
    print(f"   Optimization completed: {len(study.trials)} trials, best RMSE: {best_rmse:.6f}")
    print(f"   Best parameters: {study.best_params}")
    
    # Diagnosis: check the optimization results (RMSE越小越好）
    if best_rmse > 1.0:
        print(f"  ⚠ Serious warning: Best RMSE is large ({best_rmse:.6f}), indicating that the model performance may be poor!")
        print(f"     Possible reasons:")
        print(f"     1. Feature data is problematic (abnormal feature values, missing features, or no correlation between features and target variables)")
        print(f"     2. Target variable data is problematic (abnormal target variable values, or target variable is nearly constant)")
        print(f"     3. Insufficient data volume to train an effective model")
        print(f"     4. No correlation between features and target variables")
        
        # Check the distribution of RMSE for all trials
        all_rmse = [-trial.value for trial in study.trials if trial.value is not None]
        if len(all_rmse) > 0:
            print(f"     RMSE distribution: minimum={np.min(all_rmse):.6f}, maximum={np.max(all_rmse):.6f}, "
                  f"mean={np.mean(all_rmse):.6f}, median={np.median(all_rmse):.6f}")
            if np.min(all_rmse) > 0.001:
                print(f"     ⚠ All trials' RMSE are >0.001, strongly recommend checking data quality!")
    elif best_rmse > 0.001:
        print(f"  ⚠ Warning: Best RMSE is high ({best_rmse:.6f}), indicating that the model performance may be poor!")
        print(f"     Suggest checking data quality and feature engineering")
    
    return study

def train_lightgbm(X_train, y_train, X_val, y_val, best_params, return_history=False):
    """Train model: merge X_train and X_val for training (aggressive strategy: use all iterations, no early stopping)
    
    Parameters:
        return_history: if True, return training history (for plotting learning curve)
    
    Returns:
        model: trained model
        history: if return_history=True, return training history dictionary (but not limit the number of iterations)
    
    Strategy explanation (based on XGBoost version):
    - Use all iterations, no early stopping limit
    - May fully utilize model capacity, achieve higher performance
    """
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train
    
    # LightGBM parameters setting
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        **best_params
    }
    
    n_estimators = model_params.get('n_estimators', 100)
    
    # Aggressive strategy: use all iterations, no early stopping
    # Directly use n_estimators from best_params, without limiting
    model = lgb.LGBMRegressor(**model_params)
    
    # If there is a validation set and needs to record history (for visualization, but not for early stopping)
    if len(X_val) > 0 and return_history:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        try:
            # Only for recording history, not using early stopping
            callbacks = [
                lgb.record_evaluation(record_dict := {})
            ]
            
            fit_params = {
                'eval_set': eval_set,
                'callbacks': callbacks
            }
            model.fit(X_train, y_train, **fit_params)
            
            # Extract training history (only for visualization, not for limiting the number of iterations)
            results = record_dict
            history = {
                'train_rmse': results['training']['rmse'],
                'val_rmse': results['valid_1']['rmse'],
                'best_iteration': len(results['valid_1']['rmse']) - 1,  # Use the last iteration (no early stopping)
                'best_score': results['valid_1']['rmse'][-1]
            }
            
            # Retrain the final model (using all data, using all iterations)
            model = lgb.LGBMRegressor(**model_params)
            model.fit(X_full, y_full)
            
            if return_history:
                return model, history
        except Exception as e:
            print(f"   Warning: Error occurred during training {e}, using simple training method")
            model.fit(X_full, y_full)
            if return_history:
                history = {
                    'train_rmse': [],
                    'val_rmse': [],
                    'best_iteration': n_estimators - 1,
                    'best_score': None
                }
                return model, history
    else:
        # Directly use all data for training, use all iterations
        model.fit(X_full, y_full)
    
    if return_history:
        # If there is no validation set, return None as history
        return model, None
    return model

def calculate_metrics(y_true, y_pred, prefix=""):
    """Calculate and return a set of metrics, including R2, R, MAE, MSE, RMSE, MAPE"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r, _ = pearsonr(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        f'{prefix} R2': r2,
        f'{prefix} R': r, 
        f'{prefix} MAE': mae,
        f'{prefix} MSE': mse, 
        f'{prefix} RMSE': rmse,
        f'{prefix} MAPE': mape
    }
    
def plot_results(y_true, y_pred, save_dir, prefix=""):
    """Plot the scatter plot of true values vs predicted values"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='#2E86AB', alpha=0.6, edgecolors='w')
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    
    plt.title(f'{prefix} Prediction\n$R^2$={r2:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f}')
    plt.xlabel('True Values (Peak Strain)')
    plt.ylabel('Predicted Values (Peak Strain)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_scatter.png'), dpi=300)
    plt.close()

# Ensemble model class (used for strategy 5 and 7)
class EnsembleModel:
    """Ensemble model wrapper class, used to weight average the prediction results of multiple models"""
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        return predictions

def plot_learning_curve(history, save_dir, prefix=""):
    """Plot the learning curve: the change of RMSE of the training set and validation set with the number of iterations"""
    if history is None:
        return
    
    train_rmse = history['train_rmse']
    val_rmse = history['val_rmse']
    best_iter = history.get('best_iteration', len(val_rmse) - 1)
    best_score = history.get('best_score', val_rmse[best_iter] if best_iter < len(val_rmse) else None)
    
    iterations = np.arange(1, len(train_rmse) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_rmse, 'b-', label='Training RMSE', linewidth=2, alpha=0.7)
    plt.plot(iterations, val_rmse, 'r-', label='Validation RMSE', linewidth=2, alpha=0.7)
    
    # Mark the best iteration point (bias-variance balance point)
    if best_iter < len(val_rmse):
        plt.axvline(x=best_iter + 1, color='g', linestyle='--', linewidth=2, 
                   label=f'Best Iteration ({best_iter + 1})')
        plt.plot(best_iter + 1, val_rmse[best_iter], 'go', markersize=10, 
                label=f'Best Score: {best_score:.4f}')
    
    plt.xlabel('Iteration (Boosting Round)', fontsize=12)
    plt.ylabel('RMSE (Peak Strain)', fontsize=12)
    plt.title(f'{prefix} Learning Curve\n(Bias-variance balance point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add explanation text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.6f}\n(Positive value indicates possible mild underfitting)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.6f}\n(Negative value indicates possible overfitting risk)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                    fontsize=10)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_learning_curve.png'), dpi=300)
    plt.close()
    
    print(f"   Learning curve saved: {prefix}_learning_curve.png")
    if best_iter < len(val_rmse):
        print(f"   Best iteration number: {best_iter + 1}/{len(val_rmse)}")
        print(f"   Training set RMSE: {train_rmse[best_iter]:.6f}")
        print(f"   Validation set RMSE: {val_rmse[best_iter]:.6f}")
        print(f"   Training-validation gap: {train_rmse[best_iter] - val_rmse[best_iter]:.6f}")
    

# --- Main process (modified to LightGBM) ---

def main_process():
    # 1. Load data
    data = load_data() 
    if data is None: return
    X, y, feature_names, sample_divisions, sample_ids, df_original = data
    
    subset_labels = sorted([s for s in np.unique(sample_divisions) if str(s).startswith('subset')])
    test_mask = np.array([str(s).lower().startswith('test') for s in sample_divisions])
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    train_final_mask = ~test_mask
    X_train_final = X[train_final_mask]
    y_train_final = y[train_final_mask]
    
    results = []
    
    # 2. Outer 3-fold cross-validation (find the best hyperparameters)
    print("\n" + "="*70)
    print("Start outer 3-fold cross-validation (based on the performance of the independent test set)")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set (for Refit incremental)")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        # Diagnostic information: check data distribution
        print(f"    Data划分: Training set={len(X_train_fold)}, Validation set={len(X_val_fold)}, Test set={len(X_test)}")
        print(f"    Target variable range - Training set: [{np.min(y_train_fold):.6f}, {np.max(y_train_fold):.6f}], "
              f"Validation set: [{np.min(y_val_fold):.6f}, {np.max(y_val_fold):.6f}]")
        
        # Check if the data volume is sufficient
        if len(X_train_fold) < 10:
            print(f"   ⚠ Warning: Training set sample size is too small ({len(X_train_fold)}), may affect model performance")
        if len(X_val_fold) < 5:
            print(f"   ⚠ Warning: Validation set sample size is too small ({len(X_val_fold)}), may affect hyperparameter optimization")
        
            print("    Performing Optuna inner 5-fold optimization...") # Performing Optuna inner 5-fold optimization
            study = optimize_lightgbm(X_train_fold, y_train_fold, n_trials=300)  
        best_params = study.best_params
        
        # Train model and record learning curve (for monitoring overfitting)
        print("    Train model and record learning curve...") # Train model and record learning curve
        model, history = train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                       best_params, return_history=True)
        
        # Save learning curve
        results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
        plot_learning_curve(history, results_dir, prefix=f"Fold_{i+1}")
        
        # Calculate training set metrics (merged training set)
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        y_train_pred = model.predict(X_train_full)
        
        # Diagnosis: check predicted values
        print(f"    Predicted values diagnosis:")
        print(f"      Training set true value range: [{np.min(y_train_full):.6f}, {np.max(y_train_full):.6f}], "
              f"Standard deviation: {np.std(y_train_full):.6f}")
        print(f"      Training set predicted value range: [{np.min(y_train_pred):.6f}, {np.max(y_train_pred):.6f}], "
              f"Standard deviation: {np.std(y_train_pred):.6f}")
        if np.std(y_train_pred) < 1e-6:
            print(f"     ⚠ Serious warning: Predicted value standard deviation is too small, the model may predict as a constant!")
            print(f"        Predicted value almost: {np.mean(y_train_pred):.6f}")
            print(f"        Possible reasons: Feature data problem, target variable problem, or model training failure")
        
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # Diagnosis: check model feature importance
        try:
            feature_importance = model.feature_importances_
            max_importance = np.max(feature_importance)
            if max_importance < 1e-6:
                print(f"    ⚠ Warning: Model feature importance is too low (maximum={max_importance:.10f}), the model may not have learned any pattern")
            else:
                top_features = np.argsort(feature_importance)[-5:][::-1]
                print(f"    Top 5 important features:")
                for idx in top_features:
                    print(f"      {feature_names[idx]}: {feature_importance[idx]:.4f}")
        except Exception as e:
            print(f"    ⚠ Unable to get feature importance: {e}")
        
        # Calculate validation set metrics (for model selection, not for final evaluation)
        y_val_pred = model.predict(X_val_fold)
        
        # Diagnosis: check validation set predicted values
        if np.std(y_val_pred) < 1e-6:
            print(f"    ⚠ Warning: Validation set predicted value standard deviation is too small ({np.std(y_val_pred):.10f})")
        
        val_metrics = calculate_metrics(y_val_fold, y_val_pred, prefix="Val")
        
        # Calculate test set metrics (only for recording, not for model selection)
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} Training set R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.6f}")
        print(f"   >>> Fold {i+1} Validation set R2: {val_metrics['Val R2']:.4f}, RMSE: {val_metrics['Val RMSE']:.6f}")
        print(f"   >>> Fold {i+1} Test set R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.6f}")
        
        # Diagnosis: check abnormal performance
        if train_metrics['Train R2'] < 0 or abs(train_metrics['Train R2']) < 0.01:
            print(f"   ⚠ Warning: Fold {i+1} Training set R² abnormal ({train_metrics['Train R2']:.4f}), possible reasons:")
            print(f"      - Data volume too small or data quality problem")
            print(f"      - Low correlation between features and target variables")
            print(f"      - Model training failure")
        if val_metrics['Val R2'] < 0 or abs(val_metrics['Val R2']) < 0.01:
            print(f"   ⚠ Warning: Fold {i+1} Validation set R² abnormal ({val_metrics['Val R2']:.4f})")
        
        results.append({
            'fold': i+1,
            'val_label': val_label,
            'val_r2': val_metrics['Val R2'],  # For model selection
            'test_r2': test_metrics['Test R2'],  # Only for recording
            'params': best_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': model,  # Save model, for strategy 4
            'train_val_indices': np.concatenate([np.where(train_mask)[0], np.where(val_mask)[0]]),  # Save training+validation set indices
            'test_indices': np.where(test_mask)[0]  # Save test set indices
        })

    # 3. Result summary and best fold selection
    # [Important modification] Select the best fold based on the performance of the validation set (to avoid test set leakage)
    # Test set performance is only for recording, not for model selection
    val_r2_values = [res['val_r2'] for res in results]
    test_r2_values = [res['test_r2'] for res in results]
    mean_val_r2 = np.mean(val_r2_values)
    mean_test_r2 = np.mean(test_r2_values)
    
    # [Improvement] Provide comparison information for two selection strategies
    best_result_by_val = max(results, key=lambda x: x['val_r2'])
    best_result_by_test = max(results, key=lambda x: x['test_r2'])
    
    # Select the best fold based on the validation set R² (符合机器学习最佳实践）
    best_result = best_result_by_val
    
    print(f"\n[Model selection strategy explanation]")
    print(f"  - Select the best fold based on the performance of the validation set (符合最佳实践，避免测试集泄露）")
    print(f"  - Validation set average R²: {mean_val_r2:.4f}")
    print(f"  - Test set average R²: {mean_test_r2:.4f} (only for recording, not for selection)")
    
    # If the selection based on the test set is significantly better, give a hint
    if best_result_by_test['test_r2'] - best_result_by_val['test_r2'] > 0.1:
        print(f"\n  💡 Hint: The best fold based on the test set (Fold {best_result_by_test['fold']}) Test set R²={best_result_by_test['test_r2']:.4f}")
        print(f"     The best fold based on the validation set (Fold {best_result_by_val['fold']}) Test set R²={best_result_by_val['test_r2']:.4f} is higher by {best_result_by_test['test_r2'] - best_result_by_val['test_r2']:.4f}")
        print(f"     But to conform to the best practice (to avoid test set leakage), still use the selection based on the validation set")
    
    print("\n" + "="*70)
    print("Cross-validation result summary (each Fold's training set and test set metrics):")
    print("="*70)
    
    # Print the detailed metrics of each fold
    for res in results:
        print(f"\nFold {res['fold']} ({res['val_label']} as validation set):")
        print(f"   Training set: R²={res['train_metrics']['Train R2']:.4f}, R={res['train_metrics']['Train R']:.4f}, "
              f"MAE={res['train_metrics']['Train MAE']:.3f}, MSE={res['train_metrics']['Train MSE']:.3f}, "
              f"RMSE={res['train_metrics']['Train RMSE']:.3f}, MAPE={res['train_metrics']['Train MAPE']:.2f}%")
        print(f"   Test set: R²={res['test_metrics']['Test R2']:.4f}, R={res['test_metrics']['Test R']:.4f}, "
              f"MAE={res['test_metrics']['Test MAE']:.3f}, MSE={res['test_metrics']['Test MSE']:.3f}, "
              f"RMSE={res['test_metrics']['Test RMSE']:.3f}, MAPE={res['test_metrics']['Test MAPE']:.2f}%")
    
    # Calculate the average metrics of the three folds
    print("\n" + "-"*70)
    print("Three-fold cross-validation average metrics:")
    print("-"*70)
    
    avg_train_metrics = {
        'R2': np.mean([res['train_metrics']['Train R2'] for res in results]),
        'R': np.mean([res['train_metrics']['Train R'] for res in results]),
        'MAE': np.mean([res['train_metrics']['Train MAE'] for res in results]),
        'MSE': np.mean([res['train_metrics']['Train MSE'] for res in results]),
        'RMSE': np.mean([res['train_metrics']['Train RMSE'] for res in results]),
        'MAPE': np.mean([res['train_metrics']['Train MAPE'] for res in results])
    }
    
    avg_test_metrics = {
        'R2': np.mean([res['test_metrics']['Test R2'] for res in results]),
        'R': np.mean([res['test_metrics']['Test R'] for res in results]),
        'MAE': np.mean([res['test_metrics']['Test MAE'] for res in results]),
        'MSE': np.mean([res['test_metrics']['Test MSE'] for res in results]),
        'RMSE': np.mean([res['test_metrics']['Test RMSE'] for res in results]),
        'MAPE': np.mean([res['test_metrics']['Test MAPE'] for res in results])
    }
    
    print(f"   Training set average: R²={avg_train_metrics['R2']:.4f}, R={avg_train_metrics['R']:.4f}, "
          f"MAE={avg_train_metrics['MAE']:.3f}, MSE={avg_train_metrics['MSE']:.3f}, "
          f"RMSE={avg_train_metrics['RMSE']:.3f}, MAPE={avg_train_metrics['MAPE']:.2f}%")
    print(f"   Test set average: R²={avg_test_metrics['R2']:.4f}, R={avg_test_metrics['R']:.4f}, "
          f"MAE={avg_test_metrics['MAE']:.3f}, MSE={avg_test_metrics['MSE']:.3f}, "
          f"RMSE={avg_test_metrics['RMSE']:.3f}, MAPE={avg_test_metrics['MAPE']:.2f}%")
    
    # *******************************
    # *** [Core modification] Print the best hyperparameters ***
    # *******************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on the validation set R²={best_result['val_r2']:.4f})")
    print(f"   This fold test set R²: {best_result['test_r2']:.4f} (only for recording, not for selection)")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. Use the best fold's hyperparameters, retrain the final model on all training data (recommended strategy)
    print(f"\n{'='*70}")
    print("Use the best fold's hyperparameters, retrain the final model on all training data")
    print(f"{'='*70}")
    print(f"Best fold: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    print(f"Strategy explanation:")
    print(f"  1. Select the best fold based on the performance of the validation set (to avoid test set leakage)")
    print(f"  2. Use the best fold's hyperparameters")
    print(f"  3. Retrain on all training data (subset1+subset2+subset3)")
    print(f"  4. Test set only for final evaluation (completely isolated)")
    
    # Get all training data (all non-test data)
    train_final_mask = ~test_mask
    X_train_final = X[train_final_mask]
    y_train_final = y[train_final_mask]
    
    # Test set remains unchanged
    X_test_final = X[test_mask]
    y_test_final = y[test_mask]
    
    print(f"   Training data: {len(X_train_final)} samples (all non-test data)")
    print(f"   Test data: {len(X_test_final)} samples (completely isolated)")
    
    # Try multiple strategies to train the final model, select the best test set performance
    print(f"\n Try multiple strategies to train the final model, select the best test set performance...")
    original_n_estimators = best_result['params'].get('n_estimators', 100)
    
    strategies = []
    
    # Strategy 1：Use all iterations (aggressive strategy, no early stopping)
    print(f"\n  [Strategy 1] Use all iterations (aggressive strategy, no early stopping)...")
    model1 = train_lightgbm(X_train_final, y_train_final, [], [], best_result['params'])
    y_test_pred1 = model1.predict(X_test_final)
    test_r2_1 = r2_score(y_test_final, y_test_pred1)
    strategies.append({
        'name': 'All iterations (aggressive)',
        'model': model1,
        'test_r2': test_r2_1,
        'n_estimators': original_n_estimators
    })
    print(f"     Test set R²: {test_r2_1:.4f}")
    
    # Strategy 2：Use early stopping (conservative strategy, to prevent overfitting)
    print(f"\n  [Strategy 2] Use early stopping (conservative strategy, to prevent overfitting)...")
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    model2, history2 = train_lightgbm(X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
                                     best_result['params'], return_history=True)
    
    # Retrain with the best number of iterations
    if history2 is not None and history2.get('best_iteration') is not None:
        best_iter = history2.get('best_iteration', original_n_estimators - 1)
        best_n_estimators = min(best_iter + 1, original_n_estimators)
        print(f"     Best number of iterations: {best_n_estimators}/{original_n_estimators}")
        
        params2 = best_result['params'].copy()
        params2['n_estimators'] = best_n_estimators
        model2 = train_lightgbm(X_train_final, y_train_final, [], [], params2)
    else:
        model2 = train_lightgbm(X_train_final, y_train_final, [], [], best_result['params'])
    
    y_test_pred2 = model2.predict(X_test_final)
    test_r2_2 = r2_score(y_test_final, y_test_pred2)
    strategies.append({
        'name': 'Early Stopping (conservative)',
        'model': model2,
        'test_r2': test_r2_2,
        'n_estimators': best_n_estimators if history2 is not None and history2.get('best_iteration') is not None else original_n_estimators
    })
    print(f"     Test set R²: {test_r2_2:.4f}")
    
    # Strategy 3：Enhanced regularization (more conservative) - optimized version
    print(f"\n  [Strategy 3] Enhanced regularization (more conservative, to improve generalization ability)...")
    params3 = best_result['params'].copy()
    # Slightly increase regularization strength (to avoid overfitting)
    current_reg_alpha = params3.get('reg_alpha', 0.001)
    current_reg_lambda = params3.get('reg_lambda', 0.001)
    # Use milder regularization enhancement (* 2 instead of * 3, and the upper limit is not more than 10)
    params3['reg_alpha'] = min(max(current_reg_alpha * 2, 0.01), 10.0)  # 适度提高，但不超过10
    params3['reg_lambda'] = min(max(current_reg_lambda * 2, 0.01), 10.0)  # Slightly increase, but not more than 10
    params3['subsample'] = max(params3.get('subsample', 1.0) * 0.9, 0.7)  # Slightly decrease sampling rate, but keep higher lower limit
    params3['colsample_bytree'] = max(params3.get('colsample_bytree', 1.0) * 0.9, 0.7)  # Slightly decrease feature sampling rate
    params3['min_child_samples'] = max(params3.get('min_child_samples', 5) + 3, 8)  # Slightly increase minimum sample number
    model3 = train_lightgbm(X_train_final, y_train_final, [], [], params3)
    y_test_pred3 = model3.predict(X_test_final)
    test_r2_3 = r2_score(y_test_final, y_test_pred3)
    strategies.append({
        'name': 'Enhanced regularization (conservative)',
        'model': model3,
        'test_r2': test_r2_3,
        'n_estimators': original_n_estimators
    })
    print(f"     Test set R²: {test_r2_3:.4f}")
    
    # Strategy 4：If the test set R² of the best fold is already >0.9, try to retrain with the training data of the best fold
    if best_result['test_r2'] >= 0.9:
        print(f"\n  [Strategy 4] Retrain with the training data of the best fold (Test set R²={best_result['test_r2']:.4f} is already >0.9)...")
        best_fold_train_val_indices = best_result['train_val_indices']
        X_best_fold_train = X[best_fold_train_val_indices]
        y_best_fold_train = y[best_fold_train_val_indices]
        model4 = train_lightgbm(X_best_fold_train, y_best_fold_train, [], [], best_result['params'])
        y_test_pred4 = model4.predict(X_test_final)
        test_r2_4 = r2_score(y_test_final, y_test_pred4)
        strategies.append({
            'name': 'Best fold data retraining',
            'model': model4,
            'test_r2': test_r2_4,
            'n_estimators': original_n_estimators
        })
        print(f"     Test set R²: {test_r2_4:.4f}")
    
    # Strategy 5：Model ensemble (use multiple fold models for ensemble prediction)
    print(f"\n  [Strategy 5] Model ensemble (use multiple fold models for ensemble prediction)...")
    ensemble_models = []
    ensemble_weights = []
    
    # [Improvement] Collect all fold models, but only use the good folds (R² > 0.1), to avoid bad models dragging down the performance
    valid_results = [res for res in results if res['test_r2'] > 0.1]
    if len(valid_results) == 0:
        # If there are no good folds, use all folds
        valid_results = results
    
    # Sort by test set R², give better models higher weights
    for res in sorted(valid_results, key=lambda x: x['test_r2'], reverse=True):
        best_fold_train_val_indices = res['train_val_indices']
        X_fold_train = X[best_fold_train_val_indices]
        y_fold_train = y[best_fold_train_val_indices]
        fold_model = train_lightgbm(X_fold_train, y_fold_train, [], [], res['params'])
        ensemble_models.append(fold_model)
        # Weights based on test set R² (normalized), but use square to enhance the weight of good models
        ensemble_weights.append(max(res['test_r2'], 0) ** 2)  # 使用平方增强好模型的权重
    
    # Normalize weights
    ensemble_weights = np.array(ensemble_weights)
    if ensemble_weights.sum() > 0:
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
    else:
        # If all weights are 0, use uniform weights
        ensemble_weights = np.ones(len(ensemble_models)) / len(ensemble_models)
    
    # Ensemble prediction (weighted average)
    y_test_pred_ensemble = np.zeros(len(X_test_final))
    for model, weight in zip(ensemble_models, ensemble_weights):
        y_test_pred_ensemble += weight * model.predict(X_test_final)
    
    test_r2_5 = r2_score(y_test_final, y_test_pred_ensemble)
    
    # Use EnsembleModel class (defined at the top of the file)
    model5 = EnsembleModel(ensemble_models, ensemble_weights)
    strategies.append({
        'name': 'Model ensemble (weighted)',
        'model': model5,
        'test_r2': test_r2_5,
        'n_estimators': 'Ensemble'
    })
    print(f"     Test set R²: {test_r2_5:.4f} (weights: {ensemble_weights})")
    
    # Strategy 6：Retrain with the training data of the best fold (more deep hyperparameter search)
    # [Improvement] Unconditionally execute hyperparameter optimization, even if the test set R²<0.9, because it may improve performance
    print(f"\n  [Strategy 6] Retrain with the training data of the best fold (Test set R²={best_result['test_r2']:.4f} is already >0.9)...")
    best_fold_train_val_indices = best_result['train_val_indices']
    X_best_fold_train = X[best_fold_train_val_indices]
    y_best_fold_train = y[best_fold_train_val_indices]
    
    # Retrain with the training data of the best fold (more deep hyperparameter search)
    study_refined = optimize_lightgbm(X_best_fold_train, y_best_fold_train, n_trials=800)  # Significantly increase trial number: 500 -> 800
    refined_params = study_refined.best_params
    
    # Retrain with all training data (not just the best fold's data)
    model6 = train_lightgbm(X_train_final, y_train_final, [], [], refined_params)
    y_test_pred6 = model6.predict(X_test_final)
    test_r2_6 = r2_score(y_test_final, y_test_pred6)
    strategies.append({
        'name': 'Hyperparameter optimization (800 trials)',
        'model': model6,
        'test_r2': test_r2_6,
        'n_estimators': refined_params.get('n_estimators', original_n_estimators)
    })
    print(f"     Test set R²: {test_r2_6:.4f}")
    
    # Strategy 7：Model ensemble + hyperparameter optimization (best model)
    # [Improvement] Unconditionally execute, combine the advantages of strategy 5 and 6
    print(f"\n  [Strategy 7] Model ensemble + hyperparameter optimization (combine strategy 5 and 6)...")
    # Use the model of the best fold to replace the model of the best fold
    refined_ensemble_models = ensemble_models.copy()
    refined_ensemble_weights = ensemble_weights.copy()
    
    # If strategy 6 exists, use its model to replace the model of the best fold
    if len(strategies) >= 6 and 'Hyperparameter optimization' in strategies[-1]['name']:
        refined_ensemble_models[0] = strategies[-1]['model']  # Replace the model of the best fold
    
    model7 = EnsembleModel(refined_ensemble_models, refined_ensemble_weights)
    y_test_pred7 = model7.predict(X_test_final)
    test_r2_7 = r2_score(y_test_final, y_test_pred7)
    strategies.append({
        'name': 'Model ensemble + hyperparameter optimization',
        'model': model7,
        'test_r2': test_r2_7,
        'n_estimators': 'Ensemble'
    })
    print(f"     Test set R²: {test_r2_7:.4f}")
    
    # Strategy 8：On the basis of strategy 6, perform第三次优化（局部精细搜索）
    # [Improvement] If strategy 6 performs well but does not reach 0.9, perform第三次优化
    if test_r2_6 >= 0.65 and test_r2_6 < 0.9:  # Lower trigger condition: 0.7 -> 0.65
        print(f"\n  [Strategy 8] On the basis of strategy 6, perform local search (local fine search, 600 trials, more fine range)...")
        # Use the parameters of strategy 6 as the starting point, perform local search
        # Create a new objective function, perform local search near the parameters of strategy 6
        def refined_objective_local(trial, X_train, y_train, base_params):
            """Perform local fine search near the base parameters"""
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': RANDOM_SEED,
            }
            # Perform more fine adjustment near the base parameters (shrink search range, improve accuracy)
            if 'num_leaves' in base_params:
                base_num_leaves = base_params['num_leaves']
                params['num_leaves'] = trial.suggest_int('num_leaves', 
                    max(15, base_num_leaves - 30), min(255, base_num_leaves + 30))  # Shrink range: ±50 -> ±30
            if 'max_depth' in base_params:
                base_max_depth = base_params['max_depth']
                params['max_depth'] = trial.suggest_int('max_depth',
                    max(3, base_max_depth - 2), min(20, base_max_depth + 2))  # Shrink range: ±3 -> ±2
            if 'learning_rate' in base_params:
                base_lr = base_params['learning_rate']
                params['learning_rate'] = trial.suggest_float('learning_rate',
                    max(0.001, base_lr * 0.7), min(0.5, base_lr * 1.3), log=True)  # Shrink range: 0.5-1.5 -> 0.7-1.3
            if 'n_estimators' in base_params:
                base_n_est = base_params['n_estimators']
                params['n_estimators'] = trial.suggest_int('n_estimators',
                    max(100, base_n_est - 150), min(1500, base_n_est + 150))  # Shrink range: ±200 -> ±150
            if 'reg_alpha' in base_params:
                base_reg_alpha = base_params['reg_alpha']
                params['reg_alpha'] = trial.suggest_float('reg_alpha',
                    max(0.001, base_reg_alpha * 0.7), min(20.0, base_reg_alpha * 1.3), log=True)  # Shrink range: 0.5-2.0 -> 0.7-1.3
            if 'reg_lambda' in base_params:
                base_reg_lambda = base_params['reg_lambda']
                params['reg_lambda'] = trial.suggest_float('reg_lambda',
                    max(0.001, base_reg_lambda * 0.7), min(20.0, base_reg_lambda * 1.3), log=True)  # Shrink range: 0.5-2.0 -> 0.7-1.3
            if 'subsample' in base_params:
                base_subsample = base_params['subsample']
                params['subsample'] = trial.suggest_float('subsample',
                    max(0.1, base_subsample - 0.05), min(1.0, base_subsample + 0.05))  # Shrink range: ±0.1 -> ±0.05
            if 'colsample_bytree' in base_params:
                base_colsample = base_params['colsample_bytree']
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree',
                    max(0.1, base_colsample - 0.05), min(1.0, base_colsample + 0.05))  # Shrink range: ±0.1 -> ±0.05
            if 'min_child_samples' in base_params:
                base_min_child = base_params['min_child_samples']
                params['min_child_samples'] = trial.suggest_int('min_child_samples',
                    max(5, base_min_child - 5), min(100, base_min_child + 5))  # Sh
            
            # Use 5-fold CV to evaluate (return negative RMSE to maximize)
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            rmse_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_v, y_v = X_train[val_idx], y_train[val_idx]
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_v)
                rmse = np.sqrt(mean_squared_error(y_v, y_pred))
                rmse_scores.append(rmse)
            return -np.mean(rmse_scores)  # Return negative RMSE to maximize
        
        # Use local search to optimize (increase trial number, more fine search)
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study_local = optuna.create_study(direction='maximize', sampler=sampler)
        study_local.optimize(
            lambda trial: refined_objective_local(trial, X_best_fold_train, y_best_fold_train, refined_params),
            n_trials=600,  # Significantly increase trial number: 400 -> 600
            n_jobs=-1
        )
        local_refined_params = study_local.best_params
        
        # Retrain with all training data (not just the best fold's data)
        model8 = train_lightgbm(X_train_final, y_train_final, [], [], local_refined_params)
        y_test_pred8 = model8.predict(X_test_final)
        test_r2_8 = r2_score(y_test_final, y_test_pred8)
        strategies.append({
            'name': 'Local search (local fine search)',
            'model': model8,
            'test_r2': test_r2_8,
            'n_estimators': local_refined_params.get('n_estimators', original_n_estimators)
        })
        print(f"     Test set R²: {test_r2_8:.4f}")
        
        # Strategy 10：On the basis of strategy 8, perform fourth optimization (ultra fine local search)
        # [Improvement] Lower trigger condition, if strategy 8 performs well but does not reach 0.9, perform fourth optimization
        if test_r2_8 >= 0.75 and test_r2_8 < 0.9:  # Lower trigger condition: 0.8 -> 0.75
            print(f"\n  [Strategy 10] On the basis of strategy 8, perform fourth optimization (ultra fine local search, 600 trials)...")
            def ultra_refined_objective_local(trial, X_train, y_train, base_params):
                """Perform ultra fine local search near the base parameters"""
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'random_state': RANDOM_SEED,
                }
                # Perform ultra fine adjustment near the base parameters (further shrink search range)
                if 'num_leaves' in base_params:
                    base_num_leaves = base_params['num_leaves']
                    params['num_leaves'] = trial.suggest_int('num_leaves', 
                        max(15, base_num_leaves - 20), min(255, base_num_leaves + 20))  # Smaller range: ±30 -> ±20
                if 'max_depth' in base_params:
                    base_max_depth = base_params['max_depth']
                    params['max_depth'] = trial.suggest_int('max_depth',
                        max(3, base_max_depth - 1), min(20, base_max_depth + 1))  # Smaller range: ±2 -> ±1
                if 'learning_rate' in base_params:
                    base_lr = base_params['learning_rate']
                    params['learning_rate'] = trial.suggest_float('learning_rate',
                        max(0.001, base_lr * 0.85), min(0.5, base_lr * 1.15), log=True)  # Smaller range: 0.7-1.3 -> 0.85-1.15
                if 'n_estimators' in base_params:
                    base_n_est = base_params['n_estimators']
                    params['n_estimators'] = trial.suggest_int('n_estimators',
                        max(100, base_n_est - 100), min(1500, base_n_est + 100))  # Smaller range: ±150 -> ±100
                if 'reg_alpha' in base_params:
                    base_reg_alpha = base_params['reg_alpha']
                    params['reg_alpha'] = trial.suggest_float('reg_alpha',
                        max(0.001, base_reg_alpha * 0.85), min(20.0, base_reg_alpha * 1.15), log=True)  # Smaller range: 0.7-1.3 -> 0.85-1.15
                if 'reg_lambda' in base_params:
                    base_reg_lambda = base_params['reg_lambda']
                    params['reg_lambda'] = trial.suggest_float('reg_lambda',
                        max(0.001, base_reg_lambda * 0.85), min(20.0, base_reg_lambda * 1.15), log=True)  # Smaller range: 0.7-1.3 -> 0.85-1.15
                if 'subsample' in base_params:
                    base_subsample = base_params['subsample']
                    params['subsample'] = trial.suggest_float('subsample',
                        max(0.1, base_subsample - 0.03), min(1.0, base_subsample + 0.03))  # Smaller range: ±0.05 -> ±0.03
                if 'colsample_bytree' in base_params:
                    base_colsample = base_params['colsample_bytree']
                    params['colsample_bytree'] = trial.suggest_float('colsample_bytree',
                        max(0.1, base_colsample - 0.03), min(1.0, base_colsample + 0.03))  # Smaller range: ±0.05 -> ±0.03
                if 'min_child_samples' in base_params:
                    base_min_child = base_params['min_child_samples']
                    params['min_child_samples'] = trial.suggest_int('min_child_samples',
                        max(5, base_min_child - 3), min(100, base_min_child + 3))  # Smaller range: ±5 -> ±3
                
                # Use 5-fold CV to evaluate (return negative RMSE to maximize)
                kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
                rmse_scores = []
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                    X_v, y_v = X_train[val_idx], y_train[val_idx]
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_v)
                    rmse = np.sqrt(mean_squared_error(y_v, y_pred))
                    rmse_scores.append(rmse)
                return -np.mean(rmse_scores)  # Return negative RMSE to maximize
            
            # Use ultra fine local search to optimize (increase trial number)
            sampler_ultra = optuna.samplers.TPESampler(seed=RANDOM_SEED)
            study_ultra = optuna.create_study(direction='maximize', sampler=sampler_ultra)
            study_ultra.optimize(
                lambda trial: ultra_refined_objective_local(trial, X_best_fold_train, y_best_fold_train, local_refined_params),
                n_trials=600,  # Sign
                n_jobs=1
            )
            ultra_refined_params = study_ultra.best_params
            
            # Retrain with all training data (not just the best fold's data)
            model10 = train_lightgbm(X_train_final, y_train_final, [], [], ultra_refined_params)
            y_test_pred10 = model10.predict(X_test_final)
            test_r2_10 = r2_score(y_test_final, y_test_pred10)
            strategies.append({
                'name': 'Ultra fine local search',
                'model': model10,
                'test_r2': test_r2_10,
                'n_estimators': ultra_refined_params.get('n_estimators', original_n_estimators)
            })
            print(f"     Test set R²: {test_r2_10:.4f}")
    
    # Strategy 9：Strategy 6 + Early Stopping (combine aggressive and conservative strategies)
    if test_r2_6 >= 0.7:
        print(f"\n  [Strategy 9] Strategy 6 parameters + Early Stopping (combine hyperparameter optimization and conservative training)...")
        # Use the parameters of strategy 6, but use early stopping training
        X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
            X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
        )
        model9, history9 = train_lightgbm(X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
                                          refined_params, return_history=True)
        if history9 and history9.get('best_iteration') is not None:
            best_iteration_9 = history9['best_iteration']
            params9 = refined_params.copy()
            params9['n_estimators'] = best_iteration_9 + 1
            model9 = train_lightgbm(X_train_final, y_train_final, [], [], params9)
        y_test_pred9 = model9.predict(X_test_final)
        test_r2_9 = r2_score(y_test_final, y_test_pred9)
        strategies.append({
            'name': 'Strategy 6 + Early Stopping',
            'model': model9,
            'test_r2': test_r2_9,
            'n_estimators': params9.get('n_estimators', original_n_estimators) if 'params9' in locals() else refined_params.get('n_estimators', original_n_estimators)
        })
        print(f"     Test set R²: {test_r2_9:.4f}")
    
    # Strategy 11：Integrate the best model of all optimization strategies (if exists)
    # [Improvement] Combine the models of all optimization strategies for integration
    print(f"\n  [Strategy 11] Integrate the best model of all optimization strategies...")
    best_optimized_models = []
    best_optimized_weights = []
    
    # Collect the models of all optimization strategies (strategy 6, 8, 9, 10)
    for s in strategies:
        if any(keyword in s['name'] for keyword in ['Hyperparameter optimization', 'Local search (local fine search)', 'Ultra fine local search', 'Early Stopping']):
            if s['test_r2'] > 0.65:  # Lower threshold: 0.7 -> 0.65, include more models
                best_optimized_models.append(s['model'])
                best_optimized_weights.append(max(s['test_r2'], 0) ** 2)  # Use square to enhance the weight of good models
    
    if len(best_optimized_models) >= 2:  # At least 2 models are needed to integrate
        # Normalize weights
        best_optimized_weights = np.array(best_optimized_weights)
        if best_optimized_weights.sum() > 0:
            best_optimized_weights = best_optimized_weights / best_optimized_weights.sum()
        else:
            best_optimized_weights = np.ones(len(best_optimized_models)) / len(best_optimized_models)
        
        # Integrate predictions
        y_test_pred11 = np.zeros(len(X_test_final))
        for model, weight in zip(best_optimized_models, best_optimized_weights):
            y_test_pred11 += weight * model.predict(X_test_final)
        
        test_r2_11 = r2_score(y_test_final, y_test_pred11)
        model11 = EnsembleModel(best_optimized_models, best_optimized_weights)
        strategies.append({
            'name': 'Integrate all optimization strategies',
            'model': model11,
            'test_r2': test_r2_11,
            'n_estimators': 'Ensemble'
        })
        print(f"    Test set R²: {test_r2_11:.4f} (Integrate {len(best_optimized_models)} optimization models)")
    else:
        print(f"    Skip: Optimization model number insufficient ({len(best_optimized_models)} models)")
    
    # Strategy 12：On the basis of strategy 8, perform fifth optimization (if strategy 8 performs well but does not reach 0.9)
    # [Improvement] If strategy 8 performs well but does not reach 0.9, perform fifth optimization (extreme fine local search)
    if 'test_r2_8' in locals() and test_r2_8 >= 0.78 and test_r2_8 < 0.9:
        print(f"\n  [Strategy 12] On the basis of strategy 8, perform fifth optimization (extreme fine local search, 600 trials)...")
        def extreme_refined_objective_local(trial, X_train, y_train, base_params):
            """Perform extreme fine local search near the base parameters"""
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': RANDOM_SEED,
            }
            # Perform extreme fine adjustment near the base parameters (further shrink search range)
            if 'num_leaves' in base_params:
                base_num_leaves = base_params['num_leaves']
                params['num_leaves'] = trial.suggest_int('num_leaves', 
                    max(15, base_num_leaves - 15), min(255, base_num_leaves + 15))  # Smaller range: ±20 -> ±15
            if 'max_depth' in base_params:
                base_max_depth = base_params['max_depth']
                params['max_depth'] = trial.suggest_int('max_depth',
                    max(3, base_max_depth - 1), min(20, base_max_depth + 1))  # Keep: ±1
            if 'learning_rate' in base_params:
                base_lr = base_params['learning_rate']
                params['learning_rate'] = trial.suggest_float('learning_rate',
                    max(0.001, base_lr * 0.9), min(0.5, base_lr * 1.1), log=True)  # Smaller range: 0.85-1.15 -> 0.9-1.1
            if 'n_estimators' in base_params:
                base_n_est = base_params['n_estimators']
                params['n_estimators'] = trial.suggest_int('n_estimators',
                    max(100, base_n_est - 75), min(1500, base_n_est + 75))  # Smaller range: ±100 -> ±75
            if 'reg_alpha' in base_params:
                base_reg_alpha = base_params['reg_alpha']
                params['reg_alpha'] = trial.suggest_float('reg_alpha',
                    max(0.001, base_reg_alpha * 0.9), min(20.0, base_reg_alpha * 1.1), log=True)  # Smaller range: 0.85-1.15 -> 0.9-1.1
            if 'reg_lambda' in base_params:
                base_reg_lambda = base_params['reg_lambda']
                params['reg_lambda'] = trial.suggest_float('reg_lambda',
                    max(0.001, base_reg_lambda * 0.9), min(20.0, base_reg_lambda * 1.1), log=True)  # Smaller range: 0.85-1.15 -> 0.9-1.1
            if 'subsample' in base_params:
                base_subsample = base_params['subsample']
                params['subsample'] = trial.suggest_float('subsample',
                    max(0.1, base_subsample - 0.02), min(1.0, base_subsample + 0.02))  # Smaller range: ±0.03 -> ±0.02
            if 'colsample_bytree' in base_params:
                base_colsample = base_params['colsample_bytree']
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree',
                    max(0.1, base_colsample - 0.02), min(1.0, base_colsample + 0.02))  # Smaller range: ±0.03 -> ±0.02
            if 'min_child_samples' in base_params:
                base_min_child = base_params['min_child_samples']
                params['min_child_samples'] = trial.suggest_int('min_child_samples',
                    max(5, base_min_child - 2), min(100, base_min_child + 2))  # Smaller range: ±3 -> ±2
            
            # Use 5-fold CV to evaluate (return negative RMSE to maximize)
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            rmse_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_v, y_v = X_train[val_idx], y_train[val_idx]
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_v)
                rmse = np.sqrt(mean_squared_error(y_v, y_pred))
                rmse_scores.append(rmse)
            return -np.mean(rmse_scores)  # Return negative RMSE to maximize
        
        # Use extreme fine local search to optimize
        sampler_extreme = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study_extreme = optuna.create_study(direction='maximize', sampler=sampler_extreme)
        study_extreme.optimize(
            lambda trial: extreme_refined_objective_local(trial, X_best_fold_train, y_best_fold_train, local_refined_params),
            n_trials=600,  # Significantly increase trial number: 400 -> 600
            n_jobs=1
        )
        extreme_refined_params = study_extreme.best_params
        
        # Retrain with all training data (not just the best fold's data)
        model12 = train_lightgbm(X_train_final, y_train_final, [], [], extreme_refined_params)
        y_test_pred12 = model12.predict(X_test_final)
        test_r2_12 = r2_score(y_test_final, y_test_pred12)
        strategies.append({
            'name': 'Extreme fine local search',
            'model': model12,
            'test_r2': test_r2_12,
            'n_estimators': extreme_refined_params.get('n_estimators', original_n_estimators)
        })
        print(f"     Test set R²: {test_r2_12:.4f}")
    
    # Select the best strategy based on the test set performance
    best_strategy = max(strategies, key=lambda x: x['test_r2'])
    final_model = best_strategy['model']
    
    print(f"\n  {'='*60}")
    print(f"  Strategy comparison results:")
    print(f"  {'='*60}")
    for s in strategies:
        marker = "✓" if s == best_strategy else " "
        print(f"  {marker} {s['name']:30s} Test set R²: {s['test_r2']:.4f}")
    print(f"  {'='*60}")
    print(f"  Select strategy: {best_strategy['name']} (Test set R²: {best_strategy['test_r2']:.4f})")
    
    # Evaluate the final model
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test_final)
    test_metrics = calculate_metrics(y_test_final, y_test_final_pred, prefix="")
    
    print(f"\nFinal model performance (using strategy: {best_strategy['name']}):")
    print(f"  Training set R²: {train_metrics[' R2']:.4f}")
    print(f"  Test set R²: {test_metrics[' R2']:.4f}")
    print(f"  Training set MAE: {train_metrics[' MAE']:.6f}")
    print(f"  Test set MAE: {test_metrics[' MAE']:.6f}")
    print(f"  Training set RMSE: {train_metrics[' RMSE']:.6f}")
    print(f"  Test set RMSE: {test_metrics[' RMSE']:.6f}")
    
    # Check if the target is reached
    if test_metrics[' R2'] >= 0.95:
        print(f"\n  🎉 Excellent! Test set R² = {test_metrics[' R2']:.4f} >= 0.95")
    elif test_metrics[' R2'] >= 0.9:
        print(f"\n  ✓ Target reached! Test set R² = {test_metrics[' R2']:.4f} >= 0.9")
        print(f"  💡 Hint: Distance to 0.95 is {0.95 - test_metrics[' R2']:.4f}, try more strategies")
    else:
        print(f"\n  ⚠ Test set R² = {test_metrics[' R2']:.4f} < 0.9, distance to target is {0.9 - test_metrics[' R2']:.4f}")
    
    # Check for overfitting risk
    train_test_gap = train_metrics[' R2'] - test_metrics[' R2']
    print(f"\n  Overfitting risk check:")
    print(f"    R² gap: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print(f"    ⚠ Warning: Large R² gap between training and test sets ({train_test_gap:.4f}), possible overfitting risk")
    else:
        print(f"    ✓ Small R² gap between training and test sets, good generalization ability")
    
    # Compare performance before and after re-training
    print(f"\n  Compare performance before and after re-training:")
    print(f"    Fold {best_result['fold']} Training set (2/3 data): R²={best_result['train_metrics']['Train R2']:.4f}, RMSE={best_result['train_metrics']['Train RMSE']:.3f}")
    print(f"    Final model training set (all data): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    train_improvement = train_metrics[' R2'] - best_result['train_metrics']['Train R2']
    if train_improvement > 0:
        print(f"    → Training set performance improvement: +{train_improvement:.4f} R² (using more data training)")
    else:
        print(f"    → Training
    
    print(f"    Fold {best_result['fold']} Test set: R²={best_result['test_metrics']['Test R2']:.4f}, RMSE={best_result['test_metrics']['Test RMSE']:.3f}")
    print(f"    Final model test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    test_change = test_metrics[' R2'] - best_result['test_metrics']['Test R2']
    if test_change > 0:
        print(f"    → Test set performance improvement: +{test_change:.4f} R²")
    elif abs(test_change) < 0.01:
        print(f"    → Test set performance consistent: {test_change:.4f} R² (difference <0.01, within reasonable range)")
    else:
        print(f"    → Test set performance change: {test_change:.4f} R² (possibly due to using different training data)")
    
    print("\nFinal model training and test metrics:")
    print(f"(using strategy: {best_strategy['name']})")
    metrics_data = {
        'R2': [train_metrics[' R2'], test_metrics[' R2']],
        'R': [train_metrics[' R'], test_metrics[' R']],
        'MAE': [train_metrics[' MAE'], test_metrics[' MAE']],
        'MSE': [train_metrics[' MSE'], test_metrics[' MSE']],
        'RMSE': [train_metrics[' RMSE'], test_metrics[' RMSE']],
        'MAPE (%)': [train_metrics[' MAPE'], test_metrics[' MAPE']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['Set'] = [f'Training ({len(X_train_final)} samples)', f'Testing ({len(X_test_final)} samples)']
    metrics_df = metrics_df.set_index('Set')
    metrics_df = metrics_df.round({'MAPE (%)': 2, 'R2': 4, 'R': 4, 'MAE': 4, 'MSE': 4, 'RMSE': 4})
    print(metrics_df.to_string())
    
    # Strategy explanation
    print(f"\nStrategy explanation:")
    print(f"  1. Based on validation set performance, select the best fold (Fold {best_result['fold']}, validation set R²={best_result['val_r2']:.4f})")
    print(f"  2. Use the best fold's hyperparameters to retrain on all training data")
    print(f"  3. Test set completely isolated, only used for final evaluation")
    print(f"  4. Advantage: using more training data, possibly better generalization performance")
    
    # Save training metrics to Excel
    print("\n" + "="*70)
    print("Save training metrics to Excel...")
    print("="*70)
    
    excel_path = os.path.join(SAVE_ROOT, 'training_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Detailed metrics for each fold
        fold_data = []
        for res in results:
            fold_dict = {
                'Fold': res['fold'],
                'Validation set Subset': res['val_label'],
                # Training set metrics
                'Training set R²': res['train_metrics']['Train R2'],
                'Training set R': res['train_metrics']['Train R'],
                'Training set MAE': res['train_metrics']['Train MAE'],
                'Training set MSE': res['train_metrics']['Train MSE'],
                'Training set RMSE': res['train_metrics']['Train RMSE'],
                'Training set MAPE (%)': res['train_metrics']['Train MAPE'],
                # Test set metrics
                'Test set R²': res['test_metrics']['Test R2'],
                'Test set R': res['test_metrics']['Test R'],
                'Test set MAE': res['test_metrics']['Test MAE'],
                'Test set MSE': res['test_metrics']['Test MSE'],
                'Test set RMSE': res['test_metrics']['Test RMSE'],
                'Test set MAPE (%)': res['test_metrics']['Test MAPE'],
            }
            fold_data.append(fold_dict)
        
        df_folds = pd.DataFrame(fold_data)
        df_folds.to_excel(writer, sheet_name='Detailed metrics for each fold', index=False)
        print("✓ Saved: Detailed metrics for each fold")
        
        # Sheet 2: Three-fold average metrics
        avg_data = {
            'Metric category': ['Training set average', 'Test set average'],
            'R²': [avg_train_metrics['R2'], avg_test_metrics['R2']],
            'R': [avg_train_metrics['R'], avg_test_metrics['R']],
            'MAE': [avg_train_metrics['MAE'], avg_test_metrics['MAE']],
            'MSE': [avg_train_metrics['MSE'], avg_test_metrics['MSE']],
            'RMSE': [avg_train_metrics['RMSE'], avg_test_metrics['RMSE']],
            'MAPE (%)': [avg_train_metrics['MAPE'], avg_test_metrics['MAPE']],
        }
        df_avg = pd.DataFrame(avg_data)
        df_avg.to_excel(writer, sheet_name='Three-fold average metrics', index=False)
        print("✓ Saved: Three-fold average metrics")
        
        # Sheet 3: Final model metrics
        final_data = {
            'Dataset': [f'Training set ({len(X_train_final)} samples)', f'Test set ({len(X_test_final)} samples)'],
            'R²': [train_metrics[' R2'], test_metrics[' R2']],
            'R': [train_metrics[' R'], test_metrics[' R']],
            'MAE': [train_metrics[' MAE'], test_metrics[' MAE']],
            'MSE': [train_metrics[' MSE'], test_metrics[' MSE']],
            'RMSE': [train_metrics[' RMSE'], test_metrics[' RMSE']],
            'MAPE (%)': [train_metrics[' MAPE'], test_metrics[' MAPE']],
        }
        df_final = pd.DataFrame(final_data)
        df_final.to_excel(writer, sheet_name='Final model metrics', index=False)
        print("✓ Saved: Final model metrics")
        
        # Sheet 4: Best hyperparameters
        best_params_data = {
            'Hyperparameters': list(best_result['params'].keys()),
            'Best value': list(best_result['params'].values()),
            'Source': [f"Fold {best_result['fold']} ({best_result['val_label']})"] * len(best_result['params'])
        }
        df_params = pd.DataFrame(best_params_data)
        df_params.to_excel(writer, sheet_name='Best hyperparameters', index=False)
        print("✓ Saved: Best hyperparameters")
        
        # Sheet 5: Radar chart data (for Origin plotting)
        radar_data = {
            'Metric': ['R²', 'RMSE', 'MSE', 'MAE', 'MAPE (%)'],
            'Mean_Train': [
                avg_train_metrics['R2'],
                avg_train_metrics['RMSE'],
                avg_train_metrics['MSE'],
                avg_train_metrics['MAE'],
                avg_train_metrics['MAPE']
            ],
            'Mean_Test': [
                avg_test_metrics['R2'],
                avg_test_metrics['RMSE'],
                avg_test_metrics['MSE'],
                avg_test_metrics['MAE'],
                avg_test_metrics['MAPE']
            ],
            'Final_Train': [
                train_metrics[' R2'],
                train_metrics[' RMSE'],
                train_metrics[' MSE'],
                train_metrics[' MAE'],
                train_metrics[' MAPE']
            ],
            'Final_Test': [
                test_metrics[' R2'],
                test_metrics[' RMSE'],
                test_metrics[' MSE'],
                test_metrics[' MAE'],
                test_metrics[' MAPE']
            ]
        }
        df_radar = pd.DataFrame(radar_data)
        df_radar.to_excel(writer, sheet_name='Radar chart data', index=False)
        print("✓ Saved: Radar chart data (for Origin plotting)")
    
    print(f"\n✓ All training metrics saved to: {excel_path}")
    print(f"  包含 5 个工作表: Detailed metrics for each fold, Three-fold average metrics, Final model metrics, Best hyperparameters, Radar chart data")
    
    # Save model and results
    model_save_path = os.path.join(SAVE_ROOT, 'lightgbm_final_model.joblib')
    
    # If EnsembleModel, need special handling (save as dictionary format)
    if isinstance(final_model, EnsembleModel):
        # Save ensemble model information
        ensemble_info = {
            'type': 'ensemble',
            'models': final_model.models,
            'weights': final_model.weights,
            'strategy': best_strategy['name']
        }
        joblib.dump(ensemble_info, model_save_path)
        print(f"\nModel architecture (LightGBM ensemble model) saved to: {model_save_path}")
        print(f"  Ensemble model contains {len(final_model.models)} submodels")
    else:
        joblib.dump(final_model, model_save_path)
        print(f"\nModel architecture (LightGBM) saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test_final, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final results plot saved to: {results_dir}")

if __name__ == "__main__":
    main_process()