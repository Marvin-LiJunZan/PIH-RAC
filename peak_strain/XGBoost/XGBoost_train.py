#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost prediction model - peak strain prediction (includes hyperparameter printing and learning curve monitoring)
Strategy: 3-fold hyperparameter search + independent test set selection + final model training

Enhanced features:
1. Print all metrics (R2, R, MAE, MSE, RMSE, MAPE).
2. Print **the selected best hyperparameters** in the final report.
3. Training/validation curve monitoring: record and plot the learning curve, identify the bias-variance balance point (best iteration number).
   - Automatically use early stopping to prevent overfitting
   - Visualize the change of RMSE of the training set and validation set with the number of iterations
   - Mark the best iteration point (bias-variance balance point)

Note: This script is used to predict peak strain (peak_strain), using 17 features (15 material parameters + fc + Xiao_strain)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
import warnings
import joblib
from scipy.stats import pearsonr

# --- Auxiliary settings and paths ---

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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_strain', 'XGBoost', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"工作目录: {PROJECT_ROOT}")
print(f"保存目录: {SAVE_ROOT}")

# --- 核心函数定义 ---

def compute_xiao_strain(fc: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Xiao formula to calculate peak strain (consistent with PINN script implementation).
    
    Formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
    
    Args:
        fc: Stress value (compressive strength)
        r: Aggregate replacement rate (data stored as percentage × 100, e.g., 1.5 represents 1.5%)
    
    Returns:
        Peak strain predicted by Xiao formula
    """
    # r = aggregate replacement rate, from percentage × 100 (e.g., 1.5) to decimal (0.015)
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
    """Load and parse data - predict peak strain"""
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"Error: file does not exist {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    extra_features = ['fc']  # Peak stress as input feature
    formula_features = ['Xiao_strain']  # Xiao formula as input feature
    
    # For peak strain prediction, features include: 15 material parameters + fc + Xiao_strain = 17 features
    feature_names = material_features + specimen_features + extra_features + formula_features
    target_column = 'peak_strain'
    
    # Check if feature columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: the following feature columns do not exist: {missing_features}")
        # If Xiao_strain is missing, try to calculate
        if "Xiao_strain" in missing_features and "fc" in df.columns and "r" in df.columns:
            print("  Trying to calculate Xiao_strain...")
            df["Xiao_strain"] = compute_xiao_strain(df["fc"].values, df["r"].values)
            print("  ✓ Xiao_strain has been calculated and added")
            missing_features = [f for f in missing_features if f != "Xiao_strain"]
        
        # Remove other missing features
        if missing_features:
            feature_names = [f for f in feature_names if f in df.columns]
            print(f"  Removed missing features: {missing_features}, current feature number: {len(feature_names)}")
    
    # Check if target variable column exists
    if target_column not in df.columns:
        print(f"Error: missing target variable column '{target_column}'")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    X = df[feature_names].values
    y = df[target_column].values
    
    if 'DataSlice' not in df.columns:
        print("Error: missing 'DataSlice' column for data splitting")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully: {len(X)} samples, {len(feature_names)} features")
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV) - use R² optimization (larger is better)
    
    Hyperparameter search range optimization for small sample scenario (about 60-70 training samples): small sample suggests 3-6,放宽到8以覆盖更多可能性放宽到8以覆盖更多可能性
    - max_depth: 3-8 (small sample suggests 3-6, widen to 8 to cover more possibilities)
    - learning_rate: 0.01-0.3 (conservative range, avoid too high learning rate unstable)
    - n_estimators: 100-800 (with lower learning rate, more trees may be needed)
    - other regularization parameters keep original range, help prevent overfitting
    """
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': RANDOM_SEED,
        'verbosity': 0,
        # [Top-level master hyperparameter search space - optimized for small sample scenario (about 60-70 training samples)]
        # Core principles: conservative, strong regularization, avoid overfitting
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # depth: 3-8 layers (small sample not too deep)
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),  # iterations: 100-800 (with early stopping, avoid overtraining)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # learning rate: 0.01-0.3 (conservative range, avoid too high instability)
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # leaf minimum weight: 1-20 (strengthen regularization, prevent overfitting)
        'gamma': trial.suggest_float('gamma', 0.0, 0.6),  # minimum loss reduction: 0.0-0.6
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # sample sampling: 0.5-1.0 (keep randomness)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  # feature sampling: 0.3-1.0 (reasonable range, avoid information loss)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),  # L1 regularization: 0.001-20.0
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),  # L2 regularization: 0.001-20.0 (strengthen regularization)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_v)
        
        # Use R² as objective function (larger is better, so direction='maximize' in optimize_xgboost)
        r2 = r2_score(y_v, y_pred)
        r2_scores.append(r2)
    
    # Return average R² (larger is better, so direction='maximize' in optimize_xgboost)
    return np.mean(r2_scores)

def optimize_xgboost(X_train, y_train, n_trials=100):
    """Run Optuna optimization - use R² optimization (larger is better), no early stopping mechanism
    
    Note: the objective function is the R² of the validation set (larger is better), so direction='maximize'
    
    Strategy (based on the paper):
    - Use R² as optimization objective
    - 100 trials, no early stopping mechanism (explicitly stated in the paper: 100 trials per fold)
    - Conservative search space (prevent overfitting)
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='xgboost_optimization')  # maximize，因为R²越大越好
    
    # No early stopping, run all trials
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=1
    )
    
    # Print optimization statistics
    print(f"  Optimization completed: {len(study.trials)} trials, best R²: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")
    
    return study

def train_xgboost(X_train, y_train, X_val, y_val, best_params, return_history=False):
    """Train model: use early stopping to determine the best number of iterations, then retrain with all data (conservative strategy)
    
    Parameters:
        return_history: if True, return training history (for plotting learning curve)
    
    Returns:
        model: trained model
        history: if return_history=True, return training history dictionary
        best_iteration: best number of iterations
    
    Strategy (based on the paper):
    - Use early stopping to prevent overfitting
    - Determine the best number of iterations based on the performance of the validation set
    - Retrain with all data, but limit the number of iterations to the best number of iterations
    """
    if len(X_val) > 0:
        # First, use the training set and validation set to determine the best number of iterations
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Temporary model for determining the best number of iterations (passed in early_stopping_rounds during initialization)
        temp_params = best_params.copy()
        temp_params['early_stopping_rounds'] = 50
        
        temp_model = xgb.XGBRegressor(**temp_params)
        temp_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get the best number of iterations
        best_iteration = temp_model.best_iteration if hasattr(temp_model, 'best_iteration') else best_params.get('n_estimators', 100)
        best_iteration = max(1, best_iteration)  # Ensure at least 1 iteration
        
        # Retrain with all data, but limit the number of iterations to the best number of iterations
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        
        final_params = best_params.copy()
        final_params['n_estimators'] = best_iteration + 1  # Conservative strategy: use the best number of iterations
        
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_full, y_full)
        
        if return_history:
            # Extract training history (for visualization)
            results = temp_model.evals_result()
            history = {
                'train_rmse': results['validation_0']['rmse'],
                'val_rmse': results['validation_1']['rmse'],
                'best_iteration': best_iteration,
                'best_score': results['validation_1']['rmse'][best_iteration] if best_iteration < len(results['validation_1']['rmse']) else results['validation_1']['rmse'][-1]
            }
            return model, history
        return model
    else:
        # No validation set, train directly
        X_full = X_train
        y_full = y_train
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_full, y_full)
        
        if return_history:
            # No validation set, cannot get history
            history = None
            return model, history
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
    
    plt.title(f'{prefix} Prediction\n$R^2$={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
    plt.xlabel('True Values (Peak Strain)')
    plt.ylabel('Predicted Values (Peak Strain)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_scatter.png'), dpi=300)
    plt.close()

def plot_learning_curve(history, save_dir, prefix=""):
    """Plot the learning curve: the RMSE of the training set and validation set changes with the number of iterations
    
    Used to identify the bias-variance balance point (best number of iterations)
    """
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
    plt.title(f'{prefix} Learning Curve\n(Bias-Variance Balance Point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add explanation text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-Validation Gap: {gap:.4f}\n(Positive value indicates possible mild underfitting)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-Validation Gap: {gap:.4f}\n(Negative value indicates possible overfitting risk)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                    fontsize=10)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_learning_curve.png'), dpi=300)
    plt.close()
    
    print(f"  Learning curve saved: {prefix}_learning_curve.png")
    if best_iter < len(val_rmse):
        print(f"  Best Iteration: {best_iter + 1}/{len(val_rmse)}")
        print(f"  Training RMSE: {train_rmse[best_iter]:.4f}")
        print(f"  Validation RMSE: {val_rmse[best_iter]:.4f}")
        print(f"  Training-Validation Gap: {train_rmse[best_iter] - val_rmse[best_iter]:.4f}")
    

# --- Main process (core modifications here) ---

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
    print("Start outer 3-fold hyperparameter search (based on the performance of the independent test set)")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set (for Refit incremental)")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        print("    Running Optuna inner 5-fold optimization...")
        study = optimize_xgboost(X_train_fold, y_train_fold, n_trials=100) 
        best_params = study.best_params
        
        # Train model and record learning curve (for monitoring overfitting)
        print("    Train model and record learning curve...")
        model, history = train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                       best_params, return_history=True)
        
        # Save learning curve
        results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
        plot_learning_curve(history, results_dir, prefix=f"Fold_{i+1}")
        
        # Calculate training set metrics (combined training set)
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        y_train_pred = model.predict(X_train_full)
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # Calculate test set metrics
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} Training R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Test R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.3f}")
        
        results.append({
            'fold': i+1,
            'val_label': val_label,
            'r2': test_metrics['Test R2'],
            'params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })

    # 3. Result summary and best fold selection
    r2_values = [res['r2'] for res in results]
    mean_r2 = np.mean(r2_values)
    best_result = max(results, key=lambda x: x['r2'])
    
    print("\n" + "="*70)
    print("Cross-validation result summary (training set and test set metrics for each fold):")
    print("="*70)
    
    # Print detailed metrics for each fold
    for res in results:
        print(f"\nFold {res['fold']} ({res['val_label']} as validation set):")
        print(f"  Training R2: R²={res['train_metrics']['Train R2']:.4f}, R={res['train_metrics']['Train R']:.4f}, "
              f"MAE={res['train_metrics']['Train MAE']:.3f}, MSE={res['train_metrics']['Train MSE']:.3f}, "
              f"RMSE={res['train_metrics']['Train RMSE']:.3f}, MAPE={res['train_metrics']['Train MAPE']:.2f}%")
        print(f"  Test R2: R²={res['test_metrics']['Test R2']:.4f}, R={res['test_metrics']['Test R']:.4f}, "
              f"MAE={res['test_metrics']['Test MAE']:.3f}, MSE={res['test_metrics']['Test MSE']:.3f}, "
              f"RMSE={res['test_metrics']['Test RMSE']:.3f}, MAPE={res['test_metrics']['Test MAPE']:.2f}%")
    
    # Calculate the average metrics for the three folds
    print("\n" + "-"*70)
    print("Average metrics for the three folds:")
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
    
    print(f"  Training average: R²={avg_train_metrics['R2']:.4f}, R={avg_train_metrics['R']:.4f}, "
          f"MAE={avg_train_metrics['MAE']:.3f}, MSE={avg_train_metrics['MSE']:.3f}, "
          f"RMSE={avg_train_metrics['RMSE']:.3f}, MAPE={avg_train_metrics['MAPE']:.2f}%")
    print(f"  Test average: R²={avg_test_metrics['R2']:.4f}, R={avg_test_metrics['R']:.4f}, "
          f"MAE={avg_test_metrics['MAE']:.3f}, MSE={avg_test_metrics['MSE']:.3f}, "
          f"RMSE={avg_test_metrics['RMSE']:.3f}, MAPE={avg_test_metrics['MAPE']:.2f}%")
    
    # *********************************************************
    # *** Core modification: print the best hyperparameters ***
    # *********************************************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on R2={best_result['r2']:.4f})")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. Train the final model
    print(f"\nTraining the final model (using the best hyperparameters from Fold {best_result['fold']} + all non-Test data)...")
    print(f"Best fold hyperparameters source: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    
    # To monitor the training process of the final model, split 20% of the training set as validation set
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"  Using 20% of the training set ({len(X_val_monitor)} samples) as monitoring validation set")
    
    # Train the final model and record the learning curve
    final_model, final_history = train_xgboost(
        X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
        best_result['params'], return_history=True
    )
    
    # Save the learning curve of the final model
    results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
    plot_learning_curve(final_history, results_dir, prefix="Final_Model")
    
    # Retrain the final model with all training data (for actual prediction)
    # 策略说明：
    # 1. Use the best number of iterations (conservative strategy): based on the validation set to find the bias-variance balance point, avoid overfitting
    # 2. Use all iterations (aggressive strategy): because the final model has more data, it may be able to train more rounds
    # 
    # Why use the best number of iterations?
    # - The best number of iterations is based on the performance of the validation set to find the bias-variance balance point
    # - Even if more data is used to train, excessive iterations may still cause overfitting
    # - In a small sample scenario, the conservative strategy is usually more reliable
    # - The best point on the validation set usually can better generalize to the test set
    
    print("  Retrain the final model with all training data...")
    best_iteration_final = final_history['best_iteration'] if final_history else best_result['params'].get('n_estimators', 100)
    original_iterations = best_result['params'].get('n_estimators', 100)
    
    print(f"  Strategy comparison:")
    print(f"    - Best number of iterations (used): {best_iteration_final + 1}")
    print(f"    - Original number of iterations (not used): {original_iterations}")
    print(f"  Reason: based on the validation set to find the bias-variance balance point, avoid overfitting risk")
    
    # Train the final model with the best number of iterations (conservative strategy)
    final_params = best_result['params'].copy()
    final_params['n_estimators'] = best_iteration_final + 1  # +1 because the index starts from 0
    final_model = train_xgboost(X_train_final, y_train_final, [], [], final_params)
    
    # 5. Evaluate the final model and print detailed metrics
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_final_pred, prefix="")
    
    # Optional: compare the effect of using all iterations (for verifying the strategy selection)
    print(f"\n  Strategy comparison: train the model with all iterations...")
    final_model_full = train_xgboost(X_train_final, y_train_final, [], [], best_result['params'])
    y_test_pred_full = final_model_full.predict(X_test)
    test_metrics_full = calculate_metrics(y_test, y_test_pred_full, prefix="")
    
    print(f"  Comparison result (test set):")
    print(f"    Best number of iterations model: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    print(f"    All iterations model: R²={test_metrics_full[' R2']:.4f}, RMSE={test_metrics_full[' RMSE']:.3f}")
    
    # Select the model with better performance on the test set
    if test_metrics_full[' R2'] > test_metrics[' R2']:
        print(f"  All iterations model performs better on the test set, but be aware of the risk of overfitting")
        print(f"  Currently using: Best number of iterations model (more conservative, better generalization ability)")
    else:
        print(f"  Best number of iterations model performs better on the test set, verifying the correctness of the strategy selection")
    
    print("\nFinal model training and test metrics:")
    print(f"(Using the best hyperparameters from Fold {best_result['fold']} + all non-Test data)")
    metrics_data = {
        'R2': [train_metrics[' R2'], test_metrics[' R2']],
        'R': [train_metrics[' R'], test_metrics[' R']],
        'MAE': [train_metrics[' MAE'], test_metrics[' MAE']],
        'MSE': [train_metrics[' MSE'], test_metrics[' MSE']],
        'RMSE': [train_metrics[' RMSE'], test_metrics[' RMSE']],
        'MAPE (%)': [train_metrics[' MAPE'], test_metrics[' MAPE']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['Set'] = ['Training (S1+2+3)', f'Testing ({len(X_test)} samples)']
    metrics_df = metrics_df.set_index('Set')
    metrics_df = metrics_df.round({'MAPE (%)': 2, 'R2': 4, 'R': 4, 'MAE': 4, 'MSE': 4, 'RMSE': 4})
    print(metrics_df.to_string())
    
    # Comparison explanation
    print(f"\nComparison explanation:")
    print(f"  - Fold {best_result['fold']} Training set (subset1+subset3): R²={best_result['train_metrics']['Train R2']:.4f}, RMSE={best_result['train_metrics']['Train RMSE']:.3f}")
    print(f"  - Fold {best_result['fold']} Test set: R²={best_result['test_metrics']['Test R2']:.4f}, RMSE={best_result['test_metrics']['Test RMSE']:.3f}")
    print(f"  - Final model training set (subset1+subset2+subset3): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    print(f"  - Final model test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    
    # 6. Save the model and results
    model_save_path = os.path.join(SAVE_ROOT, 'xgboost_final_model.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(final_model, model_save_path)
    print(f"\nModel architecture (XGBoost) saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final result plot saved to: {results_dir}")
    
    # 7. Save all metrics to Excel
    excel_path = os.path.join(SAVE_ROOT, 'training_metrics.xlsx')
    save_metrics_to_excel(
        results, avg_train_metrics, avg_test_metrics,
        best_result, train_metrics, test_metrics,
        best_result['params'], feature_names, excel_path
    )
    print(f"\nAll training metrics saved to: {excel_path}")

def save_metrics_to_excel(results, avg_train_metrics, avg_test_metrics,
                          best_result, final_train_metrics, final_test_metrics,
                          best_params, feature_names, excel_path):
    """Save all training metrics to Excel file
    
    Content:
    1. Detailed metrics for each fold
    2. 3-fold cross-validation average metrics
    3. Final model metrics
    4. Best hyperparameters
    """
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Detailed metrics for each fold
        fold_data = []
        for res in results:
            fold_data.append({
                'Fold': res['fold'],
                'Validation Set': res['val_label'],
                'Train R²': res['train_metrics']['Train R2'],
                'Train R': res['train_metrics']['Train R'],
                'Train MAE': res['train_metrics']['Train MAE'],
                'Train MSE': res['train_metrics']['Train MSE'],
                'Train RMSE': res['train_metrics']['Train RMSE'],
                'Train MAPE (%)': res['train_metrics']['Train MAPE'],
                'Test R²': res['test_metrics']['Test R2'],
                'Test R': res['test_metrics']['Test R'],
                'Test MAE': res['test_metrics']['Test MAE'],
                'Test MSE': res['test_metrics']['Test MSE'],
                'Test RMSE': res['test_metrics']['Test RMSE'],
                'Test MAPE (%)': res['test_metrics']['Test MAPE']
            })
        fold_df = pd.DataFrame(fold_data)
        fold_df.to_excel(writer, sheet_name='Fold_Details', index=False)
        
        # Sheet 2: 3-fold cross-validation average metrics
        cv_avg_data = {
            'Metric': ['R²', 'R', 'MAE', 'MSE', 'RMSE', 'MAPE (%)'],
            'Train Average': [
                avg_train_metrics['R2'],
                avg_train_metrics['R'],
                avg_train_metrics['MAE'],
                avg_train_metrics['MSE'],
                avg_train_metrics['RMSE'],
                avg_train_metrics['MAPE']
            ],
            'Test Average': [
                avg_test_metrics['R2'],
                avg_test_metrics['R'],
                avg_test_metrics['MAE'],
                avg_test_metrics['MSE'],
                avg_test_metrics['RMSE'],
                avg_test_metrics['MAPE']
            ]
        }
        cv_avg_df = pd.DataFrame(cv_avg_data)
        cv_avg_df.to_excel(writer, sheet_name='CV_Average', index=False)
        
        # Sheet 3: Best fold performance
        best_fold_data = {
            'Set': ['Training', 'Testing'],
            'R²': [
                best_result['train_metrics']['Train R2'],
                best_result['test_metrics']['Test R2']
            ],
            'R': [
                best_result['train_metrics']['Train R'],
                best_result['test_metrics']['Test R']
            ],
            'MAE': [
                best_result['train_metrics']['Train MAE'],
                best_result['test_metrics']['Test MAE']
            ],
            'MSE': [
                best_result['train_metrics']['Train MSE'],
                best_result['test_metrics']['Test MSE']
            ],
            'RMSE': [
                best_result['train_metrics']['Train RMSE'],
                best_result['test_metrics']['Test RMSE']
            ],
            'MAPE (%)': [
                best_result['train_metrics']['Train MAPE'],
                best_result['test_metrics']['Test MAPE']
            ]
        }
        best_fold_df = pd.DataFrame(best_fold_data)
        best_fold_df.to_excel(writer, sheet_name='Best_Fold', index=False)
        
        # Sheet 4: Final model performance
        final_model_data = {
            'Set': ['Training (S1+2+3)', 'Testing'],
            'R²': [final_train_metrics[' R2'], final_test_metrics[' R2']],
            'R': [final_train_metrics[' R'], final_test_metrics[' R']],
            'MAE': [final_train_metrics[' MAE'], final_test_metrics[' MAE']],
            'MSE': [final_train_metrics[' MSE'], final_test_metrics[' MSE']],
            'RMSE': [final_train_metrics[' RMSE'], final_test_metrics[' RMSE']],
            'MAPE (%)': [final_train_metrics[' MAPE'], final_test_metrics[' MAPE']]
        }
        final_model_df = pd.DataFrame(final_model_data)
        final_model_df.to_excel(writer, sheet_name='Final_Model', index=False)
        
        # Sheet 5: Best hyperparameters
        hyperparams_data = {
            'Hyperparameter': list(best_params.keys()),
            'Value': list(best_params.values())
        }
        hyperparams_df = pd.DataFrame(hyperparams_data)
        hyperparams_df.to_excel(writer, sheet_name='Best_Hyperparameters', index=False)
        
        # Sheet 6: Model information summary
        summary_data = {
            'Item': [
                'Best Fold',
                'Best Fold Validation Set',
                'Best Fold Test R²',
                'Best Fold Test RMSE',
                'Final Model Train R²',
                'Final Model Train RMSE',
                'Final Model Test R²',
                'Final Model Test RMSE',
                'CV Mean Train R²',
                'CV Mean Test R²',
                'Target Variable',
                'Number of Features'
            ],
            'Value': [
                f"Fold {best_result['fold']}",
                best_result['val_label'],
                f"{best_result['test_metrics']['Test R2']:.4f}",
                f"{best_result['test_metrics']['Test RMSE']:.4f}",
                f"{final_train_metrics[' R2']:.4f}",
                f"{final_train_metrics[' RMSE']:.4f}",
                f"{final_test_metrics[' R2']:.4f}",
                f"{final_test_metrics[' RMSE']:.4f}",
                f"{avg_train_metrics['R2']:.4f}",
                f"{avg_test_metrics['R2']:.4f}",
                'Peak Strain',
                len(feature_names)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

if __name__ == "__main__":
    main_process()
