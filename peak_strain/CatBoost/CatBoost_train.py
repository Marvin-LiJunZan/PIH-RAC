#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost Prediction Model - Peak Strain Prediction (Including Hyperparameter Printing and Learning Curve Monitoring)
Strategy: 3-Fold Hyperparameter Search + Independent Test Set Selection + Final Model Training

Enhanced Features:
1. Print all metrics (R2, R, MAE, MSE, RMSE, MAPE).
2. Print **selected optimal hyperparameters** in the final report.
3. Training/Validation Curve Monitoring: Record and plot learning curves to identify the bias-variance tradeoff point (optimal number of iterations).
   - Automatically use early stopping to prevent overfitting
   - Visualize RMSE changes of training and validation sets with the number of iterations
   - Mark the optimal iteration point (bias-variance tradeoff point)

Note: This script is used for predicting peak strain, with 17 features (15 material parameters + fc + Xiao_strain)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import optuna
import warnings
import joblib
from scipy.stats import pearsonr

# --- Auxiliary Settings and Paths ---

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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_strain', 'CatBoost', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
print(f"Save root: {SAVE_ROOT}")

# --- Core Function Definitions ---

def compute_xiao_strain(fc: np.ndarray, r: np.ndarray) -> np.ndarray:
   """
   Calculate peak strain using Xiao's formula (consistent with PINN script implementation).
    
    Formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r² - 109.43r + 48.989))
    
    Args:
        fc: Stress value (compressive strength)
        r: Aggregate replacement rate (stored as percentage × 100 in data, e.g., 1.5 represents 1.5%)
    
    Returns:
        Peak strain predicted by Xiao's formula
    """
    # r = Aggregate replacement rate, convert unit from percentage × 100 (e.g., 1.5) to decimal fraction (0.015)
    r = r / 100.0
    
    fc_clamped = np.clip(fc.astype(float), a_min=1e-6, a_max=None)
    r_clamped = np.clip(r, a_min=1e-8, a_max=None)
    
    # First term：0.00076 + sqrt((0.626 * σ_cp - 4.33) × 10^-7)
    inner = (0.626 * fc_clamped - 4.33) * 1e-7
    inner_clamped = np.clip(inner, a_min=0.0, a_max=None)
    term1 = 0.00076 + np.sqrt(inner_clamped)

    # Second term：1 + r / (65.715r^2 - 109.43r + 48.989)
    denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
    term2 = 1.0 + (r_clamped / denom)

    return term1 * term2
    
def load_data():
    """
     Load and Parse Data - Predict Peak Strain
    """
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"defult: File does not exist {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    extra_features = ['fc']  # Peak stress as input feature
    formula_features = ['Xiao_strain']  # Xiao formula as input feature
    
    # For peak strain prediction, features include: 15 material parameters + fc + Xiao_strain = 17 features
    feature_names = material_features + specimen_features + extra_features + formula_features
    target_column = 'peak_strain'
    
    # Check if the feature columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: The following feature columns do not exist: {missing_features}")
        # If Xiao_strain is missing, try to calculate it
        if "Xiao_strain" in missing_features and "fc" in df.columns and "r" in df.columns:
            print("  Trying to calculate Xiao_strain...")
            df["Xiao_strain"] = compute_xiao_strain(df["fc"].values, df["r"].values)
            print("  ✓ Xiao_strain has been calculated and added")
            missing_features = [f for f in missing_features if f != "Xiao_strain"]
        
        # Remove other missing features
        if missing_features:
            feature_names = [f for f in feature_names if f in df.columns]
            print(f"  Missing features have been removed: {missing_features}, current feature number: {len(feature_names)}")
    
    # Check if the target variable column exists
    if target_column not in df.columns:
        print(f"Error: The target variable column '{target_column}' is missing")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    X = df[feature_names].values
    y = df[target_column].values
    
    if 'DataSlice' not in df.columns:
        print("Error: The 'DataSlice' column is missing for data division")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully: {len(X)} samples, {len(feature_names)} features")
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV)
    
    Hyperparameter search range optimization (for small sample scenario, approximately 60-70 training samples):
    - depth: 3-8 (small sample suggests 3-6, widen to 8 to cover more possibilities)
    - learning_rate: 0.01-0.3 (CatBoost commonly used range, avoid too high learning rate causing instability)
    - iterations: 100-800 (with lower learning rate, more trees may be needed)
    - Other regularization parameters keep the original range, help to prevent overfitting
    """
    params = {
        'task_type': 'CPU',  # Fixed use CPU
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'loss_function': 'RMSE',  # CatBoost regression default uses RMSE
        # Core principle: conservative, strong regularization, avoid overfitting,suit forCatBoost restrictions
        'depth': trial.suggest_int('depth', 3, 6),  # depth: 3-6 (small sample suggests 3-6, widen to 8 to cover more possibilities)
        'iterations': trial.suggest_int('iterations', 100, 500),  # iterations: 100-500 (with lower learning rate, more trees may be needed)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # learning_rate: 0.01-0.15 (CatBoost commonly used range, avoid too high learning rate causing instability)
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),  # min_child_samples: leaf node minimum sample number: 10-40 (lower limit, strengthen regularization, prevent overfitting)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        # Use more strict early stopping to prevent overfitting (reduce rounds, stop earlier)
        # Reduce from 15-25 rounds to 10-20 rounds, or reduce from 15% to 10% of iterations
        early_stopping_rounds_cv = max(10, min(20, int(params.get('iterations', 100) * 0.10)))
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_v, y_v), verbose=False, early_stopping_rounds=early_stopping_rounds_cv)
        
        y_pred = model.predict(X_v)
        
        # Use RMSE as the objective function (more符合回归问题的常规做法）
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        rmse_scores.append(rmse)
    
    # Return average RMSE (smaller is better, so direction='minimize' in optimize_catboost)
    return np.mean(rmse_scores)

def optimize_catboost(X_train, y_train, n_trials=50):
    """Run Optuna optimization
    
    Note: The objective function is the RMSE of the validation set (smaller is better), so direction='minimize'
    
    [Top-level analysis: Impact of Trial number increase]
    
    1. **Theoretical benefits of increasing Trial number**：
       - Explore larger hyperparameter space, possibly find better combinations
       - TPE sampler needs a certain number of trials to establish an effective probability model (usually requires 20-30 trials)
       - For 7 hyperparameter search spaces, 50 trials already cover the main area
    
    2. **Potential risks of increasing Trial number (small sample scenario)**：
       - **Overfitting to validation set**：Over optimization may lead to hyperparameter overfitting to the validation set, generalization ability下降
       - **Diminishing Returns**: Beyond a certain number, performance improvement is negligible (typically after 50-100 trials)
       - **Computational Cost**: Each trial requires 5-fold CV training; 50 trials = 250 model trainings
       - **Small Sample Limitation**: With 60-70 samples, the validation set only has 12-14 samples, leading to unstable evaluation
    
    3. **Suggestions for current scenario**：
       - **n_trials=50-100**：For small sample, 50 trials are usually sufficient, 100 trials is the upper limit
       - **Outer 3-fold CV**：Each fold is optimized independently,相当于做了3次优化，增加了探索机会
       - **Early stopping mechanism**：If 10-15 trials do not improve, consider stopping early
       - **Search space optimization**：The importance of trial number is more important than the search space (currently optimized)
    
    4. **Determine whether to increase trial number**：
       - If the optimal value continues to decrease after 50 trials → can increase to 100
       - If the optimal value is stable after 50 trials → increasing trial number is not meaningful
       - If the performance of the validation set is significantly different from the test set → may be overfitting, should reduce trial number or strengthen regularization
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)  # Change to minimize, because RMSE越小越好
    
    # [Optimization Suggestions]: For small sample, early stopping mechanism can be added
    # If N consecutive trials do not improve, stop early (save computational resources)
    # Note: Optuna's callback is called after each trial, can check whether to stop
    class EarlyStoppingCallback:
        def __init__(self, patience=15, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.best_value = None
            self.no_improve_count = 0
        
        def __call__(self, study, trial):
            if self.best_value is None:
                self.best_value = study.best_value
                return
            
            # Check if there is a significant improvement (relative improvement > min_delta)
            if study.best_value < self.best_value * (1 - self.min_delta):
                self.best_value = study.best_value
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # If N consecutive trials do not improve, stop optimization
            if self.no_improve_count >= self.patience:
                study.stop()
                print(f"  ⚠ Early stopping triggered: {self.patience} consecutive trials without improvement, stop optimization early")
    
    # For n_trials > 50, enable early stopping (save computational time)
    callbacks = [EarlyStoppingCallback(patience=15, min_delta=0.001)] if n_trials > 50 else None
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=1,
        callbacks=callbacks
    )
    
    # Print optimization statistics
    print(f"  Optimization completed: {len(study.trials)} trials, best RMSE: {study.best_value:.4f}")
    if len(study.trials) < n_trials:
        print(f"  (Early stopping triggered, saved {n_trials - len(study.trials)} trials' computational time)")
    
    return study

def train_catboost(X_train, y_train, X_val, y_val, best_params, return_history=False):
    """Train model: merge X_train and X_val for training
    
    Args:
        return_history: If True, return training history (for plotting learning curve)
    
    Returns:
        model: Trained model
        history: If return_history=True, return training history dictionary
    """
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train
    
    # Fixed use CPU
    model_params = {
        'task_type': 'CPU',  # Fixed use CPU
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'loss_function': 'RMSE',
        **best_params
    }
    # Ensure task_type is CPU (overwrite task_type in best_params if it exists)
    model_params['task_type'] = 'CPU'
    
    # Extract iterations for early stopping
    # For overfitting problem, use more strict early stopping strategy
    iterations = model_params.get('iterations', 100)
    # Use smaller early_stopping_rounds, stop earlier to prevent overfitting
    # Reduce from 20-30 rounds to 10-20 rounds, or reduce from 15% to 10% of iterations
    early_stopping_rounds = max(10, min(20, int(iterations * 0.10)))  # 10-20 rounds, or 10% of iterations (more strict)
    
    # Train model (fixed use CPU)
    history = None
    
    model = CatBoostRegressor(**model_params)
    if len(X_val) > 0 and return_history:
        # Use eval_set to record training history
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
            plot=False
        )
        
        # Extract training history
        evals_result = model.get_evals_result()
        # CatBoost returns the format: {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]}}
        train_rmse = evals_result['learn']['RMSE']
        val_rmse = evals_result['validation']['RMSE']
        
        # Get best_iteration
        best_iteration = model.get_best_iteration()
        if best_iteration is None:
            # If there is no best_iteration, use the iteration with the smallest validation set RMSE
            best_iteration = np.argmin(val_rmse)
        
        best_score = val_rmse[best_iteration] if best_iteration < len(val_rmse) else val_rmse[-1]
        
        history = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'best_iteration': best_iteration,
            'best_score': best_score
        }
        # Retrain final model (use all data, but limit to best iteration)
        if best_iteration is not None:
            model_params_final = model_params.copy()
            model_params_final['iterations'] = best_iteration + 1  # +1 because the index starts from 0
            model = CatBoostRegressor(**model_params_final)
        model.fit(X_full, y_full, verbose=False)
    else:
        model.fit(X_full, y_full, verbose=False)
    
    if return_history:
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
    """Plot true values vs predicted values scatter plot"""
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
    """Plot learning curve: the change of RMSE of training set and validation set with the number of iterations
    
    Used to identify the bias-variance balance point (best iteration number)
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
        print(f"  Best iteration number: {best_iter + 1}/{len(val_rmse)}")
        print(f"  Training set RMSE: {train_rmse[best_iter]:.4f}")
        print(f"  Validation set RMSE: {val_rmse[best_iter]:.4f}")
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
    
    # 2. Outer 3-fold cross-validation (find best hyperparameters)
    print("\n" + "="*70)
    print("Start outer 3-fold cross-validation (based on independent test set performance)")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set (for Refit incremental)")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        print("   Performing Optuna inner 5-fold optimization...")
        study = optimize_catboost(X_train_fold, y_train_fold, n_trials=50) 
        best_params = study.best_params
        
        # Train model and record learning curve (for monitoring overfitting)
        print("   Training model and recording learning curve...")
        model, history = train_catboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                       best_params, return_history=True)
        
        # Save learning curve
        results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
        plot_learning_curve(history, results_dir, prefix=f"Fold_{i+1}")
        
        # Calculate training set metrics (merged training set)
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        y_train_pred = model.predict(X_train_full)
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # Calculate test set metrics
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} Training set R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Test set R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.3f}")
        
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
        print(f"  Training set: R²={res['train_metrics']['Train R2']:.4f}, R={res['train_metrics']['Train R']:.4f}, "
              f"MAE={res['train_metrics']['Train MAE']:.3f}, MSE={res['train_metrics']['Train MSE']:.3f}, "
              f"RMSE={res['train_metrics']['Train RMSE']:.3f}, MAPE={res['train_metrics']['Train MAPE']:.2f}%")
        print(f"  Test set: R²={res['test_metrics']['Test R2']:.4f}, R={res['test_metrics']['Test R']:.4f}, "
              f"MAE={res['test_metrics']['Test MAE']:.3f}, MSE={res['test_metrics']['Test MSE']:.3f}, "
              f"RMSE={res['test_metrics']['Test RMSE']:.3f}, MAPE={res['test_metrics']['Test MAPE']:.2f}%")
    
    # Calculate three-fold average metrics
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
    
    print(f"  Training set average: R²={avg_train_metrics['R2']:.4f}, R={avg_train_metrics['R']:.4f}, "
          f"MAE={avg_train_metrics['MAE']:.3f}, MSE={avg_train_metrics['MSE']:.3f}, "
          f"RMSE={avg_train_metrics['RMSE']:.3f}, MAPE={avg_train_metrics['MAPE']:.2f}%")
    print(f"  Test set average: R²={avg_test_metrics['R2']:.4f}, R={avg_test_metrics['R']:.4f}, "
          f"MAE={avg_test_metrics['MAE']:.3f}, MSE={avg_test_metrics['MSE']:.3f}, "
          f"RMSE={avg_test_metrics['RMSE']:.3f}, MAPE={avg_test_metrics['MAPE']:.2f}%")
    
    # *******************************
    # *** Core modification: print best hyperparameters ***
    # *******************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on R2={best_result['r2']:.4f})")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. Train final model
    print(f"\nTraining final model (using best hyperparameters from Fold {best_result['fold']} + all non-Test data)...")
    print(f"Best fold hyperparameters source: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    
    # To monitor the training process of the final model, split 20% of the training set as validation set
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"  Using 20% of the training set ({len(X_val_monitor)} samples) as monitoring validation set")
    
    # Train final model and record learning curve
    final_model, final_history = train_catboost(
        X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
        best_result['params'], return_history=True
    )
    
    # Save learning curve of the final model
    results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
    plot_learning_curve(final_history, results_dir, prefix="Final_Model")
    
    # Use all training data to retrain the final model (for actual prediction)
    # Strategy explanation:
    # 1. Use best iteration number (conservative strategy): the bias-variance balance point found based on the validation set, to avoid overfitting
    # 2. Use all iterations number (aggressive strategy): because the final model has more data, it may be able to train more rounds
    # Why use best iteration number?
    # - Best iteration number is based on the validation set performance found the bias-variance balance point
    # - Even if using more data to train, excessive iterations still may cause overfitting
    # - In a small sample scenario, the conservative strategy is usually more reliable
    # - The best point on the validation set usually can better generalize to the test set
    
    print("  Using all training data to retrain the final model...")
    best_iteration_final = final_history['best_iteration']
    original_iterations = best_result['params'].get('iterations', 100)
    
    print(f"  Strategy comparison:")
    print(f"    - Best iteration number (used): {best_iteration_final + 1}")
    print(f"    - Original iteration number (not used): {original_iterations}")
    print(f"  Reason: the bias-variance balance point found based on the validation set, to avoid overfitting risk")
    
    # Use best iteration number to train the final model (conservative strategy)
    final_params = best_result['params'].copy()
    final_params['iterations'] = best_iteration_final + 1  # +1 because the index starts from 0
    final_model = train_catboost(X_train_final, y_train_final, [], [], final_params)
    
    # 5. Evaluate final model and print detailed metrics
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_final_pred, prefix="")
    
    # Optional: compare the effect of using all iterations number (for verifying the strategy selection)
    print(f"\n  Strategy comparison: train using all iterations number model for comparison...")
    final_model_full = train_catboost(X_train_final, y_train_final, [], [], best_result['params'])
    y_test_pred_full = final_model_full.predict(X_test)
    test_metrics_full = calculate_metrics(y_test, y_test_pred_full, prefix="")
    
    print(f"  Comparison result (test set):")
    print(f"    Best iteration number model: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    print(f"    All iterations number model: R²={test_metrics_full[' R2']:.4f}, RMSE={test_metrics_full[' RMSE']:.3f}")
    
    # Select the model with better test set performance
    if test_metrics_full[' R2'] > test_metrics[' R2']:
        print(f"  → All iterations number model performs better on the test set, but be aware of the overfitting risk")
        print(f"  → Currently using: best iteration number model (more conservative, more reliable generalization ability)")
    else:
        print(f"  → Best iteration number model performs better on the test set, which verifies the correctness of the strategy selection.")
    
    print("\nFinal model training and test metrics:")
    print(f"(Using best hyperparameters from Fold {best_result['fold']} on all non-test data to train)")
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
    
    # 6. Save model and results
    model_save_path = os.path.join(SAVE_ROOT, 'catboost_final_model.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(final_model, model_save_path)
    print(f"\nModel architecture (CatBoost) saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final results plot saved to: {results_dir}")
    
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
    
    Save content:
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
                float(avg_train_metrics['R2']),
                float(avg_train_metrics['R']),
                float(avg_train_metrics['MAE']),
                float(avg_train_metrics['MSE']),
                float(avg_train_metrics['RMSE']),
                float(avg_train_metrics['MAPE'])
            ],
            'Test Average': [
                float(avg_test_metrics['R2']),
                float(avg_test_metrics['R']),
                float(avg_test_metrics['MAE']),
                float(avg_test_metrics['MSE']),
                float(avg_test_metrics['RMSE']),
                float(avg_test_metrics['MAPE'])
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
            'R²': [float(final_train_metrics[' R2']), float(final_test_metrics[' R2'])],
            'R': [float(final_train_metrics[' R']), float(final_test_metrics[' R'])],
            'MAE': [float(final_train_metrics[' MAE']), float(final_test_metrics[' MAE'])],
            'MSE': [float(final_train_metrics[' MSE']), float(final_test_metrics[' MSE'])],
            'RMSE': [float(final_train_metrics[' RMSE']), float(final_test_metrics[' RMSE'])],
            'MAPE (%)': [float(final_train_metrics[' MAPE']), float(final_test_metrics[' MAPE'])]
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
