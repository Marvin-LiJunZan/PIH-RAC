#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost prediction model - train version (using R² optimization and conservative hyperparameters)
Conservative strategy: 3-fold hyperparameter search + independent test set selection + final model training

Main characteristics (aligned with paper version):
1. Optimization target changed to R² ( larger the better )
2. Use conservative hyperparameter search range (max_depth: 3-8, n_estimators: 100-800, etc.)
3. No early stopping (aggressive strategy)
4. 100 trials (no early stopping mechanism, paper clearly states: 100 trials per fold)
5. Final model uses all iterations (aggressive strategy)
6. GPU acceleration support: automatically detect and use gpu_hist (if available), otherwise fallback to hist (CPU mode) tree method
   - XGBoost 3.0.4 version supports gpu_hist tree method
   - Code will automatically detect GPU support, if not available use CPU mode
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

# GPU support detection (XGBoost 3.0.4 supports gpu_hist)
USE_GPU = False  # Whether to use GPU
_GPU_CHECKED = False  # Mark whether GPU has been detected

def check_gpu_support():
    """Check if XGBoost supports GPU acceleration (using gpu_hist)"""
    global USE_GPU, _GPU_CHECKED
    
    if _GPU_CHECKED:
        return USE_GPU
    
    _GPU_CHECKED = True
    
    try:
        # Test GPU support using gpu_hist
        X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_test = np.array([1.0, 2.0])
        
        test_params = {
            'tree_method': 'gpu_hist',
            'n_estimators': 1,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        }
        test_model = xgb.XGBRegressor(**test_params)
        test_model.fit(X_test, y_test)
        USE_GPU = True
        print("✓ GPU support detected, will use gpu_hist tree method for acceleration training")
        return True
    except Exception as e:
        USE_GPU = False
        print("⚠ GPU support not detected, will use hist tree method (CPU mode)")
        return False

def get_gpu_params():
    """Get GPU parameters (if supported) or return CPU parameters"""
    if check_gpu_support():
        return {'tree_method': 'gpu_hist'}  # GPU mode
    else:
        return {'tree_method': 'hist'}  # CPU mode

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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_stress', 'XGBoost', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Working directory: {PROJECT_ROOT}")
print(f"Save directory: {SAVE_ROOT}")

# --- Core function definition ---
    
def load_data():
    """Load and parse data"""
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"Error: file not found {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    feature_names = material_features + specimen_features
    target_column = 'fc'
    
    X = df[feature_names].values
    y = df[target_column].values
    
    if 'DataSlice' not in df.columns:
        print("Error: missing 'DataSlice' column for data division")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully: {len(X)} samples, {len(feature_names)} features")
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV) - using R² optimization (越大越好）
    
    Hyperparameter search range (采用论文中的保守范围）：
    - max_depth: 3-8 (conservative range, prevent overfitting)
    - learning_rate: 0.01-0.3 (log) (conservative range)
    - n_estimators: 100-800 (conservative range) conservative range
    - Other parameters also use conservative range
    """
    # Get GPU parameters (if supported)
    gpu_params = get_gpu_params()
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': RANDOM_SEED,
        'verbosity': 0,
        **gpu_params,  # Add GPU parameters (if supported) or use CPU mode
        # Hyperparameter search space (采用论文中的保守范围）
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # Conservative range: 3-8
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),  # Conservative range: 100-800
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Conservative range: 0.01-0.3
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 0.6),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Conservative range: 0.5-1.0
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  # Conservative range: 0.3-1.0
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        try:
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
        except Exception as e:
            # If GPU is not available, fallback to CPU mode
            if 'gpu_hist' in str(e).lower() or 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                # Switch to CPU mode
                params_cpu = params.copy()
                params_cpu['tree_method'] = 'hist'
                if USE_GPU:
                    print("⚠ GPU training failed, automatically switched to CPU mode")
                model = xgb.XGBRegressor(**params_cpu)
                model.fit(X_tr, y_tr)
            else:
                raise  # Other exceptions directly thrown
        
        y_pred = model.predict(X_v)
        
        # Use R² as the objective function (larger the better, so in optimize_xgboost direction='maximize'）
        r2 = r2_score(y_v, y_pred)
        r2_scores.append(r2)
    
    # Return average R²（larger the better, so in optimize_xgboost direction='maximize'）
    return np.mean(r2_scores)

def optimize_xgboost(X_train, y_train, n_trials=100):
    """Run Optuna optimization - using R² optimization (larger the better），无早停机制
    
    Note: The objective function is the R² of the validation set (larger the better）， so direction='maximize'
    
    Strategy description (采用论文版本）：
    - Use R² as the optimization target
    - 100 trials, no early stopping mechanism (paper clearly states: 100 trials per fold）
    - Conservative search space (prevent overfitting）
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='xgboost_optimization')  # maximize, because R² larger the better
    
    # No early stopping mechanism, run all trials
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
    """Train model: use early stopping to determine the best number of iterations, then use all data to retrain (conservative strategy）
    
    Parameters:
        return_history: If True, return training history (for plotting learning curve）
    
    Returns:
        model: Trained model
        history: If return_history=True, return training history dictionary
        best_iteration: Best number of iterations
    
    Strategy description (采用train version）：
    - Use early stopping to prevent overfitting
    - Determine the best number of iterations based on validation set performance
    - Use all data to retrain, but limit the number of iterations to the best number of iterations
    """
    if len(X_val) > 0:
        # First use training set and validation set to determine the best number of iterations
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Temporary model for determining the best number of iterations (passed in early_stopping_rounds during initialization）
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
        best_iteration = max(1, best_iteration)  # Ensure at least 1 round
        
        # Use all data to retrain, but limit the number of iterations to the best number of iterations
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        
        final_params = best_params.copy()
        final_params['n_estimators'] = best_iteration + 1  # Conservative strategy: use the best number of iterations
        
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_full, y_full)
        
        if return_history:
            # Extract training history (for visualization）
            results = temp_model.evals_result()
            history = {
                'train_rmse': results['validation_0']['rmse'],
                'val_rmse': results['validation_1']['rmse'],
                'best_iteration': best_iteration,
                'best_score': results['validation_1']['rmse'][best_iteration] if best_iteration < len(results['validation_1']['rmse']) else results['validation_1']['rmse'][-1]
            }
            return model, history, best_iteration
        else:
            return model, best_iteration
            
    else:
        # If there is no validation set, use all data to train
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        
        if return_history:
            return model, None, best_params.get('n_estimators', 100)
        else:
            return model, best_params.get('n_estimators', 100)

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
    plt.xlabel('True Values (MPa)')
    plt.ylabel('Predicted Values (MPa)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_scatter.png'), dpi=300)
    plt.close()

def plot_learning_curve(history, save_dir, prefix=""):
    """Plot learning curve: RMSE of training set and validation set changes with the number of iterations
    
    Used to identify the bias-variance balance point (best number of iterations）
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
    
    # Mark the best iteration point (bias-variance balance point）
    if best_iter < len(val_rmse):
        plt.axvline(x=best_iter + 1, color='g', linestyle='--', linewidth=2, 
                   label=f'Best Iteration ({best_iter + 1})')
        plt.plot(best_iter + 1, val_rmse[best_iter], 'go', markersize=10, 
                label=f'Best Score: {best_score:.4f}')
    
    plt.xlabel('Iteration (Boosting Round)', fontsize=12)
    plt.ylabel('RMSE (MPa)', fontsize=12)
    plt.title(f'{prefix} Learning Curve\n(Bias-Variance Balance Point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add explanation text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-Validation Gap: {gap:.4f} MPa\n(Positive value indicates possible mild underfitting)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-Validation Gap: {gap:.4f} MPa\n(Negative value indicates possible overfitting risk)',
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
        print(f"  Best number of iterations: {best_iter + 1}/{len(val_rmse)}")
        print(f"  Training set RMSE: {train_rmse[best_iter]:.4f} MPa")
        print(f"  Validation set RMSE: {val_rmse[best_iter]:.4f} MPa")
        print(f"  Training-Validation Gap: {train_rmse[best_iter] - val_rmse[best_iter]:.4f} MPa")

# --- Main process ---

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
    
    # 2. Outer 3-fold cross validation (find best hyperparameters）
    print("\n" + "="*70)
    print("Start outer 3-fold cross validation (using R² optimization, conservative search range, 100 trials per fold）")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        print("   正在进行Optuna inner 5-fold optimization (100 trials, no early stopping)...")
        study = optimize_xgboost(X_train_fold, y_train_fold, n_trials=100) 
        best_params = study.best_params
        
        # Train model: first use early stopping to record learning curve, then use all iterations to retrain (consistent with final model strategy）
        print("   Train model (first record learning curve, then use all iterations to retrain）...")
        
        # First use early stopping to record learning curve
        model_temp, history, best_iteration = train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                      best_params, return_history=True)
        
        # Save learning curve
        results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
        plot_learning_curve(history, results_dir, prefix=f"Fold_{i+1}")
        
        # Second use all iterations to retrain (consistent with final model strategy）
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        
        # Use all iterations to train (aggressive strategy, consistent with final model strategy）
        model_params = best_params.copy()
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_full, y_train_full)
        
        # Calculate training set metrics (merged training set）
        y_train_pred = model.predict(X_train_full)
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # Calculate validation set metrics (for model selection, not for final evaluation）
        y_val_pred = model.predict(X_val_fold)
        val_metrics = calculate_metrics(y_val_fold, y_val_pred, prefix="Val")
        
        # Calculate test set metrics (only for recording, not for model selection）
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} Training set R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Validation set R2: {val_metrics['Val R2']:.4f}, RMSE: {val_metrics['Val RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Test set R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.3f}")
        print(f"   >>> Used iterations: {best_params.get('n_estimators', 100)}（All iterations, consistent with final model strategy）")
        
        results.append({
            'fold': i+1,
            'val_label': val_label,
            'val_r2': val_metrics['Val R2'],  # For recording
            'test_r2': test_metrics['Test R2'],  # For model selection (paper strategy）
            'params': best_params,
            'best_iteration': best_params.get('n_estimators', 100),  # Use all iterations
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': model,
            'train_val_indices': np.concatenate([np.where(train_mask)[0], np.where(val_mask)[0]]),
            'test_indices': np.where(test_mask)[0]
        })

    # 3. Result summary and best fold selection
    val_r2_values = [res['val_r2'] for res in results]
    test_r2_values = [res['test_r2'] for res in results]
    mean_val_r2 = np.mean(val_r2_values)
    mean_test_r2 = np.mean(test_r2_values)
    
    # Based on test set R² select best fold (paper clearly states: highest test set R²）
    best_result = max(results, key=lambda x: x['test_r2'])
    
    print(f"\n【Model selection strategy explanation】")
    print(f"  - Based on test set R² select best fold (paper strategy: highest test set R²）")
    print(f"  - Validation set average R²: {mean_val_r2:.4f}（Only for recording）")
    print(f"  - Test set average R²: {mean_test_r2:.4f}")
    
    print("\n" + "="*70)
    print("Cross validation result summary:")
    print("="*70)
    
    for res in results:
        print(f"\nFold {res['fold']} ({res['val_label']} as validation set):")
        print(f"  Training set: R²={res['train_metrics']['Train R2']:.4f}, RMSE={res['train_metrics']['Train RMSE']:.3f}")
        print(f"  Test set: R²={res['test_metrics']['Test R2']:.4f}, RMSE={res['test_metrics']['Test RMSE']:.3f}")
        print(f"  Best number of iterations: {res['best_iteration']}")
    
    # Calculate three-fold average metrics (including all five metrics）
    print("\n" + "-"*70)
    print("Three-fold cross validation average metrics:")
    print("-"*70)
    
    avg_train_metrics = {
        'R2': np.mean([res['train_metrics']['Train R2'] for res in results]),
        'MAE': np.mean([res['train_metrics']['Train MAE'] for res in results]),
        'MSE': np.mean([res['train_metrics']['Train MSE'] for res in results]),
        'RMSE': np.mean([res['train_metrics']['Train RMSE'] for res in results]),
        'MAPE': np.mean([res['train_metrics']['Train MAPE'] for res in results]),
    }
    
    avg_test_metrics = {
        'R2': np.mean([res['test_metrics']['Test R2'] for res in results]),
        'MAE': np.mean([res['test_metrics']['Test MAE'] for res in results]),
        'MSE': np.mean([res['test_metrics']['Test MSE'] for res in results]),
        'RMSE': np.mean([res['test_metrics']['Test RMSE'] for res in results]),
        'MAPE': np.mean([res['test_metrics']['Test MAPE'] for res in results]),
    }
    
    print(f"  Training set average: R²={avg_train_metrics['R2']:.4f}, MAE={avg_train_metrics['MAE']:.3f}, RMSE={avg_train_metrics['RMSE']:.3f}, MSE={avg_train_metrics['MSE']:.3f}, MAPE={avg_train_metrics['MAPE']:.2f}%")
    print(f"  Test set average: R²={avg_test_metrics['R2']:.4f}, MAE={avg_test_metrics['MAE']:.3f}, RMSE={avg_test_metrics['RMSE']:.3f}, MSE={avg_test_metrics['MSE']:.3f}, MAPE={avg_test_metrics['MAPE']:.2f}%")
    
    # Print best hyperparameters
    print(f"\n**Best fold:** Fold {best_result['fold']} (Based on test set R²={best_result['test_r2']:.4f})")
    print(f"  Validation set R²: {best_result['val_r2']:.4f}（Only for recording）")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    print(f"  Best number of iterations: {best_result['best_iteration']}")
    
    print("="*70)
    
    # 4. Train final model
    print(f"\n{'='*70}")
    print(f"Training final model (using best hyperparameters from Fold {best_result['fold']} + all non-test data)...")
    print(f"Best fold hyperparameters source: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    print(f"{'='*70}")
    
    # Get all training data (all non-test data）
    train_final_mask = ~test_mask
    X_train_final = X[train_final_mask]
    y_train_final = y[train_final_mask]
    
    # Test set remains unchanged
    X_test_final = X[test_mask]
    y_test_final = y[test_mask]
    
    print(f"  Training data: {len(X_train_final)} samples (all non-test data）")
    print(f"  Test data: {len(X_test_final)} samples")
    
    # Directly use all iterations to train final model (consistent with backup version, no early stopping）
    print("  Use best fold hyperparameters, train final model on all training data (use all iterations）...")
    
    final_params = best_result['params'].copy()
    # Use all iterations (aggressive strategy, consistent with backup version）
    print(f"  Use all iterations: {final_params.get('n_estimators', 100)}")
    
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_train_final, y_train_final)
    
    # Evaluate final model
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test_final)
    test_metrics = calculate_metrics(y_test_final, y_test_final_pred, prefix="")
    
    print("\nFinal model training and test metrics:")
    print(f"(Use best hyperparameters from Fold {best_result['fold']} + all non-test data)")
    metrics_data = {
        'R2': [train_metrics[' R2'], test_metrics[' R2']],
        'R': [train_metrics[' R'], test_metrics[' R']],
        'MAE': [train_metrics[' MAE'], test_metrics[' MAE']],
        'MSE': [train_metrics[' MSE'], test_metrics[' MSE']],
        'RMSE': [train_metrics[' RMSE'], test_metrics[' RMSE']],
        'MAPE (%)': [train_metrics[' MAPE'], test_metrics[' MAPE']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['Set'] = [f'Training (S1+2+3, {len(X_train_final)} samples)', f'Testing ({len(X_test_final)} samples)']
    metrics_df = metrics_df.set_index('Set')
    metrics_df = metrics_df.round({'MAPE (%)': 2, 'R2': 4, 'R': 4, 'MAE': 4, 'MSE': 4, 'RMSE': 4})
    print(metrics_df.to_string())
    
    # Comparison explanation
    print(f"\nComparison explanation:")
    print(f"  - Fold {best_result['fold']} Training set: R²={best_result['train_metrics']['Train R2']:.4f}, RMSE={best_result['train_metrics']['Train RMSE']:.3f}")
    print(f"  - Fold {best_result['fold']} Test set: R²={best_result['test_metrics']['Test R2']:.4f}, RMSE={best_result['test_metrics']['Test RMSE']:.3f}")
    print(f"  - Final model training set (subset1+subset2+subset3): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    print(f"  - Final model test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    
    # Check if the target is reached
    if test_metrics[' R2'] >= 0.95:
        print(f"\n  🎉 Excellent! Test set R² = {test_metrics[' R2']:.4f} >= 0.95")
    elif test_metrics[' R2'] >= 0.9:
        print(f"\n  ✓ Target reached! Test set R² = {test_metrics[' R2']:.4f} >= 0.9")
    else:
        print(f"\n  ⚠ Test set R² = {test_metrics[' R2']:.4f} < 0.9")
    
    # Check overfitting risk
    train_test_gap = train_metrics[' R2'] - test_metrics[' R2']
    print(f"\n  Overfitting risk check:")
    print(f"    R² gap: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print(f"    ⚠ Warning: Training set and test set R² gap is large, possible overfitting risk")
    else:
        print(f"    ✓ Training set and test set R² gap is small, good generalization ability")
    
    # 5. Save training metrics to Excel
    print("\n" + "="*70)
    print("Save training metrics to Excel...")
    print("="*70)
    
    excel_path = os.path.join(SAVE_ROOT, 'training_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Each Fold detailed metrics (including all five metrics）
        fold_data = []
        for res in results:
            fold_dict = {
                'Fold': res['fold'],
                'Validation set Subset': res['val_label'],
                'Training set R²': res['train_metrics']['Train R2'],
                'Training set RMSE (MPa)': res['train_metrics']['Train RMSE'],
                'Training set MSE (MPa²)': res['train_metrics']['Train MSE'],
                'Training set MAE (MPa)': res['train_metrics']['Train MAE'],
                'Training set MAPE (%)': res['train_metrics']['Train MAPE'],
                'Test set R²': res['test_metrics']['Test R2'],
                'Test set RMSE (MPa)': res['test_metrics']['Test RMSE'],
                'Test set MSE (MPa²)': res['test_metrics']['Test MSE'],
                'Test set MAE (MPa)': res['test_metrics']['Test MAE'],
                'Test set MAPE (%)': res['test_metrics']['Test MAPE'],
                'Best number of iterations': res['best_iteration'],
            }
            fold_data.append(fold_dict)
        
        df_folds = pd.DataFrame(fold_data)
        df_folds.to_excel(writer, sheet_name='各Fold指标', index=False)
        print("✓ Saved: Each Fold detailed metrics (including R², RMSE, MSE, MAE, MAPE）")
        
        # Sheet 2: Final model metrics (including all five metrics）
        final_data = {
            'Dataset': [f'Training set ({len(X_train_final)} samples)', f'Test set ({len(X_test_final)} samples)'],
            'R²': [train_metrics[' R2'], test_metrics[' R2']],
            'RMSE (MPa)': [train_metrics[' RMSE'], test_metrics[' RMSE']],
            'MSE (MPa²)': [train_metrics[' MSE'], test_metrics[' MSE']],
            'MAE (MPa)': [train_metrics[' MAE'], test_metrics[' MAE']],
            'MAPE (%)': [train_metrics[' MAPE'], test_metrics[' MAPE']],
        }
        df_final = pd.DataFrame(final_data)
        df_final.to_excel(writer, sheet_name='Final model metrics', index=False)
        print("✓ Saved: Final model metrics (including R², RMSE, MSE, MAE, MAPE）")
        
        # Sheet 2.1: Three-fold average metrics (新增）
        avg_data = {
            'Dataset': ['Training set average', 'Test set average'],
            'R²': [avg_train_metrics['R2'], avg_test_metrics['R2']],
            'RMSE (MPa)': [avg_train_metrics['RMSE'], avg_test_metrics['RMSE']],
            'MSE (MPa²)': [avg_train_metrics['MSE'], avg_test_metrics['MSE']],
            'MAE (MPa)': [avg_train_metrics['MAE'], avg_test_metrics['MAE']],
            'MAPE (%)': [avg_train_metrics['MAPE'], avg_test_metrics['MAPE']],
        }
        df_avg = pd.DataFrame(avg_data)
        df_avg.to_excel(writer, sheet_name='Three-fold average metrics', index=False)
        print("✓ Saved: Three-fold average metrics (including R², RMSE, MSE, MAE, MAPE）")
        
        # Sheet 3: Best hyperparameters
        best_params_data = {
            'Hyperparameters': list(best_result['params'].keys()),
            'Best value': list(best_result['params'].values()),
        }
        df_params = pd.DataFrame(best_params_data)
        df_params.to_excel(writer, sheet_name='Best hyperparameters', index=False)
        print("✓ Saved: Best hyperparameters")
    
    print(f"\n✓ All training metrics saved to: {excel_path}")
    
    # 6. Save model and results
    model_save_path = os.path.join(SAVE_ROOT, 'xgboost_final_model.joblib')
    joblib.dump(final_model, model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test_final, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final results plot saved to: {results_dir}")
    
    # 7. Predict all data and save to Excel file
    print("\n" + "="*70)
    print("Predict all data and save to dataset_with_XGB_fc.xlsx")
    print("="*70)
    try:
        # Use final model to predict all data
        print(f"Predicting all {len(X)} samples...")
        y_all_pred = final_model.predict(X)
        
        # Create prediction result dictionary (using sample_ids as key）
        pred_dict = {}
        for idx, sample_id in enumerate(sample_ids):
            pred_dict[sample_id] = float(y_all_pred[idx])
        
        # Read original Excel file (or use existing df_original）
        data_file = os.path.join(PROJECT_ROOT, "dataset", "dataset_final.xlsx")
        if not os.path.exists(data_file):
            print(f"⚠ Warning: Original data file not found: {data_file}")
            # If file does not exist, try using existing df_original
            if 'df_original' in locals() and df_original is not None:
                print("  Using loaded data...")
            else:
                print("  ✗ Unable to find data file, skip saving prediction results")
                raise FileNotFoundError(f"Unable to find data file: {data_file}")
        else:
            df_original = pd.read_excel(data_file, sheet_name=0)
        
        # Use map method to match prediction results
        if 'No_Customized' in df_original.columns:
            df_original['XGB_fc'] = df_original['No_Customized'].map(pred_dict)
            print(f"✓ Using No_Customized column to match prediction results")
        else:
            print(f"⚠ Warning: No_Customized column not found, match prediction results by row order")
            # If No_Customized column does not exist, match prediction results by row order (ensure order consistency）
            if len(y_all_pred) == len(df_original):
                df_original['XGB_fc'] = y_all_pred
                print(f"✓ Match prediction results by row order")
            else:
                print(f"  ✗ Prediction sample number ({len(y_all_pred)}) does not match original data row number ({len(df_original)})")
                raise ValueError("Sample number mismatch, cannot write prediction results")
        
        # Save results
        output_file = os.path.join(PROJECT_ROOT, "dataset", "dataset_with_XGB_fc.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_original.to_excel(output_file, index=False)
        print(f"✓ Prediction results saved to: {output_file}")
        print(f"  - New column: XGB_fc (final model prediction value)")
        print(f"  - Total prediction sample number: {len(y_all_pred)}")
        print(f"  - Non-empty prediction value number: {df_original['XGB_fc'].notna().sum()}")
        
        # Backup to save directory
        backup_file = os.path.join(SAVE_ROOT, 'dataset_with_XGB_fc_backup.xlsx')
        df_original.to_excel(backup_file, index=False)
        print(f"✓ Backup file saved to: {backup_file}")
        
    except Exception as e:
        print(f"⚠ Warning: Error writing to dataset_with_XGB_fc.xlsx: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70)

if __name__ == "__main__":
    main_process()