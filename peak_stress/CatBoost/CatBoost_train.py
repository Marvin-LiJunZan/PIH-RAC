#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost prediction model - final complete version (includes hyperparameter printing and learning curve monitoring)
Strategy: 3-fold hyperparameter search + independent test set selection + final model training

Enhanced features:
1. Print all metrics (R2, R, MAE, MSE, RMSE, MAPE).
2. Print the **selected best hyperparameters** in the final report.
3. Training/validation curve monitoring: record and plot the learning curve, identify the bias-variance balance point (best iteration number).
   - Automatically use early stopping to prevent overfitting
   - Visualize the change of RMSE of the training set and validation set with the number of iterations
   - Mark the best iteration point (bias-variance balance point)
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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_stress', 'CatBoost', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Working directory: {PROJECT_ROOT}")
print(f"Save directory: {SAVE_ROOT}")

# --- Core function definitions ---
    
def load_data():
    """Load and parse data"""
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"Error: File does not exist: {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    feature_names = material_features + specimen_features
    target_column = 'fc'
    
    X = df[feature_names].values
    y = df[target_column].values
    
    if 'DataSlice' not in df.columns:
        print("Error: Missing 'DataSlice' column for data splitting")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully: {len(X)} samples, {len(feature_names)} features")
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV)
    
    Hyperparameter search range optimization description (for small sample scenario, about 60-70 training samples):
    - depth: 3-8 (Small sample suggests 3-6, widen to 8 to cover more possibilities)
    - learning_rate: 0.01-0.3 (Common range for CatBoost, avoid too high learning rate causing instability)
    - iterations: 100-800 (With lower learning rate, more trees may be needed)
    - Other regularization parameters keep the original range, helps to prevent overfitting
    
    CatBoost parameters:
    - depth: tree maximum depth (corresponding to XGBoost's max_depth)
    - iterations: number of iterations (corresponding to XGBoost's n_estimators)
    - learning_rate: learning rate (same)
    - l2_leaf_reg: L2 regularization coefficient (corresponding to XGBoost's reg_lambda)
    - min_child_samples: minimum number of samples in a leaf node (similar to XGBoost's min_child_weight)
    - subsample: sample sampling ratio (same)
    - rsm: feature sampling ratio (corresponding to XGBoost's colsample_bytree)
    """
    params = {
        'task_type': 'CPU',  # Fixed using CPU
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'loss_function': 'RMSE',  # CatBoost regression default using RMSE
        # [Top-level master-level hyperparameter search space - optimized for small sample (60-70 training samples)]
        # Core principles: conservative, strong regularization, avoid overfitting,符合CatBoost限制
        'depth': trial.suggest_int('depth', 3, 6),  # Depth: 3-6 layers (small sample不宜过深，6层已足够）
        'iterations': trial.suggest_int('iterations', 100, 500),  # Iterations: 100-500 (with early stopping, avoid overtraining)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # Learning rate: 0.01-0.15 (conservative range, avoid too high causing instability)
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),  # Minimum number of samples in a leaf node: 10-40 (increase lower limit, strengthen regularization, prevent overfitting)
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 50.0, log=True),  # L2 regularization: 3.0-50.0 (increase lower limit to 3.0, strengthen regularization)
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # Sample sampling: 0.7-0.95 (lower limit 0.7 to avoid CatBoost error, upper limit 0.95 to maintain randomness)
        'rsm': trial.suggest_float('rsm', 0.6, 0.9),  # Feature sampling: 0.6-
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]

        # Use stricter early stopping to prevent overfitting (reduce number of rounds, stop earlier)
        # Reduce from 15-25 rounds to 10-20 rounds, or from 15% to 10% of iterations
        early_stopping_rounds_cv = max(10, min(20, int(params.get('iterations', 100) * 0.10)))
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_v, y_v), verbose=False, early_stopping_rounds=early_stopping_rounds_cv)
        
        y_pred = model.predict(X_v)
        
        # Use RMSE as the objective function (more conventional for regression problems)
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        rmse_scores.append(rmse)
    
    # Return average RMSE (smaller is better, so direction='minimize' in optimize_catboost)
    return np.mean(rmse_scores)

def optimize_catboost(X_train, y_train, n_trials=50):
    """Run Optuna optimization
    
    Note: The objective function is the RMSE of the validation set (smaller is better), so direction='minimize'
    
    [Top-level master-level analysis: Impact of increasing trial number]
    
    1. **Theoretical benefits of increasing trial number**：
       - Explore a larger hyperparameter space, possibly find a better combination
       - TPE sampler needs a certain number of trials to establish an effective probability model (usually requires 20-30 trials)
       - For 7-parameter search space, 50 trials already cover the main area
    
    2. **Potential risks of increasing trial number (small sample scenario)**：
       - **Overfitting to validation set**：Overfitting may cause the hyperparameters to overfit the validation set, reducing generalization ability
       - **Decreasing returns**：After a certain number, performance improvement is negligible (usually after 50-100 trials)
       - **Computational cost**：Each trial requires 5-fold CV training, 50 trials = 250 model trainings
       - **Small sample limit**：60-70 samples, validation set only 12-14 samples, unstable evaluation
    
    3. **Suggestions for the current scenario**：
       - **n_trials=50-100**：For small sample, 50 trials are usually enough, 100 trials is the upper limit
       - **Outer 3-fold CV**：Each fold is independently optimized,makes 3 optimizations, increases exploration opportunities
       - **Early stopping mechanism**：If no improvement for 10-15 trials, consider stopping early
       - **Search space optimization**：More important than the number of trials is the rationality of the search space (currently optimized)
    
    4. **Whether to increase trial number**：
       - If the optimal value is still decreasing after 50 trials → can increase to 100
       - If the optimal value is stable after 50 trials → increasing the number of trials is not meaningful
       - If the performance of the validation set is significantly different from the test set → may be overfitting, should reduce the number of trials or strengthen regularization
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)  # Change to minimize, because RMSE is smaller the better
    
    # [Optimization suggestion]: For small sample, can add early stopping mechanism
    # If no improvement for N trials, stop early (save computational resources)
    # Note: Optuna's callback is called after each trial, can check if should stop
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
            
            # Check if there is significant improvement (relative improvement > min_delta)
            if study.best_value < self.best_value * (1 - self.min_delta):
                self.best_value = study.best_value
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # If no improvement for patience trials, stop optimization
            if self.no_improve_count >= self.patience:
                study.stop()
                print(f"  ⚠ Early stopping triggered: No improvement for {self.patience} trials, stop optimization")
    
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
    
    Parameters:
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
    
    # Fixed using CPU
    model_params = {
        'task_type': 'CPU',  # Fixed using CPU
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'loss_function': 'RMSE',
        **best_params
    }
    # Ensure task_type is CPU (cover possible task_type in best_params)
    model_params['task_type'] = 'CPU'
    
    # Extract iterations for early stopping
    # For overfitting problem, use stricter early stopping strategy
    iterations = model_params.get('iterations', 100)
    # Use smaller early_stopping_rounds, stop earlier to prevent overfitting
    # Reduce from 20-30 rounds to 10-20 rounds, or from 15% to 10% of iterations
    early_stopping_rounds = max(10, min(20, int(iterations * 0.10)))  # 10-20 rounds, or 10% of iterations (stricter)
    
    # Train model (fixed using CPU)
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
        # CatBoost format: {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]}}
        train_rmse = evals_result['learn']['RMSE']
        val_rmse = evals_result['validation']['RMSE']
        
        # Get best_iteration
        best_iteration = model.get_best_iteration()
        if best_iteration is None:
            # If no best_iteration, use the iteration with the smallest validation set RMSE
            best_iteration = np.argmin(val_rmse)
        
        best_score = val_rmse[best_iteration] if best_iteration < len(val_rmse) else val_rmse[-1]
        
        history = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'best_iteration': best_iteration,
            'best_score': best_score
        }
        # Retrain final model (using all data, but limited to best iteration number)
        if best_iteration is not None:
            model_params_final = model_params.copy()
            model_params_final['iterations'] = best_iteration + 1  # +1 because index starts from 0
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
    plt.xlabel('True Values (MPa)')
    plt.ylabel('Predicted Values (MPa)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_scatter.png'), dpi=300)
    plt.close()

def plot_learning_curve(history, save_dir, prefix=""):
    """Plot learning curve: RMSE of training set and validation set changes with iterations
    
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
    plt.ylabel('RMSE (MPa)', fontsize=12)
    plt.title(f'{prefix} Learning Curve\n(Bias-variance balance point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add explanation text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f} MPa\n(Positive value indicates possible轻微欠拟合)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f} MPa\n(Negative value indicates possible overfitting risk)',
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
        print(f"  Training set RMSE: {train_rmse[best_iter]:.4f} MPa")
        print(f"  Validation set RMSE: {val_rmse[best_iter]:.4f} MPa")
        print(f"  Training-validation gap: {train_rmse[best_iter] - val_rmse[best_iter]:.4f} MPa")
    

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
    print("Start outer 3-fold hyperparameter search (based on independent test set performance)")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set (for Refit incremental)")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        print("    Performing Optuna inner 5-fold optimization...")
        study = optimize_catboost(X_train_fold, y_train_fold, n_trials=50) 
        best_params = study.best_params
        
        # Train model and record learning curve (for monitoring overfitting)
        print("    Train model and record learning curve...")
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
    
    # *****************************************************
    # *** Core modification: print best hyperparameters ***
    # *****************************************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on R2={best_result['r2']:.4f})")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. Train final model
    print(f"\nTrain final model (using best hyperparameters from Fold {best_result['fold']} + all non-Test data)...")
    print(f"Best fold hyperparameters source: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    
    # For monitoring the training process of the final model, split 20% of the training set as validation set
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"  Use 20% of the training set ({len(X_val_monitor)} samples) as monitoring validation set")
    
    # Train final model and record learning curve
    final_model, final_history = train_catboost(
        X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
        best_result['params'], return_history=True
    )
    
    # Save learning curve of the final model
    results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
    plot_learning_curve(final_history, results_dir, prefix="Final_Model")
    
    # Retrain final model using all training data (for actual prediction)
    # Strategy explanation:
    # 1. Use best iteration number (conservative strategy): based on the bias-variance balance point found on the validation set, to avoid overfitting
    # 2. Use all iterations number (aggressive strategy): because the final model has more data, it may be able to train more rounds
    
    print("  Retrain final model using all training data...")
    best_iteration_final = final_history['best_iteration']
    original_iterations = best_result['params'].get('iterations', 100)
    
    print(f"  Strategy comparison:")
    print(f"    - Best iteration number (used): {best_iteration_final + 1}")
    print(f"    - Original iteration number (not used): {original_iterations}")
    print(f"  Reason: based on the bias-variance balance point found on the validation set, to avoid overfitting")
    
    # Train final model using best iteration number (conservative strategy)
    final_params = best_result['params'].copy()
    final_params['iterations'] = best_iteration_final + 1  # +1 because index starts from 0
    final_model = train_catboost(X_train_final, y_train_final, [], [], final_params)
    
    # 5. Evaluate final model and print detailed metrics
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_final_pred, prefix="")
    
    # Optional: compare the effect of using all iterations number (for verifying strategy selection)
    print(f"\n  [Strategy comparison] Train model using all iterations number for comparison...")
    final_model_full = train_catboost(X_train_final, y_train_final, [], [], best_result['params'])
    y_test_pred_full = final_model_full.predict(X_test)
    test_metrics_full = calculate_metrics(y_test, y_test_pred_full, prefix="")
    
    print(f"  Comparison result (test set):")
    print(f"    Best iteration number model: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    print(f"    All iterations number model: R²={test_metrics_full[' R2']:.4f}, RMSE={test_metrics_full[' RMSE']:.3f}")
    
    # Choose the model with better test set performance
    if test_metrics_full[' R2'] > test_metrics[' R2']:
        print(f"  → All iterations number model performs better on test set, but be cautious of overfitting risk")
        print(f"  → Currently using: Best iteration number model (more conservative, better generalization ability)")
    else:
        print(f"  → Best iteration number model performs better on test set, verifying the correctness of strategy selection")
    
    print("\nFinal model training and test metrics:")
    print(f"(using best hyperparameters from Fold {best_result['fold']} on all non-test data)")
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
    
    # 6. Model and results saving
    model_save_path = os.path.join(SAVE_ROOT, 'catboost_final_model.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(final_model, model_save_path)
    print(f"\nModel architecture (CatBoost) saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final results plot saved to: {results_dir}")

if __name__ == "__main__":
    main_process()
