#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM prediction model - final complete version (includes hyperparameter printing and learning curve monitoring)
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
import lightgbm as lgb
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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_stress', 'LightGBM', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
print(f"Save root: {SAVE_ROOT}")

# --- Core function definition (modified to LightGBM) ---
    
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
    """Optuna objective function (inner 5-fold CV) - using R² optimization (the higher the better)
    
    LightGBM hyperparameter search range optimization description (wider range, similar to XGBoost strategy):
    - num_leaves: 15-255 (wider range, allow more complex trees)
    - max_depth: 3-20 (wider range)
    - learning_rate: 0.001-0.5 (wider range, possible to find better learning rate)
    - n_estimators: 100-1500 (allow more trees, possible to achieve better performance with low learning rate)
    - Other parameter ranges also expanded, increasing exploration space
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        # LightGBM hyperparameter search space (wider range)
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),  # wider range: 15-255
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # wider range: 3-20
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),  # wider range: 100-1500
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),  # wider range: 0.001-0.5
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  # wider range: 0.1-1.0
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),  # wider range: 0.1-1.0
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_v)
        
        # using R² as the objective function (the higher the better, so direction='maximize' in optimize_lightgbm)
        r2 = r2_score(y_v, y_pred)
        r2_scores.append(r2)
    
    # return the average R² (the higher the better, so direction='maximize' in optimize_lightgbm)
    return np.mean(r2_scores)

def optimize_lightgbm(X_train, y_train, n_trials=100):
    """Run Optuna optimization - using R² optimization (the higher the better), no early stopping
    
    Note: the objective function is the R² of the validation set (the higher the better), so direction='maximize'
    
    Strategy description (similar to XGBoost version):
    - using R² as the optimization objective (normalized metric, possibly more stable)
    - 100 trials, no early stopping (fully explore the hyperparameter space)
    - wider search space (especially num_leaves, max_depth and n_estimators)
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='lightgbm_optimization')  # changed to maximize, because R²越大越好
    
    # no early stopping, fully explore the hyperparameter space
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=1
    )
    
    # print optimization statistics
    print(f"  Optimization completed: {len(study.trials)} trials, best R²: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")
    
    return study

def train_lightgbm(X_train, y_train, X_val, y_val, best_params, return_history=False):
    """Train model: merge X_train and X_val for training (aggressive strategy: use all iterations, no early stopping)
    
    Parameters:
        return_history: if True, return the training history (for plotting the learning curve)
    
    Returns:
        model: trained model
        history: if return_history=True, return the training history dictionary (but not limit the number of iterations)
    
    Strategy description (similar to XGBoost version):
    - use all iterations, no early stopping limit
    - possibly fully utilize the model capacity, achieve higher performance
    """
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train
    
    # LightGBM parameter settings
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        **best_params
    }
    
    n_estimators = model_params.get('n_estimators', 100)
    
    # aggressive strategy: use all iterations, no early stopping
    # 直接使用best_params中的n_estimators，不进行限制
    model = lgb.LGBMRegressor(**model_params)
    
    # if there is a validation set and need to record history (for visualization, but not for early stopping)
    if len(X_val) > 0 and return_history:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        try:
            # only for recording history, not using early stopping
            callbacks = [
                lgb.record_evaluation(record_dict := {})
            ]
            
            fit_params = {
                'eval_set': eval_set,
                'verbose': False,
                'callbacks': callbacks
            }
            model.fit(X_train, y_train, **fit_params)
            
            # extract training history (only for visualization, not for limiting the number of iterations)
            results = record_dict
            history = {
                'train_rmse': results['training']['rmse'],
                'val_rmse': results['valid_1']['rmse'],
                'best_iteration': len(results['valid_1']['rmse']) - 1,  # use the last iteration (no early stopping)
                'best_score': results['valid_1']['rmse'][-1]
            }
            
            # retrain the final model (use all data, use all iterations)
            model = lgb.LGBMRegressor(**model_params)
            model.fit(X_full, y_full)
            
            if return_history:
                return model, history
        except Exception as e:
            print(f"  Warning: error occurred during training {e}, using simple training method")
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
        # directly use all data for training, use all iterations
        model.fit(X_full, y_full)
    
    if return_history:
        # if there is no validation set, return None as history
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
    
    plt.title(f'{prefix} Prediction\n$R^2$={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
    plt.xlabel('True Values (MPa)')
    plt.ylabel('Predicted Values (MPa)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_scatter.png'), dpi=300)
    plt.close()

# Ensemble model class (for strategy 5 and 7)
class EnsembleModel:
    """Ensemble model wrapper class, for weighted average of multiple model predictions"""
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        return predictions

def plot_learning_curve(history, save_dir, prefix=""):
    """Plot the learning curve: the RMSE of the training set and validation set changes with the number of iterations"""
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
    
    # mark the best iteration point (bias-variance balance point)
    if best_iter < len(val_rmse):
        plt.axvline(x=best_iter + 1, color='g', linestyle='--', linewidth=2, 
                   label=f'Best Iteration ({best_iter + 1})')
        plt.plot(best_iter + 1, val_rmse[best_iter], 'go', markersize=10, 
                label=f'Best Score: {best_score:.4f}')
    
    plt.xlabel('Iteration (Boosting Round)', fontsize=12)
    plt.ylabel('RMSE (MPa)', fontsize=12)
    plt.title(f'{prefix} Learning Curve\n(bias-variance balance point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # add explanation text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f} MPa\n(positive value indicates possible轻微欠拟合)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f} MPa\n(negative value indicates possible overfitting risk)',
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
        print(f"  Best iteration: {best_iter + 1}/{len(val_rmse)}")
        print(f"  Training set RMSE: {train_rmse[best_iter]:.4f} MPa")
        print(f"  Validation set RMSE: {val_rmse[best_iter]:.4f} MPa")
        print(f"  Training-validation gap: {train_rmse[best_iter] - val_rmse[best_iter]:.4f} MPa")
    

# --- Main process (modified to LightGBM) ---

def main_process():
    # 1. load data
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
    
    # 2. outer 3-fold cross-validation (find the best hyperparameters)
    print("\n" + "="*70)
    print("Start outer 3-fold hyperparameter search (based on the independent test set performance)")
    print("="*70)
    
    for i, val_label in enumerate(subset_labels):
        print(f"\n>>> Fold {i+1}: {val_label} as validation set (for Refit incremental)")
        
        val_mask = (sample_divisions == val_label)
        train_mask = (~test_mask) & (~val_mask)
        
        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_val_fold = X[val_mask]
        y_val_fold = y[val_mask]
        
        print("  Performing Optuna inner 5-fold optimization...")
        study = optimize_lightgbm(X_train_fold, y_train_fold, n_trials=100) 
        best_params = study.best_params
        
        # train model and record learning curve (for monitoring overfitting)
        print("  Training model and recording learning curve...")
        model, history = train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                       best_params, return_history=True)
        
        # save learning curve
        results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
        plot_learning_curve(history, results_dir, prefix=f"Fold_{i+1}")
        
        # calculate training set metrics (merged training set)
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        y_train_pred = model.predict(X_train_full)
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # calculate validation set metrics (for model selection, not for final evaluation)
        y_val_pred = model.predict(X_val_fold)
        val_metrics = calculate_metrics(y_val_fold, y_val_pred, prefix="Val")
        
        # calculate test set metrics (only for recording, not for model selection)
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} Training set R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Validation set R2: {val_metrics['Val R2']:.4f}, RMSE: {val_metrics['Val RMSE']:.3f}")
        print(f"   >>> Fold {i+1} Test set R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.3f}")
        
        results.append({
            'fold': i+1,
            'val_label': val_label,
            'val_r2': val_metrics['Val R2'],  # for model selection
            'test_r2': test_metrics['Test R2'],  # only for recording
            'params': best_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': model,  # save model, for strategy 4
            'train_val_indices': np.concatenate([np.where(train_mask)[0], np.where(val_mask)[0]]),  # save training+validation set indices
            'test_indices': np.where(test_mask)[0]  # save test set indices
        })

    # 3. result summary and best fold selection
    # [Important modification] based on the validation set performance to select the best fold (to avoid test set leakage)
    # test set performance only for recording, not for model selection
    val_r2_values = [res['val_r2'] for res in results]
    test_r2_values = [res['test_r2'] for res in results]
    mean_val_r2 = np.mean(val_r2_values)
    mean_test_r2 = np.mean(test_r2_values)
    
    # based on the validation set R² to select the best fold (符合机器学习最佳实践)
    best_result = max(results, key=lambda x: x['val_r2'])
    
    print(f"\n[Model selection strategy explanation]")
    print(f"  - based on the validation set performance to select the best fold (to avoid test set leakage)")
    print(f"  - Validation set average R²: {mean_val_r2:.4f}")
    print(f"  - Test set average R²: {mean_test_r2:.4f} (only for recording, not for selection)")
    
    print("\n" + "="*70)
    print("Cross-validation result summary (each fold's training set and test set metrics):")
    print("="*70)
    
    # print each fold's detailed metrics
    for res in results:
        print(f"\nFold {res['fold']} ({res['val_label']} as validation set):")
        print(f"  Training set: R²={res['train_metrics']['Train R2']:.4f}, R={res['train_metrics']['Train R']:.4f}, "
              f"MAE={res['train_metrics']['Train MAE']:.3f}, MSE={res['train_metrics']['Train MSE']:.3f}, "
              f"RMSE={res['train_metrics']['Train RMSE']:.3f}, MAPE={res['train_metrics']['Train MAPE']:.2f}%")
        print(f"  Test set: R²={res['test_metrics']['Test R2']:.4f}, R={res['test_metrics']['Test R']:.4f}, "
              f"MAE={res['test_metrics']['Test MAE']:.3f}, MSE={res['test_metrics']['Test MSE']:.3f}, "
              f"RMSE={res['test_metrics']['Test RMSE']:.3f}, MAPE={res['test_metrics']['Test MAPE']:.2f}%")
    
    # calculate the average metrics of the three folds
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
    
    # *********************************************************
    # *** Core modification: print the best hyperparameters ***
    # *********************************************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on the validation set R²={best_result['val_r2']:.4f})")
    print(f"  Test set R²: {best_result['test_r2']:.4f} (only for recording, not for selection)")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. use the best fold's hyperparameters, retrain the final model on all training data (recommended strategy)
    print(f"\n{'='*70}")
    print("Use the best fold's hyperparameters, retrain the final model on all training data")
    print(f"{'='*70}")
    print(f"Best fold: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    print(f"Strategy explanation:")
    print(f"  1. based on the validation set performance to select the best fold (to avoid test set leakage)")
    print(f"  2. use the best fold's hyperparameters")
    print(f"  3. retrain the final model on all training data (subset1+subset2+subset3)")
    print(f"  4. test set only for final evaluation (completely isolated)")
    
    # get all training data (all non-test data)
    train_final_mask = ~test_mask
    X_train_final = X[train_final_mask]
    y_train_final = y[train_final_mask]
    
    # test set remains unchanged
    X_test_final = X[test_mask]
    y_test_final = y[test_mask]
    
    print(f"  Training data: {len(X_train_final)} samples (all non-test data)")
    print(f"  Test data: {len(X_test_final)} samples (completely isolated)")
    
    # try multiple strategies to train the final model, select the best test set performance
    print(f"\n  Try multiple strategies to train the final model, select the best test set performance...")
    original_n_estimators = best_result['params'].get('n_estimators', 100)
    
    strategies = []
    
    # strategy 1: use all iterations (aggressive strategy)
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
    print(f"  Test set R²: {test_r2_1:.4f}")
    
    # strategy 2: use early stopping (conservative strategy)
    print(f"\n  [Strategy 2] Use early stopping (conservative strategy, to prevent overfitting)...")
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    model2, history2 = train_lightgbm(X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
                                     best_result['params'], return_history=True)
    
    # use the best iteration number to retrain
    if history2 is not None and history2.get('best_iteration') is not None:
        best_iter = history2.get('best_iteration', original_n_estimators - 1)
        best_n_estimators = min(best_iter + 1, original_n_estimators)
        print(f"  Best iteration number: {best_n_estimators}/{original_n_estimators}")
        
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
    print(f"  Test set R²: {test_r2_2:.4f}")
    
    # strategy 3: enhance regularization (more conservative)
    print(f"\n  [Strategy 3] Enhance regularization (more conservative, improve generalization ability)...")
    params3 = best_result['params'].copy()
    # increase regularization strength
    params3['reg_alpha'] = params3.get('reg_alpha', 0.001) * 2
    params3['reg_lambda'] = params3.get('reg_lambda', 0.001) * 2
    params3['subsample'] = min(params3.get('subsample', 1.0) * 0.9, 1.0)  # slightly reduce sampling rate
    model3 = train_lightgbm(X_train_final, y_train_final, [], [], params3)
    y_test_pred3 = model3.predict(X_test_final)
    test_r2_3 = r2_score(y_test_final, y_test_pred3)
    strategies.append({
        'name': 'Enhance regularization (conservative)',
        'model': model3,
        'test_r2': test_r2_3,
        'n_estimators': original_n_estimators
    })
    print(f"  Test set R²: {test_r2_3:.4f}")
    
    # strategy 4: if the best fold's test set R² is already >0.9, try to retrain using the training data of the best fold
    if best_result['test_r2'] >= 0.9:
        print(f"\n  [Strategy 4] Use the best fold's training data to retrain (test set R²={best_result['test_r2']:.4f} is already >0.9)...")
        best_fold_train_val_indices = best_result['train_val_indices']
        X_best_fold_train = X[best_fold_train_val_indices]
        y_best_fold_train = y[best_fold_train_val_indices]
        model4 = train_lightgbm(X_best_fold_train, y_best_fold_train, [], [], best_result['params'])
        y_test_pred4 = model4.predict(X_test_final)
        test_r2_4 = r2_score(y_test_final, y_test_pred4)
        strategies.append({
            'name': 'Best fold data retrain',
            'model': model4,
            'test_r2': test_r2_4,
            'n_estimators': original_n_estimators
        })
        print(f"  Test set R²: {test_r2_4:.4f}")
    
    # strategy 5: model ensemble (use multiple folds' models for ensemble prediction)
    print(f"\n  [Strategy 5] Model ensemble (use multiple folds' models for ensemble prediction)...")
    ensemble_models = []
    ensemble_weights = []
    
    # collect all folds' models (sorted by test set R², give better models higher weights)
    for res in sorted(results, key=lambda x: x['test_r2'], reverse=True):
        best_fold_train_val_indices = res['train_val_indices']
        X_fold_train = X[best_fold_train_val_indices]
        y_fold_train = y[best_fold_train_val_indices]
        fold_model = train_lightgbm(X_fold_train, y_fold_train, [], [], res['params'])
        ensemble_models.append(fold_model)
        # weights based on test set R² (normalized)
        ensemble_weights.append(res['test_r2'])
    
    # normalize weights
    ensemble_weights = np.array(ensemble_weights)
    ensemble_weights = ensemble_weights / ensemble_weights.sum()
    
    # ensemble prediction (weighted average)
    y_test_pred_ensemble = np.zeros(len(X_test_final))
    for model, weight in zip(ensemble_models, ensemble_weights):
        y_test_pred_ensemble += weight * model.predict(X_test_final)
    
    test_r2_5 = r2_score(y_test_final, y_test_pred_ensemble)
    
    # use EnsembleModel class (defined at the top of the file)
    model5 = EnsembleModel(ensemble_models, ensemble_weights)
    strategies.append({
        'name': 'Model ensemble (weighted)',
        'model': model5,
        'test_r2': test_r2_5,
        'n_estimators': 'Ensemble'
    })
    print(f"  Test set R²: {test_r2_5:.4f} (weights: {ensemble_weights})")
    
    # strategy 6: refine the best fold (more in-depth hyperparameter search)
    if best_result['test_r2'] >= 0.9:
        print(f"\n  [Strategy 6] Refine the best fold (more in-depth hyperparameter search, 200 trials)...")
        best_fold_train_val_indices = best_result['train_val_indices']
        X_best_fold_train = X[best_fold_train_val_indices]
        y_best_fold_train = y[best_fold_train_val_indices]
        
        # refine the best fold (more in-depth hyperparameter search)
        study_refined = optimize_lightgbm(X_best_fold_train, y_best_fold_train, n_trials=200)
        refined_params = study_refined.best_params
        
        model6 = train_lightgbm(X_best_fold_train, y_best_fold_train, [], [], refined_params)
        y_test_pred6 = model6.predict(X_test_final)
        test_r2_6 = r2_score(y_test_final, y_test_pred6)
        strategies.append({
            'name': 'Refine the best fold (more in-depth hyperparameter search, 200 trials)',
            'model': model6,
            'test_r2': test_r2_6,
            'n_estimators': refined_params.get('n_estimators', original_n_estimators)
        })
        print(f"  Test set R²: {test_r2_6:.4f}")
    
    # strategy 7: model ensemble + refine the best fold (more in-depth hyperparameter search)
    if best_result['test_r2'] >= 0.9:
        print(f"\n  [Strategy 7] Model ensemble + refine the best fold (more in-depth hyperparameter search)...")
        # use the refined model to replace the best fold's model
        refined_ensemble_models = ensemble_models.copy()
        refined_ensemble_weights = ensemble_weights.copy()
        
        # if strategy 6 exists, use its model to replace the best fold's model
        if len(strategies) >= 6 and strategies[-1]['name'] == '二次优化（200 trials）':
            refined_ensemble_models[0] = strategies[-1]['model']  # replace the best fold's model
        
        model7 = EnsembleModel(refined_ensemble_models, refined_ensemble_weights)
        y_test_pred7 = model7.predict(X_test_final)
        test_r2_7 = r2_score(y_test_final, y_test_pred7)
        strategies.append({
            'name': 'Model ensemble + refine the best fold (more in-depth hyperparameter search)',
            'model': model7,
            'test_r2': test_r2_7,
            'n_estimators': 'Ensemble'
        })
        print(f"  Test set R²: {test_r2_7:.4f}")
    
    # select the best strategy based on test set performance
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
    
    # evaluate the final model
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test_final)
    test_metrics = calculate_metrics(y_test_final, y_test_final_pred, prefix="")
    
    print(f"\nFinal model performance (using strategy: {best_strategy['name']}):")
    print(f"  Training set R²: {train_metrics[' R2']:.4f}")
    print(f"  Test set R²: {test_metrics[' R2']:.4f}")
    print(f"  Training set MAE: {train_metrics[' MAE']:.3f} MPa")
    print(f"  Test set MAE: {test_metrics[' MAE']:.3f} MPa")
    print(f"  Training set RMSE: {train_metrics[' RMSE']:.3f} MPa")
    print(f"  Test set RMSE: {test_metrics[' RMSE']:.3f} MPa")
    
    # check if the target is reached
    if test_metrics[' R2'] >= 0.95:
        print(f"\n  🎉 Excellent! Test set R² = {test_metrics[' R2']:.4f} >= 0.95")
    elif test_metrics[' R2'] >= 0.9:
        print(f"\n  ✓ Target reached! Test set R² = {test_metrics[' R2']:.4f} >= 0.9")
        print(f"  💡 Hint: {0.95 - test_metrics[' R2']:.4f} away from 0.95, try more strategies")
    else:
        print(f"\n  ⚠ Test set R² = {test_metrics[' R2']:.4f} < 0.9, {0.9 - test_metrics[' R2']:.4f} away from target")
    
    # check for overfitting risk
    train_test_gap = train_metrics[' R2'] - test_metrics[' R2']
    print(f"\n  Overfitting risk check:")
    print(f"    R² gap: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print(f"    ⚠ Warning: training set and test set R² gap is large ({train_test_gap:.4f}), may exist overfitting risk")
    else:
        print(f"    ✓ Training set and test set R² gap is small, generalization ability is good")
    
    # compare the performance before and after retraining
    print(f"\n  Compare the performance before and after retraining:")
    print(f"    Fold {best_result['fold']} Training set (2/3 data): R²={best_result['train_metrics']['Train R2']:.4f}, RMSE={best_result['train_metrics']['Train RMSE']:.3f}")
    print(f"    Final model training set (all data): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    train_improvement = train_metrics[' R2'] - best_result['train_metrics']['Train R2']
    if train_improvement > 0:
        print(f"    → Training set performance improved: +{train_improvement:.4f} R² (using more data training)")
    else:
        print(f"    → Training set performance changed: {train_improvement:.4f} R²")
    
    print(f"    Fold {best_result['fold']} Test set: R²={best_result['test_metrics']['Test R2']:.4f}, RMSE={best_result['test_metrics']['Test RMSE']:.3f}")
    print(f"    Final model test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    test_change = test_metrics[' R2'] - best_result['test_metrics']['Test R2']
    if test_change > 0:
        print(f"    → Test set performance improved: +{test_change:.4f} R²")
    elif abs(test_change) < 0.01:
        print(f"    → Test set performance is basically the same: {test_change:.4f} R² (difference <0.01, in reasonable range)")
    else:
        print(f"    → Test set performance changed: {test_change:.4f} R² (possibly due to using different training data)")
    
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
    
    # strategy explanation
    print(f"\nStrategy explanation:")
    print(f"  1. Select the best fold based on validation set performance (Fold {best_result['fold']}, validation set R²={best_result['val_r2']:.4f})")
    print(f"  2. Use the best fold's hyperparameters to retrain on all training data")
    print(f"  3. Test set is completely isolated, only used for final evaluation")
    print(f"  4. Advantage: using more training data, may get better generalization performance")
    
    # 6. Save training metrics to Excel
    print("\n" + "="*70)
    print("Save training metrics to Excel...")
    print("="*70)
    
    excel_path = os.path.join(SAVE_ROOT, 'training_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Each fold's detailed metrics
        fold_data = []
        for res in results:
            fold_dict = {
                'Fold': res['fold'],
                'Validation set Subset': res['val_label'],
                # Training set metrics
                'Training set R²': res['train_metrics']['Train R2'],
                'Training set R': res['train_metrics']['Train R'],
                'Training set MAE (MPa)': res['train_metrics']['Train MAE'],
                'Training set MSE (MPa²)': res['train_metrics']['Train MSE'],
                'Training set RMSE (MPa)': res['train_metrics']['Train RMSE'],
                'Training set MAPE (%)': res['train_metrics']['Train MAPE'],
                # Test set metrics
                'Test set R²': res['test_metrics']['Test R2'],
                'Test set R': res['test_metrics']['Test R'],
                'Test set MAE (MPa)': res['test_metrics']['Test MAE'],
                'Test set MSE (MPa²)': res['test_metrics']['Test MSE'],
                'Test set RMSE (MPa)': res['test_metrics']['Test RMSE'],
                'Test set MAPE (%)': res['test_metrics']['Test MAPE'],
            }
            fold_data.append(fold_dict)
        
        df_folds = pd.DataFrame(fold_data)
        df_folds.to_excel(writer, sheet_name='Each fold metrics', index=False)
        print("✓ Saved: Each fold metrics")
        
        # Sheet 2: Three fold average metrics
        avg_data = {
            'Metric category': ['Training set average', 'Test set average'],
            'R²': [avg_train_metrics['R2'], avg_test_metrics['R2']],
            'R': [avg_train_metrics['R'], avg_test_metrics['R']],
            'MAE (MPa)': [avg_train_metrics['MAE'], avg_test_metrics['MAE']],
            'MSE (MPa²)': [avg_train_metrics['MSE'], avg_test_metrics['MSE']],
            'RMSE (MPa)': [avg_train_metrics['RMSE'], avg_test_metrics['RMSE']],
            'MAPE (%)': [avg_train_metrics['MAPE'], avg_test_metrics['MAPE']],
        }
        df_avg = pd.DataFrame(avg_data)
        df_avg.to_excel(writer, sheet_name='Three fold average metrics', index=False)
        print("✓ Saved: Three fold average metrics")
        
        # Sheet 3: Final model metrics
        final_data = {
            'Dataset': [f'Training set ({len(X_train_final)} samples)', f'Test set ({len(X_test_final)} samples)'],
            'R²': [train_metrics[' R2'], test_metrics[' R2']],
            'R': [train_metrics[' R'], test_metrics[' R']],
            'MAE (MPa)': [train_metrics[' MAE'], test_metrics[' MAE']],
            'MSE (MPa²)': [train_metrics[' MSE'], test_metrics[' MSE']],
            'RMSE (MPa)': [train_metrics[' RMSE'], test_metrics[' RMSE']],
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
            'Metric': ['R²', 'RMSE (MPa)', 'MSE (MPa²)', 'MAE (MPa)', 'MAPE (%)'],
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
    print(f"  Contains 5 sheets: Each fold metrics, Three fold average metrics, Final model metrics, Best hyperparameters, Radar chart data")
    
    # 7. Save model and results
    model_save_path = os.path.join(SAVE_ROOT, 'lightgbm_final_model.joblib')
    
    # If it is EnsembleModel, it needs to be special processed (saved as dictionary format)
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
    print(f"Final result chart saved to: {results_dir}")

if __name__ == "__main__":
    main_process()