#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomForest prediction model - final complete version (including hyperparameter printing)
Cross-validation strategy: 3-fold hyperparameter search + independent test set selection + final model training

Enhanced features:
1. Print all metrics (R2, R, MAE, MSE, RMSE, MAPE).
2. Print the **selected best hyperparameters** in the final report.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_stress', 'RandomForest', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Working directory: {PROJECT_ROOT}")
print(f"Save directory: {SAVE_ROOT}")

# --- Core function definition (unchanged) ---
    
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
    """Optuna objective function (inner 5-fold CV)
    
    Hyperparameter search range optimization explanation (for small sample scenarios, approximately 60-70 training samples):
    - n_estimators: 50-500 (number of Random Forest trees, small samples do not need too many trees)
    - max_depth: 3-15 (tree maximum depth, None means no limit, but small samples suggest limiting)
    - min_samples_split: 2-20 (minimum number of samples required for splitting, to prevent overfitting)
    - min_samples_leaf: 1-10 (minimum number of samples required for leaf nodes)
    - max_features: 'sqrt', 'log2', None or float (number of features considered for each split)
    - bootstrap: True/False (whether to use bootstrap sampling)
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 500),  # number of trees
        'max_depth': trial.suggest_int('max_depth', 1, 20),  # maximum depth, small samples suggest limiting
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # minimum number of samples required for splitting, to prevent overfitting
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),  # minimum number of samples required for leaf nodes
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # feature sampling方式
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # whether to use bootstrap sampling
        'random_state': RANDOM_SEED,
        'n_jobs': -1,  # use all CPU cores
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_v)
        
        # Use RMSE as the objective function (more conventional for regression problems)
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        rmse_scores.append(rmse)
    
    # Return average RMSE (smaller is better, so direction='minimize' in optimize_randomforest)
    return np.mean(rmse_scores)

def optimize_randomforest(X_train, y_train, n_trials=50):
    """Run Optuna optimization
    
    Note: the objective function is the RMSE of the validation set (smaller is better), so direction='minimize'
    
    [Top-level master analysis: the impact of increasing the number of trials]
    
    1. **Theoretical benefits of increasing the number of trials**：
       - Explore a larger hyperparameter space, possibly finding a better combination
       - TPE sampler needs a certain number of trials to establish an effective probability model (usually requires 20-30 trials)
       - For a search space of 6 hyperparameters, 50 trials already cover the main area
    
    2. **Potential risks of increasing the number of trials (small sample scenarios)**：
       - **Overfitting to the validation set**：Overfitting may cause hyperparameters to overfit the validation set, reducing generalization ability
       - **diminishing returns: after a certain number, performance improvement is negligible (usually after 50-100 trials)
       - **computational cost: each trial needs 5-fold CV training, 50 trials = 250 model trainings
       - **small samples, validation set only 12-14 samples, unstable evaluation
    
    3. **Suggestions for the current scenario**：
       - **n_trials=50-100**：for small samples, 50 trials usually suffice, 100 trials is the upper limit
       - **outer 3-fold CV: each fold is independently optimized, equivalent to 3 optimizations,增加了探索机会
       - **early stopping mechanism: if no improvement for 10-15 trials, can consider stopping early
       - **search space optimization: the importance of the search space is more than the number of trials (currently optimized)
    
    4. **判断是否需要增加trial数 determine whether to increase the number of trials**：
       - if the optimal value is still decreasing after 50 trials → can increase to 100
       - if the optimal value is stable after 50 trials → increasing the number of trials is not meaningful
       - if the performance of the validation set is significantly different from the test set → possibly overfitting, should reduce the number of trials or enhance regularization
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)  # change to minimize, because RMSE越小越好
    
    # [Optimization suggestion]: for small samples, can add early stopping mechanism
    # if no improvement for N trials, stop early (save computational resources)
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
            
            # Check for significant improvement (relative improvement > min_delta)
            if study.best_value < self.best_value * (1 - self.min_delta):
                self.best_value = study.best_value
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # If no improvement for patience trials, stop optimization
            if self.no_improve_count >= self.patience:
                study.stop()
                print(f"  ⚠ Early stopping triggered: no improvement for {self.patience} trials, stop optimization early")
    
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
        print(f"  (Early stopping triggered, saved {n_trials - len(study.trials)} trial computations)")
    
    return study

def train_randomforest(X_train, y_train, X_val, y_val, best_params):
    """Train model: merge X_train and X_val for training
    
    Parameters:
        X_train: training features
        y_train: training labels
        X_val: validation features (for evaluation, but will be merged into the training set)
        y_val: validation labels
        best_params: best hyperparameters
    
    Returns:
        model: trained model
    """
    if len(X_val) > 0:
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train
    
    # Build model parameters
    model_params = {
        'random_state': RANDOM_SEED,
        'n_jobs': -1,  # use all CPU cores
        **best_params
    }
    # Ensure random_state is consistent
    model_params['random_state'] = RANDOM_SEED
    
    # Train model
    model = RandomForestRegressor(**model_params)
    model.fit(X_full, y_full)
    
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
        
        print("   Performing Optuna inner 5-fold optimization...")
        study = optimize_randomforest(X_train_fold, y_train_fold, n_trials=50) 
        best_params = study.best_params
        
        # Train model
        print("   Training model...")
        model = train_randomforest(X_train_fold, y_train_fold, X_val_fold, y_val_fold, best_params)
        
        # Calculate training set metrics (merged training set)
        X_train_full = np.vstack([X_train_fold, X_val_fold]) if len(X_val_fold) > 0 else X_train_fold
        y_train_full = np.hstack([y_train_fold, y_val_fold]) if len(y_val_fold) > 0 else y_train_fold
        y_train_pred = model.predict(X_train_full)
        train_metrics = calculate_metrics(y_train_full, y_train_pred, prefix="Train")
        
        # Calculate test set metrics
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="Test")
        
        print(f"   >>> Fold {i+1} training set R2: {train_metrics['Train R2']:.4f}, RMSE: {train_metrics['Train RMSE']:.3f}")
        print(f"   >>> Fold {i+1} test set R2: {test_metrics['Test R2']:.4f}, RMSE: {test_metrics['Test RMSE']:.3f}")
        
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
        print(f"   Training set: R²={res['train_metrics']['Train R2']:.4f}, R={res['train_metrics']['Train R']:.4f}, "
              f"MAE={res['train_metrics']['Train MAE']:.3f}, MSE={res['train_metrics']['Train MSE']:.3f}, "
              f"RMSE={res['train_metrics']['Train RMSE']:.3f}, MAPE={res['train_metrics']['Train MAPE']:.2f}%")
        print(f"   Test set: R²={res['test_metrics']['Test R2']:.4f}, R={res['test_metrics']['Test R']:.4f}, "
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
    
    print(f"   Training set average: R²={avg_train_metrics['R2']:.4f}, R={avg_train_metrics['R']:.4f}, "
          f"MAE={avg_train_metrics['MAE']:.3f}, MSE={avg_train_metrics['MSE']:.3f}, "
          f"RMSE={avg_train_metrics['RMSE']:.3f}, MAPE={avg_train_metrics['MAPE']:.2f}%")
    print(f"   Test set average: R²={avg_test_metrics['R2']:.4f}, R={avg_test_metrics['R']:.4f}, "
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
    
    # Train final model (using all non-test data)
    print("   Using all training data to train the final model...")
    final_model = train_randomforest(X_train_final, y_train_final, [], [], best_result['params'])
    
    # 5. Evaluate final model and print detailed metrics
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_final_pred, prefix="")
    
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
    print(f"  - Fold {best_result['fold']} training set (subset1+subset3): R²={best_result['train_metrics']['Train R2']:.4f}, RMSE={best_result['train_metrics']['Train RMSE']:.3f}")
    print(f"  - Fold {best_result['fold']} test set: R²={best_result['test_metrics']['Test R2']:.4f}, RMSE={best_result['test_metrics']['Test RMSE']:.3f}")
    print(f"  - Final model training set (subset1+subset2+subset3): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    print(f"  - Final model test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    
    # 6. Model and result saving
    model_save_path = os.path.join(SAVE_ROOT, 'randomforest_final_model.joblib')
    joblib.dump(final_model, model_save_path)
    print(f"\nModel architecture (RandomForest) saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final result plot saved to: {results_dir}")

if __name__ == "__main__":
    main_process()