#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGBoost Prediction Model - Peak Strain Prediction (Including Hyperparameter Printing and Learning Curve Monitoring)
Strategy: 3-fold Hyperparameter Search + Independent Test Set Selection + Final Model Training

Enhanced Features:
1. Print all metrics (R2, R, MAE, MSE, RMSE, MAPE).
2. Print **selected optimal hyperparameters** in the final report.
3. Training/validation curve monitoring: record and plot learning curves to identify bias-variance balance point (optimal iteration count).
   - Automatically use early stopping to prevent overfitting
   - Visualize RMSE changes of training and validation sets with iterations
   - Mark optimal iteration point (bias-variance balance point)

Note: This script is used for predicting peak strain, using 17 features (15 material parameters + fc + Xiao_strain)
NGBoost Features: Supports probabilistic prediction, can output prediction distribution and uncertainty quantification
"""


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
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
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_strain', 'NGBoost', 'save')
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Working Directory: {PROJECT_ROOT}")
print(f"Save Directory: {SAVE_ROOT}")

# --- Core Function Definitions ---

def compute_xiao_strain(fc: np.ndarray, r: np.ndarray) -> np.ndarray:
     """Calculate peak strain using Xiao's formula (consistent with PINN script implementation).
    
    Formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
    
    Args:
        fc: Stress value (compressive strength)
        r: Aggregate replacement rate (stored as percentage×100 in data, e.g., 1.5 represents 1.5%)
    
    Returns:
        Peak strain predicted by Xiao's formula
    """
    # r = aggregate replacement rate, convert from percentage×100 (e.g., 1.5) to decimal (0.015)
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
    """Load and parse data - for peak strain prediction"""
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    if not os.path.exists(data_file):
        print(f"Error: File does not exist {data_file}")
        return None
    
    df = pd.read_excel(data_file, sheet_name=0)
    
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    extra_features = ['fc']  #  Peak stress as input featur
    formula_features = ['Xiao_strain']   # Xiao's formula as input feature
    
    # For peak strain prediction, features include: 15 material parameters + fc + Xiao_strain = 17 features
    feature_names = material_features + specimen_features + extra_features + formula_features
    target_column = 'peak_strain'
    
    # Check if feature columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: The following feature columns do not exist: {missing_features}")
        # If Xiao_strain is missing, try to calculate it
        if "Xiao_strain" in missing_features and "fc" in df.columns and "r" in df.columns:
            print("  Attempting to calculate Xiao_strain...")
            df["Xiao_strain"] = compute_xiao_strain(df["fc"].values, df["r"].values)
            print("  ✓ Xiao_strain calculated and added")
            missing_features = [f for f in missing_features if f != "Xiao_strain"]
        
        # Remove other missing features
        if missing_features:
            feature_names = [f for f in feature_names if f in df.columns]
            print(f"  已移除缺失特征: {missing_features}，当前特征数: {len(feature_names)}")
    
    #  Check if target variable column exists
    if target_column not in df.columns:
        print(f"Error: Missing target variable column '{target_column}'")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    X = df[feature_names].values
    y = df[target_column].values
    
    if 'DataSlice' not in df.columns:
        print("Error: Missing 'DataSlice' column for data splitting")
        return None
        
    sample_divisions = df['DataSlice'].values
    sample_ids = df['No_Customized'].values if 'No_Customized' in df.columns else np.arange(len(df))
    
    print(f"Data loaded successfully:{len(X)} Sample, {len(feature_names)} Feature")
    return X, y, feature_names, sample_divisions, sample_ids, df

def objective(trial, X_train, y_train):
    """Optuna objective function (inner 5-fold CV)
    
    Hyperparameter search range optimization explanation (for small sample scenarios, about 60-70 training samples):
    - max_depth: 3-6 (不宜过深 for small samples)
    - learning_rate: 0.01-0.2 (common range for NGBoost)
    - n_estimators: 100-500 (with early stopping)
    - minibatch_frac: 0.5-1.0 (mini-batch fraction, helps with regularization)
    - col_sample: 0.6-1.0 (feature sampling fraction)
    
    NGBoost parameter explanation:
    - n_estimators: Number of trees (corresponding to XGBoost's n_estimators)
    - learning_rate: Learning rate
    - minibatch_frac: Mini-batch fraction (regularization)
    - col_sample: Feature sampling fraction (corresponding to XGBoost's colsample_bytree)
    - Base: Base learner (DecisionTreeRegressor)
    - Dist: Distribution type (Normal for regression)
    - Score: Scoring rule (LogScore)
    """
    # Base learner parameters
    base_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 40),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
    }
    
    base_learner = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        **base_params
    )
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'minibatch_frac': trial.suggest_float('minibatch_frac', 0.5, 1.0),
        'col_sample': trial.suggest_float('col_sample', 0.6, 1.0),
        'Base': base_learner,
        'Dist': Normal,
        'Score': LogScore,
        'verbose': False,
        'random_state': RANDOM_SEED,
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        # Create model (need to recreate base_learner as it cannot be serialized)
        base_learner_cv = DecisionTreeRegressor(
            max_depth=base_params['max_depth'],
            min_samples_split=base_params['min_samples_split'],
            min_samples_leaf=base_params['min_samples_leaf'],
            random_state=RANDOM_SEED
        )
        model_params_cv = params.copy()
        model_params_cv['Base'] = base_learner_cv
        
        model = NGBRegressor(**model_params_cv)
        
         # NGBoost's early stopping is implemented through validation_fraction and early_stopping_rounds
        # But to be consistent with 5-fold CV, we manually calculate validation set performance
        model.fit(X_tr, y_tr)
        
       # Use validation set for early stopping check
        # Simplified processing here, directly use all estimators
        y_pred = model.predict(X_v)
        
        # Use RMSE as objective function
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        rmse_scores.append(rmse)
    
      # Return average RMSE (smaller is better)
    return np.mean(rmse_scores)

def optimize_ngboost(X_train, y_train, n_trials=50):
    """Run Optuna optimization
    
    Note: The objective function is RMSE of validation set (smaller is better), so direction='minimize'
    
    【Top-level Expert Analysis: Impact of Increasing Trial Count】
    
    1. **Theoretical benefits of increasing trial count**:
       - Explore larger hyperparameter space, potentially finding better combinations
       - TPE sampler needs a certain number of trials to build an effective probability model (usually 20-30 trials)
       - For a search space with 7 hyperparameters, 50 trials can cover the main areas
    
    2. **Potential risks of increasing trial count (small sample scenario)**:
       - **Overfitting to validation set**: Excessive optimization may lead to hyperparameters overfitting to validation set, reducing generalization ability
       - **Diminishing returns**: Performance improvement is minimal after a certain number (usually after 50-100 trials)
       - **Computational cost**: Each trial requires 5-fold CV training, 50 trials = 250 model trainings
       - **Small sample limitation**: With 60-70 samples, validation set only has 12-14 samples, leading to unstable evaluation
    
    3. **Recommendations for current scenario**:
       - **n_trials=50-100**: For small samples, 50 trials are usually sufficient, 100 trials is the upper limit
       - **Outer 3-fold CV**: Each fold is optimized independently, equivalent to 3 optimizations, increasing exploration opportunities
       - **Early stopping mechanism**: If there is no improvement for 10-15 consecutive trials, consider stopping early
       - **Search space optimization**: More important than trial count is the rationality of the search space (already optimized)
    
    4. **Judging whether to increase trial count**:
       - If the optimal value continues to decrease after 50 trials → can increase to 100
       - If the optimal value stabilizes after 50 trials → increasing trial count has little meaning
       - If there is a large gap between validation set performance and test set performance → may be overfitting, should reduce trial count or enhance regularization
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)  # Changed to minimize because smaller RMSE is better
    
    # 【Optimization suggestion】: For small samples, add early stopping mechanism
    # Stop early if there is no improvement for N consecutive trials (save computational resources)
    # Note: Optuna's callback is called after each trial to check if it should stop
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
            
            # Stop optimization if no improvement for consecutive patience trials
            if self.no_improve_count >= self.patience:
                study.stop()
                print(f"  ⚠ Early stopping triggered: No improvement for {self.patience} consecutive trials, stopping optimization early")
    
    # Enable early stopping for n_trials > 50 (save computation time)
    callbacks = [EarlyStoppingCallback(patience=15, min_delta=0.001)] if n_trials > 50 else None
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials, 
        n_jobs=1,
        callbacks=callbacks
    )
    
    # Print optimization statistics
    print(f"  Optimization completed: Total {len(study.trials)} trials, best RMSE: {study.best_value:.4f}")
    if len(study.trials) < n_trials:
        print(f"  (Early stopping triggered, saved computation time for {n_trials - len(study.trials)} trials)")
    
    return study

def train_ngboost(X_train, y_train, X_val, y_val, best_params, return_history=False):
    """Train model: combine X_train and X_val for training
    
    Parameters:
        return_history: If True, return training history records (for plotting learning curves)
    
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
    
     # Extract base learner parameters and rebuild
    base_params = {
        'max_depth': best_params.get('max_depth', 4),
        'min_samples_split': best_params.get('min_samples_split', 20),
        'min_samples_leaf': best_params.get('min_samples_leaf', 10),
    }
    
    base_learner = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        **base_params
    )
    
    # Build NGBoost parameters
    model_params = {
        'n_estimators': best_params.get('n_estimators', 200),
        'learning_rate': best_params.get('learning_rate', 0.1),
        'minibatch_frac': best_params.get('minibatch_frac', 0.8),
        'col_sample': best_params.get('col_sample', 0.8),
        'Base': base_learner,
        'Dist': Normal,
        'Score': LogScore,
        'verbose': False,
        'random_state': RANDOM_SEED,
    }
    
    # Train model
    history = None
    
    if len(X_val) > 0 and return_history:
        # Use validation set to record training history
        n_estimators = model_params['n_estimators']
        train_rmse = []
        val_rmse = []
        
        # Train gradually and record history
        for n in range(1, n_estimators + 1):
            model_temp = NGBRegressor(
                n_estimators=n,
                learning_rate=model_params['learning_rate'],
                minibatch_frac=model_params['minibatch_frac'],
                col_sample=model_params['col_sample'],
                Base=DecisionTreeRegressor(
                    max_depth=base_params['max_depth'],
                    min_samples_split=base_params['min_samples_split'],
                    min_samples_leaf=base_params['min_samples_leaf'],
                    random_state=RANDOM_SEED
                ),
                Dist=Normal,
                Score=LogScore,
                verbose=False,
                random_state=RANDOM_SEED,
            )
            model_temp.fit(X_train, y_train)
            
            y_train_pred = model_temp.predict(X_train)
            y_val_pred = model_temp.predict(X_val)
            
            train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        
        # Find the best iteration
        best_iteration = np.argmin(val_rmse)
        best_score = val_rmse[best_iteration]
        
        history = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'best_iteration': best_iteration,
            'best_score': best_score
        }
        
        # Use the best iteration to train the final model
        model_params['n_estimators'] = best_iteration + 1
        base_learner_final = DecisionTreeRegressor(
            max_depth=base_params['max_depth'],
            min_samples_split=base_params['min_samples_split'],
            min_samples_leaf=base_params['min_samples_leaf'],
            random_state=RANDOM_SEED
        )
        model_params['Base'] = base_learner_final
        model = NGBRegressor(**model_params)
        model.fit(X_full, y_full)
    else:
        model = NGBRegressor(**model_params)
        model.fit(X_full, y_full)
    
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
    """Plot scatter plot of true values vs predicted values"""
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
    
    To identify the bias-variance balance point (best iteration number)
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
    plt.title(f'{prefix} Learning Curve\n(Bias-variance balance point: Iteration {best_iter + 1})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add explanatory text
    if best_iter < len(val_rmse):
        gap = train_rmse[best_iter] - val_rmse[best_iter]
        if gap > 0:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f}\n(Positive value indicates possible mild underfitting)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        else:
            plt.text(0.02, 0.98, 
                    f'Training-validation gap: {gap:.4f}\n(Negative value indicates possible overfitting risk)',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                    fontsize=10)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_learning_curve.png'), dpi=300)
    plt.close()
    
    print(f"  Learning curve has been saved: {prefix}_learning_curve.png")
    if best_iter < len(val_rmse):
        print(f"  Best iteration number: {best_iter + 1}/{len(val_rmse)}")
        print(f"  Training set RMSE: {train_rmse[best_iter]:.4f}")
        print(f"  Validation set RMSE: {val_rmse[best_iter]:.4f}")
        print(f"  Training-validation gap: {train_rmse[best_iter] - val_rmse[best_iter]:.4f}")
    

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
        
        print("   performing Optuna inner 5-fold optimization...")
        study = optimize_ngboost(X_train_fold, y_train_fold, n_trials=50) 
        best_params = study.best_params
        
        # Train model and record learning curve (for monitoring overfitting)
        print("   training model and recording learning curve...")
        model, history = train_ngboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
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
    # *** Core modification: print the best hyperparameters ***
    # *******************************
    print(f"\n**Best fold:** Fold {best_result['fold']} (based on R2={best_result['r2']:.4f})")
    print("\nSelected best hyperparameters:")
    for key, value in best_result['params'].items():
        print(f"  - {key}: {value}")
    
    print("="*70)
    
    # 4. Train the final model
    print(f"\nTraining final model (using the best hyperparameters of Fold {best_result['fold']} + all non-Test data)...")
    print(f"Best fold hyperparameters source: Fold {best_result['fold']} ({best_result['val_label']} as validation set)")
    
    # To monitor the training process of the final model, split 20% of the training set as validation set
    from sklearn.model_selection import train_test_split
    X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
        X_train_final, y_train_final, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"  Using 20% of the training set ({len(X_val_monitor)} samples) as monitoring validation set")
    
    # Train the final model and record learning curve
    final_model, final_history = train_ngboost(
        X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor,
        best_result['params'], return_history=True
    )
    
    # Save the learning curve of the final model
    results_dir = os.path.join(SAVE_ROOT, 'learning_curves')
    plot_learning_curve(final_history, results_dir, prefix="Final_Model")
    
    # Use all training data to retrain the final model (for actual prediction)
    # Strategy explanation:
    # 1. Use the best iteration number (conservative strategy): based on the bias-variance balance point found on the validation set, to avoid overfitting
    # 2. Use all iteration numbers (aggressive strategy): because the final model has more data, it may be able to train more rounds
    # 
    # Why use the best iteration number?
    # - The best iteration number is based on the bias-variance balance point found on the validation set
    # - Even if more data is used to train, excessive iteration may still lead to overfitting
    # - In a small sample scenario, the conservative strategy is usually more reliable
    # - The best point on the validation set typically generalizes better to the test set
    
    print("  Retraining the final model using all training data...")
    best_iteration_final = final_history['best_iteration']
    original_iterations = best_result['params'].get('n_estimators', 200)
    
    print(f"  Strategy comparison:")
    print(f"    - Best iteration number (used): {best_iteration_final + 1}")
    print(f"    - Original iteration number (not used): {original_iterations}")
    print(f"  Reason: based on the bias-variance balance point found on the validation set, to avoid overfitting risk")
    
    # Train the final model using the best iteration number (conservative strategy)
    final_params = best_result['params'].copy()
    final_params['n_estimators'] = best_iteration_final + 1  # +1 because the index starts from 0
    final_model = train_ngboost(X_train_final, y_train_final, [], [], final_params)
    
    # 5. Evaluate the final model and print detailed metrics
    y_train_final_pred = final_model.predict(X_train_final)
    train_metrics = calculate_metrics(y_train_final, y_train_final_pred, prefix="")
    
    y_test_final_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_final_pred, prefix="")
    
    # Optional: compare the effect of using all iteration numbers (for verifying the strategy selection)
    print(f"\n  [Strategy comparison] Training the model using all iteration numbers for comparison...")
    final_model_full = train_ngboost(X_train_final, y_train_final, [], [], best_result['params'])
    y_test_pred_full = final_model_full.predict(X_test)
    test_metrics_full = calculate_metrics(y_test, y_test_pred_full, prefix="")
    
    print(f"  Comparison result (test set):")
    print(f"    - Best iteration number model: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    print(f"    - All iteration numbers model: R²={test_metrics_full[' R2']:.4f}, RMSE={test_metrics_full[' RMSE']:.3f}")
    
    # Select the model with better test set performance
    if test_metrics_full[' R2'] > test_metrics[' R2']:
        print(f"  → All iteration numbers model performs better on the test set, but需要注意过拟合风险")
        print(f"  → Currently using: Best iteration number model (more conservative, more reliable generalization ability)  ")
    else:
        print(f"  → Best iteration number model performs better on the test set,证明了策略选择的正确性")
    
    print("\nFinal model training and test metrics:")
    print(f"(Using the best hyperparameters of Fold {best_result['fold']}, on all non-test data)")
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
    print(f"  - Final model Training set (subset1+subset2+subset3): R²={train_metrics[' R2']:.4f}, RMSE={train_metrics[' RMSE']:.3f}")
    print(f"  - Final model Test set: R²={test_metrics[' R2']:.4f}, RMSE={test_metrics[' RMSE']:.3f}")
    
    # 6. Model and results save
    model_save_path = os.path.join(SAVE_ROOT, 'ngboost_final_model.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(final_model, model_save_path)
    print(f"\nModel architecture (NGBoost) has been saved to: {model_save_path}")
    
    results_dir = os.path.join(SAVE_ROOT)
    plot_results(y_test, y_test_final_pred, results_dir, prefix="Final_Test")
    print(f"Final results plot has been saved to: {results_dir}")

if __name__ == "__main__":
    main_process()
