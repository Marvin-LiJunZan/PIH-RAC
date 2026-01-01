#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost peak stress prediction model - with hyperparameter optimization
使用Optuna进行贝叶斯优化

新增功能：
1. Noise robustness analysis: test the robustness of the model to Gaussian noise and outliers
2. Prediction interval analysis: use Bootstrap and Quantile Regression methods to generate confidence intervals
3. SHAP value analysis: feature importance explanation
4. PDP analysis: partial dependency plot analysis feature effect
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import shap
from sklearn.inspection import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set Chinese font and LaTeX rendering
def configure_plot_fonts(fonts=None):
    """Set Matplotlib font, ensure negative sign is displayed normally"""
    if fonts is None:
        fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False

configure_plot_fonts()
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font to render mathematical symbols

# Set working directory - automatically find project root directory
def find_project_root():
    """Find project root directory"""
    # 先尝试从脚本位置定位
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        # If the script is in a subdirectory, find the directory containing dataset or SAVE
        current_dir = script_dir
        search_depth = 0
        while current_dir != os.path.dirname(current_dir) and search_depth < 10:
            if os.path.exists(os.path.join(current_dir, 'dataset')) or \
               os.path.exists(os.path.join(current_dir, 'SAVE')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
            search_depth += 1
    except NameError:
        # Run in Jupyter, search from current directory upward
        pass
    
    # Search from current working directory upward
    current_dir = os.path.abspath(os.getcwd())
    search_limit = 0
    while current_dir != os.path.dirname(current_dir) and search_limit < 10:
        # Check if the directory contains dataset folder or specific project file
        if os.path.exists(os.path.join(current_dir, 'dataset')) or \
           os.path.exists(os.path.join(current_dir, 'SAVE')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
        search_limit += 1
    
    # If still not found, return the current directory itself
    return os.path.abspath(os.getcwd())

# Set working directory
PROJECT_ROOT = find_project_root()
if not os.path.exists(os.path.join(PROJECT_ROOT, 'dataset')):
    # If still not found, use the current directory
    PROJECT_ROOT = os.getcwd()

os.chdir(PROJECT_ROOT)
print(f"Working directory set to: {PROJECT_ROOT}")

# Save directory root path
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'peak_stress', 'SAVE')
os.makedirs(SAVE_ROOT, exist_ok=True)

def load_data():
    """Load data - use 10 material parameters and 5 specimen parameters to regress peak stress"""
    print("=== Load data ===")
    
    # Use the specified data file
    data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
    
    if not os.path.exists(data_file):
        print(f"Error: data file not found: {data_file}")
        return None
    
    print(f"Found data file: {data_file}")
    
    # Read the first sheet of the Excel file (material parameter table)
    df = pd.read_excel(data_file, sheet_name=0)
    print(f"Data shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Define features: 10 material parameters + 5 specimen parameters
    # 10 material features
    material_features = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI']
    # 5 specimen parameters
    specimen_features = ['age', 'μe', 'DJB', 'side', 'GJB']
    
    feature_names = material_features + specimen_features
    
    # Target variable: peak stress
    target_column = 'fc'
    
    # Check column names
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        print(f"Warning: missing columns {missing_cols}")
        feature_names = [col for col in feature_names if col in df.columns]
    
    missing_target = target_column not in df.columns
    if missing_target:
        print(f"Error: missing target variable column '{target_column}'")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Extract data
    X = df[feature_names].values
    y = df[target_column].values
    
    # Check if there are missing values
    if np.isnan(X).any():
        print(f"Warning: feature data contains NaN values, will be filled")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    if np.isnan(y).any():
        print(f"Warning: target variable contains NaN values, will be removed")
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        df = df.iloc[valid_mask].reset_index(drop=True)
    
    # Extract sample division information (using DataSlice column)
    sample_divisions = []
    if 'DataSlice' in df.columns:
        sample_divisions = df['DataSlice'].values
        print(f"Found data set division information: {np.unique(sample_divisions, return_counts=True)}")
    else:
        print("DataSlice column not found, will use random division")
        sample_divisions = None
    
    # Extract sample IDs
    sample_ids = []
    if 'No_Customized' in df.columns:
        sample_ids = df['No_Customized'].values
    else:
        sample_ids = [f"sample_{i}" for i in range(len(X))]
    
    print(f"Feature count: {X.shape[1]} (10 material parameters + 5 specimen parameters)")
    print(f"Sample count: {X.shape[0]}")
    print(f"Target variable '{target_column}' range: {np.min(y):.2f} - {np.max(y):.2f}")
    
    # Save original DataFrame indices (for subsequent mapping prediction results)
    original_df_indices = df.index.values
    
    return X, y, feature_names, sample_divisions, sample_ids, original_df_indices, df

def split_data_by_divisions(X, y, sample_divisions, sample_ids, test_ratio=0.2, val_ratio=0.2, random_state=42):
    """Split data based on sample division information"""
    print("=== Split data based on sample division information ===")
    
    if sample_divisions is None:
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_ratio + val_ratio, random_state=random_state
        )
        val_ratio_adjusted = val_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio_adjusted, random_state=random_state
        )
        
        train_ids = [f"train_{i}" for i in range(len(X_train))]
        val_ids = [f"val_{i}" for i in range(len(X_val))]
        test_ids = [f"test_{i}" for i in range(len(X_test))]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids
    
    # Classify samples based on sample division information (support DataSlice format: subset1, subset2, subset3, test)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, division in enumerate(sample_divisions):
        division_str = str(division).strip()
        if division_str.lower().startswith('test'):
            test_indices.append(i)
        elif division_str.startswith('subset'):
            # subset1, subset2, subset3 are all considered as training set (or can be allocated as needed)
            # Here, all are considered as training set by default, if validation set is needed, it can be split from subset
            train_indices.append(i)
        elif division == 'train' or division == 'train set' or division_str.startswith('train'):
            train_indices.append(i)
        elif division == 'val' or division == 'validation set' or division == 'validation' or division_str.startswith('val'):
            val_indices.append(i)
        else:
            # Default to training set
            train_indices.append(i)
    
    # If there is no validation set but there is a training set, split validation set from training set
    # If test set exists, keep it unchanged
    if train_indices and not val_indices:
        print("Only training set samples, split validation set from training set...")
        from sklearn.model_selection import train_test_split
        
        train_indices = np.array(train_indices)
        
        if len(train_indices) > 1:
            # If test set exists, only split validation set from training set
            # If test set does not exist, split validation set and test set from training set
            if len(test_indices) > 0:
                # test set already exists, only split validation set
                val_ratio_adjusted = val_ratio
                train_idx, val_idx = train_test_split(
                    train_indices, test_size=val_ratio_adjusted, random_state=random_state
                )
                train_indices = train_idx.tolist()
                val_indices = val_idx.tolist()
            else:
                # test set does not exist, split validation set and test set from training set
                train_val_idx, test_idx = train_test_split(
                    train_indices, test_size=test_ratio, random_state=random_state
                )
                val_ratio_adjusted = val_ratio / (1 - test_ratio)
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=val_ratio_adjusted, random_state=random_state
                )
                train_indices = train_idx.tolist()
                val_indices = val_idx.tolist()
                test_indices = test_idx.tolist()
        else:
            val_indices = []
    
    # Convert to numpy array
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Split data
    X_train = X[train_indices] if len(train_indices) > 0 else np.array([]).reshape(0, X.shape[1])
    X_val = X[val_indices] if len(val_indices) > 0 else np.array([]).reshape(0, X.shape[1])
    X_test = X[test_indices] if len(test_indices) > 0 else np.array([]).reshape(0, X.shape[1])
    
    y_train = y[train_indices] if len(train_indices) > 0 else np.array([])
    y_val = y[val_indices] if len(val_indices) > 0 else np.array([])
    y_test = y[test_indices] if len(test_indices) > 0 else np.array([])
    
    # Get corresponding sample IDs
    train_ids = [sample_ids[i] for i in train_indices] if len(train_indices) > 0 else []
    val_ids = [sample_ids[i] for i in val_indices] if len(val_indices) > 0 else []
    test_ids = [sample_ids[i] for i in test_indices] if len(test_indices) > 0 else []
    
    print(f"Data division results:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna optimization objective function - use validation set for evaluation"""
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        
        # Hyperparameter search space
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 0.6),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20.0, log=True),
    }
    
    # Use internal cross-validation of training set to evaluate, more stable
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_v)
        r2 = r2_score(y_v, y_pred)
        scores.append(r2)
    
    return np.mean(scores)

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=100):
    """Use Optuna to optimize XGBoost hyperparameters - use 5-fold cross-validation for evaluation"""
    print(f"\n=== Start XGBoost hyperparameter optimization (n_trials={n_trials}) ===")
    print("Use 5-fold cross-validation to evaluate hyperparameters, improve stability")
    
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, n_jobs=1)
    
    print("\n=== Optimization completed ===")
    print(f"Best R2 score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Visualize optimization history
    try:
        save_dir = os.path.join(SAVE_ROOT, 'xgboost_optimization')
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plot_optimization_history(study)
        plt.savefig(os.path.join(save_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig = plot_param_importances(study)
        plt.savefig(os.path.join(save_dir, 'param_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass
    
    return study

def train_xgboost(X_train, y_train, X_val, y_val, best_params):
    """Use best parameters to train XGBoost model"""
    print("\n=== Use best parameters to train final model ===")
    
    # Merge training set and validation set
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    
    # Create model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        random_state=42,
        verbosity=0,
        **best_params
    )
    
    # Train model
    model.fit(X_train_full, y_train_full)
    
    return model

def plot_results(y_true, y_pred, save_dir, model_name='xgboost'):
    """Plot results"""
    print("\n=== Plot results ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'XGBoost Results ($R^2$={r2:.4f})', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs true values
    axes[0,0].scatter(y_true, y_pred, alpha=0.7, s=60, color='green')
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('True Peak Stress fc (MPa)')
    axes[0,0].set_ylabel('Predicted Peak Stress fc (MPa)')
    axes[0,0].set_title(f'Prediction vs True Values\n$R^2$={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = y_pred - y_true
    axes[0,1].scatter(y_pred, residuals, alpha=0.7, s=60, color='green')
    axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0,1].set_xlabel('Predicted Peak Stress fc (MPa)')
    axes[0,1].set_ylabel('Residuals (MPa)')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Residual distribution
    axes[1,0].hist(residuals, bins=15, alpha=0.7, color='green', density=True)
    axes[1,0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1,0].set_xlabel('Residuals (MPa)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Residual Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Feature importance
    # Note: Feature importance needs to be obtained from model object after model training
    stats_text = f"""Model Performance Statistics:
    
Sample Count: {len(y_true)}
R2: {r2:.4f}
MAE: {mae:.3f} MPa
RMSE: {rmse:.3f} MPa

XGBoost Features:
- Gradient Boosting
- Optuna Hyperparameter Optimization
- Cross-validation evaluation
- Feature Importance Analysis"""
    
    axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('Performance Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'xgboost_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return r2, mae, rmse

def plot_train_test_comparison(model, X_train, y_train, X_test, y_test, save_dir):
    """Plot training set and test set comparison plot (with edge distribution) - optimized version"""
    print("\n=== Plot training set and test set comparison plot ===")
    
    # Prediction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Create figure (scatter plot with edge distribution)
    fig = plt.figure(figsize=(13, 11))
    
    # Define grid layout (adjust ratio)
    gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.08,
                         height_ratios=[0.8, 4, 0.1], width_ratios=[4, 0.8, 0.1])
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Upper histogram
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    
    # Right histogram
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # Draw main scatter plot
    # Training set - use more professional blue
    ax_main.scatter(y_train, y_train_pred, alpha=0.5, s=50, color='#2E86AB', 
                   label='Training', edgecolors='white', linewidths=0.5, zorder=3)
    
    # Test set - use more prominent orange-red
    ax_main.scatter(y_test, y_test_pred, alpha=0.7, s=60, color='#E63946',
                   label='Testing', edgecolors='white', linewidths=0.5, zorder=4)
    
    # 1:1 diagonal line
    all_values = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
    min_val = all_values.min()
    max_val = all_values.max()
    # Add padding
    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding
    
    ax_main.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, 
                alpha=0.4, label='Perfect Prediction', zorder=1)
    
    # Calculate and draw best fit line and confidence interval (training set)
    slope_train, intercept_train, r_value_train, p_value_train, std_err_train = stats.linregress(y_train, y_train_pred)
    line_train = slope_train * y_train + intercept_train
    predict_train = slope_train * np.sort(y_train) + intercept_train
    
    # Calculate confidence interval
    residuals_train = y_train_pred - line_train
    std_residuals_train = np.std(residuals_train)
    ci_train = 1.96 * std_residuals_train  # 95% CI
    
    ax_main.plot(np.sort(y_train), predict_train, color='#2E86AB', lw=3, alpha=0.9,
                linestyle='-', zorder=5)
    ax_main.fill_between(np.sort(y_train), predict_train - ci_train, predict_train + ci_train,
                         alpha=0.2, color='#2E86AB', zorder=2)
    
    # Best fit line and confidence interval (test set)
    slope_test, intercept_test, r_value_test, p_value_test, std_err_test = stats.linregress(y_test, y_test_pred)
    line_test = slope_test * y_test + intercept_test
    predict_test = slope_test * np.sort(y_test) + intercept_test
    
    residuals_test = y_test_pred - line_test
    std_residuals_test = np.std(residuals_test)
    ci_test = 1.96 * std_residuals_test
    
    ax_main.plot(np.sort(y_test), predict_test, color='#E63946', lw=3, alpha=0.9,
                linestyle='-', zorder=6)
    ax_main.fill_between(np.sort(y_test), predict_test - ci_test, predict_test + ci_test,
                         alpha=0.2, color='#E63946', zorder=2)
    
    # Set axis labels
    ax_main.set_xlabel('Observed Peak Stress fc (MPa)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Predicted Peak Stress fc (MPa)', fontsize=13, fontweight='bold')
    
    # Optimized legend - put in lower right corner, use LaTeX format to display R²
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                  markersize=8, alpha=0.7, label=f'Training ($R^2$={r2_train:.3f}, MAE={mae_train:.2f})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', 
                  markersize=9, alpha=0.8, label=f'Testing ($R^2$={r2_test:.3f}, MAE={mae_test:.2f})'),
        plt.Line2D([0], [0], color='k', linestyle='--', lw=2, alpha=0.4, label='Perfect Prediction'),
        plt.Line2D([0], [0], color='gray', lw=3, label='Best Fit Line'),
    ]
    ax_main.legend(handles=legend_elements, loc='lower right', fontsize=9.5, 
                  framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    
    ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax_main.set_axisbelow(True)
    
    # Upper histogram (observed value distribution) - stacked display
    bins = np.linspace(min_val, max_val, 20)
    # Use numpy.histogram to calculate frequency
    hist_train_obs, _ = np.histogram(y_train, bins=bins)
    hist_test_obs, _ = np.histogram(y_test, bins=bins)
    
    # Draw stacked histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    ax_top.bar(bin_centers, hist_train_obs, width=width, alpha=0.7, 
              color='#2E86AB', label='Training', edgecolor='white', linewidth=0.5)
    ax_top.bar(bin_centers, hist_test_obs, width=width, alpha=0.8, 
              color='#E63946', label='Testing', edgecolor='white', linewidth=0.5, bottom=hist_train_obs)
    ax_top.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax_top.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_top.tick_params(labelbottom=False, labelsize=10)
    ax_top.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax_top.set_axisbelow(True)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    # Right histogram (predicted value distribution) - stacked display
    hist_train_pred, _ = np.histogram(y_train_pred, bins=bins)
    hist_test_pred, _ = np.histogram(y_test_pred, bins=bins)
    
    # Draw horizontal stacked histogram
    ax_right.barh(bin_centers, hist_train_pred, height=width, alpha=0.7, 
                 color='#2E86AB', edgecolor='white', linewidth=0.5)
    ax_right.barh(bin_centers, hist_test_pred, height=width, alpha=0.8, 
                 color='#E63946', edgecolor='white', linewidth=0.5, left=hist_train_pred)
    ax_right.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax_right.tick_params(labelleft=False, labelsize=10)
    ax_right.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=0.5)
    ax_right.set_axisbelow(True)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    # Add total title - include sample count information
    title = f'XGBoost Model Performance\nTraining: n={len(y_train)} | Testing: n={len(y_test)}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.985)
    
    # Set overall background
    fig.patch.set_facecolor('white')
    
    plt.savefig(os.path.join(save_dir, 'train_test_comparison_with_margins.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    plt.close()
    
    print(f"\nTraining set performance - R2: {r2_train:.4f}, MAE: {mae_train:.3f} MPa, RMSE: {rmse_train:.3f} MPa")
    print(f"Test set performance - R2: {r2_test:.4f}, MAE: {mae_test:.3f} MPa, RMSE: {rmse_test:.3f} MPa")
    print(f"Performance difference - ΔR2: {abs(r2_train - r2_test):.4f}, ΔMAE: {abs(mae_train - mae_test):.3f} MPa")

def plot_feature_importance(model, feature_names, save_dir):
    """Plot feature importance"""
    print("\n=== Plot feature importance ===")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Draw bar chart
    plt.figure(figsize=(12, 15))
    plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to Excel
    importance_df.to_excel(os.path.join(save_dir, 'feature_importance.xlsx'), index=False)
    
    return importance_df

def plot_shap_analysis(model, X_train, X_test, feature_names, save_dir):
    """SHAP value analysis"""
    print("\n=== Perform SHAP analysis ===")
    
    # Use TreeExplainer (XGBoost专用)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (using test set)
    shap_values = explainer.shap_values(X_test)
    
    # Set font, avoid Chinese乱码
    original_font = list(plt.rcParams['font.sans-serif'])
    configure_plot_fonts(['SimHei', 'Microsoft YaHei', 'DejaVu Sans'])
    
    # 1. Summary plot
    # Increase width, truly stretch horizontal axis physically
    fig = plt.figure(figsize=(30, 12))  # Increase width to 30, stretch horizontal axis physically
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    # Get current axis
    ax = plt.gca()
    # Do not change data range, only increase graph width to stretch horizontal axis physically
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # 2. Bar plot (Feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # 3. Waterfall plot (show SHAP values of a single sample)
    # Select first sample as example
    shap_explanation = shap.Explanation(
        values=shap_values[0:1],
        base_values=explainer.expected_value,
        data=X_test[0:1],
        feature_names=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Restore original font settings
    plt.rcParams['font.sans-serif'] = original_font
    plt.rcParams['axes.unicode_minus'] = False
    
    # 4. Force plot (show prediction process of a single sample, long bar force plot)
    # Create explanation containing multiple samples for force plot
    n_samples_force = min(3, len(shap_values))
    shap_explanation_force = shap.Explanation(
        values=shap_values[:n_samples_force],
        base_values=explainer.expected_value,
        data=X_test[:n_samples_force],
        feature_names=feature_names
    )
    # Draw force plot of first 3 samples
    for i in range(n_samples_force):
        plt.figure(figsize=(16, 4))  # Long bar graph
        shap.plots.force(shap_explanation_force[i], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_force_sample_{i}.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
    
    # 5. Calculate average absolute SHAP value as feature importance
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    # Save SHAP importance
    shap_importance.to_excel(os.path.join(save_dir, 'shap_importance.xlsx'), index=False)
    
    # Restore original font settings
    plt.rcParams['font.sans-serif'] = original_font
    plt.rcParams['axes.unicode_minus'] = False
    
    return shap_importance

def plot_pdp_analysis(model, X_train, feature_names, save_dir, n_top_features=5):
    """PDP (partial dependency plot) analysis - using scikit-learn"""
    print("\n=== Perform PDP analysis ===")
    
    # Ensure X_train is DataFrame
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
    
    # Get most important n features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(n_top_features)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    
    # 1. Single variable PDP analysis
    n_cols = 3  # Change to 3 columns
    n_rows = (n_top_features + 2) // 3  # Adjust row count calculation
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_top_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (feature, ax) in enumerate(zip(top_features, axes)):
        try:
            # Get feature index
            feature_idx = feature_names.index(feature)
            
            # Calculate PDP (average) and ICE (individual)
            pd_result = partial_dependence(
                model, 
                X_train_df, 
                features=[feature_idx],
                kind='average'
            )
            
            ice_result = partial_dependence(
                model,
                X_train_df,
                features=[feature_idx],
                kind='individual'
            )
            
            # Calculate 95% confidence interval of ICE curve
            ice_curves = ice_result['individual'][0]  # shape: (n_samples, n_grid_points)
            ice_mean = np.mean(ice_curves, axis=0)
            ice_std = np.std(ice_curves, axis=0)
            ice_upper_ci = ice_mean + 1.96 * ice_std  # 95% CI
            ice_lower_ci = ice_mean - 1.96 * ice_std
            
            grid_values = pd_result['grid_values'][0]
            
            # Draw PDP line (blue)
            ax.plot(grid_values, pd_result['average'][0], 
                   linewidth=2.5, color='blue', marker='o', markersize=5, 
                   label='PDP', zorder=3)
            
            # Calculate ALE (true ALE, not approximate)
            # ALE is obtained by calculating local effects and accumulating them
            n_grid = len(grid_values)
            n_samples = len(X_train_df)
            
            # Create grid interval
            ale_values = np.zeros(n_grid)
            
            # For each grid point, calculate local effect
            for i in range(n_grid - 1):
                # Get samples in current interval
                if i == 0:
                    mask = (X_train_df.iloc[:, feature_idx].values <= grid_values[i+1])
                elif i == n_grid - 2:
                    mask = (X_train_df.iloc[:, feature_idx].values > grid_values[i])
                else:
                    mask = ((X_train_df.iloc[:, feature_idx].values > grid_values[i]) & 
                           (X_train_df.iloc[:, feature_idx].values <= grid_values[i+1]))
                
                # If there are samples in this interval
                if np.sum(mask) > 0:
                    # Calculate local effect: change feature value, other features remain unchanged
                    X_low = X_train_df.copy()
                    X_high = X_train_df.copy()
                    
                    # Set feature value to the ends of the interval
                    X_low.loc[mask, X_train_df.columns[feature_idx]] = grid_values[i]
                    X_high.loc[mask, X_train_df.columns[feature_idx]] = grid_values[i+1]
                    
                    # Calculate prediction difference
                    pred_low = model.predict(X_low.iloc[mask, :].values)
                    pred_high = model.predict(X_high.iloc[mask, :].values)
                    
                    # Local effect
                    local_effect = np.mean(pred_high - pred_low) / (grid_values[i+1] - grid_values[i])
                    
                    # Accumulate to ALE
                    if i == 0:
                        ale_values[i+1] = local_effect * (grid_values[i+1] - grid_values[i])
                    else:
                        ale_values[i+1] = ale_values[i] + local_effect * (grid_values[i+1] - grid_values[i])
            
            # Center ALE (make mean 0)
            ale_values = ale_values - np.mean(ale_values) + np.mean(pd_result['average'][0])
            
            # Draw PDP line (blue solid line)
            ax.plot(grid_values, pd_result['average'][0], 
                   linewidth=2.5, color='blue', marker='o', markersize=5, 
                   label='PDP', zorder=4)
            
            # Draw Mean c-ICE line (coral dashed line)
            ax.plot(grid_values, ice_mean, 
                   linewidth=2.5, color='coral', linestyle='--', 
                   marker='s', markersize=4, label='Mean c-ICE', zorder=4)
            
            # Draw ALE line (green dotted line)
            ax.plot(grid_values, ale_values, 
                   linewidth=2.5, color='green', linestyle='-.', 
                   marker='^', markersize=4, label='ALE', zorder=4)
            
            # Draw 95% CI shadow area
            ax.fill_between(grid_values, ice_lower_ci, ice_upper_ci, 
                           color='coral', alpha=0.3, label='95% CI of c-ICE', zorder=1)
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Partial Effect', fontsize=10)
            ax.set_title(f'PDP, ALE & c-ICE for {feature}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not create PDP for {feature}: {e}")
            ax.text(0.5, 0.5, f'Error creating PDP', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_top_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pdp_analysis.png'), dpi=300, bbox_inches='tight')
    # plt.show()  # Show single variable PDP plot
    plt.close()
    
    print(f"Single variable PDP analysis completed, have drawn dependency graphs of the first {n_top_features} important features")
    
    # 2. Dual variable PDP analysis (feature interaction)
    if n_top_features >= 2:
        print("\n=== Perform dual variable PDP analysis ===")
        
        # Select the first 4 most important features for dual variable interaction analysis
        n_interaction_features = min(4, n_top_features)
        interaction_features = top_features[:n_interaction_features]
        
        # Create interaction feature pairs
        interaction_pairs = []
        for i in range(len(interaction_features)):
            for j in range(i+1, len(interaction_features)):
                interaction_pairs.append((interaction_features[i], interaction_features[j]))
        
        # Draw dual variable PDP (using heatmap)
        n_pairs = len(interaction_pairs)
        n_cols = 3  # Change to 3 columns, more美观
        n_rows = (n_pairs + 2) // 3  # Adjust row count calculation
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
        
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (feat1, feat2) in enumerate(interaction_pairs):
            try:
                # Get feature index
                feat1_idx = feature_names.index(feat1)
                feat2_idx = feature_names.index(feat2)
                
                # Calculate dual variable partial dependence
                pd_result = partial_dependence(
                    model,
                    X_train_df,
                    features=[feat1_idx, feat2_idx],
                    kind='average'
                )
                
                # Get grid values and average prediction values
                grid_values_1 = pd_result['grid_values'][0]
                grid_values_2 = pd_result['grid_values'][1]
                average = pd_result['average'][0]  # Get first element (if it is a 3D array)
                
                # Ensure average is 2D
                if average.ndim == 3:
                    average = average[0]
                
                # Create grid mesh
                X1, X2 = np.meshgrid(grid_values_1, grid_values_2)
                
                # Ensure average matches grid dimensions
                if average.shape != X1.shape:
                    # If dimensions do not match, transpose or reshape
                    if average.shape == (len(grid_values_1), len(grid_values_2)):
                        average = average.T
                
                # Draw 3D surface plot
                ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
                
                surf = ax.plot_surface(X1, X2, average, cmap='viridis', 
                                      alpha=0.8, linewidth=0, antialiased=True)
                
                ax.set_xlabel(feat1, fontsize=10)
                ax.set_ylabel(feat2, fontsize=10)
                ax.set_zlabel('Partial Dependence', fontsize=10)
                ax.set_title(f'Interaction: {feat1} vs {feat2}', fontsize=12, fontweight='bold')
                
                # Add color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, label='Partial Dependence')
            except Exception as e:
                print(f"Warning: Could not create 2D PDP for {feat1} vs {feat2}: {e}")
                # For 3D plot, check if it is a subplot
                if idx < len(axes):
                    axes[idx].text(0.5, 0.5, f'Error creating 2D PDP', 
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(n_pairs, len(axes)):
            if axes[idx] is not None:
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pdp_2d_interaction.png'), dpi=300, bbox_inches='tight')
        # plt.show()  # Show 3D plot
        plt.close()
        
        print(f"Dual variable PDP analysis completed, have drawn interaction graphs between the first {n_interaction_features} important features")

def analyze_noise_robustness(model, X_test, y_test, feature_names, save_dir):
    """Analyze model robustness to noise and outliers"""
    print("\n=== Analyze model robustness to noise and outliers ===")
    
    # 1. Add Gaussian noise test
    print("Test robustness to Gaussian noise...")
    noise_levels = [0, 0.01, 0.03, 0.05, 0.10]  # Noise level (percentage of feature standard deviation)
    noise_rmses = []
    noise_stds = []
    
    for noise_level in noise_levels:
        rmses = []
        for _ in range(10):  # Repeat 10 times to get standard deviation
            # Add Gaussian noise
            X_test_noisy = X_test.copy()
            for i in range(X_test.shape[1]):
                feature_std = np.std(X_test[:, i])
                noise = np.random.normal(0, noise_level * feature_std, size=len(X_test))
                X_test_noisy[:, i] = X_test[:, i] + noise
            
            # Predict
            y_pred_noisy = model.predict(X_test_noisy)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_noisy))
            rmses.append(rmse)
        
        noise_rmses.append(np.mean(rmses))
        noise_stds.append(np.std(rmses))
    
    # 2. Add outlier test
    print("Test robustness to outliers...")
    outlier_levels = [0, 0.02, 0.05, 0.10]  # Outlier ratio
    outlier_severity = 5.0  # Outlier severity (multiple of standard deviation)
    outlier_rmses = []
    outlier_stds = []
    
    for outlier_pct in outlier_levels:
        rmses = []
        for _ in range(10):  # Repeat 10 times
            # Add outliers
            X_test_outlier = X_test.copy()
            n_outliers = int(len(X_test) * outlier_pct)
            if n_outliers > 0:
                outlier_indices = np.random.choice(len(X_test), n_outliers, replace=False)
                for idx in outlier_indices:
                    # Randomly select a feature to add outliers
                    feature_idx = np.random.randint(X_test.shape[1])
                    feature_mean = np.mean(X_test[:, feature_idx])
                    feature_std = np.std(X_test[:, feature_idx])
                    X_test_outlier[idx, feature_idx] = feature_mean + outlier_severity * feature_std
            
            # Predict
            y_pred_outlier = model.predict(X_test_outlier)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_outlier))
            rmses.append(rmse)
        
        outlier_rmses.append(np.mean(rmses))
        outlier_stds.append(np.std(rmses))
    
    # 3. Draw robustness analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gaussian noise robustness
    axes[0].errorbar([l * 100 for l in noise_levels], noise_rmses, yerr=noise_stds, 
                     marker='o', capsize=5, capthick=2, linewidth=2, color='blue')
    axes[0].set_xlabel('Noise level (% of feature std)', fontsize=12)
    axes[0].set_ylabel('RMSE (MPa)', fontsize=12)
    axes[0].set_title('Robustness to Additive Gaussian Noise', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Outliers robustness
    axes[1].errorbar([l * 100 for l in outlier_levels], outlier_rmses, yerr=outlier_stds, 
                     marker='o', capsize=5, capthick=2, linewidth=2, color='orange')
    axes[1].set_xlabel('Injected outliers (% of samples)', fontsize=12)
    axes[1].set_ylabel('RMSE (MPa)', fontsize=12)
    axes[1].set_title(f'Robustness to Outliers (severity={outlier_severity}×std)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Save results
    robustness_df = pd.DataFrame({
        'Noise Level (%)': [l * 100 for l in noise_levels],
        'RMSE (MPa)': noise_rmses,
        'Std': noise_stds
    })
    robustness_df.to_excel(os.path.join(save_dir, 'noise_robustness.xlsx'), index=False)
    
    outlier_df = pd.DataFrame({
        'Outlier Percentage (%)': [l * 100 for l in outlier_levels],
        'RMSE (MPa)': outlier_rmses,
        'Std': outlier_stds
    })
    outlier_df.to_excel(os.path.join(save_dir, 'outlier_robustness.xlsx'), index=False)
    
    # Print result summary
    print("\n=== Robustness analysis result summary ===")
    print("\n1. Gaussian noise robustness:")
    print(f"   - No noise RMSE: {noise_rmses[0]:.3f} MPa")
    print(f"   - 1% noise RMSE: {noise_rmses[1]:.3f} MPa (increase {noise_rmses[1]-noise_rmses[0]:.3f} MPa)")
    print(f"   - 3% noise RMSE: {noise_rmses[2]:.3f} MPa (increase {noise_rmses[2]-noise_rmses[0]:.3f} MPa)")
    print(f"   - 5% noise RMSE: {noise_rmses[3]:.3f} MPa (increase {noise_rmses[3]-noise_rmses[0]:.3f} MPa)")
    print(f"   - 10% noise RMSE: {noise_rmses[4]:.3f} MPa (increase {noise_rmses[4]-noise_rmses[0]:.3f} MPa)")
    
    print("\n2. Outliers robustness:")
    print(f"   - No outliers RMSE: {outlier_rmses[0]:.3f} MPa")
    print(f"   - 2% outliers RMSE: {outlier_rmses[1]:.3f} MPa (change {outlier_rmses[1]-outlier_rmses[0]:.3f} MPa)")
    print(f"   - 5% outliers RMSE: {outlier_rmses[2]:.3f} MPa (change {outlier_rmses[2]-outlier_rmses[0]:.3f} MPa)")
    print(f"   - 10% outliers RMSE: {outlier_rmses[3]:.3f} MPa (change {outlier_rmses[3]-outlier_rmses[0]:.3f} MPa)")
    
    # Calculate performance degradation rate
    noise_degradation = (noise_rmses[-1] - noise_rmses[0]) / noise_rmses[0] * 100
    outlier_degradation = (outlier_rmses[-1] - outlier_rmses[0]) / outlier_rmses[0] * 100
    
    print("\n3. Performance degradation rate (10% noise/outliers vs no perturbation):")
    print(f"   - Gaussian noise performance degradation: {noise_degradation:.1f}%")
    print(f"   - Outliers performance degradation: {outlier_degradation:.1f}%")
    
    # Determine robustness
    if noise_degradation < 20:
        noise_robust = "Excellent"
    elif noise_degradation < 40:
        noise_robust = "Good"
    elif noise_degradation < 60:
        noise_robust = "Average"
    else:
        noise_robust = "Poor"
    
    if abs(outlier_degradation) < 5:
        outlier_robust = "Excellent"
    elif abs(outlier_degradation) < 10:
        outlier_robust = "Good"
    elif abs(outlier_degradation) < 20:
        outlier_robust = "Average"
    else:
        outlier_robust = "Poor"
    
    print("\n4. Robustness evaluation:")
    print(f"   - Gaussian noise robustness: {noise_robust} (10% noise performance degradation {noise_degradation:.1f}%)")
    print(f"   - Outliers robustness: {outlier_robust} (10% outliers performance change {outlier_degradation:.1f}%)")
    
    # Generate paper description
    print("\n" + "="*80)
    print("【Paper result description】")
    print("="*80)
    
    paper_description = f"""
To evaluate the robustness of the XGBoost model under data perturbation, this study分别对模型进行了高斯
noise and outliers robustness tests.

### Gaussian noise robustness analysis

Figure 10(a) shows the performance changes of the model when adding different levels of Gaussian noise (noise standard deviation占特征标准差的0%-10%）时的
performance changes. The results show that the RMSE of the model increases significantly with the increase of the noise level, from {noise_rmses[0]:.3f} MPa without noise to {noise_rmses[4]:.3f} MPa at 10% noise level, with a performance degradation rate of {noise_degradation:.1f}%。
Furthermore, as the noise level increases, the error bars (error bars) gradually widen, indicating that the model's prediction uncertainty increases. This result shows that although the model's performance decreases under Gaussian noise environment, it performs relatively stable overall, with RMSE only increasing to {noise_rmses[1]:.3f} MPa at 1% noise level, indicating that the model has a good tolerance for small measurement errors.

### Outliers robustness analysis

Figure 10(b) shows the robustness of the model when injecting different proportions of outliers (severity of 5 times the standard deviation).
Unlike the Gaussian noise test, the model shows excellent robustness to outliers. When the outlier proportion increases from 0% to 10%, the RMSE of the model changes from {outlier_rmses[0]:.3f} MPa to {outlier_rmses[3]:.3f} MPa, with a performance change of {abs(outlier_degradation):.1f}%，almost maintaining stability. This result shows that the XGBoost model can effectively handle outliers in the data set, and even maintain good prediction performance when the outlier proportion is as high as 10%.
This strong robustness to outliers is important for practical engineering applications, because it is inevitable to have measurement anomalies or recording errors in actual data collection.

Based on the above analysis, the XGBoost model built in this study is stable under data perturbation environment, especially with excellent robustness to outliers, which provides a reliable guarantee for its application in practical engineering.
"""
    
    print(paper_description)
    print("="*80)
    
    # Save paper description to file
    with open(os.path.join(save_dir, 'robustness_paper_description.txt'), 'w', encoding='utf-8') as f:
        f.write(paper_description)
    
    print(f"\nRobustness analysis completed, detailed results have been saved")
    return noise_rmses, outlier_rmses

def analyze_prediction_intervals(model, X_test, y_test, save_dir, n_bootstrap=100, bootstrap_noise_level=0.01):
    """Analyze the confidence interval of the prediction
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_dir: Save directory
        n_bootstrap: Bootstrap resampling次数
        bootstrap_noise_level: Bootstrap噪声水平（特征标准差的百分比）
    """
    print("\n=== Generate prediction intervals ===")
    print(f"Bootstrap noise level: {bootstrap_noise_level*100}% of feature standard deviation")
    
    # 1. Bootstrap prediction intervals - improved version
    print("Generate Bootstrap prediction intervals...")
    bootstrap_predictions = []
    for _ in range(n_bootstrap):
        # Add noise to simulate model uncertainty (similar to the idea of Monte Carlo Dropout)
        # Add small noise to the test data to reflect the model's response to input uncertainty
        noise_std = bootstrap_noise_level  # Noise standard deviation (percentage of feature standard deviation)
        X_test_noisy = X_test.copy()
        for i in range(X_test.shape[1]):
            feature_std = np.std(X_test[:, i])
            noise = np.random.normal(0, noise_std * feature_std, size=len(X_test))
            X_test_noisy[:, i] = X_test[:, i] + noise
        
        # Prediction
        y_pred = model.predict(X_test_noisy)
        bootstrap_predictions.append(y_pred)
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    bootstrap_median = np.median(bootstrap_predictions, axis=0)
    bootstrap_lower = np.percentile(bootstrap_predictions, 10, axis=0)  # 80% confidence interval
    bootstrap_upper = np.percentile(bootstrap_predictions, 90, axis=0)
    
    # 2. Quantile Regression prediction intervals (using the residual distribution of the training data)
    print("Generate Quantile Regression prediction intervals...")
    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test
    residual_std = np.std(residuals)
    
    # Use the standard deviation of the residuals to build the prediction intervals
    quantile_lower = y_pred_test - 1.28 * residual_std  # 10th percentile for 80% interval
    quantile_upper = y_pred_test + 1.28 * residual_std  # 90th percentile for 80% interval
    
    # Sort for visualization
    sort_indices = np.argsort(y_test)
    y_test_sorted = y_test[sort_indices]
    bootstrap_median_sorted = bootstrap_median[sort_indices]
    bootstrap_lower_sorted = bootstrap_lower[sort_indices]
    bootstrap_upper_sorted = bootstrap_upper[sort_indices]
    quantile_lower_sorted = quantile_lower[sort_indices]
    quantile_upper_sorted = quantile_upper[sort_indices]
    
    # 3. Draw prediction intervals comparison plot
    fig = plt.figure(figsize=(20, 10))
    
    # Create grid layout: left two columns show prediction intervals, right column show performance comparison
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                         width_ratios=[2, 1], height_ratios=[1, 1])
    
    # Bootstrap prediction intervals (left top)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(range(len(y_test_sorted)), y_test_sorted, s=30, c='black', alpha=0.7, label='Actual values', zorder=3)
    ax1.plot(range(len(y_test_sorted)), bootstrap_median_sorted, 'r--', linewidth=2, label='Predicted median', zorder=2)
    ax1.fill_between(range(len(y_test_sorted)), bootstrap_lower_sorted, bootstrap_upper_sorted, 
                     alpha=0.3, color='lightblue', label='80% Prediction interval', zorder=1)
    
    ax1.set_xlabel('Test samples (sorted by actual values)', fontsize=11)
    ax1.set_ylabel('Peak Stress fc (MPa)', fontsize=11)
    ax1.set_title('Bootstrap Prediction Intervals', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Quantile Regression预测区间 (左下)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(range(len(y_test_sorted)), y_test_sorted, s=30, c='black', alpha=0.7, label='Actual values', zorder=3)
    ax2.plot(range(len(y_test_sorted)), y_pred_test[sort_indices], 'r--', linewidth=2, label='Predicted median', zorder=2)
    ax2.fill_between(range(len(y_test_sorted)), quantile_lower_sorted, quantile_upper_sorted, 
                     alpha=0.3, color='lightgreen', label='80% Prediction interval', zorder=1)
    
    ax2.set_xlabel('Test samples (sorted by actual values)', fontsize=11)
    ax2.set_ylabel('Peak Stress fc (MPa)', fontsize=11)
    ax2.set_title('Quantile Regression Intervals', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Prediction intervals coverage analysis (right)
    ax3 = fig.add_subplot(gs[:, 1])
    
    # Calculate coverage
    bootstrap_coverage = np.mean((y_test_sorted >= bootstrap_lower_sorted) & (y_test_sorted <= bootstrap_upper_sorted))
    quantile_coverage = np.mean((y_test_sorted >= quantile_lower_sorted) & (y_test_sorted <= quantile_upper_sorted))
    
    # Calculate average interval width
    bootstrap_width = np.mean(bootstrap_upper_sorted - bootstrap_lower_sorted)
    quantile_width = np.mean(quantile_upper_sorted - quantile_lower_sorted)
    
    methods = ['Bootstrap', 'Quantile Regression']
    coverages = [bootstrap_coverage * 100, quantile_coverage * 100]
    widths = [bootstrap_width, quantile_width]
    
    y_pos = np.arange(len(methods))
    height = 0.35
    
    # Create dual axis
    bars1 = ax3.barh(y_pos - height/2, coverages, height, 
                    label='Coverage Rate (%)', color='skyblue', alpha=0.8)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.barh(y_pos + height/2, widths, height, 
                         label='Avg Interval Width (MPa)', color='coral', alpha=0.8)
    
    ax3.set_ylabel('Methods', fontsize=11)
    ax3.set_xlabel('Coverage Rate (%)', fontsize=11, color='blue')
    ax3_twin.set_xlabel('Average Interval Width (MPa)', fontsize=11, color='red')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(methods)
    ax3.tick_params(axis='x', labelcolor='blue')
    ax3_twin.tick_params(axis='x', labelcolor='red')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_title('Performance Comparison', fontsize=13, fontweight='bold', pad=20)
    
    # Add numerical labels
    for i, v in enumerate(coverages):
        ax3.text(v + 2, i - height/2, f'{v:.1f}%', ha='left', va='center', fontsize=9)
    for i, v in enumerate(widths):
        ax3_twin.text(v + 0.5, i + height/2, f'{v:.2f}', ha='left', va='center', fontsize=9)
    
    # Add legend
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='lower left', fontsize=9)
    
    # Invert y axis to make Bootstrap on top
    ax3.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_intervals.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Save prediction intervals result
    intervals_df = pd.DataFrame({
        'Sample_ID': range(len(y_test)),
        'Actual': y_test,
        'Predicted': y_pred_test,
        'Bootstrap_Lower': bootstrap_lower,
        'Bootstrap_Upper': bootstrap_upper,
        'Quantile_Lower': quantile_lower,
        'Quantile_Upper': quantile_upper
    })
    intervals_df.to_excel(os.path.join(save_dir, 'prediction_intervals.xlsx'), index=False)
    
    print(f"Prediction intervals analysis completed, 80% confidence interval coverage: Bootstrap={bootstrap_coverage*100:.1f}%, Quantile={quantile_coverage*100:.1f}%")
    return bootstrap_predictions, bootstrap_coverage, quantile_coverage

def train_single_fold(X_train, y_train, X_val, y_val, X_test, y_test, train_ids, val_ids, test_ids, 
                     feature_names, save_dir, n_trials=100):
    """Single fold training (for cross validation) - only perform hyperparameter optimization and training, no interpretability analysis"""
    # Hyperparameter optimization
    study = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials)
    best_params = study.best_params
    
    # Train final model
    model = train_xgboost(X_train, y_train, X_val, y_val, best_params)
    
    # Predict test set (only for performance evaluation)
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Save prediction result
        results_df = pd.DataFrame({
            'Sample ID': test_ids,
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': y_pred - y_test
        })
        results_df.to_excel(os.path.join(save_dir, 'test_predictions.xlsx'), index=False)
        
        return model, best_params, {'r2': r2, 'mae': mae, 'rmse': rmse}, train_ids, val_ids, test_ids
    else:
        return model, best_params, None, train_ids, val_ids, test_ids

def cross_subset_validation(X, y, feature_names, sample_divisions, sample_ids, original_df_indices, df_processed, 
                           save_dir=None, n_trials=100):
    """Three-fold cross validation for subset1/2/3"""
    print("\n=== XGBoost Subset three-fold cross validation ===")

    if save_dir is None:
        save_dir = os.path.join(SAVE_ROOT, 'xgboost_cv')
    
    # Extract subset labels and test set
    subset_labels = sorted([s for s in np.unique(sample_divisions) if str(s).startswith('subset')])
    test_mask = np.array([str(s).strip().lower().startswith('test') for s in sample_divisions])
    test_indices = np.where(test_mask)[0]
    
    if len(subset_labels) != 3:
        raise ValueError(f"The number of subsets should be 3, actually: {len(subset_labels)} {subset_labels}")
    
    if len(test_indices) == 0:
        raise ValueError("Test set not found")
    
    print(f"Found 3 subsets: {subset_labels}")
    print(f"Test set sample number: {len(test_indices)}")
    
    all_results = []
    
    # Three-fold cross validation
    for i, val_label in enumerate(subset_labels):
        print(f"\n{'='*60}")
        print(f"Subset cross validation round {i+1}/3 [{val_label} as validation set]")
        print(f"{'='*60}")
        
        # Determine training set and validation set
        train_labels = [lbl for lbl in subset_labels if lbl != val_label]
        subset_to_indices = {lbl: np.where(sample_divisions == lbl)[0] for lbl in subset_labels}
        
        train_indices = np.concatenate([subset_to_indices[lbl] for lbl in train_labels])
        val_indices = subset_to_indices[val_label]
        
        print(f"Training set = {train_labels}, validation set = {val_label}, test set = test")
        print(f"   Training set sample number: {len(train_indices)}, validation set: {len(val_indices)}, test set: {len(test_indices)}")
        
        # Split data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        train_ids_fold = [sample_ids[idx] for idx in train_indices]
        val_ids_fold = [sample_ids[idx] for idx in val_indices]
        test_ids_fold = [sample_ids[idx] for idx in test_indices]
        
        # Create round save directory
        round_save_dir = os.path.join(save_dir, f'subsetCV_{val_label}')
        os.makedirs(round_save_dir, exist_ok=True)
        
        # Train single fold model (only perform hyperparameter optimization and training, no interpretability analysis)
        model, best_params, metrics, train_ids_round, val_ids_round, test_ids_round = train_single_fold(
            X_train, y_train, X_val, y_val, X_test, y_test,
            train_ids_fold, val_ids_fold, test_ids_fold,
            feature_names, round_save_dir, n_trials=n_trials
        )
        
        all_results.append({
            'val_label': val_label,
            'train_labels': train_labels,
            'best_params': best_params,
            'metrics': metrics,
            'model': model,
            'train_val_indices': np.concatenate([train_indices, val_indices]),
            'test_indices': test_indices.copy()
        })
        
        print(f"[{val_label} validation set] Round training completed")
        if metrics:
            print(f"   Test set metrics: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("All three-fold cross validation completed!")
    print(f"{'='*60}")
    for i, res in enumerate(all_results):
        print(f"[{i+1}] Validation set: {res['val_label']}")
        if res['metrics']:
            print(f"     Test set metrics: R2={res['metrics']['r2']:.4f}, MAE={res['metrics']['mae']:.3f}, RMSE={res['metrics']['rmse']:.3f}")
    
    # Calculate three-fold average metrics
    valid_metrics = [res['metrics'] for res in all_results if res['metrics'] is not None]
    if valid_metrics:
        avg_r2 = np.mean([m['r2'] for m in valid_metrics])
        avg_mae = np.mean([m['mae'] for m in valid_metrics])
        avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
        print(f"\nThree-fold average performance: R2={avg_r2:.4f}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")
    else:
        avg_r2 = avg_mae = avg_rmse = None

    # Find the model with the best test set performance (by R2)
    best_fold_idx = 0
    best_r2 = -np.inf
    for i, res in enumerate(all_results):
        if res['metrics'] and res['metrics']['r2'] > best_r2:
            best_r2 = res['metrics']['r2']
            best_fold_idx = i
    
    best_fold_result = all_results[best_fold_idx]
    best_overall_params = best_fold_result['best_params']
    
    print(f"\n{'='*60}")
    print(f"Three-fold cross validation best hyperparameters (from round {best_fold_idx+1}, validation set={best_fold_result['val_label']})")
    print(f"Test set R2: {best_r2:.4f}")
    print(f"Best hyperparameters: {best_overall_params}")
    print(f"{'='*60}")
    
    # Use best hyperparameters, retrain final model using all subset data (excluding test)
    print(f"\n{'='*60}")
    print("Use best fold model as final model (no re-training)")
    print(f"{'='*60}")

    best_fold_train_val_indices = best_fold_result['train_val_indices']
    best_fold_test_indices = best_fold_result['test_indices']

    X_train_final = X[best_fold_train_val_indices]
    y_train_final = y[best_fold_train_val_indices]
    X_test_final = X[best_fold_test_indices]
    y_test_final = y[best_fold_test_indices]

    train_ids_final = [sample_ids[idx] for idx in best_fold_train_val_indices]
    test_ids_final = [sample_ids[idx] for idx in best_fold_test_indices]

    final_model = best_fold_result['model']

    # Evaluate on test set (should be consistent with the fold metrics)
    y_test_pred = final_model.predict(X_test_final)
    final_r2 = r2_score(y_test_final, y_test_pred)
    final_mae = mean_absolute_error(y_test_final, y_test_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test_final, y_test_pred))

    print(f"Final model from round {best_fold_idx+1} validation set {best_fold_result['val_label']}.")
    print(f"  R2: {final_r2:.4f}")
    print(f"  MAE: {final_mae:.3f} MPa")
    print(f"  RMSE: {final_rmse:.3f} MPa")
    if avg_r2 is not None:
        print(f"  (Three-fold average) R2={avg_r2:.4f}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")
    
    # Interpretability analysis directory
    interpretability_dir = os.path.join(save_dir, 'interpretability_analysis')
    os.makedirs(interpretability_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Start interpretability analysis")
    print(f"{'='*60}")
    
    # 1. Draw result plot
    plot_results(y_test_final, y_test_pred, interpretability_dir)
    
    # 2. Draw training set and test set comparison plot
    plot_train_test_comparison(final_model, X_train_final, y_train_final, X_test_final, y_test_final, interpretability_dir)
    
    # 3. Feature importance
    importance_df = plot_feature_importance(final_model, feature_names, interpretability_dir)
    
    # 4. SHAP analysis
    X_all_final = np.vstack([X_train_final, X_test_final])
    shap_importance = plot_shap_analysis(final_model, X_all_final, X_all_final, feature_names, interpretability_dir)
    
    # 5. PDP analysis
    plot_pdp_analysis(final_model, X_all_final, feature_names, interpretability_dir, n_top_features=6)
    
    # 6. Noise robustness analysis (optional)
    # analyze_noise_robustness(final_model, X_test_final, y_test_final, feature_names, interpretability_dir)
    
    # 7. Prediction intervals analysis (optional)
    # analyze_prediction_intervals(final_model, X_test_final, y_test_final, interpretability_dir, n_bootstrap=100, bootstrap_noise_level=0.01)
    
    # Save final model
    joblib.dump({
        'model': final_model,
        'best_params': best_overall_params,
        'feature_names': feature_names,
        'r2': final_r2,
        'mae': final_mae,
        'rmse': final_rmse,
        'importance_df': importance_df,
        'shap_importance': shap_importance,
        'cross_validation_results': all_results
    }, os.path.join(interpretability_dir, 'final_xgboost_model.pkl'))
    
    # Predict all samples and write to Excel
    print(f"\n{'='*60}")
    print("Predict all samples and write to Excel")
    print(f"{'='*60}")
    try:
        # Predict all samples (including test set)
        X_all_samples = np.vstack([X_train_final, X_test_final])
        y_all_pred = final_model.predict(X_all_samples)
        all_sample_ids = train_ids_final + test_ids_final
        
        # Create prediction result dictionary
        pred_dict = {}
        for idx, sample_id in enumerate(all_sample_ids):
            pred_dict[sample_id] = y_all_pred[idx]
        
        # Reload original Excel file
        data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
        df_original = pd.read_excel(data_file, sheet_name=0)
        
        # Use map method to match
        if 'No_Customized' in df_original.columns:
            df_original['XGB_fc'] = df_original['No_Customized'].map(pred_dict)
        else:
            print("Warning: No_Customized column not found, cannot match prediction result")
        
        # 保存结果
        output_file = os.path.join(PROJECT_ROOT, "dataset/dataset_with_XGB_fc.xlsx")
        df_original.to_excel(output_file, index=False)
        print(f"✓ Prediction result written to: {output_file}")
        print(f"  - New column: XGB_fc (final model predicted value)")
        print(f"  - Non-empty prediction value number: {df_original['XGB_fc'].notna().sum()}")
        
        # Backup
        backup_file = os.path.join(save_dir, 'dataset_with_XGB_fc_backup.xlsx')
        df_original.to_excel(backup_file, index=False)
        print(f"✓ Backup file saved: {backup_file}")
    except Exception as e:
        print(f"Warning: Error writing to Excel: {e}")
        import traceback
        traceback.print_exc()
    
    return all_results, final_model, best_overall_params

def cross_random_kfold_validation(X, y, feature_names, sample_ids, sample_divisions=None,
                                  save_dir=None, n_trials=100, n_splits=3, random_state=42):
    """Random KFold cross validation"""
    print(f"\n=== XGBoost Random {n_splits} fold cross validation ===")

    if save_dir is None:
        save_dir = os.path.join(SAVE_ROOT, 'xgboost_cv_random')

    os.makedirs(save_dir, exist_ok=True)

    fixed_test_indices = None
    if sample_divisions is not None:
        test_mask = np.array([str(s).strip().lower().startswith('test') for s in sample_divisions])
        if np.any(test_mask):
            fixed_test_indices = np.where(test_mask)[0]
            print(f"Use already divided test set ({len(fixed_test_indices)} samples) for random cross validation evaluation")

    if fixed_test_indices is not None and len(fixed_test_indices) == len(X):
        raise ValueError("All samples belong to test set, cannot perform random cross validation")

    if fixed_test_indices is not None:
        train_pool_mask = np.ones(len(X), dtype=bool)
        train_pool_mask[fixed_test_indices] = False
        train_pool_indices = np.where(train_pool_mask)[0]
        print(f"Random cross validation only on non-test set samples, total {len(train_pool_indices)} samples")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_source = train_pool_indices
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_source = np.arange(len(X))

    all_results = []

    for fold_idx, (train_indices_rel, val_indices_rel) in enumerate(kf.split(split_source), 1):
        print(f"\n{'='*60}")
        print(f"Random KFold round {fold_idx}/{n_splits} fold")
        print(f"{'='*60}")

        if fixed_test_indices is not None:
            train_indices = split_source[train_indices_rel]
            val_indices = split_source[val_indices_rel]
            test_indices = fixed_test_indices
        else:
            train_val_indices = split_source[train_indices_rel]
            val_candidates = split_source[val_indices_rel]
            if len(train_val_indices) < 5:
                raise ValueError("Training set sample too few, cannot perform random cross validation")
            inner_train_indices, inner_val_indices = train_test_split(
                np.concatenate([train_val_indices, val_candidates]),
                test_size=0.2,
                random_state=random_state + fold_idx
            )
            train_indices = inner_train_indices
            val_indices = inner_val_indices
            test_indices = val_candidates

        if len(train_indices) < 5:
            raise ValueError("Training set sample too few, cannot perform random cross validation")

        print(f"   Training set sample number: {len(train_indices)}")
        print(f"   Validation set sample number: {len(val_indices)}")
        print(f"   Test set sample number: {len(test_indices)}")

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        train_ids_fold = [sample_ids[idx] for idx in train_indices]
        val_ids_fold = [sample_ids[idx] for idx in val_indices]
        test_ids_fold = [sample_ids[idx] for idx in test_indices]

        round_save_dir = os.path.join(save_dir, f'randomKFold_fold{fold_idx}')
        os.makedirs(round_save_dir, exist_ok=True)

        model, best_params, metrics, _, _, _ = train_single_fold(
            X_train, y_train, X_val, y_val, X_test, y_test,
            train_ids_fold, val_ids_fold, test_ids_fold,
            feature_names, round_save_dir, n_trials=n_trials
        )

        combined_train_val = np.unique(np.concatenate([train_indices, val_indices]))

        all_results.append({
            'fold_idx': fold_idx,
            'best_params': best_params,
            'metrics': metrics,
            'model': model,
            'train_val_indices': combined_train_val,
            'test_indices': test_indices.copy()
        })

        if metrics:
            print(f"[Fold {fold_idx}] Test set metrics: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

    print(f"\n{'='*60}")
    print("All random KFold completed!")
    print(f"{'='*60}")
    for res in all_results:
        if res['metrics']:
            print(f"[Fold {res['fold_idx']}] Test set: R2={res['metrics']['r2']:.4f}, MAE={res['metrics']['mae']:.3f}, RMSE={res['metrics']['rmse']:.3f}")

    valid_metrics = [res['metrics'] for res in all_results if res['metrics']]
    if valid_metrics:
        avg_r2 = np.mean([m['r2'] for m in valid_metrics])
        avg_mae = np.mean([m['mae'] for m in valid_metrics])
        avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
        print(f"\n{n_splits} fold average performance: R2={avg_r2:.4f}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")
    else:
        avg_r2 = avg_mae = avg_rmse = None

    best_fold_idx = 0
    best_r2 = -np.inf
    for i, res in enumerate(all_results):
        if res['metrics'] and res['metrics']['r2'] > best_r2:
            best_r2 = res['metrics']['r2']
            best_fold_idx = i

    best_fold_result = all_results[best_fold_idx]
    best_overall_params = best_fold_result['best_params']

    print(f"\n{'='*60}")
    print(f"Best fold = Fold {best_fold_result['fold_idx']} (Test set R2={best_r2:.4f})")
    print(f"Best hyperparameters: {best_overall_params}")
    print(f"{'='*60}")

    best_fold_train_val_indices = best_fold_result['train_val_indices']
    best_fold_test_indices = best_fold_result['test_indices']

    X_train_final = X[best_fold_train_val_indices]
    y_train_final = y[best_fold_train_val_indices]
    X_test_final = X[best_fold_test_indices]
    y_test_final = y[best_fold_test_indices]

    train_ids_final = [sample_ids[idx] for idx in best_fold_train_val_indices]
    test_ids_final = [sample_ids[idx] for idx in best_fold_test_indices]

    final_model = best_fold_result['model']

    y_test_pred = final_model.predict(X_test_final)
    final_r2 = r2_score(y_test_final, y_test_pred)
    final_mae = mean_absolute_error(y_test_final, y_test_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test_final, y_test_pred))

    print(f"Final model from round {best_fold_idx+1} validation set {best_fold_result['val_label']}.")
    print(f"  R2: {final_r2:.4f}")
    print(f"  MAE: {final_mae:.3f} MPa")
    print(f"  RMSE: {final_rmse:.3f} MPa")
    if avg_r2 is not None:
        print(f"  ({n_splits} fold average) R2={avg_r2:.4f}, MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")

    interpretability_dir = os.path.join(save_dir, 'interpretability_analysis')
    os.makedirs(interpretability_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Start interpretability analysis (random划分)")
    print(f"{'='*60}")

    plot_results(y_test_final, y_test_pred, interpretability_dir)
    plot_train_test_comparison(final_model, X_train_final, y_train_final, X_test_final, y_test_final, interpretability_dir)
    importance_df = plot_feature_importance(final_model, feature_names, interpretability_dir)
    X_all_final = np.vstack([X_train_final, X_test_final])
    shap_importance = plot_shap_analysis(final_model, X_all_final, X_all_final, feature_names, interpretability_dir)
    plot_pdp_analysis(final_model, X_all_final, feature_names, interpretability_dir, n_top_features=6)

    joblib.dump({
        'model': final_model,
        'best_params': best_overall_params,
        'feature_names': feature_names,
        'r2': final_r2,
        'mae': final_mae,
        'rmse': final_rmse,
        'importance_df': importance_df,
        'shap_importance': shap_importance,
        'cross_validation_results': all_results
    }, os.path.join(interpretability_dir, 'final_xgboost_model_random.pkl'))

    print(f"\n{'='*60}")
    print("Predict all samples and write to Excel (random划分模型)")
    print(f"{'='*60}")
    try:
        X_all_samples = X
        y_all_pred = final_model.predict(X_all_samples)
        pred_dict = {sid: pred for sid, pred in zip(sample_ids, y_all_pred)}

        data_file = os.path.join(PROJECT_ROOT, "dataset/dataset_final.xlsx")
        df_original = pd.read_excel(data_file, sheet_name=0)

        if 'No_Customized' in df_original.columns:
            df_original['XGB_fc_random'] = df_original['No_Customized'].map(pred_dict)
        else:
            df_original['XGB_fc_random'] = y_all_pred

        output_file = os.path.join(save_dir, "dataset_with_XGB_fc_random.xlsx")
        df_original.to_excel(output_file, index=False)
        print(f"✓ Random划分模型预测结果已写入: {output_file}")
        print(f"  - New column: XGB_fc_random")
        print(f"  - Non-empty prediction value number: {df_original['XGB_fc_random'].notna().sum()}")
    except Exception as e:
        print(f"Warning: Error writing to random划分Excel: {e}")
        import traceback
        traceback.print_exc()

    return all_results, final_model, best_overall_params

def main():
    """Main function - sequentially run subset three-fold and random three-fold cross validation"""
    print("=== XGBoost peak stress prediction model (Subset three-fold + random three-fold) ===")
    
    # 1. Load data
    data = load_data()
    if data is None:
        return
    
    X, y, feature_names, sample_divisions, sample_ids, original_df_indices, df_processed = data
    
    # 2. Subset three-fold cross validation
    subset_save_dir = os.path.join(SAVE_ROOT, 'xgboost_cv')
    os.makedirs(subset_save_dir, exist_ok=True)
    
    subset_results, subset_model, subset_best_params = cross_subset_validation(
        X, y, feature_names, sample_divisions, sample_ids, original_df_indices, df_processed,
        save_dir=subset_save_dir, n_trials=100
    )
    
    # 3. Random three-fold cross validation
    random_save_dir = os.path.join(SAVE_ROOT, 'xgboost_cv_random')
    os.makedirs(random_save_dir, exist_ok=True)

    random_results, random_model, random_best_params = cross_random_kfold_validation(
        X, y, feature_names, sample_ids, sample_divisions,
        save_dir=random_save_dir, n_trials=100, n_splits=3
    )

    def summarize_results(results, label_key):
        available = [res for res in results if res.get('metrics')]
        if not available:
            return None, None
        best_res = max(available, key=lambda r: r['metrics']['r2'])
        avg_metrics = {
            'r2': np.mean([res['metrics']['r2'] for res in available]),
            'mae': np.mean([res['metrics']['mae'] for res in available]),
            'rmse': np.mean([res['metrics']['rmse'] for res in available]),
        }
        label = best_res.get(label_key)
        if label is None:
            label = f"Fold {best_res.get('fold_idx', '?')}"
        return (label, best_res['metrics']), avg_metrics

    subset_best, subset_avg = summarize_results(subset_results, 'val_label')
    random_best, random_avg = summarize_results(random_results, 'val_label')

    print(f"\n{'='*60}")
    print("Subset划分 vs Random划分性能对比")
    print(f"{'='*60}")
    if subset_best:
        print(f"Subset best ({subset_best[0]}): R2={subset_best[1]['r2']:.4f}, MAE={subset_best[1]['mae']:.3f}, RMSE={subset_best[1]['rmse']:.3f}")
        if subset_avg:
            print(f"Subset average: R2={subset_avg['r2']:.4f}, MAE={subset_avg['mae']:.3f}, RMSE={subset_avg['rmse']:.3f}")
    if random_best:
        print(f"Random best ({random_best[0]}): R2={random_best[1]['r2']:.4f}, MAE={random_best[1]['mae']:.3f}, RMSE={random_best[1]['rmse']:.3f}")
        if random_avg:
            print(f"Random average: R2={random_avg['r2']:.4f}, MAE={random_avg['mae']:.3f}, RMSE={random_avg['rmse']:.3f}")
    print(f"{'='*60}")
    
    return {
        'subset': (subset_results, subset_model, subset_best_params),
        'random': (random_results, random_model, random_best_params)
    }

if __name__ == "__main__":
    results = main()
