#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost peak stress prediction model - with hyperparameter optimization
  Bayesian optimization using Optuna
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import shap
from sklearn.inspection import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set working directory - automatically find project root directory
def find_project_root():
    """Find project root directory"""
    # Try to locate from the script location first
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        # If the script is in a subdirectory, find the directory containing dataset or SAVE
        current_dir = script_dir
        while current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'dataset')) or \
               os.path.exists(os.path.join(current_dir, 'SAVE')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
    except NameError:
        # Running in Jupyter, search from the current directory upwards
        pass
    
    # Search from the current working directory upwards
    current_dir = os.path.abspath(os.getcwd())
    search_limit = 0
    while current_dir != os.path.dirname(current_dir) and search_limit < 5:
        # Check if the directory contains the dataset folder or specific project files
        if os.path.exists(os.path.join(current_dir, 'dataset')) or \
           os.path.exists(os.path.join(current_dir, 'SAVE')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
        search_limit += 1
    
    # If still not found, return the parent directory of the current directory
    return os.path.dirname(os.path.abspath(os.getcwd()))

# Set working directory
PROJECT_ROOT = find_project_root()
if not os.path.exists(os.path.join(PROJECT_ROOT, 'dataset')):
    # If still not found, use the current directory
    PROJECT_ROOT = os.getcwd()

os.chdir(PROJECT_ROOT)
print(f"Working directory set to: {PROJECT_ROOT}")

def load_data():
    """Load data"""
    print("=== 加载数据 ===")
    
    # Use the clustered data set
    clustered_file = os.path.join(PROJECT_ROOT, "dataset/本构模型预测项目数据集_聚类标记_数据集划分.xlsx")
    if not os.path.exists(clustered_file):
        print(f"Clustered file not found: {clustered_file}")
        # Fall back to the original file
        original_file = os.path.join(PROJECT_ROOT, "dataset/peak_data_extraction_result.xlsx")
        if not os.path.exists(original_file):
            print(f"Original data file not found: {original_file}")
            return None
        clustered_file = original_file
    
    df = pd.read_excel(clustered_file)
    print(f"Data shape: {df.shape}")
    
    # Define features and target variables
    feature_names = [
        'water(kg/m3)', 'cement(kg/m3)', 'water-cement ratio', 'natural sand(kg/m3)', 'coarse aggregate用量', 
        'replacement rate(%)', 'combined aggregate water absorption rate%', 'coarse aggregate maximum particle size(mm)', 'combined aggregate crushing index', 
        'loading speed(μe)', 'cement strength(MPa)', ' curing period(days)', 'chamfer ratio(prism is 0, cylinder is 1)', 
        'side length or diameter(mm)', 'height-diameter ratio'
    ]
    
    target_column = 'peak stress(fc)'
    
    # Check column names
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        print(f"Warning: missing columns {missing_cols}")
        feature_names = [col for col in feature_names if col in df.columns]
    
    # Extract data
    X = df[feature_names].values
    y = df[target_column].values
    
    # Extract sample division information
    sample_divisions = []
    if 'data set division' in df.columns:
        sample_divisions = df['data set division'].values
        print(f"Found data set division information: {np.unique(sample_divisions, return_counts=True)}")
    elif 'sample division' in df.columns:
        sample_divisions = df['sample division'].values
    else:
        print("No sample division information found, will use random division")
        sample_divisions = None
    
    # Extract sample IDs
    sample_ids = []
    if 'custom sample ID' in df.columns:
        sample_ids = df['custom sample ID'].values
    else:
        sample_ids = [f"sample_{i}" for i in range(len(X))]
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    return X, y, feature_names, sample_divisions, sample_ids

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
    
    # Classify samples based on sample division information
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, division in enumerate(sample_divisions):
        if division == 'train' or division == 'train set' or division.startswith('train'):
            train_indices.append(i)
        elif division == 'val' or division == 'validation set' or division == 'validation' or division.startswith('val'):
            val_indices.append(i)
        elif division == 'test' or division == 'test set' or division == 'test':
            test_indices.append(i)
        else:
            train_indices.append(i)
    
    # If there are only training set samples, need to split the validation set and test set from the training set
    if train_indices and not val_indices and not test_indices:
        print("Only training set samples, need to split the validation set and test set from the training set...")
        from sklearn.model_selection import train_test_split
        
        train_indices = np.array(train_indices)
        
        if len(train_indices) > 2:
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
            test_indices = []
    
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
    
    # Get the corresponding sample IDs
    train_ids = [sample_ids[i] for i in train_indices] if len(train_indices) > 0 else []
    val_ids = [sample_ids[i] for i in val_indices] if len(val_indices) > 0 else []
    test_ids = [sample_ids[i] for i in test_indices] if len(test_indices) > 0 else []
    
    print(f"Data split results:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna optimization objective function - using validation set evaluation"""
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        
        # Hyperparameter search space
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
    }
    
        # Use internal cross-validation on the training set to evaluate, more stable
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
    """Use Optuna to optimize XGBoost hyperparameters - using 5-fold cross-validation evaluation"""
    print(f"\n=== Start XGBoost hyperparameter optimization (n_trials={n_trials}) ===")
    print("Use 5-fold cross-validation to evaluate hyperparameters, improve stability")
    
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, n_jobs=1)
    
    print("\n=== Optimization completed ===")
    print(f"Best R² score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Visualize optimization history
    try:
        save_dir = os.path.join(PROJECT_ROOT, 'SAVE/xgboost_optimization')
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
    """Use best parameters to train the final XGBoost model"""
    print("\n=== Use best parameters to train the final XGBoost model ===")
    
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
    fig.suptitle(f'XGBoost Results (R²={r2:.4f})', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs true values
    axes[0,0].scatter(y_true, y_pred, alpha=0.7, s=60, color='green')
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('True Peak Stress (MPa)')
    axes[0,0].set_ylabel('Predicted Peak Stress (MPa)')
    axes[0,0].set_title(f'Prediction vs True Values\nR²={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = y_pred - y_true
    axes[0,1].scatter(y_pred, residuals, alpha=0.7, s=60, color='green')
    axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0,1].set_xlabel('Predicted Peak Stress (MPa)')
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
    # Note: Feature importance needs to be obtained from the model object after model training
    stats_text = f"""Model Performance Statistics:

Sample Count: {len(y_true)}
R²: {r2:.4f}
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
    
    # Plot bar chart
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
    
    # Use TreeExplainer（XGBoost专用）
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values（using test set）
    shap_values = explainer.shap_values(X_test)
    
    # Set font, avoid Chinese乱码
    original_font = plt.rcParams['font.sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    
    # 1. Summary plot
    fig = plt.figure(figsize=(20, 10))  # Increase width to 20
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    # Get current axes and adjust x-axis range
    ax = plt.gca()
    # Calculate appropriate x-axis range, increase margin for clearer display
    x_min, x_max = shap_values.flatten().min(), shap_values.flatten().max()
    margin = (x_max - x_min) * 0.3  # Increase from 10% to 30%, lengthen x-axis
    ax.set_xlim(x_min - margin, x_max + margin)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 2. Bar plot (Feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 3. Waterfall plot (Display SHAP values for a single sample)
    # 选择第一个样本作为示例
    shap_explanation = shap.Explanation(
        values=shap_values[0:1],
        base_values=explainer.expected_value,
        data=X_test[0:1],
        feature_names=feature_names
    )
    # Temporarily set font, avoid Chinese乱码
    original_font = plt.rcParams['font.sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Restore original font settings
    plt.rcParams['font.sans-serif'] = original_font
    
    # 4. Force plot (Display the prediction process for a single sample, optional)
    # If you want to see the force plot for all samples, uncomment the following code
    # shap.plots.force(shap_explanation[0], matplotlib=True, show=False)
    # plt.savefig(os.path.join(save_dir, 'shap_force.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()
    
    # 5. Calculate the average absolute SHAP value as feature importance
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    # Save SHAP importance
    shap_importance.to_excel(os.path.join(save_dir, 'shap_importance.xlsx'), index=False)
    
    # Restore original font settings
    plt.rcParams['font.sans-serif'] = original_font
    
    return shap_importance

def plot_pdp_analysis(model, X_train_df, feature_names, save_dir, n_top_features=5):
    """PDP（Partial Dependence Plot）analysis - using scikit-learn"""
    print("\n=== Perform PDP analysis ===")
    
    # Get the n most important features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(n_top_features)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    
    # 1. Single variable PDP analysis
    n_cols = 2
    n_rows = (n_top_features + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
    if n_top_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (feature, ax) in enumerate(zip(top_features, axes)):
        try:
            # Get feature index
            feature_idx = feature_names.index(feature)
            
            # Calculate partial dependence
            pd_result = partial_dependence(
                model, 
                X_train_df, 
                features=[feature_idx],
                kind='average'
            )
            
            # Plot PDP
            ax.plot(pd_result['grid_values'][0], pd_result['average'][0], 
                   linewidth=2, color='blue', marker='o', markersize=4)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=10)
            ax.set_title(f'PDP for {feature}', fontsize=12, fontweight='bold')
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
    plt.close()
    
    print(f"Single variable PDP analysis completed, the dependency plot of the first {n_top_features} important features has been drawn")
    
    # 2. Double variable PDP analysis（feature interaction）
    if n_top_features >= 2:
        print("\n=== Perform double variable PDP analysis ===")
        
        # Select the first 4 most important features for double variable interaction analysis
        n_interaction_features = min(4, n_top_features)
        interaction_features = top_features[:n_interaction_features]
        
        # Create interaction feature pairs
        interaction_pairs = []
        for i in range(len(interaction_features)):
            for j in range(i+1, len(interaction_features)):
                interaction_pairs.append((interaction_features[i], interaction_features[j]))
        
        # Plot double variable PDP（using heatmap）
        n_pairs = len(interaction_pairs)
        n_cols = 3  # Change to 3 columns, more美观
        n_rows = (n_pairs + 2) // 3  # Adjust row number calculation
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
                
                # Calculate double variable partial dependence
                pd_result = partial_dependence(
                    model,
                    X_train_df,
                    features=[feat1_idx, feat2_idx],
                    kind='average'
                )
                
                # Get grid values and average predicted values
                grid_values_1 = pd_result['grid_values'][0]
                grid_values_2 = pd_result['grid_values'][1]
                average = pd_result['average'][0]  # Take the first element（if it is a 3D array）
                
                # Ensure average is 2D
                if average.ndim == 3:
                    average = average[0]
                
                # Create grid
                X1, X2 = np.meshgrid(grid_values_1, grid_values_2)
                
                # Ensure average matches grid dimensions
                if average.shape != X1.shape:
                    # If dimensions do not match, transpose or reshape
                    if average.shape == (len(grid_values_1), len(grid_values_2)):
                        average = average.T
                
                # Plot 3D surface graph
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
                # For 3D graphs, check if it is a subplot
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
        plt.show()  # Display 3D graph
        plt.close()
        
        print(f"Double variable PDP analysis completed, the interaction plot of the first {n_interaction_features} important features has been drawn")

def main():
    """Main function"""
    print("=== XGBoost peak stress prediction model (with hyperparameter optimization) ===")
    
    # 1. Load data
    data = load_data()
    if data is None:
        return
    
    X, y, feature_names, sample_divisions, sample_ids = data
    
    # 2. Data split
    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = split_data_by_divisions(
        X, y, sample_divisions, sample_ids, test_ratio=0.2, val_ratio=0.2, random_state=42
    )
    
    # 3. Hyperparameter optimization（using validation set evaluation）
    study = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=100)
    best_params = study.best_params
    
    # 4. Train the final model
    model = train_xgboost(X_train, y_train, X_val, y_val, best_params)
    
    # 5. Predict test set
    if len(X_test) > 0:
        print(f"\n=== Predict test set ({len(X_test)} samples) ===")
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test set performance:")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.3f} MPa")
        print(f"  RMSE: {rmse:.3f} MPa")
        
        # 6. Plot results
        save_dir = os.path.join(PROJECT_ROOT, 'SAVE/xgboost_optimization')
        os.makedirs(save_dir, exist_ok=True)
        
        plot_results(y_test, y_pred, save_dir)
        importance_df = plot_feature_importance(model, feature_names, save_dir)
        
        # 7. SHAP and PDP analysis（using all data）
        # Merge all data for explanatory analysis
        X_all = np.vstack([X_train, X_val, X_test])
        
        # Prepare data（convert to DataFrame for PDP）
        X_all_df = pd.DataFrame(X_all, columns=feature_names)
        
        # SHAP analysis（using all data）
        shap_importance = plot_shap_analysis(model, X_all, X_all, feature_names, save_dir)
        
        # PDP analysis（using all data）
        plot_pdp_analysis(model, X_all, feature_names, save_dir, n_top_features=6)
        
        # 8. Save model and results
        import joblib
        joblib.dump({
            'model': model,
            'best_params': best_params,
            'feature_names': feature_names,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'importance_df': importance_df,
            'shap_importance': shap_importance
        }, os.path.join(save_dir, 'xgboost_model.pkl'))
        
        print(f"\nModel has been saved to: {save_dir}")
        
        # 9. Save prediction results
        results_df = pd.DataFrame({
            'Sample ID': test_ids,
            'True value': y_test,
            'Predicted value': y_pred,
            'Residual': y_pred - y_test
        })
        results_df.to_excel(os.path.join(save_dir, 'test_predictions.xlsx'), index=False)
        
        return r2, mae, rmse
    else:
        print("Test set is empty, cannot predict")
        return None, None, None

if __name__ == "__main__":
    results = main()
