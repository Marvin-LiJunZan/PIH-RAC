#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peak strain comparison analysis script
Calculate the R², RMSE, MSE, EVS, MAPE, MPE, etc. of other columns with the peak_strain column
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)

# Configure paths
PROJECT_ROOT = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM"
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "dataset_final.xlsx")
SAVE_DIR = os.path.join(PROJECT_ROOT, "peak_strain", "SAVE")
OUTPUT_FILE = os.path.join(SAVE_DIR, "strain_comparison_metrics.xlsx")

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Reading data: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH, sheet_name=0)

# Check if the peak_strain column exists
if "peak_strain" not in df.columns:
    raise ValueError("The 'peak_strain' column does not exist in the dataset")

# Get the peak_strain column (true values)
true_values = df["peak_strain"].values

# Remove missing values
mask = ~np.isnan(true_values)
true_values = true_values[mask]

print(f"Valid sample number: {len(true_values)}")

# Only calculate the metrics for these specified columns
compare_cols = [
    "Xiao_strain",
    "XGB_Xiao_strain",
    "True_Yan_strain",
    "True_Belen_strain",
    "peak_strain_pinn_pred",
    "peak_strain_pinn_pred_xgbfc",
]

# Check which columns exist
existing_cols = [col for col in compare_cols if col in df.columns]
missing_cols = [col for col in compare_cols if col not in df.columns]

if missing_cols:
    print(f"Warning: The following columns do not exist in the dataset: {missing_cols}")
if existing_cols:
    print(f"Found {len(existing_cols)} columns to compare: {existing_cols}")
else:
    raise ValueError("No columns to compare found!")

compare_cols = existing_cols

# Store all metric results
results = []

for col in compare_cols:
    pred_values = df[col].values[mask]  # Use the same mask
    
    # Check if there are valid values
    valid_mask = ~np.isnan(pred_values)
    if valid_mask.sum() == 0:
        print(f"Warning: {col} column has no valid values, skipping")
        continue
    
    true_vals = true_values[valid_mask]
    pred_vals = pred_values[valid_mask]
    
    # Avoid division by zero error
    if len(true_vals) == 0 or np.std(true_vals) == 0:
        print(f"Warning: {col} column data is invalid, skipping")
        continue
    
    try:
        # Calculate various metrics
        r2 = r2_score(true_vals, pred_vals)
        mse = mean_squared_error(true_vals, pred_vals)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_vals, pred_vals)
        mape = mean_absolute_percentage_error(true_vals, pred_vals) * 100  # Convert to percentage
        evs = explained_variance_score(true_vals, pred_vals)
        
        # MPE (Mean Percentage Error) = mean((pred - true) / true) * 100
        mpe = np.mean((pred_vals - true_vals) / np.abs(true_vals)) * 100
        
        # Additional metrics
        # Maximum error
        max_error = np.max(np.abs(pred_vals - true_vals))
        # Median absolute error
        median_ae = np.median(np.abs(pred_vals - true_vals))
        # Correlation coefficient
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        results.append({
            "Column name": col,
            "R²": r2,
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
            "MAPE (%)": mape,
            "MPE (%)": mpe,
            "EVS": evs,
            "Maximum error": max_error,
            "Median absolute error": median_ae,
            "Correlation coefficient": correlation,
            "Valid sample number": len(true_vals),
        })
        
        print(f"✓ {col}: R²={r2:.4f}, RMSE={rmse:.6f}, MAPE={mape:.2f}%")
        
    except Exception as e:
        print(f"✗ {col} calculation failed: {e}")
        continue

# Create result DataFrame
results_df = pd.DataFrame(results)

# Sort by R² in descending order
results_df = results_df.sort_values("R²", ascending=False)

# Save to Excel
print(f"\nSaving results to: {OUTPUT_FILE}")
results_df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

print(f"\nDone! Calculated {len(results_df)} columns of metrics")
print(f"\n前10名（按R²排序）:")
print(results_df[["Column name", "R²", "RMSE", "MAPE (%)", "Correlation coefficient"]].head(10).to_string(index=False))

