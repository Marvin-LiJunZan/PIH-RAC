#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical Information Neural Network (PINN) peak strain prediction model - training script
==================================================

Function Primaryly:
---------
This script focuses on model training, outputting trained model weights, configuration files, and preprocessors.
It does not include inference and detailed evaluation functionality, use peak_strain_inference.py for inference.

Model Features:
---------
1. Data-driven: Based on experimental data to fit the concrete peak strain-stress relationship
2. Physical constraint as secondary: Integrate Xiao et al. empirical formula as a soft constraint (low weight)
3. Smart optimization: Use Optuna Bayesian optimization (TPE algorithm) to automatically search for optimal hyperparameters
4. Cross-validation: Perform cross-validation based on predefined subsets to evaluate model generalization ability
5. Normalization strategy: MinMaxScaler normalize to [0,1], match Sigmoid output layer.

Technical Highlights:
---------
- Network structure: Fully connected network + ReLU activation + Sigmoid output layer
- Regularization: Dropout + L2 weight decay, double prevention of overfitting
- Optimizer: Adam + exponential learning rate decay + gradient clipping
- Loss function: λ_data * MSE_data + λ_physics * MSE_physics
- Early stopping: Stop training early if validation performance does not improve.

Training Output Metrics:
-------------
✅ Training process:
   - Print every 20 epochs: train_loss, val_loss, val_R²
   - Early stopping information: triggered epoch and patience value

✅ Optuna hyperparameter optimization (optional):
   - Hyperparameter configuration for each Trial
   - Mean and standard deviation of validation R² for each fold
   - Optimal hyperparameters and validation R² for the best Trial

✅ Cross-validation results:
   - Validation R² and test R² for each fold
   - Best fold information

✅ Save files:
   - pinn_peak_strain.pt: model weights
   - training_summary.json: training configuration and metrics
   - scalers.pkl: data preprocessor
   - model_architecture.json: model architecture

Usage:
---------
python peak_strain_model_minmax.py --tune-trials 50 --final-epochs 500

Inference and evaluation:
---------
After training, use peak_strain_inference.py for inference and comparison with Xiao formula.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ---------------------------------------------------------------------------
# Global configuration and utility functions
# ---------------------------------------------------------------------------

# When you use it, you need to change it to your own path
PROJECT_ROOT = os.path.normpath(r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(os.path.normpath(r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\peak_strain\PINN\SAVE"), exist_ok=True)

# Set random seed to ensure reproducibility.
def set_random_seed(seed: int = 42) -> None:
    """Set random seed to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure logging output.
def configure_logging(log_path: Optional[str] = None) -> None:
    """Configure logging output."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


# Basic configuration dataclass. Will be expanded as needed.
@dataclass
class BaseConfig:
    """Basic configuration dataclass. Will be expanded as needed."""

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = os.path.normpath(r"peak_strain\PINN\SAVE")
    dataset_path: str = os.path.join(PROJECT_ROOT, "dataset", "dataset_final.xlsx")
    dataset_sheet: int = 0
    target_column: str = "peak_strain"
    stress_column: str = "fc"
    baseline_column: str = "xiao_formula"
    feature_columns: Optional[List[str]] = None
    test_ratio: float = 0.2
    val_ratio: float = 0.2
    train_batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0  # Windows/笔记本建议0
    pin_memory: bool = torch.cuda.is_available()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """PINN training related hyperparameters (based on the successful experience of 0.87 R²)."""

    input_dim: int = 0  # Fill after initialization based on the number of features
    
    # Network structure (larger capacity, reference previous 0.87 configuration)
    hidden_dim: int = 512  # Increase to 512 (previously best was 407接近512)
    num_layers: int = 6    # Increase to 6 layers (previously best was 6)
    
    # Regularization strategy (适度正则化，不要过度限制）
    dropout: float = 0.15          # Decrease dropout (too high will limit learning ability)
    weight_decay: float = 5e-5     # Decrease L2 regularization
    
    # Learning rate and optimization strategy
    learning_rate: float = 2e-3    # Increase initial learning rate
    use_scheduler: bool = True     # Use learning rate decay
    scheduler_gamma: float = 0.97  # Moderate decay
    gradient_clip: Optional[float] = 1.5  # Gradient clipping threshold
    
    # Loss function weights (try more balanced configuration)
    lambda_physics: float = 0.5    # Increase physical loss weight (may help regularization)
    lambda_data: float = 3.0       # Data loss weight
    
    # Training strategy
    max_epochs: int = 500   # Increase maximum training epochs
    patience: int = 80      # Increase early stopping patience

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrainDataset(Dataset):
    """Wrapped peak strain dataset."""

    def __init__(
        self,
        features: np.ndarray,
        strain: np.ndarray,
        stress: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        r_values: Optional[np.ndarray] = None,
    ) -> None:
        assert features.shape[0] == strain.shape[0] == stress.shape[0], "Sample number mismatch"
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.strain = torch.as_tensor(strain, dtype=torch.float32).view(-1, 1)
        self.stress = torch.as_tensor(stress, dtype=torch.float32).view(-1, 1)
        if baseline is not None:
            self.baseline = torch.as_tensor(baseline, dtype=torch.float32).view(-1, 1)
        else:
            self.baseline = None
        if r_values is not None:
            # r = recycled aggregate replacement rate, divided by 100: convert from percentage×100 (e.g. 1.5% to 0.015)
            self.r_values = torch.as_tensor(r_values / 100.0, dtype=torch.float32).view(-1, 1)
        else:
            self.r_values = None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.features[idx],
            "strain": self.strain[idx],
            "stress": self.stress[idx],
        }
        if self.baseline is not None:
            item["baseline"] = self.baseline[idx]
        if self.r_values is not None:
            item["r"] = self.r_values[idx]
        return item


def get_column_values(
    df: pd.DataFrame,
    column: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Get the values of a specified column from a DataFrame. If the column does not exist, raise an exception."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the data. Existing columns: {list(df.columns)}")
    return df[column].astype(dtype).to_numpy()


def _xiao_formula_core(stress, r, lib):
    """Xiao formula core calculation logic (NumPy/PyTorch通用).
    
    Formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
    
    Args:
        stress: Stress value (fc)
        r: Recycled aggregate replacement rate (decimal form, e.g. 0.015 for 1.5%)
        lib: Library used (np or torch), providing clip/clamp and sqrt functions
    
    Returns:
        Peak strain prediction value
    """
    # Distinguish between NumPy and PyTorch by checking the library name
    is_numpy = lib.__name__ == 'numpy'
    
    if is_numpy:
        # NumPy uses clip, parameter name a_min/a_max
        stress_clamped = lib.clip(stress, a_min=1e-6, a_max=None)
        r_clamped = lib.clip(r, a_min=1e-8, a_max=None)
        
        inner = (0.626 * stress_clamped - 4.33) * 1e-7
        inner_clamped = lib.clip(inner, a_min=0.0, a_max=None)
    else:
        # PyTorch uses clamp, parameter name min/max
        stress_clamped = lib.clamp(stress, min=1e-6)
        r_clamped = lib.clamp(r, min=1e-8)
        
        inner = (0.626 * stress_clamped - 4.33) * 1e-7
        inner_clamped = lib.clamp(inner, min=0.0)
    
    term1 = 0.00076 + lib.sqrt(inner_clamped)

    # Second term: 1 + r / (65.715r^2 - 109.43r + 48.989)
    denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
    term2 = 1.0 + (r_clamped / denom)

    return term1 * term2


def compute_xiao_formula(df: pd.DataFrame, stress_column: str) -> np.ndarray:
    """Xiao formula calculation (NumPy version, for data preprocessing).
    
    Args:
        df: Dataset
        stress_column: Stress column name
    
    Returns:
        Xiao formula calculated peak strain
    
    Note: r = Recycled Aggregate replacement rate
          Stored in the data as percentage×100 (e.g. 1.5% stored as 1.5), need to divide by 100 to convert to decimal form (0.015)
    """
    stress = get_column_values(df, stress_column)
    r = get_column_values(df, "r")  # r = Recycled Aggregate replacement rate
    
    # r value divided by 100: convert from percentage×100 (e.g. 1.5 to 0.015)
    r = r / 100.0
    
    return _xiao_formula_core(stress.astype(float), r, np)


def infer_feature_columns(df: pd.DataFrame, config: BaseConfig) -> List[str]:
    """Infer feature columns based on default rules."""
    if config.feature_columns:
        return config.feature_columns

    material_features = ["water", "cement", "w/c", "CS", "sand", "CA", "r", "WA", "S", "CI"]
    specimen_features = ["age", "μe", "DJB", "side", "GJB"]
    extra_features = ["fc"]
    formula_features = ["xiao_formula"]

    candidate_columns = material_features + specimen_features + extra_features + formula_features
    available = [col for col in candidate_columns if col in df.columns]

    if not available:
        # If all preset columns are unavailable, fall back to all numeric columns (excluding target/stress/ID)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        blacklist = {config.target_column, config.stress_column}
        available = [col for col in numeric_cols if col not in blacklist]
        logging.warning("No preset feature columns matched, falling back to all numeric columns: %s", available)

    return available


def load_dataframe(config: BaseConfig) -> pd.DataFrame:
    """Load the original data table."""
    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Data file not found: {config.dataset_path}")

    df = pd.read_excel(config.dataset_path, sheet_name=config.dataset_sheet)
    logging.info("Successfully loaded data, shape: %s", df.shape)
    return df


def preprocess_data(config: BaseConfig) -> Dict[str, Any]:
    """Data loading and preprocessing, returning information containing all samples and DataSlice indices."""
    df = load_dataframe(config)
    df = df.copy()

    df["xiao_formula"] = compute_xiao_formula(df, config.stress_column)

    # Infer feature columns
    feature_cols = infer_feature_columns(df, config)
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' does not exist, existing columns: {list(df.columns)}")
    if config.stress_column not in df.columns:
        raise ValueError(f"Stress column '{config.stress_column}' does not exist, existing columns: {list(df.columns)}")
    if config.baseline_column not in df.columns:
        logging.warning("Baseline column '%s' not found, will not use baseline information", config.baseline_column)

    # Remove missing data
    subset_cols = feature_cols + [config.target_column, config.stress_column]
    if config.baseline_column in df.columns:
        subset_cols.append(config.baseline_column)
    df_clean = df.dropna(subset=subset_cols).reset_index(drop=True)
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        logging.warning("Missing data found, %d records have been deleted", dropped)

    features = df_clean[feature_cols].values.astype(np.float32)
    strain = df_clean[config.target_column].values.astype(np.float32)
    stress = df_clean[config.stress_column].values.astype(np.float32)
    baseline = (
        df_clean[config.baseline_column].values.astype(np.float32)
        if config.baseline_column in df_clean.columns
        else None
    )
    # Extract r parameter for physical loss calculation (if exists)
    if "r" in df_clean.columns:
        r_values = df_clean["r"].values.astype(np.float32)
        logging.info("Successfully extracted r parameter, sample number=%d, range=[%.6f, %.6f]", 
                    len(r_values), np.min(r_values), np.max(r_values))
    else:
        r_values = None
        logging.warning("r column missing in the data, Xiao formula will not be able to calculate correctly! Please check the data file.")
    if "DataSlice" not in df_clean.columns:
        raise ValueError("DataSlice column missing in the dataset, cannot split the dataset in the specified way.")

    sample_divisions_raw = df_clean["DataSlice"].astype(str).str.strip().to_numpy()
    sample_ids = (
        df_clean["No_Customized"].values.tolist()
        if "No_Customized" in df_clean.columns
        else [f"sample_{i}" for i in range(len(df_clean))]
    )

    # Record original stress/strain statistics for physical loss usage
    strain_stats = {"mean": float(np.mean(strain)), "std": float(np.std(strain))}
    stress_stats = {"mean": float(np.mean(stress)), "std": float(np.std(stress))}

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    strain_scaled = target_scaler.fit_transform(strain.reshape(-1, 1)).flatten()

    # Organize DataSlice indices
    subset_indices: Dict[str, List[int]] = {}
    explicit_val_indices: Dict[str, List[int]] = {}
    explicit_train_indices: List[int] = []
    test_indices: List[int] = []
    fallback_indices: List[int] = []

    for idx, division in enumerate(sample_divisions_raw):
        lower_div = division.lower()
        if lower_div.startswith("subset"):
            subset_indices.setdefault(division, []).append(idx)
        elif lower_div.startswith("test"):
            test_indices.append(idx)
        elif lower_div.startswith("val") or division in {"validation set", "validation"}:
            explicit_val_indices.setdefault(division, []).append(idx)
        elif lower_div.startswith("train") or division in {"train set"}:
            explicit_train_indices.append(idx)
        else:
            # Store unknown labels for later subset processing
            fallback_indices.append(idx)

    if fallback_indices:
        subset_indices.setdefault("subset_fallback", []).extend(fallback_indices)

    subset_indices_np = {
        label: np.array(idxs, dtype=int) for label, idxs in subset_indices.items()
    }
    explicit_val_indices_np = {
        label: np.array(idxs, dtype=int) for label, idxs in explicit_val_indices.items()
    }
    explicit_train_indices_np = np.array(explicit_train_indices, dtype=int)
    test_indices_np = np.array(test_indices, dtype=int)

    subset_labels = [
        label
        for label in sorted(subset_indices_np.keys(), key=lambda x: x.lower())
        if label.lower().startswith("subset")
    ]

    data_dict: Dict[str, Any] = {
        "feature_cols": feature_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "strain_stats": strain_stats,
        "stress_stats": stress_stats,
        "df_clean": df_clean,
        "sample_ids": sample_ids,
        "sample_divisions": sample_divisions_raw,
        "features_raw": features,
        "features_scaled": features_scaled,
        "strain_scaled": strain_scaled,
        "stress_raw": stress,
        "baseline_raw": baseline,
        "r_values": r_values,  # 添加r参数
        "subset_indices": subset_indices_np,
        "subset_labels": subset_labels,
        "explicit_val_indices": explicit_val_indices_np,
        "explicit_train_indices": explicit_train_indices_np,
        "test_indices": test_indices_np,
        "num_samples": len(df_clean),
        "all_indices": np.arange(len(df_clean), dtype=int),
    }

    return data_dict


def create_datasets(
    data_dict: Dict[str, Any],
    split_indices: Dict[str, np.ndarray],
) -> Dict[str, StrainDataset]:
    """Build Dataset objects based on index information."""

    datasets: Dict[str, StrainDataset] = {}
    baseline = data_dict["baseline_raw"]
    r_values = data_dict.get("r_values")

    for split_name, indices in split_indices.items():
        if indices.size == 0:
            continue
        features = data_dict["features_scaled"][indices]
        strain = data_dict["strain_scaled"][indices]
        stress = data_dict["stress_raw"][indices]
        baseline_split = baseline[indices] if baseline is not None else None
        r_split = r_values[indices] if r_values is not None else None
        datasets[split_name] = StrainDataset(features, strain, stress, baseline_split, r_split)
    return datasets

def create_dataloaders(
    datasets: Dict[str, StrainDataset],
    config: BaseConfig,
) -> Dict[str, DataLoader]:
    """Create DataLoader."""

    loaders: Dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        is_train = split_name == "train"
        batch_size = config.train_batch_size if is_train else config.eval_batch_size
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )
    return loaders


class PINNRegressor(nn.Module):
    """Simple fully connected network for PINN regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if activation is None:
            # MinMaxScaler normalize to [0,1], ReLU更适合 (output [0,+∞))
            # Output layer uses Sigmoid to ensure the final output is within the [0,1] range
            activation = nn.ReLU()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer: use Sigmoid to ensure the output is within the [0,1] range, matching the normalization range of MinMaxScaler
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure the output is within the [0,1] range
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def xiao_equation(stress: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Xiao formula calculation (PyTorch version, for physical loss calculation, supports automatic differentiation).
    
    Args:
        stress: Stress value σ_cp (fc)
        r: Recycled aggregate replacement rate (decimal form, e.g. 0.015 for 1.5%)
        Note: Input is converted from percentage×100 to decimal (divided by 100) when input
    
    Returns:
        Xiao formula predicted peak strain (supports gradient backpropagation)
    """
    return _xiao_formula_core(stress, r, torch)


def compute_physics_loss(
    pred_scaled: torch.Tensor,
    stress_raw: torch.Tensor,
    target_data_min: torch.Tensor,
    target_range: torch.Tensor,
    r_values: torch.Tensor,
) -> torch.Tensor:
    """Calculate physical loss based on the complete Xiao equation.
    
    Args:
        pred_scaled: Normalized model prediction value
        stress_raw: Original stress value
        target_data_min: Minimum value of target value (for de-normalization)
        target_range: Range of target value (for de-normalization)
        r_values: Reinforcement ratio parameter (decimal form, e.g. 0.015 for 1.5%)
    """
    # MinMaxScaler de-normalization: x = x_scaled * (data_max_ - data_min_) + data_min_
    strain_pred = pred_scaled * target_range + target_data_min
    strain_phys = xiao_equation(stress_raw, r_values)
    return nn.functional.mse_loss(strain_pred, strain_phys)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_data_min: torch.Tensor,
    target_range: torch.Tensor,
    lambda_data: float,
    lambda_physics: float,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["strain"].to(device)
        stress = batch["stress"].to(device)
        r = batch.get("r")  # Get r parameter (if exists)
        if r is not None:
            r = r.to(device)  # Ensure r parameter is on the correct device

        optimizer.zero_grad()
        preds = model(x)

        data_loss = mse_loss(preds, y)
        physics_loss = compute_physics_loss(preds, stress, target_data_min, target_range, r)
        loss = lambda_data * data_loss + lambda_physics * physics_loss

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_data_loss += data_loss.item() * batch_size
        total_physics_loss += physics_loss.item() * batch_size
        total_samples += batch_size

    avg_data_loss = total_data_loss / total_samples
    avg_physics_loss = total_physics_loss / total_samples

    return {
        "loss": total_loss / total_samples,
        "data_loss": avg_data_loss,
        "physics_loss": avg_physics_loss,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_data_min: torch.Tensor,
    target_range: torch.Tensor,
    lambda_data: float,
    lambda_physics: float,
) -> Dict[str, Any]:
    model.eval()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_samples = 0

    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    preds_physical: List[np.ndarray] = []
    stress_list: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["strain"].to(device)
        stress = batch["stress"].to(device)
        r = batch.get("r")  # Get r parameter (if exists)
        if r is not None:
            r = r.to(device)  # Ensure r parameter is on the correct device

        preds = model(x)
        data_loss = mse_loss(preds, y)
        physics_loss = compute_physics_loss(preds, stress, target_data_min, target_range, r)
        loss = lambda_data * data_loss + lambda_physics * physics_loss

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_data_loss += data_loss.item() * batch_size
        total_physics_loss += physics_loss.item() * batch_size
        total_samples += batch_size

        preds_list.append(preds.detach().cpu().numpy())
        targets_list.append(y.detach().cpu().numpy())
        stress_list.append(stress.detach().cpu().numpy())

        strain_phys = xiao_equation(stress, r)
        preds_physical.append(strain_phys.detach().cpu().numpy())

    preds_scaled = np.concatenate(preds_list, axis=0)
    targets_scaled = np.concatenate(targets_list, axis=0)
    stress_raw = np.concatenate(stress_list, axis=0)
    phys_expect = np.concatenate(preds_physical, axis=0)

    return {
        "loss": total_loss / total_samples,
        "data_loss": total_data_loss / total_samples,
        "physics_loss": total_physics_loss / total_samples,
        "preds_scaled": preds_scaled,
        "targets_scaled": targets_scaled,
        "stress_raw": stress_raw,
        "phys_expect": phys_expect,
    }


def compute_metrics_from_scaled(
    preds_scaled: np.ndarray,
    targets_scaled: np.ndarray,
    target_scaler: MinMaxScaler,
) -> Dict[str, float]:
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    metrics = {
        "r2": float(r2_score(targets, preds)),
        "evs": float(explained_variance_score(targets, preds)),
        "mae": float(mean_absolute_error(targets, preds)),
        "mse": float(mean_squared_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "medae": float(median_absolute_error(targets, preds)),
        "mape": float(mean_absolute_percentage_error(targets, preds)),
        "max_error": float(max_error(targets, preds)),
    }
    return metrics


def train_pinn(
    data_dict: Dict[str, Any],
    base_config: BaseConfig,
    train_config: TrainingConfig,
    split_indices: Dict[str, np.ndarray],
    run_test: bool = True,
    trial: Optional[Trial] = None,  # For Optuna pruning
) -> Dict[str, Any]:
    """Train PINN model and return training process and results.
    
    Args:
        trial: Optuna Trial object, for reporting intermediate results and pruning
    """

    device = torch.device(base_config.device)
    datasets = create_datasets(data_dict, split_indices)
    dataloaders = create_dataloaders(datasets, base_config)

    if "train" not in datasets or "val" not in datasets:
        raise ValueError("train_pinn requires both training and validation set indices.")

    input_dim = datasets["train"].features.shape[1]
    train_config.input_dim = input_dim

    model = PINNRegressor(
        input_dim=input_dim,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config.scheduler_gamma)
        if train_config.use_scheduler
        else None
    )

    # MinMaxScaler de-normalization formula: x = x_scaled * (data_max_ - data_min_) + data_min_
    # Note: min_ and scale_ are internal properties, not the minimum and range of the original data
    # Should use data_min_ and data_max_ to calculate de-normalization
    target_scaler_obj = data_dict["target_scaler"]
    # data_min_ is the minimum value of the original data, data_max_ is the maximum value of the original data
    target_data_min = torch.tensor(
        target_scaler_obj.data_min_[0], dtype=torch.float32, device=device
    )
    target_data_max = torch.tensor(
        target_scaler_obj.data_max_[0], dtype=torch.float32, device=device
    )
    # Calculate range for de-normalization
    target_range = target_data_max - target_data_min
    # To avoid division by zero, add a small epsilon
    target_range = torch.clamp(target_range, min=1e-8)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    patience_counter = 0

    for epoch in range(1, train_config.max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            target_data_min,
            target_range,
            train_config.lambda_data,
            train_config.lambda_physics,
            train_config.gradient_clip,
        )
        val_stats = evaluate_model(
            model,
            dataloaders["val"],
            device,
            target_data_min,
            target_range,
            train_config.lambda_data,
            train_config.lambda_physics,
        )
        val_metrics = compute_metrics_from_scaled(
            val_stats["preds_scaled"], val_stats["targets_scaled"], data_dict["target_scaler"]
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_stats,
                "val": {**val_stats, "metrics": val_metrics},
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Only print information for every 20th epoch or the last epoch (reduce log volume)
        if epoch % 20 == 0 or epoch == train_config.max_epochs:
            logging.info(
                "[Epoch %03d] train_loss=%.6f | val_loss=%.6f | val_R²=%.4f",
                epoch,
                train_stats["loss"],
                val_stats["loss"],
                val_metrics["r2"],
            )
        
        # Report intermediate results to Optuna (for pruning)
        # Note: pruning is not used in cross-validation, because it will cause duplicate step warning reporting
        # Pruning is only useful in single training, cross-validation has already been evaluated at the fold level
        if trial is not None:
            # In cross-validation scenario, do not report intermediate results during training
            # Only report the final result of each fold in the objective function
            pass

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history.copy(),
                "val_stats": val_stats,
                "val_metrics": val_metrics,
            }
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                logging.info("Early stopping triggered (patience=%d), stopping training at epoch %d", train_config.patience, epoch)
                break

        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("Best model state not recorded during training.")

    model.load_state_dict(best_state["model_state"])

    test_stats: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, float]] = None
    test_xiao_metrics: Optional[Dict[str, float]] = None

    if run_test and "test" in dataloaders and split_indices.get("test", np.array([])).size > 0:
        test_stats = evaluate_model(
            model,
            dataloaders["test"],
            device,
            target_data_min,
            target_range,
            train_config.lambda_data,
            train_config.lambda_physics,
        )
        test_metrics = compute_metrics_from_scaled(
            test_stats["preds_scaled"], test_stats["targets_scaled"], data_dict["target_scaler"]
        )
        
        # Calculate test set Xiao formula metrics
        test_targets_original = data_dict["target_scaler"].inverse_transform(
            test_stats["targets_scaled"].reshape(-1, 1)
        ).flatten()
        test_xiao_preds = test_stats["phys_expect"].flatten()
        test_xiao_metrics = {
            "r2": float(r2_score(test_targets_original, test_xiao_preds)),
            "evs": float(explained_variance_score(test_targets_original, test_xiao_preds)),
            "mae": float(mean_absolute_error(test_targets_original, test_xiao_preds)),
            "mse": float(mean_squared_error(test_targets_original, test_xiao_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test_targets_original, test_xiao_preds))),
            "medae": float(median_absolute_error(test_targets_original, test_xiao_preds)),
            "mape": float(mean_absolute_percentage_error(test_targets_original, test_xiao_preds)),
            "max_error": float(max_error(test_targets_original, test_xiao_preds)),
        }

        # Simplify test set output (will be printed later)
        pass

    results = {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "best_state": best_state,
        "test_stats": test_stats,
        "test_metrics": test_metrics,
        "test_xiao_metrics": test_xiao_metrics,  # Add Xiao formula test set metrics
        "dataloaders": dataloaders,
        "datasets": datasets,
        "target_data_min": float(target_data_min.cpu().item()),
        "target_range": float(target_range.cpu().item()),
        "split_indices": {k: v.copy() for k, v in split_indices.items()},
    }
    return results


def sample_training_config(trial: Trial, base_train_config: TrainingConfig) -> TrainingConfig:
    """Generate training configuration based on Optuna trial.
    
    Optimization strategy (goal: restore 0.87 R² performance):
    1. Expand network capacity search range (previously best 407/hidden, 6 layers)
    2. Decrease regularization strength (avoid over-restricting learning ability)
    3. Wider physical loss weight range (explore best balance point)
    4. Sufficient training (more epochs, larger patience)
    """
    cfg = replace(base_train_config)
    
    # ==================== Network structure ====================
    # Hidden layer dimension: expand to 64-1024 (previously best 407, now includes larger range)
    cfg.hidden_dim = trial.suggest_int("hidden_dim", 64, 1024, step=32)
    
    # Hidden layer number: 2-10 layers (previously best 6 layers, expand search space)
    cfg.num_layers = trial.suggest_int("num_layers", 2, 10)
    
    # ==================== Regularization ====================
    # Dropout: 0.0-0.35 (decrease upper limit, avoid over-regularization)
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.35)
    
    # L2 regularization: 1e-6 to 1e-3 (expand search range)L2 regularization: 1e-6 to 1e-3 (expand search range)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # ==================== Learning rate and optimizer ====================
    # Learning rate: 1e-4 to 5e-3 (standard range)
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    
    # Learning rate decay: 0.90-0.999 (expand range, previously best 0.987 close to upper limit)
    cfg.scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.90, 0.999)
    cfg.use_scheduler = True
    
    # ==================== Physical loss weight ====================
    # Physical loss weight: 0.1-5.0 (expand range, explore positive effect of physical constraints)
    # Although Xiao formula R²<0, physical constraints may have a regularization effect
    cfg.lambda_physics = trial.suggest_float("lambda_physics", 0.1, 5.0, log=True)
    
    # Data loss weight: 0.5-5.0
    cfg.lambda_data = trial.suggest_float("lambda_data", 0.5, 5.0, log=True)
    
    # ==================== Training strategy ====================
    # Maximum training epochs: 150-600 (expand range)
    cfg.max_epochs = trial.suggest_int("max_epochs", 150, 600, step=50)
    
    # Early stopping patience: 30-120 (expand range, previously best 59)
    cfg.patience = trial.suggest_int("patience", 30, 120, step=10)
    
    # Gradient clipping: 0.5-3.0
    cfg.gradient_clip = trial.suggest_float("gradient_clip", 0.5, 3.0)
    
    return cfg


def optimize_hyperparameters(
    data_dict: Dict[str, Any],
    base_config: BaseConfig,
    base_train_config: TrainingConfig,
    n_trials: int = 30,
    n_folds: int = 3,
) -> optuna.study.Study:
    """Use Optuna for Bayesian optimization, using random K-fold cross-validation."""

    # Get fixed test set and samples available for cross-validation
    test_indices = data_dict.get("test_indices", np.array([], dtype=int))
    all_indices = data_dict["all_indices"]
    train_val_indices = np.setdiff1d(all_indices, test_indices)
    
    if len(train_val_indices) < n_folds:
        raise ValueError(f"Available sample number ({len(train_val_indices)}) less than fold number ({n_folds}), cannot perform cross-validation")

    def objective(trial: Trial) -> float:
        """Optuna optimization objective: maximize mean validation R²."""
        try:
            # Sample training configuration
            train_cfg = sample_training_config(trial, base_train_config)
            
            # Simple progress hint
            logging.info("\n" + "-"*80)
            logging.info("Trial %d/%d | hidden=%d, layers=%d, dropout=%.2f, lr=%.1e", 
                        trial.number + 1, n_trials,
                        train_cfg.hidden_dim, train_cfg.num_layers, 
                        train_cfg.dropout, train_cfg.learning_rate)
            logging.info("-"*80)

            fold_scores: List[float] = []
            fold_details: List[Dict[str, Any]] = []

            # Use KFold for random划分
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=base_config.seed)
            
            for fold_idx, (train_idx_local, val_idx_local) in enumerate(kfold.split(train_val_indices), start=1):
                train_idx = train_val_indices[train_idx_local]
                val_idx = train_val_indices[val_idx_local]

                split_indices = {
                    "train": np.sort(train_idx),
                    "val": np.sort(val_idx),
                }
                if test_indices.size > 0:
                    split_indices["test"] = test_indices.copy()

                # Train model (log is controlled in train_pinn)
                results = train_pinn(
                    data_dict,
                    base_config,
                    train_cfg,
                    split_indices=split_indices,
                    run_test=False,
                    trial=trial,
                )
                val_metrics = results["best_state"]["val_metrics"]
                fold_scores.append(float(val_metrics["r2"]))
                fold_details.append({
                    "fold": fold_idx,
                    "val_metrics": val_metrics,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                })

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            trial.set_user_attr("fold_val_metrics", fold_details)
            trial.set_user_attr("mean_val_r2", mean_score)
            trial.set_user_attr("std_val_r2", std_score)
            
            # Report final result to Optuna (for recording, not for pruning)
            # Use trial.number as step, avoid duplicate report
            trial.report(mean_score, trial.number)
            
            # Output trial result
            logging.info("✓ Trial %d completed: mean validation R² = %.4f ± %.4f", 
                        trial.number + 1, mean_score, std_score)
            
            return mean_score
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Trial %d failed: %s", trial.number + 1, exc)
            raise

    # ==================== Bayesian optimization configuration ====================
    # TPESampler: Tree-structured Parzen Estimator (one of the most advanced Bayesian optimization algorithms)
    # Principle: maintain two probability models (good configuration and bad configuration), smartly select next trial point
    sampler = TPESampler(
        n_startup_trials=10,  # Randomly sample the first 10 trials (fully explore, establish good prior)
        n_ei_candidates=50,   # Increase to 50 (more accurate expected improvement estimation)
        multivariate=True,    # Consider the correlation between hyperparameters (capture interaction between hidden_dim and num_layers)
        seed=base_config.seed,  # Use global random seed, ensure reproducibility
    )
    
    # MedianPruner: Early stopping strategy (not suitable for cross-validation scenario, therefore disabled)
    # Reason: each fold is independently trained in cross-validation, pruning will cause duplicate step warning reporting
    # Solution: do not use pruner, let all trials run fully
    pruner = None  # Disable pruning, avoid warning
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="pinn_peak_strain_minmax",
        sampler=sampler,  # Use TPE sampler for Bayesian optimization
        # pruner=None do not use pruner (not suitable for cross-validation scenario)
    )
    
    logging.info("\n" + "="*80)
    logging.info("Start Optuna Bayesian hyperparameter optimization")
    logging.info("="*80)
    logging.info("Optimization configuration: trials=%d, folds=%d, sampler=TPE", n_trials, n_folds)
    logging.info("Search space: hidden_dim=[64,1024], layers=[2,10], dropout=[0,0.35]")
    logging.info("Optimization goal: restore to 0.87+ R² performance (Xiao formula R²~0.42)")
    logging.info("Estimated time: ~%.1f minutes (estimated based on average time per trial)\n", n_trials * n_folds * 2 / 60)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    logging.info("\n" + "="*80)
    logging.info("✓ Optuna optimization completed")
    logging.info("="*80)
    logging.info("Best validation R² = %.4f", study.best_value)
    logging.info("Best hyperparameters:")
    for key, value in sorted(study.best_params.items()):
        logging.info("  - %s: %s", key, value)
    logging.info("="*80 + "\n")
    return study


def _build_final_split_indices(
    data_dict: Dict[str, Any],
    base_config: BaseConfig,
    subset_labels: List[str],
    subset_indices_map: Dict[str, np.ndarray],
    test_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build final training/validation split based on all subset samples."""
    if not subset_labels:
        raise ValueError("Missing subset labels, cannot build final training set.")

    all_subset_indices = np.concatenate([subset_indices_map[label] for label in subset_labels])
    if all_subset_indices.size < 2:
        raise ValueError("Subset sample number insufficient to divide into training and validation sets.")

    val_ratio = base_config.val_ratio
    rng = np.random.default_rng(base_config.seed)
    shuffled = all_subset_indices.copy()
    rng.shuffle(shuffled)

    if val_ratio <= 0:
        tentative_val = max(1, len(shuffled) // 5)
    else:
        tentative_val = int(np.ceil(len(shuffled) * val_ratio))

    tentative_val = max(1, min(tentative_val, len(shuffled) - 1))

    val_indices = np.sort(shuffled[:tentative_val])
    train_indices = np.sort(np.setdiff1d(all_subset_indices, val_indices, assume_unique=False))

    split_indices = {
        "train": train_indices,
        "val": val_indices,
    }
    if test_indices.size > 0:
        split_indices["test"] = np.sort(test_indices)
    else:
        split_indices["test"] = np.array([], dtype=int)

    logging.info(
        "Final training set split: train=%d, val=%d, test=%d",
        split_indices["train"].size,
        split_indices["val"].size,
        split_indices["test"].size,
    )
    return split_indices


def train_with_subset_strategy(
    data_dict: Dict[str, Any],
    base_config: BaseConfig,
    train_config: TrainingConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Use predefined subset split for three-fold cross-validation, and collect detailed metrics."""

    subset_labels = data_dict.get("subset_labels", [])
    subset_indices_map = data_dict.get("subset_indices", {})
    test_indices = data_dict.get("test_indices", np.array([], dtype=int)) if test_indices.size == 0:
        logging.warning("No test set labels detected (test), cross-validation will not evaluate test set performance.")

    logging.info("\n" + "="*120)
    logging.info("Subset split cross-validation (%d folds)", len(subset_labels)) 
    logging.info("="*120)
    
    cv_results: List[Dict[str, Any]] = []
    best_test_r2 = -np.inf
    best_fold_idx = 0

    for fold_idx, val_label in enumerate(subset_labels, start=1):
        train_labels = [lbl for lbl in subset_labels if lbl != val_label]
        train_idx = np.concatenate([subset_indices_map[lbl] for lbl in train_labels])
        val_idx = subset_indices_map[val_label]

        split_indices = {
            "train": np.sort(train_idx),
            "val": np.sort(val_idx),
        }
        if test_indices.size > 0:
            split_indices["test"] = np.sort(test_indices)

        logging.info("\n" + "-"*120)
        logging.info("[Fold %d/%d] 验证集: %s", fold_idx, len(subset_labels), val_label)
        logging.info("-"*120)

        fold_config = replace(train_config)
        results = train_pinn(
            data_dict, base_config, fold_config,
            split_indices=split_indices,
            run_test=True,
        )

        # Collect detailed metrics for training set, validation set, and test set
        device = torch.device(base_config.device)
        target_data_min = torch.tensor(results["target_data_min"], dtype=torch.float32, device=device)
        target_range = torch.tensor(results["target_range"], dtype=torch.float32, device=device)
        
        # Training set metrics
        train_stats = evaluate_model(
            results["model"],
            results["dataloaders"]["train"],
            device,
            target_data_min,
            target_range,
            train_config.lambda_data,
            train_config.lambda_physics,
        )
        train_metrics = compute_metrics_from_scaled(
            train_stats["preds_scaled"],
            train_stats["targets_scaled"],
            data_dict["target_scaler"],
        )
        
        # Validation set metrics (already exists)
        val_metrics = results["best_state"]["val_metrics"]
        
        # Test set metrics (already exists)
        test_metrics = results["test_metrics"]
        test_xiao_metrics = results.get("test_xiao_metrics")
        
        best_epoch = results["best_state"]["epoch"]
        test_r2 = test_metrics["r2"] if test_metrics else val_metrics["r2"]
        
        # Print detailed metrics
        logging.info("\n✓ Fold %d training completed [best epoch=%d]", fold_idx, best_epoch)
        logging.info("  Sample number: training set=%d, validation set=%d, test set=%d", 
                    len(train_idx), len(val_idx), len(test_indices) if test_indices.size > 0 else 0)
        
        logging.info("\n  [Training set metrics]")
        logging.info("    R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    train_metrics["r2"], train_metrics["evs"], train_metrics["mae"], train_metrics["mse"],
                    train_metrics["rmse"], train_metrics["medae"], train_metrics["mape"]*100, train_metrics["max_error"])
        
        logging.info("  [Validation set metrics]")
        logging.info("    R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    val_metrics["r2"], val_metrics["evs"], val_metrics["mae"], val_metrics["mse"],
                    val_metrics["rmse"], val_metrics["medae"], val_metrics["mape"]*100, val_metrics["max_error"])
        
        if test_metrics:
            logging.info("  [Test set metrics - PINN]")
            logging.info("    R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                        test_metrics["r2"], test_metrics["evs"], test_metrics["mae"], test_metrics["mse"],
                        test_metrics["rmse"], test_metrics["medae"], test_metrics["mape"]*100, test_metrics["max_error"])
            
            if test_xiao_metrics:
                logging.info("  [Test set metrics - Xiao formula]")
                logging.info("    R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                            test_xiao_metrics["r2"], test_xiao_metrics["evs"], test_xiao_metrics["mae"], test_xiao_metrics["mse"],
                            test_xiao_metrics["rmse"], test_xiao_metrics["medae"], test_xiao_metrics["mape"]*100, test_xiao_metrics["max_error"])

        cv_results.append({
            "fold": fold_idx,
            "subset_label": val_label,
            "split_indices": split_indices,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
                "test_xiao_metrics": test_xiao_metrics,
            "best_epoch": best_epoch,
            "model_results": results,
        })

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_fold_idx = fold_idx - 1

    # Calculate three-fold average metrics
    logging.info("\n" + "="*120)
    logging.info("Three-fold cross-validation average metrics")
    logging.info("="*120)
    
    # Training set average
    avg_train = {
        "r2": np.mean([r["train_metrics"]["r2"] for r in cv_results]),
        "evs": np.mean([r["train_metrics"]["evs"] for r in cv_results]),
        "mae": np.mean([r["train_metrics"]["mae"] for r in cv_results]),
        "mse": np.mean([r["train_metrics"]["mse"] for r in cv_results]),
        "rmse": np.mean([r["train_metrics"]["rmse"] for r in cv_results]),
        "medae": np.mean([r["train_metrics"]["medae"] for r in cv_results]),
        "mape": np.mean([r["train_metrics"]["mape"] for r in cv_results]),
        "max_error": np.mean([r["train_metrics"]["max_error"] for r in cv_results]),
    }
    
    # Validation set average
    avg_val = {
        "r2": np.mean([r["val_metrics"]["r2"] for r in cv_results]),
        "evs": np.mean([r["val_metrics"]["evs"] for r in cv_results]),
        "mae": np.mean([r["val_metrics"]["mae"] for r in cv_results]),
        "mse": np.mean([r["val_metrics"]["mse"] for r in cv_results]),
        "rmse": np.mean([r["val_metrics"]["rmse"] for r in cv_results]),
        "medae": np.mean([r["val_metrics"]["medae"] for r in cv_results]),
        "mape": np.mean([r["val_metrics"]["mape"] for r in cv_results]),
        "max_error": np.mean([r["val_metrics"]["max_error"] for r in cv_results]),
    }
    
    logging.info("[Training set average] R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                avg_train["r2"], avg_train["evs"], avg_train["mae"], avg_train["mse"],
                avg_train["rmse"], avg_train["medae"], avg_train["mape"]*100, avg_train["max_error"])
    
    logging.info("[Validation set average] R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                avg_val["r2"], avg_val["evs"], avg_val["mae"], avg_val["mse"],
                avg_val["rmse"], avg_val["medae"], avg_val["mape"]*100, avg_val["max_error"])
    
    if cv_results[0]["test_metrics"]:
        avg_test = {
            "r2": np.mean([r["test_metrics"]["r2"] for r in cv_results]),
            "evs": np.mean([r["test_metrics"]["evs"] for r in cv_results]),
            "mae": np.mean([r["test_metrics"]["mae"] for r in cv_results]),
            "mse": np.mean([r["test_metrics"]["mse"] for r in cv_results]),
            "rmse": np.mean([r["test_metrics"]["rmse"] for r in cv_results]),
            "medae": np.mean([r["test_metrics"]["medae"] for r in cv_results]),
            "mape": np.mean([r["test_metrics"]["mape"] for r in cv_results]),
            "max_error": np.mean([r["test_metrics"]["max_error"] for r in cv_results]),
        }
        logging.info("[Test set average - PINN] R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    avg_test["r2"], avg_test["evs"], avg_test["mae"], avg_test["mse"],
                    avg_test["rmse"], avg_test["medae"], avg_test["mape"]*100, avg_test["max_error"])
        
        if cv_results[0].get("test_xiao_metrics"):
            avg_xiao = {
                "r2": np.mean([r["test_xiao_metrics"]["r2"] for r in cv_results]),
                "evs": np.mean([r["test_xiao_metrics"]["evs"] for r in cv_results]),
                "mae": np.mean([r["test_xiao_metrics"]["mae"] for r in cv_results]),
                "mse": np.mean([r["test_xiao_metrics"]["mse"] for r in cv_results]),
                "rmse": np.mean([r["test_xiao_metrics"]["rmse"] for r in cv_results]),
                "medae": np.mean([r["test_xiao_metrics"]["medae"] for r in cv_results]),
                "mape": np.mean([r["test_xiao_metrics"]["mape"] for r in cv_results]),
                "max_error": np.mean([r["test_xiao_metrics"]["max_error"] for r in cv_results]),
            }
            logging.info("[Test set average - Xiao formula] R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                        avg_xiao["r2"], avg_xiao["evs"], avg_xiao["mae"], avg_xiao["mse"],
                        avg_xiao["rmse"], avg_xiao["medae"], avg_xiao["mape"]*100, avg_xiao["max_error"])

    best_fold = cv_results[best_fold_idx]
    logging.info("\n" + "="*120)
    logging.info("✓ Best Fold: Fold %d (test R²=%.4f)", 
                best_fold["fold"], best_test_r2)
    logging.info("="*120)
    
    # Use best fold hyperparameters configuration, retrain final model using all non-test data
    logging.info("\n" + "="*120)
    logging.info("Use best fold hyperparameters configuration, retrain final model using all non-test data")
    logging.info("="*120)
    
    # Get all non-test data (subset1+subset2+subset3)
    all_subset_indices = np.concatenate([subset_indices_map[label] for label in subset_labels])
    
    # Split training set and validation set from all non-test data (for final model training)
    # Use same validation set ratio as best fold
    val_ratio = base_config.val_ratio
    rng = np.random.default_rng(base_config.seed)
    shuffled = all_subset_indices.copy()
    rng.shuffle(shuffled)
    
    if val_ratio <= 0:
        tentative_val = max(1, len(shuffled) // 5)
    else:
        tentative_val = int(np.ceil(len(shuffled) * val_ratio))
    
    tentative_val = max(1, min(tentative_val, len(shuffled) - 1))
    
    val_indices_final = np.sort(shuffled[:tentative_val])
    train_indices_final = np.sort(np.setdiff1d(all_subset_indices, val_indices_final, assume_unique=False))
    
    # Test set remains unchanged, for final evaluation
    test_indices_final = test_indices.copy() if test_indices.size > 0 else np.array([], dtype=int)
    
    logging.info("Final model training data split:")
    logging.info("  Training set: %d samples (part of all non-test data)", len(train_indices_final))
    logging.info("  Validation set: %d samples (for early stopping and model selection)", len(val_indices_final))
    logging.info("  Test set: %d samples (independent test set, not participating in training)", len(test_indices_final))
    
    # Use best fold hyperparameters configuration (train_config already contains Optuna optimized hyperparameters)
    final_train_config = replace(train_config)
    
    # Build final training data split
    final_split_indices = {
        "train": train_indices_final,
        "val": val_indices_final,
    }
    if test_indices_final.size > 0:
        final_split_indices["test"] = test_indices_final
    
    # Use best fold configuration to retrain final model
    logging.info("\n开始训练最终模型（使用所有非测试集数据）...")
    final_results = train_pinn(
        data_dict,
        base_config,
        final_train_config,
        split_indices=final_split_indices,
        run_test=True,
    )
    
    # Calculate final model training set metrics (using all non-test data)
    device = torch.device(base_config.device)
    target_data_min = torch.tensor(final_results["target_data_min"], dtype=torch.float32, device=device)
    target_range = torch.tensor(final_results["target_range"], dtype=torch.float32, device=device)
    
    # Training set metrics (all non-test data)
    train_stats_final = evaluate_model(
        final_results["model"],
        final_results["dataloaders"]["train"],
        device,
        target_data_min,
        target_range,
        final_train_config.lambda_data,
        final_train_config.lambda_physics,
    )
    train_metrics_final = compute_metrics_from_scaled(
        train_stats_final["preds_scaled"],
        train_stats_final["targets_scaled"],
        data_dict["target_scaler"],
    )
    
    # Validation set metrics
    val_metrics_final = final_results["best_state"]["val_metrics"]
    
    # Test set metrics
    test_metrics_final = final_results["test_metrics"]
    
    logging.info("\n" + "="*120)
    logging.info("Final model performance (using best fold hyperparameters, trained using all non-test data)")
    logging.info("="*120)
    logging.info("[Training set metrics]")
    logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                train_metrics_final["r2"], train_metrics_final["evs"], train_metrics_final["mae"], train_metrics_final["mse"],
                train_metrics_final["rmse"], train_metrics_final["medae"], train_metrics_final["mape"]*100, train_metrics_final["max_error"])
    logging.info("[Validation set metrics]")
    logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                val_metrics_final["r2"], val_metrics_final["evs"], val_metrics_final["mae"], val_metrics_final["mse"],
                val_metrics_final["rmse"], val_metrics_final["medae"], val_metrics_final["mape"]*100, val_metrics_final["max_error"])
    if test_metrics_final:
        logging.info("[Test set metrics - PINN]")
        logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    test_metrics_final["r2"], test_metrics_final["evs"], test_metrics_final["mae"], test_metrics_final["mse"],
                    test_metrics_final["rmse"], test_metrics_final["medae"], test_metrics_final["mape"]*100, test_metrics_final["max_error"])
    logging.info("="*120)
    
    # Add final model training set and test set metrics to final_results for subsequent saving
    final_results["train_metrics_final"] = train_metrics_final
    final_results["test_metrics_final"] = test_metrics_final
    
    return cv_results, final_results




def evaluate_best_model_on_full_dataset(
    model: nn.Module,
    data_dict: Dict[str, Any],
    base_config: BaseConfig,
    target_data_min: float,
    target_range: float,
    train_config: TrainingConfig,
) -> Dict[str, Any]:
    """Evaluate best model on full dataset and separate test set."""
    device = torch.device(base_config.device)
    model = model.to(device)
    model.eval()
    
    # Evaluate full dataset
    all_indices = data_dict["all_indices"]
    test_indices = data_dict.get("test_indices", np.array([], dtype=int))
    
    split_indices_full = {"full": all_indices}
    if test_indices.size > 0:
        split_indices_full["test_only"] = test_indices
    
    datasets_full = create_datasets(data_dict, split_indices_full)
    dataloaders_full = create_dataloaders(datasets_full, base_config)

    tm_tensor = torch.tensor(target_data_min, dtype=torch.float32, device=device)
    ts_tensor = torch.tensor(target_range, dtype=torch.float32, device=device)

    results = {}
    
    # Evaluate full dataset
    logging.info("\n" + "="*120)
    logging.info("Best model performance on full dataset and test set")
    logging.info("="*120)
    
    full_stats = evaluate_model(
            model,
        dataloaders_full["full"],
            device,
            tm_tensor,
            ts_tensor,
            train_config.lambda_data,
            train_config.lambda_physics,
        )
    full_metrics = compute_metrics_from_scaled(
        full_stats["preds_scaled"],
        full_stats["targets_scaled"],
        data_dict["target_scaler"],
    )
    
    # Calculate full dataset Xiao formula metrics
    full_targets = data_dict["target_scaler"].inverse_transform(
        full_stats["targets_scaled"].reshape(-1, 1)
    ).flatten()
    full_xiao_preds = full_stats["phys_expect"].flatten()
    full_xiao_metrics = {
        "r2": float(r2_score(full_targets, full_xiao_preds)),
        "evs": float(explained_variance_score(full_targets, full_xiao_preds)),
        "mae": float(mean_absolute_error(full_targets, full_xiao_preds)),
        "mse": float(mean_squared_error(full_targets, full_xiao_preds)),
        "rmse": float(np.sqrt(mean_squared_error(full_targets, full_xiao_preds))),
        "medae": float(median_absolute_error(full_targets, full_xiao_preds)),
        "mape": float(mean_absolute_percentage_error(full_targets, full_xiao_preds)),
        "max_error": float(max_error(full_targets, full_xiao_preds)),
    }
    
    logging.info("\n[Full dataset (n=%d) - PINN]", len(all_indices))
    logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                full_metrics["r2"], full_metrics["evs"], full_metrics["mae"], full_metrics["mse"],
                full_metrics["rmse"], full_metrics["medae"], full_metrics["mape"]*100, full_metrics["max_error"])
    
    logging.info("[Full dataset - Xiao formula]")
    logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                full_xiao_metrics["r2"], full_xiao_metrics["evs"], full_xiao_metrics["mae"], full_xiao_metrics["mse"],
                full_xiao_metrics["rmse"], full_xiao_metrics["medae"], full_xiao_metrics["mape"]*100, full_xiao_metrics["max_error"])
    
    results["full_dataset"] = {
        "pinn_metrics": full_metrics,
        "xiao_metrics": full_xiao_metrics,
        "sample_count": len(all_indices),
    }
    
    # Evaluate separate test set
    if "test_only" in dataloaders_full:
        test_only_stats = evaluate_model(
            model,
            dataloaders_full["test_only"],
            device,
            tm_tensor,
            ts_tensor,
            train_config.lambda_data,
            train_config.lambda_physics,
        )
        test_only_metrics = compute_metrics_from_scaled(
            test_only_stats["preds_scaled"],
            test_only_stats["targets_scaled"],
            data_dict["target_scaler"],
        )
        
        # Test set Xiao formula metrics
        test_only_targets = data_dict["target_scaler"].inverse_transform(
            test_only_stats["targets_scaled"].reshape(-1, 1)
        ).flatten()
        test_only_xiao_preds = test_only_stats["phys_expect"].flatten()
        test_only_xiao_metrics = {
            "r2": float(r2_score(test_only_targets, test_only_xiao_preds)),
            "evs": float(explained_variance_score(test_only_targets, test_only_xiao_preds)),
            "mae": float(mean_absolute_error(test_only_targets, test_only_xiao_preds)),
            "mse": float(mean_squared_error(test_only_targets, test_only_xiao_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test_only_targets, test_only_xiao_preds))),
            "medae": float(median_absolute_error(test_only_targets, test_only_xiao_preds)),
            "mape": float(mean_absolute_percentage_error(test_only_targets, test_only_xiao_preds)),
            "max_error": float(max_error(test_only_targets, test_only_xiao_preds)),
        }
        
        logging.info("\n[Separate test set (n=%d) - PINN]", len(test_indices))
        logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    test_only_metrics["r2"], test_only_metrics["evs"], test_only_metrics["mae"], test_only_metrics["mse"],
                    test_only_metrics["rmse"], test_only_metrics["medae"], test_only_metrics["mape"]*100, test_only_metrics["max_error"])
        
        logging.info("[Separate test set - Xiao formula]")
        logging.info("  R²=%.4f, EVS=%.4f, MAE=%.2e, MSE=%.2e, RMSE=%.2e, MedAE=%.2e, MAPE=%.2f%%, MaxErr=%.2e",
                    test_only_xiao_metrics["r2"], test_only_xiao_metrics["evs"], test_only_xiao_metrics["mae"], test_only_xiao_metrics["mse"],
                    test_only_xiao_metrics["rmse"], test_only_xiao_metrics["medae"], test_only_xiao_metrics["mape"]*100, test_only_xiao_metrics["max_error"])
        
        results["test_only"] = {
            "pinn_metrics": test_only_metrics,
            "xiao_metrics": test_only_xiao_metrics,
            "sample_count": len(test_indices),
        }
    
    logging.info("="*120)
    
    return results


def export_cv_metrics_to_excel(
    cv_results: List[Dict[str, Any]],
    best_model_results: Dict[str, Any],
    save_path: str,
    final_train_metrics: Optional[Dict[str, float]] = None,
    final_test_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Export cross-validation metrics and best model results to Excel.
    
    Args:
        cv_results: Cross-validation results list
        best_model_results: Best model performance on full dataset
        save_path: Excel save path
        final_train_metrics: Final model training set metrics (optional)
        final_test_metrics: Final model test set metrics (optional)
    """
    
    # Create Excel writer
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        
        # Sheet 1: Detailed metrics for each fold
        fold_data = []
        for cv in cv_results:
            fold_dict = {
                "Fold": cv["fold"],
                "Validation set Subset": cv["subset_label"],
                "Best Epoch": cv["best_epoch"],
                # Training set
                "Training set_R²": cv["train_metrics"]["r2"],
                "Training set_EVS": cv["train_metrics"]["evs"],
                "Training set_MAE": cv["train_metrics"]["mae"],
                "Training set_MSE": cv["train_metrics"]["mse"],
                "Training set_RMSE": cv["train_metrics"]["rmse"],
                "Training set_MedAE": cv["train_metrics"]["medae"],
                "Training set_MAPE(%)": cv["train_metrics"]["mape"] * 100,
                "Training set_MaxError": cv["train_metrics"]["max_error"],
                # Validation set
                "Validation set_R²": cv["val_metrics"]["r2"],
                "Validation set_EVS": cv["val_metrics"]["evs"],
                "Validation set_MAE": cv["val_metrics"]["mae"],
                "Validation set_MSE": cv["val_metrics"]["mse"],
                "Validation set_RMSE": cv["val_metrics"]["rmse"],
                "Validation set_MedAE": cv["val_metrics"]["medae"],
                "Validation set_MAPE(%)": cv["val_metrics"]["mape"] * 100,
                "Validation set_MaxError": cv["val_metrics"]["max_error"],
            }
            
            # Test set - PINN
            if cv["test_metrics"]:
                fold_dict.update({
                    "Test set_PINN_R²": cv["test_metrics"]["r2"],
                    "Test set_PINN_EVS": cv["test_metrics"]["evs"],
                    "Test set_PINN_MAE": cv["test_metrics"]["mae"],
                    "Test set_PINN_MSE": cv["test_metrics"]["mse"],
                    "Test set_PINN_RMSE": cv["test_metrics"]["rmse"],
                    "Test set_PINN_MedAE": cv["test_metrics"]["medae"],
                    "Test set_PINN_MAPE(%)": cv["test_metrics"]["mape"] * 100,
                    "Test set_PINN_MaxError": cv["test_metrics"]["max_error"],
                })
            
            # Test set - Xiao formula
            if cv.get("test_xiao_metrics"):
                fold_dict.update({
                    "Test set_Xiao_R²": cv["test_xiao_metrics"]["r2"],
                    "Test set_Xiao_EVS": cv["test_xiao_metrics"]["evs"],
                    "Test set_Xiao_MAE": cv["test_xiao_metrics"]["mae"],
                    "Test set_Xiao_MSE": cv["test_xiao_metrics"]["mse"],
                    "Test set_Xiao_RMSE": cv["test_xiao_metrics"]["rmse"],
                    "Test set_Xiao_MedAE": cv["test_xiao_metrics"]["medae"],
                    "Test set_Xiao_MAPE(%)": cv["test_xiao_metrics"]["mape"] * 100,
                    "Test set_Xiao_MaxError": cv["test_xiao_metrics"]["max_error"],
                })
            
            fold_data.append(fold_dict)
        
        df_folds = pd.DataFrame(fold_data)
        df_folds.to_excel(writer, sheet_name="Metrics for each fold", index=False)
        
        # Sheet 2: Average metrics for three folds
        avg_data = {
            "Metric category": ["Training set average", "Validation set average"],
            "R²": [
                np.mean([r["train_metrics"]["r2"] for r in cv_results]),
                np.mean([r["val_metrics"]["r2"] for r in cv_results]),
            ],
            "EVS": [
                np.mean([r["train_metrics"]["evs"] for r in cv_results]),
                np.mean([r["val_metrics"]["evs"] for r in cv_results]),
            ],
            "MAE": [
                np.mean([r["train_metrics"]["mae"] for r in cv_results]),
                np.mean([r["val_metrics"]["mae"] for r in cv_results]),
            ],
            "MSE": [
                np.mean([r["train_metrics"]["mse"] for r in cv_results]),
                np.mean([r["val_metrics"]["mse"] for r in cv_results]),
            ],
            "RMSE": [
                np.mean([r["train_metrics"]["rmse"] for r in cv_results]),
                np.mean([r["val_metrics"]["rmse"] for r in cv_results]),
            ],
            "MedAE": [
                np.mean([r["train_metrics"]["medae"] for r in cv_results]),
                np.mean([r["val_metrics"]["medae"] for r in cv_results]),
            ],
            "MAPE(%)": [
                np.mean([r["train_metrics"]["mape"] for r in cv_results]) * 100,
                np.mean([r["val_metrics"]["mape"] for r in cv_results]) * 100,
            ],
            "MaxError": [
                np.mean([r["train_metrics"]["max_error"] for r in cv_results]),
                np.mean([r["val_metrics"]["max_error"] for r in cv_results]),
            ],
        }
        
        if cv_results[0]["test_metrics"]:
            avg_data["Metric category"].append("Test set average_PINN")
            avg_data["R²"].append(np.mean([r["test_metrics"]["r2"] for r in cv_results]))
            avg_data["EVS"].append(np.mean([r["test_metrics"]["evs"] for r in cv_results]))
            avg_data["MAE"].append(np.mean([r["test_metrics"]["mae"] for r in cv_results]))
            avg_data["MSE"].append(np.mean([r["test_metrics"]["mse"] for r in cv_results]))
            avg_data["RMSE"].append(np.mean([r["test_metrics"]["rmse"] for r in cv_results]))
            avg_data["MedAE"].append(np.mean([r["test_metrics"]["medae"] for r in cv_results]))
            avg_data["MAPE(%)"].append(np.mean([r["test_metrics"]["mape"] for r in cv_results]) * 100)
            avg_data["MaxError"].append(np.mean([r["test_metrics"]["max_error"] for r in cv_results]))
        
        if cv_results[0].get("test_xiao_metrics"):
            avg_data["Metric category"].append("Test set average_Xiao")
            avg_data["R²"].append(np.mean([r["test_xiao_metrics"]["r2"] for r in cv_results]))
            avg_data["EVS"].append(np.mean([r["test_xiao_metrics"]["evs"] for r in cv_results]))
            avg_data["MAE"].append(np.mean([r["test_xiao_metrics"]["mae"] for r in cv_results]))
            avg_data["MSE"].append(np.mean([r["test_xiao_metrics"]["mse"] for r in cv_results]))
            avg_data["RMSE"].append(np.mean([r["test_xiao_metrics"]["rmse"] for r in cv_results]))
            avg_data["MedAE"].append(np.mean([r["test_xiao_metrics"]["medae"] for r in cv_results]))
            avg_data["MAPE(%)"].append(np.mean([r["test_xiao_metrics"]["mape"] for r in cv_results]) * 100)
            avg_data["MaxError"].append(np.mean([r["test_xiao_metrics"]["max_error"] for r in cv_results]))
        
        df_avg = pd.DataFrame(avg_data)
        df_avg.to_excel(writer, sheet_name="Average metrics for three folds", index=False)
        
        # Sheet 3: Best model performance on full dataset and test set
        best_model_data = []
        
        if "full_dataset" in best_model_results:
            best_model_data.append({
                "Dataset": "Full dataset",
                "Model": "PINN",
                "Sample count": best_model_results["full_dataset"]["sample_count"],
                "R²": best_model_results["full_dataset"]["pinn_metrics"]["r2"],
                "EVS": best_model_results["full_dataset"]["pinn_metrics"]["evs"],
                "MAE": best_model_results["full_dataset"]["pinn_metrics"]["mae"],
                "MSE": best_model_results["full_dataset"]["pinn_metrics"]["mse"],
                "RMSE": best_model_results["full_dataset"]["pinn_metrics"]["rmse"],
                "MedAE": best_model_results["full_dataset"]["pinn_metrics"]["medae"],
                "MAPE(%)": best_model_results["full_dataset"]["pinn_metrics"]["mape"] * 100,
                "MaxError": best_model_results["full_dataset"]["pinn_metrics"]["max_error"],
            })
            best_model_data.append({
                "Dataset": "Full dataset",
                "Model": "Xiao formula",
                "Sample count": best_model_results["full_dataset"]["sample_count"],
                "R²": best_model_results["full_dataset"]["xiao_metrics"]["r2"],
                "EVS": best_model_results["full_dataset"]["xiao_metrics"]["evs"],
                "MAE": best_model_results["full_dataset"]["xiao_metrics"]["mae"],
                "MSE": best_model_results["full_dataset"]["xiao_metrics"]["mse"],
                "RMSE": best_model_results["full_dataset"]["xiao_metrics"]["rmse"],
                "MedAE": best_model_results["full_dataset"]["xiao_metrics"]["medae"],
                "MAPE(%)": best_model_results["full_dataset"]["xiao_metrics"]["mape"] * 100,
                "MaxError": best_model_results["full_dataset"]["xiao_metrics"]["max_error"],
            })
        
        if "test_only" in best_model_results:
            best_model_data.append({
                "Dataset": "Separate test set",
                "Model": "PINN",
                "Sample count": best_model_results["test_only"]["sample_count"],
                "R²": best_model_results["test_only"]["pinn_metrics"]["r2"],
                "EVS": best_model_results["test_only"]["pinn_metrics"]["evs"],
                "MAE": best_model_results["test_only"]["pinn_metrics"]["mae"],
                "MSE": best_model_results["test_only"]["pinn_metrics"]["mse"],
                "RMSE": best_model_results["test_only"]["pinn_metrics"]["rmse"],
                "MedAE": best_model_results["test_only"]["pinn_metrics"]["medae"],
                "MAPE(%)": best_model_results["test_only"]["pinn_metrics"]["mape"] * 100,
                "MaxError": best_model_results["test_only"]["pinn_metrics"]["max_error"],
            })
            best_model_data.append({
                "Dataset": "Separate test set",
                "Model": "Xiao formula",
                "Sample count": best_model_results["test_only"]["sample_count"],
                "R²": best_model_results["test_only"]["xiao_metrics"]["r2"],
                "EVS": best_model_results["test_only"]["xiao_metrics"]["evs"],
                "MAE": best_model_results["test_only"]["xiao_metrics"]["mae"],
                "MSE": best_model_results["test_only"]["xiao_metrics"]["mse"],
                "RMSE": best_model_results["test_only"]["xiao_metrics"]["rmse"],
                "MedAE": best_model_results["test_only"]["xiao_metrics"]["medae"],
                "MAPE(%)": best_model_results["test_only"]["xiao_metrics"]["mape"] * 100,
                "MaxError": best_model_results["test_only"]["xiao_metrics"]["max_error"],
            })
        
        df_best = pd.DataFrame(best_model_data)
        df_best.to_excel(writer, sheet_name="Best model performance on full dataset and test set", index=False)
        
        # Sheet 4: Final model metrics (training set and test set)
        if final_train_metrics is not None and final_test_metrics is not None:
            final_model_data = {
                "Dataset": ["Training set (S1+2+3)", "Test set"],
                "R²": [final_train_metrics["r2"], final_test_metrics["r2"]],
                "EVS": [final_train_metrics["evs"], final_test_metrics["evs"]],
                "MAE": [final_train_metrics["mae"], final_test_metrics["mae"]],
                "MSE": [final_train_metrics["mse"], final_test_metrics["mse"]],
                "RMSE": [final_train_metrics["rmse"], final_test_metrics["rmse"]],
                "MedAE": [final_train_metrics["medae"], final_test_metrics["medae"]],
                "MAPE (%)": [final_train_metrics["mape"] * 100, final_test_metrics["mape"] * 100],
                "MaxError": [final_train_metrics["max_error"], final_test_metrics["max_error"]],
            }
            df_final = pd.DataFrame(final_model_data)
            df_final.to_excel(writer, sheet_name="Final model metrics (training set and test set)", index=False)
        
        # Sheet 5: Radar chart data (for Origin plotting, contains 5 metrics: R², RMSE, MSE, MAE, MAPE)
        # Calculate average training set metrics for three folds
        avg_train_r2 = np.mean([r["train_metrics"]["r2"] for r in cv_results])
        avg_train_rmse = np.mean([r["train_metrics"]["rmse"] for r in cv_results])
        avg_train_mse = np.mean([r["train_metrics"]["mse"] for r in cv_results])
        avg_train_mae = np.mean([r["train_metrics"]["mae"] for r in cv_results])
        avg_train_mape = np.mean([r["train_metrics"]["mape"] for r in cv_results]) * 100
        
        # Calculate average test set metrics for three folds
        if cv_results[0]["test_metrics"]:
            avg_test_r2 = np.mean([r["test_metrics"]["r2"] for r in cv_results])
            avg_test_rmse = np.mean([r["test_metrics"]["rmse"] for r in cv_results])
            avg_test_mse = np.mean([r["test_metrics"]["mse"] for r in cv_results])
            avg_test_mae = np.mean([r["test_metrics"]["mae"] for r in cv_results])
            avg_test_mape = np.mean([r["test_metrics"]["mape"] for r in cv_results]) * 100
        else:
            avg_test_r2 = avg_test_rmse = avg_test_mse = avg_test_mae = avg_test_mape = np.nan
        
        # Final model metrics
        if final_train_metrics is not None and final_test_metrics is not None:
            final_train_r2 = final_train_metrics["r2"]
            final_train_rmse = final_train_metrics["rmse"]
            final_train_mse = final_train_metrics["mse"]
            final_train_mae = final_train_metrics["mae"]
            final_train_mape = final_train_metrics["mape"] * 100
            
            final_test_r2 = final_test_metrics["r2"]
            final_test_rmse = final_test_metrics["rmse"]
            final_test_mse = final_test_metrics["mse"]
            final_test_mae = final_test_metrics["mae"]
            final_test_mape = final_test_metrics["mape"] * 100
        else:
            final_train_r2 = final_train_rmse = final_train_mse = final_train_mae = final_train_mape = np.nan
            final_test_r2 = final_test_rmse = final_test_mse = final_test_mae = final_test_mape = np.nan
        
        radar_data = {
            "Metric": ["R²", "RMSE", "MSE", "MAE", "MAPE (%)"],
            "Mean_Train": [avg_train_r2, avg_train_rmse, avg_train_mse, avg_train_mae, avg_train_mape],
            "Mean_Test": [avg_test_r2, avg_test_rmse, avg_test_mse, avg_test_mae, avg_test_mape],
            "Final_Train": [final_train_r2, final_train_rmse, final_train_mse, final_train_mae, final_train_mape],
            "Final_Test": [final_test_r2, final_test_rmse, final_test_mse, final_test_mae, final_test_mape],
        }
        df_radar = pd.DataFrame(radar_data)
        df_radar.to_excel(writer, sheet_name="Radar chart data", index=False)
    
    logging.info("✓ Cross-validation metrics exported to Excel: %s", save_path)


def save_artifacts(
    base_config: BaseConfig,
    train_config: TrainingConfig,
    results: Dict[str, Any],
    data_dict: Dict[str, Any],
    cv_results: Optional[List[Dict[str, Any]]] = None,
    best_model_results: Optional[Dict[str, Any]] = None,
    study: Optional[optuna.study.Study] = None,
    strategy_name: str = "Subset",
) -> None:
    """Save trained model, configuration, and necessary preprocessors (for subsequent inference)."""

    os.makedirs(base_config.save_dir, exist_ok=True)

    # 1. Save model weights
    model_path = os.path.join(base_config.save_dir, "pinn_peak_strain.pt")
    torch.save(results["model"].state_dict(), model_path)
    logging.info("✓ Model weights saved: %s", model_path)

    # 2. Save training configuration and best hyperparameters
    train_summary = {
        "strategy": strategy_name,
        "base_config": base_config.to_dict(),
        "train_config": train_config.to_dict(),
        "best_epoch": results["best_state"]["epoch"],
        "validation_metrics": results["best_state"]["val_metrics"],
        "test_metrics": results["test_metrics"] if results["test_metrics"] else None,
        "test_xiao_metrics": results.get("test_xiao_metrics"),  # Test set Xiao formula metrics
        "best_model_results": best_model_results,  # Best model performance on full dataset
        "optuna_best_params": study.best_params if study is not None else None,
        "optuna_best_value": study.best_value if study is not None else None,
        "cv_summary": [
            {
                "fold": item["fold"],
                "subset_label": item.get("subset_label", f"subset{item['fold']}"),
                "best_epoch": item.get("best_epoch"),
                "train_r2": item["train_metrics"]["r2"],
                "train_mae": item["train_metrics"]["mae"],
                "val_r2": item["val_metrics"]["r2"],
                "val_mae": item["val_metrics"]["mae"],
                "test_r2": item["test_metrics"]["r2"] if item["test_metrics"] else None,
                "test_mae": item["test_metrics"]["mae"] if item["test_metrics"] else None,
            }
            for item in cv_results
        ]
        if cv_results
        else None,
    }
    summary_path = os.path.join(base_config.save_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, ensure_ascii=False, indent=2)
    logging.info("✓ Training summary saved: %s", summary_path)

    # 3. Save data preprocessors (required for inference with the same scaler)
    import pickle
    scaler_data = {
        "feature_scaler": data_dict["feature_scaler"],
        "target_scaler": data_dict["target_scaler"],
        "feature_cols": data_dict["feature_cols"],
    }
    scaler_path = os.path.join(base_config.save_dir, "scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)
    logging.info("✓ Data preprocessors saved: %s", scaler_path)
    
    # 4. Save model architecture information (for model reconstruction)
    model_arch = {
        "input_dim": train_config.input_dim,
        "hidden_dim": train_config.hidden_dim,
        "num_layers": train_config.num_layers,
        "dropout": train_config.dropout,
    }
    arch_path = os.path.join(base_config.save_dir, "model_architecture.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(model_arch, f, ensure_ascii=False, indent=2)
    logging.info("✓ Model architecture saved: %s", arch_path)
    
    # 5. Export cross-validation metrics to Excel
    if cv_results and best_model_results:
        excel_path = os.path.join(base_config.save_dir, "training_metrics.xlsx")
        # Extract final model training set and test set metrics from results
        final_train_metrics = results.get("train_metrics_final")
        final_test_metrics = results.get("test_metrics_final")
        export_cv_metrics_to_excel(
            cv_results, 
            best_model_results, 
            excel_path,
            final_train_metrics=final_train_metrics,
            final_test_metrics=final_test_metrics
        )



def init_environment(config: BaseConfig) -> None:
    """Initialize random seed, logging, and save directory."""
    os.makedirs(config.save_dir, exist_ok=True)
    set_random_seed(config.seed)
    configure_logging(os.path.join(config.save_dir, "training.log"))
    
    logging.info("\n" + "="*80)
    logging.info("PINN Peak Strain Prediction Model - Environment initialization")
    logging.info("="*80)
    logging.info("Project root directory: %s", PROJECT_ROOT)
    logging.info("Result save directory: %s", config.save_dir)
    logging.info("Running device: %s", config.device)
    logging.info("Random seed: %d", config.seed)
    logging.info("="*80 + "\n")


def print_training_config_summary(train_config: TrainingConfig) -> None:
    """Print training configuration summary."""
    logging.info("\n" + "="*80)
    logging.info("Training configuration summary")
    logging.info("="*80)
    logging.info("Network structure:")
    logging.info("  - Hidden layer dimension: %d", train_config.hidden_dim)
    logging.info("  - Number of hidden layers: %d", train_config.num_layers)
    logging.info("  - Dropout rate: %.3f", train_config.dropout)
    logging.info("\nRegularization:")
    logging.info("  - L2 weight decay: %.1e", train_config.weight_decay)
    logging.info("  - Gradient clipping: %s", train_config.gradient_clip)
    logging.info("\nOptimization strategy:")
    logging.info("  - Learning rate: %.1e", train_config.learning_rate)
    logging.info("  - Learning rate decay: %.3f (per epoch)", train_config.scheduler_gamma)
    logging.info("\nLoss function weights:")
    logging.info("  - Data loss λ_data: %.2f", train_config.lambda_data)
    logging.info("  - Physical loss λ_physics: %.4f", train_config.lambda_physics)
    logging.info("  - Weight ratio: %.1f:1 (data:physical)", 
                train_config.lambda_data / train_config.lambda_physics)
    logging.info("\nTraining strategy:")
    logging.info("  - Maximum training epochs: %d", train_config.max_epochs)
    logging.info("  - Early stopping patience: %d", train_config.patience)
    logging.info("="*80 + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINN Peak Strain Trainer (MinMaxScaler version)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna hyperparameter search, use optimized default configuration")
    parser.add_argument("--tune-trials", type=int, default=50, help="Optuna search trials (recommended 50+ for 0.87 performance)")
    parser.add_argument("--final-epochs", type=int, default=500, help="Final model training epochs")
    parser.add_argument("--save-dir", type=str, default=None, help="Custom save directory")
    parser.add_argument("--device", type=str, default=None, help="Specify running device, e.g. cpu or cuda:0")
    parser.add_argument("--lambda-physics", type=float, default=None, help="Final training physical loss weight")
    parser.add_argument("--lambda-data", type=float, default=None, help="Final training data loss weight")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader threads")
    parser.add_argument("--train-batch", type=int, default=None, help="Training batch size")
    parser.add_argument("--eval-batch", type=int, default=None, help="Validation/test batch size")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test set ratio")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation set ratio (relative to remaining data)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset Excel path")
    parser.add_argument("--sheet", type=int, default=None, help="Excel sheet index")
    parser.add_argument("--feature-columns", type=str, nargs="+", default=None, help="Manually specify feature columns")
    return parser.parse_args(args=argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    base_config = BaseConfig()
    if args.save_dir:
        base_config.save_dir = os.path.abspath(args.save_dir)
    if args.device:
        base_config.device = args.device
    if args.num_workers is not None:
        base_config.num_workers = args.num_workers
    if args.train_batch is not None:
        base_config.train_batch_size = args.train_batch
    if args.eval_batch is not None:
        base_config.eval_batch_size = args.eval_batch
    if args.test_ratio is not None:
        base_config.test_ratio = args.test_ratio
    if args.val_ratio is not None:
        base_config.val_ratio = args.val_ratio
    if args.dataset is not None:
        base_config.dataset_path = os.path.abspath(args.dataset)
    if args.sheet is not None:
        base_config.dataset_sheet = args.sheet
    if args.feature_columns is not None:
        base_config.feature_columns = args.feature_columns

    init_environment(base_config)
    data_dict = preprocess_data(base_config)

    base_train_config = TrainingConfig()
    base_train_config.max_epochs = args.final_epochs
    base_train_config.patience = max(20, base_train_config.max_epochs // 5)
    if args.lambda_physics is not None:
        base_train_config.lambda_physics = args.lambda_physics
    if args.lambda_data is not None:
        base_train_config.lambda_data = args.lambda_data

    study: Optional[optuna.study.Study] = None
    if not args.skip_tuning:
        study = optimize_hyperparameters(
            data_dict=data_dict,
            base_config=base_config,
            base_train_config=base_train_config,
            n_trials=args.tune_trials,
        )
        # 应用最优超参数
        for key, value in study.best_params.items():
            if hasattr(base_train_config, key):
                setattr(base_train_config, key, value)
        base_train_config.max_epochs = args.final_epochs
        base_train_config.patience = max(20, base_train_config.max_epochs // 5)
        logging.info("\n✓ Optuna best hyperparameters applied to final training configuration")
    else:
        logging.info("\n⚠ Skip hyperparameter search, use default configuration for training")

    # Print final training configuration
    print_training_config_summary(base_train_config)

    # ==================== Start cross-validation based on predefined Subset ====================
    logging.info("\n" + "#"*80)
    logging.info("Start cross-validation based on predefined Subset")
    logging.info("#"*80)
    
    cv_results, final_results = train_with_subset_strategy(
        data_dict=data_dict,
        base_config=base_config,
        train_config=base_train_config,
    )
    
    strategy_name = "Subset"
    
    # ==================== Evaluate best model ====================
    best_model_results = evaluate_best_model_on_full_dataset(
        model=final_results["model"],
        data_dict=data_dict,
        base_config=base_config,
        target_data_min=final_results["target_data_min"],
        target_range=final_results["target_range"],
        train_config=base_train_config,
    )

    # ==================== Save training results ====================
    save_artifacts(
        base_config=base_config,
        train_config=base_train_config,
        results=final_results,
        data_dict=data_dict,
        cv_results=cv_results,
        best_model_results=best_model_results,
        study=study,
        strategy_name=strategy_name,
    )

    # ==================== Predict all data and write to Excel ====================
    logging.info("\n" + "="*80)
    logging.info("Predict all data (including test set) and write to Excel")
    logging.info("="*80)
    
    try:
        # Use final model to predict all data
        device = torch.device(base_config.device)
        final_model = final_results["model"].to(device)
        final_model.eval()
        
        # Get all data features (scaled)
        all_features_scaled = data_dict["features_scaled"]
        all_sample_ids = data_dict["sample_ids"]
        
        # Predict all data
        with torch.no_grad():
            features_tensor = torch.from_numpy(all_features_scaled).to(device)
            preds_scaled = final_model(features_tensor).cpu().numpy().flatten()
            # Inverse transform
            preds = data_dict["target_scaler"].inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).flatten()
        
        # Create prediction result dictionary
        pred_dict = {}
        for idx, sample_id in enumerate(all_sample_ids):
            pred_dict[sample_id] = float(preds[idx])
        
        # Reload original Excel file
        data_file = base_config.dataset_path
        df_original = pd.read_excel(data_file, sheet_name=base_config.dataset_sheet)
        
        # Use map method to match
        if 'No_Customized' in df_original.columns:
            df_original['PINN_peak_strain'] = df_original['No_Customized'].map(pred_dict)
        else:
            logging.warning("No_Customized column not found, will match predictions by row order")
            # If No_Customized column is not found, match predictions by row order (ensure order consistency)
            if len(preds) == len(df_original):
                df_original['PINN_peak_strain'] = preds
            else:
                logging.error("Prediction sample number (%d) does not match the number of rows in the original data (%d), cannot automatically match", 
                            len(preds), len(df_original))
                raise ValueError("Sample number does not match, cannot write prediction results")
        
        # Save results
        output_file = os.path.join(PROJECT_ROOT, "dataset", "dataset_with_PINN_peak_strain.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_original.to_excel(output_file, index=False)
        logging.info("✓ Prediction results written to: %s", output_file)
        logging.info("  - New column: PINN_peak_strain (final model prediction value, trained using all non-test data)")
        logging.info("  - Total prediction sample number: %d", len(preds))
        logging.info("  - Number of non-empty prediction values: %d", df_original['PINN_peak_strain'].notna().sum())
        
        # Backup to save directory
        backup_file = os.path.join(base_config.save_dir, 'dataset_with_PINN_peak_strain_backup.xlsx')
        df_original.to_excel(backup_file, index=False)
        logging.info("✓ Backup file saved to: %s", backup_file)
        
    except Exception as e:
        logging.error("Warning: Error writing to Excel: %s", e)
        import traceback
        logging.error(traceback.format_exc())

    logging.info("\n" + "="*80)
    logging.info("✓ Training completed! Model and configuration saved to: %s", base_config.save_dir)
    logging.info("="*80)
    logging.info("Saved files:")
    logging.info("  - pinn_peak_strain.pt: model weights")
    logging.info("  - training_summary.json: training configuration and metrics")
    logging.info("  - training_metrics.xlsx: cross-validation detailed metrics")
    logging.info("  - scalers.pkl: data preprocessor")
    logging.info("  - model_architecture.json: model architecture")
    logging.info("  - dataset_with_PINN_peak_strain.xlsx: all data prediction results")
    logging.info("\n Follow-up Actions:")
    logging.info("  1. View training_metrics.xlsx to understand detailed training metrics")
    logging.info("  2. View dataset_with_PINN_peak_strain.xlsx to view all sample prediction values")
    logging.info("  3. Run inference script for model evaluation and comparison with Xiao formula")
    logging.info("     python peak_strain_inference.py --model-dir %s --dataset <data.xlsx>", base_config.save_dir)
    logging.info("="*80 + "\n")


if __name__ == "__main__":
    main()

