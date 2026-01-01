"""
Data loading, normalization, and clustering module
Contains functions for loading data from Excel, assigning cluster labels, normalization, etc.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Sequence, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ========== Clustering-related configurations ==========
CLUSTER_FEATURE_COLUMNS = [
    'water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI',
    'age', 'μe', 'DJB', 'side', 'GJB'
]
CLUSTER_CACHE_NAME = 'cluster_labels.npy'


def search_best_n_clusters(X_scaled: np.ndarray, min_n: int = 2, max_n: int = 30, verbose: bool = True) -> Tuple[int, list]:
    """
    Automatically search for the optimal number of clusters based on silhouette coefficient (refer to cluster_label_dataset_cvfixed.py)
    
    Fully aligns with the implementation logic of the reference code:
    - Uses AgglomerativeClustering for hierarchical clustering
    - Uses silhouette_score to evaluate clustering quality
    - Sets silhouette coefficient to -1 when the number of clusters is less than 2
    
    Args:
        X_scaled: Standardized feature matrix
        min_n: Minimum number of clusters (default 2, consistent with reference code)
        max_n: Maximum number of clusters (default 30, consistent with reference code)
        verbose: Whether to print detailed information
    
    Returns:
        best_n_clusters: Optimal number of clusters
        all_scores: List of silhouette coefficients corresponding to all cluster counts
    """
    best_score = -1
    best_n_clusters = min_n
    all_scores = []
    
    for n_clusters in range(min_n, max_n + 1):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = agg.fit_predict(X_scaled)
        
        if len(np.unique(clusters)) < 2:
            score = -1
        else:
            score = silhouette_score(X_scaled, clusters)
        
        all_scores.append(score)
        
        if verbose:
            print(f'n_clusters={n_clusters}, silhouette={score:.4f}')
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    if verbose:
        print(f'Best n_clusters: {best_n_clusters}, silhouette: {best_score:.4f}')
    
    return best_n_clusters, all_scores


def assign_cluster_labels(material_df: pd.DataFrame, cache_dir: Optional[Path] = None, 
                          default_cluster_count: Optional[int] = None,
                          min_clusters: int = 2, max_clusters: int = 30,
                          verbose: bool = True, force_regenerate: bool = False) -> np.ndarray:
    """
    Assign cluster labels to samples, prioritizing reading from cache.
    Refers to the algorithm in cluster_label_dataset_cvfixed.py, using silhouette coefficient to automatically search for optimal cluster count (range 2-30)
    
    Args:
        material_df: Material parameter DataFrame
        cache_dir: Cache directory (skip caching if None)
        default_cluster_count: Default number of clusters (use directly if specified, no automatic search)
        min_clusters: Minimum number of clusters for automatic search
        max_clusters: Maximum number of clusters for automatic search
        verbose: Whether to print detailed information
        force_regenerate: Whether to force regenerate cluster labels (ignore cache)
    
    Returns:
        cluster_labels: Array of cluster labels
    """
    # If DataFrame already has 'cluster_id' column and not forcing regeneration, return directly
    if 'cluster_id' in material_df.columns and not force_regenerate:
        labels = material_df['cluster_id'].to_numpy()
        return labels.astype(int)
    
    # If forcing regeneration, remove 'cluster_id' column from DataFrame
    if force_regenerate and 'cluster_id' in material_df.columns:
        material_df = material_df.drop(columns=['cluster_id'])
    
    # Try to read from cache (only when cache_dir is not None and not forcing regeneration)
    cache_path = None
    if cache_dir is not None and not force_regenerate:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / CLUSTER_CACHE_NAME
        if cache_path.exists():
            try:
                labels = np.load(cache_path)
                if len(labels) == len(material_df):
                    material_df['cluster_id'] = labels
                    if verbose:
                        print(f"Loaded cluster labels from cache: {cache_path}")
                    return labels.astype(int)
            except Exception as exc:
                if verbose:
                    print(f"[Warning] Failed to read cluster cache: {exc}, will recalculate.")
    
    # If forcing regeneration, delete old cache file
    if force_regenerate and cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / CLUSTER_CACHE_NAME
        if cache_path.exists():
            try:
                cache_path.unlink()
                if verbose:
                    print(f"Deleted old cluster cache file: {cache_path}")
            except Exception as exc:
                if verbose:
                    print(f"[Warning] Failed to delete old cache file: {exc}")
    
    # Check if required feature columns for clustering exist
    available_features = [col for col in CLUSTER_FEATURE_COLUMNS if col in material_df.columns]
    if not available_features:
        raise ValueError("Missing required feature columns for clustering, cannot assign cluster_id.")
    
    # Extract features and standardize (exactly consistent with reference code)
    X = material_df[available_features].astype(float).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of clusters (refer to logic in cluster_label_dataset_cvfixed.py)
    if default_cluster_count is not None and default_cluster_count > 0:
        n_clusters = default_cluster_count
        if verbose:
            print(f"Using specified number of clusters: {n_clusters}")
    else:
        # Automatically search for optimal number of clusters (refer to cluster_label_dataset_cvfixed.py, range 2-30)
        if verbose:
            print(f"Automatically searching for optimal number of clusters (range: {min_clusters}-{max_clusters})...")
        n_clusters, _ = search_best_n_clusters(X_scaled, min_n=min_clusters, max_n=max_clusters, verbose=verbose)
        n_clusters = max(1, n_clusters)
    
    # Perform clustering (using AgglomerativeClustering, consistent with reference code)
    if n_clusters <= 1:
        labels = np.zeros(len(material_df), dtype=int)
    else:
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X_scaled)
    
    # Save to DataFrame and cache (only when cache_dir is not None)
    material_df['cluster_id'] = labels
    if cache_dir is not None:
        # Ensure cache_path is set
        if cache_path is None:
            cache_dir = Path(cache_dir)
            cache_path = cache_dir / CLUSTER_CACHE_NAME
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, labels)
            if verbose:
                print(f"Cluster labels saved to cache: {cache_path}")
        except Exception as exc:
            if verbose:
                print(f"[Warning] Failed to save cluster cache: {exc}")
    
    return labels.astype(int)


def calculate_xiao_curve_with_real_peaks(sigma_cp: float, epsilon_cp: float, r_value: float,
                                         strain_sequence: np.ndarray) -> np.ndarray:
    """
    Calculate stress-strain curve using Xiao et al.'s formula (original scale, unit: MPa).
    
    Args:
        sigma_cp: Peak stress
        epsilon_cp: Peak strain
        r_value: Mass replacement rate
        strain_sequence: Strain sequence
    
    Returns:
        stress_curve: Calculated stress curve
    """
    sigma_cp = float(sigma_cp) if sigma_cp is not None else 0.0
    epsilon_cp = float(epsilon_cp) if epsilon_cp is not None else 0.0
    if not np.isfinite(sigma_cp):
        sigma_cp = 0.0
    if not np.isfinite(epsilon_cp) or epsilon_cp <= 0:
        epsilon_cp = max(float(np.max(strain_sequence)), 1e-4)

    r_value = float(r_value) if r_value is not None else 0.0
    if r_value > 1.0:
        r_value = r_value / 100.0

    # Calculate coefficients
    a = 2.2 * (0.748 * r_value**2 - 1.231 * r_value + 0.975)
    b = 0.8 * (7.6483 * r_value + 1.142)

    strain_sequence = np.asarray(strain_sequence, dtype=float)
    if strain_sequence.size == 0:
        return np.zeros(0, dtype=float)

    x = np.divide(strain_sequence, epsilon_cp, out=np.zeros_like(strain_sequence, dtype=float), where=epsilon_cp != 0)

    stress_ratio = np.zeros_like(x, dtype=float)
    mask_ascending = x < 1
    if np.any(mask_ascending):
        xa = x[mask_ascending]
        stress_ratio[mask_ascending] = a * xa + (3 - 2 * a) * xa**2 + (a - 2) * xa**3

    mask_descending = ~mask_ascending
    if np.any(mask_descending):
        xd = x[mask_descending]
        denominator = b * (xd - 1) ** 2 + xd
        denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
        stress_ratio[mask_descending] = xd / denominator

    stress_curve = stress_ratio * sigma_cp
    # Replace invalid values
    stress_curve = np.nan_to_num(stress_curve, nan=0.0, posinf=sigma_cp, neginf=0.0)
    return stress_curve


def compute_xiao_and_residual_curves(strain_original: np.ndarray,
                                     stress_original: np.ndarray,
                                     material_params_original: np.ndarray,
                                     peak_stress_original: np.ndarray,
                                     peak_strain_original: np.ndarray,
                                     material_param_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch compute Xiao reference curves and residual curves based on original scale data.
    
    Args:
        strain_original: Original strain data [samples, curve_length]
        stress_original: Original stress data [samples, curve_length]
        material_params_original: Original material parameters [samples, num_params]
        peak_stress_original: Original peak stress [samples]
        peak_strain_original: Original peak strain [samples]
        material_param_cols: List of material parameter column names
    
    Returns:
        xiao_curves: Xiao reference curves [samples, curve_length]
        residual_curves: Residual curves [samples, curve_length]
    """
    if 'r' in material_param_cols:
        r_index = material_param_cols.index('r')
    else:
        r_index = 0

    xiao_curves = []
    residual_curves = []
    num_samples = stress_original.shape[0]

    for i in range(num_samples):
        strain_seq = strain_original[i]
        stress_seq = stress_original[i]
        sigma_cp = peak_stress_original[i] if i < len(peak_stress_original) else np.max(stress_seq)
        epsilon_cp = peak_strain_original[i] if i < len(peak_strain_original) else strain_seq[np.argmax(stress_seq)]

        if not np.isfinite(sigma_cp):
            sigma_cp = np.max(stress_seq)
        if not np.isfinite(epsilon_cp) or epsilon_cp <= 0:
            epsilon_cp = strain_seq[np.argmax(stress_seq)] if strain_seq.size else 1.0
            epsilon_cp = epsilon_cp if epsilon_cp > 0 else 1e-4

        r_value = material_params_original[i, r_index] if material_params_original.ndim > 1 else material_params_original[r_index]
        xiao_curve = calculate_xiao_curve_with_real_peaks(sigma_cp, epsilon_cp, r_value, strain_seq)
        if xiao_curve.shape != stress_seq.shape:
            xiao_curve = np.interp(
                np.linspace(0, len(xiao_curve) - 1, len(stress_seq)),
                np.arange(len(xiao_curve)),
                xiao_curve
            )
        residual_curve = stress_seq - xiao_curve

        xiao_curves.append(xiao_curve)
        residual_curves.append(residual_curve)

    return np.asarray(xiao_curves, dtype=float), np.asarray(residual_curves, dtype=float)


def load_excel_data(material_params_file: str, stress_data_file: str, curve_length: int = 800,
                    train_indices: Optional[np.ndarray] = None,
                    cache_dir: Optional[Path] = None, use_cache: bool = True,
                    default_cluster_count: Optional[int] = None,
                    verbose: bool = True) -> Tuple:
    """
    Load material parameters and stress-strain data from Excel files, and perform normalization
    
    Args:
        material_params_file: Excel file path (first worksheet: material parameters)
        stress_data_file: Excel file path (second worksheet: stress-strain data)
        curve_length: Uniform sampling length (default 800 points)
        train_indices: Training set indices (used to calculate normalization parameters to avoid data leakage)
        cache_dir: Cache directory
        use_cache: Whether to use cache
        default_cluster_count: Default number of clusters (None for automatic search)
        verbose: Whether to print detailed information
    
    Returns:
        X_strain: Normalized strain data [samples, curve_length]
        X_stress: Normalized stress data [samples, curve_length] 
        X_material: Normalized material parameters [samples, num_params]
        X_peak_stress: Normalized peak stress [samples]
        X_peak_strain: Normalized peak strain [samples]
        material_param_cols: Material parameter column names
        strain_scaler: Strain normalizer
        stress_scaler: Stress normalizer
        material_scaler: Material parameter normalizer
        peak_stress_scaler: Peak stress normalizer
        peak_strain_scaler: Peak strain normalizer
        X_material_original: Original material parameters
        X_stress_original: Original stress data
        X_peak_stress_original: Original peak stress
        X_peak_strain_original: Original peak strain
        sample_divisions: List of dataset division labels
        cluster_labels: Cluster labels
        extra_data: Dictionary of additional data (including Xiao curves and residual curves)
    """
    def log(message: str, force: bool = False):
        if verbose or force:
            print(message)
    
    log("=== Data Loading ===", force=True)
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"{Path(material_params_file).stem}_curve{curve_length}.npz"
        cache_path = cache_dir / cache_key
    
    # Try to load from cache
    if use_cache and cache_path and cache_path.exists():
        log(f"Loading data from cache: {cache_path}", force=True)
        cached = np.load(cache_path, allow_pickle=True)
        cached_files = set(cached.files)
        required_curve_keys = {'X_xiao_curve', 'X_residual_curve', 'X_xiao_curve_original', 'X_residual_curve_original'}

        X_strain = cached['X_strain']
        X_stress = cached['X_stress']
        X_material = cached['X_material']
        X_peak_stress = cached['X_peak_stress']
        X_peak_strain = cached['X_peak_strain']
        material_param_cols = cached['material_param_cols'].tolist()
        strain_scaler = cached['strain_scaler'].item()
        stress_scaler = cached['stress_scaler'].item()
        material_scaler = cached['material_scaler'].item()
        peak_stress_scaler = cached['peak_stress_scaler'].item()
        peak_strain_scaler = cached['peak_strain_scaler'].item()
        X_material_original = cached['X_material_original']
        X_stress_original = cached['X_stress_original']
        X_peak_stress_original = cached['X_peak_stress_original']
        X_peak_strain_original = cached['X_peak_strain_original']
        sample_divisions = cached['sample_divisions']

        if 'cluster_labels' in cached_files:
            cluster_labels = cached['cluster_labels']
        else:
            log("Cache missing cluster label information, recalculating cluster_id", force=True)
            material_df_cached = pd.read_excel(material_params_file, sheet_name=0)
            cluster_labels = assign_cluster_labels(material_df_cached, cache_dir, default_cluster_count, verbose=verbose)

        if required_curve_keys.issubset(cached_files):
            X_xiao_curve = cached['X_xiao_curve']
            X_residual_curve = cached['X_residual_curve']
            X_xiao_curve_original = cached['X_xiao_curve_original']
            X_residual_curve_original = cached['X_residual_curve_original']
            cached.close()

            extra_data = {
                'xiao_curve_normalized': X_xiao_curve,
                'residual_curve_normalized': X_residual_curve,
                'xiao_curve_original': X_xiao_curve_original,
                'residual_curve_original': X_residual_curve_original,
            }
            log("Cache loading completed", force=True)
            return (X_strain, X_stress, X_material, X_peak_stress, X_peak_strain, material_param_cols,
                    strain_scaler, stress_scaler, material_scaler, peak_stress_scaler, peak_strain_scaler,
                    X_material_original, X_stress_original, X_peak_stress_original, X_peak_strain_original,
                    sample_divisions, cluster_labels, extra_data)

        log("Cache missing Xiao reference or residual information, recalculating...", force=True)
        cached.close()

    log("Reading data from Excel...", force=True)
    
    # 1. Load material parameter data (first worksheet)
    material_df = pd.read_excel(material_params_file, sheet_name=0)
    if verbose:
        print(f"Material parameter table column names: {list(material_df.columns)}")
    
    # 2. Assign cluster labels (refer to algorithm in cluster_label_dataset_cvfixed.py)
    cluster_labels = assign_cluster_labels(material_df, cache_dir, default_cluster_count, verbose=verbose)
    
    # 3. Load stress-strain data (second worksheet)
    stress_df = pd.read_excel(stress_data_file, sheet_name=1)
    
    # 4. Get column names
    strain_col = stress_df.columns[0]  # First column is strain (shared by all samples)
    stress_cols = stress_df.columns[1:]  # Subsequent columns are stresses of different samples

    # Build column name normalization mapping, fault-tolerant matching (remove spaces, case-insensitive, full-width spaces, remove .0)
    def _normalize_name(value):
        s = str(value)
        s = s.strip().replace(' ', '').replace('\u3000', '').lower()
        if s.endswith('.0'):
            s = s[:-2]
        return s
    
    normalized_col_to_actual = {}
    for col in stress_cols:
        key = _normalize_name(col)
        normalized_col_to_actual[key] = col
    
    # 5. Directly specify material parameter column names (15 material parameters)
    material_param_cols = [
        # 10 material features
        'water', 'cement', 'w/c', 'CS', 'sand',
        'CA', 'r', 'WA', 'S', 'CI',
        # 5 specimen parameters
        'age', 'μe', 'DJB', 'side', 'GJB'
    ]
    
    # 6. Read dataset division information
    if verbose:
        print("Reading dataset division information...")

    # Read real peak stress/strain (directly from material parameter table)
    fc_col = None
    eps_col = None
    for col in material_df.columns:
        col_str = str(col).strip()
        if col_str.lower() == 'fc' or col_str == 'fc':
            fc_col = col
        if col_str.lower() == 'peak_strain':
            eps_col = col
    
    if fc_col is not None and eps_col is not None:
        X_peak_stress_groundtruth = material_df[fc_col].values
        X_peak_strain_groundtruth = material_df[eps_col].values
        if verbose:
            print(f"Successfully read real peak stress/strain columns: '{fc_col}', '{eps_col}'")
            print(f"Successfully read real peak stress/strain: {len(X_peak_stress_groundtruth)} samples")
    else:
        X_peak_stress_groundtruth = None
        X_peak_strain_groundtruth = None
        if verbose:
            print(f"[Warning] Real peak columns not found (fc column: {fc_col}, εc column: {eps_col}); will fall back to curve-based calculation")

    # Read dataset division information
    sample_divisions = material_df['DataSlice'].values
    if verbose:
        print(f"Successfully read dataset divisions: {len(sample_divisions)} samples")
        print(f"Dataset division distribution: {np.unique(sample_divisions, return_counts=True)}")

    # 7. Data cleaning - keep original data unchanged
    cleaned_material_df = material_df.copy()
    
    # 8. Match samples - use custom specimen number as index
    matched_samples = []
    custom_id_col = 'NO'  # Custom specimen number
    
    for idx, row in cleaned_material_df.iterrows():
        sample_name_raw = row[custom_id_col]
        sample_name = str(sample_name_raw)
        material_params = row[material_param_cols].values.astype(float)

        # Normalized key
        norm_key = _normalize_name(sample_name)

        # Try to match
        actual_col = normalized_col_to_actual.get(norm_key)

        gt_peak_stress = X_peak_stress_groundtruth[idx] if X_peak_stress_groundtruth is not None else None
        gt_peak_strain = X_peak_strain_groundtruth[idx] if X_peak_strain_groundtruth is not None else None
        division = sample_divisions[idx] if sample_divisions is not None else 'unknown'

        matched_samples.append({
            'sample_name': actual_col,
            'material_params': material_params,
            'gt_peak_stress': gt_peak_stress,
            'gt_peak_strain': gt_peak_strain,
            'division': division
        })
   
    # 9. Uniform sampling to specified length
    X_input = []
    y_output = []
    xiao_curves_list = []
    residual_curves_list = []
    
    # Get complete strain data (shared by all samples)
    strain_data_full = stress_df[strain_col].values
    
    # Store real peak stress-strain data
    X_peak_stress_groundtruth_list = []
    X_peak_strain_groundtruth_list = []
    sample_divisions_list = []
    successful_sample_indices = []
    
    for sample_idx, sample_info in enumerate(matched_samples):
        # Get original data
        stress_series = stress_df[sample_info['sample_name']]
        material_params = sample_info['material_params']
        division = sample_info['division']
        
        # Get length of valid data (remove nan)
        valid_stress_data = stress_series.dropna()
        stress_data = valid_stress_data.values
        valid_length = len(stress_data)
        
        # Truncate strain data to the same length as valid stress data
        strain_data = strain_data_full[:valid_length]
        
        # Use original logic: sample half points from ascending and descending segments each
        original_length = len(strain_data)
        peak_idx = np.argmax(stress_data)
        
        rising_points = curve_length // 2
        falling_points = curve_length - rising_points
        
        # Ascending segment indices (including 0 to peak_idx)
        if peak_idx > 0:
            rising_indices = np.linspace(0, peak_idx, rising_points, dtype=int)
        else:
            rising_indices = np.array([0])
        
        # Descending segment indices (including peak_idx to end)
        falling_length = original_length - peak_idx
        if falling_length > 1:
            falling_indices = np.linspace(peak_idx, original_length - 1, falling_points, dtype=int)
        else:
            falling_indices = np.full(falling_points, peak_idx, dtype=int)
        
        # Merge indices, remove duplicates and sort
        indices = np.concatenate([rising_indices, falling_indices])
        indices = np.unique(indices)
        indices = np.sort(indices)
        
        # Ensure final length is strictly equal to target length
        if len(indices) != curve_length:
            if len(indices) > curve_length:
                indices = indices[np.linspace(0, len(indices)-1, curve_length, dtype=int)]
            else:
                last_idx = indices[-1]
                additional_indices = np.full(curve_length - len(indices), last_idx)
                indices = np.concatenate([indices, additional_indices])
        
        # Sample to get fixed-length data
        strain_sampled = strain_data[indices]
        stress_sampled = stress_data[indices]
        
        # Build input matrix
        try:
            material_params_expanded = np.tile(material_params, (curve_length, 1))
            strain_expanded = strain_sampled.reshape(-1, 1)
            input_matrix = np.concatenate([strain_expanded, material_params_expanded], axis=1)
        except Exception as e:
            if verbose:
                print(f"Error building input matrix (sample {sample_info.get('sample_name', 'unknown')}): {e}")
                print(f"Skipping this sample, continuing with next...")
            continue
        
        # Calculate Xiao reference and residual (original scale)
        sigma_cp_original = sample_info.get('gt_peak_stress')
        epsilon_cp_original = sample_info.get('gt_peak_strain')
        if sigma_cp_original is None or not np.isfinite(sigma_cp_original):
            sigma_cp_original = float(np.max(stress_sampled))
        if epsilon_cp_original is None or not np.isfinite(epsilon_cp_original) or epsilon_cp_original <= 0:
            epsilon_cp_original = float(strain_sampled[np.argmax(stress_sampled)])
            if not np.isfinite(epsilon_cp_original) or epsilon_cp_original <= 0:
                epsilon_cp_original = float(np.max(strain_sampled)) if strain_sampled.size else 1e-4
                epsilon_cp_original = epsilon_cp_original if epsilon_cp_original > 0 else 1e-4
        
        r_value = float(material_params[6]) if len(material_params) > 6 else 0.0
        xiao_curve_sample = calculate_xiao_curve_with_real_peaks(sigma_cp_original, epsilon_cp_original, r_value, strain_sampled)
        if xiao_curve_sample.shape != stress_sampled.shape:
            xiao_curve_sample = np.interp(
                np.linspace(0, len(xiao_curve_sample) - 1, len(stress_sampled)),
                np.arange(len(xiao_curve_sample)),
                xiao_curve_sample
            )
        residual_sample = stress_sampled - xiao_curve_sample

        # Append samples
        X_input.append(input_matrix)
        y_output.append(stress_sampled)
        xiao_curves_list.append(xiao_curve_sample)
        residual_curves_list.append(residual_sample)
        
        # Store real peak stress-strain and dataset division information
        X_peak_stress_groundtruth_list.append(sample_info.get('gt_peak_stress'))
        X_peak_strain_groundtruth_list.append(sample_info.get('gt_peak_strain'))
        sample_divisions_list.append(division)
        successful_sample_indices.append(sample_idx)
    
    # Convert to numpy arrays
    X_input = np.array(X_input)  # [samples, curve_length, 16]
    y_output = np.array(y_output)  # [samples, curve_length]
    X_xiao_curve_original = np.array(xiao_curves_list)  # [samples, curve_length]
    X_residual_curve_original = np.array(residual_curves_list)  # [samples, curve_length]
    
    # Separate data for subsequent processing
    X_strain = X_input[:, :, 0]  # [samples, curve_length] - strain data
    X_material = X_input[:, 0, 1:]  # [samples, material_params] - material parameters
    X_stress = y_output  # [samples, curve_length] - stress data
    
    # Use real peak stress-strain from fc/εc columns
    X_peak_stress = np.array(X_peak_stress_groundtruth_list)
    X_peak_strain = np.array(X_peak_strain_groundtruth_list)
    
    # Process dataset division labels
    sample_divisions = np.array(sample_divisions_list)  # [samples]
    
    # Data preprocessing - normalize using peak average
    # Save original data (for physical equation calculation and denormalization)
    X_strain_original = X_strain.copy()
    X_stress_original = X_stress.copy()
    X_material_original = X_material.copy()
    X_peak_stress_original = X_peak_stress.copy()
    X_peak_strain_original = X_peak_strain.copy()
    
    # Calculate peak stress-strain averages as normalization factors
    mean_peak_stress = np.mean(X_peak_stress_original)
    mean_peak_strain = np.mean(X_peak_strain_original)
    
    # Strain data: normalize by dividing by peak strain average
    X_strain_normalized = X_strain_original / mean_peak_strain
    strain_scaler = {'type': 'peak_average', 'factor': mean_peak_strain}
    
    # Stress data: normalize by dividing by peak stress average
    stress_factor = mean_peak_stress if abs(mean_peak_stress) > 1e-8 else 1.0
    X_stress_normalized = X_stress_original / stress_factor
    stress_scaler = {'type': 'peak_average', 'factor': stress_factor}
    X_xiao_curve_normalized = np.nan_to_num(X_xiao_curve_original / stress_factor, nan=0.0, posinf=0.0, neginf=0.0)
    X_residual_curve_normalized = np.nan_to_num(X_residual_curve_original / stress_factor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Material parameters: normalize to [0,1] using MinMaxScaler
    material_scaler = MinMaxScaler()
    X_material_normalized = material_scaler.fit_transform(X_material_original)
    
    # Peak stress-strain: normalize by dividing by their respective averages
    X_peak_stress_normalized = X_peak_stress_original / mean_peak_stress
    X_peak_strain_normalized = X_peak_strain_original / mean_peak_strain
    
    # Create peak normalizers (for denormalization)
    peak_stress_scaler = {'type': 'peak_average', 'factor': mean_peak_stress}
    peak_strain_scaler = {'type': 'peak_average', 'factor': mean_peak_strain}
    
    # Use normalized data as main data
    X_strain = X_strain_normalized
    X_stress = X_stress_normalized
    X_material = X_material_normalized
    X_peak_stress = X_peak_stress_normalized
    X_peak_strain = X_peak_strain_normalized
    
    log(f"\n=== Data Loading Completed ===", force=True)
    log(f"Successfully loaded samples: {len(X_input)} / {len(matched_samples)}", force=True)
    if len(successful_sample_indices) < len(matched_samples):
        failed_count = len(matched_samples) - len(successful_sample_indices)
        log(f"Warning: {failed_count} samples failed to load and were skipped", force=True)
    
    if use_cache and cache_path:
        log(f"Caching data to: {cache_path}", force=True)
        np.savez_compressed(
            cache_path,
            X_strain=X_strain,
            X_stress=X_stress,
            X_material=X_material,
            X_peak_stress=X_peak_stress,
            X_peak_strain=X_peak_strain,
            material_param_cols=np.array(material_param_cols, dtype=object),
            strain_scaler=np.array([strain_scaler], dtype=object),
            stress_scaler=np.array([stress_scaler], dtype=object),
            material_scaler=np.array([material_scaler], dtype=object),
            peak_stress_scaler=np.array([peak_stress_scaler], dtype=object),
            peak_strain_scaler=np.array([peak_strain_scaler], dtype=object),
            X_material_original=X_material_original,
            X_stress_original=X_stress_original,
            X_peak_stress_original=X_peak_stress_original,
            X_peak_strain_original=X_peak_strain_original,
            sample_divisions=sample_divisions,
            cluster_labels=cluster_labels,
            X_xiao_curve=X_xiao_curve_normalized,
            X_residual_curve=X_residual_curve_normalized,
            X_xiao_curve_original=X_xiao_curve_original,
            X_residual_curve_original=X_residual_curve_original
        )
    
    extra_data = {
        'xiao_curve_normalized': X_xiao_curve_normalized,
        'residual_curve_normalized': X_residual_curve_normalized,
        'xiao_curve_original': X_xiao_curve_original,
        'residual_curve_original': X_residual_curve_original,
    }

    return (X_strain, X_stress, X_material, X_peak_stress, X_peak_strain, material_param_cols,
            strain_scaler, stress_scaler, material_scaler, peak_stress_scaler, peak_strain_scaler,
            X_material_original, X_stress_original, X_peak_stress_original, X_peak_strain_original,
            sample_divisions, cluster_labels, extra_data)


# ========== Dataset class ==========
class ConstitutiveDataset(Dataset):
    """Constitutive relationship dataset class"""
    
    def __init__(self, strain_data, stress_data, material_params, peak_stress, peak_strain,
                 output_length=800, cluster_labels=None, num_clusters=None,
                 xiao_curve=None, residual_curve=None):
        """
        Args:
            strain_data: Normalized strain data [samples, curve_length]
            stress_data: Normalized stress data [samples, curve_length] 
            material_params: Normalized material parameters [samples, num_params]
            peak_stress: Normalized peak stress [samples]
            peak_strain: Normalized peak strain [samples]
            output_length: Output curve length
            xiao_curve: Normalized Xiao reference curve [samples, curve_length]
            residual_curve: Normalized residual curve [samples, curve_length]
        """
        self.strain_data = torch.FloatTensor(strain_data)
        self.stress_data = torch.FloatTensor(stress_data)
        self.material_params = torch.FloatTensor(material_params)
        self.peak_stress = torch.FloatTensor(peak_stress)
        self.peak_strain = torch.FloatTensor(peak_strain)
        if xiao_curve is None:
            xiao_curve = np.zeros_like(stress_data)
        if residual_curve is None:
            residual_curve = stress_data
        self.xiao_curve = torch.FloatTensor(xiao_curve)
        self.residual_curve = torch.FloatTensor(residual_curve)
        self.output_length = output_length
        
        if cluster_labels is None:
            cluster_labels = np.zeros(len(self.strain_data), dtype=int)
        cluster_labels = np.asarray(cluster_labels, dtype=int)
        inferred_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 1
        if num_clusters is None:
            self.num_clusters = max(1, inferred_clusters)
        else:
            self.num_clusters = max(1, int(num_clusters))
            if cluster_labels.size:
                self.num_clusters = max(self.num_clusters, inferred_clusters)
        if cluster_labels.size:
            cluster_labels = np.clip(cluster_labels, 0, self.num_clusters - 1)
        self.cluster_labels = torch.LongTensor(cluster_labels)
        self.cluster_one_hot = F.one_hot(self.cluster_labels, num_classes=self.num_clusters).float()
        
    def __len__(self):
        return len(self.strain_data)
    
    def __getitem__(self, idx):
        strain = self.strain_data[idx]  # [curve_length]
        stress = self.stress_data[idx]  # [curve_length]
        material_params = self.material_params[idx]  # [num_params]
        peak_stress = self.peak_stress[idx]  # scalar
        peak_strain = self.peak_strain[idx]  # scalar
        
        # Copy material parameters to each time step
        material_params_expanded = material_params.unsqueeze(0).expand(self.output_length, -1)  # [curve_length, num_params]
        cluster_one_hot = self.cluster_one_hot[idx]  # [num_clusters]
        cluster_expanded = cluster_one_hot.unsqueeze(0).expand(self.output_length, -1)  # [curve_length, num_clusters]
        
        # Copy peak stress-strain to each time step
        peak_stress_expanded = peak_stress.unsqueeze(0).expand(self.output_length)  # [curve_length]
        peak_strain_expanded = peak_strain.unsqueeze(0).expand(self.output_length)  # [curve_length]
        
        return {
            'strain': strain,
            'stress': stress,
            'residual_target': self.residual_curve[idx],
            'xiao_curve': self.xiao_curve[idx],
            'material_params': material_params_expanded,
            'peak_stress': peak_stress_expanded,
            'peak_strain': peak_strain_expanded,
            'cluster_features': cluster_expanded,
            'cluster_id': self.cluster_labels[idx],
            'cluster_one_hot': cluster_one_hot
        }


def create_dataloader(dataset, batch_size, shuffle, num_workers=None, pin_memory=None, prefetch_factor=None):
    """
    Create a data loader with pin_memory and multi-process prefetching.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (None to read from environment variable)
        pin_memory: Whether to use pin_memory (None to auto-detect)
        prefetch_factor: Prefetch factor (None to read from environment variable)
    """
    if num_workers is None:
        _default_workers = 4
        num_workers = int(os.getenv('BILSTM_NUM_WORKERS', str(_default_workers)))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if prefetch_factor is None:
        prefetch_factor = int(os.getenv('BILSTM_PREFETCH_FACTOR', '4'))
    
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def prepare_cluster_features(batch, num_clusters, device):
    """
    Get cluster one-hot features expanded to time steps.
    
    Args:
        batch: Data batch
        num_clusters: Number of clusters
        device: Device
    
    Returns:
        cluster_features: Cluster features [batch_size, curve_length, num_clusters]
    """
    # Prefer using cluster_features if available and valid
    cluster_features = batch.get('cluster_features')
    if cluster_features is not None and cluster_features.numel() > 0:
        return cluster_features.to(device)
    
    # If cluster_features doesn't exist, try to build from cluster_id
    cluster_id = batch.get('cluster_id')
    if cluster_id is not None:
        cluster_ids = cluster_id.to(device)
        cluster_one_hot = F.one_hot(cluster_ids, num_classes=num_clusters).float()
        curve_length = batch['strain'].shape[1]
        return cluster_one_hot.unsqueeze(1).expand(-1, curve_length, -1)
    
    # If neither exists, create all-zero cluster features (degraded processing)
    batch_size = batch['strain'].shape[0]
    curve_length = batch['strain'].shape[1]
    return torch.zeros(batch_size, curve_length, num_clusters, device=device, dtype=torch.float32)
