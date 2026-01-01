"""
Multi-Objective Optimization - NSGA-III Algorithm
For finding RAC (Recycled Aggregate Concrete) mix proportions that balance strength, ductility, cost, and carbon emissions

Algorithm Description:
- Uses the NSGA-III algorithm (more suitable for 4-objective optimization, providing better convergence and Pareto front distribution)
- Model predictions in the evaluation function are accelerated via GPU batch processing (Pi-BiLSTM batch prediction)
- Population size is automatically adjusted to a multiple of the number of reference directions

Optimization Objectives:
1. Peak stress (f_c) - Maximization
2. Normalized Ductility Area - Maximization
   - Methodology: Normalize strain/peak strain and stress/peak stress
   - Take the residual 20% of the post-peak stress decline phase as the critical point
   - Calculate the area under the normalized curve from the initial point to the critical point
   - Larger area indicates higher ductility
3. Material cost (Cost) - Minimization
4. Carbon emissions (CO₂) - Minimization

Decision Variables: RAC material parameters (water content, cement dosage, water-cement ratio (w/c), cement strength, natural sand content, coarse aggregate dosage,
                    mass replacement rate, mixed aggregate water absorption, maximum particle size of coarse aggregate, crushing index of mixed aggregate, etc.)

Constraints:
- The upper and lower bounds of decision variables are automatically read from the dataset (using the minimum and maximum values of the dataset)
- If reading from the dataset fails, default values are used as backups
- Default constraint ranges (only used when dataset reading is not possible):
  - 0% ≤ r ≤ 100% (mass replacement rate)
  - 0.38 ≤ w/c ≤ 0.71 (water-cement ratio)
  - 20 ≤ S ≤ 31.5 mm (maximum particle size of coarse aggregate)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set font display and LaTeX rendering (supporting Chinese characters if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False  # Do not use LaTeX, use matplotlib's mathematical mode
plt.rcParams['mathtext.default'] = 'regular'  # Use regular mathematical text mode

# Check and import pymoo library (NSGA-III implementation)
PYMOO_AVAILABLE = False
try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    # Try new version import method
    try:
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError:
        # Old version import method
        from pymoo.factory import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    print("Error: pymoo library not installed or NSGA-III not available. Please run: pip install pymoo")
    print("NSGA-III is more suitable for 4-objective optimization, providing better convergence and Pareto front distribution")
    # Create a placeholder Problem class to avoid NameError
    class Problem:
        def __init__(self, *args, **kwargs):
            raise ImportError("pymoo library not installed, cannot use multi-objective optimization functionality. Please run: pip install pymoo")

# Set project path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add LSTM directory to path
LSTM_DIR = PROJECT_ROOT / "LSTM"
# Import necessary classes and functions
from dataset.dataloader import load_excel_data as load_dataset_with_clusters

# Import LSTM related modules (need to specify full path or use importlib)
# First add LSTM directory to sys.path, ensure modules can be imported from each other
if str(LSTM_DIR) not in sys.path:
    sys.path.insert(0, str(LSTM_DIR))

import importlib.util

# Load Bidirectional_LSTM_Enhanced cross-validation module
bilstm_module_path = LSTM_DIR / "Bidirectional_LSTM_Enhanced交叉验证.py"
if bilstm_module_path.exists():
    spec = importlib.util.spec_from_file_location("Bidirectional_LSTM_Enhanced交叉验证", bilstm_module_path)
    bilstm_module = importlib.util.module_from_spec(spec)
    # Before loading module, ensure sys.path contains LSTM directory
    spec.loader.exec_module(bilstm_module)
    # Add module to sys.modules so it can be imported by other modules
    sys.modules["Bidirectional_LSTM_Enhanced交叉验证"] = bilstm_module
    BidirectionalLSTMRegressor = bilstm_module.BidirectionalLSTMRegressor
else:
    raise FileNotFoundError(f"File not found: {bilstm_module_path}")

# Import trained model module
trained_model_path = LSTM_DIR / "训练好的模型_交叉验证.py"
if not trained_model_path.exists():
    raise FileNotFoundError(f"File not found: {trained_model_path}")

try:
    # Use importlib to load module, ensure dependencies can be found
    spec = importlib.util.spec_from_file_location("训练好的模型_交叉验证", trained_model_path)
    trained_model_module = importlib.util.module_from_spec(spec)
    # Before loading, ensure Bidirectional_LSTM_Enhanced cross-validation is in sys.modules
    spec.loader.exec_module(trained_model_module)
    # Add module to sys.modules
    sys.modules["训练好的模型_交叉验证"] = trained_model_module
    
    load_trained_model = trained_model_module.load_trained_model
    compute_curve_energy = trained_model_module.compute_curve_energy
    compute_energy_metrics = trained_model_module.compute_energy_metrics
    calculate_xiao_curve_with_real_peaks = trained_model_module.calculate_xiao_curve_with_real_peaks
    calculate_yan_curve_with_real_peaks = trained_model_module.calculate_yan_curve_with_real_peaks
    predict_test_set = trained_model_module.predict_test_set
except Exception as e:
    # If importlib fails, try direct import (need to ensure in LSTM directory)
    print(f"Warning: importlib loading module failed: {e}")
    print(f"Try using direct import method...")
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(str(LSTM_DIR))
        # Ensure sys.path contains LSTM directory
        if str(LSTM_DIR) not in sys.path:
            sys.path.insert(0, str(LSTM_DIR))
        import importlib
        trained_model_module = importlib.import_module("Trained model_cross-validation")
        load_trained_model = trained_model_module.load_trained_model
        compute_curve_energy = trained_model_module.compute_curve_energy
        compute_energy_metrics = trained_model_module.compute_energy_metrics
        calculate_xiao_curve_with_real_peaks = trained_model_module.calculate_xiao_curve_with_real_peaks
        calculate_yan_curve_with_real_peaks = trained_model_module.calculate_yan_curve_with_real_peaks
        predict_test_set = trained_model_module.predict_test_set
    finally:
        os.chdir(original_cwd)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== Ductility index calculation function ==========
def calculate_normalized_ductility_area(strain_curve, stress_curve, peak_strain, peak_stress, 
                                        residual_ratio=0.2):
    """
    Calculate normalized curve area as ductility index
    
    Method:
    1. Divide strain by peak strain, stress by peak stress to normalize
    2. Find residual 20% of the post-peak stress decline phase as the critical point
    3. Calculate the area under the normalized curve from the initial point to the critical point
    4. Larger area indicates higher ductility
    
    Args:
        strain_curve: Strain curve array [curve_length]
        stress_curve: Stress curve array [curve_length]
        peak_strain: Peak strain
        peak_stress: Peak stress (MPa)
        residual_ratio: Residual strength ratio (default 0.2, i.e. 20%)
    
    Returns:
        ductility_area: Normalized curve area (dimensionless), larger area indicates higher ductility
        critical_idx: Critical point index
    """
    # Ensure input is numpy array
    strain_curve = np.asarray(strain_curve, dtype=float)
    stress_curve = np.asarray(stress_curve, dtype=float)
    
    # Check if peak is valid
    if peak_strain <= 0 or peak_stress <= 0:
        return 0.0, 0
    
    # 1. Normalize: strain/peak strain, stress/peak stress
    normalized_strain = strain_curve / peak_strain
    normalized_stress = stress_curve / peak_stress
    
    # 2. Find peak position
    peak_idx = np.argmax(stress_curve)
    
    # 3. Find critical point to residual 20% strength
    # Residual strength = peak stress × residual ratio (e.g. 20% of peak stress)
    residual_stress = peak_stress * residual_ratio
    critical_idx = peak_idx
    
    # Find critical point in descending segment (after peak)
    if peak_idx < len(stress_curve) - 1:
        # Find first point where stress falls below residual strength
        # Note: descending segment refers to points after peak
        descending_stress = stress_curve[peak_idx + 1:]  # Start from first point after peak
        descending_indices = np.where(descending_stress <= residual_stress)[0]
        
        if len(descending_indices) > 0:
            # Find first point that satisfies condition (index relative to peak_idx+1)
            critical_idx = peak_idx + 1 + descending_indices[0]
        else:
            # If no point found in descending segment, curve does not fall below residual strength
            # In this case, use last point as critical point
            critical_idx = len(stress_curve) - 1
    else:
        # If peak is last, cannot calculate descending segment, use last point as critical point
        critical_idx = len(stress_curve) - 1
    
    # Ensure critical point index is valid
    # Critical point should be at peak position or after (because we calculate area from initial point to critical point, including peak)
    critical_idx = min(critical_idx, len(strain_curve) - 1)
    critical_idx = max(critical_idx, peak_idx)  # Critical point should be at peak position or after
    
    # 4. Calculate area under normalized curve from initial point to critical point
    # Use trapezoidal integration method
    if critical_idx > 0:
        # Extract normalized curve segment from 0 to critical point
        norm_strain_segment = normalized_strain[:critical_idx + 1]
        norm_stress_segment = normalized_stress[:critical_idx + 1]
        
        # Ensure strain is sorted in ascending order (prevent small disorder in curve)
        order = np.argsort(norm_strain_segment)
        norm_strain_sorted = norm_strain_segment[order]
        norm_stress_sorted = norm_stress_segment[order]
        
        # Calculate area under normalized curve (trapezoidal integration)
        ductility_area = float(np.trapz(norm_stress_sorted, norm_strain_sorted))
    else:
        ductility_area = 0.0
    
    return ductility_area, critical_idx


# ========== Cost and carbon emissions calculation function ==========
# 
# About parameters in table:
# 1. Price (price): Material unit price, unit $/kg
# 2. Density (density): Material density, unit kg/m³
# 3. Energy intensity (energy intensity): Energy consumption per unit mass of material, unit GJ/ton (GJ/ton)
#    - This is an important indicator in life cycle assessment (LCA), used to evaluate energy consumption during material production
#    - 1 GJ = 10^9 Joules = approximately 277.78 kWh (kilowatt hours)
#    - This indicator is not used in the current code, but can be used to calculate total energy consumption during material production
# 4. Carbon emission factor (carbon emission factor): Carbon emissions per unit mass of material, unit kgCO₂e/ton
#
# Note: Fly ash in the table is not used in the current model, therefore not included in the calculation
#
def calculate_material_cost(material_params_dict):
    """
    Calculate material cost (unit: $/m³)
    
    Data source: Table 4: Characteristic parameters for RAC mixture ingredients
    
    Material price (based on table data, USD price):
    - Water: 0.000691 $/kg
    - Cement: Depending on strength grade
      * 32.5 MPa: 0.0531 $/kg
      * 42.5 MPa: 0.0631 $/kg
      * 52.5 MPa: 0.0689 $/kg
    - Natural sand: 0.0287 $/kg
    - Natural coarse aggregate (NCA): 0.0287 $/kg
    - Recycled coarse aggregate (RCA): 0.0215 $/kg
    
    Note: Fly ash is not used in the current model, therefore not included in the calculation
    
    Args:
        material_params_dict: Material parameters dictionary, containing:
            - water: Water content (kg/m³)
            - cement: Cement quantity (kg/m³)
            - cement_strength: Cement strength (MPa) - used to select corresponding price
            - sand: Natural sand (kg/m³)
            - coarse_aggregate: Coarse aggregate quantity (kg/m³)
            - replacement_ratio: Replacement ratio (%)
    
    Returns:
        cost: Material cost ($/m³)
    """
    water = material_params_dict.get('water', 0)
    cement = material_params_dict.get('cement', 0)
    cement_strength = material_params_dict.get('cement_strength', 42.5)  # Default 42.5 MPa
    sand = material_params_dict.get('sand', 0)
    coarse_aggregate = material_params_dict.get('coarse_aggregate', 0)
    r = material_params_dict.get('replacement_ratio', 0) / 100.0  # Convert to decimal
    
    # Cost price ($/kg, from table)
    cost_water = 0.000691  # Water: 0.000691 $/kg
    
    # Select price based on cement strength (from table)
    if cement_strength <= 35:
        cost_cement = 0.0531  # 32.5 MPa: 0.0531 $/kg
    elif cement_strength <= 47:
        cost_cement = 0.0631  # 42.5 MPa: 0.0631 $/kg
    else:
        cost_cement = 0.0689  # 52.5 MPa: 0.0689 $/kg
    
    cost_sand = 0.0287  # Natural sand: 0.0287 $/kg
    cost_natural_coarse = 0.0287  # Natural coarse aggregate: 0.0287 $/kg
    cost_recycled_coarse = 0.0215  # Recycled coarse aggregate: 0.0215 $/kg
    
    # Calculate quantity of recycled coarse aggregate and natural coarse aggregate
    # Logic: Total coarse aggregate = RCA quantity + NCA quantity
    # RCA quantity = Total coarse aggregate × Replacement ratio
    recycled_coarse = coarse_aggregate * r
    # NCA quantity = Total coarse aggregate - RCA quantity = Total coarse aggregate × (1 - Replacement ratio)
    natural_coarse = coarse_aggregate - recycled_coarse  # Equivalent to coarse_aggregate * (1 - r)
    
    # Total cost
    cost = (water * cost_water + 
            cement * cost_cement + 
            sand * cost_sand + 
            natural_coarse * cost_natural_coarse + 
            recycled_coarse * cost_recycled_coarse)
    
    return cost


def calculate_carbon_emissions(material_params_dict):
    """
    Calculate carbon emissions (unit: kg CO₂/m³)
    
    Carbon emission factor data source: Table 4: Characteristic parameters for RAC mixture ingredients
    Unit conversion: kgCO₂e/ton → kg CO₂/kg (divide by 1000)
    
    Material carbon emission factor (based on table data):
    1. Cement: Depending on strength grade
       * 32.5 MPa: 631 kgCO₂e/ton = 0.631 kg CO₂/kg
       * 42.5 MPa: 795 kgCO₂e/ton = 0.795 kg CO₂/kg
       * 52.5 MPa: 889 kgCO₂e/ton = 0.889 kg CO₂/kg    
    2. Natural sand: 2.5 kgCO₂e/ton = 0.0025 kg CO₂/kg
       - Includes carbon emissions from extraction, transportation process
    3. Natural coarse aggregate (NCA): 2.2 kgCO₂e/ton = 0.0022 kg CO₂/kg
       - Includes carbon emissions from extraction, crushing, transportation process
    4. Recycled coarse aggregate (RCA): 1.5 kgCO₂e/ton = 0.0015 kg CO₂/kg
       - Lower than natural aggregate by approximately 32%, mainly from crushing, screening process, avoiding extraction process
    5. Water: 0.2 kgCO₂e/ton = 0.0002 kg CO₂/kg
       - Carbon emissions from water treatment process (usually negligible)
    
    Note: Fly ash is not used in the current model, therefore not included in the calculation
    
    Args:
        material_params_dict: Material parameters dictionary, containing:
            - water: Water content (kg/m³)
            - cement: Cement quantity (kg/m³)
            - cement_strength: Cement strength (MPa) - used to select corresponding carbon emission factor
            - sand: Natural sand (kg/m³)
            - coarse_aggregate: Coarse aggregate quantity (kg/m³)
            - replacement_ratio: Replacement ratio (%)
    
    Returns:
        co2: Carbon emissions (kg CO₂/m³)
    """
    water = material_params_dict.get('water', 0)
    cement = material_params_dict.get('cement', 0)
    cement_strength = material_params_dict.get('cement_strength', 42.5)  # Default 42.5 MPa
    sand = material_params_dict.get('sand', 0)
    coarse_aggregate = material_params_dict.get('coarse_aggregate', 0)
    r = material_params_dict.get('replacement_ratio', 0) / 100.0
    
    # Carbon emission factor (kgCO₂e/ton, from table) - convert to kg CO₂/kg (divide by 1000)
    co2_water = 0.2 / 1000.0  # Water: 0.2 kgCO₂e/ton
    
    # Select carbon emission factor based on cement strength (from table)
    if cement_strength <= 35:
        co2_cement = 631.0 / 1000.0  # 32.5 MPa: 631 kgCO₂e/ton
    elif cement_strength <= 47:
        co2_cement = 795.0 / 1000.0  # 42.5 MPa: 795 kgCO₂e/ton
    else:
        co2_cement = 889.0 / 1000.0  # 52.5 MPa: 889 kgCO₂e/ton
    
    co2_sand = 2.5 / 1000.0  # Sand: 2.5 kgCO₂e/ton
    co2_natural_coarse = 2.2 / 1000.0  # Natural coarse aggregate: 2.2 kgCO₂e/ton
    co2_recycled_coarse = 1.5 / 1000.0  # Recycled coarse aggregate: 1.5 kgCO₂e/ton
    
    # Calculate quantity of recycled coarse aggregate and natural coarse aggregate
    # Logic: Total coarse aggregate = RCA quantity + NCA quantity
    # RCA quantity = Total coarse aggregate × Replacement ratio
    recycled_coarse = coarse_aggregate * r
    # NCA quantity = Total coarse aggregate - RCA quantity = Total coarse aggregate × (1 - Replacement ratio)
    natural_coarse = coarse_aggregate - recycled_coarse  # Equivalent to coarse_aggregate * (1 - r)
    
    # Total carbon emissions
    co2 = (water * co2_water + 
           cement * co2_cement + 
           sand * co2_sand + 
           natural_coarse * co2_natural_coarse + 
           recycled_coarse * co2_recycled_coarse)
    
    return co2


# ========== Multi-objective optimization problem definition ==========
if PYMOO_AVAILABLE:
    class RACMixOptimizationProblem(Problem):
        """
        Multi-objective optimization problem for RAC mix design
        
        Optimization objectives:
        1. Maximize peak stress (f_c)
        2. Maximize normalized ductility area (Normalized Ductility Area)
           - Normalized curve area: strain/peak strain, stress/peak stress
           - Critical point: residual 20% of peak stress
           - Larger area, higher ductility
        3. Minimize material cost (Cost)
        4. Minimize carbon emissions (CO₂)
        
        Decision variables (15 material parameters):
        1. Water content (kg/m³)
        2. Cement quantity (kg/m³)
        3. Water-cement ratio
        4. Cement strength (MPa)
        5. Natural sand (kg/m³)
        6. Coarse aggregate quantity (kg/m³)
        7. Replacement ratio (%)
        8. Water absorption of mixed aggregate (%)
        9. Maximum particle size of coarse aggregate (mm)
        10. Crushing index of mixed aggregate
        11. Curing age (days)
        12. Loading rate (μe)
        13. Chamfer ratio (prism=0, cylinder=1)
        14. Side length or diameter (mm)
        15. Height-diameter ratio
        """
        
        def __init__(self, xgb_model, catboost_model, bilstm_model, model_info, 
                     material_scaler, peak_stress_scaler, peak_strain_scaler,
                     strain_scaler, stress_scaler, curve_length=1000,
                     material_param_names=None, combine_cost_co2=False,
                     cost_weight=0.5, co2_weight=0.5, co2_price_per_kg=0.1,
                     variable_bounds=None):
            """
            Initialize optimization problem
            
            Args:
                xgb_model: XGBoost model (for predicting peak stress)
                catboost_model: CatBoost model (for predicting peak strain)
                bilstm_model: Pi-BiLSTM model (for predicting full curve)
                model_info: Model information dictionary
                material_scaler: Material parameter scaler
                peak_stress_scaler: Peak stress scaler
                peak_strain_scaler: Peak strain scaler
                strain_scaler: Strain scaler
                stress_scaler: Stress scaler
                curve_length: Curve length
                material_param_names: Material parameter names list
                combine_cost_co2: Whether to combine cost and carbon emissions into a single objective (default False, i.e. 4 independent objectives)
                cost_weight: Cost weight (when combine_cost_co2=True)
                co2_weight: Carbon emission weight (when combine_cost_co2=True)
                co2_price_per_kg: Carbon emission price ($/kg CO₂), used to convert carbon emission to cost equivalent
                variable_bounds: Decision variable bounds dictionary, format as {'xl': array, 'xu': array}, if None then use default value
            """
            self.xgb_model = xgb_model
            self.catboost_model = catboost_model
            self.bilstm_model = bilstm_model
            self.model_info = model_info
            self.material_scaler = material_scaler
            self.peak_stress_scaler = peak_stress_scaler
            self.peak_strain_scaler = peak_strain_scaler
            self.strain_scaler = strain_scaler
            self.stress_scaler = stress_scaler
            self.curve_length = curve_length
            self.material_param_names = material_param_names or [
                'water', 'cement', 'w_c_ratio', 'cement_strength', 'sand',
                'coarse_aggregate', 'replacement_ratio', 'water_absorption',
                'max_particle_size', 'crushing_index', 'curing_age', 'loading_rate',
                'chamfer_ratio', 'side_length', 'height_diameter_ratio'
            ]
            self.combine_cost_co2 = combine_cost_co2
            self.cost_weight = cost_weight
            self.co2_weight = co2_weight
            self.co2_price_per_kg = co2_price_per_kg
            
            # Define decision variable bounds
            n_var = 15  # 15 material parameters
            
            # If variable bounds are provided, use provided bounds; otherwise use default value
            if variable_bounds is not None and 'xl' in variable_bounds and 'xu' in variable_bounds:
                xl = np.asarray(variable_bounds['xl'], dtype=float)
                xu = np.asarray(variable_bounds['xu'], dtype=float)
                if len(xl) != n_var or len(xu) != n_var:
                    raise ValueError(f"Variable bounds length mismatch: Expected {n_var} but xl has {len(xl)} and xu has {len(xu)}")
                print(f"   Using provided variable bounds (automatically read from dataset)")
            else:
                # Default variable bounds (reserved as backup)
                print(f"   Warning: No variable bounds provided, using default value (recommended to read from dataset)")
                xl = np.array([
                    150,    # Water content (kg/m³)
                    311,   # Cement quantity (kg/m³)
                    0.38,  # Water-cement ratio
                    32.5,    # Cement strength (MPa)
                    492,   # Natural sand (kg/m³)
                    1030,   # Coarse aggregate quantity (kg/m³)
                    0,    # Replacement ratio (%)
                    0.5,   # Water absorption of mixed aggregate (%)
                    20,     # Maximum particle size of coarse aggregate (mm)
                    3.10,     # Crushing index of mixed aggregate
                    7,     # Curing age (days)
                    5.56,   # Loading rate (μe)
                    0,     # Chamfer ratio (prism=0, cylinder=1)
                    100,    # Side length or diameter (mm)
                    2.0    # Height-diameter ratio
                ])
                
                xu = np.array([
                    221.04,   # Water content (kg/m³)
                    539,   # Cement quantity (kg/m³)
                    0.71,  # Water-cement ratio
                    45,    # Cement strength (MPa)
                    829.03,   # Natural sand (kg/m³)
                    1246,  # Coarse aggregate quantity (kg/m³)
                    100,    # Replacement ratio (%)
                    9.25,   # Water absorption of mixed aggregate (%)
                    31.5,    # Maximum particle size of coarse aggregate (mm)
                    24.74,    # Crushing index of mixed aggregate
                    120,    # Curing age (days)
                    100,    # Loading rate (μe)
                    1,     # Chamfer ratio (prism=0, cylinder=1)
                    150,   # Side length or diameter (mm)
                    3.0    # Height-diameter ratio
                ])
            
            # Set number of objectives based on whether to combine cost and carbon emissions
            if combine_cost_co2:
                # 3 objectives: maximize f_c and η, minimize combined cost (cost + carbon emissions)
                n_obj = 3
            else:
                # 4 objectives: maximize f_c and η, minimize Cost and CO₂ (default)
                n_obj = 4
            
            # In pymoo, all objectives should be minimized, so conversion is needed
            super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu)
            
            # Generate strain sequence (normalized)
            self.strain_sequence = np.linspace(0, 2.0, curve_length)  # Normalized strain sequence
            
            # Record all evaluated points (for plotting and saving)
            self.all_evaluated_X = []  # All evaluated decision variables
            self.all_evaluated_F = []   # All evaluated objective values
            
            # Detect XGBoost model output type (normalized or original value)
            # If normalization factor is large (>100), it is likely that XGBoost output is already original value
            self.xgb_output_is_original = False
            if peak_stress_scaler and peak_stress_scaler.get('type') == 'peak_average':
                if peak_stress_scaler['factor'] > 100:
                    self.xgb_output_is_original = True
                    print(f"   Detection: Normalization factor is large ({peak_stress_scaler['factor']:.2f}),")
                    print(f"        It is likely that XGBoost model output is already original value (MPa), will automatically detect and correct")
        
        def _evaluate(self, X, out, *args, **kwargs):
            """
            Evaluate objective functions (support batch GPU acceleration)
            
            Prediction process (cascade agent model strategy):
            ============================================
            Step 1: 15 decision variables → XGBoost → Peak stress
            --------------------------------------------
            Input: 15 material parameters (original value, not normalized)
              - 10 material parameters: water, cement, w/c, CS, sand, CA, r, WA, S, CI
              - 5 test parameters: age, μe, DJB, side, GJB
            Output: Peak stress (fc, MPa)
            
            Step 2: 17 features → CatBoost → Peak strain
            --------------------------------------------
            Input: 15 material parameters (original value) + Peak stress (original value) + Xiao formula value (original value)
              - 15 material parameters (original value, not normalized)
              - Peak stress (from step 1, original value)
              - Xiao formula calculated peak strain baseline value (original value)
            Output: Peak strain (ε_cp, original value)
            
            Step 3: 18 features → Pi-BiLSTM → Full stress-strain curve
            --------------------------------------------
            Input (each time step): 
              - Strain value (normalized, sequence feature)
              - 15 material parameters (normalized, copied to each time step)
              - Peak stress (normalized, copied to each time step)
              - Peak strain (normalized, copied to each time step)
            Output: Full stress curve (normalized, length=curve_length)
            
            Step 4: Calculate objective function values
            --------------------------------------------
            - Peak stress (from step 1, maximize)
            - Normalized ductility area (from curve in step 3, maximize)
            - Material cost (from 15 decision variables, minimize)
            - Carbon emissions (from 15 decision variables, minimize)
            
            Args:
                X: Decision variable matrix [n_population, n_var], n_var=15
                out: Output dictionary, containing objective function values
            
            Returns:
                out['F']: Objective function value matrix [n_population, n_obj]
            """
            n_pop = X.shape[0]
            n_obj = self.n_obj
            objectives = np.zeros((n_pop, n_obj))
            
            # Initialize variables
            xgb_output_is_original = False
            peak_stresses_original_from_xgb = None
            
            # Batch predict peak (using GPU acceleration)
            try:
                # Batch normalize material parameters (for CatBoost and Pi-BiLSTM, XGBoost uses original value)
                material_params_normalized = self.material_scaler.transform(X)
                
                # Batch predict peak stress (XGBoost) - using original value, not normalized
                peak_stresses_normalized = None
                if self.xgb_model is not None:
                    try:
                        # XGBoost batch prediction - directly using original value X (not normalized)
                        xgb_outputs = self.xgb_model.predict(X)
                        
                        # Ensure output is float array, keep precision
                        xgb_outputs = np.asarray(xgb_outputs, dtype=np.float64).flatten()
                        
                        # XGBoost uses original value input, output is also original value (MPa)
                        # Directly use output as peak stress
                        peak_stresses_original_from_xgb = xgb_outputs.copy()
                        
                        # Calculate normalized value for subsequent calculation (CatBoost and Pi-BiLSTM need normalized value)
                        if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                            peak_stresses_normalized = peak_stresses_original_from_xgb / self.peak_stress_scaler['factor']
                        else:
                            peak_stresses_normalized = peak_stresses_original_from_xgb.copy()
                        
                        # Mark XGBoost output as original value
                        xgb_output_is_original = True
                    except Exception as e:
                        print(f"Warning: XGBoost batch prediction failed, revert to single prediction: {e}")
                        peak_stresses_normalized = None
                
                # Batch predict peak strain (CatBoost)
                peak_strains_normalized = None
                if self.catboost_model is not None and peak_stresses_normalized is not None:
                    try:
                        # Calculate Xiao formula value (batch)
                        # If XGBoost output is already original value, use directly; otherwise need to denormalize
                        if xgb_output_is_original:
                            peak_stresses_original = peak_stresses_original_from_xgb
                        else:
                            peak_stresses_original = peak_stresses_normalized.copy()
                            if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                                peak_stresses_original = peak_stresses_normalized * self.peak_stress_scaler['factor']
                        
                        replacement_ratios = X[:, 6] / 100.0  # r is the 7th parameter (index 6)
                        fc_clamped = np.maximum(peak_stresses_original, 1e-6)
                        r_clamped = np.maximum(replacement_ratios, 1e-8)
                        
                        inner = (0.626 * fc_clamped - 4.33) * 1e-7
                        inner_clamped = np.maximum(inner, 0.0)
                        term1 = 0.00076 + np.sqrt(inner_clamped)
                        denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
                        term2 = 1.0 + (r_clamped / denom)
                        xiao_formula_values = term1 * term2
                        
                        # Normalize xiao_formula
                        if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                            xiao_formula_normalized = xiao_formula_values / self.peak_strain_scaler['factor']
                        else:
                            xiao_formula_normalized = xiao_formula_values
                        
                        # Build CatBoost batch input - using original value (not normalized)
                        # CatBoost training uses original value, prediction should also use original value
                        # Need to denormalize peak stress and Xiao formula value
                        peak_stresses_original = peak_stresses_original_from_xgb
                        xiao_formula_original = xiao_formula_values  # Already original value
                        
                        catboost_input = np.column_stack([
                            X,  # 15 material parameters (original value)
                            peak_stresses_original.reshape(-1, 1),  # Peak stress (original value)
                            xiao_formula_original.reshape(-1, 1)  # Xiao formula value (original value)
                        ])
                        
                        # CatBoost batch prediction - using original value input
                        catboost_outputs = self.catboost_model.predict(catboost_input)
                        catboost_outputs = np.asarray(catboost_outputs, dtype=np.float64).flatten()
                        
                        # Detect CatBoost output type: Peak strain is usually in the range of 0.001-0.003 (0.1%-0.3%)
                        # If output value is in this range, it means output is already original value (according to paper, CatBoost outputs original value)
                        n_samples = len(catboost_outputs)
                        n_in_range = np.sum((catboost_outputs > 0.0005) & (catboost_outputs < 0.005))
                        ratio_in_range = n_in_range / n_samples if n_samples > 0 else 0
                        
                        if ratio_in_range > 0.8:  # More than 80% of values are in reasonable range, considered as original value
                            # CatBoost output is already original value, use directly
                            peak_strains = catboost_outputs.copy()
                            # Calculate normalized value for subsequent calculation
                            if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                                peak_strains_normalized = peak_strains / self.peak_strain_scaler['factor']
                            else:
                                peak_strains_normalized = peak_strains.copy()
                        else:
                            # CatBoost output may be normalized value, need to denormalize
                            peak_strains_normalized = catboost_outputs.copy()
                            if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                                peak_strains = peak_strains_normalized * self.peak_strain_scaler['factor']
                            else:
                                peak_strains = peak_strains_normalized.copy()
                            
                            # Reasonability check: Peak strain should be in the range of 0.001-0.003
                            if np.any(peak_strains < 0.0001) or np.any(peak_strains > 0.01):
                                # If denormalized value is abnormal, limit to reasonable range
                                peak_strains = np.clip(peak_strains, 0.001, 0.003)
                                if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                                    peak_strains_normalized = peak_strains / self.peak_strain_scaler['factor']
                    except Exception as e:
                        print(f"Warning: CatBoost batch prediction failed, revert to single prediction: {e}")
                        peak_strains_normalized = None
                        peak_strains = None
                
                # Batch predict full curve (using GPU batch processing, batch to avoid memory overflow)
                stress_curves = None
                if peak_stresses_normalized is not None and peak_strains_normalized is not None:
                    try:
                        # Denormalize peak (for full curve prediction)
                        # If XGBoost output is already original value, peak_stresses_normalized is already normalized value, need to denormalize
                        # If XGBoost output is normalized value, peak_stresses_normalized is also normalized value, need to denormalize
                        if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                            peak_stresses_for_curve = peak_stresses_normalized * self.peak_stress_scaler['factor']
                        else:
                            peak_stresses_for_curve = peak_stresses_normalized
                        
                        # If peak_strains hasn't been calculated (possible already calculated in batch prediction)
                        if 'peak_strains' not in locals() or peak_strains is None:
                            if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                                peak_strains = peak_strains_normalized * self.peak_strain_scaler['factor']
                            else:
                                peak_strains = peak_strains_normalized.copy()
                        
                        # Reasonability check: Peak strain should be in the range of 0.001-0.003
                        if np.any(peak_strains < 0.0001) or np.any(peak_strains > 0.01):
                            # Limit to reasonable range
                            peak_strains = np.clip(peak_strains, 0.001, 0.003)
                            if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                                peak_strains_normalized = peak_strains / self.peak_strain_scaler['factor']
                        
                        # Batch processing to avoid GPU memory overflow (adjust batch size dynamically based on GPU memory)
                        batch_size = min(64, n_pop)  # Default batch size is 64; if the population size is smaller, the population size will be used instead
                        # If GPU memory is less than 16GB, use smaller batch
                        if torch.cuda.is_available():
                            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                            if gpu_memory_gb < 8:
                                batch_size = min(32, n_pop)
                            elif gpu_memory_gb < 16:
                                batch_size = min(48, n_pop)
                        
                        # If population is large, output batch information
                        n_batches = (n_pop + batch_size - 1) // batch_size
                        if n_batches > 1:
                            print(f"  Using batch processing (batch size: {batch_size}, total batches: {n_batches}) to avoid GPU memory overflow")
                        
                        strain_seq_normalized = self.strain_sequence.copy()
                        stress_curves_list = []
                        
                        # Batch processing
                        for batch_idx, batch_start in enumerate(range(0, n_pop, batch_size)):
                            batch_end = min(batch_start + batch_size, n_pop)
                            batch_indices = slice(batch_start, batch_end)
                            batch_n = batch_end - batch_start
                            
                            # Prepare input for current batch
                            material_params_batch = material_params_normalized[batch_indices]
                            peak_stresses_batch = peak_stresses_normalized[batch_indices]
                            peak_strains_batch = peak_strains_normalized[batch_indices]
                            
                            material_params_expanded = np.tile(material_params_batch[:, np.newaxis, :], 
                                                              (1, self.curve_length, 1))
                            peak_stress_expanded = np.tile(peak_stresses_batch[:, np.newaxis], 
                                                           (1, self.curve_length))
                            peak_strain_expanded = np.tile(peak_strains_batch[:, np.newaxis], 
                                                          (1, self.curve_length))
                            
                            # Build batch input tensor [batch_n, curve_length, 18]
                            x_batch = np.concatenate([
                                np.tile(strain_seq_normalized[np.newaxis, :, np.newaxis], (batch_n, 1, 1)),
                                material_params_expanded,
                                peak_stress_expanded[:, :, np.newaxis],
                                peak_strain_expanded[:, :, np.newaxis]
                            ], axis=2)
                            
                            x_tensor_batch = torch.FloatTensor(x_batch).to(device)
                            
                            # Batch prediction (GPU acceleration)
                            self.bilstm_model.eval()
                            with torch.no_grad():
                                stress_curves_batch_normalized = self.bilstm_model(x_tensor_batch).cpu().numpy()
                            
                            # Immediately release GPU memory
                            del x_tensor_batch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            stress_curves_list.append(stress_curves_batch_normalized)
                            
                            # Show progress (only when batch is large)
                            if n_batches > 5 and (batch_idx + 1) % max(1, n_batches // 5) == 0:
                                progress = (batch_idx + 1) / n_batches * 100
                                print(f"    Batch prediction progress: {batch_idx + 1}/{n_batches} ({progress:.1f}%)")
                        
                        # Merge all batch results
                        stress_curves_normalized = np.concatenate(stress_curves_list, axis=0)
                        
                        # Denormalize stress curve
                        if self.stress_scaler and self.stress_scaler.get('type') == 'peak_average':
                            stress_curves = stress_curves_normalized * self.stress_scaler['factor']
                        else:
                            stress_curves = stress_curves_normalized
                    except Exception as e:
                        print(f"Warning: Batch prediction of full curve failed, revert to single prediction: {e}")
                        stress_curves = None
                
            except Exception as e:
                print(f"Warning: Batch prediction failed, revert to single prediction: {e}")
                peak_stresses_normalized = None
                peak_strains_normalized = None
                stress_curves = None
            
            # If batch prediction failed, revert to single prediction
            use_batch = (peak_stresses_normalized is not None and 
                        peak_strains_normalized is not None and 
                        stress_curves is not None)
            
            if use_batch:
                # Batch processing mode (GPU acceleration)
                # Denormalize peak
                # If XGBoost output is already original value, use directly; otherwise need to denormalize
                if xgb_output_is_original:
                    peak_stresses = peak_stresses_original_from_xgb
                elif self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                    peak_stresses = peak_stresses_normalized * self.peak_stress_scaler['factor']
                else:
                    peak_stresses = peak_stresses_normalized
                
                # Reasonability check: Concrete compressive strength should be in the range of 10-150 MPa
                # But don't over-limit, only correct the obviously abnormal values
                # Use a wider range, avoid over-truncation causing discontinuity
                if np.any(peak_stresses > 200) or np.any(peak_stresses < 1):
                    # Only limit the obviously abnormal values, use a wider range
                    peak_stresses = np.clip(peak_stresses, 5, 200)
                    # Check if there are too many values truncated (possibly indicating model or normalization problem)
                    n_clipped = np.sum((peak_stresses == 5) | (peak_stresses == 200))
                    if n_clipped > len(peak_stresses) * 0.1:  # If more than 10% of values are truncated
                        print(f"  Warning: {n_clipped}/{len(peak_stresses)} peak stress values are truncated, possibly indicating model prediction range problem")
                
                # If peak_strains hasn't been calculated (possible already calculated in batch prediction)
                if 'peak_strains' not in locals() or peak_strains is None:
                    if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                        peak_strains = peak_strains_normalized * self.peak_strain_scaler['factor']
                    else:
                        peak_strains = peak_strains_normalized.copy()
                
                # Reasonability check: Peak strain should be in the range of 0.001-0.003
                if np.any(peak_strains < 0.0001) or np.any(peak_strains > 0.01):
                    # Limit to reasonable range
                    peak_strains = np.clip(peak_strains, 0.001, 0.003)
                    if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                        peak_strains_normalized = peak_strains / self.peak_strain_scaler['factor']
                
                # Batch calculate normalized ductility area
                if self.strain_scaler and self.strain_scaler.get('type') == 'peak_average':
                    strain_sequence = self.strain_sequence * self.strain_scaler['factor']
                else:
                    strain_sequence = self.strain_sequence.copy()
                
                normalized_ductility_areas = np.zeros(n_pop)
                for i in range(n_pop):
                    try:
                        normalized_ductility_areas[i], _ = calculate_normalized_ductility_area(
                            strain_sequence, stress_curves[i], peak_strains[i], peak_stresses[i], residual_ratio=0.2
                        )
                    except:
                        normalized_ductility_areas[i] = 0.0
                
                # Batch calculate cost and carbon emissions
                costs = np.array([calculate_material_cost(self._array_to_dict(X[i])) for i in range(n_pop)])
                co2s = np.array([calculate_carbon_emissions(self._array_to_dict(X[i])) for i in range(n_pop)])
                
                # Set objective function values
                objectives[:, 0] = -peak_stresses  # Maximize -> Minimize negative value
                objectives[:, 1] = -normalized_ductility_areas  # Maximize -> Minimize negative value
                
                if self.combine_cost_co2:
                    co2_cost_equivalent = co2s * self.co2_price_per_kg
                    combined_costs = (self.cost_weight * costs + self.co2_weight * co2_cost_equivalent)
                    objectives[:, 2] = combined_costs
                else:
                    objectives[:, 2] = costs
                    objectives[:, 3] = co2s
            else:
                # Single prediction mode (fallback solution)
                for i in range(n_pop):
                    try:
                        # 1. Predict peak stress and peak strain
                        # Single prediction mode (fallback solution)
                        peak_stress, peak_strain = self._predict_peaks(X[i])
                        
                        # 2. Predict full stress-strain curve
                        stress_curve = self._predict_full_curve(X[i], peak_stress, peak_strain)
                        
                        # 3. Calculate normalized ductility area
                        if self.strain_scaler and self.strain_scaler.get('type') == 'peak_average':
                            strain_sequence = self.strain_sequence * self.strain_scaler['factor']
                        else:
                            strain_sequence = self.strain_sequence.copy()
                        
                        normalized_ductility_area, critical_idx = calculate_normalized_ductility_area(
                            strain_sequence, stress_curve, peak_strain, peak_stress, residual_ratio=0.2
                        )
                        
                        # 4. Calculate cost and carbon emissions
                        material_params_dict = self._array_to_dict(X[i])
                        cost = calculate_material_cost(material_params_dict)
                        co2 = calculate_carbon_emissions(material_params_dict)
                        
                        # 5. Set objective function values
                        objectives[i, 0] = -peak_stress  # Maximize -> Minimize negative value
                        objectives[i, 1] = -normalized_ductility_area  # Maximize -> Minimize negative value
                        
                        if self.combine_cost_co2:
                            co2_cost_equivalent = co2 * self.co2_price_per_kg
                            combined_cost = (self.cost_weight * cost + self.co2_weight * co2_cost_equivalent)
                            objectives[i, 2] = combined_cost
                        else:
                            objectives[i, 2] = cost
                            objectives[i, 3] = co2
                        
                    except Exception as e:
                        # If prediction failed, set penalty value
                        if i % 100 == 0:  # Print every 100 individuals to avoid too many outputs
                            print(f"Warning: Error evaluating individual {i}: {e}")
                        if self.combine_cost_co2:
                            objectives[i, :] = [1e6, 1e6, 1e6]
                        else:
                            objectives[i, :] = [1e6, 1e6, 1e6, 1e6]
            
            # Record all evaluated points (for subsequent plotting and saving)
            self.all_evaluated_X.append(X.copy())
            self.all_evaluated_F.append(objectives.copy())
            
            # Diagnosis: Check the number of unique values of peak stress (only output on the first evaluation)
            if not hasattr(self, '_diagnostic_printed'):
                if use_batch and peak_stresses is not None:
                    unique_stresses = np.unique(peak_stresses)
                    print(f"\n  [Diagnosis] Peak stress prediction value statistics:")
                    print(f"    Unique value count: {len(unique_stresses)}")
                    print(f"    Value range: [{np.min(peak_stresses):.4f}, {np.max(peak_stresses):.4f}] MPa")
                    print(f"    Average value: {np.mean(peak_stresses):.4f} MPa")
                    print(f"    Standard deviation: {np.std(peak_stresses):.4f} MPa")
                    if len(unique_stresses) <= 10:
                        print(f"    Unique value list: {unique_stresses}")
                        print(f"    ⚠️ Warning: Peak stress unique value count is too few ({len(unique_stresses)}), may cause discontinuous Pareto front")
                        print(f"    Possible reasons:")
                        print(f"      1. XGBoost model output precision is insufficient")
                        print(f"      2. Normalization factor causes value quantization")
                        print(f"      3. Decision variable range is too small, limiting the prediction value")
                    else:
                        print(f"    ✅ Peak stress value distribution is normal（{len(unique_stresses)} unique values）")
                    
                    # Diagnosis: Display the input parameter space to XGBoost
                    print(f"\n  [Diagnosis] Input parameter space to XGBoost（Decision variable range）:")
                    print(f"    Parameter names: {self.material_param_names}")
                    
                    # Get XGBoost feature mapping
                    try:
                        mapping, xgb_feature_names = self._get_xgb_feature_mapping()
                        print(f"\n    XGBoost model expected feature order:")
                        for i, xgb_name in enumerate(xgb_feature_names):
                            # Find the corresponding optimization parameter name
                            opt_name = None
                            for opt_n, map_info in mapping.items():
                                if map_info['xgb_name'] == xgb_name:
                                    opt_name = opt_n
                                    break
                            if opt_name:
                                opt_idx = mapping[opt_name]['opt_idx']
                                xl_val = self.xl[opt_idx]
                                xu_val = self.xu[opt_idx]
                                actual_min = np.min(X[:, opt_idx])
                                actual_max = np.max(X[:, opt_idx])
                                print(f"      {i+1:2d}. {xgb_name:15s} <- {opt_name:25s}: Range [{xl_val:8.2f}, {xu_val:8.2f}], Actual [{actual_min:8.2f}, {actual_max:8.2f}]")
                            else:
                                print(f"      {i+1:2d}. {xgb_name:15s} <- (No corresponding parameter found)")
                    except Exception as e:
                        print(f"     Warning: Unable to get XGBoost feature mapping: {e}")
                        # Revert to simple display
                        for i, (name, xl_val, xu_val) in enumerate(zip(self.material_param_names, self.xl, self.xu)):
                            actual_min = np.min(X[:, i])
                            actual_max = np.max(X[:, i])
                            print(f"      {i+1:2d}. {name:20s}: Range [{xl_val:8.2f}, {xu_val:8.2f}], Actual [{actual_min:8.2f}, {actual_max:8.2f}]")
                    
                    # Diagnosis: Display the normalized parameter range (for CatBoost and Pi-BiLSTM)
                    if material_params_normalized is not None:
                        print(f"\n  [Diagnosis] Normalized parameter range（Input to CatBoost and Pi-BiLSTM, MinMaxScaler normalized to [0,1]）:")
                        print(f"    Note: XGBoost uses original values（not normalized）, CatBoost and Pi-BiLSTM use normalized values")
                        for i, name in enumerate(self.material_param_names):
                            norm_min = np.min(material_params_normalized[:, i])
                            norm_max = np.max(material_params_normalized[:, i])
                            norm_mean = np.mean(material_params_normalized[:, i])
                            print(f"      {i+1:2d}. {name:20s}: [{norm_min:8.4f}, {norm_max:8.4f}], Mean: {norm_mean:8.4f}")
                        
                        # Check if any parameter is normalized to the boundary (possibly causing discontinuity in prediction)
                        boundary_params = []
                        for i, name in enumerate(self.material_param_names):
                            norm_min = np.min(material_params_normalized[:, i])
                            norm_max = np.max(material_params_normalized[:, i])
                            if norm_min < 0.01 or norm_max > 0.99:
                                boundary_params.append((name, norm_min, norm_max))
                        if boundary_params:
                            print(f"\n    ⚠️ Warning: The following parameters are normalized to the boundary（possibly limiting the prediction value）:")
                            for name, nmin, nmax in boundary_params:
                                print(f"      {name}: [{nmin:.4f}, {nmax:.4f}]")
                    
                    # Diagnosis: Check XGBoost output（using original values as input, output is also original values）
                    if peak_stresses_original_from_xgb is not None:
                        print(f"\n  [Diagnosis] XGBoost output（using original values as input, output is also original values）:")
                        print(f"    Input: 15 material parameters（original values, not normalized）")
                        print(f"    Output range: [{np.min(peak_stresses_original_from_xgb):.4f}, {np.max(peak_stresses_original_from_xgb):.4f}] MPa")
                        print(f"    Unique value count: {len(np.unique(peak_stresses_original_from_xgb))}")
                        print(f"    Value distribution: Mean={np.mean(peak_stresses_original_from_xgb):.4f}, Standard deviation={np.std(peak_stresses_original_from_xgb):.4f}")
                        if len(np.unique(peak_stresses_original_from_xgb)) < 10:
                            unique_vals = np.unique(peak_stresses_original_from_xgb)
                            print(f"    ⚠️ Unique value list (first 20): {unique_vals[:20]}")
                        
                        # Display normalized values（for CatBoost and Pi-BiLSTM）
                        if peak_stresses_normalized is not None:
                            print(f"\n  [Diagnosis] Normalized peak stress value（for CatBoost and Pi-BiLSTM）:")
                            print(f"    Normalized output range: [{np.min(peak_stresses_normalized):.4f}, {np.max(peak_stresses_normalized):.4f}]")
                            if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                                print(f"    Normalization factor: {self.peak_stress_scaler['factor']:.4f} MPa")
                self._diagnostic_printed = True
            
            out['F'] = objectives
        
        def _predict_peaks(self, material_params):
            """
            Predict peak stress and peak strain
            
            Args:
                material_params: Material parameters array [15]
            
            Returns:
                peak_stress: Peak stress (MPa)
                peak_strain: Peak strain
            """
            # Normalize material parameters（for CatBoost and Pi-BiLSTM, XGBoost uses original values）
            material_params_normalized = self.material_scaler.transform(
                material_params.reshape(1, -1)
            )[0]
            
            # Predict peak stress（using XGBoost）- using original values, not normalized
            if self.xgb_model is not None:
                try:
                    # XGBoost prediction - directly using original values material_params（not normalized）
                    xgb_output = self.xgb_model.predict(
                        material_params.reshape(1, -1)
                    )[0]
                    
                    # Ensure output is a float, keep precision
                    xgb_output = float(xgb_output)
                    
                    # XGBoost uses original values as input, output is also original values（MPa）
                    # Directly use output as peak stress
                    peak_stress = xgb_output
                    
                    # Calculate normalized value for subsequent calculations（CatBoost and Pi-BiLSTM need normalized values）
                    if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                        peak_stress_normalized = peak_stress / self.peak_stress_scaler['factor']
                    else:
                        peak_stress_normalized = peak_stress
                    
                    # Reasonability check: Concrete compressive strength should be in the range of 10-150 MPa
                    # Use a wider range, avoid over-truncation
                    if peak_stress > 200 or peak_stress < 1:
                        # Only limit the obviously abnormal values
                        peak_stress = np.clip(peak_stress, 5, 200)
                        if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                            peak_stress_normalized = peak_stress / self.peak_stress_scaler['factor']
                        else:
                            peak_stress_normalized = peak_stress
                except Exception as e:
                    print(f"Warning: XGBoost prediction failed, using empirical formula: {e}")
                    # Use empirical formula as backup
                    peak_stress_normalized = self._predict_peak_stress_empirical(material_params)
                    if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                        peak_stress = peak_stress_normalized * self.peak_stress_scaler['factor']
                    else:
                        peak_stress = peak_stress_normalized
            else:
                # Use empirical formula as backup
                peak_stress_normalized = self._predict_peak_stress_empirical(material_params)
                if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                    peak_stress = peak_stress_normalized * self.peak_stress_scaler['factor']
                else:
                    peak_stress = peak_stress_normalized
            
            # Predict peak strain（using CatBoost）
            if self.catboost_model is not None:
                try:
                    # CatBoost model uses 17 features: 15 material + specimen parameters + fc + xiao_formula
                    # Currently material_params_normalized contains 15 parameters（10 material + 5 specimen）
                    # Need to add: fc（peak stress）and xiao_formula（peak strain calculated by Xiao formula）
                    
                    # Calculate Xiao formula value（using peak stress）
                    # Xiao formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
                    # Simplified version: ε_cp = (0.0706 * fc + 0.7811) * (0.172 * √fc + 0.7)
                    peak_stress_for_xiao = peak_stress  # Using original peak stress（not normalized）
                    replacement_ratio = material_params[6]  # r is the 7th parameter（index 6）
                    
                    # Use the complete Xiao formula
                    fc_clamped = max(peak_stress_for_xiao, 1e-6)
                    r_clamped = max(replacement_ratio / 100.0, 1e-8)  # r from percentage to decimal
                    
                    inner = (0.626 * fc_clamped - 4.33) * 1e-7
                    inner_clamped = max(inner, 0.0)
                    term1 = 0.00076 + np.sqrt(inner_clamped)
                    
                    denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
                    term2 = 1.0 + (r_clamped / denom)
                    xiao_formula_value = term1 * term2
                    
                    # CatBoost training uses original values, prediction should also use original values（not normalized）
                    # Build 17-dimensional input: [15 material + specimen parameters（original values）] + [fc（original values）] + [xiao_formula（original values）]
                    catboost_input = np.concatenate([
                        material_params,  # 15 material parameters（original values, not normalized）
                        [peak_stress],  # Peak stress（original values, not normalized）
                        [xiao_formula_value]  # Xiao formula value（original values, not normalized）
                    ]).reshape(1, -1)
                    
                    # CatBoost prediction
                    catboost_output = self.catboost_model.predict(catboost_input)[0]
                    
                    # CatBoost uses original values as input, output is also original values
                    # Directly use output as peak strain
                    peak_strain = float(catboost_output)
                    
                    # Calculate normalized value for subsequent calculations（Pi-BiLSTM needs normalized values）
                    if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                        peak_strain_normalized = peak_strain / self.peak_strain_scaler['factor']
                    else:
                        peak_strain_normalized = peak_strain
                    
                    # Reasonability check: Peak strain should be in the range of 0.001-0.003
                    if peak_strain < 0.0001 or peak_strain > 0.01:
                        # If the value is abnormal, use Xiao formula as backup
                        print(f"Warning: CatBoost prediction failed, using Xiao formula: {peak_strain:.6f}")
                        peak_strain = self._predict_peak_strain_empirical(material_params, peak_stress)
                        if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                            peak_strain_normalized = peak_strain / self.peak_strain_scaler['factor']
                        else:
                            peak_strain_normalized = peak_strain
                except Exception as e:
                    print(f"Warning: CatBoost prediction failed, using Xiao formula: {e}")
                    # Use Xiao formula as backup
                    peak_strain = self._predict_peak_strain_empirical(material_params, peak_stress)
                    if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                        peak_strain_normalized = peak_strain / self.peak_strain_scaler['factor']
                    else:
                        peak_strain_normalized = peak_strain
            else:
                # Use Xiao formula as backup
                peak_strain = self._predict_peak_strain_empirical(material_params, peak_stress)
                if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                    peak_strain_normalized = peak_strain / self.peak_strain_scaler['factor']
                else:
                    peak_strain_normalized = peak_strain
            
            return peak_stress, peak_strain
        
        def _predict_peak_stress_empirical(self, material_params):
            """
            Use empirical formula to predict peak stress（backup method）
            
            Args:
                material_params: Material parameters array [15]（original values）
            
            Returns:
                peak_stress_normalized: Normalized peak stress
            """
            # Here you can use a simple empirical formula
            # For example: Empirical formula based on water-cement ratio and cement strength
            w_c_ratio = material_params[2]  # Water-cement ratio
            cement_strength = material_params[3]  # Cement strength (MPa)
            
            # Simple empirical formula: f_c ≈ k * f_cement * (w/c)^(-α)
            # Here we use a simplified formula, actually it should be fitted to data
            k = 0.5
            alpha = 0.5
            peak_stress_estimate = k * cement_strength * (w_c_ratio ** (-alpha))
            
            # Normalize
            if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                peak_stress_normalized = peak_stress_estimate / self.peak_stress_scaler['factor']
            else:
                peak_stress_normalized = peak_stress_estimate
            
            return peak_stress_normalized
        
        def _predict_peak_strain_empirical(self, material_params, peak_stress):
            """
            Use Xiao formula to predict peak strain（backup method）
            
            Args:
                material_params: Material parameters array [15]（original values）
                peak_stress: Peak stress (MPa)
            
            Returns:
                peak_strain: Original peak strain（not normalized）
            """
            # Use the complete Xiao formula to calculate peak strain
            # Xiao formula: ε_cp = {0.00076 + [(0.626σ_cp - 4.33) × 10^-7]^0.5} × (1 + r / (65.715r^2 - 109.43r + 48.989))
            replacement_ratio = material_params[6]  # r is the 7th parameter（index 6）, unit is %
            r_value = replacement_ratio / 100.0 if replacement_ratio > 1 else replacement_ratio  # Convert to decimal
            
            # Use the complete Xiao formula
            fc_clamped = max(peak_stress, 1e-6)
            r_clamped = max(r_value, 1e-8)
            
            inner = (0.626 * fc_clamped - 4.33) * 1e-7
            inner_clamped = max(inner, 0.0)
            term1 = 0.00076 + np.sqrt(inner_clamped)
            
            denom = 65.715 * r_clamped ** 2 - 109.43 * r_clamped + 48.989
            term2 = 1.0 + (r_clamped / denom)
            peak_strain_estimate = term1 * term2
            
            # Reasonability check: Peak strain should be in the range of 0.001-0.003
            if peak_strain_estimate < 0.0001 or peak_strain_estimate > 0.01:
                # If the calculated value is abnormal, use the simplified empirical formula
                E_estimate = peak_stress * 1000  # Simplified elastic modulus estimation（MPa）
                peak_strain_estimate = peak_stress / E_estimate  # Simplified peak strain estimation
                # Limit to a reasonable range
                peak_strain_estimate = np.clip(peak_strain_estimate, 0.001, 0.003)
            
            return float(peak_strain_estimate)
        
        def _predict_full_curve(self, material_params, peak_stress, peak_strain):
            """
            Predict the full stress-strain curve
            
            Args:
                material_params: Material parameters array [15]
                peak_stress: Peak stress (MPa)
                peak_strain: Peak strain
            
            Returns:
                stress_curve: Stress curve array [curve_length]
            """
            # Normalize material parameters
            material_params_normalized = self.material_scaler.transform(
                material_params.reshape(1, -1)
            )[0]
            
            # Normalize peak stress and peak strain
            if self.peak_stress_scaler and self.peak_stress_scaler.get('type') == 'peak_average':
                peak_stress_normalized = peak_stress / self.peak_stress_scaler['factor']
            else:
                peak_stress_normalized = peak_stress
            
            if self.peak_strain_scaler and self.peak_strain_scaler.get('type') == 'peak_average':
                peak_strain_normalized = peak_strain / self.peak_strain_scaler['factor']
            else:
                peak_strain_normalized = peak_strain
            
            # Prepare input
            strain_seq_normalized = self.strain_sequence.copy()
            material_params_expanded = np.tile(material_params_normalized, (self.curve_length, 1))
            peak_stress_expanded = np.full(self.curve_length, peak_stress_normalized)
            peak_strain_expanded = np.full(self.curve_length, peak_strain_normalized)
            
            # Build input tensor
            x = np.concatenate([
                strain_seq_normalized.reshape(-1, 1),
                material_params_expanded,
                peak_stress_expanded.reshape(-1, 1),
                peak_strain_expanded.reshape(-1, 1)
            ], axis=1)
            
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)  # [1, curve_length, 18]
            
            # Model prediction
            self.bilstm_model.eval()
            with torch.no_grad():
                stress_curve_normalized = self.bilstm_model(x_tensor).cpu().numpy()[0]
            
            # Unnormalize stress curve
            if self.stress_scaler and self.stress_scaler.get('type') == 'peak_average':
                stress_curve = stress_curve_normalized * self.stress_scaler['factor']
            else:
                stress_curve = stress_curve_normalized
            
            return stress_curve
        
        def _calculate_energy_toughness_ratio(self, stress_curve, peak_stress, peak_strain):
            """
            Calculate energy toughness ratio η = W_p / W_u
            
            Use the same calculation method as in the model_cross_validation.py file, ensure consistency
            
            Args:
                stress_curve: Stress curve array [curve_length]
                peak_stress: Peak stress (MPa)
                peak_strain: Peak strain
            
            Returns:
                eta: Energy toughness ratio
            """
            # Unnormalize strain sequence
            if self.strain_scaler and self.strain_scaler.get('type') == 'peak_average':
                strain_sequence = self.strain_sequence * self.strain_scaler['factor']
            else:
                strain_sequence = self.strain_sequence.copy()
            
            # Find peak position
            peak_idx = np.argmax(stress_curve)
            
            # Calculate cumulative strain energy W_u (energy before peak, i.e., ascending segment energy)
            if peak_idx > 0:
                W_u = compute_curve_energy(
                    strain_sequence[:peak_idx+1],
                    stress_curve[:peak_idx+1]
                )
            else:
                W_u = 0.0
            
            # Calculate failure energy dissipation W_p (energy after peak, i.e., descending segment energy)
            if peak_idx < len(stress_curve) - 1:
                W_p = compute_curve_energy(
                    strain_sequence[peak_idx:],
                    stress_curve[peak_idx:]
                )
            else:
                W_p = 0.0
            
            # Calculate energy toughness ratio η = W_p / W_u
            # This ratio reflects the material's toughness: the larger the ratio, the more energy in the descending segment, the tougher the material
            if W_u > 1e-8:
                eta = W_p / W_u
            else:
                eta = 0.0
            
            return eta
        
        def _array_to_dict(self, material_params):
            """
            Convert material parameters array to dictionary
            
            Args:
                material_params: Material parameters array [15]
            
            Returns:
                material_params_dict: Material parameters dictionary
            """
            return {
                'water': material_params[0],
                'cement': material_params[1],
                'w_c_ratio': material_params[2],
                'cement_strength': material_params[3],
                'sand': material_params[4],
                'coarse_aggregate': material_params[5],
                'replacement_ratio': material_params[6],
                'water_absorption': material_params[7],
                'max_particle_size': material_params[8],
                'crushing_index': material_params[9],
                'curing_age': material_params[10],
                'loading_rate': material_params[11],
                'chamfer_ratio': material_params[12],
                'side_length': material_params[13],
                'height_diameter_ratio': material_params[14]
            }
        
        def _get_xgb_feature_mapping(self):
            """
            Get the mapping from optimization code parameter names to XGBoost model feature names
            
            The expected feature order of the XGBoost model（according to peak_stress/XGBoost code）：
            10 material parameters: water, cement, w/c, CS, sand, CA, r, WA, S, CI
            5 specimen parameters: age, μe, DJB, side, GJB
            
            The parameter order in the optimization code:
            water, cement, w_c_ratio, cement_strength, sand, coarse_aggregate, 
            replacement_ratio, water_absorption, max_particle_size, crushing_index,
            curing_age, loading_rate, chamfer_ratio, side_length, height_diameter_ratio
            
            Returns:
                mapping: Mapping dictionary, key is the optimization code parameter name, value is the XGBoost feature name and index
            """
            # The expected feature names of the XGBoost model（in order）
            xgb_feature_names = [
                'water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI',  # 10 material parameters
                'age', 'μe', 'DJB', 'side', 'GJB'  # 5 specimen parameters
            ]
            
            # The parameter names in the optimization code（in order）
            opt_param_names = self.material_param_names
            
            # Create mapping
            mapping = {}
            for i, opt_name in enumerate(opt_param_names):
                # Try to find the corresponding XGBoost feature name
                xgb_name = None
                xgb_idx = None
                
                # Direct matching
                if opt_name in xgb_feature_names:
                    xgb_idx = xgb_feature_names.index(opt_name)
                    xgb_name = xgb_feature_names[xgb_idx]
                else:
                    # Try fuzzy matching
                    name_mapping = {
                        'w_c_ratio': 'w/c',
                        'cement_strength': 'CS',
                        'coarse_aggregate': 'CA',
                        'replacement_ratio': 'r',
                        'water_absorption': 'WA',
                        'max_particle_size': 'S',
                        'crushing_index': 'CI',
                        'curing_age': 'age',
                        'loading_rate': 'μe',
                        'chamfer_ratio': 'DJB',
                        'side_length': 'side',
                        'height_diameter_ratio': 'GJB'
                    }
                    if opt_name in name_mapping:
                        xgb_name = name_mapping[opt_name]
                        xgb_idx = xgb_feature_names.index(xgb_name) if xgb_name in xgb_feature_names else None
                
                mapping[opt_name] = {
                    'xgb_name': xgb_name,
                    'xgb_idx': xgb_idx,
                    'opt_idx': i
                }
            
            return mapping, xgb_feature_names
else:
    # If pymoo is not available, create a placeholder class
    class RACMixOptimizationProblem:
        def __init__(self, *args, **kwargs):
            raise ImportError("pymoo library is not installed, cannot use multi-objective optimization. Please run: pip install pymoo")


# ========== Main function ==========
def run_multi_objective_optimization(
    xgb_model_path=None,
    catboost_model_path=None,
    bilstm_model_path=None,
    excel_file=None,
    n_generations=100,
    pop_size=500,
    save_dir=None,
    combine_cost_co2=False,
    cost_weight=0.5,
    co2_weight=0.5,
    co2_price_per_kg=0.1
):
    """
    Run multi-objective optimization
    
    Args:
        xgb_model_path: XGBoost model path
        catboost_model_path: CatBoost model path（for predicting peak strain）
        bilstm_model_path: Pi-BiLSTM model path
        excel_file: Excel data file path
        n_generations: Number of generations
        pop_size: Population size
        save_dir: Save directory
                combine_cost_co2: Whether to combine cost and carbon emission into a single objective (default False, i.e., 4 independent objectives)
                cost_weight: Cost weight (used when combine_cost_co2=True, default 0.5)
                co2_weight: Carbon emission weight (used when combine_cost_co2=True, default 0.5)
                co2_price_per_kg: Carbon emission price ($/kg CO₂), used to convert carbon emissions to cost equivalent (default 0.1)
    """
    if not PYMOO_AVAILABLE:
        print("Error: pymoo library is not installed, cannot run multi-objective optimization")
        print("Please run: pip install pymoo")
        return None
    
    print("="*80)
    print("RAC mix design multi-objective optimization - NSGA-III")
    print("="*80)
    
    # Set default path
    if save_dir is None:
        # Default save to multi_objective_optimization/SAVE directory
        save_dir = str(CURRENT_DIR / "SAVE")
    
    if bilstm_model_path is None:
        bilstm_model_path = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\LSTM\SAVE\bidirectional_lstm_cv\best_model.pth"
    if excel_file is None:
        excel_file = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\dataset\dataset_final.xlsx"
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load Pi-BiLSTM model
    print("\n1. Load Pi-BiLSTM model...")
    bilstm_model, model_info = load_trained_model(bilstm_model_path)
    
    # Load data to get normalizers
    print("\n2. Load data to get normalizers...")
    (X_strain, X_stress, X_material, X_peak_stress, X_peak_strain, material_param_names,
     strain_scaler, stress_scaler, material_scaler, peak_stress_scaler, peak_strain_scaler,
     X_material_original, X_stress_original, X_peak_stress_original, X_peak_strain_original,
     sample_divisions, cluster_labels, extra_data) = load_dataset_with_clusters(
        material_params_file=excel_file,
        stress_data_file=excel_file,
        curve_length=model_info['curve_length'],
        train_indices=None,
        cache_dir=None,
        use_cache=True,
        default_cluster_count=None,
        verbose=False
    )
    
    # Load XGBoost and CatBoost models
    print("\n3. Load XGBoost and CatBoost models...")
    import joblib
    import xgboost as xgb
    from catboost import CatBoostRegressor
    
    # Load XGBoost model
    if xgb_model_path is None:
        # Use correct path
        xgb_model_path = os.path.join(PROJECT_ROOT, "peak_stress", "XGBoost", "save", "xgboost_final_model.joblib")
        if not os.path.exists(xgb_model_path):
            # Try other possible paths as backup
            possible_paths = [
                r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\peak_stress\SAVE\xgboost_cv\interpretability_analysis\final_xgboost_model.pkl",
                r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\peak_stress\SAVE\xgboost_cv\best_xgboost_model.pkl",
                r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\peak_stress\SAVE\xgboost_cv_random\interpretability_analysis\final_xgboost_model.pkl"
            ]
            xgb_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    xgb_model_path = path
                    break
    
    xgb_model = None
    if xgb_model_path and os.path.exists(xgb_model_path):
        try:
            xgb_data = joblib.load(xgb_model_path)
            if isinstance(xgb_data, dict):
                xgb_model = xgb_data.get('model')
            else:
                xgb_model = xgb_data
            print(f"  XGBoost model loaded: {xgb_model_path}")
        except Exception as e:
            print(f"  Warning: Failed to load XGBoost model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Warning: XGBoost model file does not exist, will use empirical formula as backup")
    
    # Load CatBoost model
    if catboost_model_path is None:
        catboost_model_path = os.path.join(PROJECT_ROOT, "peak_strain", "CatBoost", "SAVE", "catboost_final_model.joblib")
    
    catboost_model = None
    if os.path.exists(catboost_model_path):
        try:
            # CatBoost model uses joblib to save
            catboost_data = joblib.load(catboost_model_path)
            if isinstance(catboost_data, dict):
                catboost_model = catboost_data.get('model')
                if catboost_model is None:
                    # If the dictionary does not have the 'model' key, try to use the entire dictionary as the model
                    catboost_model = catboost_data
            else:
                catboost_model = catboost_data
            print(f"  CatBoost model loaded: {catboost_model_path}")
        except Exception as e:
            print(f"  Warning: Failed to load CatBoost model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Warning: CatBoost model file does not exist: {catboost_model_path}, will use empirical formula as backup")
    
    if xgb_model is None or catboost_model is None:
        print("  Warning: XGBoost or CatBoost model not loaded, will try to read predicted values from Excel file")
        # Try to read predicted values from Excel file
        try:
            material_df = pd.read_excel(excel_file, sheet_name=0)
            if 'XGB_fc' in material_df.columns and 'peak_strain_catboost_pred_xgbfc' in material_df.columns:
                print("  Found predicted peak columns in Excel file, will use these values as reference")
        except:
            pass
    
    # Create optimization problem
    print("\n4. Create optimization problem...")
    
    # Automatically calculate variable bounds from dataset（using original data, not normalized）
    print("  4.1. Automatically calculate variable bounds from dataset...")
    if X_material_original is not None and X_material_original.shape[1] == len(material_param_names):
        # Calculate the minimum and maximum values of each parameter
        xl_from_data = np.min(X_material_original, axis=0)
        xu_from_data = np.max(X_material_original, axis=0)
        
        # Ensure the boundaries are reasonable（avoid numerical errors）
        xl_from_data = np.maximum(xl_from_data, -1e10)  # Prevent very small values
        xu_from_data = np.minimum(xu_from_data, 1e10)    # Prevent very large values
        
        variable_bounds = {
            'xl': xl_from_data,
            'xu': xu_from_data
        }
        
        print(f"    Variable bounds read from dataset (total {len(material_param_names)} parameters):")
        for i, name in enumerate(material_param_names):
            print(f"      {i+1:2d}. {name:25s}: [{xl_from_data[i]:10.4f}, {xu_from_data[i]:10.4f}]")
    else:
        print(f"  Warning: Unable to read variable bounds from dataset（X_material_original shape: {X_material_original.shape if X_material_original is not None else 'None'}）")
        print(f"  Will use default boundary values")
        variable_bounds = None
    
    # Print normalization factor information for debugging
    if peak_stress_scaler and peak_stress_scaler.get('type') == 'peak_average':
        print(f"\n  Peak stress normalization factor: {peak_stress_scaler['factor']:.2f} MPa")
        print(f"  Peak strain normalization factor: {peak_strain_scaler['factor']:.6f}")
        print(f"  Note: If the normalization factor is too large（>100）, XGBoost output may already be the original value（MPa）")
    
    problem = RACMixOptimizationProblem(
        xgb_model=xgb_model,
        catboost_model=catboost_model,
        bilstm_model=bilstm_model,
        model_info=model_info,
        material_scaler=material_scaler,
        peak_stress_scaler=peak_stress_scaler,
        peak_strain_scaler=peak_strain_scaler,
        strain_scaler=strain_scaler,
        stress_scaler=stress_scaler,
        curve_length=model_info['curve_length'],
        material_param_names=material_param_names,
        combine_cost_co2=combine_cost_co2,
        cost_weight=cost_weight,
        co2_weight=co2_weight,
        co2_price_per_kg=co2_price_per_kg,
        variable_bounds=variable_bounds
    )
    
    # Configure NSGA-III algorithm (more suitable for 4-objective optimization)
    print("\n5. Configure NSGA-III algorithm...")
    n_obj = 3 if combine_cost_co2 else 4
    
    # For 4-objective problem, increase the number of reference directions to get more Pareto solutions
    # The larger the n_partitions, the more reference directions, but the larger the computational cost
    # For 4-objective: n_partitions=12 will produce approximately 455 reference directions
    # If you want more solutions, you can increase it to 15 or 20
    n_partitions = 12
    if n_obj == 4:
        # For 4-objective, you can increase the number of reference directions
        # But note: the number of reference directions = C(n_partitions + n_obj - 1, n_obj - 1)
        # n_partitions=12: approximately 455 reference directions
        # n_partitions=15: approximately 816 reference directions
        # n_partitions=20: approximately 1771 reference directions
        n_partitions = 15  # Increase to 15 to get more reference directions
    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    
    # Ensure the population size is a multiple of the number of reference directions（NSGA-III requires）
    if pop_size < len(ref_dirs):
        pop_size = len(ref_dirs)
        print(f"  Adjust population size to match the number of reference directions: {pop_size}")
    elif pop_size % len(ref_dirs) != 0:
        pop_size = ((pop_size // len(ref_dirs)) + 1) * len(ref_dirs)
        print(f"  Adjust population size to match the number of reference directions: {pop_size}")
    
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    print(f"  Number of reference directions: {len(ref_dirs)}")
    print(f"  Population size: {pop_size}")
    print(f"  Note: The number of reference directions determines the maximum number of Pareto solutions, currently can obtain approximately {len(ref_dirs)} different Pareto solutions")
    
    # GPU acceleration status
    print(f"\n  GPU acceleration status:")
    print(f"    - Pi-BiLSTM model: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"    - Batch prediction: Enabled (utilizing GPU batch processing, significantly accelerating evaluation speed)")
    print(f"    - XGBoost/CatBoost: CPU (if GPU acceleration is needed, configure during training)")
    print(f"    Note: NSGA-III algorithm framework itself is CPU-based, but the model prediction in the evaluation function is using GPU batch acceleration")
    
    # Test if the predicted values are reasonable (before optimization)
    print(f"\n5.5. Test the reasonableness of the predicted values...")
    try:
        # Create multiple test samples (using different positions of the variable range)
        test_samples = [
            problem.xl + (problem.xu - problem.xl) * 0.1,  # 10% position
            (problem.xl + problem.xu) / 2.0,                # middle value
            problem.xl + (problem.xu - problem.xl) * 0.9,  # 90% position
        ]
        
        test_stresses = []
        for i, test_X in enumerate(test_samples):
            test_peak_stress, test_peak_strain = problem._predict_peaks(test_X)
            test_stresses.append(test_peak_stress)
            print(f"  Test sample {i+1} (position {10*(i*4+1) if i < 2 else 90}%):")
            print(f"    Peak stress: {test_peak_stress:.4f} MPa")
            print(f"    Peak strain: {test_peak_strain:.6f}")
        
        # Check the continuity and distribution of the predicted values
        test_stresses = np.array(test_stresses)
        unique_stresses = np.unique(test_stresses)
        print(f"\n  Check the continuity of the predicted values:")
        print(f"    Number of unique values: {len(unique_stresses)} / {len(test_stresses)}")
        print(f"    Value range: [{np.min(test_stresses):.4f}, {np.max(test_stresses):.4f}] MPa")
        print(f"    Value difference: {np.max(test_stresses) - np.min(test_stresses):.4f} MPa")
        
        if len(unique_stresses) < len(test_stresses):
            print(f"    ⚠️ Warning: Different inputs produce the same predicted values, may indicate model output is discontinuous")
            print(f"    Suggestion: Check the training data and prediction accuracy of the XGBoost model")
        
        # Check the reasonableness
        if np.any(test_stresses > 200):
            print(f"  ⚠️ Warning: Some peak stress predicted values are abnormal (> 200 MPa)")
            print(f"    Possible reasons: The XGBoost model output is already the original value (MPa), no need to multiply by normalization factor")
            print(f"    Suggestion: Check the training method, confirm whether output is normalized value or original value")
        elif np.any(test_stresses < 5):
            print(f"  ⚠️ Warning: Some peak stress predicted values are too small (< 5 MPa)")
        elif np.all((test_stresses >= 10) & (test_stresses <= 150)):
            print(f"  ✅ All peak stress predicted values are in the reasonable range (10-150 MPa)")
        else:
            print(f"  ⚠️ Note: Some peak stress predicted values are on the boundary range")
    except Exception as e:
        print(f"  Warning: Error testing predicted values: {e}")
        import traceback
        traceback.print_exc()
    
    # Run optimization
    print(f"\n6. Start optimization (population size: {pop_size}, number of generations: {n_generations})...")
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        verbose=True,
        seed=42
    )
    
    # Save the results
    print("\n7. Save the optimization results...")
    
    # Extract the Pareto front solutions
    pareto_solutions = res.X
    pareto_objectives = res.F
    
    # Collect all evaluated points（for plotting and saving）
    print("\n7.1. Collect all evaluated points...")
    all_evaluated_X = np.vstack(problem.all_evaluated_X) if len(problem.all_evaluated_X) > 0 else np.array([])
    all_evaluated_F = np.vstack(problem.all_evaluated_F) if len(problem.all_evaluated_F) > 0 else np.array([])
    
    # Convert the objective values of all evaluated points（restore the original direction）
    if len(all_evaluated_F) > 0:
        all_evaluated_F_original = all_evaluated_F.copy()
        all_evaluated_F_original[:, 0] = -all_evaluated_F[:, 0]  # Peak stress（maximization）
        all_evaluated_F_original[:, 1] = -all_evaluated_F[:, 1]  # Normalized ductility area（maximization）
    else:
        all_evaluated_F_original = np.array([])
    
    print(f"  Total number of evaluated points: {len(all_evaluated_X)}")
    print(f"  Number of Pareto optimal solutions: {len(pareto_solutions)}")
    
    # Diagnosis: analyze why the Pareto solutions are so few
    if len(pareto_solutions) < 10:
        print(f"\n  ⚠️ Diagnosis: The number of Pareto solutions is too few ({len(pareto_solutions)} solutions)")
        print(f"    Possible reasons:")
        print(f"      1. The objective values are not continuous (XGBoost/CatBoost model outputs discrete values)")
        print(f"      2. The cost and carbon emissions values are completely identical or very close")
        print(f"      3. The number of reference directions is insufficient (current: {len(ref_dirs)} directions)")
        print(f"      4. The decision variable range is too small, limiting diversity")
        
        # Check the uniqueness of the objective values
        if len(all_evaluated_F_original) > 0:
            unique_stresses = len(np.unique(np.round(all_evaluated_F_original[:, 0], decimals=2)))
            unique_ductilities = len(np.unique(np.round(all_evaluated_F_original[:, 1], decimals=4)))
            unique_costs = len(np.unique(np.round(all_evaluated_F_original[:, 2], decimals=2)))
            if not combine_cost_co2:
                unique_co2s = len(np.unique(np.round(all_evaluated_F_original[:, 3], decimals=2)))
            
            print(f"\n    Analysis of the uniqueness of the objective values:")
            print(f"    Number of unique peak stress values: {unique_stresses} (range: {np.min(all_evaluated_F_original[:, 0]):.2f} - {np.max(all_evaluated_F_original[:, 0]):.2f} MPa)")
            print(f"    Number of unique ductility area values: {unique_ductilities} (range: {np.min(all_evaluated_F_original[:, 1]):.4f} - {np.max(all_evaluated_F_original[:, 1]):.4f})")
            print(f"    Number of unique cost values: {unique_costs} (range: {np.min(all_evaluated_F_original[:, 2]):.2f} - {np.max(all_evaluated_F_original[:, 2]):.2f} $/m³)")
            if not combine_cost_co2:
                print(f"    Number of unique carbon emission values: {unique_co2s} (range: {np.min(all_evaluated_F_original[:, 3]):.2f} - {np.max(all_evaluated_F_original[:, 3]):.2f} kg CO₂/m³)")
            
            # If the number of unique values is few, it means the model output is not continuous
                if unique_stresses < 10:
                    print(f"      ⚠️ Peak stress unique value is small（{unique_stresses}），it means the XGBoost model output is not continuous")
            if unique_costs < 10:
                print(f"      ⚠️ Cost unique value is small（{unique_costs}），it means the decision variable change has little effect on the cost")
            if not combine_cost_co2 and unique_co2s < 10:
                print(f"      ⚠️ Carbon emission unique value is few ({unique_co2s}), it means the decision variable change has little effect on the carbon emission")
        
        # Check the objective values of the Pareto solutions
        if len(pareto_solutions) > 0:
            print(f"\n    Analysis of the objective values of the Pareto solutions:")
            for i, (sol, obj) in enumerate(zip(pareto_solutions, pareto_objectives_original)):
                print(f"      Solution {i+1}:")
                print(f"        Peak stress: {obj[0]:.2f} MPa")
                print(f"        Ductility area: {obj[1]:.4f}")
                print(f"        Cost: {obj[2]:.2f} $/m³")
                if not combine_cost_co2:
                    print(f"        Carbon emission: {obj[3]:.2f} kg CO₂/m³")
        
        print(f"\n    Suggestions:")
        print(f"      1. Increase the number of reference directions（n_partitions）to get more Pareto solutions")
        print(f"      2. Check the output of the XGBoost and CatBoost models, ensure the predicted values are continuous")
        print(f"      3. Expand the range of the decision variables, increase the diversity of the solutions")
        print(f"      4. Increase the number of generations, give the algorithm more time to explore")
    
    # Convert the objective values（restore the original direction）
    pareto_objectives_original = pareto_objectives.copy()
    pareto_objectives_original[:, 0] = -pareto_objectives[:, 0]  # Peak stress（maximization）
    pareto_objectives_original[:, 1] = -pareto_objectives[:, 1]  # Normalized ductility area（maximization）
    
    # Identify the best points
    best_stress_idx = np.argmax(pareto_objectives_original[:, 0])  # Maximum peak stress
    best_toughness_idx = np.argmax(pareto_objectives_original[:, 1])  # Maximum normalized ductility area
    best_cost_idx = np.argmin(pareto_objectives_original[:, 2])  # Minimum cost
    if not combine_cost_co2:
        best_co2_idx = np.argmin(pareto_objectives_original[:, 3])  # Minimum CO₂
    else:
        # If the combined cost is used, the CO₂ needs to be recalculated to find the minimum CO₂ point
        co2s_temp = np.array([calculate_carbon_emissions(problem._array_to_dict(sol)) for sol in pareto_solutions])
        best_co2_idx = np.argmin(co2s_temp)
    
    # Calculate the best balanced point（using TOPSIS method: the point closest to the ideal solution）
    # Ideal solution: peak stress maximum, ductility area maximum, cost minimum, CO₂ minimum
    # Negative ideal solution: peak stress minimum, ductility area minimum, cost maximum, CO₂ maximum
    if not combine_cost_co2:
        # 4 objectives cases
        # Normalize the objectives（to make them in the same order of magnitude）
        stress_norm = (pareto_objectives_original[:, 0] - pareto_objectives_original[:, 0].min()) / (pareto_objectives_original[:, 0].max() - pareto_objectives_original[:, 0].min() + 1e-10)
        ductility_norm = (pareto_objectives_original[:, 1] - pareto_objectives_original[:, 1].min()) / (pareto_objectives_original[:, 1].max() - pareto_objectives_original[:, 1].min() + 1e-10)
        cost_norm = 1 - (pareto_objectives_original[:, 2] - pareto_objectives_original[:, 2].min()) / (pareto_objectives_original[:, 2].max() - pareto_objectives_original[:, 2].min() + 1e-10)  # Lower cost is better, so invert
        co2_norm = 1 - (pareto_objectives_original[:, 3] - pareto_objectives_original[:, 3].min()) / (pareto_objectives_original[:, 3].max() - pareto_objectives_original[:, 3].min() + 1e-10)  # Lower CO₂ is better, so invert
        
        # Ideal solution（all objectives are optimal）
        ideal_solution = np.array([1.0, 1.0, 1.0, 1.0])
        # Negative ideal solution（all objectives are worst）
        nadir_solution = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Calculate the distance to the ideal and negative ideal solutions
        distances_to_ideal = np.sqrt(
            (stress_norm - ideal_solution[0])**2 +
            (ductility_norm - ideal_solution[1])**2 +
            (cost_norm - ideal_solution[2])**2 +
            (co2_norm - ideal_solution[3])**2
        )
        distances_to_nadir = np.sqrt(
            (stress_norm - nadir_solution[0])**2 +
            (ductility_norm - nadir_solution[1])**2 +
            (cost_norm - nadir_solution[2])**2 +
            (co2_norm - nadir_solution[3])**2
        )
        
        # TOPSIS scores: the closer to the negative ideal solution, the farther from the ideal solution, the higher the score
        topsis_scores = distances_to_nadir / (distances_to_ideal + distances_to_nadir + 1e-10)
        best_balanced_idx = np.argmax(topsis_scores)
    else:
        # 3 objectives cases（combined cost）
        stress_norm = (pareto_objectives_original[:, 0] - pareto_objectives_original[:, 0].min()) / (pareto_objectives_original[:, 0].max() - pareto_objectives_original[:, 0].min() + 1e-10)
        ductility_norm = (pareto_objectives_original[:, 1] - pareto_objectives_original[:, 1].min()) / (pareto_objectives_original[:, 1].max() - pareto_objectives_original[:, 1].min() + 1e-10)
        combined_cost_norm = 1 - (pareto_objectives_original[:, 2] - pareto_objectives_original[:, 2].min()) / (pareto_objectives_original[:, 2].max() - pareto_objectives_original[:, 2].min() + 1e-10)
        
        ideal_solution = np.array([1.0, 1.0, 1.0])
        nadir_solution = np.array([0.0, 0.0, 0.0])
        
        distances_to_ideal = np.sqrt(
            (stress_norm - ideal_solution[0])**2 +
            (ductility_norm - ideal_solution[1])**2 +
            (combined_cost_norm - ideal_solution[2])**2
        )
        distances_to_nadir = np.sqrt(
            (stress_norm - nadir_solution[0])**2 +
            (ductility_norm - nadir_solution[1])**2 +
            (combined_cost_norm - nadir_solution[2])**2
        )
        
        topsis_scores = distances_to_nadir / (distances_to_ideal + distances_to_nadir + 1e-10)
        best_balanced_idx = np.argmax(topsis_scores)
    
    # Save to Excel
    results_df = pd.DataFrame(
        pareto_solutions,
        columns=material_param_names
    )
    results_df['Peak_Stress_MPa'] = pareto_objectives_original[:, 0]
    results_df['Normalized_Ductility_Area'] = pareto_objectives_original[:, 1]  # Normalized ductility area
    
    if combine_cost_co2:
        # 3 objectives: the cost and carbon emissions need to be recalculated for saving
        results_df['Combined_Cost_USD_per_m3'] = pareto_objectives_original[:, 2]
        # Recalculate the cost and carbon emissions for each solution for saving
        costs = []
        co2s = []
        for solution in pareto_solutions:
            material_params_dict = problem._array_to_dict(solution)
            costs.append(calculate_material_cost(material_params_dict))
            co2s.append(calculate_carbon_emissions(material_params_dict))
        results_df['Cost_USD_per_m3'] = costs
        results_df['CO2_kg_per_m3'] = co2s
    else:
        # 4 objectives
        results_df['Cost_USD_per_m3'] = pareto_objectives_original[:, 2]
        results_df['CO2_kg_per_m3'] = pareto_objectives_original[:, 3]
    
    # Add label column（using English labels）
    labels = [''] * len(pareto_solutions)
    labels[best_stress_idx] = 'Best Strength'
    labels[best_toughness_idx] = 'Best Ductility'
    labels[best_cost_idx] = 'Lowest Cost'
    labels[best_co2_idx] = 'Lowest CO₂'
    labels[best_balanced_idx] = 'Best Balanced'
    
    # If a point satisfies multiple conditions, merge the labels
    if best_stress_idx == best_toughness_idx:
        labels[best_stress_idx] = 'Best Strength, Best Ductility'
    if best_stress_idx == best_cost_idx:
        labels[best_stress_idx] = labels[best_stress_idx] + ', Lowest Cost' if labels[best_stress_idx] else 'Best Strength, Lowest Cost'
    if best_stress_idx == best_co2_idx:
        labels[best_stress_idx] = labels[best_stress_idx] + ', Lowest CO₂' if labels[best_stress_idx] else 'Best Strength, Lowest CO₂'
    if best_toughness_idx == best_cost_idx:
        labels[best_toughness_idx] = labels[best_toughness_idx] + ', Lowest Cost' if labels[best_toughness_idx] else 'Best Ductility, Lowest Cost'
    if best_toughness_idx == best_co2_idx:
        labels[best_toughness_idx] = labels[best_toughness_idx] + ', Lowest CO₂' if labels[best_toughness_idx] else 'Best Ductility, Lowest CO₂'
    if best_cost_idx == best_co2_idx:
        labels[best_cost_idx] = labels[best_cost_idx] + ', Lowest CO₂' if labels[best_cost_idx] else 'Lowest Cost, Lowest CO₂'
    
    # If the best balanced point coincides with other points, the labels also need to be merged
    if best_balanced_idx == best_stress_idx:
        labels[best_balanced_idx] = labels[best_balanced_idx] + ', Best Balanced' if labels[best_balanced_idx] else 'Best Strength, Best Balanced'
    elif best_balanced_idx == best_toughness_idx:
        labels[best_balanced_idx] = labels[best_balanced_idx] + ', Best Balanced' if labels[best_balanced_idx] else 'Best Ductility, Best Balanced'
    elif best_balanced_idx == best_cost_idx:
        labels[best_balanced_idx] = labels[best_balanced_idx] + ', Best Balanced' if labels[best_balanced_idx] else 'Lowest Cost, Best Balanced'
    elif best_balanced_idx == best_co2_idx:
        labels[best_balanced_idx] = labels[best_balanced_idx] + ', Best Balanced' if labels[best_balanced_idx] else 'Lowest CO₂, Best Balanced'
    
    results_df['Label'] = labels
    
    excel_path = os.path.join(save_dir, 'pareto_optimal_solutions.xlsx')
    results_df.to_excel(excel_path, index=False)
    print(f"Pareto optimal solutions have been saved to: {excel_path}")
    print(f"  Labeled: Best Strength (index {best_stress_idx}), Best Ductility (index {best_toughness_idx}), Lowest Cost (index {best_cost_idx}), Lowest CO₂ (index {best_co2_idx}), Best Balanced (index {best_balanced_idx})")
    
    # Save all evaluated points to Excel（including non-Pareto solutions）
    if len(all_evaluated_X) > 0 and len(all_evaluated_F_original) > 0:
        print("\n7.2. Save all evaluated points to Excel...")
        # Check which points are Pareto solutions
        pareto_indices_set = set()
        for pareto_sol in pareto_solutions:
            # Find matching indices（using tolerance comparison）
            matches = np.where(np.all(np.abs(all_evaluated_X - pareto_sol) < 1e-6, axis=1))[0]
            if len(matches) > 0:
                pareto_indices_set.add(matches[0])
        
        # Create a DataFrame for all evaluated points
        all_results_df = pd.DataFrame(
            all_evaluated_X,
            columns=material_param_names
        )
        
        # Add objective values
        all_results_df['Peak_Stress_MPa'] = all_evaluated_F_original[:, 0]
        all_results_df['Normalized_Ductility_Area'] = all_evaluated_F_original[:, 1]
        if combine_cost_co2:
            all_results_df['Combined_Cost_USD_per_m3'] = all_evaluated_F_original[:, 2]
            # Recalculate the cost and carbon emissions
            costs_all = []
            co2s_all = []
            for solution in all_evaluated_X:
                material_params_dict = problem._array_to_dict(solution)
                costs_all.append(calculate_material_cost(material_params_dict))
                co2s_all.append(calculate_carbon_emissions(material_params_dict))
            all_results_df['Cost_USD_per_m3'] = costs_all
            all_results_df['CO2_kg_per_m3'] = co2s_all
        else:
            all_results_df['Cost_USD_per_m3'] = all_evaluated_F_original[:, 2]
            all_results_df['CO2_kg_per_m3'] = all_evaluated_F_original[:, 3]
        
        # Add whether the point is a Pareto solution
        all_results_df['Is_Pareto'] = [i in pareto_indices_set for i in range(len(all_evaluated_X))]
        
        # Calculate the Closeness value（using TOPSIS method）
        if not combine_cost_co2:
            # 4 objectives cases
            stress_norm_all = (all_evaluated_F_original[:, 0] - all_evaluated_F_original[:, 0].min()) / (all_evaluated_F_original[:, 0].max() - all_evaluated_F_original[:, 0].min() + 1e-10)
            ductility_norm_all = (all_evaluated_F_original[:, 1] - all_evaluated_F_original[:, 1].min()) / (all_evaluated_F_original[:, 1].max() - all_evaluated_F_original[:, 1].min() + 1e-10)
            cost_norm_all = 1 - (all_evaluated_F_original[:, 2] - all_evaluated_F_original[:, 2].min()) / (all_evaluated_F_original[:, 2].max() - all_evaluated_F_original[:, 2].min() + 1e-10)
            co2_norm_all = 1 - (all_evaluated_F_original[:, 3] - all_evaluated_F_original[:, 3].min()) / (all_evaluated_F_original[:, 3].max() - all_evaluated_F_original[:, 3].min() + 1e-10)
            
            ideal_solution = np.array([1.0, 1.0, 1.0, 1.0])
            nadir_solution = np.array([0.0, 0.0, 0.0, 0.0])
            
            distances_to_ideal_all = np.sqrt(
                (stress_norm_all - ideal_solution[0])**2 +
                (ductility_norm_all - ideal_solution[1])**2 +
                (cost_norm_all - ideal_solution[2])**2 +
                (co2_norm_all - ideal_solution[3])**2
            )
            distances_to_nadir_all = np.sqrt(
                (stress_norm_all - nadir_solution[0])**2 +
                (ductility_norm_all - nadir_solution[1])**2 +
                (cost_norm_all - nadir_solution[2])**2 +
                (co2_norm_all - nadir_solution[3])**2
            )
            
            closeness_all = distances_to_nadir_all / (distances_to_ideal_all + distances_to_nadir_all + 1e-10)
        else:
            # 3 objectives cases
            stress_norm_all = (all_evaluated_F_original[:, 0] - all_evaluated_F_original[:, 0].min()) / (all_evaluated_F_original[:, 0].max() - all_evaluated_F_original[:, 0].min() + 1e-10)
            ductility_norm_all = (all_evaluated_F_original[:, 1] - all_evaluated_F_original[:, 1].min()) / (all_evaluated_F_original[:, 1].max() - all_evaluated_F_original[:, 1].min() + 1e-10)
            combined_cost_norm_all = 1 - (all_evaluated_F_original[:, 2] - all_evaluated_F_original[:, 2].min()) / (all_evaluated_F_original[:, 2].max() - all_evaluated_F_original[:, 2].min() + 1e-10)
            
            ideal_solution = np.array([1.0, 1.0, 1.0])
            nadir_solution = np.array([0.0, 0.0, 0.0])
            
            distances_to_ideal_all = np.sqrt(
                (stress_norm_all - ideal_solution[0])**2 +
                (ductility_norm_all - ideal_solution[1])**2 +
                (combined_cost_norm_all - ideal_solution[2])**2
            )
            distances_to_nadir_all = np.sqrt(
                (stress_norm_all - nadir_solution[0])**2 +
                (ductility_norm_all - nadir_solution[1])**2 +
                (combined_cost_norm_all - nadir_solution[2])**2
            )
            
            closeness_all = distances_to_nadir_all / (distances_to_ideal_all + distances_to_nadir_all + 1e-10)
        
        all_results_df['Closeness'] = closeness_all
        
        # Save to Excel（including all evaluated points, including test results and Pareto points）
        all_excel_path = os.path.join(save_dir, 'all_evaluated_solutions.xlsx')
        all_results_df.to_excel(all_excel_path, index=False)
        print(f"All evaluated points have been saved to: {all_excel_path}")
        print(f"  Total evaluated points: {len(all_evaluated_X)}")
        print(f"  Pareto solutions: {len(pareto_indices_set)}")
        print(f"  Non-Pareto solutions: {len(all_evaluated_X) - len(pareto_indices_set)}")
        
        # Save another Excel file only containing Pareto points（from all_results_df）
        pareto_only_df = all_results_df[all_results_df['Is_Pareto'] == True].copy()
        pareto_only_excel_path = os.path.join(save_dir, 'pareto_only_solutions.xlsx')
        pareto_only_df.to_excel(pareto_only_excel_path, index=False)
        print(f"Only Pareto points have been saved to: {pareto_only_excel_path}")
        print(f"  Pareto solutions: {len(pareto_only_df)}")
    else:
        all_results_df = None
        print("Warning: Unable to collect all evaluated points (may not be recorded during optimization)")
    
    # Use the energy analysis function in the training_model_cross_validation.py for detailed evaluation
    print("\n7.5. Use the energy analysis function in the training_model_cross_validation.py for detailed evaluation...")
    try:
        # Calculate the complete energy indicators for each Pareto optimal solution
        energy_analysis_results = []
        for i, solution in enumerate(pareto_solutions):
            material_params = solution
            # Predict the peaks
            peak_stress, peak_strain = problem._predict_peaks(material_params)
            # Predict the full curve
            stress_curve = problem._predict_full_curve(material_params, peak_stress, peak_strain)
            # Calculate the energy
            if problem.strain_scaler and problem.strain_scaler.get('type') == 'peak_average':
                strain_sequence = problem.strain_sequence * problem.strain_scaler['factor']
            else:
                strain_sequence = problem.strain_sequence.copy()
            
            # Use compute_curve_energy to calculate the total energy
            total_energy = compute_curve_energy(strain_sequence, stress_curve)
            
            # Calculate the energy toughness ratio
            eta = problem._calculate_energy_toughness_ratio(stress_curve, peak_stress, peak_strain)
            
            # Calculate the normalized ductility area
            normalized_ductility_area, critical_idx = calculate_normalized_ductility_area(
                strain_sequence, stress_curve, peak_strain, peak_stress, residual_ratio=0.2
            )
            
            energy_analysis_results.append({
                'Solution_ID': i + 1,
                'Total_Energy_MPa': total_energy,
                'Energy_Toughness_Ratio': eta,
                'Normalized_Ductility_Area': normalized_ductility_area,  # Normalized ductility area
                'Critical_Index': critical_idx,  # Critical point index
                'Peak_Stress_MPa': peak_stress,
                'Peak_Strain': peak_strain
            })
        
        # Add the energy analysis results to the DataFrame
        energy_df = pd.DataFrame(energy_analysis_results)
        results_df = pd.concat([results_df, energy_df[['Total_Energy_MPa']]], axis=1)
        
        # Save the energy analysis results
        energy_excel_path = os.path.join(save_dir, 'energy_analysis_results.xlsx')
        energy_df.to_excel(energy_excel_path, index=False)
        print(f"Energy analysis results have been saved to: {energy_excel_path}")
        
        # Print the energy statistics
        print("\nEnergy analysis statistics:")
        print(f"  Average total energy: {energy_df['Total_Energy_MPa'].mean():.4f} MPa")
        print(f"  Average energy toughness ratio: {energy_df['Energy_Toughness_Ratio'].mean():.4f}")
        print(f"  Average normalized ductility area: {energy_df['Normalized_Ductility_Area'].mean():.4f}")
        print(f"  Total energy range: [{energy_df['Total_Energy_MPa'].min():.4f}, {energy_df['Total_Energy_MPa'].max():.4f}] MPa")
        print(f"  Energy toughness ratio range: [{energy_df['Energy_Toughness_Ratio'].min():.4f}, {energy_df['Energy_Toughness_Ratio'].max():.4f}]")
        print(f"  Normalized ductility area range: [{energy_df['Normalized_Ductility_Area'].min():.4f}, {energy_df['Normalized_Ductility_Area'].max():.4f}]")
        print(f" Note: The larger the normalized ductility area, the better the material ductility (larger area indicates higher ductility)")
        
    except Exception as e:
        print(f"Warning: Error in energy analysis: {e}")  
        import traceback
        traceback.print_exc()
    
    # Plot the Pareto front
    print("\n8. Plot the Pareto front...")
    try:
        # First plot the 2D scatter plot
        # Plot different figures according to the number of objectives
        if combine_cost_co2:
            # 3 objectives: plot 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            # Peak Stress vs Normalized Ductility Area
            axes[0].scatter(pareto_objectives_original[:, 0], pareto_objectives_original[:, 1], 
                          c='blue', alpha=0.6, s=50)
            axes[0].set_xlabel('Peak Stress (MPa)')
            axes[0].set_ylabel('Normalized Ductility Area')
            axes[0].set_title('Peak Stress vs Normalized Ductility Area')
            axes[0].grid(True, alpha=0.3)
            
            # Peak Stress vs Combined Cost
            axes[1].scatter(pareto_objectives_original[:, 0], pareto_objectives_original[:, 2], 
                          c='red', alpha=0.6, s=50)
            axes[1].set_xlabel('Peak Stress (MPa)')
            axes[1].set_ylabel('Combined Cost ($/m³)')
            axes[1].set_title('Peak Stress vs Combined Cost (Cost + CO$_2$)')
            axes[1].grid(True, alpha=0.3)
            
            # Normalized Ductility Area vs Combined Cost
            axes[2].scatter(pareto_objectives_original[:, 1], pareto_objectives_original[:, 2], 
                          c='orange', alpha=0.6, s=50)
            axes[2].set_xlabel('Normalized Ductility Area')
            axes[2].set_ylabel('Combined Cost ($/m³)')
            axes[2].set_title('Normalized Ductility Area vs Combined Cost')
            axes[2].grid(True, alpha=0.3)
            
            # Cost vs CO2 (recalculated values)
            costs = [calculate_material_cost(problem._array_to_dict(sol)) for sol in pareto_solutions]
            co2s = [calculate_carbon_emissions(problem._array_to_dict(sol)) for sol in pareto_solutions]
            axes[3].scatter(costs, co2s, c='green', alpha=0.6, s=50)
            axes[3].set_xlabel('Cost ($/m³)')
            axes[3].set_ylabel('CO$_2$ Emissions (kg/m³)')
            axes[3].set_title('Cost vs CO$_2$ Emissions (Individual)')
            axes[3].grid(True, alpha=0.3)
        else:
            # 4 objectives: plot 2x3 subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Prepare all evaluated points data（if available）
            if len(all_evaluated_F_original) > 0 and all_results_df is not None:
                # Separate Pareto solutions and non-Pareto solutions
                pareto_mask = all_results_df['Is_Pareto'].values
                closeness_values = all_results_df['Closeness'].values
                
                # Peak Stress vs Normalized Ductility Area
                # Only plot Pareto solutions（using color bar）
                scatter1 = axes[0, 0].scatter(all_evaluated_F_original[pareto_mask, 0], 
                                             all_evaluated_F_original[pareto_mask, 1],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, label='Pareto front', marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[0, 0].set_xlabel('Compressive strength (MPa)', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Normalized Ductility Area', fontsize=12, fontweight='bold')
                axes[0, 0].set_title('(a) Bi-objective', fontsize=14, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter1, ax=axes[0, 0], label='Closeness', shrink=0.8)
                
                # Peak Stress vs Cost
                # Only plot Pareto solutions
                scatter2 = axes[0, 1].scatter(all_evaluated_F_original[pareto_mask, 0], 
                                             all_evaluated_F_original[pareto_mask, 2],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[0, 1].set_xlabel('Compressive strength (MPa)', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Price ($/m³)', fontsize=12, fontweight='bold')
                axes[0, 1].set_title('Peak Stress vs Cost', fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=axes[0, 1], label='Closeness', shrink=0.8)
                
                # Peak Stress vs CO2
                # Only plot Pareto solutions
                scatter3 = axes[0, 2].scatter(all_evaluated_F_original[pareto_mask, 0], 
                                             all_evaluated_F_original[pareto_mask, 3],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[0, 2].set_xlabel('Compressive strength (MPa)', fontsize=12, fontweight='bold')
                axes[0, 2].set_ylabel('Carbon emission (kgCO$_2$/m³)', fontsize=12, fontweight='bold')
                axes[0, 2].set_title('Peak Stress vs CO$_2$ Emissions', fontsize=12, fontweight='bold')
                axes[0, 2].grid(True, alpha=0.3)
                plt.colorbar(scatter3, ax=axes[0, 2], label='Closeness', shrink=0.8)
                
                # Normalized Ductility Area vs Cost
                # Only plot Pareto solutions
                scatter4 = axes[1, 0].scatter(all_evaluated_F_original[pareto_mask, 1], 
                                             all_evaluated_F_original[pareto_mask, 2],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[1, 0].set_xlabel('Normalized Ductility Area', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('Price ($/m³)', fontsize=12, fontweight='bold')
                axes[1, 0].set_title('Normalized Ductility Area vs Cost', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter4, ax=axes[1, 0], label='Closeness', shrink=0.8)
                
                # Normalized Ductility Area vs CO2
                # Only plot Pareto solutions
                scatter5 = axes[1, 1].scatter(all_evaluated_F_original[pareto_mask, 1], 
                                             all_evaluated_F_original[pareto_mask, 3],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[1, 1].set_xlabel('Normalized Ductility Area', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('Carbon emission (kgCO$_2$/m³)', fontsize=12, fontweight='bold')
                axes[1, 1].set_title('Normalized Ductility Area vs CO$_2$ Emissions', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter5, ax=axes[1, 1], label='Closeness', shrink=0.8)
                
                # Cost vs CO2
                # Only plot Pareto solutions
                scatter6 = axes[1, 2].scatter(all_evaluated_F_original[pareto_mask, 2], 
                                             all_evaluated_F_original[pareto_mask, 3],
                                             c=closeness_values[pareto_mask], cmap='viridis',
                                             alpha=0.8, s=80, marker='*',
                                             edgecolors='black', linewidths=0.5)
                axes[1, 2].set_xlabel('Price ($/m³)', fontsize=12, fontweight='bold')
                axes[1, 2].set_ylabel('Carbon emission (kgCO$_2$/m³)', fontsize=12, fontweight='bold')
                axes[1, 2].set_title('Cost vs CO$_2$ Emissions', fontsize=12, fontweight='bold')
                axes[1, 2].grid(True, alpha=0.3)
                plt.colorbar(scatter6, ax=axes[1, 2], label='Closeness', shrink=0.8)
            else:
                # If there is no data for all evaluated points, only plot Pareto solutions
                axes[0, 0].scatter(pareto_objectives_original[:, 0], pareto_objectives_original[:, 1], 
                                  c='blue', alpha=0.6, s=50)
                axes[0, 0].set_xlabel('Peak Stress (MPa)')
                axes[0, 0].set_ylabel('Normalized Ductility Area')
                axes[0, 0].set_title('Peak Stress vs Normalized Ductility Area')
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].scatter(pareto_objectives_original[:, 0], pareto_objectives_original[:, 2], 
                                  c='red', alpha=0.6, s=50)
                axes[0, 1].set_xlabel('Peak Stress (MPa)')
                axes[0, 1].set_ylabel('Cost ($/m³)')
                axes[0, 1].set_title('Peak Stress vs Cost')
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[0, 2].scatter(pareto_objectives_original[:, 0], pareto_objectives_original[:, 3], 
                                  c='green', alpha=0.6, s=50)
                axes[0, 2].set_xlabel('Peak Stress (MPa)')
                axes[0, 2].set_ylabel('CO$_2$ Emissions (kg/m³)')
                axes[0, 2].set_title('Peak Stress vs CO$_2$ Emissions')
                axes[0, 2].grid(True, alpha=0.3)
                
                axes[1, 0].scatter(pareto_objectives_original[:, 1], pareto_objectives_original[:, 2], 
                                  c='orange', alpha=0.6, s=50)
                axes[1, 0].set_xlabel('Normalized Ductility Area')
                axes[1, 0].set_ylabel('Cost ($/m³)')
                axes[1, 0].set_title('Normalized Ductility Area vs Cost')
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].scatter(pareto_objectives_original[:, 1], pareto_objectives_original[:, 3], 
                                  c='purple', alpha=0.6, s=50)
                axes[1, 1].set_xlabel('Normalized Ductility Area')
                axes[1, 1].set_ylabel('CO$_2$ Emissions (kg/m³)')
                axes[1, 1].set_title('Normalized Ductility Area vs CO$_2$ Emissions')
                axes[1, 1].grid(True, alpha=0.3)
                
                axes[1, 2].scatter(pareto_objectives_original[:, 2], pareto_objectives_original[:, 3], 
                                  c='brown', alpha=0.6, s=50)
                axes[1, 2].set_xlabel('Cost ($/m³)')
                axes[1, 2].set_ylabel('CO$_2$ Emissions (kg/m³)')
                axes[1, 2].set_title('Cost vs CO$_2$ Emissions')
                axes[1, 2].grid(True, alpha=0.3)
        
        # Comment out the 2D scatter plot, only keep the core 3D plot and parallel coordinate plot
        # plt.tight_layout()
        # plot_path = os.path.join(save_dir, 'pareto_front.png')
        # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # print(f"Pareto front plot has been saved to: {plot_path}")
        plt.close()
        
        # Plot the 3D Pareto front
        print("\n8.5. Plot the 3D Pareto front...")
        try:
            # Find the best points（the optimal solution for each objective）
            if combine_cost_co2:
                # 3 objectives case
                best_stress_idx = np.argmax(pareto_objectives_original[:, 0])  # Maximum peak stress
                best_toughness_idx = np.argmax(pareto_objectives_original[:, 1])  # Maximum normalized ductility area
                best_cost_idx = np.argmin(pareto_objectives_original[:, 2])  # Minimum combined cost
                
                # Create 3D plot: Peak stress vs Normalized ductility area vs Combined cost
                fig = plt.figure(figsize=(16, 12))
                
                # First 3D plot: Peak stress vs Normalized ductility area vs Combined cost
                ax1 = fig.add_subplot(221, projection='3d')
                scatter1 = ax1.scatter(
                    pareto_objectives_original[:, 0],  # Peak stress
                    pareto_objectives_original[:, 1],  # Normalized ductility area
                    pareto_objectives_original[:, 2],  # Combined cost
                    c=pareto_objectives_original[:, 2],  # Use combined cost to color
                    cmap='viridis',
                    alpha=0.6,
                    s=50
                )
                # Mark the best points
                ax1.scatter(
                    pareto_objectives_original[best_stress_idx, 0],
                    pareto_objectives_original[best_stress_idx, 1],
                    pareto_objectives_original[best_stress_idx, 2],
                    c='red', marker='*', s=300, label='Best strength', edgecolors='black', linewidths=1
                )
                ax1.scatter(
                    pareto_objectives_original[best_toughness_idx, 0],
                    pareto_objectives_original[best_toughness_idx, 1],
                    pareto_objectives_original[best_toughness_idx, 2],
                    c='blue', marker='*', s=300, label='Best ductility', edgecolors='black', linewidths=1
                )
                ax1.scatter(
                    pareto_objectives_original[best_cost_idx, 0],
                    pareto_objectives_original[best_cost_idx, 1],
                    pareto_objectives_original[best_cost_idx, 2],
                    c='green', marker='*', s=300, label='Lowest cost', edgecolors='black', linewidths=1
                )
                ax1.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax1.set_ylabel('Normalized Ductility Area', fontsize=10)
                ax1.set_zlabel('Combined Cost ($/m³)', fontsize=10)
                ax1.set_title('3D Pareto Front: Strength vs Ductility vs Cost', fontsize=12)
                ax1.legend(fontsize=8)
                plt.colorbar(scatter1, ax=ax1, label='Combined Cost ($/m³)', shrink=0.8)
                
                # Recalculate the cost and carbon emissions for the 3D plot
                costs = np.array([calculate_material_cost(problem._array_to_dict(sol)) for sol in pareto_solutions])
                co2s = np.array([calculate_carbon_emissions(problem._array_to_dict(sol)) for sol in pareto_solutions])
                
                # Second 3D plot: Peak stress vs Normalized ductility area vs Cost
                ax2 = fig.add_subplot(222, projection='3d')
                scatter2 = ax2.scatter(
                    pareto_objectives_original[:, 0],
                    pareto_objectives_original[:, 1],
                    costs,
                    c=costs,
                    cmap='Reds',
                    alpha=0.6,
                    s=50
                )
                ax2.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax2.set_ylabel('Normalized Ductility Area', fontsize=10)
                ax2.set_zlabel('Cost ($/m³)', fontsize=10)
                ax2.set_title('3D Pareto Front: Strength vs Ductility vs Cost', fontsize=12)
                plt.colorbar(scatter2, ax=ax2, label='Cost ($/m³)', shrink=0.8)
                
                # Third 3D plot: Peak stress vs Normalized ductility area vs CO₂
                ax3 = fig.add_subplot(223, projection='3d')
                scatter3 = ax3.scatter(
                    pareto_objectives_original[:, 0],
                    pareto_objectives_original[:, 1],
                    co2s,
                    c=co2s,
                    cmap='Greens',
                    alpha=0.6,
                    s=50
                )
                ax3.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax3.set_ylabel('Normalized Ductility Area', fontsize=10)
                ax3.set_zlabel('CO$_2$ Emissions (kg/m³)', fontsize=10)
                ax3.set_title('3D Pareto Front: Strength vs Ductility vs CO$_2$', fontsize=12)
                plt.colorbar(scatter3, ax=ax3, label='CO$_2$ Emissions (kg/m³)', shrink=0.8)
                
                # Fourth 3D plot: Cost vs CO₂ vs Peak stress
                ax4 = fig.add_subplot(224, projection='3d')
                scatter4 = ax4.scatter(
                    costs,
                    co2s,
                    pareto_objectives_original[:, 0],
                    c=pareto_objectives_original[:, 0],
                    cmap='coolwarm',
                    alpha=0.6,
                    s=50
                )
                ax4.set_xlabel('Cost ($/m³)', fontsize=10)
                ax4.set_ylabel('CO$_2$ Emissions (kg/m³)', fontsize=10)
                ax4.set_zlabel('Peak Stress (MPa)', fontsize=10)
                ax4.set_title('3D Pareto Front: Cost vs CO$_2$ vs Strength', fontsize=12)
                plt.colorbar(scatter4, ax=ax4, label='Peak Stress (MPa)', shrink=0.8)
                
            else:
                # 4 objectives case
                best_stress_idx = np.argmax(pareto_objectives_original[:, 0])  # Maximum peak stress
                best_toughness_idx = np.argmax(pareto_objectives_original[:, 1])  # Maximum normalized ductility area
                best_cost_idx = np.argmin(pareto_objectives_original[:, 2])  # Minimum cost
                best_co2_idx = np.argmin(pareto_objectives_original[:, 3])  # Minimum CO₂
                
                # Save the data for the first four 3D plots to Excel for plotting in Origin
                try:
                    stress_vals = pareto_objectives_original[:, 0]
                    ductility_vals = pareto_objectives_original[:, 1]
                    cost_vals = pareto_objectives_original[:, 2]
                    co2_vals = pareto_objectives_original[:, 3]
                    
                    # Add a label column to the Excel plot data for filtering specific solutions in Origin/Excel
                    plot_labels = [''] * len(stress_vals)
                    plot_labels[best_stress_idx] = 'Best Strength'
                    plot_labels[best_toughness_idx] = 'Best Ductility'
                    plot_labels[best_cost_idx] = 'Lowest Cost'
                    plot_labels[best_co2_idx] = 'Lowest CO₂'
                    plot_labels[best_balanced_idx] = 'Best Balanced'
                    # Merge labels（consistent with the main table）
                    if best_stress_idx == best_toughness_idx:
                        plot_labels[best_stress_idx] = 'Best Strength, Best Ductility'
                    if best_stress_idx == best_cost_idx:
                        plot_labels[best_stress_idx] = plot_labels[best_stress_idx] + ', Lowest Cost' if plot_labels[best_stress_idx] else 'Best Strength, Lowest Cost'
                    if best_stress_idx == best_co2_idx:
                        plot_labels[best_stress_idx] = plot_labels[best_stress_idx] + ', Lowest CO₂' if plot_labels[best_stress_idx] else 'Best Strength, Lowest CO₂'
                    if best_toughness_idx == best_cost_idx:
                        plot_labels[best_toughness_idx] = plot_labels[best_toughness_idx] + ', Lowest Cost' if plot_labels[best_toughness_idx] else 'Best Ductility, Lowest Cost'
                    if best_toughness_idx == best_co2_idx:
                        plot_labels[best_toughness_idx] = plot_labels[best_toughness_idx] + ', Lowest CO₂' if plot_labels[best_toughness_idx] else 'Best Ductility, Lowest CO₂'
                    if best_cost_idx == best_co2_idx:
                        plot_labels[best_cost_idx] = plot_labels[best_cost_idx] + ', Lowest CO₂' if plot_labels[best_cost_idx] else 'Lowest Cost, Lowest CO₂'
                    if best_balanced_idx == best_stress_idx:
                        plot_labels[best_balanced_idx] = plot_labels[best_balanced_idx] + ', Best Balanced' if plot_labels[best_balanced_idx] else 'Best Strength, Best Balanced'
                    elif best_balanced_idx == best_toughness_idx:
                        plot_labels[best_balanced_idx] = plot_labels[best_balanced_idx] + ', Best Balanced' if plot_labels[best_balanced_idx] else 'Best Ductility, Best Balanced'
                    elif best_balanced_idx == best_cost_idx:
                        plot_labels[best_balanced_idx] = plot_labels[best_balanced_idx] + ', Best Balanced' if plot_labels[best_balanced_idx] else 'Lowest Cost, Best Balanced'
                    elif best_balanced_idx == best_co2_idx:
                        plot_labels[best_balanced_idx] = plot_labels[best_balanced_idx] + ', Best Balanced' if plot_labels[best_balanced_idx] else 'Lowest CO₂, Best Balanced'

                    df_fig1 = pd.DataFrame({
                        'Peak_Stress_MPa': stress_vals,
                        'Normalized_Ductility_Area': ductility_vals,
                        'Cost_USD_per_m3': cost_vals,
                        'Label': plot_labels
                    })
                    df_fig2 = pd.DataFrame({
                        'Peak_Stress_MPa': stress_vals,
                        'Normalized_Ductility_Area': ductility_vals,
                        'CO2_kg_per_m3': co2_vals,
                        'Label': plot_labels
                    })
                    df_fig3 = pd.DataFrame({
                        'Peak_Stress_MPa': stress_vals,
                        'Cost_USD_per_m3': cost_vals,
                        'CO2_kg_per_m3': co2_vals,
                        'Label': plot_labels
                    })
                    df_fig4 = pd.DataFrame({
                        'Normalized_Ductility_Area': ductility_vals,
                        'Cost_USD_per_m3': cost_vals,
                        'CO2_kg_per_m3': co2_vals,
                        'Label': plot_labels
                    })
                    # 4D scatter data: Strength, Ductility, CO₂, color mapped to Cost
                    df_fig5 = pd.DataFrame({
                        'Peak_Stress_MPa': stress_vals,
                        'Normalized_Ductility_Area': ductility_vals,
                        'CO2_kg_per_m3': co2_vals,
                        'Cost_USD_per_m3': cost_vals,
                        'ColorValue': cost_vals,  # Use cost to color map
                        'Label': plot_labels
                    })
                    
                    excel_3d_path = os.path.join(save_dir, 'pareto_front_3d_data.xlsx')
                    with pd.ExcelWriter(excel_3d_path, engine='openpyxl', mode='w') as writer:
                        df_fig1.to_excel(writer, sheet_name='Fig1_Stress_Duct_Cost', index=False)
                        df_fig2.to_excel(writer, sheet_name='Fig2_Stress_Duct_CO2', index=False)
                        df_fig3.to_excel(writer, sheet_name='Fig3_Stress_Cost_CO2', index=False)
                        df_fig4.to_excel(writer, sheet_name='Fig4_Duct_Cost_CO2', index=False)
                        df_fig5.to_excel(writer, sheet_name='Fig5_4DScatter', index=False)
                    print(f"3D/4D plot data has been saved to: {excel_3d_path}")
                except Exception as e:
                    print(f"Warning: Error saving 3D Pareto plot data to Excel: {e}")
                
                # Create 3D plot: Peak stress vs Normalized ductility area vs Cost/CO₂
                fig = plt.figure(figsize=(18, 14))
                
                # Prepare all evaluated points data（if available）
                if len(all_evaluated_F_original) > 0 and all_results_df is not None:
                    pareto_mask = all_results_df['Is_Pareto'].values
                    closeness_values = all_results_df['Closeness'].values
                    
                    # First 3D plot: Peak stress vs Normalized ductility area vs Cost
                    ax1 = fig.add_subplot(231, projection='3d')
                    # Only plot Pareto solutions（using Closeness to color）
                    scatter1 = ax1.scatter(
                        all_evaluated_F_original[pareto_mask, 0],  # Peak stress
                        all_evaluated_F_original[pareto_mask, 1],  # Normalized ductility area
                        all_evaluated_F_original[pareto_mask, 2],  # Cost
                        c=closeness_values[pareto_mask],  # Use Closeness to color
                        cmap='viridis',
                        alpha=0.8,
                        s=80,
                        marker='*',
                        edgecolors='black',
                        linewidths=0.5,
                        label='Pareto front'
                    )
                else:
                    # First 3D plot: Peak stress vs Normalized ductility area vs Cost
                    ax1 = fig.add_subplot(231, projection='3d')
                    scatter1 = ax1.scatter(
                        pareto_objectives_original[:, 0],  # Peak stress
                        pareto_objectives_original[:, 1],  # Normalized ductility area
                        pareto_objectives_original[:, 2],  # Cost
                        c=pareto_objectives_original[:, 2],  # Use cost to color
                        cmap='Reds',
                        alpha=0.6,
                        s=50
                    )
                # Mark the best points（if there is data for all evaluated points, do not mark, because Pareto solutions are already marked with colors）
                if len(all_evaluated_F_original) == 0 or all_results_df is None:
                    ax1.scatter(
                        pareto_objectives_original[best_stress_idx, 0],
                        pareto_objectives_original[best_stress_idx, 1],
                        pareto_objectives_original[best_stress_idx, 2],
                        c='red', marker='*', s=300, label='Best strength', edgecolors='black', linewidths=1
                    )
                    ax1.scatter(
                        pareto_objectives_original[best_toughness_idx, 0],
                        pareto_objectives_original[best_toughness_idx, 1],
                        pareto_objectives_original[best_toughness_idx, 2],
                        c='blue', marker='*', s=300, label='Best ductility', edgecolors='black', linewidths=1
                    )
                    ax1.scatter(
                        pareto_objectives_original[best_cost_idx, 0],
                        pareto_objectives_original[best_cost_idx, 1],
                        pareto_objectives_original[best_cost_idx, 2],
                        c='green', marker='*', s=300, label='Lowest cost', edgecolors='black', linewidths=1
                    )
                    ax1.legend(fontsize=8)
                    plt.colorbar(scatter1, ax=ax1, label='Cost ($/m³)', shrink=0.6)
                else:
                    ax1.legend(fontsize=8)
                    plt.colorbar(scatter1, ax=ax1, label='Closeness', shrink=0.6)
                ax1.set_xlabel('Compressive strength (MPa)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Normalized Ductility Area', fontsize=12, fontweight='bold')
                ax1.set_zlabel('Price ($/m³)', fontsize=12, fontweight='bold')
                ax1.set_title('(b) Tri-objective', fontsize=14, fontweight='bold')

                # New 4D scatter plot（Strength / Ductility / CO2, color mapped to Closeness, point size fixed）
                try:
                    fig4d = plt.figure(figsize=(9, 7))
                    ax4d = fig4d.add_subplot(111, projection='3d')
                    if len(all_evaluated_F_original) > 0 and all_results_df is not None:
                        # Only plot Pareto solutions（using Closeness to color）
                        scatter4d = ax4d.scatter(
                            all_evaluated_F_original[pareto_mask, 0],  # Strength
                            all_evaluated_F_original[pareto_mask, 1],  # Ductility
                            all_evaluated_F_original[pareto_mask, 3],  # CO2
                            c=closeness_values[pareto_mask],  # Color mapped to Closeness
                            cmap='viridis',
                            s=80,
                            alpha=0.8,
                            marker='*',
                            edgecolors='black',
                            linewidths=0.5,
                            label='Pareto front'
                        )
                        ax4d.legend(fontsize=10)
                        cbar = plt.colorbar(scatter4d, ax=ax4d, pad=0.12, shrink=0.8)
                        cbar.set_label('Closeness', fontname='Arial', fontsize=11, fontweight='bold')
                    else:
                        scatter4d = ax4d.scatter(
                            stress_vals,              # Strength
                            ductility_vals,           # Ductility
                            co2_vals,                 # CO2
                            c=cost_vals,              # Color mapped to cost
                            cmap='viridis_r',         # Darker colors represent higher cost
                            s=50,
                            alpha=0.75
                        )
                        cbar = plt.colorbar(scatter4d, ax=ax4d, pad=0.12, shrink=0.8)
                        cbar.set_label('Cost ($/m³)', fontname='Arial', fontsize=11, fontweight='bold')
                    ax4d.set_xlabel('Compressive Strength (MPa)', fontname='Arial', fontsize=14, fontweight='bold')
                    ax4d.set_ylabel('Normalized Ductility Area', fontname='Arial', fontsize=14, fontweight='bold')
                    ax4d.set_zlabel('Carbon Emission (kgCO₂/m³)', fontname='Arial', fontsize=14, fontweight='bold')
                    if len(all_evaluated_F_original) > 0 and all_results_df is not None:
                        ax4d.set_title('(b) Tri-objective', fontname='Arial', fontsize=16, fontweight='bold')
                    else:
                        ax4d.set_title('4D Scatter: Strength / Ductility / CO₂ (Color=Cost)', fontname='Arial', fontsize=16, fontweight='bold')
                    for label in ax4d.get_xticklabels() + ax4d.get_yticklabels() + ax4d.get_zticklabels():
                        label.set_fontname('Arial')
                        label.set_fontsize(11)
                        label.set_fontweight('bold')
                    ax4d.tick_params(axis='both', width=1.8)
                    ax4d.tick_params(axis='z', width=1.8)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontname('Arial')
                        t.set_fontsize(10)
                        t.set_fontweight('bold')
                    scatter4d_path = os.path.join(save_dir, 'pareto_front_4d_scatter_cost_color.png')
                    plt.savefig(scatter4d_path, dpi=300, bbox_inches='tight')
                    plt.close(fig4d)
                    print(f"4D scatter plot has been saved to: {scatter4d_path}")
                except Exception as e:
                    print(f"Warning: Error plotting 4D scatter plot: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Second 3D plot: Peak stress vs Normalized ductility area vs CO₂
                ax2 = fig.add_subplot(232, projection='3d')
                if len(all_evaluated_F_original) > 0 and all_results_df is not None:
                    # Only plot Pareto solutions（using Closeness to color）
                    scatter2 = ax2.scatter(
                        all_evaluated_F_original[pareto_mask, 0],
                        all_evaluated_F_original[pareto_mask, 1],
                        all_evaluated_F_original[pareto_mask, 3],
                        c=closeness_values[pareto_mask],
                        cmap='viridis',
                        alpha=0.8,
                        s=80,
                        marker='*',
                        edgecolors='black',
                        linewidths=0.5
                    )
                    plt.colorbar(scatter2, ax=ax2, label='Closeness', shrink=0.6)
                else:
                    scatter2 = ax2.scatter(
                        pareto_objectives_original[:, 0],
                        pareto_objectives_original[:, 1],
                        pareto_objectives_original[:, 3],
                        c=pareto_objectives_original[:, 3],
                        cmap='Greens',
                        alpha=0.6,
                        s=50
                    )
                    ax2.scatter(
                        pareto_objectives_original[best_stress_idx, 0],
                        pareto_objectives_original[best_stress_idx, 1],
                        pareto_objectives_original[best_stress_idx, 3],
                        c='red', marker='*', s=300, label='Best strength', edgecolors='black', linewidths=1
                    )
                    ax2.scatter(
                        pareto_objectives_original[best_toughness_idx, 0],
                        pareto_objectives_original[best_toughness_idx, 1],
                        pareto_objectives_original[best_toughness_idx, 3],
                        c='blue', marker='*', s=300, label='Best ductility', edgecolors='black', linewidths=1
                    )
                    ax2.scatter(
                        pareto_objectives_original[best_co2_idx, 0],
                        pareto_objectives_original[best_co2_idx, 1],
                        pareto_objectives_original[best_co2_idx, 3],
                        c='orange', marker='*', s=300, label='Lowest CO₂', edgecolors='black', linewidths=1
                    )
                    ax2.legend(fontsize=8)
                    plt.colorbar(scatter2, ax=ax2, label='CO$_2$ Emissions (kg/m³)', shrink=0.6)
                ax2.set_xlabel('Compressive strength (MPa)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Normalized Ductility Area', fontsize=12, fontweight='bold')
                ax2.set_zlabel('Carbon emission (kgCO$_2$/m³)', fontsize=12, fontweight='bold')
                ax2.set_title('3D Pareto Front: Strength vs Ductility vs CO$_2$', fontsize=11)
                
                # Third 3D plot: Peak stress vs Cost vs CO₂
                ax3 = fig.add_subplot(233, projection='3d')
                scatter3 = ax3.scatter(
                    pareto_objectives_original[:, 0],
                    pareto_objectives_original[:, 2],
                    pareto_objectives_original[:, 3],
                    c=pareto_objectives_original[:, 0],
                    cmap='coolwarm',
                    alpha=0.6,
                    s=50
                )
                ax3.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax3.set_ylabel('Cost ($/m³)', fontsize=10)
                ax3.set_zlabel('CO$_2$ Emissions (kg/m³)', fontsize=10)
                ax3.set_title('3D Pareto Front: Strength vs Cost vs CO$_2$', fontsize=11)
                plt.colorbar(scatter3, ax=ax3, label='Peak Stress (MPa)', shrink=0.6)
                
                # Fourth 3D plot: Normalized ductility area vs Cost vs CO₂
                ax4 = fig.add_subplot(234, projection='3d')
                scatter4 = ax4.scatter(
                    pareto_objectives_original[:, 1],
                    pareto_objectives_original[:, 2],
                    pareto_objectives_original[:, 3],
                    c=pareto_objectives_original[:, 1],
                    cmap='viridis',
                    alpha=0.6,
                    s=50
                )
                ax4.set_xlabel('Normalized Ductility Area', fontsize=10)
                ax4.set_ylabel('Cost ($/m³)', fontsize=10)
                ax4.set_zlabel('CO$_2$ Emissions (kg/m³)', fontsize=10)
                ax4.set_title('3D Pareto Front: Ductility vs Cost vs CO$_2$', fontsize=11)
                plt.colorbar(scatter4, ax=ax4, label='Normalized Ductility Area', shrink=0.6)
                
                # Fifth 3D plot: Peak stress vs Normalized ductility area vs Cost（using CO₂ to color）
                ax5 = fig.add_subplot(235, projection='3d')
                scatter5 = ax5.scatter(
                    pareto_objectives_original[:, 0],
                    pareto_objectives_original[:, 1],
                    pareto_objectives_original[:, 2],
                    c=pareto_objectives_original[:, 3],
                    cmap='YlOrRd',
                    alpha=0.6,
                    s=50
                )
                ax5.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax5.set_ylabel('Normalized Ductility Area', fontsize=10)
                ax5.set_zlabel('Cost ($/m³)', fontsize=10)
                ax5.set_title('3D Pareto Front: Strength vs Ductility vs Cost\n(Colored by CO$_2$)', fontsize=11)
                plt.colorbar(scatter5, ax=ax5, label='CO$_2$ Emissions (kg/m³)', shrink=0.6)
                
                # Sixth 3D plot: Peak stress vs Cost vs CO₂（using Normalized ductility area to color）
                ax6 = fig.add_subplot(236, projection='3d')
                scatter6 = ax6.scatter(
                    pareto_objectives_original[:, 0],
                    pareto_objectives_original[:, 2],
                    pareto_objectives_original[:, 3],
                    c=pareto_objectives_original[:, 1],
                    cmap='plasma',
                    alpha=0.6,
                    s=50
                )
                ax6.set_xlabel('Peak Stress (MPa)', fontsize=10)
                ax6.set_ylabel('Cost ($/m³)', fontsize=10)
                ax6.set_zlabel('CO$_2$ Emissions (kg/m³)', fontsize=10)
                ax6.set_title('3D Pareto Front: Strength vs Cost vs CO$_2$\n(Colored by Ductility)', fontsize=11)
                plt.colorbar(scatter6, ax=ax6, label='Normalized Ductility Area', shrink=0.6)
            
            plt.tight_layout()
            plot_3d_path = os.path.join(save_dir, 'pareto_front_3d.png')
            plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
            print(f"3D Pareto front plot has been saved to: {plot_3d_path}")
            plt.close()

            # Comment out the separate 3D subplots, the main 3D plot already contains all information
            #（The separate save code for Figure 1-4 has been removed, and the main 3D plot pareto_front_3d.png already contains all information）
            
        except Exception as e:
            print(f"Warning: Error plotting 3D Pareto front: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error plotting Pareto front: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot the four-objective parallel coordinate plot（similar to Figure c in the paper）
    if not combine_cost_co2:
        print("\n8.6. Plotting the four-objective parallel coordinate plot...")
        try:
            # Use a more professional color scheme and a larger canvas
            fig, ax = plt.subplots(figsize=(16, 10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Four objectives
            # Rearrange the order to match the paper: Price, Strength, CO₂, Ductility
            # The order of pareto_objectives_original is: [Peak_Stress, Ductility, Cost, CO₂]
            # Need to rearrange to: [Cost, Peak_Stress, CO₂, Ductility]
            objectives_reordered = np.zeros_like(pareto_objectives_original)
            objectives_reordered[:, 0] = pareto_objectives_original[:, 2]  # Cost -> position 0
            objectives_reordered[:, 1] = pareto_objectives_original[:, 0]  # Peak_Stress -> position 1
            objectives_reordered[:, 2] = pareto_objectives_original[:, 3]  # CO₂ -> position 2
            objectives_reordered[:, 3] = pareto_objectives_original[:, 1]  # Ductility -> position 3
            
            objectives = objectives_reordered
            n_solutions = len(objectives)
            n_objectives = 4
            print(f"\n Parallel coordinate plot: {n_solutions} Pareto solutions")
            
            # Objective names and units（according to the paper order, using the correct superscripts）
            # Use matplotlib's math mode to correctly display superscripts and subscripts
            obj_names = [
                'Price ($/m$^3$)',
                'Compressive strength (MPa)',
                'Carbon emission (kgCO$_2$/m$^3$)',
                'Normalized Ductility Area'
            ]
            
            # Normalize data to 0-1 range（for plotting）
            normalized_data = np.zeros_like(objectives)
            obj_ranges = []
            for i in range(n_objectives):
                min_val = objectives[:, i].min()
                max_val = objectives[:, i].max()
                obj_ranges.append((min_val, max_val))
                if max_val > min_val:
                    normalized_data[:, i] = (objectives[:, i] - min_val) / (max_val - min_val)
                else:
                    normalized_data[:, i] = 0.5
            
            # Set x-axis positions（four vertical axes）
            x_positions = np.arange(n_objectives)
            
            # Draw background grid（finer）
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='major')
            ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.3, which='minor')
            ax.set_axisbelow(True)
            
            # Add more horizontal grid lines on the compressive strength axis（index 1）, help see the distribution
            strength_axis_idx = 1
            if strength_axis_idx < len(obj_ranges):
                strength_min, strength_max = obj_ranges[strength_axis_idx]
                strength_range = strength_max - strength_min
                if strength_range > 1e-10:
                    # Add more grid lines（corresponding to the compressive strength axis ticks）
                    # Use the same position as the tick labels
                    strength_norm_positions = np.linspace(0.1, 0.9, 9)
                    for norm_pos in strength_norm_positions:
                        # Add horizontal grid lines at the compressive strength axis position
                        ax.axhline(y=norm_pos, xmin=strength_axis_idx-0.1, xmax=strength_axis_idx+0.1,
                                  color='#BDC3C7', linestyle='--', linewidth=0.8, alpha=0.3, zorder=0)
            
            # Draw vertical axis lines（thicker, more visible）
            for x_pos in x_positions:
                ax.axvline(x=x_pos, color='#2C3E50', linewidth=2.5, alpha=0.8, zorder=1)
            
            # Add density distribution visualization on the compressive strength axis（on the left side of the axis）
            strength_axis_idx = 1
            if strength_axis_idx < len(obj_ranges):
                # Get the normalized value of compressive strength
                strength_normalized = normalized_data[:, strength_axis_idx]
                
                # Create density histogram（on the left side of the axis）
                # Use a smaller width, not blocking main content
                hist_width = 0.08
                hist_x_start = strength_axis_idx - hist_width - 0.02
                
                # Calculate the histogram
                hist_bins = 15  # Use 15 bins to show the distribution
                hist, bin_edges = np.histogram(strength_normalized, bins=hist_bins, range=(0, 1))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Normalize the histogram height (make it not exceed hist_width)
                max_hist = hist.max() if hist.max() > 0 else 1
                hist_normalized = hist / max_hist * hist_width
                
                # Draw horizontal bar chart（from the axis to the left）
                for i, (center, height) in enumerate(zip(bin_centers, hist_normalized)):
                    if height > 0:
                        ax.barh(center, height, left=hist_x_start, height=(bin_edges[1]-bin_edges[0]),
                               color='#3498DB', alpha=0.4, zorder=1, edgecolor='none')
            
            # Use smooth curves to draw all Pareto solutions' parallel coordinate lines
            from scipy.interpolate import make_interp_spline
            
            # Create interpolation points for smooth curves（add more points between each two axes）
            x_smooth = np.linspace(0, n_objectives - 1, 100)
            
            # Draw all Pareto solutions' parallel coordinate lines（using gray）
            # Increase visibility: increase alpha and linewidth, make the lines more visible
            for i in range(n_solutions):
                if i != best_balanced_idx:
                    # Use gray to draw other solutions, but increase visibility
                    line_color = '#808080'  # Gray
                    # Use spline interpolation to create smooth curves
                    try:
                        spline = make_interp_spline(x_positions, normalized_data[i, :], k=3)
                        y_smooth = spline(x_smooth)
                        # Increase alpha and linewidth, make the lines more visible
                        ax.plot(x_smooth, y_smooth, 
                               color=line_color, alpha=0.6, linewidth=1.2, zorder=2)
                    except:
                        # If interpolation fails, use a straight line
                        ax.plot(x_positions, normalized_data[i, :], 
                               color=line_color, alpha=0.6, linewidth=1.2, zorder=2)
            
            # Draw the best balanced solution（using smooth curves, bright red, thicker lines）
            optimal_color = '#E74C3C'  # Bright red
            try:
                spline_optimal = make_interp_spline(x_positions, normalized_data[best_balanced_idx, :], k=3)
                y_smooth_optimal = spline_optimal(x_smooth)
                ax.plot(x_smooth, y_smooth_optimal, 
                       color=optimal_color, linewidth=4.5, label='Optimal Solution', 
                       zorder=15, alpha=0.95)
            except:
                # If interpolation fails, use a straight line
                ax.plot(x_positions, normalized_data[best_balanced_idx, :], 
                       color=optimal_color, linewidth=4.5, label='Optimal Solution', 
                       zorder=15, alpha=0.95)
            
            # Mark the best balanced solution's point（larger, more noticeable）
            ax.scatter(x_positions, normalized_data[best_balanced_idx, :], 
                      color=optimal_color, s=180, zorder=16, 
                      edgecolors='white', linewidths=2.5, alpha=0.95)
            
            # Set x-axis labels and ticks (Arial bold, increase font size by 2)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(obj_names, fontsize=14, fontweight='bold', 
                              color='#2C3E50', fontname='Arial')
            
            # Set y-axis（normalized 0-1 range）
            ax.set_ylim(-0.08, 1.08)
            ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold', 
                         color='#2C3E50', family='Arial')
            
            # Add y-axis ticks on the left (display normalized values, Arial bold, increase font size by 2)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], 
                              fontsize=12, fontweight='bold', color='#34495E', fontname='Arial')
            ax.tick_params(axis='y', width=1.5, length=6, labelsize=12)
            
            # Add actual value labels on both sides of each vertical axis（top, bottom, and more intermediate ticks）
            for i, (obj_name, (min_val, max_val)) in enumerate(zip(obj_names, obj_ranges)):
                # Display the maximum value at the top
                ax.text(i, 1.05, f'{max_val:.1f}', ha='center', va='bottom', 
                       fontsize=11, fontweight='bold', color='#2C3E50', fontname='Arial',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', 
                               edgecolor='#BDC3C7', alpha=0.8))
                # Display the minimum value at the bottom
                ax.text(i, -0.05, f'{min_val:.1f}', ha='center', va='top', 
                       fontsize=11, fontweight='bold', color='#2C3E50', fontname='Arial',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', 
                               edgecolor='#BDC3C7', alpha=0.8))
                # Add tick values at the middle position (add more tick points, especially for compressive strength axis)
                range_val = max_val - min_val
                if range_val > 1e-10:  # Avoid division by zero
                    # For the compressive strength axis（index 1）, use more tick points
                    if i == 1:  # The compressive strength axis
                        # Use 10 tick points（0.1, 0.2, ..., 0.9）
                        norm_positions = np.linspace(0.1, 0.9, 9)
                    else:
                        # Other axes use 5 tick points（0.2, 0.4, 0.6, 0.8）
                        norm_positions = np.linspace(0.2, 0.8, 4)
                    
                    # Add tick labels on the right side of the axis
                    for norm_pos in norm_positions:
                        actual_val = min_val + norm_pos * range_val
                        # Select the appropriate format based on the value size
                        if abs(actual_val) < 0.01:
                            fmt_str = f'{actual_val:.4f}'
                        elif abs(actual_val) < 1:
                            fmt_str = f'{actual_val:.3f}'
                        elif abs(actual_val) < 100:
                            fmt_str = f'{actual_val:.2f}'
                        else:
                            fmt_str = f'{actual_val:.1f}'
                        ax.text(i + 0.08, norm_pos, fmt_str, ha='left', va='center',
                               fontsize=9, fontweight='bold', color='#34495E', fontname='Arial',
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                       edgecolor='#BDC3C7', alpha=0.7, linewidth=0.5))
            
            # Set tick label font size (Arial bold, increase font size by 2)
            ax.tick_params(axis='both', labelsize=12, width=1.5, length=6)
            ax.tick_params(axis='x', pad=15)
            # Ensure x-axis tick labels use Arial bold
            for label in ax.get_xticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')
                label.set_fontsize(14)
            # Ensure y-axis tick labels use Arial bold
            for label in ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')
                label.set_fontsize(12)
            
            # Add the actual values of the best balanced solution（more beautiful text box）
            best_solution_values = objectives[best_balanced_idx, :]
            # Use matplotlib's math mode to correctly display superscripts and subscripts
            annotation_text = 'Optimal Solution:\n'
            annotation_text += f'Price: {best_solution_values[0]:.2f} $/m$^3$\n'
            annotation_text += f'Strength: {best_solution_values[1]:.2f} MPa\n'
            annotation_text += f'CO$_2$: {best_solution_values[2]:.2f} kg/m$^3$\n'
            annotation_text += f'Ductility: {best_solution_values[3]:.4f}'
            
            # Add text box in the upper right corner（more professional style）
            text_box = ax.text(0.985, 0.985, annotation_text, 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', 
                           edgecolor='#F39C12', linewidth=2, alpha=0.95),
                   family='Arial', color='#2C3E50', fontweight='bold',
                   usetex=False)  # Use matplotlib's math mode instead of LaTeX
            
            # Add legend（in the upper left corner, avoid overlapping with the text box）
            legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
                             edgecolor='#BDC3C7', facecolor='white', 
                             frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_linewidth(1.5)
            
            # Remove the top and right border lines, make the figure more concise
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#BDC3C7')
            ax.spines['bottom'].set_color('#BDC3C7')
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Comment out the old version of the parallel coordinate plot, only keep the v2 version
            # plt.tight_layout()
            # parallel_coords_path = os.path.join(save_dir, 'pareto_front_parallel_coordinates.png')
            # plt.savefig(parallel_coords_path, dpi=300, bbox_inches='tight', 
            #            facecolor='white', edgecolor='none')
            # print(f"Four-objective parallel coordinate plot has been saved to: {parallel_coords_path}")
            plt.close()
            
            # Draw another version: use the actual value range（not normalized to 0-1）
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            
            # Create independent y-axis ranges for each objective
            # Use matplotlib's subplot to create multiple y-axes
            fig2, axes = plt.subplots(1, n_objectives, figsize=(16, 8), sharey=False)
            
            # Draw for each objective
            for obj_idx in range(n_objectives):
                ax = axes[obj_idx]
                
                # Get the value range of this objective
                obj_values = objectives[:, obj_idx]
                min_val = obj_values.min()
                max_val = obj_values.max()
                range_val = max_val - min_val
                
                # Draw all solutions' lines（gray）
                for i in range(n_solutions):
                    if i != best_balanced_idx:
                        # Calculate the normalized position of this solution on all objectives（for x-axis）
                        x_coords = np.arange(n_objectives)
                        y_coords = []
                        for j in range(n_objectives):
                            obj_j_min = objectives[:, j].min()
                            obj_j_max = objectives[:, j].max()
                            if obj_j_max > obj_j_min:
                                y_coords.append((objectives[i, j] - obj_j_min) / (obj_j_max - obj_j_min))
                            else:
                                y_coords.append(0.5)
                        
                        # Only draw the lines on the current objective axis
                        if obj_idx < n_objectives - 1:
                            # Draw the lines to the next axis
                            ax.plot([obj_idx, obj_idx + 1], 
                                   [y_coords[obj_idx], y_coords[obj_idx + 1]],
                                   color='gray', alpha=0.3, linewidth=0.8)
                        else:
                            # The last axis, only show the points
                            ax.scatter([obj_idx], [y_coords[obj_idx]], 
                                     color='gray', alpha=0.3, s=20)
                
                # Draw the best balanced solution's line（blue thick line）
                best_y_coords = []
                for j in range(n_objectives):
                    obj_j_min = objectives[:, j].min()
                    obj_j_max = objectives[:, j].max()
                    if obj_j_max > obj_j_min:
                        best_y_coords.append((objectives[best_balanced_idx, j] - obj_j_min) / (obj_j_max - obj_j_min))
                    else:
                        best_y_coords.append(0.5)
                
                if obj_idx < n_objectives - 1:
                    ax.plot([obj_idx, obj_idx + 1], 
                           [best_y_coords[obj_idx], best_y_coords[obj_idx + 1]],
                           color='blue', linewidth=3, zorder=10)
                ax.scatter([obj_idx], [best_y_coords[obj_idx]], 
                          color='blue', s=150, zorder=11, 
                          edgecolors='darkblue', linewidths=2)
                
                # Set y-axis ticks to actual values
                y_ticks = np.linspace(0, 1, 5)
                y_tick_labels = [f'{min_val + tick * range_val:.1f}' for tick in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels, fontsize=10)
                ax.set_ylim(-0.05, 1.05)
                
                # Set x-axis
                ax.set_xlim(-0.5, n_objectives - 0.5)
                ax.set_xticks([obj_idx])
                ax.set_xticklabels([obj_names[obj_idx]], fontsize=12, fontweight='bold', rotation=0)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axvline(x=obj_idx, color='black', linewidth=2, alpha=0.5)
            
            # Hide the x-axis labels of the intermediate subplots（only show on the first and last one）
            for i in range(1, n_objectives - 1):
                axes[i].set_xticklabels([])
            
            # Set the overall title
            fig2.suptitle('Tetra-objective Optimization Results: RAC Mixture Design\n(Parallel Coordinates Plot)', 
                         fontsize=16, fontweight='bold', y=1.02)
            
            # Add y-axis labels（only on the first subplot）
            axes[0].set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            parallel_coords_path2 = os.path.join(save_dir, 'pareto_front_parallel_coordinates_v2.png')
            plt.savefig(parallel_coords_path2, dpi=300, bbox_inches='tight')
            print(f"Four-objective parallel coordinate plot（version 2）has been saved to: {parallel_coords_path2}")
            plt.close()
            
        except Exception as e:
            print(f"Warning: error drawing parallel coordinate plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Save the optimization results summary
    summary = {
        'n_generations': n_generations,
        'pop_size': pop_size,
        'n_pareto_solutions': len(pareto_solutions),
        'combine_cost_co2': combine_cost_co2,
        'objectives': {
            'peak_stress': {
                'min': float(np.min(pareto_objectives_original[:, 0])),
                'max': float(np.max(pareto_objectives_original[:, 0])),
                'mean': float(np.mean(pareto_objectives_original[:, 0]))
            },
            'normalized_ductility_area': {
                'min': float(np.min(pareto_objectives_original[:, 1])),
                'max': float(np.max(pareto_objectives_original[:, 1])),
                'mean': float(np.mean(pareto_objectives_original[:, 1]))
            },
            'cost': {
                'min': float(np.min(pareto_objectives_original[:, 2])),
                'max': float(np.max(pareto_objectives_original[:, 2])),
                'mean': float(np.mean(pareto_objectives_original[:, 2]))
            }
        }
    }
    
    # If using 4 objectives, add carbon emission statistics
    if not combine_cost_co2:
        summary['objectives']['co2'] = {
            'min': float(np.min(pareto_objectives_original[:, 3])),
            'max': float(np.max(pareto_objectives_original[:, 3])),
            'mean': float(np.mean(pareto_objectives_original[:, 3]))
        }
    else:
        # If using 3 objectives, still calculate carbon emission statistics（from the recalculated values）
        co2s = [calculate_carbon_emissions(problem._array_to_dict(sol)) for sol in pareto_solutions]
        summary['objectives']['co2'] = {
            'min': float(np.min(co2s)),
            'max': float(np.max(co2s)),
            'mean': float(np.mean(co2s))
        }
        summary['objectives']['combined_cost'] = {
            'min': float(np.min(pareto_objectives_original[:, 2])),
            'max': float(np.max(pareto_objectives_original[:, 2])),
            'mean': float(np.mean(pareto_objectives_original[:, 2]))
        }
    
    summary_path = os.path.join(save_dir, 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Optimization summary has been saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("Multi-objective optimization completed!")
    print("="*80)
    print(f"Found {len(pareto_solutions)} Pareto optimal solutions")
    print(f"\nObjective function range:")
    print(f"  1. Peak stress（strength）: {np.min(pareto_objectives_original[:, 0]):.2f} - {np.max(pareto_objectives_original[:, 0]):.2f} MPa")
    print(f"  2. Normalized ductility area（ductility）: {np.min(pareto_objectives_original[:, 1]):.4f} - {np.max(pareto_objectives_original[:, 1]):.4f}")
    print(f" Note: The larger the area, the higher the ductility (normalized area under the curve, the critical point is the residual 20% of the peak stress descending segment)")
    
    if combine_cost_co2:
        print(f"  3. Combined cost（cost + carbon emission）: {np.min(pareto_objectives_original[:, 2]):.2f} - {np.max(pareto_objectives_original[:, 2]):.2f} $/m³")
        # Recalculate cost and carbon emission for display
        costs = [calculate_material_cost(problem._array_to_dict(sol)) for sol in pareto_solutions]
        co2s = [calculate_carbon_emissions(problem._array_to_dict(sol)) for sol in pareto_solutions]
        print(f"    Among them:")
        print(f"      - Cost: {np.min(costs):.2f} - {np.max(costs):.2f} $/m³")
        print(f"      - Carbon emission: {np.min(co2s):.2f} - {np.max(co2s):.2f} kg CO₂/m³")
    else:
        print(f"  3. Cost: {np.min(pareto_objectives_original[:, 2]):.2f} - {np.max(pareto_objectives_original[:, 2]):.2f} $/m³")
        print(f"  4. Carbon emission: {np.min(pareto_objectives_original[:, 3]):.2f} - {np.max(pareto_objectives_original[:, 3]):.2f} kg CO₂/m³")
    print("="*80)
    
    # Return the results, including all evaluated points' data
    return res, pareto_solutions, pareto_objectives_original, all_evaluated_X, all_evaluated_F_original


def plot_parallel_coordinates_from_file(excel_file_path, save_path=None):
    """
    Read the Pareto optimal solutions from the saved Excel file, and draw the four-objective parallel coordinate plot
    
    Args:
        excel_file_path: The Excel file path of the Pareto optimal solutions（pareto_optimal_solutions.xlsx）
        save_path: The path to save the picture（if None, save to the same directory as the Excel file）
    """
    # Read the Excel file
    df = pd.read_excel(excel_file_path)
    
    # Extract the values of the four objectives
    # Note: The column names in the Excel file may be different, so adjust according to actual situation
    peak_stress = df['Peak_Stress_MPa'].values
    ductility = df['Normalized_Ductility_Area'].values
    cost = df['Cost_USD_per_m3'].values
    co2 = df['CO2_kg_per_m3'].values
    
    # Rearrange the order to match the paper: Price, Strength, CO₂, Ductility
    objectives = np.column_stack([cost, peak_stress, co2, ductility])
    
    # Objective names and units（according to the paper order, use the correct superscripts）
    # Use matplotlib's math mode to correctly display the superscripts and subscripts
    obj_names = [
        'Price ($/m$^3$)',
        'Compressive strength (MPa)',
        'Carbon emission (kgCO$_2$/m$^3$)',
        'Normalized Ductility Area'
    ]
    
    n_solutions = len(objectives)
    n_objectives = 4
    
    # Normalize the data to the 0-1 range
    normalized_data = np.zeros_like(objectives)
    for i in range(n_objectives):
        min_val = objectives[:, i].min()
        max_val = objectives[:, i].max()
        if max_val > min_val:
            normalized_data[:, i] = (objectives[:, i] - min_val) / (max_val - min_val)
        else:
            normalized_data[:, i] = 0.5
    
    # Use the TOPSIS method to find the best balanced solution
    # Normalize each objective
    stress_norm = (objectives[:, 1] - objectives[:, 1].min()) / (objectives[:, 1].max() - objectives[:, 1].min() + 1e-10)
    ductility_norm = (objectives[:, 3] - objectives[:, 3].min()) / (objectives[:, 3].max() - objectives[:, 3].min() + 1e-10)
    cost_norm = 1 - (objectives[:, 0] - objectives[:, 0].min()) / (objectives[:, 0].max() - objectives[:, 0].min() + 1e-10)
    co2_norm = 1 - (objectives[:, 2] - objectives[:, 2].min()) / (objectives[:, 2].max() - objectives[:, 2].min() + 1e-10)
    
    ideal_solution = np.array([1.0, 1.0, 1.0, 1.0])
    nadir_solution = np.array([0.0, 0.0, 0.0, 0.0])
    
    distances_to_ideal = np.sqrt(
        (cost_norm - ideal_solution[0])**2 +
        (stress_norm - ideal_solution[1])**2 +
        (co2_norm - ideal_solution[2])**2 +
        (ductility_norm - ideal_solution[3])**2
    )
    distances_to_nadir = np.sqrt(
        (cost_norm - nadir_solution[0])**2 +
        (stress_norm - nadir_solution[1])**2 +
        (co2_norm - nadir_solution[2])**2 +
        (ductility_norm - nadir_solution[3])**2
    )
    
    topsis_scores = distances_to_nadir / (distances_to_ideal + distances_to_nadir + 1e-10)
    best_balanced_idx = np.argmax(topsis_scores)
    
    # Draw the parallel coordinate plot（using the improved style）
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Calculate the objective range
    obj_ranges = []
    for i in range(n_objectives):
        min_val = objectives[:, i].min()
        max_val = objectives[:, i].max()
        obj_ranges.append((min_val, max_val))
    
    # Set the x-axis position（four vertical axes）
    x_positions = np.arange(n_objectives)
    
    # Draw the background grid（finer）
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='major')
    ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.3, which='minor')
    ax.set_axisbelow(True)
    
    # Draw the vertical axis lines（thicker, more obvious）
    for x_pos in x_positions:
        ax.axvline(x=x_pos, color='#2C3E50', linewidth=2.5, alpha=0.8, zorder=1)
    
    # Use smooth curves to draw the parallel coordinate lines of all Pareto solutions
    from scipy.interpolate import make_interp_spline
    
    # Create the interpolation points for the smooth curves（add more points between each two axes）
    x_smooth = np.linspace(0, n_objectives - 1, 100)
    
    # Draw the parallel coordinate lines of all Pareto solutions（using gray）
    for i in range(n_solutions):
        if i != best_balanced_idx:
            # Use gray to draw the other solutions
            line_color = '#808080'  # gray
            # Use the spline interpolation to create the smooth curves
            try:
                spline = make_interp_spline(x_positions, normalized_data[i, :], k=3)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, 
                       color=line_color, alpha=0.4, linewidth=0.8, zorder=2)
            except:
                # If the interpolation fails, use the straight line
                ax.plot(x_positions, normalized_data[i, :], 
                       color=line_color, alpha=0.4, linewidth=0.8, zorder=2)
    
    # Draw the best balanced solution（using the smooth curves, bright red, thicker line）
    optimal_color = '#E74C3C'  # Bright red
    try:
        spline_optimal = make_interp_spline(x_positions, normalized_data[best_balanced_idx, :], k=3)
        y_smooth_optimal = spline_optimal(x_smooth)
        ax.plot(x_smooth, y_smooth_optimal, 
               color=optimal_color, linewidth=4.5, label='Optimal Solution', 
               zorder=15, alpha=0.95)
    except:
        # If the interpolation fails, use the straight line
        ax.plot(x_positions, normalized_data[best_balanced_idx, :], 
               color=optimal_color, linewidth=4.5, label='Optimal Solution', 
               zorder=15, alpha=0.95)
    
    # Mark the best balanced solution's points（larger, more obvious）
    ax.scatter(x_positions, normalized_data[best_balanced_idx, :], 
              color=optimal_color, s=180, zorder=16, 
              edgecolors='white', linewidths=2.5, alpha=0.95)
    
    # Set the x-axis labels and ticks (Arial bold, increase font size by 2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(obj_names, fontsize=18, fontweight='bold', 
                      color='#2C3E50', fontname='Arial')
    
    # Set the y-axis（the 0-1 range after normalization）
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel('Normalized Value', fontsize=15, fontweight='bold', 
                 color='#2C3E50', family='Arial')
    
    # Add the y-axis ticks on the left (display the normalized values, Arial bold, increase font size by 2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], 
                      fontsize=15, fontweight='bold', color='#34495E', fontname='Arial')
    ax.tick_params(axis='y', width=1.5, length=6, labelsize=15)
    
    # Add the actual value labels on both sides of each vertical axis (top, bottom and more intermediate ticks)
    for i, (obj_name, (min_val, max_val)) in enumerate(zip(obj_names, obj_ranges)):
        # Display the maximum value on the top
        ax.text(i, 1.05, f'{max_val:.1f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='#2C3E50', fontname='Arial',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', 
                       edgecolor='#BDC3C7', alpha=0.8))
        # Display the minimum value on the bottom
        ax.text(i, -0.05, f'{min_val:.1f}', ha='center', va='top', 
               fontsize=12, fontweight='bold', color='#2C3E50', fontname='Arial',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', 
                       edgecolor='#BDC3C7', alpha=0.8))
        # Add the intermediate ticks (add more tick points, especially for compressive strength axis)
        range_val = max_val - min_val
        if range_val > 1e-10:  # Avoid division by zero
            # For the compressive strength axis（index 1）, use more ticks
            if i == 1:  # The compressive strength axis
                # Use 10 ticks（0.1, 0.2, ..., 0.9）
                norm_positions = np.linspace(0.1, 0.9, 9)
            else:
                # Other axes use 5 ticks（0.2, 0.4, 0.6, 0.8）
                norm_positions = np.linspace(0.2, 0.8, 4)
            
            # Add the ticks labels on the right side of the axis
            for norm_pos in norm_positions:
                actual_val = min_val + norm_pos * range_val
                # Select the appropriate format according to the value size
                if abs(actual_val) < 0.01:
                    fmt_str = f'{actual_val:.4f}'
                elif abs(actual_val) < 1:
                    fmt_str = f'{actual_val:.3f}'
                elif abs(actual_val) < 100:
                    fmt_str = f'{actual_val:.2f}'
                else:
                    fmt_str = f'{actual_val:.1f}'
                ax.text(i + 0.08, norm_pos, fmt_str, ha='left', va='center',
                       fontsize=9, fontweight='bold', color='#34495E', fontname='Arial',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                               edgecolor='#BDC3C7', alpha=0.7, linewidth=0.5))
    
    # Set the tick labels font size (Arial bold, increase font size by 2)
    ax.tick_params(axis='both', labelsize=15, width=1.5, length=6)
    ax.tick_params(axis='x', pad=15)
    # Ensure the x-axis tick labels use Arial bold
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(18)
    # Ensure the y-axis tick labels use Arial bold
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(15)
    
    # Add the actual values of the best balanced solution to the figure（more beautiful text box）
    best_solution_values = objectives[best_balanced_idx, :]
    # Use matplotlib's math mode to correctly display the superscripts and subscripts
    annotation_text = 'Optimal Solution:\n'
    annotation_text += f'Price: {best_solution_values[0]:.2f} $/m$^3$\n'
    annotation_text += f'Strength: {best_solution_values[1]:.2f} MPa\n'
    annotation_text += f'CO$_2$: {best_solution_values[2]:.2f} kg/m$^3$\n'
    annotation_text += f'Ductility: {best_solution_values[3]:.4f}'
    
    # Add the text box in the upper right corner of the figure（more professional style）
    text_box = ax.text(0.985, 0.985, annotation_text, 
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', 
                   edgecolor='#F39C12', linewidth=2, alpha=0.95),
           family='Arial', color='#2C3E50', fontweight='bold',
           usetex=False)  # Use matplotlib's math mode instead of LaTeX
    
    # Add the legend（put it in the upper left corner, avoid overlapping with the text box）
    legend = ax.legend(loc='upper left', fontsize=13, framealpha=0.95,
                     edgecolor='#BDC3C7', facecolor='white', 
                     frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(1.5)
    
    # Remove the top and right border lines, making the figure more concise
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Determine the save path
    if save_path is None:
        excel_dir = os.path.dirname(excel_file_path)
        save_path = os.path.join(excel_dir, 'pareto_front_parallel_coordinates.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Four-objective parallel coordinate plot has been saved to: {save_path}")
    plt.close()
    
    return best_balanced_idx, objectives[best_balanced_idx, :]


if __name__ == "__main__":
    # Run the multi-objective optimization
    # Default save to the multi_objective_optimization/SAVE directory
    # Increase the number of generations and population size to get more Pareto solutions
    res, pareto_solutions, pareto_objectives, all_evaluated_X, all_evaluated_F_original = run_multi_objective_optimization(
        n_generations=200,  # Increase to 200 generations
        pop_size=1000       # Increase to 1000 individuals
        # save_dir uses default value: multi_objective_optimization/SAVE
    )
    
    print(f"\nOptimization completed!")
    print(f"  Number of Pareto optimal solutions: {len(pareto_solutions)}")
    print(f"  Total evaluation points: {len(all_evaluated_X) if len(all_evaluated_X) > 0 else 'N/A'}")
    print(f"  All evaluation points have been saved to: multi_objective_optimization/SAVE/all_evaluated_solutions.xlsx")
    
    # If you need to draw the parallel coordinate plot from the saved file, you can use:
    # plot_parallel_coordinates_from_file(
    #     r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\multi_objective_optimization\SAVE\pareto_optimal_solutions.xlsx"
    # )

