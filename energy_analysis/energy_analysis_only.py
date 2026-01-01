"""
Special energy analysis script
Only performs energy analysis without other predictions and evaluations

Functions:
1. Load the trained BiLSTM model
2. Analyze the impact of water-cement ratio (w/c) and aggregate replacement rate (r) on energy indicators
3. Calculate two types of energy indicators simultaneously:
   - Energy indicators for the entire curve (W_u, W_ascending, W_p, η)
   - Energy indicators at 20% residual strength (based on normalized stress-strain curve, x = ε/ε_cp, y = σ/σ_cp)
4. Generate energy analysis charts and Excel files

Usage:
    python energy_analysis_only.py

Output:
    - parameter_impact_on_energy.png: 4x4 subplots showing the impact of parameters on energy indicators
    - parameter_impact_on_energy.xlsx: Excel file containing all energy indicator data
"""
import numpy as np
import pandas as pd
import torch
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Set random seeds and device
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== Import necessary classes and functions ==========
# Ensure the project root directory can be found
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent  # energy_analysis directory
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent  # Pi_BiLSTM root directory
LSTM_DIR = PROJECT_ROOT / 'LSTM'  # LSTM directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent  # constitutive_relation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import energy analysis related functions (import from LSTM directory)
sys.path.insert(0, str(LSTM_DIR))
from 训练好的模型_交叉验证 import (
    load_trained_model,
    analyze_parameter_impact_on_energy
)


def main():
    """Main function - only perform energy analysis"""
    print("="*100)
    print("="*100)
    print("Special energy analysis script")
    print("="*100)
    print("="*100)
    
     # Set file paths
    model_path = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\LSTM\SAVE\bidirectional_lstm_cv\best_model.pth"
    excel_file = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\dataset\dataset_final.xlsx"
    save_dir = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\energy_analysis\SAVE"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        print("Please run the main training script first to generate the model file")
        return None
    
    if not os.path.exists(excel_file):
        print(f"Error: Excel file not found {excel_file}")
        return None
    
    print(f"Model file：{model_path}")
    print(f"Data file：{excel_file}")
    print(f"Saving directory：{save_dir}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
         # Load the trained model
        print("\n" + "="*100)
        print("加载模型...")
        print("="*100)
        model, model_info = load_trained_model(model_path)
        
       # Perform analysis of parameter impact on energy indicators
        print("\n" + "="*100)
        print("Starting analysis of parameter impact on energy indicators...")
        print("="*100)
        print("Note: Energy analysis uses normalized stress-strain curves (x = ε/ε_cp, y = σ/σ_cp")
        print("      The 20% residual strength critical point is found in the normalized coordinate system (y = 0.2)")
        print("="*100)
        
        df_wc, df_r = analyze_parameter_impact_on_energy(
            model=model,
            model_info=model_info,
            excel_file=excel_file,
            save_dir=save_dir
        )
        
        print("\n" + "="*100)
        print("Energy analysis completed!")
        print("="*100)
        print(f"Results saved to: {save_dir}")
        print(f"  - Graph: parameter_impact_on_energy.png")
        print(f"  - Excel: parameter_impact_on_energy.xlsx")
        print("="*100)
        
        return df_wc, df_r
        
    except Exception as e:
        print(f"Error occurred during energy analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
   # Run energy analysis
    results = main()

