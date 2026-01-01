"""
Read all curve data from dataset_final.xlsx and plot them into multiple canvases
Each canvas has 4 rows and 4 columns (16 curves) and is saved to a folder
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_all_curves(excel_file, output_dir='save'):
    """
    Read all curves from Excel file and plot them
    
    Args:
        excel_file: Path to the Excel file
        output_dir: Output folder path (saved to 'save' folder in the script's directory by default)
    """
    print("=== Starting data reading ===")
    
    # Read stress-strain data (second worksheet)
    stress_df = pd.read_excel(excel_file, sheet_name=1)
    print(f"Data shape: {stress_df.shape}")
    print(f"Number of columns: {len(stress_df.columns)}")
    
    # The first column is strain (shared by all samples)
    strain_col = stress_df.columns[0]
    stress_cols = stress_df.columns[1:]  # Subsequent columns are stresses of different samples
    
    print(f"Strain column: {strain_col}")
    print(f"Number of stress columns: {len(stress_cols)}")
    
    # Get strain data
    strain_data = stress_df[strain_col].values
    print(f"Length of strain data: {len(strain_data)}")
    print(f"Strain range: [{strain_data.min():.6f}, {strain_data.max():.6f}]")
    
    # Create output folder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {output_path.absolute()}")
    
    # Configuration for each canvas
    rows_per_page = 4
    cols_per_page = 4
    curves_per_page = rows_per_page * cols_per_page
    
    # Calculate the number of canvases needed
    total_curves = len(stress_cols)
    num_pages = (total_curves + curves_per_page - 1) // curves_per_page  # Round up
    
    print(f"\nTotal {total_curves} curves")
    print(f"{curves_per_page} curves per canvas ({rows_per_page} rows x {cols_per_page} columns)")
    print(f"{num_pages} canvases needed")
    
    # Plot page by page
    for page in range(num_pages):
        print(f"\nPlotting canvas {page + 1}/{num_pages}...")
        
        # Create canvas
        fig, axes = plt.subplots(rows_per_page, cols_per_page, 
                                figsize=(16, 16), 
                                facecolor='white')
        
        # Ensure axes is a 2D array if there's only one row or column
        if rows_per_page == 1:
            axes = axes.reshape(1, -1)
        elif cols_per_page == 1:
            axes = axes.reshape(-1, 1)
        
        # Calculate curve range for current page
        start_idx = page * curves_per_page
        end_idx = min(start_idx + curves_per_page, total_curves)
        
        # Plot each curve on the current page
        for idx in range(start_idx, end_idx):
            # Calculate position in the canvas
            curve_idx = idx - start_idx
            row = curve_idx // cols_per_page
            col = curve_idx % cols_per_page
            ax = axes[row, col]
            
            # Get stress data
            stress_col_name = stress_cols[idx]
            stress_data = stress_df[stress_col_name].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(stress_data)
            if np.sum(valid_mask) == 0:
                ax.text(0.5, 0.5, f'Sample {idx+1}\nNo valid data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Sample {idx+1}: {stress_col_name}', fontsize=10)
                ax.grid(True, alpha=0.3)
                continue
            
            strain_valid = strain_data[valid_mask]
            stress_valid = stress_data[valid_mask]
            
            # Plot curve
            ax.plot(strain_valid, stress_valid, 'b-', linewidth=1.5, alpha=0.8)
            
            # Mark peak point
            peak_idx = np.argmax(stress_valid)
            peak_strain = strain_valid[peak_idx]
            peak_stress = stress_valid[peak_idx]
            ax.plot(peak_strain, peak_stress, 'ro', markersize=6, 
                   label=f'Peak: ({peak_strain:.4f}, {peak_stress:.2f})')
            
            # Set title and labels
            ax.set_title(f'Sample {idx+1}: {stress_col_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Strain', fontsize=9)
            ax.set_ylabel('Stress (MPa)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=7, loc='best')
            
            # Set axis ranges
            ax.set_xlim(strain_valid.min() * 0.95, strain_valid.max() * 1.05)
            ax.set_ylim(0, stress_valid.max() * 1.1)
        
        # Hide extra subplots
        for idx in range(end_idx - start_idx, curves_per_page):
            row = idx // cols_per_page
            col = idx % cols_per_page
            axes[row, col].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Stress-Strain Curves (Page {page + 1}/{num_pages}, Samples {start_idx+1}-{end_idx})', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save image
        output_file = output_path / f'curves_page_{page+1:02d}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_file}")
        
        # Close figure to free memory
        plt.close(fig)
    
    print(f"\n=== Plotting completed ===")
    print(f"All images saved to: {output_path.absolute()}")
    print(f"Total {num_pages} images generated")
    
    return output_path

if __name__ == "__main__":
    # Set file path
    excel_file = r"dataset\dataset_final.xlsx"
    # Set save path to 'save' folder in the script's directory
    current_dir = Path(__file__).parent
    output_dir = str(current_dir / "save")
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"Error: File does not exist - {excel_file}")
        print("Please check if the file path is correct")
        # Try using absolute path
        current_dir = Path(__file__).parent
        excel_file = current_dir / "dataset" / "dataset_final.xlsx"
        if excel_file.exists():
            print(f"File found: {excel_file}")
        else:
            print("Could not find the file, please check the path")
            exit(1)
    
    # Execute plotting
    print(f"Starting to process file: {excel_file}")
    plot_all_curves(str(excel_file), output_dir)
