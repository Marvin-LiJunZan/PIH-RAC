"""
Extract peak stress/strain data and save to a new Excel file.
Index by custom specimen ID column "NO" to match the latest data layout.
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Set working directory to project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
os.chdir(project_root)

def extract_peak_data():
    """Extract peak stress/strain data and save to Excel."""
    
    print("=== Extract peak stress/strain data ===")
    
    # 1. Load raw data
    excel_file = os.path.join("dataset", "dataset_final.xlsx")
    
    if not os.path.exists(excel_file):
        excel_file = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\dataset\dataset_final.xlsx"
    
    print(f"Load Excel file: {excel_file}")
    
    # Load material parameter data (first sheet)
    material_df = pd.read_excel(excel_file, sheet_name=0)
    print(f"Material parameter shape: {material_df.shape}")
    print(f"Material parameter columns: {list(material_df.columns)}")
    
    # Load stress-strain data (second sheet)
    stress_df = pd.read_excel(excel_file, sheet_name=1)
    print(f"Stress-strain shape: {stress_df.shape}")
    
    # 2. Get column names
    strain_col = stress_df.columns[0]  # First column is strain
    stress_cols = stress_df.columns[1:]  # Remaining columns are stress per sample
    
    # 3. Define material parameter columns (15 params, aligned with dataloader)
    material_param_cols = [
        # 10 material features
        'water', 'cement', 'w/c', 'CS', 'sand',
        'CA', 'r', 'WA', 'S', 'CI',
        # 5 specimen parameters
        'age', 'μe', 'DJB', 'side', 'GJB'
    ]
    
    # Check which columns exist
    existing_material_cols = [col for col in material_param_cols if col in material_df.columns]
    missing_cols = [col for col in material_param_cols if col not in material_df.columns]
    
    if missing_cols:
        print(f"Warning: missing material columns: {missing_cols}")
        print(f"Will use existing columns: {existing_material_cols}")
    
    material_param_cols = existing_material_cols
    
    print(f"Material param count: {len(material_param_cols)}")
    print(f"Material param names: {material_param_cols}")
    
    # 4. Compute peak stress/strain
    print("\n=== Compute peak stress/strain ===")
    
    peak_data = []
    custom_id_col = 'NO'  # Use NO as custom specimen ID
    
    # Ensure NO column exists
    if custom_id_col not in material_df.columns:
        # Try alternative names
        possible_cols = ['自定义试件编号', 'NO', '编号', 'ID']
        custom_id_col = None
        for col in possible_cols:
            if col in material_df.columns:
                custom_id_col = col
                break
        if custom_id_col is None:
            raise ValueError(f"Custom specimen ID column not found, tried: {possible_cols}")
    
    print(f"Use column '{custom_id_col}' as custom specimen ID")
    
    # Full strain data
    strain_data_full = stress_df[strain_col].values
    
    # Check DataSlice column
    has_dataslice = 'DataSlice' in material_df.columns
    
    # Check existing peak columns (prefer existing)
    fc_col = None
    peak_strain_col = None
    for col in material_df.columns:
        col_str = str(col).strip().lower()
        if col_str in ['fc', '峰值应力fc', 'peak_stress']:
            fc_col = col
        if col_str in ['peak_strain', '峰值应变εc', 'εc']:
            peak_strain_col = col
    
    use_existing_peaks = (fc_col is not None and peak_strain_col is not None)
    if use_existing_peaks:
        print(f"Detected existing peak columns: '{fc_col}', '{peak_strain_col}', will prefer existing values")
    else:
        print("No existing peak columns found, will compute from curves")
    
    for idx, row in material_df.iterrows():
        sample_name = str(row[custom_id_col])
        
        # Ensure sample exists in stress data
        if sample_name not in stress_df.columns:
            print(f"Warning: sample {sample_name} not found in stress data")
            continue
        
        # Stress data (always compute index/length from curve)
        stress_series = stress_df[sample_name]
        valid_stress_data = stress_series.dropna()
        stress_data = valid_stress_data.values
        valid_length = len(stress_data)
        
        if valid_length == 0:
            print(f"Warning: sample {sample_name} has no valid stress data")
            continue
        
        # Trim strain to match valid stress length
        strain_data = strain_data_full[:valid_length]
        
        # Always compute peak index and length from curve
        peak_idx = np.argmax(stress_data)
        peak_stress_from_curve = stress_data[peak_idx]
        peak_strain_from_curve = strain_data[peak_idx]
        
        # Prefer existing peak values if present, keep index/length from curve
        if use_existing_peaks:
            peak_stress_existing = row[fc_col]
            peak_strain_existing = row[peak_strain_col]
            
            # Validate existing values
            if pd.isna(peak_stress_existing) or pd.isna(peak_strain_existing):
                # Fallback to curve-derived values
                use_existing_peaks_for_sample = False
                peak_stress = peak_stress_from_curve
                peak_strain = peak_strain_from_curve
            else:
                # Use existing values, keep curve-derived index/length
                use_existing_peaks_for_sample = True
                peak_stress = peak_stress_existing
                peak_strain = peak_strain_existing
        else:
            # No existing columns, use curve-derived values
            use_existing_peaks_for_sample = False
            peak_stress = peak_stress_from_curve
            peak_strain = peak_strain_from_curve
        
        # Create data row (regardless of source)
        # Collect material parameters
        material_params = row[material_param_cols].values if len(material_param_cols) > 0 else []
        
        # Build data row (always includes peak index and valid length)
        data_row = {
            'NO': sample_name,  # Use NO as column name
            'fc': peak_stress,
            'peak_strain': peak_strain,
            'PeakStressIndex': peak_idx,  # Always from curve
            'ValidDataLength': valid_length,  # Always from curve
        }
        
        # Add data source flag
        if use_existing_peaks_for_sample:
            data_row['DataSource'] = 'existing_column'
        else:
            data_row['DataSource'] = 'curve_computed'
        
        # Add DataSlice info if present
        if has_dataslice:
            data_row['DataSlice'] = row['DataSlice']
        
        # Add material parameters
        for i, col_name in enumerate(material_param_cols):
            if i < len(material_params):
                data_row[col_name] = material_params[i]
        
        peak_data.append(data_row)
        
        if (idx + 1) % 10 == 0 or idx == 0:
            source_str = "(existing)" if use_existing_peaks_for_sample else "(curve)"
            print(f"Sample {sample_name}: fc={peak_stress:.3f}, peak_strain={peak_strain:.6f} {source_str}")
    
    # 5. Build DataFrame
    peak_df = pd.DataFrame(peak_data)
    
    if len(peak_df) == 0:
        print("Error: no peak data extracted!")
        return None
    
    print(f"\n=== Data stats ===")
    print(f"Samples extracted: {len(peak_df)}")
    print(f"Peak stress range: {peak_df['fc'].min():.3f} - {peak_df['fc'].max():.3f}")
    print(f"Peak strain range: {peak_df['peak_strain'].min():.6f} - {peak_df['peak_strain'].max():.6f}")
    
    # Show DataSlice distribution if available
    if 'DataSlice' in peak_df.columns:
        print(f"\nDataset split distribution:")
        slice_counts = peak_df['DataSlice'].value_counts()
        for slice_name, count in slice_counts.items():
            print(f"  {slice_name}: {count} samples")
    
    # 6. Save to Excel
    output_dir = "dataset/Peak_Value_Extract"
    output_file = os.path.join(output_dir, "Peak_Value_Extract.xlsx")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main data sheet
        peak_df.to_excel(writer, sheet_name='PeakData', index=False)
        
        # Summary stats sheet
        stats_data = {
            'Metric': [
                'Sample Count',
                'fc_min',
                'fc_max', 
                'fc_mean',
                'fc_std',
                'peak_strain_min',
                'peak_strain_max',
                'peak_strain_mean',
                'peak_strain_std'
            ],
            'Value': [
                len(peak_df),
                peak_df['fc'].min(),
                peak_df['fc'].max(),
                peak_df['fc'].mean(),
                peak_df['fc'].std(),
                peak_df['peak_strain'].min(),
                peak_df['peak_strain'].max(),
                peak_df['peak_strain'].mean(),
                peak_df['peak_strain'].std()
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Material parameter stats sheet
        material_stats = []
        for col in material_param_cols:
            if col in peak_df.columns:
                material_stats.append({
                    'MaterialParam': col,
                    'Min': peak_df[col].min(),
                    'Max': peak_df[col].max(),
                    'Mean': peak_df[col].mean(),
                    'Std': peak_df[col].std()
                })
        
        material_stats_df = pd.DataFrame(material_stats)
        material_stats_df.to_excel(writer, sheet_name='MaterialStats', index=False)
    
    print(f"\n=== Save done ===")
    print(f"Data saved to: {output_file}")
    print(f"Sheets included:")
    print(f"  - PeakData: main data table")
    print(f"  - Summary: peak stress/strain stats")
    print(f"  - MaterialStats: material parameter stats")
    
    # 7. Preview first rows
    print(f"\n=== Preview ===")
    print("First 5 rows:")
    print(peak_df.head())
    
    return peak_df

if __name__ == "__main__":
    # Run extraction
    peak_data = extract_peak_data()
    
    if peak_data is not None:
        print("\nPeak data extraction completed!")
    else:
        print("\nPeak data extraction failed!")
