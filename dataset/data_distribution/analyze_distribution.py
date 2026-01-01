import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Configure fonts (supporting Chinese labels if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data from dataset_final.xlsx (script located in dataset/data_distribution/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
excel_file = os.path.join(project_root, 'dataset', 'dataset_final.xlsx')

df = pd.read_excel(excel_file, sheet_name=0)

# Material parameter column names (15 parameters)
material_param_cols = [
    # 10 material features
    'water', 'cement', 'w/c', 'CS', 'sand',
    'CA', 'r', 'WA', 'S', 'CI',  # coarse aggregate, replacement ratio, water absorption, max size, crushing index
    # 5 specimen parameters
    'age', 'μe', 'DJB', 'side', 'GJB'
]

# Material parameters (15)
X_material = df[material_param_cols].values

# Peak stress and strain columns
peak_stress_col = 'fc'
peak_strain_col = 'peak_strain'

# Ensure columns exist (fallback to matching lowercase names)
if peak_stress_col not in df.columns:
    # Try to find similar column names
    for col in df.columns:
        if str(col).strip().lower() == 'fc':
            peak_stress_col = col
            break

if peak_strain_col not in df.columns:
    # Try to find similar column names
    for col in df.columns:
        if str(col).strip().lower() == 'peak_strain':
            peak_strain_col = col
            break

# Validate columns
if peak_stress_col not in df.columns:
    raise ValueError(f"未找到峰值应力列 'fc'。可用列: {list(df.columns)}")
if peak_strain_col not in df.columns:
    raise ValueError(f"未找到峰值应变列 'peak_strain'。可用列: {list(df.columns)}")

X_stress_strain = df[[peak_stress_col, peak_strain_col]].values

# Parameter names (used for plotting/statistics)
material_names = [
# 10 material features
    'water', 'cement', 'w/c', 'CS', 'sand',
    'CA', 'r', 'WA', 'S', 'CI',  # coarse aggregate, replacement ratio, water absorption, max size, crushing index
    # 5 specimen parameters
    'age', 'μe', 'DJB', 'side', 'GJB'
]

stress_strain_names = ['fc', 'peak_strain']

print('=== Distribution analysis ===')
print(f'Sample count: {X_material.shape[0]}')
print(f'Material parameter count: {X_material.shape[1]}')
print(f'Stress/strain parameter count: {X_stress_strain.shape[1]}')

# Analyze distribution characteristics of each parameter
all_data = np.column_stack([X_material, X_stress_strain])
all_names = material_names + stress_strain_names

print('\n=== Parameter-wise distribution analysis ===')

# Lists to store statistical information (overall statistics + histogram percentage distribution)
distribution_stats = []
histogram_data = {}  # key: parameter name, value: DataFrame with histogram / percentage data

for i, name in enumerate(all_names):
    data = all_data[:, i]
    
    # Remove NaN and infinite values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        continue
    
   # Basic statistics
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    median_val = np.median(data_clean)
    min_val = np.min(data_clean)
    max_val = np.max(data_clean)
    q25_val = np.percentile(data_clean, 25)
    q75_val = np.percentile(data_clean, 75)
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean)
    
    # Distribution test
    # Shapiro-Wilk test (for sample size < 5000）
    if len(data_clean) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data_clean)
        normal_test_stat = shapiro_stat
        normal_test_p = shapiro_p
        normal_test_name = 'Shapiro-Wilk'
    else:
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data_clean, 'norm', args=(mean_val, std_val))
        normal_test_stat = ks_stat
        normal_test_p = ks_p
        normal_test_name = 'KS-test'
    
    # Classify distribution characteristics for summary
    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        dist_type = 'Approximately normal'
    elif abs(skewness) > 1:
        dist_type = 'Skewed distribution'
    elif abs(kurtosis) > 1:
        dist_type = 'Leptokurtic or platykurtic'
    else:
        dist_type = 'Other distribution'
    
    # Identify discrete parameters (chamfer ratio, side/diameter, height-to-diameter)
    is_discrete = i in [12, 13, 14]
    
    # Collect summary statistics
    distribution_stats.append({
        'Parameter Index': i + 1,
        'Parameter Name': name,
        'Distribution Type': 'Discrete' if is_discrete else 'Continuous',
        'Sample Size': len(data_clean),
        'Min': min_val,
        'Max': max_val,
        'Q1 (25%)': q25_val,
        'Q3 (75%)': q75_val,
        'Mean': mean_val,
        'Median': median_val,
        'Std Dev': std_val,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Distribution Feature': dist_type,
        'Normality Test Method': normal_test_name,
        'Normality Test Statistic': normal_test_stat,
        'Normality Test p-value': normal_test_p
    })
    
    # Histogram percentages for downstream visualization
    if is_discrete:
        # Discrete variables: value + percentage
        unique_values, counts = np.unique(data_clean, return_counts=True)
        percentages = counts / counts.sum() * 100.0
        hist_df = pd.DataFrame({
            'Value': unique_values,
            'Percentage (%)': percentages
        })
    else:
        # Continuous variables: bin centers + percentage
        counts, bin_edges = np.histogram(data_clean, bins=30)
        if counts.sum() == 0:
            continue
        percentages = counts / counts.sum() * 100.0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist_df = pd.DataFrame({
            'Bin Center': bin_centers,
            'Percentage (%)': percentages
        })
    histogram_data[name] = hist_df
    
    print(f'{i+1:2d}. {name:25s}:')
    print(f'    Range: [{min_val:8.3f}, {max_val:8.3f}]')
    print(f'    Mean: {mean_val:8.3f}, Std: {std_val:8.3f}')
    print(f'    Skewness: {skewness:8.3f}, Kurtosis: {kurtosis:8.3f}')
    print(f'    Distribution feature: {dist_type}')
    print(f'    Normality test: {normal_test_name}: stat={normal_test_stat:.4f}, p={normal_test_p:.4f}')
    print()

# Analyze specific parameters in detail
print('\n=== Detailed analysis of special parameters ===')

# Chamfer ratio (expected discrete 0/1)
chamfer_data = X_material[:, 12]
print(f'Chamfer ratio distribution:')
print(f'  Unique values: {np.unique(chamfer_data)}')
print(f'  Count of 0: {np.sum(chamfer_data == 0)}')
print(f'  Count of 1: {np.sum(chamfer_data == 1)}')
print(f'  Other values: {np.sum((chamfer_data != 0) & (chamfer_data != 1))}')

# Side length or diameter (should be positive integers)
size_data = X_material[:, 13]
print(f'\nSide length or diameter distribution:')
print(f'  All integers: {np.all(size_data == np.round(size_data))}')
print(f'  All non-negative: {np.all(size_data >= 0)}')
print(f'  Min: {size_data.min()}')
print(f'  Max: {size_data.max()}')
print(f'  Unique count: {len(np.unique(size_data))}')

# Height-to-diameter ratio (positive integers)
ratio_data = X_material[:, 14]
print(f'\nHeight/diameter ratio distribution:')
print(f'  All integers: {np.all(ratio_data == np.round(ratio_data))}')
print(f'  All non-negative: {np.all(ratio_data >= 0)}')
print(f'  Min: {ratio_data.min()}')
print(f'  Max: {ratio_data.max()}')
print(f'  Unique count: {len(np.unique(ratio_data))}')

# Peak stress (non-negative)
stress_data = X_stress_strain[:, 0]
print(f'\nPeak stress distribution:')
print(f'  All non-negative: {np.all(stress_data >= 0)}')
print(f'  Min: {stress_data.min()}')
print(f'  Max: {stress_data.max()}')
print(f'  Mean: {stress_data.mean():.3f}')
print(f'  Std: {stress_data.std():.3f}')

# Peak strain
strain_data = X_stress_strain[:, 1]
print(f'\nPeak strain distribution:')
print(f'  Min: {strain_data.min()}')
print(f'  Max: {strain_data.max()}')
print(f'  Mean: {strain_data.mean():.6f}')
print(f'  Std: {strain_data.std():.6f}')

print('\n=== Distribution summary ===')
print('1. Chamfer ratio: discrete distribution (0 or 1)')
print('2. Side length or diameter: discrete positive integer distribution')
print('3. Height–diameter ratio: discrete positive integer distribution')
print('4. Peak stress: continuous non-negative distribution')
print('5. Peak strain: continuous distribution')
print('6. Other material parameters: mostly continuous, some skewed')

# ========== Visualization ==========
print('\n=== Starting to plot distribution charts ===')

# Identify indices of discrete and continuous distributions
discrete_indices = [12, 13, 14]   # chamfer ratio, side length or diameter, height-to-diameter ratio
discrete_names = [all_names[i] for i in discrete_indices]
discrete_data_list = [all_data[:, i] for i in discrete_indices]

# Indices of continuous distributions (12 parameters excluding 3 discrete parameters and 2 peak stress/strain parameters)
# Peak stress and strain indices in all_data are 15 and 16 (X_material has 15 columns, indices 0-14)
peak_stress_strain_indices = [15, 16]  # peak stress, peak strain
continuous_indices = [i for i in range(len(all_names)) if i not in discrete_indices + peak_stress_strain_indices]
continuous_names = [all_names[i] for i in continuous_indices]
continuous_data_list = [all_data[:, i] for i in continuous_indices]

print(f'Number of discrete distributions: {len(discrete_names)}')
print(f'Discrete distribution parameters: {discrete_names}')
print(f'Number of continuous distributions:  {len(continuous_names)}')
print(f'Continuous distribution parameters: {continuous_names}')

# Create output directory
output_dir = os.path.join(current_dir, 'distribution_plots')
os.makedirs(output_dir, exist_ok=True)

# Create combined chart: upper half for continuous distribution histograms, lower half for discrete distribution donut charts
# Use GridSpec for layout: upper half 3 rows x 4 columns (continuous distributions), lower half 1 row x 3 columns (discrete distributions)
fig = plt.figure(figsize=(16, 14))
fig = plt.figure(figsize=(16, 14))
gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 0.8])

# Upper half: histograms for continuous distributions (3 rows x 4 columns, total 12 positions)
axes_continuous = []
for i in range(3):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(continuous_names):
            ax = fig.add_subplot(gs[i, j])
            axes_continuous.append(ax)

for idx, (ax, name, data) in enumerate(zip(axes_continuous, continuous_names, continuous_data_list)):
    # Remove NaN and infinite value
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) > 0:
        # Plot histogram
        ax.hist(data_clean, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
       # Add statistical information
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.2f}')
        ax.legend(fontsize=8, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=10, fontweight='bold')

# Lower half: donut charts for discrete distributions (1 row x 3 columns)）
axes_discrete = []
axes_discrete.append(fig.add_subplot(gs[3, 0]))
axes_discrete.append(fig.add_subplot(gs[3, 1]))
axes_discrete.append(fig.add_subplot(gs[3, 2:4]))  # Third chart occupies last two columns for balance
for ax, name, data in zip(axes_discrete, discrete_names, discrete_data_list):
    # Remove NaN and infinite values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) > 0:
       # Count unique values and their occurrences
        unique_values, counts = np.unique(data_clean, return_counts=True)
        
         # For chamfer ratio (only 0 and 1), use specific labels
        if name == '倒角比(棱柱为0,圆柱为1)':
            # Create label mapping, ensuring 0 and 1 have corresponding labels
            label_map = {0: '棱柱 (0)', 1: '圆柱 (1)'}
            # Sort by 0, 1 order
            sorted_indices = np.argsort(unique_values)
            unique_values_sorted = unique_values[sorted_indices]
            counts_sorted = counts[sorted_indices]
            
            labels = [label_map.get(val, f'{val}') for val in unique_values_sorted]
            counts_ordered = counts_sorted
        else:
            #  Other discrete distributions use numerical labels, sorted by value
            sorted_indices = np.argsort(unique_values)
            unique_values_sorted = unique_values[sorted_indices]
            counts_sorted = counts[sorted_indices]
            
            labels = [f'{val:.0f}' if val == int(val) else f'{val:.2f}' for val in unique_values_sorted]
            counts_ordered = counts_sorted
        
        # Plot donut chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
        wedges, texts, autotexts = ax.pie(counts_ordered, labels=labels, autopct='%1.1f%%',
                                          startangle=90, colors=colors, 
                                          wedgeprops=dict(width=0.5, edgecolor='w', linewidth=2))
        
         # Set text styles
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        ax.set_title(name, fontsize=12, fontweight='bold', pad=20)
        
       # Add total count information
        total = np.sum(counts_ordered)
        ax.text(0, -1.3, f'Total: {total}', ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=12, fontweight='bold')

# Add main title
fig.suptitle('材料与试件参数分布分析', fontsize=16, fontweight='bold', y=0.98)

# Save combined chart
plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'), dpi=300, bbox_inches='tight')
print(f'Combined distribution chart saved: {os.path.join(output_dir, "parameter_distributions.png")}')

#  Also save separate continuous and discrete distribution charts (for backward compatibility)
#  Continuous distribution charts
fig1, axes1 = plt.subplots(3, 4, figsize=(16, 12))
axes1 = axes1.flatten()

for idx, (ax, name, data) in enumerate(zip(axes1, continuous_names, continuous_data_list)):
    data_clean = data[np.isfinite(data)]
    if len(data_clean) > 0:
        ax.hist(data_clean, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        mean_val = np.mean(data_clean)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.2f}')
        ax.legend(fontsize=8, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=10, fontweight='bold')

plt.suptitle('Continuous Parameter Distribution Charts (Histograms)）', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'continuous_distributions.png'), dpi=300, bbox_inches='tight')
print(f'Continuous distribution charts saved: {os.path.join(output_dir, "continuous_distributions.png")}')
plt.close()

# Discrete distribution charts
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

for ax, name, data in zip(axes2, discrete_names, discrete_data_list):
    data_clean = data[np.isfinite(data)]
    if len(data_clean) > 0:
        unique_values, counts = np.unique(data_clean, return_counts=True)
        if name == 'Chamfer ratio (prism=0, cylinder=1)':
            label_map = {0: 'Prism (0)', 1: 'Cylinder (1)'}
            sorted_indices = np.argsort(unique_values)
            unique_values_sorted = unique_values[sorted_indices]
            counts_sorted = counts[sorted_indices]
            labels = [label_map.get(val, f'{val}') for val in unique_values_sorted]
            counts_ordered = counts_sorted
        else:
            sorted_indices = np.argsort(unique_values)
            unique_values_sorted = unique_values[sorted_indices]
            counts_sorted = counts[sorted_indices]
            labels = [f'{val:.0f}' if val == int(val) else f'{val:.2f}' for val in unique_values_sorted]
            counts_ordered = counts_sorted
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
        wedges, texts, autotexts = ax.pie(counts_ordered, labels=labels, autopct='%1.1f%%', 
                                          startangle=90, colors=colors, 
                                          wedgeprops=dict(width=0.5, edgecolor='w', linewidth=2))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        ax.set_title(name, fontsize=12, fontweight='bold', pad=20)
        total = np.sum(counts_ordered)
        ax.text(0, -1.3, f'Total: {total}', ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=12, fontweight='bold')

plt.suptitle('Discrete Parameter Distribution Charts (Donut Charts)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'discrete_distributions.png'), dpi=300, bbox_inches='tight')
print(f'Discrete distribution charts saved: {os.path.join(output_dir, "discrete_distributions.png")}')
plt.close()

print(f'\nAll distribution charts saved to: {output_dir}')
print(f'  - Combined distribution chart: parameter_distributions.png')
print(f'  - Continuous distribution charts: continuous_distributions.png')
print(f'  - Discrete distribution charts: discrete_distributions.png')

# ========== Save distribution statistics table ==========
print('\n=== Saving distribution statistics table ===')

# 创建DataFrame
df_stats = pd.DataFrame(distribution_stats)

# Set column order
column_order = [
    'Parameter Index', 'Parameter Name', 'Distribution Type', 'Sample Size',
    'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'Mean', 'Median', 'Std Dev',
    'Skewness', 'Kurtosis', 'Distribution Feature',
    'Normality Test Method', 'Normality Test Statistic', 'Normality Test p-value'
]
df_stats = df_stats[column_order]

# Save overall statistics and histogram percentage data for each parameter to the same Excel file (multiple worksheets)
excel_path = os.path.join(output_dir, 'distribution_statistics.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Overall statistics in first worksheet
    df_stats.to_excel(writer, sheet_name='Summary', index=False)
    # Histogram percentage data for each parameter in subsequent worksheets
    for param_name, hist_df in histogram_data.items():
        # Excel sheet names can't contain certain special characters and have length limits, so clean them
        safe_name = ''.join(c if c not in '[]:*?/\\' else '_' for c in str(param_name))
        safe_name = safe_name[:31]  # Excel sheet name max length
        # Avoid duplicate names with Summary or other worksheets
        if safe_name.lower() == 'summary':
            safe_name = 'Summary_1'
        hist_df.to_excel(writer, sheet_name=safe_name, index=False)

print(f'Distribution statistics and histogram percentage data saved: {excel_path}')

print(f'\n=== All analyses completed ===')
print(f'Output directory: {output_dir}')
print(f'  - Combined distribution chart: parameter_distributions.png')
print(f'  - Continuous distribution charts: continuous_distributions.png')
print(f'  - Discrete distribution charts: discrete_distributions.png')
print(f'  - Distribution statistics and histogram data: distribution_statistics.xlsx')

# ========== Output statistical summary of peak stress and strain (for paper tables) ==========
print('\n=== Statistical summary of peak stress and strain (for paper tables) ===')
fc_stats = None
peak_strain_stats = None

for stat in distribution_stats:
    if stat['Parameter Name'] == 'fc':
        fc_stats = stat
    elif stat['Parameter Name'] == 'peak_strain':
        peak_strain_stats = stat

if fc_stats:
    print(f"\nPeak stress (fc):")
    print(f"  Mean: {fc_stats['Mean']:.2f}")
    print(f"  Std: {fc_stats['Std Dev']:.2f}")
    print(f"  Min: {fc_stats['Min']:.2f}")
    print(f"  Q2 (25%): {fc_stats['Q1 (25%)']:.2f}")
    print(f"  Median: {fc_stats['Median']:.2f}")
    print(f"  Q4 (75%): {fc_stats['Q3 (75%)']:.2f}")
    print(f"  Max: {fc_stats['Max']:.2f}")
    print(f"\nLaTeXtable row:")
    print(f"\\hline")
    print(f"fc & {fc_stats['Mean']:.2f}  & {fc_stats['Std Dev']:.2f}  & {fc_stats['Min']:.2f}  & {fc_stats['Q1 (25%)']:.2f}  & {fc_stats['Median']:.2f}  & {fc_stats['Q3 (75%)']:.2f}  & {fc_stats['Max']:.2f}  \\\\")

if peak_strain_stats:
    print(f"\nPeak strain (peak_strain):")
    print(f"  Mean: {peak_strain_stats['Mean']:.6f}")
    print(f"  Std: {peak_strain_stats['Std Dev']:.6f}")
    print(f"  Min: {peak_strain_stats['Min']:.6f}")
    print(f"  Q2 (25%): {peak_strain_stats['Q1 (25%)']:.6f}")
    print(f"  Median: {peak_strain_stats['Median']:.6f}")
    print(f"  Q4 (75%): {peak_strain_stats['Q3 (75%)']:.6f}")
    print(f"  Max: {peak_strain_stats['Max']:.6f}")
    print(f"\nLaTeX table row:")
    print(f"\\hline")
    print(f"peak\\_strain & {peak_strain_stats['Mean']:.6f}  & {peak_strain_stats['Std Dev']:.6f}  & {peak_strain_stats['Min']:.6f}  & {peak_strain_stats['Q1 (25%)']:.6f}  & {peak_strain_stats['Median']:.6f}  & {peak_strain_stats['Q3 (75%)']:.6f}  & {peak_strain_stats['Max']:.6f}  \\\\")

if not fc_stats or not peak_strain_stats:
    print("\nWarning: Statistical information for peak stress or peak strain not found. Please check the data file.")
