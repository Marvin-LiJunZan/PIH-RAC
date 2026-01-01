import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
import os
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm

# Hyperparameter automatic cluster number search

def search_best_n_clusters(mat_X_scaled, min_n=2, max_n=30, verbose=True, save_dir='save'):
    best_score = -1
    best_n_clusters = min_n
    all_scores = []
    n_samples = len(mat_X_scaled)
    # Silhouette score requirement: number of clusters must be >= 2 and <= n_samples - 1
    # Because if the number of clusters equals the number of samples, each sample is a cluster, and Silhouette score cannot be calculated
    original_max_n = max_n  # Save original value for plotting
    max_n = min(max_n, n_samples - 1)  # Ensure not exceeding n_samples - 1
    
    if max_n < min_n:
        raise ValueError(f'Maximum number of clusters ({max_n}) must be greater than or equal to minimum number of clusters ({min_n})')
    
    for n_clusters in range(min_n, max_n+1):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = agg.fit_predict(mat_X_scaled)
        unique_clusters = len(np.unique(clusters))
        
        # Check if the number of clusters is valid (at least 2 clusters, and each cluster has at least 2 samples)
        if unique_clusters < 2:
            score = -1
        elif n_clusters >= n_samples:
            # If the number of clusters >= number of samples, Silhouette score cannot be calculated
            score = -1
        else:
            try:
                score = silhouette_score(mat_X_scaled, clusters)
            except ValueError as e:
                # If Silhouette calculation fails (e.g., some clusters have only one sample), set to -1
                score = -1
                if verbose:
                    print(f'n_clusters={n_clusters}, silhouette calculation failed: {e}')
        
        all_scores.append(score)
        if verbose:
            print(f'n_clusters={n_clusters}, silhouette={score:.4f}')
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    if verbose:
        print(f'Best n_clusters: {best_n_clusters}, silhouette: {best_score:.4f}')
        if original_max_n > n_samples - 1:
            print(f'Note: The requested maximum number of clusters ({original_max_n}) exceeds the valid range and has been adjusted to {max_n}')
    
    plt.figure(figsize=(7,6), dpi=200)
    # Plot using the actual searched range
    n_clusters_range = list(range(min_n, max_n+1))
    plt.plot(n_clusters_range, all_scores, marker='o', color='#2b83ba', linewidth=2, markersize=6)
    plt.xlabel('n_clusters', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Method for n_clusters Selection', fontsize=16, pad=10)
    plt.grid(visible=True, ls='--', color='#adadad', alpha=0.4)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    save_path = os.path.join(save_dir, 'silhouette_curve.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    #plt.show()
    
    # Export Silhouette curve data to Excel
    silhouette_df = pd.DataFrame({
        'n_clusters': n_clusters_range,
        'Silhouette_Score': all_scores
    })
    silhouette_excel_path = os.path.join(save_dir, 'silhouette_curve_data.xlsx')
    silhouette_df.to_excel(silhouette_excel_path, index=False, engine='openpyxl')
    print(f'Silhouette curve data exported to: {silhouette_excel_path}')
    
    return best_n_clusters, all_scores, silhouette_df

def get_many_markers(n):
    base = ["o", "D", "^", "s", "P", "v", "<", ">", "h", "H", "*", "X", "p", "8"]
    return [base[i % len(base)] for i in range(n)]

def get_many_colors(n):
    cmap = cm.get_cmap('tab10', n)
    return [cmap(i) for i in range(n)]

# ==== parameters ====
DATA_COLUMNS = ['water', 'cement', 'w/c', 'CS', 'sand', 'CA', 'r', 'WA', 'S', 'CI', 'age', 'μe', 'DJB', 'side', 'GJB']
INPUT_FILE = 'dataset/dataset_final.xlsx'
# Save directory setting - save to dataset/cluster_analysis/save/ directory
SAVE_DIR = 'dataset/cluster_analysis/save'
os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure save directory exists
OUTPUT_FILE = os.path.join(SAVE_DIR, 'dataset_cv_labeled_fixed.xlsx')
DATA_SLICE_COLUMN = 'DataSlice'
N_CLUSTERS = None
TEST_SIZE = 18
N_SPLITS = 3

print('=== Cluster & CV split script (fixed CV labeling) ===')
# 1. load data
print(f'Reading: {INPUT_FILE}')
df = pd.read_excel(INPUT_FILE)
print(f'Data size: {len(df)}')

# 2. cluster
mat_X = df[DATA_COLUMNS].values.astype(float)
scaler = StandardScaler()
mat_X_scaled = scaler.fit_transform(mat_X)

# auto optimize n_clusters - dynamically set search range based on number of samples
n_samples = len(mat_X_scaled)
# Search range: 2 to n_samples - 1 (Silhouette score requires number of clusters <= n_samples - 1)
# In practice, the number of clusters usually does not exceed 1/3 of the number of samples, but for comprehensiveness, we search up to n_samples - 1
min_n_clusters = 2
max_n_clusters = n_samples - 1  # Maximum number of clusters cannot be equal to number of samples (otherwise Silhouette score cannot be calculated)
print(f'Number of dataset samples: {n_samples}')
print(f'Searching best n_clusters ({min_n_clusters}~{max_n_clusters})...')
N_CLUSTERS, silhouette_scores, silhouette_df = search_best_n_clusters(
    mat_X_scaled, min_n=min_n_clusters, max_n=max_n_clusters, save_dir=SAVE_DIR)
print(f'AgglomerativeClustering, n_clusters={N_CLUSTERS}')
agg = AgglomerativeClustering(n_clusters=N_CLUSTERS)
clusters = agg.fit_predict(mat_X_scaled)

# Add cluster label column to dataframe
df['Cluster'] = clusters
print(f'Cluster labels added to dataframe. Cluster distribution: {dict(zip(*np.unique(clusters, return_counts=True)))}')

# color/marker for all clusters
n_current_clusters = len(np.unique(clusters))
sci_colors = get_many_colors(n_current_clusters)
marker_shapes = get_many_markers(n_current_clusters)

# PCA visualize
pca = PCA(n_components=2, random_state=42)
mat_X_pca = pca.fit_transform(mat_X_scaled)
plt.figure(figsize=(7,6), dpi=200)
for i, clu in enumerate(np.unique(clusters)):
    pts = (clusters == clu)
    plt.scatter(mat_X_pca[pts, 0], mat_X_pca[pts, 1],
                c=[sci_colors[i]],
                marker=marker_shapes[i],
                label=f'Cluster {clu+1}',
                edgecolor='k', alpha=0.82, s=60, linewidths=0.7)
from matplotlib.patches import Ellipse
def draw_ellipse(x, y, ax, **kwargs):
    if len(x) < 3: return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height,
                      angle=theta, edgecolor=kwargs.get('color', 'gray'),
                      lw=2, fill=False, ls='--', alpha=0.7)
    ax.add_patch(ellipse)
ax = plt.gca()
for i, clu in enumerate(np.unique(clusters)):
    pts = (clusters == clu)
    draw_ellipse(mat_X_pca[pts, 0], mat_X_pca[pts, 1], ax,
                 color=sci_colors[i])
plt.xlabel('PCA Component 1', fontdict={'fontsize': 17})
plt.ylabel('PCA Component 2', fontdict={'fontsize': 17})
plt.title(f'Agglomerative Clustering (n={N_CLUSTERS})', fontsize=20, pad=10)
leg = plt.legend(
    title='Cluster',
    bbox_to_anchor=(1.02, 0.5),
    loc='center left',
    borderaxespad=0.0,
    frameon=True,
    framealpha=0.97,
    fancybox=True,
    borderpad=0.9
)
leg.get_title().set_fontsize(15)
plt.grid(visible=True, ls='--', color='#adadad', alpha=0.4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout(rect=[0, 0, 0.87, 1])
save_path_pdf = os.path.join(SAVE_DIR, 'cluster_vis.pdf')
save_path_png = os.path.join(SAVE_DIR, 'cluster_vis.png')
plt.savefig(save_path_pdf, dpi=400, bbox_inches='tight')
plt.savefig(save_path_png, dpi=400, bbox_inches='tight')
#plt.show()
plt.close()
print(f'Cluster visualization saved as {save_path_pdf} and {save_path_png}')

# Export PCA cluster data to Excel
pca_cluster_df = pd.DataFrame({
    'PCA_Component_1': mat_X_pca[:, 0],
    'PCA_Component_2': mat_X_pca[:, 1],
    'Cluster': clusters,
    'Cluster_Label': [f'Cluster {c+1}' for c in clusters]  # Add readable labels
})
pca_excel_path = os.path.join(SAVE_DIR, 'pca_cluster_data.xlsx')
pca_cluster_df.to_excel(pca_excel_path, index=False, engine='openpyxl')
print(f'PCA cluster data exported to: {pca_excel_path}')

# Export ellipse parameters for each cluster (for plotting ellipses in Origin)
ellipse_data = []
for i, clu in enumerate(np.unique(clusters)):
    pts = (clusters == clu)
    x_data = mat_X_pca[pts, 0]
    y_data = mat_X_pca[pts, 1]
    if len(x_data) >= 3:
        cov = np.cov(x_data, y_data)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width = 2 * np.sqrt(vals[0])
        height = 2 * np.sqrt(vals[1])
        center_x = np.mean(x_data)
        center_y = np.mean(y_data)
        ellipse_data.append({
            'Cluster': clu,
            'Cluster_Label': f'Cluster {clu+1}',
            'Center_X': center_x,
            'Center_Y': center_y,
            'Width': width,
            'Height': height,
            'Angle_deg': theta
        })

if ellipse_data:
    ellipse_df = pd.DataFrame(ellipse_data)
    ellipse_excel_path = os.path.join(SAVE_DIR, 'cluster_ellipse_params.xlsx')
    ellipse_df.to_excel(ellipse_excel_path, index=False, engine='openpyxl')
    print(f'Cluster ellipse parameters exported to: {ellipse_excel_path}')

# Create comprehensive Excel file containing all visualization data (for redrawing in Origin)
comprehensive_excel_path = os.path.join(SAVE_DIR, 'cluster_visualization_data.xlsx')
with pd.ExcelWriter(comprehensive_excel_path, engine='openpyxl') as writer:
    # Sheet 1: Silhouette curve data
    silhouette_df.to_excel(writer, sheet_name='Silhouette Curve', index=False)
    
    # Sheet 2: PCA cluster scatter plot data
    pca_cluster_df.to_excel(writer, sheet_name='PCA Cluster Scatter Plot', index=False)
    
    # Sheet 3: Ellipse parameters (if any)
    if ellipse_data:
        ellipse_df.to_excel(writer, sheet_name='Cluster Ellipse Parameters', index=False)
    
    # Sheet 4: PCA component contribution rate
    pca_variance_df = pd.DataFrame({
        'Component': ['PCA_Component_1', 'PCA_Component_2'],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Explained_Variance': pca.explained_variance_,
        'Cumulative_Ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    pca_variance_df.to_excel(writer, sheet_name='PCA Contribution Rate', index=False)

print(f'\nComprehensive visualization data exported to: {comprehensive_excel_path}')
print('  - Sheet 1: Silhouette curve data (n_clusters, Silhouette_Score)')
print('  - Sheet 2: PCA cluster scatter plot data (PCA_Component_1, PCA_Component_2, Cluster)')
if ellipse_data:
    print('  - Sheet 3: Cluster ellipse parameters (for drawing ellipse boundaries)')
print('  - Sheet 4: PCA contribution rate (variance ratio explained by principal components)')

# 4. stratified sample test set by cluster
test_size = TEST_SIZE
labels = clusters.copy()
all_indices = np.arange(len(labels))
unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
print(f'Cluster sample distribution: {dict(zip(unique_clusters, cluster_counts))}')
# assign test set size per cluster, round to int
cluster_test_sizes = np.round(cluster_counts * test_size / len(labels)).astype(int)
# make sure total is TEST_SIZE
while cluster_test_sizes.sum() < test_size:
    idx = np.argmax(cluster_counts - cluster_test_sizes)
    if cluster_test_sizes[idx] < cluster_counts[idx] - 1:
        cluster_test_sizes[idx] += 1
    else:
        break
while cluster_test_sizes.sum() > test_size:
    idx = np.argmax(cluster_test_sizes)
    if cluster_test_sizes[idx] > 0:
        cluster_test_sizes[idx] -= 1
    else:
        break
np.random.seed(7)
test_indices = []
train_val_indices = []
for clu, csize in zip(unique_clusters, cluster_test_sizes):
    clu_idx = np.where(labels == clu)[0]
    np.random.shuffle(clu_idx)
    test_indices.extend(clu_idx[:csize])
    train_val_indices.extend(clu_idx[csize:])
test_indices = np.array(test_indices)
train_val_indices = np.array(train_val_indices)
print(f'Test set size: {len(test_indices)}, remaining train/val: {len(train_val_indices)}')

# 5. stratified Kfold CV - fixed DataSlice encoding
slice_labels = [''] * len(df)
for i in test_indices:
    slice_labels[i] = 'test'
# only assign valN label, rest should remain 'train'
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
trainval_labels = labels[train_val_indices]
cv_slices = np.array(['train']*len(train_val_indices), dtype='<U10')
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, trainval_labels), 1):
    cv_slices[val_idx] = f'val{fold}'
for idx, s in zip(train_val_indices, cv_slices):
    slice_labels[idx] = s
assert all([lbl != '' for lbl in slice_labels]), 'Some samples not labeled!'
df[DATA_SLICE_COLUMN] = slice_labels
from collections import Counter
print(Counter(slice_labels))
print(df[DATA_SLICE_COLUMN].value_counts())
# save
print(f'Writing result to: {OUTPUT_FILE}')
df.to_excel(OUTPUT_FILE, index=False)
print('=== Split & DataSlice annotation finished ===')

# === Cluster-based sampling split (test/subset1/2/3 tripartite method) ===
slice_labels = [None]*len(df)
np.random.seed(7)
test_indices_dict = {}  # Use dictionary to record test set indices for each cluster
part_indices = [[], [], []]
for clu in unique_clusters:
    clu_idx = np.where(labels == clu)[0]
    np.random.shuffle(clu_idx)
    n_samples = len(clu_idx)
    
    # Rule for taking 20% from each cluster as test set:
    # 1. If cluster size * 0.2 < 1, round up to 1
    # 2. If cluster size * 0.2 is an integer, take that integer
    # 3. If cluster size * 0.2 > 1 and not an integer, round down
    test_ratio = 0.2
    n_test_float = n_samples * test_ratio
    
    # Use safer floating point comparison method
    n_test_int = int(n_test_float)
    is_integer = abs(n_test_float - n_test_int) < 1e-10  # Floating point precision tolerance
    
    if n_test_float < 1:
        n_test = 1  # Round up to 1
    elif is_integer:
        n_test = n_test_int  # If integer, take the integer
    else:
        n_test = int(np.floor(n_test_float))  # Greater than 1 and not integer, round down
    
    n_test = min(n_test, n_samples)  # Ensure not exceeding cluster size
    
    # Debug output
    print(f'  Cluster {clu}: {n_samples} samples -> {n_test_float:.2f} -> {n_test} test samples')
    test = clu_idx[:n_test]
    rest = clu_idx[n_test:]
    
    # Record test set indices for this cluster, using cluster number as key
    test_indices_dict[clu] = list(test)
    
    # Distribute remaining samples as evenly as possible into 3 subsets, using cyclic distribution to ensure each subset has representatives
    if len(rest) > 0:
        if len(rest) <= 3:
            # If number of samples is less than or equal to 3, use cyclic distribution to ensure each subset has representatives
            # For example: 2 samples → [1, 0, 1] or [0, 1, 1], not [1, 1, 0]
            # This avoids some subsets completely missing representatives from a cluster
            for i, idx in enumerate(rest):
                # Use cyclic distribution: i-th sample assigned to subset (i % 3)
                # But to balance, we start with the subset with the fewest samples currently
                subset_sizes = [len(part_indices[j]) for j in range(3)]
                # Find the subset with the fewest samples currently (if multiple, choose the one with smallest index)
                min_size = min(subset_sizes)
                target_subset = subset_sizes.index(min_size)
                part_indices[target_subset].append(idx)
        else:
            # If more than 3 samples, distribute evenly
            parts = np.array_split(rest, 3)
            for i in range(3):
                part_indices[i].extend(list(parts[i]))
# Verification and printing
total_test_count = sum(len(test_list) for test_list in test_indices_dict.values())
print('Test set sample count:', total_test_count)
print('Test set distribution by cluster:')
for clu in sorted(test_indices_dict.keys()):
    print(f'  Cluster {clu+1}: {len(test_indices_dict[clu])} test samples')
for k in range(3):
    print(f'Subset{k+1} count:', len(part_indices[k]))

# test labels: marked with test + global sequential number (test1, test2, test3...)
test_counter = 1
for clu in sorted(test_indices_dict.keys()):  # Process in cluster number order
    test_idx_list = test_indices_dict[clu]
    for idx in test_idx_list:
        slice_labels[idx] = f'test{test_counter}'
        test_counter += 1

# Verify test_counter is correct
expected_test_counter = total_test_count + 1  # +1 because it will increment one more time at the end
if test_counter != expected_test_counter:
    print(f'Warning: test_counter ({test_counter}) is not equal to expected value ({expected_test_counter})')
    print(f'Actual test count: {test_counter - 1}, expected test count: {total_test_count}')

# subsetN
for k in range(3):
    for i in part_indices[k]:
        slice_labels[i] = f'subset{k+1}'

# Final verification
assert all(lbl is not None for lbl in slice_labels), 'Some samples not labeled!'

# Verify number of test labels
actual_test_count = sum(1 for lbl in slice_labels if str(lbl).startswith('test'))
if actual_test_count != total_test_count:
    print(f'Error: Actual number of test labels ({actual_test_count}) is not equal to calculated test count ({total_test_count})')
    # Find missing test labels
    test_labels_in_slice = [str(lbl) for lbl in slice_labels if str(lbl).startswith('test')]
    test_numbers = sorted([int(lbl.replace('test', '')) for lbl in test_labels_in_slice])
    print(f'Existing test label numbers: {test_numbers}')
    expected_numbers = list(range(1, total_test_count + 1))
    missing_numbers = [n for n in expected_numbers if n not in test_numbers]
    if missing_numbers:
        print(f'Missing test label numbers: {missing_numbers}')
    # Force repair: reassign test labels
    print('Repairing test labels...')
    test_counter = 1
    for clu in sorted(test_indices_dict.keys()):
        test_idx_list = test_indices_dict[clu]
        for idx in test_idx_list:
            slice_labels[idx] = f'test{test_counter}'
            test_counter += 1
    print(f'Repair completed, number of reallocated test labels: {test_counter - 1}')

df[DATA_SLICE_COLUMN] = slice_labels
from collections import Counter
print('Counts:', Counter(slice_labels))
print(df[DATA_SLICE_COLUMN].value_counts())

# Final verification and forced repair before saving
print(f'\nVerification before saving:')
print(f'Expected number of tests: {total_test_count}')

# Force reallocation of test labels to ensure all test labels are correctly written to DataFrame
print('Forcibly reallocating test labels to ensure correctness...')
test_counter = 1
for clu in sorted(test_indices_dict.keys()):
    test_idx_list = test_indices_dict[clu]
    for idx in test_idx_list:
        # Use .loc to ensure correct assignment
        df.loc[idx, DATA_SLICE_COLUMN] = f'test{test_counter}'
        test_counter += 1

print(f'Reallocated {test_counter - 1} test labels')

# Verify test labels in DataFrame
final_test_labels = [str(lbl) for lbl in df[DATA_SLICE_COLUMN].values if str(lbl).startswith('test')]
final_test_count = len(final_test_labels)
print(f'Number of test labels in DataFrame: {final_test_count}')

if final_test_count != total_test_count:
    print(f'Error: Number of tests in DataFrame ({final_test_count}) is not equal to expected value ({total_test_count})')
    # Attempt repair again
    test_numbers = sorted([int(lbl.replace('test', '')) for lbl in final_test_labels])
    print(f'Existing test label numbers: {test_numbers}')
    missing_numbers = [n for n in range(1, total_test_count + 1) if n not in test_numbers]
    if missing_numbers:
        print(f'Missing test label numbers: {missing_numbers}')
else:
    print(f'✓ DataFrame verification passed: contains {final_test_count} test samples')

# Save new file - use engine='openpyxl' to ensure correct saving
OUTPUT_FILE2 = os.path.join(SAVE_DIR, 'dataset_cv_labeled_tripartite.xlsx')
print(f'\nWriting result to: {OUTPUT_FILE2}')

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE2) if os.path.dirname(OUTPUT_FILE2) else '.', exist_ok=True)

# Save file
try:
    df.to_excel(OUTPUT_FILE2, index=False, engine='openpyxl')
    print('File saved successfully')
except Exception as e:
    print(f'Error while saving: {e}')
    # Try using xlsxwriter
    try:
        df.to_excel(OUTPUT_FILE2, index=False, engine='xlsxwriter')
        print('Saved successfully using xlsxwriter engine')
    except Exception as e2:
        print(f'Failed with xlsxwriter as well: {e2}')
        raise

# Post-save verification - detailed check
print('\nPost-save verification:')
df_verify = pd.read_excel(OUTPUT_FILE2, engine='openpyxl')
verify_test_labels = [str(lbl) for lbl in df_verify[DATA_SLICE_COLUMN].values if str(lbl).startswith('test')]
verify_test_count = len(verify_test_labels)
verify_test_numbers = sorted([int(lbl.replace('test', '')) for lbl in verify_test_labels])

print(f'Number of test labels in Excel file: {verify_test_count}')
print(f'Test label numbers in Excel file: {verify_test_numbers}')

if verify_test_count != total_test_count:
    print(f'✗ Error: Number of tests in Excel file ({verify_test_count}) is not equal to expected value ({total_test_count})')
    missing_numbers = [n for n in range(1, total_test_count + 1) if n not in verify_test_numbers]
    if missing_numbers:
        print(f'Missing test label numbers: {missing_numbers}')
    # Try saving again
    print('Attempting to save file again...')
    df.to_excel(OUTPUT_FILE2, index=False, engine='openpyxl')
    # Verify again
    df_verify2 = pd.read_excel(OUTPUT_FILE2, engine='openpyxl')
    verify_test_labels2 = [str(lbl) for lbl in df_verify2[DATA_SLICE_COLUMN].values if str(lbl).startswith('test')]
    verify_test_count2 = len(verify_test_labels2)
    print(f'Number of test labels after re-saving: {verify_test_count2}')
    if verify_test_count2 == total_test_count:
        print('✓ Verification passed after re-saving!')
    else:
        print(f'✗ Still problematic after re-saving, test count: {verify_test_count2}')
else:
    print(f'✓ Verification passed: Excel file contains {verify_test_count} test samples')
    print(f'✓ All test labels are complete: test1 to test{total_test_count}')

print('=== Tripartite split & DataSlice annotation finished ===')