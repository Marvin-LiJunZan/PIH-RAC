"""
Analyze cluster distribution imbalance in subset division
Used to explain why certain folds perform poorly in three-fold cross-validation
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def analyze_subset_cluster_distribution(excel_file='dataset/cluster_analysis/save/dataset_交叉验证标记_tripartite.xlsx'):
    """Analyze the distribution of clusters in each subset"""
    
    print("="*70)
    print("Analyzing cluster distribution imbalance in subset division")
    print("="*70)
    
    # Read data
    df = pd.read_excel(excel_file, engine='openpyxl')
    
    # Get cluster labels and DataSlice labels
    clusters = df['Cluster'].values
    data_slices = df['DataSlice'].values
    
    # Count the number of samples in each cluster
    cluster_counts = Counter(clusters)
    print(f"\n1. Total number of samples in each cluster:")
    for clu in sorted(cluster_counts.keys()):
        print(f"   Cluster {clu}: {cluster_counts[clu]} samples")
    
    # Count the number of samples in each cluster in the test set
    test_mask = np.array([str(s).startswith('test') for s in data_slices])
    test_cluster_counts = Counter(clusters[test_mask])
    print(f"\n2. Number of samples in each cluster in the Test set:")
    for clu in sorted(test_cluster_counts.keys()):
        print(f"   Cluster {clu}: {test_cluster_counts[clu]} test samples")
    
    # Count the number of samples in each cluster in each subset
    print(f"\n3. Cluster distribution in each Subset:")
    subset_cluster_dist = defaultdict(lambda: defaultdict(int))
    
    for subset_num in [1, 2, 3]:
        subset_mask = np.array([str(s) == f'subset{subset_num}' for s in data_slices])
        subset_clusters = clusters[subset_mask]
        subset_cluster_counts = Counter(subset_clusters)
        
        print(f"\n   Subset {subset_num} (total {np.sum(subset_mask)} samples):")
        for clu in sorted(cluster_counts.keys()):
            count = subset_cluster_counts.get(clu, 0)
            subset_cluster_dist[subset_num][clu] = count
            status = "✓" if count > 0 else "✗ Missing"
            print(f"      Cluster {clu}: {count} samples {status}")
    
    # Identify problematic clusters (small clusters with uneven distribution)
    print(f"\n4. Problematic cluster identification (small clusters with uneven distribution):")
    problem_clusters = []
    for clu in sorted(cluster_counts.keys()):
        total = cluster_counts[clu]
        test_count = test_cluster_counts.get(clu, 0)
        remaining = total - test_count
        
        # Check distribution of remaining samples across three subsets
        subset_counts = [subset_cluster_dist[i][clu] for i in [1, 2, 3]]
        missing_subsets = [i+1 for i, count in enumerate(subset_counts) if count == 0]
        
        if total <= 5 and len(missing_subsets) > 0:  # Small clusters missing in some subsets
            problem_clusters.append({
                'cluster': clu,
                'total': total,
                'test': test_count,
                'remaining': remaining,
                'subset_distribution': subset_counts,
                'missing_subsets': missing_subsets
            })
            print(f"\n   ⚠️  Cluster {clu}:")
            print(f"      Total samples: {total}")
            print(f"      Test set: {test_count} samples")
            print(f"      Remaining: {remaining} samples")
            print(f"      Subset distribution: Subset1={subset_counts[0]}, Subset2={subset_counts[1]}, Subset3={subset_counts[2]}")
            print(f"      Missing subsets: {missing_subsets}")
            print(f"      Impact: When Subset{missing_subsets} is used as validation set, the model may not have seen features of Cluster{clu}")
    
    # Analyze impact on cross-validation
    print(f"\n5. Impact analysis on three-fold cross-validation:")
    print(f"\n   Fold 1 (Subset1 as validation set):")
    fold1_missing = []
    for prob in problem_clusters:
        if 1 in prob['missing_subsets']:
            fold1_missing.append(prob['cluster'])
    if fold1_missing:
        print(f"      ⚠️  Missing clusters: {fold1_missing} (but Subset1 is not in the validation set, so impact is minor)")
    else:
        print(f"      ✓ No missing clusters")
    
    print(f"\n   Fold 2 (Subset2 as validation set):")
    fold2_missing = []
    for prob in problem_clusters:
        if 2 in prob['missing_subsets']:
            fold2_missing.append(prob['cluster'])
    if fold2_missing:
        print(f"      ⚠️  Missing clusters: {fold2_missing} (but Subset2 is not in the validation set, so impact is minor)")
    else:
        print(f"      ✓ No missing clusters")
    
    print(f"\n   Fold 3 (Subset3 as validation set):")
    fold3_missing = []
    for prob in problem_clusters:
        if 3 in prob['missing_subsets']:
            fold3_missing.append(prob['cluster'])
    if fold3_missing:
        print(f"      ⚠️  Missing clusters: {fold3_missing}")
        print(f"      ⚠️  Impact: Training set lacks representation of these clusters, model may fail to learn their feature patterns")
        print(f"      ⚠️  This may result in poor performance for Fold 3 test set")
    else:
        print(f"      ✓ No missing clusters")
    
    # Calculate cluster coverage for each subset
    print(f"\n6. Cluster coverage for each Subset:")
    total_clusters = len(cluster_counts)
    for subset_num in [1, 2, 3]:
        subset_mask = np.array([str(s) == f'subset{subset_num}' for s in data_slices])
        subset_clusters = clusters[subset_mask]
        unique_clusters = len(np.unique(subset_clusters))
        coverage = unique_clusters / total_clusters * 100
        print(f"   Subset {subset_num}: {unique_clusters}/{total_clusters} clusters ({coverage:.1f}%)")
    
    # Recommendations
    print(f"\n7. Improvement suggestions:")
    print(f"   a) For small clusters (≤5 samples), consider:")
    print(f"      - Ensuring each subset contains at least 1 sample (if possible)")
    print(f"      - Or merging these small clusters into similar large clusters")
    print(f"   b) For cases where remaining samples < 3:")
    print(f"      - Use cyclic allocation instead of sequential allocation to ensure each subset has representation")
    print(f"      - Example: 2 samples → [1, 1, 0] changed to [1, 0, 1] or [0, 1, 1]")
    print(f"   c) Consider using stratified sampling to ensure each subset contains representatives of all clusters")
    
    return problem_clusters, subset_cluster_dist

if __name__ == '__main__':
    try:
        problem_clusters, subset_dist = analyze_subset_cluster_distribution()
        print(f"\n{'='*70}")
        print("Analysis completed!")
        print(f"{'='*70}")
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        print("Please run 聚类标记数据集_cvfixed.py first to generate the data file")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()