# Analysis of Poor Performance in Fold 3 of Three-Fold Cross-Validation

According to cross-validation results:

- **Fold 1** (with Subset1 as the validation set): R²=0.908, RMSE=2.52 MPa
- **Fold 2** (with Subset2 as the validation set): R²=0.902, RMSE=2.61 MPa
- **Fold 3** (with Subset3 as the validation set): R²=0.853, RMSE=3.19 MPa ⚠️

The performance of Fold 3 on the test set is significantly lower than that of Fold 1 and Fold 2.

## Root Cause

Through analysis of the data partitioning process, an **imbalanced cluster distribution** issue was identified:

### Situation of Cluster 8

- **Total number of samples** : 3
- **Test set allocation** : 1 sample (according to the 20% ratio)
- **Remaining samples** : 2
- **Original allocation strategy** : Sequential allocation to Subset1 and Subset2
  - Subset1: 1 sample ✓
  - Subset2: 1 sample ✓
  - Subset3: 0 samples ✗ ****Missing****

### Impact Analysis

When **Fold 3** is executed:

* **Validation set** : Subset3 (contains representatives of all clusters but lacks Cluster 8)
* **Training set** : Subset1 + Subset2 (contains representatives of Cluster 8)
* **Test set** : Independent test set (contains 1 sample from Cluster 8)

**Problem** : Although the training set includes Cluster 8, when Subset3 is used as the validation set, the model does not encounter the feature patterns of Cluster 8 during the validation phase. More importantly, when Subset3 serves as the validation set, the training set is actually Subset1 + Subset2, both of which contain Cluster 8, but the  **test set may include samples from Cluster 8** , and the model may not have sufficiently learned the features of Cluster 8 during training (since there are only 2 training samples).

### Cluster Coverage Statistics

- Subset 1: 10/10 个clusters (100.0%) ✓
- Subset 2: 10/10 个clusters (100.0%) ✓
- Subset 3: 9/10 个clusters (90.0%) ⚠️ **Missing Cluster 8**

## Solutions

### 1.Improved Allocation Strategy (Implemented)

Change the sequential allocation to  **sample count balance-based allocation** :

* For cases where the number of remaining samples ≤ 3, prioritize allocation to the subset with the smallest current sample count
* Ensure that each subset contains representatives of all clusters as much as possible

### 2. Small Cluster Handling Strategies

For very small clusters (≤3 samples):

* **Option A** : Merge into similar large clusters
* **Option B** : Ensure each subset contains at least 1 sample (if possible)
* **Option C** : Use stratified sampling to ensure balanced cluster distribution

### 3. Discussion Suggestions in the Paper

In the **Limitations** section of the paper, it can be discussed:

> "The hierarchical clustering-based data partitioning strategy ensures balanced representation across material characteristics. However, for very small clusters (≤3 samples), complete cluster representation across all validation folds cannot be guaranteed. In this study, Cluster 8 (3 samples) was not represented in Subset 3, which may have contributed to the slightly lower performance of Fold 3 (R²=0.853) compared to Folds 1 and 2 (R²=0.908 and 0.902, respectively). This limitation is inherent to small-sample learning scenarios and highlights the importance of ensuring adequate sample sizes for robust model evaluation."

## Verification of Improvement Effects

After running the improved allocation script, you can:

1. Re-run the cross-validation
2. Check if the performance of Fold 3 is improved
3. Verify whether the cluster coverage of all subsets reaches 100%
