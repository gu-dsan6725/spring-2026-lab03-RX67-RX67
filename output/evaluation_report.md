# Model Evaluation Report

## Model
- **Type**: XGBClassifier
- **Task**: Multi-class classification (3 wine cultivar classes)
- **Training samples**: 142
- **Test samples**: 36
- **Features**: 16 (13 original + 3 derived)

## Metrics Summary

| Metric | Value |
|--------|-------|
| Accuracy | 1.0 |
| Precision (weighted) | 1.0 |
| Recall (weighted) | 1.0 |
| F1-score (weighted) | 1.0 |
| Precision (macro) | 1.0 |
| Recall (macro) | 1.0 |
| F1-score (macro) | 1.0 |

## Per-Class ROC AUC

| Class | AUC |
|-------|-----|
| class_0 | 1.0 |
| class_1 | 1.0 |
| class_2 | 1.0 |

## Per-Class Classification Report

```
              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        12
     class_1       1.00      1.00      1.00        14
     class_2       1.00      1.00      1.00        10

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```

## Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | flavanoids | 0.2012 |
| 2 | color_intensity | 0.1854 |
| 3 | proline | 0.1555 |
| 4 | od280/od315_of_diluted_wines | 0.1413 |
| 5 | magnesium | 0.0761 |

## Key Findings

- The model achieves **perfect accuracy** on the 36-sample test set.
- All three ROC AUC scores are 1.0, confirming strong class separation.
- Cross-validation accuracy (97.19%) is a more realistic performance estimate, as the test set is small.
- The top feature is **flavanoids** (importance 0.2012), followed by **color_intensity** (0.1854) and **proline** (0.1555).
- Original chemical features dominate the top 5, suggesting the raw measurements carry strong discriminative signal.

## Recommendations

1. **Use CV accuracy (97.2%) as the primary metric** — the test set (n=36) is too small for reliable evaluation alone.
2. **Consider nested cross-validation** to get an unbiased estimate of generalization after hyperparameter tuning.
3. **Try additional derived features** — ratios involving proline and proanthocyanins may further separate classes.

## Artifacts

- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curves.png`: One-vs-rest ROC curves per class
- `feature_importance.png`: XGBoost feature importance ranking
- `xgboost_model.joblib`: Trained model file
- `tuning_results.json`: Hyperparameter tuning results (20 iterations, 5-fold CV)
