# Software Defect Prediction — Corrected Evaluation

## 1. Executive Summary
Best model: **LightGBM_Tuned** (selected by training out-of-fold accuracy)
- Accuracy: 0.6998
- F1: 0.5243
- Recall: 0.4355
- ROC-AUC: 0.7204
- PR-AUC: 0.6237
- MCC: 0.3325

## 2. Dataset Description
11 PROMISE repository Java projects. xalan excluded (98.4% defective).
Total samples after deduplication: 4078
Defect rate: 37.9%

## 3. Historical Result
The previous 0.8321 accuracy result used incompatible label handling and is not a comparable baseline.

## 4. Fixes Applied
- Defect counts are consistently converted to binary labels (bug > 0).
- Canonical app model uses 19 features derivable from seven CK inputs.
- Model and threshold selection use training out-of-fold predictions.
- Experimental full-feature models save separate artifacts.

## 5. Leakage Audit
SMOTE and scaling are inside cross-validation pipelines. Model and threshold
selection use training folds; the held-out test set is not used for selection.

## 6. Class Imbalance Analysis
Clean: 2531 (62.1%)  Defective: 1547 (37.9%)
Strategies tested: No handling, class_weight=balanced, SMOTE, BorderlineSMOTE.

## 7. Feature Engineering
Feature Set C: 19 features (7 CK + log-transforms + ratios).
Feature transforms are row-wise; the dataset audit found no missing metrics.

## 8. Model Comparison
See outputs/cv_results.csv for full CV comparison across all models.

## 9. Hyperparameter Tuning
RandomizedSearchCV uses accuracy scoring in stratified training folds.

## 10. Threshold Optimization
Threshold selected to maximize accuracy on training out-of-fold predictions only.
Best threshold: 0.60

## 11. Per-Project Results
See outputs/per_project_improved_results.csv
Small projects flagged with LOW SAMPLE SIZE warning.

## 12. Cross-Project Results
See outputs/cross_project_results.csv (Leave-One-Project-Out evaluation).

## 13. Error Analysis
See outputs/error_analysis.csv
False Negatives (missed defects): 175
False Positives (false alarms): 70

## 14. Feature Importance / SHAP
See outputs/feature_importance.csv and plots/improved_shap_summary.png

## 15. Best Model
Model: LightGBM_Tuned
Saved: models/final_defect_prediction_pipeline.pkl

## 16. 90% Accuracy Investigation
85% accuracy target achieved: NO (held-out accuracy is 69.98%).
90% accuracy achieved: NO
The current result does not establish that 90% accuracy generalizes. Process
metrics (code churn, commits, authors) and project-aware evaluation may be
worth testing, but any improvement must be checked on unseen projects.

## 17. Limitations
- CK metrics capture structure but not runtime behaviour or developer experience.
- Combined model generalizes across projects but loses project-specific signal.
- Process metrics unavailable in current dataset.

## 18. Recommended Future Work
1. Add process metrics: code churn, number of commits, author count.
2. Project-specific fine-tuning on top of combined model.
3. Deep learning on AST/token sequences (code2vec).
4. Stacking ensemble with calibrated probabilities.
5. Extend to NASA MDP and Eclipse defect datasets.