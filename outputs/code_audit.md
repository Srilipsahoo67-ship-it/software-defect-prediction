# Code Audit — Historical Findings

> **Stale audit:** Some findings below have since been fixed, including defect
> count-to-label conversion, training-only threshold selection, and the mismatch
> between the canonical app model and its CK inputs. Re-run the audit before
> treating this file as a current description of the code.

## Files Inspected
- train_models.py, save_best_model.py, validate_pipeline.py, visualize.py
- experiment.py, per_dataset.py, per_project_model.py, high_accuracy_model.py
- predict.py, app.py, requirements.txt
- datasets/ (11 CSV files), outputs/, results/, models/

---

## 1. Data Loading
- All scripts load CSVs from datasets/ and normalise column names to lowercase.
- Label column detected dynamically (bug/defect/class/label).
- **CRITICAL BUG**: The `bug` column in PROMISE datasets is a COUNT (0,1,2,3,...), not binary.
  Current mapping `{"1":1,"0":0}` only maps exact string "1" or "0".
  Rows with bug=2,3,4,... are dropped as NaN → data loss.
  **Fix**: map `bug > 0 → 1`, `bug == 0 → 0` after numeric conversion.
- xalan excluded in most scripts (98.4% defective) — correct decision.

## 2. Feature Engineering
- train_models.py: 20 raw features + 12 engineered = 32 total.
- experiment.py: 20 raw features + 11 engineered = 31 total (slightly different set).
- high_accuracy_model.py: 20 raw + 13 engineered (adds wmc*cbo interaction).
- **LEAKAGE RISK**: In experiment.py, `add_engineered_features()` is called on the full
  `raw_df` BEFORE train/test split. Log transforms are row-wise (safe), but ratios
  involving division are also row-wise (safe). No dataset-level statistics used → OK.
- **INCONSISTENCY**: Different scripts use different engineered feature sets.
  predict.py and app.py use only 7 CK features with no engineering → mismatch with
  the model saved by experiment.py which was trained on 20 raw features.

## 3. Train/Test Splitting
- All scripts use stratified 80/20 split with random_state=42 — correct.
- experiment.py Experiment 8 uses ALL_RAW (20 features) for tuning, but the best
  model from Experiments 1-7 was from Experiment 5 (CK + engineered, 24 features).
  The tuned model uses a DIFFERENT feature set than the best experiment — inconsistency.

## 4. Scaling
- StandardScaler fitted on training set only, applied to test — correct in all scripts.
- train_models.py: scaler fitted BEFORE SMOTE, then SMOTE applied to scaled data — correct.
- experiment.py: same pattern — correct.
- high_accuracy_model.py: scaler fitted on combined (other_projects + target_train) — 
  this is a leakage risk because test project's training data influences the scaler.

## 5. SMOTE
- All scripts apply SMOTE only to training data — correct.
- validate_pipeline.py uses ImbPipeline for CV — correct, no leakage.
- experiment.py Experiments 1-7: SMOTE applied outside CV loop — LEAKAGE in CV scores.
  The cv_score() helper uses ImbPipeline correctly, but run_experiment() applies SMOTE
  before calling evaluate_model() directly, not inside CV.
- per_dataset.py: uses ImbPipeline with StratifiedKFold — correct.

## 6. Cross-Validation
- validate_pipeline.py: correct ImbPipeline CV.
- experiment.py: cv_score() helper is correct but NOT used in the main experiment loop.
  The main experiments evaluate on a fixed test set, not via CV.
- CV results in results/cv_results.csv are computed on SMOTE-balanced data (not raw) —
  this inflates CV scores because SMOTE was applied before CV splitting.

## 7. Hyperparameter Tuning
- experiment.py Experiment 8: Optuna tunes LightGBM on X8_tr_bal (SMOTE-balanced training).
  The CV inside Optuna objective uses StratifiedKFold on already-balanced data — 
  this is acceptable but slightly optimistic.
- RandomizedSearchCV for XGBoost also on balanced data — same note.
- **GOOD**: Neither uses the test set for tuning.

## 8. Threshold Optimization
- train_models.py: threshold tuned on a held-out 15% split from SMOTE training data — correct.
- experiment.py Experiment 8: `best_threshold_cv()` uses CV on training data — correct.
- high_accuracy_model.py: threshold tuned on `X_tr_sc` (training set predictions) — 
  **LEAKAGE**: optimizing threshold on the same data the model was trained on inflates F1.
- per_project_model.py: trains a second model (model2) on a split of training data for
  threshold tuning — correct approach but wasteful.

## 9. Model Selection
- save_best_model.py: selects by ROC-AUC only — suboptimal for imbalanced data.
- experiment.py: selects best by ROC-AUC from all experiments — same issue.
- The "best model" saved to models/ (Experiment 8 LightGBM tuned on ALL_RAW features)
  is DIFFERENT from the best experiment result (Experiment 5 LightGBM on CK+engineered).
  This is a major inconsistency.

## 10. Leakage Risks Summary
| Risk | Location | Severity |
|------|----------|----------|
| Bug label is count not binary → data loss | All scripts | HIGH |
| Scaler fitted on combined train+other in transfer learning | high_accuracy_model.py | MEDIUM |
| Threshold tuned on training predictions | high_accuracy_model.py | HIGH |
| CV on pre-SMOTE-balanced data | experiment.py cv_results | MEDIUM |
| Best model (Exp8) uses different features than best result (Exp5) | experiment.py | HIGH |
| predict.py/app.py use 7 CK features, model trained on 20+ | predict.py, app.py | HIGH |

## 11. Duplicate Handling
- All scripts call drop_duplicates() — correct.
- experiment.py drops duplicates on feature+label subset — correct.

## 12. Project Leakage
- high_accuracy_model.py: scaler fitted on combined (other + target_train) data.
  This means the scaler has seen target project statistics before test evaluation.
  Minor but technically a leakage.

## 13. Test-Set Contamination
- No direct test-set contamination found in main experiments.
- high_accuracy_model.py threshold tuning on training predictions is the closest issue.

## 14. Class Imbalance
- Dataset is ~78% Clean / ~22% Defective (after correct binary conversion).
- SMOTE used in all scripts — appropriate.
- No BalancedRandomForest or class_weight experiments — opportunity for improvement.

## 15. Inconsistencies: Training vs Prediction
- predict.py loads best_model.pkl (from save_best_model.py, trained on 7 CK features).
- app.py loads best_model.pkl + scaler.pkl (same 7 CK features).
- experiment.py saves models/best_defect_prediction_model.pkl (trained on 20 features).
- These are DIFFERENT models with DIFFERENT feature sets — predict.py/app.py are
  inconsistent with the best experiment model.

## 16. Inconsistencies: Training vs Streamlit
- app.py hardcodes "Model: XGBoost" in the idle state tile, but the actual loaded
  model may be RandomForest or XGBoost depending on which script was run last.
- SHAP TreeExplainer in app.py may fail if model is not tree-based.

---

## Current Best Results (from experiment.py)
| Metric | Value |
|--------|-------|
| Model | LightGBM (Experiment 5) |
| Accuracy | 0.8321 |
| ROC-AUC | 0.8151 |
| PR-AUC | 0.6007 |
| F1 | 0.4957 |
| Recall | 0.3958 |
| Precision | 0.6628 |
| MCC | 0.4218 |

**Note**: Recall of 0.3958 means 60% of defective modules are missed.
The main improvement opportunity is increasing Recall/F1 while maintaining AUC.

---

## What Needs to Change
1. Fix bug label: use `pd.to_numeric` then `> 0` for binary conversion.
2. Unify feature set: use CK + engineered features consistently across train/predict/app.
3. Save a single pipeline (scaler + model) so predict.py/app.py cannot mismatch.
4. Fix threshold tuning in high_accuracy_model.py (use validation split, not train).
5. Add BalancedRandomForest and class_weight experiments.
6. Select best model by Defective F1, not ROC-AUC.
7. Add PR-AUC, Balanced Accuracy, MCC to all evaluations.
8. Fix CV: run SMOTE inside CV folds for unbiased CV scores.
