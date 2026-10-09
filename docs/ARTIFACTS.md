# Artifact index

Use this index to identify which generated files belong to which workflow. The folders contain experiments from different stages, so a file's presence does not mean it is the current default result.

## Current POI artifacts

| File | Meaning |
| --- | --- |
| `outputs/poi_f1_results_seed2026.csv` | POI 3.0 candidate table for the seeded within-release workflow. |
| `models/poi_f1_model_seed2026.pkl` | Model bundle selected by training-partition cross-validation F1. |
| `outputs/poi_temporal_2.5_to_3.0_results.csv` | Results for training on POI 2.5 and evaluating on POI 3.0. |
| `models/poi_temporal_model.pkl` | Selected model bundle from the temporal workflow. |
| `models/poi_xgboost_app.pkl` | XGBoost depth-2 app model, trained with the POI 3.0 training partition and verified against the saved held-out row. |

## Demo artifacts

| File | Meaning |
| --- | --- |
| `models/final_defect_prediction_pipeline.pkl` | Earlier unified pipeline used by `predict.py`; the Streamlit app now uses `poi_xgboost_app.pkl`. |
| `models/final_metadata.json` | Metadata for the earlier unified pipeline. |

`best_model.pkl` and `scaler.pkl` at the project root are older artifacts and are not the same pipeline as the one loaded by the current app code. Avoid describing them as the active demo model without checking the consuming script.

## Historical experiment outputs

Files in `outputs/` such as `FINAL_RESULTS.csv`, `results_smote.csv`, `results_tuned.csv`, `push_to_85_results.csv`, and the investigation/audit files document earlier approaches. They are useful for tracing project history, but they do not supersede the explicitly named POI outputs above. `results/`, `plots/`, `experiment_plots/`, `high_acc_models/`, and `per_project_models/` hold historical reports, visuals, or per-project model artifacts.

Keep outputs and model bundles together with the script and configuration that produced them. If rerunning an experiment for evaluation, record the command, date, data files, split or release protocol, and seed alongside the copied result.
