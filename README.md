# Software Defect Prediction

This project trains machine learning models to classify software classes as defective or non-defective from PROMISE-style software metrics. It includes experiments on individual releases, a POI temporal experiment, and a Streamlit demo.

## Start here

Use Python 3.10 or newer. Install packages from the project root and run the POI experiment:

```powershell
python -m pip install -r requirements.txt
python train_models.py
```

`requirements.txt` and `requirements-app.txt` pin scikit-learn 1.9.1 to match the POI XGBoost model. Generate that app model once from the saved evaluation row, then start Streamlit. The export script verifies that held-out metrics match the existing result table and does not overwrite it.

```powershell
py -m venv .venv-app
.venv-app\Scripts\activate
python -m pip install -r requirements-app.txt
python export_poi_xgboost_app.py
python -m streamlit run app.py
```

The project has multiple research workflows. Follow [docs/PROJECT_EVALUATION.md](docs/PROJECT_EVALUATION.md) for the evaluation narrative, current metrics, limitations, and demo sequence. See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the generated model/result index and [datasets/README.md](datasets/README.md) for dataset roles.

## Workflows

| Command | Purpose | Main artifacts |
| --- | --- | --- |
| `python train_models.py` | Tune and evaluate POI 3.0 candidates; choose the default model from training-fold CV F1. | `outputs/poi_f1_results_seed2026.csv`, `models/poi_f1_model_seed2026.pkl` |
| `python train_poi_temporal.py` | Train on POI 2.5 and evaluate on POI 3.0. | `outputs/poi_temporal_2.5_to_3.0_results.csv`, `models/poi_temporal_model.pkl` |
| `python train_models.py --all-projects` | Run the earlier pooled multi-project workflow. | Files written under `outputs/` (consult script output) |
| `python improved_pipeline.py` | Run the earlier unified multi-project pipeline workflow. | `models/final_defect_prediction_pipeline.pkl` and metadata |
| `python export_poi_xgboost_app.py` | Export the POI 3.0 XGBoost candidate used in the panel demo. | `models/poi_xgboost_app.pkl` |
| `python -m streamlit run app.py` | Start the 20-metric XGBoost interface. | Reads `models/poi_xgboost_app.pkl` |

The Streamlit demo uses the POI XGBoost depth-2, all-20-raw-metrics candidate at threshold 0.52. Its 84.27% accuracy is the held-out POI 3.0 score recorded in the results CSV. The overall model bundle selected by CV F1 remains a different model. Explain this choice clearly when presenting.

## Repository map

Scripts stay at the project root because their paths are written relative to that location.

```text
datasets/                 Input CSV datasets and dataset notes
docs/                     Evaluation and project documentation
models/                   Serialized trained models / pipelines
outputs/                  Current metric tables and generated reports
plots/, experiment_plots/  Visualizations from experiments
results/                  Earlier experiment result files
tests/                    Existing project checks
app.py                    Streamlit interface
predict.py                Command-line prediction entry point
train_poi_model.py        POI 3.0 CV tuning and held-out evaluation
train_poi_temporal.py     POI 2.5 to POI 3.0 temporal evaluation
train_models.py           Main training command and legacy pooled mode
improved_pipeline.py      Earlier unified multi-project pipeline
```

The remaining root Python scripts are diagnostics, visualization, or earlier experimental approaches. Their output folders are retained as research history; they are not all part of the default run.

## Evaluation notes

- The POI 3.0 model selection uses five-fold cross-validation on the training partition; the holdout partition is used for the reported final metrics in that run.
- The temporal experiment fits on POI 2.5 and reports on POI 3.0. POI 3.0 has also appeared in earlier development experiments, so treat this temporal result as exploratory rather than a pristine final validation.
- A classification threshold changes predicted labels and therefore accuracy, precision, recall, and F1. ROC-AUC is calculated from probabilities and does not use the threshold.
- Scores describe these datasets and splits. They are not a guarantee of performance on future projects.
- Do not compare a CV score, a random holdout score, and a temporal score as if they came from the same evaluation protocol.

