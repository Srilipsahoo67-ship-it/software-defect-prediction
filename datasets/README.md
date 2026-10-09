# Dataset inventory

The CSV files in this directory are the input data for the project. Keep original datasets unchanged; preprocessing is performed by the Python workflows.

## Primary POI data

| File | Role |
| --- | --- |
| `poi-3.0.csv` | Primary POI 3.0 within-release training/holdout workflow and temporal evaluation target. |
| `poi-2.5.csv` | Training release for the POI temporal experiment. |

Other CSV files cover additional project/release datasets used by earlier pooled and per-project experiments. Their names encode project and release where available. Refer to the corresponding script (`train_models.py --all-projects`, `per_project_model.py`, or `per_dataset.py`) before combining datasets; these scripts do not all apply the same inclusion rules.

## Schema and handling

The POI workflows use software metrics such as WMC, DIT, NOC, CBO, RFC, LCOM, LOC, CA, CE, NPM, LCOM3, DAM, MOA, MFA, CAM, IC, CBM, AMC, maximum cyclomatic complexity, and average cyclomatic complexity when present. A label column named `bug`, `defect`, `class`, or `label` is normalized to a binary target by `data_utils.py`.

Rows are class records. Repeated metric values can occur for different classes and should not be removed as duplicates unless the complete source records are known duplicates. The POI loader retains these class rows.

## Reproducibility

Check the workflow's source code for the exact file list, split seed, label mapping, and preprocessing. The default `python train_models.py` workflow uses POI 3.0; the `--all-projects` option uses a separate pooled workflow. Do not treat their scores as directly comparable without describing their evaluation protocols.

