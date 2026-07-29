# Software Defect Prediction — Research Workflow

## Abstract
This project predicts software defects using CK object-oriented metrics
extracted from Java projects in the PROMISE repository. Four ML classifiers
(Random Forest, Decision Tree, SVM, XGBoost) are trained and evaluated.

---

## 1. Dataset Collection
- Source     : PROMISE Repository (Jureczko et al.)
- Projects   : ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces
- Format     : CSV
- GitHub     : https://github.com/klainfo/DefectData

## 2. Features (CK Metrics)
| Feature | Description                          |
|---------|--------------------------------------|
| WMC     | Weighted Methods per Class           |
| DIT     | Depth of Inheritance Tree            |
| NOC     | Number of Children                   |
| CBO     | Coupling Between Objects             |
| RFC     | Response For a Class                 |
| LCOM    | Lack of Cohesion in Methods          |
| LOC     | Lines of Code                        |
| Bug     | Defect Label (1=Defective, 0=Clean)  |

## 3. Preprocessing
1. Load all CSV files and merge into one dataframe
2. Standardize column names to lowercase
3. Map defect labels: true/yes → 1, false/no → 0
4. Fill missing values with column median
5. Remove duplicate rows
6. Apply StandardScaler normalization

## 4. Class Imbalance
- PROMISE datasets are typically imbalanced (~20-30% defective)
- Handled via stratified train-test split (80/20)
- Optional: SMOTE oversampling (imbalanced-learn)

## 5. Models Used
| Model         | Key Hyperparameters                    |
|---------------|----------------------------------------|
| Random Forest | n_estimators=100, random_state=42      |
| Decision Tree | random_state=42                        |
| SVM           | kernel=rbf, probability=True           |
| XGBoost       | eval_metric=logloss, random_state=42   |

## 6. Evaluation Metrics
- Accuracy   : Overall correct predictions
- Precision  : TP / (TP + FP)
- Recall     : TP / (TP + FN)
- F1-Score   : Harmonic mean of Precision and Recall
- AUC-ROC    : Area under the ROC curve (best for imbalanced data)

## 7. Visualizations
- Confusion Matrix  → plots/confusion_matrices.png
- ROC Curve         → plots/roc_curves.png
- Feature Importance→ plots/feature_importance.png
- Correlation Heatmap→ plots/correlation_heatmap.png
- Class Distribution→ plots/class_distribution.png

## 8. Results Interpretation
- AUC-ROC > 0.80 → Good model
- AUC-ROC > 0.90 → Excellent model
- F1-Score is the primary metric for imbalanced defect datasets
- Random Forest and XGBoost typically outperform SVM and DT

## 9. How to Run
```
pip install -r requirements.txt
python download_datasets.py
python train_models.py
python visualize.py
```

## 10. References
- Jureczko & Madeyski (2010). Towards Identifying Software Project Clusters
  with Regard to Defect Prediction. PROMISE 2010.
- PROMISE Repository: http://promise.site.uottawa.ca/SERepository/
- GitHub Data Mirror: https://github.com/klainfo/DefectData
