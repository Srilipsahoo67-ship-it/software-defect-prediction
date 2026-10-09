import pandas as pd, glob, os, warnings
warnings.filterwarnings('ignore')
from data_utils import normalize_bug_labels
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

dataset_dir = 'datasets'
for f in sorted(glob.glob(os.path.join(dataset_dir, '*.csv'))):
    df = pd.read_csv(f)
    df.columns = [c.lower().strip() for c in df.columns]
    lbl = next((c for c in df.columns if c in ['bug','defect','class','label']), None)
    if not lbl: continue
    df = df.rename(columns={lbl:'bug'})
    df['bug'] = normalize_bug_labels(df['bug'])
    df = df.dropna(subset=['bug'])
    df['bug'] = df['bug'].astype(int)
    feats = [c for c in ['wmc','dit','noc','cbo','rfc','lcom','loc'] if c in df.columns]
    for col in feats:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    df = df.drop_duplicates()
    X = df[feats].values
    y = df['bug'].values
    total = len(df)
    defrate = round(100*y.sum()/total, 1)
    if y.sum() < 5 or (total - y.sum()) < 5:
        print('%s | n=%d | defect=%.1f%% | acc=SKIPPED (too few samples in one class)' % (os.path.basename(f).replace('.csv',''), total, defrate))
        continue
    pipe = ImbPipeline([('sc',StandardScaler()),('sm',SMOTE(random_state=42)),('m',XGBClassifier(eval_metric='logloss',random_state=42))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    acc = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy').mean()
    print('%s | n=%d | defect=%.1f%% | acc=%.2f%%' % (os.path.basename(f).replace('.csv',''), total, defrate, acc*100))
