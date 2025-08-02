!pip install pyarrow

import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = 'xgb_amped_model.json'
SUBMISSION_PATH = 'xgb_amped_submission.csv'
DATA_DICT_PATH  = 'data_dictionary.csv'

# ── 1) Load Parquet data & data dictionary ────────────────────────────────
print("Loading data…")
train = pd.read_parquet('train_data.parquet', engine='pyarrow')
test  = pd.read_parquet('test_data.parquet',  engine='pyarrow')
dd    = pd.read_csv(DATA_DICT_PATH)

# ── Identify key columns ──────────────────────────────────────────────────
id_cols    = ['id1','id2','id3','id4','id5']
target_col = 'y'

# ── 1.1) Coerce target ─────────────────────────────────────────────────────
train[target_col] = pd.to_numeric(train[target_col], errors='coerce')
if train[target_col].isnull().any():
    raise ValueError("Non-numeric target values found in 'y'")
train[target_col] = train[target_col].astype(int)

# ── 2) Dtype casting ───────────────────────────────────────────────────────
for _, row in dd.iterrows():
    col  = row['masked_column']
    kind = str(row['Type']).strip().lower()
    if col not in train.columns:
        continue
    if col == 'id4':
        train[col] = pd.to_datetime(train[col])
        test[col]  = pd.to_datetime(test[col])
    elif kind == 'numerical':
        train[col] = pd.to_numeric(train[col], errors='coerce', downcast='float')
        test[col]  = pd.to_numeric(test[col], errors='coerce', downcast='float')

# ── 3) Impute + flags ──────────────────────────────────────────────────────
num_feats = [c for c in train.columns 
             if train[c].dtype.kind in ('i','u','f') 
             and c not in id_cols + [target_col]]
for col in num_feats:
    train[f"{col}_na"] = train[col].isna().astype(int)
    test[f"{col}_na"]  = test[col].isna().astype(int)
    med = train[col].median()
    train[col].fillna(med, inplace=True)
    test[col].fillna(med, inplace=True)

# ── 4) Datetime feats ──────────────────────────────────────────────────────
for df in (train, test):
    df['hour']       = df['id4'].dt.hour
    df['is_weekend'] = (df['id4'].dt.weekday >= 5).astype(int)
    df['month_sin']  = np.sin(2*np.pi * df['id4'].dt.month / 12)
    df['month_cos']  = np.cos(2*np.pi * df['id4'].dt.month / 12)

# ── 5) Encoding ────────────────────────────────────────────────────────────
cat_feats = [c for c in ['f42','f48','f50','f51','f52','f53','f54',
                         'f55','f56','f57','f349'] if c in train.columns]
low_card  = [c for c in cat_feats if train[c].nunique() <= 10]
high_card = [c for c in cat_feats if train[c].nunique() > 10]

# one-hot
train = pd.get_dummies(train, columns=low_card, dummy_na=True)
test  = pd.get_dummies(test,  columns=low_card, dummy_na=True)
train, test = train.align(test, join='left', axis=1, fill_value=0)

# K-Fold target + frequency encode
def kfold_target_encode(tr, te, col, tgt, n_splits=5, alpha=10):
    avg = tr[tgt].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    te_s = pd.Series(index=tr.index, dtype=float)
    for ti, vi in skf.split(tr, tr[tgt]):
        sub = tr.iloc[ti]
        stats = sub.groupby(col)[tgt].agg(['mean','count'])
        stats['smooth'] = (stats['mean']*stats['count'] + avg*alpha) / (stats['count']+alpha)
        te_s.iloc[vi] = tr.iloc[vi][col].map(stats['smooth'])
    te_s.fillna(avg, inplace=True)
    overall = tr.groupby(col)[tgt].agg(['mean','count'])
    overall['smooth'] = (overall['mean']*overall['count'] + avg*alpha) / (overall['count']+alpha)
    te_test = te[col].map(overall['smooth']).fillna(avg)
    return te_s, te_test

for c in high_card:
    tr_e, te_e = kfold_target_encode(train, test, c, target_col)
    train[c], test[c] = tr_e, te_e
    freq = train[c].value_counts(normalize=True)
    train[f"{c}_freq"] = train[c].map(freq)
    test[f"{c}_freq"]  = test[c].map(freq).fillna(0)

# auto-encode leftover objects
for col in train.select_dtypes('object'):
    train[col], un = pd.factorize(train[col].fillna('m'))
    test[col]  = pd.Categorical(test[col].fillna('m'), categories=un).codes

# ── 6) Prep & DMatrix ─────────────────────────────────────────────────────
features = [c for c in train.columns if c not in id_cols+[target_col]]
X, y     = train[features], train[target_col]
dtest    = xgb.DMatrix(test[features])

# ── 7) RandomizedSearchCV ─────────────────────────────────────────────────
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_dist = {
    'max_depth':        randint(6,16),
    'learning_rate':    uniform(0.005,0.045),
    'subsample':        uniform(0.6,0.4),
    'colsample_bytree': uniform(0.6,0.4),
    'reg_alpha':        uniform(0.0,1.0),
    'reg_lambda':       uniform(0.0,1.0),
}
base = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='auc',
    use_label_encoder=False,
    n_estimators=200,
    seed=42
)
search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=kf,
    verbose=2,
    n_jobs=1,
    random_state=42
)
search.fit(X, y)
best = search.best_estimator_
print("Best params:", search.best_params_)

# ── 8) Calibrate & save ───────────────────────────────────────────────────
calib = CalibratedClassifierCV(best, cv=kf, method='sigmoid')
calib.fit(X, y)
best.get_booster().save_model(MODEL_PATH)

# ── 9) Predict & write submission ─────────────────────────────────────────
preds = calib.predict_proba(test[features])[:,1]
sub   = test[id_cols][['id1','id2','id3','id5']].copy()
sub['pred'] = preds
sub.to_csv(SUBMISSION_PATH, index=False)
print("Done →", SUBMISSION_PATH)