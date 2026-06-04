"""Re-evaluate every cross-cohort experiment (train on 2020 ``old`` →
test on 2025 ``horizontal``) using a SOURCE-derived Youden decision
threshold rather than one fit on the held-out 2025 labels.

For each model in the original Chapter 5 cross-cohort tables the threshold
comes from the corresponding within-cohort 2020 LOOCV CSV that the user
already produced. This is functionally identical to running the patched
``joint_loocv_eval.py`` / ``xgb_loocv_eval.py`` with ``--use_youden
--cross_dataset`` (those scripts now compute the same thresholds inline via
a fresh source LOOCV pass) but completes in seconds because the LOOCV
output is already on disk.

Outputs are written **back into the original result folders** alongside the
existing files, with the same naming convention. The original (leaky) CSVs
and PNGs are first copied to ``<folder>/_backup_pre_youden_fix/`` for
reference.

Generated per configuration:
  - ``metrics_<orig_name>.csv``  (overwritten)
  - ``roc_<orig_name>.png``      (overwritten)
  - ``cm_<orig_name>.png``       (overwritten)

Configurations covered (each writes into the indicated folder):
  - ``joint4_exp/``      thumb-tip (LR / RF / XGB)
  - ``all_joint_exp/``   bilateral all-joint (LR / RF / XGB)
  - ``xgb_select_exp/``  bilateral FS + XGB (ANOVA / L1 / XGB-gain)
  - ``xgb_select_exp/single_hand/`` left- and right-hand FS + XGB
  - ``2020_to_2025_corr/``  graph-concat correlation (3 FS × 3 clf)
"""
from __future__ import annotations
import os
import shutil
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'extact_feature_latest.csv')
BACKUP_DIR_NAME = '_backup_pre_youden_fix'


# -------------------- threshold table --------------------
# Each value is read from a within-cohort 2020 LOOCV metrics CSV
# (column "Threshold"); these ARE the source-LOOCV Youden values.
THRESH = {
    'joint4': {  # joint4_exp/metrics_Joint_4_2020.csv
        'Logistic Regression': 0.3013296410211198,
        'Random Forest':       0.3147832553752067,
        'XGBoost':             0.31961530447006226,
    },
    'bilateral': {  # all_joint_exp/metrics_All_Joints_2020.csv
        'Logistic Regression': 0.26441898718476475,
        'Random Forest':       0.22151871249759136,
        'XGBoost':             0.27945154905319214,
    },
    'fs_bilateral': {  # xgb_select_exp/metrics_Standard_LOOCV_old.csv
        'ANOVA + XGB':              0.4033743,
        'L1-Regularization + XGB':  0.63850087,
        'XGB Importance + XGB':     0.5849607,
    },
    'fs_left_hand': {  # xgb_select_exp/single_hand/metrics_Standard_Left_Hand_LOOCV_old.csv
        'ANOVA + XGB':              0.4033743,
        'L1-Regularization + XGB':  0.49753645,
        'XGB Importance + XGB':     0.7107539,
    },
    'fs_right_hand': {  # xgb_select_exp/single_hand/metrics_Standard_Right_Hand_LOOCV_old.csv
        'ANOVA + XGB':              0.66220194,
        'L1-Regularization + XGB':  0.2501472,
        'XGB Importance + XGB':     0.36605254,
    },
    # 2020_corr/metrics_GraphConcat_LOOCV_old.csv
    # 注意：注意：the within-cohort CSV labels L1 as "L1-Reg" and XGB-FS as
    # "XGB-FS", whereas the cross-cohort CSV uses "L1-Reg + XGB" / "XGB-FS +
    # XGB" etc. We follow the cross-cohort naming below.
    'graph': {
        'ANOVA + XGB':  0.4683791995048523,
        'ANOVA + LR':   0.5077847440238619,
        'ANOVA + RF':   0.43943868641463524,
        'L1-Reg + XGB': 0.6407507658004761,
        'L1-Reg + LR':  0.5371963511176455,
        'L1-Reg + RF':  0.42897108990106403,
        'XGB-FS + XGB': 0.5240277051925659,
        'XGB-FS + LR':  0.5514552142473991,
        'XGB-FS + RF':  0.3865254296987001,
    },
}


# -------------------- shared utilities --------------------
def backup_and_overwrite(target_path):
    """Move the existing file to ``<dir>/_backup_pre_youden_fix/`` if it
    exists and has not been backed up already."""
    d, fn = os.path.split(target_path)
    bdir = os.path.join(d, BACKUP_DIR_NAME)
    os.makedirs(bdir, exist_ok=True)
    if os.path.exists(target_path):
        bk = os.path.join(bdir, fn)
        if not os.path.exists(bk):
            shutil.copy2(target_path, bk)


def load_old_horiz_offmed():
    df = pd.read_csv(CSV)
    mask = df['on_medication'].isin([False, 0])
    df_old = df[(df['dataset_source'] == 'old') & mask].drop_duplicates(
        subset=['patient_id', 'date'], keep='first').reset_index(drop=True)
    df_new = df[(df['dataset_source'] == 'horizontal') & mask].drop_duplicates(
        subset=['patient_id', 'date'], keep='first').reset_index(drop=True)
    meta = ['patient_id', 'date', 'pd_stage', 'on_medication', 'dataset_source']
    y_old = (df_old['pd_stage'].astype(int) > 0).astype(int).values
    y_new = (df_new['pd_stage'].astype(int) > 0).astype(int).values
    X_old = df_old.drop(columns=meta).select_dtypes(include=[np.number]).reset_index(drop=True)
    X_new = df_new.drop(columns=meta).select_dtypes(include=[np.number]).reset_index(drop=True)
    return X_old, y_old, X_new, y_new


def filter_joint4(cols):
    return [c for c in cols if ('joint04_' in c) or ('_j4_' in c)]


def filter_left(cols):
    return [c for c in cols if str(c).startswith('L_')]


def filter_right(cols):
    return [c for c in cols if str(c).startswith('R_')]


def metrics_one(y_true, y_prob, thr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    y_pred = (y_prob >= thr).astype(int)
    return {
        'Threshold': thr,
        'AUROC': roc_auc,
        'Acc': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'CM': confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def write_outputs(folder, base_name, results, title_prefix):
    """Write CSV + ROC + CM into ``folder`` using the original
    ``<prefix>_<base_name>.png`` / ``metrics_<base_name>.csv`` naming.

    ``results``: list of dicts each having Model + metrics_one output."""
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f'metrics_{base_name}.csv')
    roc_path = os.path.join(folder, f'roc_{base_name}.png')
    cm_path  = os.path.join(folder, f'cm_{base_name}.png')

    # backup originals
    for p in (csv_path, roc_path, cm_path):
        backup_and_overwrite(p)

    # ---- CSV ----
    rows = [{
        'Model': r['Model'],
        'Threshold': r['Threshold'],
        'AUROC': r['AUROC'],
        'Acc': r['Acc'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1-score': r['F1-score'],
    } for r in results]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- ROC ----
    fig_roc, ax = plt.subplots(figsize=(8, 8))
    for r in results:
        ax.plot(r['fpr'], r['tpr'], lw=2,
                 label=f"{r['Model']} (AUC={r['AUROC']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title_prefix} ROC (source-derived Youden)',
                  fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    fig_roc.savefig(roc_path, bbox_inches='tight'); plt.close(fig_roc)

    # ---- Confusion matrices ----
    n = len(results)
    fig_cm, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax_i, r in zip(axes, results):
        sns.heatmap(r['CM'], annot=True, fmt='d', cmap='Blues', ax=ax_i,
                     annot_kws={"size": 14})
        ax_i.set_title(f"{r['Model']}\nThresh={r['Threshold']:.3f}",
                        fontsize=11, fontweight='bold')
        ax_i.set_xlabel('Predicted'); ax_i.set_ylabel('True')
        ax_i.set_xticklabels(['Healthy', 'PD'])
        ax_i.set_yticklabels(['Healthy', 'PD'])
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, bbox_inches='tight'); plt.close(fig_cm)

    print(f"  wrote: {csv_path}")
    print(f"         {roc_path}")
    print(f"         {cm_path}")
    return rows


# -------------------- classifier blocks --------------------
def classifiers_classical(spw):
    return [
        ('Logistic Regression', LogisticRegression(C=1.0, solver='lbfgs',
                                                   max_iter=1000,
                                                   class_weight='balanced',
                                                   random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  class_weight='balanced',
                                                  random_state=42, n_jobs=1)),
        ('XGBoost', XGBClassifier(n_estimators=50, max_depth=3,
                                   learning_rate=0.1,
                                   scale_pos_weight=spw,
                                   eval_metric='logloss',
                                   random_state=42, n_jobs=1)),
    ]


def fit_eval_classical(X_tr, y_tr, X_te, y_te, thr_map):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    med = X_tr.median(numeric_only=True)
    X_tr_s = StandardScaler().fit(X_tr.fillna(med))
    sc = StandardScaler(); X_tr_arr = sc.fit_transform(X_tr.fillna(med))
    X_te_arr = sc.transform(X_te.fillna(med))
    out = []
    for name, clf in classifiers_classical(spw):
        clf.fit(X_tr_arr, y_tr)
        prob = clf.predict_proba(X_te_arr)[:, 1]
        r = metrics_one(y_te, prob, thr_map[name]); r['Model'] = name
        out.append(r)
    return out


# ---- feature-selection helpers for FS + XGB ----
def fs_anova(X, y, k=10):
    Xf = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(Xf)
    sel = SelectKBest(f_classif, k=min(k, X.shape[1])).fit(Xs, y)
    s = pd.Series(sel.scores_, index=X.columns).fillna(0).sort_values(ascending=False)
    return s.head(k).index.tolist()


def fs_l1(X, y, k=10, C=0.5):
    Xf = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(Xf)
    m = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                            class_weight='balanced', random_state=42,
                            max_iter=1000).fit(Xs, y)
    s = pd.Series(np.abs(m.coef_[0]), index=X.columns).sort_values(ascending=False)
    return s.head(k).index.tolist()


def fs_xgb(X, y, k=10):
    Xf = X.fillna(X.median(numeric_only=True))
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                       scale_pos_weight=spw, eval_metric='logloss',
                       random_state=42, n_jobs=1).fit(Xf, y)
    imp = m.get_booster().get_score(importance_type='gain')
    s = pd.Series(imp).reindex(X.columns).fillna(0).sort_values(ascending=False)
    return s.head(k).index.tolist()


FS_METHODS_STD = [
    ('ANOVA + XGB',             fs_anova),
    ('L1-Regularization + XGB', fs_l1),
    ('XGB Importance + XGB',    fs_xgb),
]


def fit_eval_fs_xgb(X_tr, y_tr, X_te, y_te, thr_map):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    out = []
    for fs_name, fs_func in FS_METHODS_STD:
        cols = fs_func(X_tr, y_tr, k=10)
        if not cols:
            continue
        clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric='logloss',
                             random_state=42, n_jobs=1)
        clf.fit(X_tr[cols], y_tr)
        prob = clf.predict_proba(X_te[cols])[:, 1]
        thr = thr_map.get(fs_name, 0.5)
        r = metrics_one(y_te, prob, thr); r['Model'] = fs_name
        out.append(r)
    return out


# -------------------- graph-concat block --------------------
def build_graph_features_for_patient(X_n):
    assert X_n.shape == (42, 42)
    X_n = X_n.copy().astype(float)
    for j in range(X_n.shape[1]):
        col = X_n[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            med = np.nanmedian(col)
            X_n[bad, j] = med if np.isfinite(med) else 0.0
    lw = LedoitWolf(); lw.fit(X_n.T)
    cov = lw.covariance_
    diag_sqrt = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    A_corr = cov / np.outer(diag_sqrt, diag_sqrt)
    A = (A_corr + 1.0) / 2.0; np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    d_inv = np.where(deg > 1e-12, deg ** (-0.5), 0.0)
    A_norm = d_inv[:, None] * A * d_inv[None, :]
    H = A_norm @ X_n
    h_mean = H.mean(axis=1); h_std = H.std(axis=1); h_max = H.max(axis=1)
    H_l = H[:21]; H_r = H[21:]
    asym_vec = np.abs(H_l - H_r).mean(axis=1)
    asym_scalar = float(np.linalg.norm(H_l - H_r))
    A_LL = A[:21, :21]; A_RR = A[21:, 21:]; A_LR = A[:21, 21:]
    blk = np.array([
        A_LL.mean(), A_LL.std(), A_LL.max(), A_LL.sum() / 441,
        A_RR.mean(), A_RR.std(), A_RR.max(), A_RR.sum() / 441,
        A_LR.mean(), A_LR.std(), A_LR.max(), A_LR.sum() / 441,
    ])
    coup_asym = float(np.abs(A_LL.mean() - A_RR.mean()))
    return np.concatenate([h_mean, h_std, h_max, asym_vec, [asym_scalar],
                            blk, [coup_asym]])


def build_graph_features(X_df, n_joints=42):
    X = np.array(X_df, dtype=float)
    N, n = X.shape
    npj = max(n // n_joints, 1)
    X = X[:, :n_joints * npj]
    out = []
    for i in range(N):
        Xn = X[i].reshape(n_joints, npj)
        if npj < 42:
            Xn = np.pad(Xn, ((0, 0), (0, 42 - npj)))
        elif npj > 42:
            Xn = Xn[:, :42]
        out.append(build_graph_features_for_patient(Xn))
    return np.array(out)


def graph_classifiers(spw):
    return {
        'XGB': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                              scale_pos_weight=spw, eval_metric='logloss',
                              random_state=42, n_jobs=1),
        'LR':  LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000,
                                   class_weight='balanced', random_state=42),
        'RF':  RandomForestClassifier(n_estimators=200, max_depth=5,
                                       class_weight='balanced',
                                       random_state=42, n_jobs=1),
    }


def fit_eval_graph(X_old, y_old, X_new, y_new, thr_map):
    print("  building graph features (Ledoit-Wolf concat)…")
    Vtr = build_graph_features(X_old)
    Vte = build_graph_features(X_new)
    g_cols = [f'g_{i}' for i in range(Vtr.shape[1])]
    Xtr = pd.concat([
        pd.DataFrame(np.array(X_old, dtype=float), columns=X_old.columns).reset_index(drop=True),
        pd.DataFrame(Vtr, columns=g_cols).reset_index(drop=True),
    ], axis=1)
    Xte = pd.concat([
        pd.DataFrame(np.array(X_new, dtype=float), columns=X_new.columns).reset_index(drop=True),
        pd.DataFrame(Vte, columns=g_cols).reset_index(drop=True),
    ], axis=1)
    print(f"  feature matrix train={Xtr.shape}  test={Xte.shape}")

    fs_methods_graph = [
        ('ANOVA',  fs_anova),
        ('L1-Reg', fs_l1),
        ('XGB-FS', fs_xgb),
    ]
    spw = (y_old == 0).sum() / max((y_old == 1).sum(), 1)
    out = []
    sc = StandardScaler()
    for fs_name, fs_func in fs_methods_graph:
        cols = fs_func(Xtr, y_old, k=10)
        if not cols:
            continue
        Xtr_s = sc.fit_transform(Xtr[cols].values)
        Xte_s = sc.transform(Xte[cols].values)
        for clf_name, clf in graph_classifiers(spw).items():
            key = f"{fs_name} + {clf_name}"
            clf.fit(Xtr_s, y_old)
            prob = clf.predict_proba(Xte_s)[:, 1]
            thr = thr_map.get(key, 0.5)
            r = metrics_one(y_new, prob, thr); r['Model'] = key
            out.append(r)
    return out


# -------------------- main --------------------
def main():
    print("Loading features…")
    X_old, y_old, X_new, y_new = load_old_horiz_offmed()
    print(f"  2020 train: {X_old.shape}, PD={int(y_old.sum())}/{len(y_old)}")
    print(f"  2025 test : {X_new.shape}, PD={int(y_new.sum())}/{len(y_new)}")

    summary = []

    # --- thumb-tip cross-cohort → joint4_exp/ ---
    print("\n[1/6] thumb-tip cross-cohort → joint4_exp/")
    cols4 = filter_joint4(X_old.columns)
    res = fit_eval_classical(X_old[cols4], y_old, X_new[cols4], y_new,
                              THRESH['joint4'])
    rows = write_outputs(os.path.join(ROOT, 'joint4_exp'),
                          'Joint_4_CrossDataset_Old2Horiz', res,
                          'Thumb-tip (joint 4) Cross-cohort')
    for r in rows: r['Config'] = 'thumb-tip'; summary.append(r)

    # --- bilateral all-joint cross-cohort → all_joint_exp/ ---
    print("\n[2/6] bilateral all-joint cross-cohort → all_joint_exp/")
    res = fit_eval_classical(X_old, y_old, X_new, y_new, THRESH['bilateral'])
    rows = write_outputs(os.path.join(ROOT, 'all_joint_exp'),
                          'All_Joints_CrossDataset_Old2Horiz', res,
                          'Bilateral all-joint Cross-cohort')
    for r in rows: r['Config'] = 'bilateral all-joint'; summary.append(r)

    # --- bilateral FS + XGBoost → xgb_select_exp/ ---
    print("\n[3/6] bilateral FS + XGBoost cross-cohort → xgb_select_exp/")
    res = fit_eval_fs_xgb(X_old, y_old, X_new, y_new, THRESH['fs_bilateral'])
    rows = write_outputs(os.path.join(ROOT, 'xgb_select_exp'),
                          'CrossDataset_Standard_CrossDataset_Old2Horiz', res,
                          'Bilateral FS + XGBoost Cross-cohort')
    for r in rows: r['Config'] = 'bilateral FS + XGBoost'; summary.append(r)

    # --- left-hand FS + XGBoost → xgb_select_exp/single_hand/ ---
    print("\n[4/6] left-hand FS + XGBoost cross-cohort → xgb_select_exp/single_hand/")
    cL = filter_left(X_old.columns)
    res = fit_eval_fs_xgb(X_old[cL], y_old, X_new[cL], y_new,
                           THRESH['fs_left_hand'])
    rows = write_outputs(os.path.join(ROOT, 'xgb_select_exp', 'single_hand'),
                          'CrossDataset_Standard_Left_Hand_CrossDataset_Old2Horiz',
                          res, 'Left-hand FS + XGBoost Cross-cohort')
    for r in rows: r['Config'] = 'left-hand FS + XGBoost'; summary.append(r)

    # --- right-hand FS + XGBoost → xgb_select_exp/single_hand/ ---
    print("\n[5/6] right-hand FS + XGBoost cross-cohort → xgb_select_exp/single_hand/")
    cR = filter_right(X_old.columns)
    res = fit_eval_fs_xgb(X_old[cR], y_old, X_new[cR], y_new,
                           THRESH['fs_right_hand'])
    rows = write_outputs(os.path.join(ROOT, 'xgb_select_exp', 'single_hand'),
                          'CrossDataset_Standard_Right_Hand_CrossDataset_Old2Horiz',
                          res, 'Right-hand FS + XGBoost Cross-cohort')
    for r in rows: r['Config'] = 'right-hand FS + XGBoost'; summary.append(r)

    # --- graph-concat (correlation features) → 2020_to_2025_corr/ ---
    print("\n[6/6] graph-concat correlation cross-cohort → 2020_to_2025_corr/")
    res = fit_eval_graph(X_old, y_old, X_new, y_new, THRESH['graph'])
    rows = write_outputs(os.path.join(ROOT, '2020_to_2025_corr'),
                          'GraphConcat_CrossDataset_CrossDataset_Old2Horiz',
                          res, 'Graph-Concat correlation Cross-cohort')
    for r in rows: r['Config'] = 'graph correlation + FS'; summary.append(r)

    # Master summary CSV
    out_dir = os.path.join(ROOT, 'xgb_exp', 'cross_cohort_fixed')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([{
        'Config': r['Config'], 'Model': r['Model'],
        'Threshold': r['Threshold'], 'AUROC': r['AUROC'],
        'Acc': r['Acc'], 'Precision': r['Precision'],
        'Recall': r['Recall'], 'F1-score': r['F1-score'],
    } for r in summary])
    df.to_csv(os.path.join(out_dir, 'cross_cohort_fixed_threshold.csv'),
              index=False)
    print(f"\nMaster summary → {os.path.join(out_dir, 'cross_cohort_fixed_threshold.csv')}")
    pd.set_option('display.width', 180)
    pd.set_option('display.max_colwidth', 45)
    print('\n=== Cross-cohort metrics with SOURCE-derived Youden threshold ===')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
