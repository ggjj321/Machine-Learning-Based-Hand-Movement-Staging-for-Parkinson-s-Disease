import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 定義特徵選擇函式
# ==========================================
def selectkbest_fs(X_train, y_train, k=10):
    X_num = X_train.fillna(X_train.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    k_adjusted = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_adjusted)
    selector.fit(X_scaled, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).fillna(0).sort_values(ascending=False)
    return scores.head(k).index.tolist()

def logistic_l1_fs(X_train, y_train, k=10, C=0.5):
    X_num = X_train.fillna(X_train.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    if len(np.unique(y_train)) < 2: return X_train.columns[:k].tolist()
    model = LogisticRegression(penalty='l1', C=C, solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_scaled, y_train)
    coefs = np.abs(model.coef_[0]) if y_train.ndim == 1 else np.abs(model.coef_).mean(axis=0)
    scores = pd.Series(coefs, index=X_train.columns).sort_values(ascending=False)
    return scores.head(k).index.tolist()

def xgboost_fs(X_train, y_train, k=10):
    X_num = X_train.fillna(X_train.median(numeric_only=True))
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    if len(np.unique(y_train)) < 2: return X_train.columns[:k].tolist()
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                          eval_metric='logloss', random_state=42, n_jobs=1)
    model.fit(X_num, y_train)
    imp_dict = model.get_booster().get_score(importance_type="gain")
    scores = pd.Series(imp_dict).reindex(X_train.columns).fillna(0).sort_values(ascending=False)
    return scores.head(k).index.tolist()

# ==========================================
# 2. 定義計算 Youden's J 與評估指標的輔助函式
# ==========================================
def evaluate_predictions(y_true, y_prob, use_youden=False, external_threshold=None):
    """計算最佳閾值(Youden Index)及相關指標.

    ``external_threshold`` (optional): if not None, the operating point comes
    from this value (e.g. a Youden threshold derived from a source-cohort
    LOOCV pass in cross-cohort mode) and the Youden lookup on (y_true, y_prob)
    is skipped. This is required to avoid fitting the decision threshold on
    the held-out test labels.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 0.0, 0.0, 0.0, confusion_matrix(y_true, (y_prob >= 0.5).astype(int), labels=[0, 1]), [0,1], [0,1]

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if external_threshold is not None:
        best_threshold = float(external_threshold)
    elif use_youden:
        best_threshold = _best_youden_threshold(fpr, tpr, thresholds)
    else:
        best_threshold = 0.5

    y_pred = (y_prob >= best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return best_threshold, roc_auc, acc, prec, rec, f1, cm, fpr, tpr


# ==========================================
# 2b. Source-cohort LOOCV Youden helpers (for cross-dataset)
# ==========================================
def _best_youden_threshold(fpr, tpr, thresholds):
    return float(thresholds[np.argmax(tpr - fpr)])


def _youden_from_oof(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return _best_youden_threshold(fpr, tpr, thr)


def compute_source_loocv_youden_fs_xgb(X_train, y_train_bin, fs_methods,
                                       k_features, y_train_mul=None,
                                       use_lda=False):
    """Run LOOCV on the SOURCE (training) cohort with the same
    feature-selection + XGBoost path that ``run_standard_cross_dataset`` uses
    on the full source, and return ``{method_name: Youden_threshold}``.
    """
    n = len(y_train_bin)
    y_arr = y_train_bin.values if hasattr(y_train_bin, 'values') else np.asarray(y_train_bin)
    if y_train_mul is not None:
        y_mul_arr = y_train_mul.values if hasattr(y_train_mul, 'values') else np.asarray(y_train_mul)
    else:
        y_mul_arr = None

    method_names = [m[0] for m in fs_methods]
    if use_lda:
        method_names.append('LDA (Multi) + XGB')
    oof = {name: np.full(n, 0.5) for name in method_names}

    loo = LeaveOneOut()
    for tr_idx, te_idx in tqdm(loo.split(X_train), total=n,
                                desc="Source LOOCV (for cross-cohort threshold)"):
        X_tr = X_train.iloc[tr_idx]
        X_te = X_train.iloc[te_idx]
        y_tr = y_arr[tr_idx]
        for fs_name, fs_func in fs_methods:
            try:
                p = train_and_predict_xgb(X_tr, y_tr, X_te, fs_func, k_features, None)
            except Exception:
                p = 0.5
            oof[fs_name][te_idx[0]] = p
        if use_lda and y_mul_arr is not None:
            try:
                p = train_and_predict_lda_xgb(X_tr, y_tr, y_mul_arr[tr_idx], X_te)
            except Exception:
                p = 0.5
            oof['LDA (Multi) + XGB'][te_idx[0]] = p

    thresholds = {name: _youden_from_oof(y_arr, oof[name]) for name in method_names}
    print("\n[Source-LOOCV Youden thresholds | Standard Cross-Dataset]")
    for k, v in thresholds.items():
        print(f"  {k:<26s}  thr={v:.4f}")
    return thresholds

# ==========================================
# 3. XGBoost 訓練與預測核心輔助函式
# ==========================================
def train_and_predict_xgb(X_train, y_train, X_test, fs_func, k_features, feature_counter_dict=None):
    """特徵選擇後訓練 XGBoost 並回傳預測機率"""
    top_cols = fs_func(X_train, y_train, k=k_features)
    
    if feature_counter_dict is not None and top_cols:
        feature_counter_dict.update(top_cols)

    if not top_cols:
         return 0.5
         
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
    
    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                        eval_metric='logloss', random_state=42, n_jobs=1)
    
    clf.fit(X_train[top_cols], y_train)
    prob = clf.predict_proba(X_test[top_cols])[:, 1][0]
    return prob

def train_and_predict_lda_xgb(X_train, y_train_bin, y_train_mul, X_test):
    """LDA 降維後訓練 XGBoost 並回傳預測機率"""
    if len(np.unique(y_train_mul)) < 2:
        return 0.5
    
    train_medians = X_train.median(numeric_only=True)
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train_mul)

    X_train_lda = lda.transform(X_train_scaled)
    X_test_lda = lda.transform(X_test_scaled)

    scale_pos_weight = np.sum(y_train_bin == 0) / np.sum(y_train_bin == 1) if np.sum(y_train_bin == 1) > 0 else 1.0
    
    clf_lda = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                            eval_metric='logloss', random_state=42, n_jobs=1)
    clf_lda.fit(X_train_lda, y_train_bin)

    return clf_lda.predict_proba(X_test_lda)[:, 1][0]


def _make_graph_classifiers(spw):
    return {
        "XGB": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=1,
        ),
        "LR": LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=2000,
            class_weight='balanced', random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight='balanced',
            random_state=42, n_jobs=1,
        ),
    }


# ==========================================
# 4. 評估流程：一般 LOOCV (所有特徵混合)
# ==========================================
def run_standard_loocv(X, y_binary, y_multi, args, dataset_name, fs_methods, hand_label="Both"):
    exp_name = "Standard_LOOCV" if hand_label == "Both" else f"Standard_{hand_label}_Hand_LOOCV"
    print(f"\n--- 開始執行 {exp_name} (總樣本數: {len(y_binary)}) ---")
    
    all_method_names = [m[0] for m in fs_methods]
    if getattr(args, 'use_lda', False):
        all_method_names.append("LDA (Multi) + XGB")
        
    aggregated_results = {name: {'y_true': [], 'y_prob': []} for name in all_method_names} 
    feature_counters = {name: Counter() for name in [m[0] for m in fs_methods]}
    
    loo = LeaveOneOut()
    
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="Standard LOOCV"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_bin, y_test_bin = y_binary.iloc[train_index].values, y_binary.iloc[test_index].values
        y_train_mul = y_multi.iloc[train_index].values 

        y_test_val = y_test_bin[0]

        # 1. 執行基礎特徵選擇法
        for name, fs_func in fs_methods:
            try:
                prob = train_and_predict_xgb(X_train, y_train_bin, X_test, fs_func, args.k_features, feature_counters[name])
                aggregated_results[name]['y_true'].append(y_test_val)
                aggregated_results[name]['y_prob'].append(prob)
            except Exception:
                aggregated_results[name]['y_true'].append(y_test_val)
                aggregated_results[name]['y_prob'].append(0.5)

        # 2. 執行 LDA 方法
        if getattr(args, 'use_lda', False):
            lda_name = "LDA (Multi) + XGB"
            try:
                prob = train_and_predict_lda_xgb(X_train, y_train_bin, y_train_mul, X_test)
                aggregated_results[lda_name]['y_true'].append(y_test_val)
                aggregated_results[lda_name]['y_prob'].append(prob)
            except Exception:
                aggregated_results[lda_name]['y_true'].append(y_test_val)
                aggregated_results[lda_name]['y_prob'].append(0.5)
            
    # 繪製圖表與產生報表
    plot_and_report_results(aggregated_results, exp_name, dataset_name, args)
    
    # 新增 Top 2 特徵散佈圖 (只有當包含雙手或 Both 時才繪製)
    if hand_label == "Both":
        X_dict = {name: X for name in feature_counters.keys()}
        plot_scatter_top2(feature_counters, X_dict, y_binary, exp_name, dataset_name, args)
    
    # 顯示特徵頻率
    print_top_features(feature_counters, args.k_features, len(y_binary))


# ==========================================
# 5b. 評估流程：Cross-Dataset Evaluation (Old -> Horizontal)
# ==========================================
def run_standard_cross_dataset(X_train, y_train_bin, y_train_mul, X_test, y_test_bin, args, dataset_name, fs_methods, hand_label="Both"):
    exp_name = "CrossDataset_Standard" if hand_label == "Both" else f"CrossDataset_Standard_{hand_label}_Hand"
    print(f"\n--- 開始執行 {exp_name} (Train: {len(y_train_bin)}, Test: {len(y_test_bin)}) ---")
    
    all_method_names = [m[0] for m in fs_methods]
    if getattr(args, 'use_lda', False):
        all_method_names.append("LDA (Multi) + XGB")
        
    aggregated_results = {name: {'y_true': y_test_bin.tolist(), 'y_prob': []} for name in all_method_names} 
    feature_counters = {name: Counter() for name in [m[0] for m in fs_methods]}
    
    for name, fs_func in fs_methods:
        try:
            top_cols = fs_func(X_train, y_train_bin, k=args.k_features)
            if top_cols:
                feature_counters[name].update(top_cols)
                scale_pos_weight = np.sum(y_train_bin == 0) / np.sum(y_train_bin == 1) if np.sum(y_train_bin == 1) > 0 else 1.0
                clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                                    eval_metric='logloss', random_state=42, n_jobs=1)
                clf.fit(X_train[top_cols], y_train_bin)
                prob = clf.predict_proba(X_test[top_cols])[:, 1]
            else:
                prob = np.full(len(X_test), 0.5)
            aggregated_results[name]['y_prob'] = prob.tolist()
        except:
            aggregated_results[name]['y_prob'] = np.full(len(X_test), 0.5).tolist()

    if getattr(args, 'use_lda', False):
        lda_name = "LDA (Multi) + XGB"
        try:
            train_medians = X_train.median(numeric_only=True)
            X_train_filled = X_train.fillna(train_medians)
            X_test_filled = X_test.fillna(train_medians)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_filled)
            X_test_scaled = scaler.transform(X_test_filled)

            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_scaled, y_train_mul)
            X_train_lda = lda.transform(X_train_scaled)
            X_test_lda = lda.transform(X_test_scaled)

            scale_pos_weight = np.sum(y_train_bin == 0) / np.sum(y_train_bin == 1) if np.sum(y_train_bin == 1) > 0 else 1.0
            clf_lda = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                                    eval_metric='logloss', random_state=42, n_jobs=1)
            clf_lda.fit(X_train_lda, y_train_bin)
            prob = clf_lda.predict_proba(X_test_lda)[:, 1]
            aggregated_results[lda_name]['y_prob'] = prob.tolist()
        except Exception:
            aggregated_results[lda_name]['y_prob'] = np.full(len(X_test), 0.5).tolist()

    # ---- Derive Youden threshold from a LOOCV pass on the SOURCE cohort,
    # so the operating point applied to the held-out target cohort is not
    # tuned on test labels.
    src_thresholds = None
    if args.use_youden:
        src_thresholds = compute_source_loocv_youden_fs_xgb(
            X_train, y_train_bin, fs_methods, args.k_features,
            y_train_mul=y_train_mul, use_lda=getattr(args, 'use_lda', False))

    plot_and_report_results(aggregated_results, exp_name, dataset_name, args,
                             thresholds_override=src_thresholds)
    
    if hand_label == "Both":
        X_dict = {name: X_train for name in feature_counters.keys()}
        plot_scatter_top2(feature_counters, X_dict, y_train_bin, exp_name, dataset_name, args)
        
    print_top_features(feature_counters, args.k_features, 1)

# ==========================================
# 6. 視覺化及輸出輔助函式
# ==========================================
def _dataset_prefix(args, dataset_name):
    if getattr(args, 'cross_dataset', False):
        return "cross dataset"
    if dataset_name == "horizontal":
        return "2025"
    if dataset_name == "old":
        return "2020"
    return dataset_name


def plot_and_report_results(results_dict, exp_name, dataset_name, args,
                            thresholds_override=None):
    """結果報表、ROC 曲線與混淆矩陣視覺化統一介面.

    ``thresholds_override`` (optional dict: model_name → float): if a model
    name is present in this dict, the supplied value is used as the decision
    threshold instead of computing Youden on (y_true, y_prob). Cross-dataset
    runners use this to apply a source-LOOCV-derived threshold.
    """
    table_results = []

    os.makedirs(args.save_dir, exist_ok=True)
    n_models = len(results_dict)

    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 8))

    fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1: axes_cm = [axes_cm]

    for i, (name, data) in enumerate(results_dict.items()):
        y_true_all = np.array(data['y_true'])
        y_prob_all = np.array(data['y_prob'])

        ext_thr = None
        if thresholds_override is not None and name in thresholds_override:
            ext_thr = thresholds_override[name]
        thresh, roc_auc, acc, prec, rec, f1, cm, fpr, tpr = evaluate_predictions(
            y_true_all, y_prob_all, use_youden=args.use_youden,
            external_threshold=ext_thr)

        table_results.append({
            'Model': name,
            'Threshold': thresh,
            'AUROC': roc_auc,
            'Acc': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })

        if len(np.unique(y_true_all)) > 1:
            ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i], annot_kws={"size": 14})
        axes_cm[i].set_title(f"{name}\nThresh={thresh:.3f}", fontsize=12, fontweight='bold')
        axes_cm[i].set_xlabel('Predicted')
        axes_cm[i].set_ylabel('True')
        axes_cm[i].set_xticklabels(['Healthy', 'PD'])
        axes_cm[i].set_yticklabels(['Healthy', 'PD'])

    prefix = _dataset_prefix(args, dataset_name)

    # ROC Figure 設定
    ax_roc.plot([0, 1], [0, 1], 'k--')
    # 移除 "Standard" 標示、重複的 CrossDataset 標示並處理底線
    exp_label = exp_name.replace('Standard_', '')
    if prefix == "cross dataset":
        exp_label = exp_label.replace('CrossDataset_', '').replace('CrossDataset', '')
    exp_label = exp_label.replace('_', ' ').strip()
    
    ax_roc.set_title(f'{prefix} {exp_label} ROC Curves', fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect('equal') 

    # 儲存與顯示
    fig_suffix = f"{exp_name.replace(' ', '_')}_{dataset_name}"
    fig_roc.savefig(os.path.join(args.save_dir, f'roc_{fig_suffix}.png'), bbox_inches='tight')
    fig_cm.savefig(os.path.join(args.save_dir, f'cm_{fig_suffix}.png'), bbox_inches='tight')
        
    plt.tight_layout()
    if not args.no_show:
        plt.show()

    df_res = pd.DataFrame(table_results)
    cols_order = ['Model', 'Threshold', 'AUROC', 'Acc', 'Precision', 'Recall', 'F1-score']
    print(f"\n=== Performance Report: {prefix} | {exp_label} ===")
    print(df_res[cols_order].round(4).to_string(index=False))
    
    df_res[cols_order].to_csv(os.path.join(args.save_dir, f'metrics_{fig_suffix}.csv'), index=False)

def plot_scatter_top2(feature_counters_dict, X_dict, y_binary, exp_name, dataset_name, args):
    """
    根據紀錄的 feature counters 繪製出每個方法 Top 2 特徵的 2D 散佈圖。
    """
    prefix = _dataset_prefix(args, dataset_name)

    # 移除 "Standard" 標示、重複的 CrossDataset 標示並處理底線
    exp_label = exp_name.replace('Standard_', '')
    if prefix == "cross dataset":
        exp_label = exp_label.replace('CrossDataset_', '').replace('CrossDataset', '')
    exp_label = exp_label.replace('_', ' ').replace('-', ' ').strip()

    for name, counter in feature_counters_dict.items():
        if name not in X_dict:
            continue
        X_df = X_dict[name]
        most_common = counter.most_common(2)
        
        # 確保至少有兩個特徵才能畫 2D Scatter Plot
        if len(most_common) >= 2:
            top1_feat, top2_feat = most_common[0][0], most_common[1][0]
            
            # 確保欄位名稱正確存在
            if top1_feat in X_df.columns and top2_feat in X_df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # y_binary 通常是 pandas Series，可以對應 X_df 的行順序
                scatter_data = pd.DataFrame({
                    top1_feat: X_df[top1_feat].values,
                    top2_feat: X_df[top2_feat].values,
                    'Label': ['PD' if y == 1 else 'Healthy' for y in y_binary]
                })
                
                sns.scatterplot(
                    data=scatter_data, 
                    x=top1_feat, 
                    y=top2_feat, 
                    hue='Label', 
                    palette={'Healthy': 'blue', 'PD': 'red'}, 
                    ax=ax
                )
                
                ax.set_title(f"{name} Top 2 Features\n{prefix} {exp_label}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                safe_name = name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
                safe_exp = exp_name.replace(' ', '_')
                fig_suffix = f"{safe_exp}_{safe_name}_{dataset_name}"
                
                fig.savefig(os.path.join(args.save_dir, f'scatter_{fig_suffix}.png'), bbox_inches='tight')
                plt.close(fig)

def print_top_features(feature_counters, k_features, total_folds):
    print("\n" + "="*50)
    print(f"=== Top {k_features} Most Frequently Selected Features ===")
    print("="*50)

    for name, counter in feature_counters.items():
        print(f"\n>> Method: {name}")

        most_common = counter.most_common(k_features)

        if not most_common:
            print("  No features were selected.")
        else:
            df_counts = pd.DataFrame(most_common, columns=['Feature Name', 'Selection Count'])
            df_counts['Frequency (%)'] = (df_counts['Selection Count'] / total_folds * 100).round(1)
            print(df_counts.to_string(index=False))


# ==========================================
# 6b. Graph Feature Construction (GCN-style)
# ==========================================

def build_graph_features_for_patient(X_n):
    """
    X_n: (42, 42) — 42 joints × 42 features.
    Returns a ~161-dim feature vector capturing graph structure and asymmetry.
    """
    assert X_n.shape == (42, 42), f"Expect (nodes=42, features=42), got {X_n.shape}"

    X_n = X_n.copy().astype(float)
    # Fill NaN / Inf with per-column median, fallback to 0
    for j in range(X_n.shape[1]):
        col = X_n[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            med = np.nanmedian(col)
            X_n[bad, j] = med if np.isfinite(med) else 0.0

    # Step 1.2: Ledoit-Wolf shrinkage covariance → correlation matrix
    lw = LedoitWolf()
    lw.fit(X_n.T)          # fit on (42 features × 42 nodes)
    cov = lw.covariance_   # (42, 42)
    diag_sqrt = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    denom = np.outer(diag_sqrt, diag_sqrt)
    A_corr = cov / denom   # correlation, values in [-1, 1]

    # Step 1.3: Map to non-negative [0, 1]
    A = (A_corr + 1.0) / 2.0
    np.fill_diagonal(A, 0.0)

    # Step 1.5: Symmetric normalisation (GCN-style)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 1e-12, deg ** (-0.5), 0.0)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    # Step 1.6: Graph fusion — each node aggregates neighbour info
    H = A_norm @ X_n       # (42, 42)

    # Step 1.7: Pooling
    h_mean = H.mean(axis=1)   # 42
    h_std  = H.std(axis=1)    # 42
    h_max  = H.max(axis=1)    # 42

    # Step 1.8: Left / Right asymmetry (PD hallmark)
    H_left  = H[:21]           # (21, 42)
    H_right = H[21:]           # (21, 42)
    asymmetry_vec    = np.abs(H_left - H_right).mean(axis=1)   # 21
    asymmetry_scalar = float(np.linalg.norm(H_left - H_right)) # 1

    # Step 1.9: Block-level graph statistics
    A_LL = A[:21, :21]
    A_RR = A[21:, 21:]
    A_LR = A[:21, 21:]
    block_stats = np.array([
        A_LL.mean(), A_LL.std(), A_LL.max(), A_LL.sum() / (21 * 21),
        A_RR.mean(), A_RR.std(), A_RR.max(), A_RR.sum() / (21 * 21),
        A_LR.mean(), A_LR.std(), A_LR.max(), A_LR.sum() / (21 * 21),
    ])  # 12
    coupling_asymmetry = float(np.abs(A_LL.mean() - A_RR.mean()))  # 1

    # Step 1.10: Concatenate → 161 dims
    return np.concatenate([
        h_mean, h_std, h_max,
        asymmetry_vec, [asymmetry_scalar],
        block_stats, [coupling_asymmetry],
    ])


def build_graph_features(X_patients, n_joints=42):
    """
    X_patients : (N, n_features) DataFrame or ndarray.
    Reshapes each row into (n_joints, k) then pads/truncates k to 42
    before calling build_graph_features_for_patient.
    Returns V_all of shape (N, 161).
    """
    X = np.array(X_patients, dtype=float)
    N, n_feats = X.shape
    n_feats_per_joint = max(n_feats // n_joints, 1)
    X = X[:, : n_joints * n_feats_per_joint]

    vectors = []
    for i in range(N):
        X_n = X[i].reshape(n_joints, n_feats_per_joint)
        if n_feats_per_joint < 42:
            X_n = np.pad(X_n, ((0, 0), (0, 42 - n_feats_per_joint)))
        elif n_feats_per_joint > 42:
            X_n = X_n[:, :42]
        vectors.append(build_graph_features_for_patient(X_n))

    return np.array(vectors)  # (N, 161)


_GRAPH_FEAT_NAMES = (
    [f"gcn_mean_{i}" for i in range(42)] +
    [f"gcn_std_{i}"  for i in range(42)] +
    [f"gcn_max_{i}"  for i in range(42)] +
    [f"asym_vec_{i}" for i in range(21)] +
    ["asym_scalar"] +
    [f"blk_LL_{s}" for s in ("mean", "std", "max", "density")] +
    [f"blk_RR_{s}" for s in ("mean", "std", "max", "density")] +
    [f"blk_LR_{s}" for s in ("mean", "std", "max", "density")] +
    ["coupling_asym"]
)  # 161


# ==========================================
# 6c. Joint–Stage Spearman Analysis
# ==========================================

def _build_adjacency_matrices(X, n_joints=42):
    """
    Return all_A of shape (N, n_joints, n_joints) — one Ledoit-Wolf
    correlation matrix per patient, mapped to non-negative [0, 1].
    """
    X_arr = np.array(X, dtype=float)
    N, n_feats = X_arr.shape
    n_fpj = max(n_feats // n_joints, 1)
    X_arr = X_arr[:, : n_joints * n_fpj]

    all_A = np.zeros((N, n_joints, n_joints))
    for idx in range(N):
        X_n = X_arr[idx].reshape(n_joints, n_fpj)
        if n_fpj < 42:
            X_n = np.pad(X_n, ((0, 0), (0, 42 - n_fpj)))
        elif n_fpj > 42:
            X_n = X_n[:, :42]
        X_n = X_n.copy()
        for j in range(X_n.shape[1]):
            bad = ~np.isfinite(X_n[:, j])
            if bad.any():
                med = np.nanmedian(X_n[:, j])
                X_n[bad, j] = med if np.isfinite(med) else 0.0
        lw = LedoitWolf()
        lw.fit(X_n.T)
        cov = lw.covariance_
        d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        A_corr = cov / np.outer(d, d)
        A = (A_corr + 1.0) / 2.0
        np.fill_diagonal(A, 0.0)
        all_A[idx] = A
    return all_A


def analyze_joint_stage_spearman(X, y_stage, dataset_name, args, n_joints=42, joint_names=None):
    """
    Compute Spearman ρ between every joint-pair similarity A[i,j] and PD stage.
    Produces a heatmap (all pairs + significant-only) and a ranked table.

    Parameters
    ----------
    X        : (N, n_features) DataFrame / ndarray
    y_stage  : (N,) array-like of continuous PD stage values
    """
    print(f"\n--- Joint–Stage Spearman Analysis (dataset: {dataset_name}) ---")
    all_A   = _build_adjacency_matrices(X, n_joints)
    y_s     = np.array(y_stage, dtype=float)
    N       = len(y_s)

    if joint_names is None:
        joint_names = [f"L{i}" for i in range(21)] + [f"R{i}" for i in range(21)]

    print(f"Running Spearman on {n_joints*(n_joints-1)//2} joint pairs × {N} patients …")

    rho_mat  = np.zeros((n_joints, n_joints))
    pval_mat = np.ones((n_joints, n_joints))

    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            rho, pval = spearmanr(all_A[:, i, j], y_s)
            rho_mat[i, j]  = rho_mat[j, i]  = rho
            pval_mat[i, j] = pval_mat[j, i] = pval

    # --- ranked table ---
    pairs   = [(i, j) for i in range(n_joints) for j in range(i + 1, n_joints)]
    rho_vec = np.array([rho_mat[i, j]  for i, j in pairs])
    p_vec   = np.array([pval_mat[i, j] for i, j in pairs])
    order   = np.argsort(np.abs(rho_vec))[::-1]

    n_report = 20
    print(f"\n{'Rank':<5} {'Joint A':<8} {'Joint B':<8} {'Rho':>8} {'p-value':>12}  Sig")
    print("-" * 52)
    for rank, k in enumerate(order[:n_report]):
        i, j   = pairs[k]
        sig    = ("***" if p_vec[k] < 0.001 else
                  "**"  if p_vec[k] < 0.01  else
                  "*"   if p_vec[k] < 0.05  else "")
        print(f"{rank+1:<5} {joint_names[i]:<8} {joint_names[j]:<8} "
              f"{rho_vec[k]:>8.4f} {p_vec[k]:>12.4e}  {sig}")

    # --- heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    kw = dict(cmap='RdBu_r', center=0, vmin=-1, vmax=1,
              xticklabels=joint_names, yticklabels=joint_names,
              cbar_kws={'label': 'Spearman ρ'})

    sns.heatmap(rho_mat, ax=axes[0], **kw)
    axes[0].set_title(f"All Joint Pairs — Spearman ρ vs Stage\n({dataset_name})",
                      fontweight='bold')

    sig_rho = np.where(pval_mat < 0.05, rho_mat, 0.0)
    sns.heatmap(sig_rho, ax=axes[1], **kw)
    axes[1].set_title(f"Significant Pairs Only (p < 0.05)\n({dataset_name})",
                      fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0,  labelsize=6)
        ax.axhline(21, color='black', lw=1.5)
        ax.axvline(21, color='black', lw=1.5)

    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'spearman_joint_stage_{dataset_name}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"Saved → {out_path}")

    return rho_mat, pval_mat


# ==========================================
# 6d. Graph-Concat LOOCV (original + graph features + FS + XGB / LR / RF)
# ==========================================

def run_graph_loocv(X, y_binary, args, dataset_name, hand_label="Both",
                    y_stage=None, fs_methods=None):
    exp_name = "GraphConcat_LOOCV" if hand_label == "Both" else f"GraphConcat_{hand_label}_LOOCV"
    print(f"\n--- 開始執行 {exp_name} (總樣本數: {len(y_binary)}) ---")
    print("Building graph features (Ledoit-Wolf + GCN-style fusion)...")

    try:
        V_all = build_graph_features(X)
        print(f"Graph feature matrix: {V_all.shape}  (target: N × 161)")
    except Exception as e:
        print(f"Graph feature construction failed: {e}")
        return

    # --- Concatenate original features + graph features ---
    orig_cols  = [str(c) for c in X.columns] if hasattr(X, 'columns') else [f'orig_{i}' for i in range(np.array(X).shape[1])]
    graph_cols = [f'g_{n}' for n in _GRAPH_FEAT_NAMES[:V_all.shape[1]]]
    X_orig_df  = pd.DataFrame(np.array(X, dtype=float), columns=orig_cols).reset_index(drop=True)
    X_graph_df = pd.DataFrame(V_all, columns=graph_cols).reset_index(drop=True)
    X_concat   = pd.concat([X_orig_df, X_graph_df], axis=1)
    print(f"Concatenated feature matrix: {X_concat.shape}  "
          f"({len(orig_cols)} original + {len(graph_cols)} graph)")

    # --- Feature-selection methods ---
    if fs_methods is None:
        fs_methods = [
            ("ANOVA",  selectkbest_fs),
            ("L1-Reg", logistic_l1_fs),
            ("XGB-FS", xgboost_fs),
        ]

    combo_keys = [f"{fs} + {clf}" for fs, _ in fs_methods for clf in ("XGB", "LR", "RF")]
    aggregated_results = {k: {'y_true': [], 'y_prob': []} for k in combo_keys}
    # feature counters indexed by fs_name (same selection applies to all 3 classifiers)
    feature_counters   = {fs: Counter() for fs, _ in fs_methods}
    # per-combo: list of (selected_col_names, importance_array)
    imp_records        = {k: [] for k in combo_keys}

    loo    = LeaveOneOut()
    scaler = StandardScaler()

    for train_idx, test_idx in tqdm(loo.split(X_concat), total=len(X_concat), desc=exp_name):
        X_tr = X_concat.iloc[train_idx]
        X_te = X_concat.iloc[test_idx]
        y_tr = y_binary.iloc[train_idx].values
        y_te = y_binary.iloc[test_idx].values[0]
        fold_spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        clfs     = _make_graph_classifiers(fold_spw)

        for fs_name, fs_func in fs_methods:
            try:
                top_cols = fs_func(X_tr, y_tr, k=args.k_features)
            except Exception:
                top_cols = []

            if not top_cols:
                for clf_name in clfs:
                    key = f"{fs_name} + {clf_name}"
                    aggregated_results[key]['y_true'].append(y_te)
                    aggregated_results[key]['y_prob'].append(0.5)
                continue

            feature_counters[fs_name].update(top_cols)

            X_tr_sc = scaler.fit_transform(X_tr[top_cols].values)
            X_te_sc = scaler.transform(X_te[top_cols].values)

            for clf_name, clf in clfs.items():
                key = f"{fs_name} + {clf_name}"
                try:
                    clf.fit(X_tr_sc, y_tr)
                    prob = clf.predict_proba(X_te_sc)[:, 1][0]
                    aggregated_results[key]['y_true'].append(y_te)
                    aggregated_results[key]['y_prob'].append(prob)
                    if hasattr(clf, 'feature_importances_'):
                        imp_records[key].append((top_cols, clf.feature_importances_))
                    elif hasattr(clf, 'coef_'):
                        imp_records[key].append((top_cols, np.abs(clf.coef_[0])))
                except Exception:
                    aggregated_results[key]['y_true'].append(y_te)
                    aggregated_results[key]['y_prob'].append(0.5)

    plot_and_report_results(aggregated_results, exp_name, dataset_name, args)

    # --- Feature frequency report ---
    fs_counters_display = {f"{fs} (all clfs)": cnt for fs, cnt in feature_counters.items()}
    print_top_features(fs_counters_display, args.k_features, len(y_binary))

    # --- Top-feature importance per combo ---
    print("\n=== GraphConcat Top Selected Features (by avg importance) ===")
    for key, records in imp_records.items():
        if not records:
            continue
        # Aggregate: sum importance per column name across folds
        col_imp = Counter()
        col_cnt = Counter()
        for cols, imp_arr in records:
            for col, imp in zip(cols, imp_arr):
                col_imp[col] += imp
                col_cnt[col] += 1
        avg_col_imp = {c: col_imp[c] / col_cnt[c] for c in col_imp}
        top = sorted(avg_col_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n[{key}]")
        for rank, (col, imp) in enumerate(top):
            tag = " ← graph" if col.startswith("g_") else ""
            print(f"  {rank+1:2d}. {col:<35s}  {imp:.4f}{tag}")

    # --- Spearman: joint relationship vs stage ---
    if y_stage is not None:
        analyze_joint_stage_spearman(X, y_stage, dataset_name, args)


# ==========================================
# 6e. Graph-Concat Cross-Dataset
# ==========================================

def run_graph_cross_dataset(X_train, X_test, y_train_bin, y_test_bin,
                             args, dataset_name, hand_label="Both",
                             y_train_stage=None, fs_methods=None):
    exp_name = "GraphConcat_CrossDataset" if hand_label == "Both" else f"GraphConcat_CrossDataset_{hand_label}"
    print(f"\n--- 開始執行 {exp_name} "
          f"(Train: {len(y_train_bin)}, Test: {len(y_test_bin)}) ---")
    print("Building graph features for train / test sets...")

    try:
        V_train = build_graph_features(X_train)
        V_test  = build_graph_features(X_test)
        print(f"Graph feature matrix — train: {V_train.shape}, test: {V_test.shape}")
    except Exception as e:
        print(f"Graph feature construction failed: {e}")
        return

    # --- Concatenate original + graph features ---
    orig_cols  = [str(c) for c in X_train.columns] if hasattr(X_train, 'columns') else [f'orig_{i}' for i in range(np.array(X_train).shape[1])]
    graph_cols = [f'g_{n}' for n in _GRAPH_FEAT_NAMES[:V_train.shape[1]]]

    X_tr = pd.concat([
        pd.DataFrame(np.array(X_train, dtype=float), columns=orig_cols).reset_index(drop=True),
        pd.DataFrame(V_train, columns=graph_cols).reset_index(drop=True),
    ], axis=1)
    X_te = pd.concat([
        pd.DataFrame(np.array(X_test, dtype=float), columns=orig_cols).reset_index(drop=True),
        pd.DataFrame(V_test, columns=graph_cols).reset_index(drop=True),
    ], axis=1)
    print(f"Concatenated feature matrix: {X_tr.shape}  "
          f"({len(orig_cols)} original + {len(graph_cols)} graph)")

    if fs_methods is None:
        fs_methods = [
            ("ANOVA",  selectkbest_fs),
            ("L1-Reg", logistic_l1_fs),
            ("XGB-FS", xgboost_fs),
        ]

    y_tr = y_train_bin.values if hasattr(y_train_bin, 'values') else np.array(y_train_bin)
    y_te = y_test_bin.values  if hasattr(y_test_bin,  'values') else np.array(y_test_bin)
    spw  = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    aggregated_results = {}
    feature_counters   = {fs: Counter() for fs, _ in fs_methods}
    imp_records        = {}
    scaler             = StandardScaler()

    for fs_name, fs_func in fs_methods:
        try:
            top_cols = fs_func(X_tr, y_tr, k=args.k_features)
        except Exception:
            top_cols = []

        if not top_cols:
            for clf_name in ("XGB", "LR", "RF"):
                key = f"{fs_name} + {clf_name}"
                aggregated_results[key] = {'y_true': y_te.tolist(), 'y_prob': [0.5] * len(y_te)}
            continue

        feature_counters[fs_name].update(top_cols)
        X_tr_sc = scaler.fit_transform(X_tr[top_cols].values)
        X_te_sc = scaler.transform(X_te[top_cols].values)

        for clf_name, clf in _make_graph_classifiers(spw).items():
            key = f"{fs_name} + {clf_name}"
            try:
                clf.fit(X_tr_sc, y_tr)
                probs = clf.predict_proba(X_te_sc)[:, 1]
                aggregated_results[key] = {'y_true': y_te.tolist(), 'y_prob': probs.tolist()}
                if hasattr(clf, 'feature_importances_'):
                    imp_records[key] = (top_cols, clf.feature_importances_)
                elif hasattr(clf, 'coef_'):
                    imp_records[key] = (top_cols, np.abs(clf.coef_[0]))
            except Exception:
                aggregated_results[key] = {'y_true': y_te.tolist(), 'y_prob': [0.5] * len(y_te)}

    # ---- Source-cohort LOOCV for cross-dataset operating point ----
    # Run an additional LOOCV pass on the source (training) cohort using the
    # same concatenated graph features, FS methods, and 3 classifiers, then
    # take per (fs, clf) Youden thresholds. This avoids fitting the threshold
    # on the held-out target cohort.
    src_thresholds = None
    if args.use_youden:
        src_thresholds = {}
        n_src = len(y_tr)
        loo_src = LeaveOneOut()
        # per-(fs, clf) OOF arrays
        oof = {f"{fs} + {c}": np.full(n_src, 0.5)
                for fs, _ in fs_methods for c in ("XGB", "LR", "RF")}
        sc_src = StandardScaler()
        for tr_idx, te_idx in tqdm(loo_src.split(X_tr), total=n_src,
                                    desc="Source LOOCV (graph, for threshold)"):
            X_fold_tr = X_tr.iloc[tr_idx]
            X_fold_te = X_tr.iloc[te_idx]
            y_fold_tr = y_tr[tr_idx]
            spw_fold = (y_fold_tr == 0).sum() / max((y_fold_tr == 1).sum(), 1)
            for fs_name, fs_func in fs_methods:
                try:
                    cols_f = fs_func(X_fold_tr, y_fold_tr, k=args.k_features)
                except Exception:
                    cols_f = []
                if not cols_f:
                    continue
                Xtr_s = sc_src.fit_transform(X_fold_tr[cols_f].values)
                Xte_s = sc_src.transform(X_fold_te[cols_f].values)
                for clf_name, clf in _make_graph_classifiers(spw_fold).items():
                    key = f"{fs_name} + {clf_name}"
                    try:
                        clf.fit(Xtr_s, y_fold_tr)
                        oof[key][te_idx[0]] = clf.predict_proba(Xte_s)[:, 1][0]
                    except Exception:
                        oof[key][te_idx[0]] = 0.5
        for key, probs in oof.items():
            src_thresholds[key] = _youden_from_oof(y_tr, probs)
        print("\n[Source-LOOCV Youden thresholds | Graph Cross-Dataset]")
        for k, v in src_thresholds.items():
            print(f"  {k:<30s}  thr={v:.4f}")

    plot_and_report_results(aggregated_results, exp_name, dataset_name, args,
                             thresholds_override=src_thresholds)

    # Feature frequency
    print_top_features({f"{fs} (all clfs)": cnt for fs, cnt in feature_counters.items()},
                       args.k_features, 1)

    # Top-feature importance
    print("\n=== GraphConcat Cross-Dataset Top Selected Features ===")
    for key, (cols, imp_arr) in imp_records.items():
        top = sorted(zip(cols, imp_arr), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n[{key}]")
        for rank, (col, imp) in enumerate(top):
            tag = " ← graph" if col.startswith("g_") else ""
            print(f"  {rank+1:2d}. {col:<35s}  {imp:.4f}{tag}")

    # Spearman on train set
    if y_train_stage is not None:
        analyze_joint_stage_spearman(X_train, y_train_stage, dataset_name + "_train", args)


# ==========================================
# 7. 主流程與參數解析
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="XGBoost Feature Selection with Modulized Standard/Stacked LOOCV")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to features CSV file")
    parser.add_argument('--k_features', type=int, default=10, help="Number of top features to select")
    parser.add_argument('--use_youden', action='store_true', help="Enable Youden's J Index for threshold optimization")
    parser.add_argument('--dataset_source', type=str, choices=['horizontal', 'old', 'all'], default='all', help="Filter by dataset source")
    parser.add_argument('--mode', type=str, choices=['standard', 'separate_hands', 'graph', 'all'], default='standard', help="Which LOOCV mode to run: standard, separate_hands, graph, or all")
    parser.add_argument('--save_dir', type=str, default='xgb_exp/results', help="Directory to save plots and metrics")
    parser.add_argument('--no_show', action='store_true', help="Do not display plots (useful for headless environments)")
    parser.add_argument('--add_diff_features', action='store_true', help="Add left-right difference features (L - R)")
    parser.add_argument('--diff_abs', action='store_true', help="Use absolute difference |L - R| instead of L - R")
    parser.add_argument('--cross_dataset', action='store_true', help="Train on 2020 (old) and test on 2025 (horizontal). Ignores --dataset_source if set.")
    parser.add_argument('--use_lda', action='store_true', help="Enable LDA (Multi-class) + XGB method in Standard mode")
    
    args = parser.parse_args()
    
    try:
        df_full = pd.read_csv(args.csv_path)
        print(f"成功讀取 CSV: {args.csv_path}, 形狀: {df_full.shape}")
    except FileNotFoundError:
        print(f"找不到檔案: {args.csv_path}。")
        return

    if df_full.empty:
        return

    if args.dataset_source == 'all':
        target_datasets = df_full['dataset_source'].dropna().unique()
    else:
        target_datasets = [args.dataset_source]

    metadata_cols = ['patient_id', 'date', 'pd_stage', 'on_medication', 'dataset_source']

    # 定義使用的特徵選擇方法
    fs_methods = [
        ("ANOVA + XGB", selectkbest_fs),
        ("L1-Regularization + XGB", logistic_l1_fs),
        ("XGB Importance + XGB", xgboost_fs)
    ]

    if args.cross_dataset:
        print(f"\n{'='*70}")
        print(f"模式: Cross-Dataset (Train on 'old', Test on 'horizontal')")
        print(f"{'='*70}")
        
        mask_med = (df_full['on_medication'] == 0) | (df_full['on_medication'] == False)
        
        df_train = df_full[(df_full['dataset_source'] == 'old') & mask_med].copy()
        df_test = df_full[(df_full['dataset_source'] == 'horizontal') & mask_med].copy()
        
        if 'date' in df_train.columns:
            df_train = df_train.drop_duplicates(subset=['patient_id', 'date'], keep='first')
        if 'date' in df_test.columns:
            df_test = df_test.drop_duplicates(subset=['patient_id', 'date'], keep='first')
            
        print(f"訓練集 (old) 樣本數 (No Med): {len(df_train)}")
        print(f"測試集 (horizontal) 樣本數 (No Med): {len(df_test)}")
        
        if len(df_train) < 5 or len(df_test) < 5:
            print("訓練集或測試集樣本數不足。")
            return
            
        y_train_bin = pd.Series((df_train['pd_stage'] > 0).astype(int).values)
        y_train_mul = pd.Series(df_train['pd_stage'].fillna(0).astype(int).values)
        y_test_bin = pd.Series((df_test['pd_stage'] > 0).astype(int).values)
        
        X_train_all = df_train.drop(columns=[c for c in metadata_cols if c in df_train.columns]).select_dtypes(include=[np.number]).reset_index(drop=True)
        X_test_all = df_test.drop(columns=[c for c in metadata_cols if c in df_test.columns]).select_dtypes(include=[np.number]).reset_index(drop=True)
        
        if args.add_diff_features:
            left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
            for l_col in left_cols:
                r_col = 'R_' + l_col[2:]
                if r_col in X_train_all.columns:
                    diff_colname = 'Diff_' + l_col[2:]
                    if args.diff_abs:
                        X_train_all[diff_colname] = (X_train_all[l_col] - X_train_all[r_col]).abs()
                        X_test_all[diff_colname] = (X_test_all[l_col] - X_test_all[r_col]).abs()
                    else:
                        X_train_all[diff_colname] = X_train_all[l_col] - X_train_all[r_col]
                        X_test_all[diff_colname] = X_test_all[l_col] - X_test_all[r_col]

        source_name = "CrossDataset_Old2Horiz"
        left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
        right_cols = [c for c in X_train_all.columns if str(c).startswith('R_')]

        if args.mode in ['standard', 'all']:
            hand_label = "Both"
            if left_cols and not right_cols:
                hand_label = "Left"
            elif right_cols and not left_cols:
                hand_label = "Right"
            run_standard_cross_dataset(X_train_all, y_train_bin, y_train_mul, X_test_all, y_test_bin, args, source_name, fs_methods, hand_label=hand_label)

        if args.mode in ['separate_hands', 'all']:
            if left_cols:
                run_standard_cross_dataset(X_train_all[left_cols], y_train_bin, y_train_mul, X_test_all[left_cols], y_test_bin, args, source_name, fs_methods, hand_label="Left")
            if right_cols:
                run_standard_cross_dataset(X_train_all[right_cols], y_train_bin, y_train_mul, X_test_all[right_cols], y_test_bin, args, source_name, fs_methods, hand_label="Right")

        if args.mode in ['graph', 'all']:
            if left_cols and right_cols:
                use_cols   = left_cols + right_cols
                hand_label = "Both"
            elif left_cols:
                use_cols, hand_label = left_cols, "Left"
            elif right_cols:
                use_cols, hand_label = right_cols, "Right"
            else:
                use_cols, hand_label = list(X_train_all.columns), "Both"
            run_graph_cross_dataset(
                X_train_all[use_cols], X_test_all[use_cols],
                y_train_bin, y_test_bin,
                args, source_name, hand_label=hand_label,
                y_train_stage=y_train_mul,
            )

        return

    for source_name in target_datasets:
        mask_source = df_full['dataset_source'].astype(str).str.contains(source_name, case=False, na=False)
        if not mask_source.any(): continue

        print(f"\n{'='*70}")
        print(f"正在處理資料集: {source_name}")
        print(f"{'='*70}")

        mask_med = (df_full['on_medication'] == 0) | (df_full['on_medication'] == False)
        df_subset = df_full[mask_source & mask_med].copy()

        # 移除重複病人 (同 patient_id + 同 date)
        if 'date' in df_subset.columns:
            before = len(df_subset)
            df_subset = df_subset.drop_duplicates(subset=['patient_id', 'date'], keep='first')
            n_dropped = before - len(df_subset)
            if n_dropped > 0:
                print(f"移除 {n_dropped} 筆重複資料 (同 patient_id + 同 date)")

        print(f"符合條件的樣本數 (No Med): {len(df_subset)}")
        if len(df_subset) < 5:
            print("樣本數不足 (< 5)，跳過。")
            continue

        y_binary = pd.Series((df_subset['pd_stage'] > 0).astype(int).values)
        y_multi = pd.Series(df_subset['pd_stage'].fillna(0).astype(int).values)

        X_all = df_subset.drop(columns=[c for c in metadata_cols if c in df_subset.columns])
        X_all = X_all.select_dtypes(include=[np.number]).reset_index(drop=True)

        if args.add_diff_features:
            left_cols = [c for c in X_all.columns if str(c).startswith('L_')]
            diff_count = 0
            for l_col in left_cols:
                r_col = 'R_' + l_col[2:]
                if r_col in X_all.columns:
                    diff_colname = 'Diff_' + l_col[2:]
                    if args.diff_abs:
                        X_all[diff_colname] = (X_all[l_col] - X_all[r_col]).abs()
                    else:
                        X_all[diff_colname] = X_all[l_col] - X_all[r_col]
                    diff_count += 1
            print(f"已新增 {diff_count} 個左右手差距特徵 (Diff_)。總特徵數: {X_all.shape[1]}")

        if args.mode in ['standard', 'all']:
            left_cols = [c for c in X_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_all.columns if str(c).startswith('R_')]
            hand_label = "Both"
            if len(left_cols) > 0 and len(right_cols) == 0:
                hand_label = "Left"
            elif len(right_cols) > 0 and len(left_cols) == 0:
                hand_label = "Right"

            run_standard_loocv(X_all, y_binary, y_multi, args, source_name, fs_methods, hand_label=hand_label)

        if args.mode in ['separate_hands', 'all']:
            left_cols = [c for c in X_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_all.columns if str(c).startswith('R_')]
            if len(left_cols) > 0:
                run_standard_loocv(X_all[left_cols], y_binary, y_multi, args, source_name, fs_methods, hand_label="Left")
            if len(right_cols) > 0:
                run_standard_loocv(X_all[right_cols], y_binary, y_multi, args, source_name, fs_methods, hand_label="Right")

        if args.mode in ['graph', 'all']:
            left_cols  = [c for c in X_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_all.columns if str(c).startswith('R_')]
            if left_cols and right_cols:
                run_graph_loocv(X_all[left_cols + right_cols], y_binary, args, source_name,
                                hand_label="Both", y_stage=y_multi)
            elif left_cols:
                run_graph_loocv(X_all[left_cols], y_binary, args, source_name,
                                hand_label="Left", y_stage=y_multi)
            elif right_cols:
                run_graph_loocv(X_all[right_cols], y_binary, args, source_name,
                                hand_label="Right", y_stage=y_multi)
            else:
                run_graph_loocv(X_all, y_binary, args, source_name, y_stage=y_multi)

if __name__ == "__main__":
    main()
