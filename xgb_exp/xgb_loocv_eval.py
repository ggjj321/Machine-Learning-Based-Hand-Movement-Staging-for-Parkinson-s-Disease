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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
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
                          eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
    model.fit(X_num, y_train)
    imp_dict = model.get_booster().get_score(importance_type="gain")
    scores = pd.Series(imp_dict).reindex(X_train.columns).fillna(0).sort_values(ascending=False)
    return scores.head(k).index.tolist()

# ==========================================
# 2. 定義計算 Youden's J 與評估指標的輔助函式
# ==========================================
def evaluate_predictions(y_true, y_prob, use_youden=False):
    """計算最佳閾值(Youden Index)及相關指標"""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 0.0, 0.0, 0.0, confusion_matrix(y_true, (y_prob >= 0.5).astype(int), labels=[0, 1]), [0,1], [0,1]

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if use_youden:
        # Youden's J Index = TPR - FPR
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_threshold = thresholds[best_idx]
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
                        eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
    
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
                            eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
    clf_lda.fit(X_train_lda, y_train_bin)

    return clf_lda.predict_proba(X_test_lda)[:, 1][0]

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
            except Exception as e:
                aggregated_results[name]['y_true'].append(y_test_val)
                aggregated_results[name]['y_prob'].append(0.5)

        # 2. 執行 LDA 方法
        if getattr(args, 'use_lda', False):
            lda_name = "LDA (Multi) + XGB"
            try:
                prob = train_and_predict_lda_xgb(X_train, y_train_bin, y_train_mul, X_test)
                aggregated_results[lda_name]['y_true'].append(y_test_val)
                aggregated_results[lda_name]['y_prob'].append(prob)
            except Exception as e:
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
# 5. 評估流程：Stacking LOOCV (左右手獨立分離)
# ==========================================
def run_stacked_loocv(X_left, X_right, y_binary, args, dataset_name, fs_methods):
    n_samples = len(y_binary)
    print(f"\n--- 開始執行 Stacked LOOCV (左右手獨立，線性權重組合，總樣本數: {n_samples}) ---")

    loo = LeaveOneOut()
    all_results = []

    # 注意：LDA 難以單純直接用於左右手分離 Stack (需要修改多分類標籤策略) 這裡暫時只用原本的 FS Methods
    for fs_name, fs_func in fs_methods:
        print(f"\n[{fs_name}] 正在處理中...")

        prob_left_oof = np.zeros(n_samples)
        prob_right_oof = np.zeros(n_samples)

        left_feature_counter = Counter()
        right_feature_counter = Counter()

        # 第一階段：左右手分離計算
        for train_index, test_index in tqdm(loo.split(X_left), total=n_samples, desc=f"Stage 1 (Left/Right XGB)"):
            X_l_train, X_l_test = X_left.iloc[train_index], X_left.iloc[test_index]
            X_r_train, X_r_test = X_right.iloc[train_index], X_right.iloc[test_index]
            y_train = y_binary.iloc[train_index].values

            # 左手處理
            prob_left_oof[test_index] = train_and_predict_xgb(X_l_train, y_train, X_l_test, fs_func, args.k_features, left_feature_counter)
            
            # 右手處理
            prob_right_oof[test_index] = train_and_predict_xgb(X_r_train, y_train, X_r_test, fs_func, args.k_features, right_feature_counter)

        # 第二階段：Linear Regression (Stacking)
        X_meta = np.vstack((prob_left_oof, prob_right_oof)).T
        prob_final_oof = np.zeros(n_samples)

        for train_index, test_index in loo.split(X_meta):
            X_meta_train, X_meta_test = X_meta[train_index], X_meta[test_index]
            y_train = y_binary.iloc[train_index].values

            lr_model = LinearRegression()
            lr_model.fit(X_meta_train, y_train)

            # 預測並限制範圍在 [0, 1] 之間
            pred_score = lr_model.predict(X_meta_test)[0]
            prob_final_oof[test_index] = np.clip(pred_score, 0, 1)

        # 整理給繪圖函式的資料結構
        models_to_eval = {
            f'{fs_name} - Left Hand': {'y_true': y_binary.values, 'y_prob': prob_left_oof},
            f'{fs_name} - Right Hand': {'y_true': y_binary.values, 'y_prob': prob_right_oof},
            f'{fs_name} - Final Stack': {'y_true': y_binary.values, 'y_prob': prob_final_oof}
        }
        
        # 繪圖及報表產出
        plot_and_report_results(models_to_eval, f"Stacked_{fs_name}", dataset_name, args, grid_layout=True)

        counters_to_print = {f"{fs_name} Left": left_feature_counter, f"{fs_name} Right": right_feature_counter}
        X_dict = {f"{fs_name} Left": X_left, f"{fs_name} Right": X_right}
        plot_scatter_top2(counters_to_print, X_dict, y_binary, f"Stacked_{fs_name}", dataset_name, args)

        print_top_features(counters_to_print, args.k_features, len(y_binary))


# ==========================================
# 5b. 評估流程：Asymmetric Stacking LOOCV
#     （左手為主 + 右手輔助 + 差異特徵捕捉不對稱性）
# ==========================================
def run_asymmetric_stacked_loocv(X_left, X_right, y_binary, args, dataset_name, fs_methods):
    n_samples = len(y_binary)
    print(f"\n--- 開始執行 Asymmetric Stacked LOOCV (左手主導 + 右手輔助 + 差異特徵，總樣本數: {n_samples}) ---")

    # 建立差異特徵：根據欄位名稱配對 L_ 與 R_
    left_col_names = list(X_left.columns)
    right_col_names = list(X_right.columns)
    # 建立名稱映射：去掉 L_/R_ prefix 後對齊
    left_suffix = {c[2:]: c for c in left_col_names if c.startswith('L_')}
    right_suffix = {c[2:]: c for c in right_col_names if c.startswith('R_')}
    common_suffixes = sorted(set(left_suffix.keys()) & set(right_suffix.keys()))

    diff_data = pd.DataFrame(index=X_left.index)
    for suffix in common_suffixes:
        diff_data[f'Diff_{suffix}'] = X_left[left_suffix[suffix]].values - X_right[right_suffix[suffix]].values
    print(f"差異特徵數量: {diff_data.shape[1]}")

    loo = LeaveOneOut()

    for fs_name, fs_func in fs_methods:
        print(f"\n[Asymmetric - {fs_name}] 正在處理中...")

        prob_left_oof = np.zeros(n_samples)
        prob_right_oof = np.zeros(n_samples)
        prob_diff_oof = np.zeros(n_samples)

        left_feature_counter = Counter()
        right_feature_counter = Counter()
        diff_feature_counter = Counter()

        # ----- 第一階段：三個分支分別計算 OOF 預測 -----
        for train_index, test_index in tqdm(loo.split(X_left), total=n_samples, desc="Stage 1 (3-branch)"):
            X_l_train, X_l_test = X_left.iloc[train_index], X_left.iloc[test_index]
            X_r_train, X_r_test = X_right.iloc[train_index], X_right.iloc[test_index]
            X_d_train, X_d_test = diff_data.iloc[train_index], diff_data.iloc[test_index]
            y_train = y_binary.iloc[train_index].values

            # 左手（主要分支）
            prob_left_oof[test_index] = train_and_predict_xgb(
                X_l_train, y_train, X_l_test, fs_func, args.k_features, left_feature_counter)

            # 右手（輔助分支）
            prob_right_oof[test_index] = train_and_predict_xgb(
                X_r_train, y_train, X_r_test, fs_func, args.k_features, right_feature_counter)

            # 差異特徵（不對稱性分支）
            prob_diff_oof[test_index] = train_and_predict_xgb(
                X_d_train, y_train, X_d_test, fs_func, args.k_features, diff_feature_counter)

        # ----- 第二階段：LogisticRegression Stacking -----
        X_meta = np.vstack((prob_left_oof, prob_right_oof, prob_diff_oof)).T
        prob_final_oof = np.zeros(n_samples)

        # 同時記錄每一 fold 的 stacking 權重
        stacking_coefs = []

        for train_index, test_index in loo.split(X_meta):
            X_meta_train, X_meta_test = X_meta[train_index], X_meta[test_index]
            y_train = y_binary.iloc[train_index].values

            meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            meta_model.fit(X_meta_train, y_train)

            prob_final_oof[test_index] = meta_model.predict_proba(X_meta_test)[:, 1][0]
            stacking_coefs.append(meta_model.coef_[0].copy())

        # 印出平均 stacking 權重
        avg_coefs = np.mean(stacking_coefs, axis=0)
        print(f"\n[Asymmetric Stacking 平均權重]")
        print(f"  Left Hand  : {avg_coefs[0]:.4f}")
        print(f"  Right Hand : {avg_coefs[1]:.4f}")
        print(f"  Diff (L-R) : {avg_coefs[2]:.4f}")

        # 整理給繪圖函式的資料結構
        models_to_eval = {
            f'{fs_name} - Left Hand':  {'y_true': y_binary.values, 'y_prob': prob_left_oof},
            f'{fs_name} - Right Hand': {'y_true': y_binary.values, 'y_prob': prob_right_oof},
            f'{fs_name} - Diff (L-R)': {'y_true': y_binary.values, 'y_prob': prob_diff_oof},
            f'{fs_name} - Asymmetric Stack': {'y_true': y_binary.values, 'y_prob': prob_final_oof},
        }

        plot_and_report_results(models_to_eval, f"Asymmetric_{fs_name}", dataset_name, args, grid_layout=True)

        counters_to_print = {
            f"{fs_name} Left": left_feature_counter,
            f"{fs_name} Right": right_feature_counter,
            f"{fs_name} Diff": diff_feature_counter,
        }
        X_dict = {
            f"{fs_name} Left": X_left,
            f"{fs_name} Right": X_right,
            f"{fs_name} Diff": diff_data,
        }
        plot_scatter_top2(counters_to_print, X_dict, y_binary, f"Asymmetric_{fs_name}", dataset_name, args)

        print_top_features(counters_to_print, args.k_features, len(y_binary))

# ==========================================
# 5c. 評估流程：Cross-Dataset Evaluation (Old -> Horizontal)
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
                                    eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
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
            from sklearn.preprocessing import StandardScaler
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
                                    eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
            clf_lda.fit(X_train_lda, y_train_bin)
            prob = clf_lda.predict_proba(X_test_lda)[:, 1]
            aggregated_results[lda_name]['y_prob'] = prob.tolist()
        except Exception as e:
            aggregated_results[lda_name]['y_prob'] = np.full(len(X_test), 0.5).tolist()
        
    plot_and_report_results(aggregated_results, exp_name, dataset_name, args)
    
    if hand_label == "Both":
        X_dict = {name: X_train for name in feature_counters.keys()}
        plot_scatter_top2(feature_counters, X_dict, y_train_bin, exp_name, dataset_name, args)
        
    print_top_features(feature_counters, args.k_features, 1)

def run_stacked_cross_dataset(X_l_train, X_r_train, y_train_bin, X_l_test, X_r_test, y_test_bin, args, dataset_name, fs_methods):
    print(f"\n--- 開始執行 Stacked Cross-Dataset ---")
    
    for fs_name, fs_func in fs_methods:
        left_feature_counter = Counter()
        right_feature_counter = Counter()
        
        top_cols_l = fs_func(X_l_train, y_train_bin, k=args.k_features)
        if top_cols_l: left_feature_counter.update(top_cols_l)
        scale_pos_weight = np.sum(y_train_bin == 0) / np.sum(y_train_bin == 1) if np.sum(y_train_bin == 1) > 0 else 1.0
        
        if top_cols_l:
            clf_l = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
            clf_l.fit(X_l_train[top_cols_l], y_train_bin)
            prob_left_train = clf_l.predict_proba(X_l_train[top_cols_l])[:, 1]
            prob_left_test = clf_l.predict_proba(X_l_test[top_cols_l])[:, 1]
        else:
            prob_left_train, prob_left_test = np.full(len(X_l_train), 0.5), np.full(len(X_l_test), 0.5)

        top_cols_r = fs_func(X_r_train, y_train_bin, k=args.k_features)
        if top_cols_r: right_feature_counter.update(top_cols_r)
        
        if top_cols_r:
            clf_r = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
            clf_r.fit(X_r_train[top_cols_r], y_train_bin)
            prob_right_train = clf_r.predict_proba(X_r_train[top_cols_r])[:, 1]
            prob_right_test = clf_r.predict_proba(X_r_test[top_cols_r])[:, 1]
        else:
            prob_right_train, prob_right_test = np.full(len(X_r_train), 0.5), np.full(len(X_r_test), 0.5)

        X_meta_train = np.vstack((prob_left_train, prob_right_train)).T
        X_meta_test = np.vstack((prob_left_test, prob_right_test)).T
        
        from sklearn.linear_model import LinearRegression
        lr_model = LinearRegression()
        lr_model.fit(X_meta_train, y_train_bin)
        
        prob_final_test = np.clip(lr_model.predict(X_meta_test), 0, 1)

        models_to_eval = {
            f'{fs_name} - Left Hand': {'y_true': y_test_bin, 'y_prob': prob_left_test},
            f'{fs_name} - Right Hand': {'y_true': y_test_bin, 'y_prob': prob_right_test},
            f'{fs_name} - Final Stack': {'y_true': y_test_bin, 'y_prob': prob_final_test}
        }
        
        plot_and_report_results(models_to_eval, f"CrossDataset_Stacked_{fs_name}", dataset_name, args, grid_layout=True)
        counters_to_print = {f"{fs_name} Left": left_feature_counter, f"{fs_name} Right": right_feature_counter}
        X_dict = {f"{fs_name} Left": X_l_train, f"{fs_name} Right": X_r_train}
        plot_scatter_top2(counters_to_print, X_dict, y_train_bin, f"CrossDataset_Stacked_{fs_name}", dataset_name, args)
        print_top_features(counters_to_print, args.k_features, 1)

def run_asymmetric_cross_dataset(X_l_train, X_r_train, diff_train, y_train_bin, X_l_test, X_r_test, diff_test, y_test_bin, args, dataset_name, fs_methods):
    print(f"\n--- 開始執行 Asymmetric Stacked Cross-Dataset ---")
    
    for fs_name, fs_func in fs_methods:
        left_feature_counter = Counter()
        right_feature_counter = Counter()
        diff_feature_counter = Counter()
        
        scale_pos_weight = np.sum(y_train_bin == 0) / np.sum(y_train_bin == 1) if np.sum(y_train_bin == 1) > 0 else 1.0

        def train_and_eval(X_tr, X_te, counter):
            top_cols = fs_func(X_tr, y_train_bin, k=args.k_features)
            if top_cols: counter.update(top_cols)
            if top_cols:
                clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
                clf.fit(X_tr[top_cols], y_train_bin)
                return clf.predict_proba(X_tr[top_cols])[:, 1], clf.predict_proba(X_te[top_cols])[:, 1]
            return np.full(len(X_tr), 0.5), np.full(len(X_te), 0.5)

        prob_l_tr, prob_l_te = train_and_eval(X_l_train, X_l_test, left_feature_counter)
        prob_r_tr, prob_r_te = train_and_eval(X_r_train, X_r_test, right_feature_counter)
        prob_d_tr, prob_d_te = train_and_eval(diff_train, diff_test, diff_feature_counter)

        from sklearn.linear_model import LogisticRegression
        X_meta_train = np.vstack((prob_l_tr, prob_r_tr, prob_d_tr)).T
        X_meta_test = np.vstack((prob_l_te, prob_r_te, prob_d_te)).T
        
        meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        meta.fit(X_meta_train, y_train_bin)
        prob_final_te = meta.predict_proba(X_meta_test)[:, 1]

        models_to_eval = {
            f'{fs_name} - Left Hand': {'y_true': y_test_bin, 'y_prob': prob_l_te},
            f'{fs_name} - Right Hand': {'y_true': y_test_bin, 'y_prob': prob_r_te},
            f'{fs_name} - Diff (L-R)': {'y_true': y_test_bin, 'y_prob': prob_d_te},
            f'{fs_name} - Asymmetric Stack': {'y_true': y_test_bin, 'y_prob': prob_final_te},
        }
        
        plot_and_report_results(models_to_eval, f"CrossDataset_Asym_{fs_name}", dataset_name, args, grid_layout=True)
        counters_to_print = {f"{fs_name} Left": left_feature_counter, f"{fs_name} Right": right_feature_counter, f"{fs_name} Diff": diff_feature_counter}
        X_dict = {f"{fs_name} Left": X_l_train, f"{fs_name} Right": X_r_train, f"{fs_name} Diff": diff_train}
        plot_scatter_top2(counters_to_print, X_dict, y_train_bin, f"CrossDataset_Asym_{fs_name}", dataset_name, args)
        print_top_features(counters_to_print, args.k_features, 1)

# ==========================================
# 6. 視覺化及輸出輔助函式
# ==========================================
def plot_and_report_results(results_dict, exp_name, dataset_name, args, grid_layout=False):
    """結果報表、ROC 曲線與混淆矩陣視覺化統一介面"""
    table_results = []
    
    os.makedirs(args.save_dir, exist_ok=True)
    n_models = len(results_dict)
    
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 8))
    
    if grid_layout:
        # 為了 Stacked 模式呈現一行 (預設為 Left, Right, Stacked 共 3 個)
        fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    else:
        # Standard 的話依模型數量排佈
        fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1: axes_cm = [axes_cm]

    for i, (name, data) in enumerate(results_dict.items()):
        y_true_all = np.array(data['y_true'])
        y_prob_all = np.array(data['y_prob'])

        thresh, roc_auc, acc, prec, rec, f1, cm, fpr, tpr = evaluate_predictions(y_true_all, y_prob_all, use_youden=args.use_youden)

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

    # ROC Figure 設定
    ax_roc.plot([0, 1], [0, 1], 'k--')
    title_suffix = "(Youden's J)" if args.use_youden else "(Thresh=0.5)"
    ax_roc.set_title(f'{dataset_name} - {exp_name} ROC Curves {title_suffix}', fontsize=14, fontweight='bold')
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
    print(f"\n=== Performance Report: {dataset_name} | {exp_name} ===")
    print(df_res[cols_order].round(4).to_string(index=False))
    
    df_res[cols_order].to_csv(os.path.join(args.save_dir, f'metrics_{fig_suffix}.csv'), index=False)

def plot_scatter_top2(feature_counters_dict, X_dict, y_binary, exp_name, dataset_name, args):
    """
    根據紀錄的 feature counters 繪製出每個方法 Top 2 特徵的 2D 散佈圖。
    """
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
                
                ax.set_title(f"{name} Top 2 Features", fontsize=14, fontweight='bold')
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
# 7. 主流程與參數解析
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="XGBoost Feature Selection with Modulized Standard/Stacked LOOCV")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to features CSV file")
    parser.add_argument('--k_features', type=int, default=10, help="Number of top features to select")
    parser.add_argument('--use_youden', action='store_true', help="Enable Youden's J Index for threshold optimization")
    parser.add_argument('--dataset_source', type=str, choices=['horizontal', 'old', 'all'], default='all', help="Filter by dataset source")
    parser.add_argument('--mode', type=str, choices=['standard', 'stacked', 'asymmetric', 'both', 'separate_hands', 'all'], default='both', help="Which LOOCV mode to run: standard, stacked, asymmetric, both, separate_hands, or all")
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
        ("SelectKBest + XGB", selectkbest_fs),
        ("Logistic L1 + XGB", logistic_l1_fs),
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

        if args.mode in ['standard', 'both']:
            left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_train_all.columns if str(c).startswith('R_')]
            hand_label = "Both"
            if len(left_cols) > 0 and len(right_cols) == 0:
                hand_label = "Left"
            elif len(right_cols) > 0 and len(left_cols) == 0:
                hand_label = "Right"

            run_standard_cross_dataset(X_train_all, y_train_bin, y_train_mul, X_test_all, y_test_bin, args, source_name, fs_methods, hand_label=hand_label)

        if args.mode in ['separate_hands', 'all']:
            left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_train_all.columns if str(c).startswith('R_')]
            if len(left_cols) > 0:
                run_standard_cross_dataset(X_train_all[left_cols], y_train_bin, y_train_mul, X_test_all[left_cols], y_test_bin, args, source_name, fs_methods, hand_label="Left")
            if len(right_cols) > 0:
                run_standard_cross_dataset(X_train_all[right_cols], y_train_bin, y_train_mul, X_test_all[right_cols], y_test_bin, args, source_name, fs_methods, hand_label="Right")

        if args.mode in ['stacked', 'both']:
            left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_train_all.columns if str(c).startswith('R_')]
            if len(left_cols) > 0 and len(right_cols) > 0:
                run_stacked_cross_dataset(X_train_all[left_cols], X_train_all[right_cols], y_train_bin, 
                                          X_test_all[left_cols], X_test_all[right_cols], y_test_bin, args, source_name, fs_methods)

        if args.mode in ['asymmetric', 'all']:
            left_cols = [c for c in X_train_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_train_all.columns if str(c).startswith('R_')]
            
            left_suffix = {c[2:]: c for c in left_cols}
            right_suffix = {c[2:]: c for c in right_cols}
            common_suffixes = sorted(set(left_suffix.keys()) & set(right_suffix.keys()))

            diff_train = pd.DataFrame(index=X_train_all.index)
            diff_test = pd.DataFrame(index=X_test_all.index)
            for suffix in common_suffixes:
                diff_train[f'Diff_{suffix}'] = X_train_all[left_suffix[suffix]].values - X_train_all[right_suffix[suffix]].values
                diff_test[f'Diff_{suffix}'] = X_test_all[left_suffix[suffix]].values - X_test_all[right_suffix[suffix]].values

            if len(left_cols) > 0 and len(right_cols) > 0:
                run_asymmetric_cross_dataset(X_train_all[left_cols], X_train_all[right_cols], diff_train, y_train_bin, 
                                             X_test_all[left_cols], X_test_all[right_cols], diff_test, y_test_bin, args, source_name, fs_methods)
        
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

        if args.mode in ['standard', 'both']:
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

        if args.mode in ['stacked', 'both']:
            left_cols = [c for c in X_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_all.columns if str(c).startswith('R_')]
            
            if len(left_cols) == 0 or len(right_cols) == 0:
                print(f"警告：資料集 {source_name} 找不到具有 'L_' 或 'R_' 標註的左右手特徵。已跳過 Stacked 模式。")
                print(f"您的欄位範例：{list(X_all.columns[:5])}...")
            else:
                X_left = X_all[left_cols]
                X_right = X_all[right_cols]
                run_stacked_loocv(X_left, X_right, y_binary, args, source_name, fs_methods)

        if args.mode in ['asymmetric', 'all']:
            left_cols = [c for c in X_all.columns if str(c).startswith('L_')]
            right_cols = [c for c in X_all.columns if str(c).startswith('R_')]

            if len(left_cols) == 0 or len(right_cols) == 0:
                print(f"警告：資料集 {source_name} 找不到具有 'L_' 或 'R_' 標註的左右手特徵。已跳過 Asymmetric 模式。")
            else:
                X_left = X_all[left_cols]
                X_right = X_all[right_cols]
                run_asymmetric_stacked_loocv(X_left, X_right, y_binary, args, source_name, fs_methods)

if __name__ == "__main__":
    main()
