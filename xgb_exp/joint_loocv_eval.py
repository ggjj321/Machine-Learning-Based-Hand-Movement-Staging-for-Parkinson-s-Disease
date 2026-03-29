"""
Joint-Selective Feature LOOCV Evaluation
=========================================
可選擇特定關節點 (或全部) 的特徵，
以 Logistic Regression、Random Forest、XGBoost 進行二元分類 (PD vs Healthy)，
使用 Leave-One-Out Cross-Validation (LOOCV)。

用法範例：
  --joints 1,2,3,4    # 只用大拇指 (joint 1-4)
  --joints 4           # 只用大拇指指尖 (joint 4)
  --joints all         # 使用全部 21 個關節 (0-20)

輸出：ROC Curve、Confusion Matrix、Accuracy、Precision、Recall、F1-score、AUROC。
"""

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# 1. 篩選指定關節特徵
# ==========================================
def filter_joint_columns(columns, joint_indices):
    """
    從所有欄位中篩選出指定關節的特徵。
    若 joint_indices 為 None 則回傳全部數值特徵欄位。

    特徵命名格式：
      - L_joint01_x_frequency_without_trend  (joint{XX} 格式)
      - L_j1_x_cycle_amp_mean               (j{X} 格式)
    """
    if joint_indices is None:
        return list(columns)

    # 建立 regex：例如 joint_indices=[1,4] → _(j1_|j4_|joint01_|joint04_)
    short_alts = [f'j{j}_' for j in joint_indices]      # j1_, j4_
    long_alts = [f'joint{j:02d}_' for j in joint_indices]  # joint01_, joint04_
    all_alts = '|'.join(short_alts + long_alts)
    pattern = re.compile(rf'_({all_alts})')

    selected = [col for col in columns if pattern.search(col)]
    return selected


# ==========================================
# 2. 評估指標計算
# ==========================================
def evaluate_predictions(y_true, y_prob, use_youden=False):
    """計算最佳閾值 (Youden Index) 及相關指標"""
    if len(np.unique(y_true)) < 2:
        return (0.5, 0.0,
                accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
                0.0, 0.0, 0.0,
                confusion_matrix(y_true, (y_prob >= 0.5).astype(int), labels=[0, 1]),
                [0, 1], [0, 1])

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if use_youden:
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
# 3. 定義分類器
# ==========================================
def get_classifiers(scale_pos_weight=1.0):
    """回傳 (名稱, 分類器) 的清單"""
    return [
        ("Logistic Regression", LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, max_depth=5,
            class_weight='balanced', random_state=42, n_jobs=1)),
        ("XGBoost", XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42, n_jobs=1)),
    ]


# ==========================================
# 4. LOOCV 主流程
# ==========================================
def run_joint_loocv(X, y_binary, args, dataset_name, joint_label):
    """使用指定關節特徵進行 LOOCV"""
    n_samples = len(y_binary)
    print(f"\n--- 開始執行 {joint_label} LOOCV (樣本數: {n_samples}, 特徵數: {X.shape[1]}) ---")

    scale_pos_weight = np.sum(y_binary == 0) / np.sum(y_binary == 1) if np.sum(y_binary == 1) > 0 else 1.0
    classifiers = get_classifiers(scale_pos_weight)

    loo = LeaveOneOut()
    aggregated = {name: {'y_true': [], 'y_prob': [], 'importances': []} for name, _ in classifiers}

    for train_index, test_index in tqdm(loo.split(X), total=n_samples, desc=f"{joint_label} LOOCV"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test_val = y_binary.iloc[train_index].values, y_binary.iloc[test_index].values[0]

        # 中位數填補 + 標準化
        train_medians = X_train.median(numeric_only=True)
        X_train_filled = X_train.fillna(train_medians)
        X_test_filled = X_test.fillna(train_medians)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_test_scaled = scaler.transform(X_test_filled)

        for name, clf_template in classifiers:
            try:
                # clone classifier for each fold
                from sklearn.base import clone
                clf = clone(clf_template)
                clf.fit(X_train_scaled, y_train)
                prob = clf.predict_proba(X_test_scaled)[:, 1][0]
                
                if hasattr(clf, 'feature_importances_'):
                    imp = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    imp = np.abs(clf.coef_[0])
                else:
                    imp = np.zeros(X_train_scaled.shape[1])
                aggregated[name]['importances'].append(imp)
            except Exception:
                prob = 0.5
                aggregated[name]['importances'].append(np.zeros(X.shape[1]))

            aggregated[name]['y_true'].append(y_test_val)
            aggregated[name]['y_prob'].append(prob)

    # 繪圖及報表
    plot_and_report(aggregated, dataset_name, args, joint_label, X, y_binary)


# ==========================================
# 5. 視覺化及報表輸出
# ==========================================
def plot_and_report(results_dict, dataset_name, args, joint_label, X=None, y_binary=None):
    """ROC 曲線、混淆矩陣、指標表格"""
    os.makedirs(args.save_dir, exist_ok=True)

    n_models = len(results_dict)
    table_results = []

    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 8))
    fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes_cm = [axes_cm]

    for i, (name, data) in enumerate(results_dict.items()):
        y_true_all = np.array(data['y_true'])
        y_prob_all = np.array(data['y_prob'])

        thresh, roc_auc, acc, prec, rec, f1, cm, fpr, tpr = \
            evaluate_predictions(y_true_all, y_prob_all, use_youden=args.use_youden)

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

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i],
                    annot_kws={"size": 14})
        axes_cm[i].set_title(f"{name}\nThresh={thresh:.3f}", fontsize=12, fontweight='bold')
        axes_cm[i].set_xlabel('Predicted')
        axes_cm[i].set_ylabel('True')
        axes_cm[i].set_xticklabels(['Healthy', 'PD'])
        axes_cm[i].set_yticklabels(['Healthy', 'PD'])
    
    if args.cross_dataset:
        prefix = "cross dataset"
    elif dataset_name == "horizontal":
        prefix = "2025"
    elif dataset_name == "old":
        prefix = "2020"
    else:
        prefix = dataset_name

    # ROC 設定
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title(f'{prefix} {joint_label} LOOCV ROC',
                     fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect('equal')

    safe_label = joint_label.replace(' ', '_').replace(',', '-')
    fig_suffix = f"{safe_label}_{prefix.replace(' ', '_')}"
    fig_roc.savefig(os.path.join(args.save_dir, f'roc_{fig_suffix}.png'), bbox_inches='tight')
    fig_cm.savefig(os.path.join(args.save_dir, f'cm_{fig_suffix}.png'), bbox_inches='tight')

    # 畫出每個模型前兩大特徵的 2D Scatter Plot
    if X is not None and y_binary is not None:
        for name, data in results_dict.items():
            if 'importances' in data and len(data['importances']) > 0:
                avg_imp = np.mean(data['importances'], axis=0)
                if np.sum(avg_imp) > 0:
                    top2_idx = np.argsort(avg_imp)[-2:][::-1]
                    top1_feat = X.columns[top2_idx[0]]
                    top2_feat = X.columns[top2_idx[1]]
                    
                    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                    scatter_data = pd.DataFrame({
                        top1_feat: X[top1_feat].values,
                        top2_feat: X[top2_feat].values,
                        'Label': ['PD' if y == 1 else 'Healthy' for y in y_binary]
                    })
                    
                    sns.scatterplot(
                        data=scatter_data, 
                        x=top1_feat, 
                        y=top2_feat, 
                        hue='Label', 
                        palette={'Healthy': 'blue', 'PD': 'red'},
                        ax=ax_scatter
                    )
                    
                    ax_scatter.set_title(f"{name} Top 2 Features\n({prefix} {joint_label})", fontsize=14, fontweight='bold')
                    ax_scatter.grid(True, alpha=0.3)
                    
                    safe_name = name.replace(' ', '_')
                    fig_scatter.savefig(os.path.join(args.save_dir, f'scatter_{safe_name}_{fig_suffix}.png'), bbox_inches='tight')
                    plt.close(fig_scatter)

    plt.tight_layout()
    if not args.no_show:
        plt.show()

    df_res = pd.DataFrame(table_results)
    cols_order = ['Model', 'Threshold', 'AUROC', 'Acc', 'Precision', 'Recall', 'F1-score']
    print(f"\n=== Performance Report: {prefix} | {joint_label} LOOCV ===")
    print(df_res[cols_order].round(4).to_string(index=False))

    df_res[cols_order].to_csv(os.path.join(args.save_dir, f'metrics_{fig_suffix}.csv'), index=False)


# ==========================================
# 6. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Joint-Selective Feature LOOCV: LR / RF / XGBoost 二元分類")
    parser.add_argument('--csv_path', type=str, required=True,
                        help="Path to features CSV file")
    parser.add_argument('--joints', type=str, default='1,2,3,4',
                        help="Comma-separated joint indices to use, or 'all' for all 21 joints. "
                             "Examples: '4' (thumb tip), '1,2,3,4' (thumb), 'all' (all joints)")
    parser.add_argument('--use_youden', action='store_true',
                        help="Enable Youden's J Index for threshold optimization")
    parser.add_argument('--dataset_source', type=str,
                        choices=['horizontal', 'old', 'all'], default='all',
                        help="Filter by dataset source")
    parser.add_argument('--save_dir', type=str, default='xgb_exp/results',
                        help="Directory to save plots and metrics")
    parser.add_argument('--cross_dataset', action='store_true',
                        help="Train on 2020 (old) and test on 2025 (horizontal). Ignores --dataset_source if set.")
    parser.add_argument('--no_show', action='store_true',
                        help="Do not display plots (useful for headless environments)")
    args = parser.parse_args()

    # 解析關節選擇
    if args.joints.strip().lower() == 'all':
        joint_indices = None  # None = 使用全部
        joint_label = "all joint"
    else:
        joint_indices = [int(j.strip()) for j in args.joints.split(',')]
        if len(joint_indices) == 1:
            joint_label = f"joint {joint_indices[0]}"
        else:
            joint_label = f"joint {args.joints.replace(',', ' ')}"

    # 讀取 CSV
    try:
        df_full = pd.read_csv(args.csv_path)
        print(f"成功讀取 CSV: {args.csv_path}, 形狀: {df_full.shape}")
    except FileNotFoundError:
        print(f"找不到檔案: {args.csv_path}。")
        return

    if df_full.empty:
        return

    metadata_cols = ['patient_id', 'date', 'pd_stage', 'on_medication', 'dataset_source']

    if args.cross_dataset:
        print(f"\n{'='*70}")
        print(f"模式: Cross-Dataset (Train on 'old', Test on 'horizontal')")
        print(f"{'='*70}")
        
        mask_med = (df_full['on_medication'] == 0) | (df_full['on_medication'] == False)
        
        # 分割 Train (old) / Test (horizontal)
        df_train = df_full[(df_full['dataset_source'] == 'old') & mask_med].copy()
        df_test = df_full[(df_full['dataset_source'] == 'horizontal') & mask_med].copy()
        
        # 移除重複 (已經在 extract_features 處理過，但多一道保險)
        if 'date' in df_train.columns:
            df_train = df_train.drop_duplicates(subset=['patient_id', 'date'], keep='first')
        if 'date' in df_test.columns:
            df_test = df_test.drop_duplicates(subset=['patient_id', 'date'], keep='first')
            
        print(f"訓練集 (old) 樣本數 (No Med): {len(df_train)}")
        print(f"測試集 (horizontal) 樣本數 (No Med): {len(df_test)}")
        
        if len(df_train) == 0 or len(df_test) == 0:
            print("訓練集或測試集樣本數為 0，無法執行。")
            return
            
        y_train = pd.Series((df_train['pd_stage'] > 0).astype(int).values)
        y_test_binary = pd.Series((df_test['pd_stage'] > 0).astype(int).values)
        
        X_train_all = df_train.drop(columns=[c for c in metadata_cols if c in df_train.columns])
        X_train_all = X_train_all.select_dtypes(include=[np.number]).reset_index(drop=True)
        
        X_test_all = df_test.drop(columns=[c for c in metadata_cols if c in df_test.columns])
        X_test_all = X_test_all.select_dtypes(include=[np.number]).reset_index(drop=True)
        
        # 篩選指定關節特徵
        selected_cols = filter_joint_columns(X_train_all.columns, joint_indices)
        if len(selected_cols) == 0:
            print("找不到指定關節的欄位。")
            return
            
        X_train_sel = X_train_all[selected_cols]
        X_test_sel = X_test_all[selected_cols]
        print(f"篩選出 {len(selected_cols)} 個特徵 ({joint_label})")
        
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
        classifiers = get_classifiers(scale_pos_weight)
        
        # 中位數填補 + 標準化 (以 train_set fit)
        train_medians = X_train_sel.median(numeric_only=True)
        X_train_filled = X_train_sel.fillna(train_medians)
        X_test_filled = X_test_sel.fillna(train_medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_test_scaled = scaler.transform(X_test_filled)
        
        aggregated = {name: {'y_true': y_test_binary.tolist(), 'y_prob': [], 'importances': []} for name, _ in classifiers}
        
        for name, clf in classifiers:
            clf.fit(X_train_scaled, y_train.values)
            prob = clf.predict_proba(X_test_scaled)[:, 1]
            aggregated[name]['y_prob'] = prob.tolist()
            if hasattr(clf, 'feature_importances_'):
                imp = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                imp = np.abs(clf.coef_[0])
            else:
                imp = np.zeros(X_train_sel.shape[1])
            aggregated[name]['importances'] = [imp]
            
        global_year_override = "CrossDataset_Old2Horiz"
        plot_and_report(aggregated, global_year_override, args, joint_label, X_test_sel, y_test_binary)

    else:
        if args.dataset_source == 'all':
            target_datasets = df_full['dataset_source'].dropna().unique()
        else:
            target_datasets = [args.dataset_source]

        for source_name in target_datasets:
            mask_source = df_full['dataset_source'].astype(str).str.contains(
                source_name, case=False, na=False)
            if not mask_source.any():
                continue

            print(f"\n{'='*70}")
            print(f"正在處理資料集: {source_name}")
            print(f"{'='*70}")

            mask_med = (df_full['on_medication'] == 0) | (df_full['on_medication'] == False)
            df_subset = df_full[mask_source & mask_med].copy()

            print(f"符合條件的樣本數 (No Med): {len(df_subset)}")
            if len(df_subset) < 5:
                print("樣本數不足 (< 5)，跳過。")
                continue

            y_binary = pd.Series((df_subset['pd_stage'] > 0).astype(int).values)

            X_all = df_subset.drop(columns=[c for c in metadata_cols if c in df_subset.columns])
            X_all = X_all.select_dtypes(include=[np.number]).reset_index(drop=True)

            # 篩選指定關節特徵
            selected_cols = filter_joint_columns(X_all.columns, joint_indices)
            if len(selected_cols) == 0:
                print(f"警告：資料集 {source_name} 找不到指定關節的欄位，已跳過。")
                continue

            X_selected = X_all[selected_cols]
            print(f"篩選出 {len(selected_cols)} 個特徵 ({joint_label})")

            # 印出特徵欄位摘要
            left_count = sum(1 for c in selected_cols if c.startswith('L_'))
            right_count = sum(1 for c in selected_cols if c.startswith('R_'))
            print(f"  左手: {left_count} 個, 右手: {right_count} 個")

            run_joint_loocv(X_selected, y_binary, args, source_name, joint_label)


if __name__ == '__main__':
    main()
