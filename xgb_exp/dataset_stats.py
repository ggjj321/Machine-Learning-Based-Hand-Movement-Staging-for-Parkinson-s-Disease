"""
病人統計表產生工具
==================
讀取 extract_features.py 產生的 CSV，
依 dataset_source 分組列出各 PD Stage × 服藥狀態 的人數統計表。

用法：
  python xgb_exp/dataset_stats.py --csv extracted_features_merged.csv
  python xgb_exp/dataset_stats.py --csv extracted_features_merged.csv --source horizontal
"""

import argparse
import pandas as pd
import numpy as np


def print_stats_table(df, title=""):
    """印出一個 dataset_source 的統計表"""
    stages = sorted(df['pd_stage'].unique())
    stage_cols = [f"Stage {s}" for s in stages]

    # 分組
    med_on = df[df['on_medication'] == True]
    med_off = df[df['on_medication'] == False]

    row_on = [len(med_on[med_on['pd_stage'] == s]) for s in stages]
    row_off = [len(med_off[med_off['pd_stage'] == s]) for s in stages]
    row_total = [len(df[df['pd_stage'] == s]) for s in stages]

    row_on.append(sum(row_on))
    row_off.append(sum(row_off))
    row_total.append(sum(row_total))

    cols = stage_cols + ['All']

    table = pd.DataFrame(
        [row_on, row_off, row_total],
        index=['服藥中', '未服藥', '總計'],
        columns=cols
    )

    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")
    print(table.to_string())
    print()


def print_duplicate_patients(df):
    """列出重複出現的病人 (同一 patient_id + 同一 date 出現多次才算重複)"""

    has_date = 'date' in df.columns

    # --- 各 dataset_source 內部重複 ---
    sources = sorted(df['dataset_source'].dropna().unique())
    found_any = False

    for src in sources:
        subset = df[df['dataset_source'] == src]

        if has_date:
            # 以 (patient_id, date) 為 key 判斷重複
            dup_mask = subset.duplicated(subset=['patient_id', 'date'], keep=False)
            dup_rows = subset[dup_mask]
            if len(dup_rows) == 0:
                continue
            dup_groups = dup_rows.groupby(['patient_id', 'date'])
        else:
            dup_ids = subset['patient_id'].value_counts()
            dup_ids = dup_ids[dup_ids > 1]
            if len(dup_ids) == 0:
                continue
            dup_groups = None

        found_any = True
        n_dup = len(dup_groups) if dup_groups is not None else len(dup_ids)
        print(f"\n{'='*60}")
        print(f"  [{src}] 內部重複 ({n_dup} 組)")
        print(f"{'='*60}")

        if has_date and dup_groups is not None:
            for (pid, date), group in sorted(dup_groups, key=lambda x: (x[0])):
                print(f"\n  Patient {pid}, Date {date} (出現 {len(group)} 次):")
                for _, r in group.iterrows():
                    med_str = "服藥中" if r['on_medication'] else "未服藥"
                    print(f"    Stage {int(r['pd_stage'])}, {med_str}")
        else:
            for pid, count in dup_ids.sort_index().items():
                rows = subset[subset['patient_id'] == pid]
                print(f"\n  Patient {pid} (出現 {count} 次):")
                for _, r in rows.iterrows():
                    med_str = "服藥中" if r['on_medication'] else "未服藥"
                    print(f"    Stage {int(r['pd_stage'])}, {med_str}")

    # --- 跨 dataset_source 重複 ---
    if len(sources) > 1 and has_date:
        # 同 patient_id + 同 date 出現在不同 source
        cross = df.groupby(['patient_id', 'date'])['dataset_source'].nunique()
        cross_dup = cross[cross > 1]

        if len(cross_dup) > 0:
            found_any = True
            print(f"\n{'='*60}")
            print(f"  跨資料集重複 ({len(cross_dup)} 組)")
            print(f"{'='*60}")

            for (pid, date) in sorted(cross_dup.index):
                rows = df[(df['patient_id'] == pid) & (df['date'] == date)]
                print(f"\n  Patient {pid}, Date {date} (出現在 {cross_dup[(pid, date)]} 個資料集):")
                for _, r in rows.iterrows():
                    med_str = "服藥中" if r['on_medication'] else "未服藥"
                    print(f"    [{r['dataset_source']}] Stage {int(r['pd_stage'])}, {med_str}")

    if not found_any:
        print("\n無重複病人。")


def main():
    parser = argparse.ArgumentParser(description="病人統計表：依 dataset_source 和 PD Stage 分組")
    parser.add_argument('--csv', type=str, required=True, help="特徵 CSV 檔案路徑")
    parser.add_argument('--source', type=str, default=None,
                        help="只顯示特定 dataset_source (例如 horizontal, old)，不指定則全部顯示")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"讀取 CSV: {args.csv}, 總列數: {len(df)}")

    # 確保 on_medication 為 bool
    df['on_medication'] = df['on_medication'].astype(bool)

    if args.source:
        sources = [args.source]
    else:
        sources = sorted(df['dataset_source'].dropna().unique())

    # 各 dataset_source 分別列表
    for src in sources:
        subset = df[df['dataset_source'] == src]
        if len(subset) == 0:
            print(f"\n[{src}] 無資料")
            continue
        print_stats_table(subset, title=f"Dataset: {src} ({len(subset)} samples)")

    # 如果有多個 source，也印出合計
    if len(sources) > 1:
        print_stats_table(df, title=f"All Datasets Combined ({len(df)} samples)")

    # 列出重複病人
    print_duplicate_patients(df if not args.source else df[df['dataset_source'] == args.source])


if __name__ == '__main__':
    main()
