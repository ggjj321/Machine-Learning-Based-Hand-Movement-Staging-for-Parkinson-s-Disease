"""Add a Dominant column to the feature table for patients who have 2025 data.

Dominant values come from 收案_CAREs 20251009-加密 - deID.csv, matched to feature
rows by (patient_id == 收案號) and (date == 日期). Rows with pd_stage == 0 are
written as Dominant = 0; rows with pd_stage in 1-4 take the Dominant value
looked up from the CARE file (which is always present for those stages).
"""

import pandas as pd

FEATURE_PATH = "extact_feature_latest.csv"
CARE_PATH = "收案_CAREs 20251009-加密 - deID.csv"
OUTPUT_PATH = "extact_feature_latest_with_dominant.csv"

feat = pd.read_csv(FEATURE_PATH)
care = pd.read_csv(CARE_PATH)

care_date = pd.to_datetime(care["日期"], errors="coerce")
care["日期_norm"] = pd.to_numeric(care_date.dt.strftime("%Y%m%d"), errors="coerce")
care_lookup = care[["收案號", "日期_norm", "Dominant"]].dropna(subset=["收案號", "日期_norm"])
care_lookup = care_lookup.set_index(["收案號", "日期_norm"])["Dominant"]

patients_2025 = feat.loc[feat["date"] // 10000 == 2025, "patient_id"].unique()
mask_2025_patients = feat["patient_id"].isin(patients_2025)

dominant = pd.Series(pd.NA, index=feat.index, dtype="object")
dominant[mask_2025_patients & (feat["pd_stage"] == 0)] = 0

need_lookup = mask_2025_patients & (feat["pd_stage"] != 0)
keys = pd.MultiIndex.from_arrays([feat.loc[need_lookup, "patient_id"], feat.loc[need_lookup, "date"]])
dominant[need_lookup] = care_lookup.reindex(keys).values

feat["Dominant"] = dominant
feat.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote {OUTPUT_PATH}")
print(f"Patients with 2025 data: {len(patients_2025)}")
print(f"Rows updated: {mask_2025_patients.sum()}")
print(f"Missing Dominant among updated rows: {dominant[mask_2025_patients].isna().sum()}")
