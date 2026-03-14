import pandas as pd
import sys

df1_path = 'pd_features_with_medication(1).csv'
df2_path = 'extracted_features.csv'

try:
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
except Exception as e:
    print(f"Error reading CSVs: {e}")
    sys.exit(1)

def get_patient_row(df, pid_str):
    mask = df['patient_id'].astype(str).str.contains(str(pid_str))
    return df[mask]

p1 = get_patient_row(df1, '105002')
p2 = get_patient_row(df2, '105002')

if p1.empty:
    print(f"Patient 105002 not found in {df1_path}")
    sys.exit(0)
if p2.empty:
    print(f"Patient 105002 not found in {df2_path}")
    sys.exit(0)

s1 = p1.iloc[0]
s2 = p2.iloc[0]

print(f"Patient 1 ID in {df1_path}: {s1['patient_id']}")
print(f"Patient 2 ID in {df2_path}: {s2['patient_id']}")

common_cols = set(df1.columns).intersection(set(df2.columns))
feature_cols = [c for c in df1.columns if c in common_cols and c not in ['patient_id', 'pd_stage', 'on_medication', 'dataset_source']]

diff_count = 0
diffs = []
for c in feature_cols:
    val1 = s1[c]
    val2 = s2[c]
    
    if pd.isna(val1) and pd.isna(val2):
        continue
    if pd.isna(val1) or pd.isna(val2):
        diff_count += 1
        diffs.append((c, val1, val2))
        continue
        
    try:
        f1 = float(val1)
        f2 = float(val2)
        # Using a very small epsilon for floating point comparison
        if abs(f1 - f2) > 1e-5:
            diff_count += 1
            diffs.append((c, f1, f2))
    except ValueError:
        if val1 != val2:
            diff_count += 1
            diffs.append((c, val1, val2))

print(f"Total features compared: {len(feature_cols)}")
print(f"Found {diff_count} feature values that are different.")
if diff_count > 0:
    print("Showing up to 20 differences:")
    for c, v1, v2 in diffs[:20]:
        if isinstance(v1, float) and isinstance(v2, float):
            print(f"{c}: old = {v1:.5f}, new = {v2:.5f}, diff = {abs(v1-v2):.5f}")
        else:
            print(f"{c}: old = {v1}, new = {v2}")
