import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import correlate, find_peaks
from typing import List, Tuple, Dict, Optional

class PDPatient:
    """Stores data for a single PD patient."""
    def __init__(self, patient_id: str, pd_stage: int, date: str, on_medication: bool):
        self.patient_id = patient_id
        self.pd_stage = pd_stage
        self.date = date
        self.on_medication = on_medication
        self.left_trajectory = None   # shape: (frames, 21, 3)
        self.right_trajectory = None  # shape: (frames, 21, 3)

    @classmethod
    def from_pt(cls, patient_id: str, pd_stage: int, date: str, on_medication: bool, left_pt_path: str, right_pt_path: str):
        patient = cls(patient_id, pd_stage, date, on_medication)
        
        # 載入 .pt 檔案 / Load .pt files
        # 為了相容新舊版本，處理兩種格式：包含 meta 資訊的 dict 或純 tensor
        left_data = torch.load(left_pt_path, map_location='cpu')
        right_data = torch.load(right_pt_path, map_location='cpu')
        
        if isinstance(left_data, dict) and 'skeleton_sequence' in left_data:
            patient.left_trajectory = left_data['skeleton_sequence']
        else:
            patient.left_trajectory = left_data
            
        if isinstance(right_data, dict) and 'skeleton_sequence' in right_data:
            patient.right_trajectory = right_data['skeleton_sequence']
        else:
            patient.right_trajectory = right_data
            
        # 確保維度為 (T, 21, 3) / Ensure shape is (T, 21, 3)
        if patient.left_trajectory.shape[-1] == 2:
            zeros = torch.zeros((*patient.left_trajectory.shape[:-1], 1), dtype=patient.left_trajectory.dtype)
            patient.left_trajectory = torch.cat([patient.left_trajectory, zeros], dim=-1)
        if patient.right_trajectory.shape[-1] == 2:
            zeros = torch.zeros((*patient.right_trajectory.shape[:-1], 1), dtype=patient.right_trajectory.dtype)
            patient.right_trajectory = torch.cat([patient.right_trajectory, zeros], dim=-1)

        return patient


def find_pt_file(base_dirs: List[str], date_str: str, patient_id: str, hand_suffix: str) -> Optional[str]:
    """
    尋找對應的 .pt 檔案 (支援遞迴收尋 stage_* 目錄)
    容許 patient_id 前有補零
    支援多個基礎目錄 (例如: dir1,dir2)
    """
    possible_files = []
    
    for bdir in base_dirs:
        base_path = Path(bdir)
        if not base_path.exists():
            continue
            
        # 搜尋所有 stage 子資料夾，或是如果只有一層結構也一起收納
        for file_path in base_path.rglob("*.pt"):
            possible_files.append(file_path)
            
    # File name example: 2025-04-18 14:36:55_gesture_20250418_093409__128_左手旋轉_REC_...pt
    # OR: 20250703_111023_L.pt (older formats, fallback)
    # The regex r"0*{patient_id}" handles CSV inputs like '93409' when file is '093409'
    pattern1 = re.compile(rf".*{re.escape(date_str)}_0*{re.escape(patient_id)}.*{re.escape(hand_suffix)}")
    
    matches = [f for f in possible_files if pattern1.search(f.name)]
    
    if len(matches) >= 1:
        if len(matches) > 1:
            print(f"Warning: Multiple matches found for {date_str}_{patient_id}*{hand_suffix}: {[m.name for m in matches]}. Using the first one.")
        return str(matches[0])
    return None


def load_patients_from_csv(csv_path: str, pt_base_dirs: List[str]) -> List[PDPatient]:
    patients = []
    try:
        df = pd.read_csv(csv_path, dtype={'Date': str, 'PID': str})
    except Exception as e:
        print(f"Failed to read CSV {csv_path}: {e}")
        return patients

    for col in ['PD 階段', '藥效']:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. Missing values will be assigned.")
            df[col] = np.nan

    for index, row in df.iterrows():
        date_str = str(row.get('日期', '')).strip().replace('.0', '').replace('/', '')
        patient_id = str(row.get('收案號', '')).strip().replace('.0', '')
        pd_stage_raw = row.get('PD 階段')
        medicine_raw = row.get('藥效')

        if not date_str or date_str == 'nan' or not patient_id or patient_id == 'nan':
            continue

        if pd.isna(pd_stage_raw) or str(pd_stage_raw).strip() == '' or str(pd_stage_raw) == '-':
            pd_stage = 0
        else:
            try:
                pd_stage = int(float(pd_stage_raw))
            except ValueError:
                pd_stage = 0

        on_medication = False
        if not pd.isna(medicine_raw):
            val = str(medicine_raw).strip()
            if val == '1' or val.lower() == 'true' or val.lower() == 'on':
                on_medication = True

        left_pt = find_pt_file(pt_base_dirs, date_str, patient_id, "_左手")
        right_pt = find_pt_file(pt_base_dirs, date_str, patient_id, "_右手")

        # Fallback to older format if newer format not found
        if not left_pt:
            left_pt = find_pt_file(pt_base_dirs, date_str, patient_id, "_L.pt")
        if not right_pt:
            right_pt = find_pt_file(pt_base_dirs, date_str, patient_id, "_R.pt")

        if not left_pt or not right_pt:
            print(f"Warning: Missing .pt files for Patient {patient_id} ({date_str}), skipping.")
            continue

        try:
            patient = PDPatient.from_pt(patient_id, pd_stage, date_str, on_medication, left_pt, right_pt)
            patients.append(patient)
        except Exception as e:
            print(f"Error loading data for Patient {patient_id}: {e}")

    return patients


# ---------- 工具函式 ----------
def _find_peaks_valleys(x, w=3):
    N = len(x); P=[]; V=[]
    for i in range(N):
        L=max(0,i-w); R=min(N,i+w+1)
        seg=x[L:R]
        if x[i] == np.max(seg): P.append(i)
        if x[i] == np.min(seg): V.append(i)
    return np.asarray(P, int), np.asarray(V, int)

def _pair_alternating_extrema(x, P, V):
    if len(P)==0 and len(V)==0:
        return np.array([], int), np.array([]), []
    eP = np.stack([P, np.ones_like(P)], 1)
    eV = np.stack([V, np.zeros_like(V)], 1)
    e  = np.concatenate([eP, eV], 0)
    e  = e[e[:,0].argsort()]
    centers=[]; amps=[]; pairs=[]
    for i in range(len(e)-1):
        i1,t1 = int(e[i,0]),   int(e[i,1])
        i2,t2 = int(e[i+1,0]), int(e[i+1,1])
        if t1!=t2:
            hi=max(x[i1], x[i2]); lo=min(x[i1], x[i2])
            amps.append((hi-lo)/2.0)
            centers.append((i1+i2)//2)
            pairs.append((i1,i2))
    return np.asarray(centers, int), np.asarray(amps, float), pairs

def _rolling_peak2peak_amp(x, W):
    N=len(x); A=np.full(N, np.nan); half=W//2
    for i in range(N):
        L=max(0,i-half); R=min(N,i+half+1)
        seg=x[L:R]
        A[i]=(np.max(seg)-np.min(seg))/2.0
    return A

def _rolling_rms(x, W):
    N=len(x); A=np.full(N, np.nan); half=W//2
    for i in range(N):
        L=max(0,i-half); R=min(N,i+half+1)
        seg=x[L:R]
        A[i]=np.sqrt(np.mean((seg-np.mean(seg))**2))
    return A

def _summarize(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))

def get_advanced_autocorr_features(wave, sampling_rate=60, degree=10):
    if len(wave) == 0:
        return 0.0, 0.0, 0.0, 0.0

    def autocorr(x):
        x_centered = x - np.mean(x)
        result = correlate(x_centered, x_centered, mode='full')
        return result[result.size // 2:]

    t = np.arange(len(wave))
    if t.max() == t.min():
        t_normalized = t
    else:
        t_normalized = (t - t.min()) / (t.max() - t.min())

    p = np.polyfit(t_normalized, wave, degree)
    trend = np.polyval(p, t_normalized)
    detrend = wave - trend

    autocorrelation_with_trend = autocorr(wave)
    autocorrelation_without_trend = autocorr(detrend)

    peaks_with_trend, _ = find_peaks(autocorrelation_with_trend, height=0)
    peaks_without_trend, _ = find_peaks(autocorrelation_without_trend, height=0)

    clarity_with_trend = 0.0
    frequency_with_trend = 0.0
    clarity_without_trend = 0.0
    frequency_without_trend = 0.0

    if peaks_with_trend.size > 0:
        first_peak = peaks_with_trend[0]
        if first_peak > 0:
            clarity_with_trend = autocorrelation_with_trend[first_peak] / (autocorrelation_with_trend[0] + 1e-9)
            frequency_with_trend = sampling_rate / first_peak

    if peaks_without_trend.size > 0:
        first_peak = peaks_without_trend[0]
        if first_peak > 0:
            clarity_without_trend = autocorrelation_without_trend[first_peak] / (autocorrelation_without_trend[0] + 1e-9)
            frequency_without_trend = sampling_rate / first_peak

    return frequency_with_trend, clarity_with_trend, frequency_without_trend, clarity_without_trend


def compute_amplitude_bundle(wave, w_extrema=10, window=None, detrend_order=None, dist=None):
    if dist is not None:
        # Avoid division by zero
        dist_safe = np.where(dist == 0, 1e-12, dist)
        wave = wave / dist_safe
    
    x = wave

    if detrend_order is not None and len(x) > detrend_order+1:
        t = np.arange(len(x), dtype=float)
        t = (t - t.min())/(t.max()-t.min() + 1e-12)
        p = np.polyfit(t, x, detrend_order)
        trend = np.polyval(p, t)
        x = x - trend

    P, V = _find_peaks_valleys(x, w=w_extrema)
    centers, amps_cycle, _pairs = _pair_alternating_extrema(x, P, V)

    if window is None:
        if len(P) > 1:
            window = int(np.median(np.diff(P)))
        else:
            window = 15
    window = max(5, int(window))

    A_roll = _rolling_peak2peak_amp(x, W=window)
    A_rms  = _rolling_rms(x, W=window)

    A_hilbert = None
    try:
        from scipy.signal import hilbert
        A_hilbert = np.abs(hilbert(x))
    except Exception:
        pass

    results = {
        'cycle_amp':  {'series': amps_cycle, 'mean': None, 'median': None, 'std': None},
        'rolling_p2p':{'series': A_roll,     'mean': None, 'median': None, 'std': None},
        'rms':        {'series': A_rms,      'mean': None, 'median': None, 'std': None},
        'hilbert':    {'series': A_hilbert,  'mean': None, 'median': None, 'std': None},
        'meta':       {'centers': centers, 'peaks': P, 'troughs': V, 'window': window}
    }
    
    for k in ['cycle_amp','rolling_p2p','rms','hilbert']:
        s = results[k]['series']
        if s is None:
            results[k]['mean'] = results[k]['median'] = results[k]['std'] = np.nan
        else:
            m, med, sd = _summarize(s)
            results[k]['mean'], results[k]['median'], results[k]['std'] = m, med, sd

    return results


def extract_features_from_patient(p: PDPatient) -> Tuple[np.ndarray, List[str]]:
    feats = []
    names = []
    
    L = p.left_trajectory.detach().cpu().numpy()
    R = p.right_trajectory.detach().cpu().numpy()

    for hand_label, arr in [('L', L), ('R', R)]:
        p0 = arr[:, 0, :]
        p4 = arr[:, 4, :]
        
        # dist is used as a normalizing factor per frame
        dist = np.linalg.norm(p4 - p0, axis=1)

        for j in range(21):
            for a, axis_name in enumerate(['x', 'y', 'z']):
                wave = arr[:, j, a]
                
                frequency_with_trend, clarity_with_trend, frequency_without_trend, clarity_without_trend = get_advanced_autocorr_features(wave)
                feats.append(frequency_without_trend)
                feats.append(clarity_without_trend)
                names.append(f"{hand_label}_joint{j:02d}_{axis_name}_frequency_without_trend")
                names.append(f"{hand_label}_joint{j:02d}_{axis_name}_clarity_without_trend")
                
                res = compute_amplitude_bundle(
                    wave,
                    w_extrema=10,
                    window=None,
                    detrend_order=None,
                    dist=dist
                )
                
                method_keys = ['cycle_amp', 'rolling_p2p', 'rms', 'hilbert']
                stat_keys   = ['mean', 'median', 'std']
                
                for mk in method_keys:
                    for sk in stat_keys:
                        feats.append(res[mk][sk])
                        names.append(f'{hand_label}_j{j}_{axis_name}_{mk}_{sk}')
                
    return np.array(feats, dtype=np.float32), names


def main():
    parser = argparse.ArgumentParser(description="從骨架 .pt 檔案與 CSV 抽取時間序列特徵 (Clarity)")
    parser.add_argument('--csv', type=str, required=True, help="病人 metadata CSV 檔案路徑")
    parser.add_argument('--pt_dir', type=str, required=True, help="儲存 .pt 的基礎資料夾,可使用逗號分隔多個路徑 (例如: dir1,dir2)")
    parser.add_argument('--output', type=str, default='extracted_features.csv', help="輸出的特徵 CSV 名稱")
    
    args = parser.parse_args()
    
    pt_base_dirs = [d.strip() for d in args.pt_dir.split(',')]
    
    print(f"Loading patient data from {args.csv}")
    print(f"Searching for .pt files in {pt_base_dirs}...")
    
    patients = load_patients_from_csv(args.csv, pt_base_dirs)
    print(f"Successfully loaded {len(patients)} patients with skeleton data.")
    
    if len(patients) == 0:
        print("No patient data found. Exiting.")
        return
        
    print("Extracting features using autocorrelation...")
    all_features = []
    feature_names = None
    
    for p in patients:
        feats, names = extract_features_from_patient(p)
        if feature_names is None:
            feature_names = names
            
        row_data = {
            'Date': p.date,
            'PID': p.patient_id,
            'PD_Stage': p.pd_stage,
            'On_Medication': p.on_medication
        }
        
        for name, val in zip(names, feats):
            row_data[name] = val
            
        all_features.append(row_data)
        
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(args.output, index=False)
    print(f"Feature extraction complete! Saved {df_features.shape[0]} rows and {df_features.shape[1]} columns to {args.output}")

if __name__ == '__main__':
    main()
