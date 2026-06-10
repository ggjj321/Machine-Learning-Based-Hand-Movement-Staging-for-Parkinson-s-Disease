"""
Plot Wave+Trend, Detrend Result, and Autocorrelation Result
for a selected joint/axis using the same method as extract_features.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, find_peaks

PT_PATH = (
    "/Users/wukeyang/mirlab_project/Machine-Learning-Based-Hand-Movement-Staging-for-Parkinson-s-Disease"
    "/hand_view_classifer/skeleton_sequences/skeleton_sequences_4_to_8/horizontal_view/stage_3"
    "/2025-05-29 13:23:20_gesture_20250529_131352__229_右手旋轉_REC_47CF8491-06AD-4605-B5C5-22279F2EE33F.pt"
)

JOINT = 4       # middle finger MCP — typically shows clear rotation signal
AXIS  = 0       # 0=x, 1=y, 2=z
AXIS_NAME = ['x', 'y', 'z'][AXIS]
DEGREE = 10     # polynomial detrend order (same as extract_features.py)
SAMPLING_RATE = 60


# ── load skeleton ──────────────────────────────────────────────────────────────
data = torch.load(PT_PATH, map_location='cpu', weights_only=False)
if isinstance(data, dict) and 'skeleton_sequence' in data:
    traj = data['skeleton_sequence']
else:
    traj = data

arr = traj.numpy()          # (T, 21, 3)
wave_raw = arr[:, JOINT, AXIS].astype(float)
T = len(wave_raw)
frames = np.arange(T)

# ── polynomial trend ───────────────────────────────────────────────────────────
t_norm = (frames - frames.min()) / (frames.max() - frames.min() + 1e-12)
p = np.polyfit(t_norm, wave_raw, DEGREE)
trend = np.polyval(p, t_norm)
detrend = wave_raw - trend

# ── autocorrelation ────────────────────────────────────────────────────────────
def autocorr(x):
    xc = x - np.mean(x)
    result = correlate(xc, xc, mode='full')
    return result[result.size // 2:]

acf = autocorr(detrend)
acf_lags = np.arange(len(acf))

peaks, _ = find_peaks(acf, height=0)

# mark first peak
first_peak_lag  = peaks[0] if len(peaks) > 0 else None
first_peak_val  = acf[first_peak_lag] if first_peak_lag is not None else None
clarity = (first_peak_val / (acf[0] + 1e-9)) if first_peak_lag is not None else 0.0
frequency = (SAMPLING_RATE / first_peak_lag) if first_peak_lag is not None else 0.0


# ── plotting ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 4.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

COLOR_WAVE  = '#1f77b4'
COLOR_TREND = '#d62728'
COLOR_ACF   = '#1f77b4'
COLOR_PEAK  = '#d62728'

# --- Panel 1: Wave + Trend ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(frames, wave_raw, color=COLOR_WAVE,  lw=1.2, label='Joint Movement')
ax1.plot(frames, trend,    color=COLOR_TREND, lw=1.5, label='Trend', linestyle='--')
ax1.set_title('Wave + Trend', fontsize=13, fontweight='bold')
ax1.set_xlabel('Frame Index')
ax1.set_ylabel('Value')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Detrend Result ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(frames, detrend, color=COLOR_WAVE, lw=1.2, label='Detrended')
ax2.axhline(0, color='gray', lw=0.8, linestyle='--')
ax2.set_title('Detrend Result', fontsize=13, fontweight='bold')
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Value')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Autocorrelation Result ---
ax3 = fig.add_subplot(gs[2])
ax3.plot(acf_lags, acf, color=COLOR_ACF, lw=1.2, label='Autocorrelation Result')
if first_peak_lag is not None:
    ax3.axvline(first_peak_lag, color=COLOR_PEAK, lw=1.5, linestyle='--',
                label=f'1st peak (lag={first_peak_lag})')
    ax3.scatter([first_peak_lag], [first_peak_val], color=COLOR_PEAK, zorder=5, s=50)
    # annotation box
    ax3.annotate(
        f"Frequency = {frequency:.2f} Hz\nClarity = {clarity:.3f}",
        xy=(first_peak_lag, first_peak_val),
        xytext=(first_peak_lag + max(T*0.08, 20), first_peak_val * 0.8),
        fontsize=8.5,
        arrowprops=dict(arrowstyle='->', color='gray'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )
ax3.set_title('Autocorrelation Result', fontsize=13, fontweight='bold')
ax3.set_xlabel('Frame Index')
ax3.set_ylabel('Variance')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

fig.suptitle(
    f'Patient 229  |  Right Hand  |  Joint {JOINT} ({AXIS_NAME}-axis)  |  Stage 3',
    fontsize=12, fontweight='bold', y=1.02
)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'acf_demo_229_right.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved → {save_path}")
plt.show()
