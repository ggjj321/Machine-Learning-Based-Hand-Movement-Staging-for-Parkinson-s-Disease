"""
Demonstration of four amplitude estimation methods used in extract_features.py:
  1. Cycle-by-Cycle (cycle_amp)
  2. Rolling Peak-to-Peak (rolling_p2p)
  3. Rolling RMS (rms)
  4. Hilbert Envelope (hilbert)

Uses the same real .pt file as plot_acf_demo.py (Patient 229, right hand, joint 4, x-axis).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import hilbert

# ── Reproduce the same helper functions as extract_features.py ─────────────────

def _find_peaks_valleys(x, w=4):
    N = len(x); P = []; V = []
    for i in range(N):
        L = max(0, i - w); R = min(N, i + w + 1)
        seg = x[L:R]
        if x[i] == np.max(seg): P.append(i)
        if x[i] == np.min(seg): V.append(i)
    print(f"Found {len(P)} peaks and {len(V)} valleys with window={w}")
    return np.asarray(P, int), np.asarray(V, int)

def _pair_alternating_extrema(x, P, V):
    if len(P) == 0 and len(V) == 0:
        return np.array([], int), np.array([]), []
    eP = np.stack([P, np.ones_like(P)], 1)
    eV = np.stack([V, np.zeros_like(V)], 1)
    e  = np.concatenate([eP, eV], 0)
    e  = e[e[:, 0].argsort()]
    centers = []; amps = []; pairs = []
    for i in range(len(e) - 1):
        i1, t1 = int(e[i, 0]),     int(e[i, 1])
        i2, t2 = int(e[i + 1, 0]), int(e[i + 1, 1])
        if t1 != t2:
            hi = max(x[i1], x[i2]); lo = min(x[i1], x[i2])
            amps.append((hi - lo) / 2.0)
            centers.append((i1 + i2) // 2)
            pairs.append((i1, i2))
    return np.asarray(centers, int), np.asarray(amps, float), pairs

def _rolling_peak2peak_amp(x, W):
    N = len(x); A = np.full(N, np.nan); half = W // 2
    for i in range(N):
        L = max(0, i - half); R = min(N, i + half + 1)
        seg = x[L:R]
        A[i] = (np.max(seg) - np.min(seg)) / 2.0
    return A

def _rolling_rms(x, W):
    N = len(x); A = np.full(N, np.nan); half = W // 2
    for i in range(N):
        L = max(0, i - half); R = min(N, i + half + 1)
        seg = x[L:R]
        A[i] = np.sqrt(np.mean((seg - np.mean(seg)) ** 2))
    return A

# ── Load real .pt file (same as plot_acf_demo.py) ─────────────────────────────
PT_PATH = (
    "/Users/wukeyang/mirlab_project/Machine-Learning-Based-Hand-Movement-Staging-for-Parkinson-s-Disease"
    "/hand_view_classifer/skeleton_sequences/skeleton_sequences_4_to_8/horizontal_view/stage_3"
    "/2025-05-29 13:23:20_gesture_20250529_131352__229_右手旋轉_REC_47CF8491-06AD-4605-B5C5-22279F2EE33F.pt"
)
JOINT = 4
AXIS  = 0
DEGREE = 10

data = torch.load(PT_PATH, map_location='cpu', weights_only=False)
traj = data['skeleton_sequence'] if isinstance(data, dict) and 'skeleton_sequence' in data else data
arr  = traj.numpy()   # (T, 21, 3)

# distance normalisation (wrist=0, middle MCP=9), same as extract_features.py
p0 = arr[:, 0, :]; p9 = arr[:, 9, :]
dist_scalar = float(np.median(np.linalg.norm(p9 - p0, axis=1)))
dist_scalar = dist_scalar if dist_scalar > 1e-12 else 1.0

wave_raw = arr[:200, JOINT, AXIS].astype(float) / dist_scalar

# polynomial detrend
T  = len(wave_raw)
t  = np.arange(T, dtype=float)
t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)
p_coef = np.polyfit(t_norm, wave_raw, DEGREE)
trend  = np.polyval(p_coef, t_norm)
wave   = wave_raw - trend

# ── Compute all four methods ───────────────────────────────────────────────────
P, V = _find_peaks_valleys(wave, w=3)
centers, amps_cycle, pairs = _pair_alternating_extrema(wave, P, V)

# window ≈ median inter-peak distance
if len(P) > 1:
    window = int(np.median(np.diff(P)))
else:
    window = 40
window = max(5, window)

A_roll    = _rolling_peak2peak_amp(wave, W=window)
A_rms     = _rolling_rms(wave, W=window)
A_hilbert = np.abs(hilbert(wave))

# ── Colours ───────────────────────────────────────────────────────────────────
C_WAVE    = '#4878CF'
C_CYCLE   = '#E84646'
C_ROLL    = '#F5A623'
C_RMS     = '#7DC462'
C_HILBERT = '#9B59B6'
C_PEAK    = '#E84646'
C_VALLEY  = '#2196F3'

def _base(ax):
    ax.plot(t, wave, color=C_WAVE, lw=1.0, alpha=0.5, zorder=1, label='Signal')
    ax.set_xlabel('Frame'); ax.set_ylabel('Normalised Amplitude')
    ax.grid(True, alpha=0.25)

# ── Fig 1: Cycle-by-Cycle ─────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 4))
_base(ax1)
pair_xs = [x for (i1, i2) in pairs for x in [i1, i2]]
pair_ys = [wave[x] for x in pair_xs]
for (i1, i2) in pairs:
    mid = (i1 + i2) / 2
    ax1.annotate('', xy=(mid, max(wave[i1], wave[i2])),
                 xytext=(mid, min(wave[i1], wave[i2])),
                 arrowprops=dict(arrowstyle='<->', color=C_CYCLE, lw=1.2))
ax1.scatter(pair_xs, pair_ys, color=C_CYCLE, s=10, zorder=5, label='cycle_amp')
ax1.set_title('Cycle by cycle', fontsize=13, fontweight='bold', color=C_CYCLE)
fig1.tight_layout()
p1 = os.path.join(os.path.dirname(__file__), 'amp_demo_cycle.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight'); print(f"Saved → {p1}")

# ── Fig 2: Rolling Peak-to-Peak ───────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
_base(ax2)
ax2.plot(t,  A_roll, color=C_ROLL, lw=2.0, label='rolling_p2p')
ax2.plot(t, -A_roll, color=C_ROLL, lw=2.0)
ax2.set_title('Rolling peak to peak', fontsize=13, fontweight='bold', color=C_ROLL)
fig2.tight_layout()
p2 = os.path.join(os.path.dirname(__file__), 'amp_demo_rolling_p2p.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight'); print(f"Saved → {p2}")

# ── Fig 3: Rolling RMS ────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 4))
_base(ax3)
ax3.plot(t,  A_rms, color=C_RMS, lw=2.0, label='rms')
ax3.plot(t, -A_rms, color=C_RMS, lw=2.0)
ax3.set_title('Rolling RMS', fontsize=13, fontweight='bold', color=C_RMS)
fig3.tight_layout()
p3 = os.path.join(os.path.dirname(__file__), 'amp_demo_rms.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight'); print(f"Saved → {p3}")

# ── Fig 4: Hilbert Envelope ───────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 4))
_base(ax4)
ax4.plot(t,  A_hilbert, color=C_HILBERT, lw=2.0, label='hilbert')
ax4.plot(t, -A_hilbert, color=C_HILBERT, lw=2.0)
ax4.set_title('Hilbert envelope', fontsize=13, fontweight='bold', color=C_HILBERT)
fig4.tight_layout()
p4 = os.path.join(os.path.dirname(__file__), 'amp_demo_hilbert.png')
fig4.savefig(p4, dpi=150, bbox_inches='tight'); print(f"Saved → {p4}")

plt.show()
