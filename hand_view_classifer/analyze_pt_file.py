#!/usr/bin/env python3
"""
讀取並分析 .pt 檔案，檢查 NaN 值
"""

import torch
import numpy as np

# 讀取 .pt 檔案
pt_file_path = "/Users/wukeyang/mirlab_project/hand-view_classifer/Individual_handling_video/2025-12-19 13_39_18_gesture_20251219_133233__547_左手旋轉_REC_57B58CDE-ED34-48B0-83D0-DFC35F3713BC.pt"

print("=" * 70)
print("讀取 .pt 檔案")
print("=" * 70)

data = torch.load(pt_file_path)

print("\n檔案內容:")
print("-" * 70)
for key, value in data.items():
    if key == 'skeleton_sequence':
        print(f"{key}: torch.Tensor with shape {value.shape}")
    else:
        print(f"{key}: {value}")

print("\n" + "=" * 70)
print("骨架序列分析")
print("=" * 70)

skeleton_sequence = data['skeleton_sequence']
print(f"\n形狀: {skeleton_sequence.shape}")
print(f"  - 總幀數: {skeleton_sequence.shape[0]}")
print(f"  - 關鍵點數: {skeleton_sequence.shape[1]}")
print(f"  - 座標維度: {skeleton_sequence.shape[2]}")

# 轉換為 numpy 以便分析
skeleton_np = skeleton_sequence.numpy()

# 檢查 NaN 值
nan_mask = np.isnan(skeleton_np)
total_elements = skeleton_np.size
nan_count = np.sum(nan_mask)

print(f"\n總元素數: {total_elements}")
print(f"NaN 值數量: {nan_count}")
print(f"NaN 值比例: {nan_count / total_elements * 100:.2f}%")

# 檢查每一幀是否有 NaN
frames_with_nan = []
for frame_idx in range(skeleton_np.shape[0]):
    if np.any(np.isnan(skeleton_np[frame_idx])):
        frames_with_nan.append(frame_idx)

print(f"\n包含 NaN 的幀數: {len(frames_with_nan)}")
if len(frames_with_nan) > 0:
    print(f"包含 NaN 的幀索引 (前 10 個): {frames_with_nan[:10]}")

# 檢查零值 (可能是填充的缺失幀)
zero_frames = []
for frame_idx in range(skeleton_np.shape[0]):
    if np.all(skeleton_np[frame_idx] == 0):
        zero_frames.append(frame_idx)

print(f"\n全零幀數 (可能是缺失幀): {len(zero_frames)}")
if len(zero_frames) > 0:
    print(f"全零幀索引 (前 10 個): {zero_frames[:10]}")

# 統計資訊
print("\n" + "=" * 70)
print("統計資訊")
print("=" * 70)
print(f"最小值: {np.nanmin(skeleton_np):.6f}")
print(f"最大值: {np.nanmax(skeleton_np):.6f}")
print(f"平均值: {np.nanmean(skeleton_np):.6f}")
print(f"標準差: {np.nanstd(skeleton_np):.6f}")

print("\n" + "=" * 70)
