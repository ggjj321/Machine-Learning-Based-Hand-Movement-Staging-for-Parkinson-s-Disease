# ACGN Experiment Execution Guide

本文件為 `acgn_exp` 目錄下的各個實驗程式碼執行指南，適用於 HackMD 筆記分享。

## 1. 特徵分類器訓練 (`train_feature.py`)

此腳本負責使用萃取好的 CSV 特徵進行模型訓練，支援 MLP 以及 AGCN-style 兩種架構，且可搭配 Linear 或 XGBoost 分類器。

### 基本指令
```bash
python acgn_exp/train_feature.py --csv_path path/to/features.csv [選項]
```

### 重要參數
- `--csv_path` (必填): 特徵 CSV 檔案的路徑。
- `--dataset_source`: 選擇要使用的資料集來源：`horizontal`, `old` 或 `all`。
- `--medication_filter`: 藥效篩選：`no_medication` (Off狀態), `with_medication` (On狀態) 或 `all`。
- `--model_type`: 模型架構，可選 `mlp` 或 `agcn_style`。
- `--classifier_type`: (當選擇 agcn_style 時) 分類器類型：可選 `linear` 或是 `xgboost`。
- `--adj_mode`: 相鄰矩陣 (Adjacency Matrix) 模式：`separate_block` (原專案各 Block 分離獨立) 或 `same_block` (基於 AGCN 論文共用的相鄰矩陣)。
- `--cv_type`: 交叉驗證方式：`kfold` (K-Fold) 或是 `loocv` (Leave-One-Out)。
- `--n_splits`: 如果選擇 `kfold`，則此處為切分數量。

---

## 2. 跨資料集評估 (`cross_dataset_eval.py`)

此腳本用於跨資料集 (Cross-Dataset) 的驗證，可將以 `old` 資料集訓練的模型應用於 `horizontal` 資料集進行評估（或反之）。

### 基本指令
```bash
python acgn_exp/cross_dataset_eval.py --csv_path path/to/features.csv \
    --old_checkpoint path/to/old_model.pt \
    --horizontal_checkpoint path/to/horizontal_model.pt [選項]
```

### 重要參數
- `--old_checkpoint`: 於 old 資料集訓練出的模型權重檔 (`.pt`)。
- `--horizontal_checkpoint`: 於 horizontal 資料集訓練出的模型權重檔 (`.pt`)。
- `--model_type`: 需與訓練時對應的模型架構一致 (`mlp` 或是 `agcn_style`)。
- `--save_dir`: 儲存評估結果、混淆矩陣圖表 (`.png`) 的資料夾路徑。

---

## 3. AGCN 骨架序列訓練 (`train_agcn.py`)

不依賴純特徵 CSV，而是直接針對骨架序列資料 (經過預處理為 `.pt` 的張量檔案) 來訓練 AGCN 網路。

### 基本指令
```bash
python acgn_exp/train_agcn.py --data_dir path/to/skeleton_sequences [選項]
```

### 重要參數
- `--data_dir`: 骨架序列資料的目錄位置。
- `--max_frames`: 長度對齊 (Padding) 所使用的最大 Frame 數量。
- `--epochs`, `--batch_size`, `--lr`: 控制訓練週期、批次量及學習率。
- `--save_dir`: Checkpoint 權重的儲存位置。

---

## 4. 相鄰矩陣視覺化 (`visualize_adjacency.py`)

訓練完成後，我們常常會想探究 AGCN 學習到了什麼樣的骨架關聯拓樸。此腳本能視覺化學習出的相鄰矩陣。

### 基本指令
```bash
python acgn_exp/visualize_adjacency.py --checkpoint path/to/matrix.pt --output output_plot.png [--show_labels]
```

### 重要參數
- `--checkpoint`: 學習後的相鄰矩陣路徑檔案 (`.pt`)。
- `--output`: 輸出的圖片路徑（如 `adacency.png`）。
- `--show_labels`: 加上此 flag 可在圖表的 x, y 軸顯示出每個骨節點 (Joint) 的標籤。
