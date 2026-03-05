# XGBoost Feature Selection & Evaluation

本程式 (`xgb_loocv_eval.py`) 為使用特徵選擇搭配 XGBoost 分類器，並基於 Leave-One-Out Cross-Validation (LOOCV) 進行模型評估的實驗腳本。

## 執行需求
請確認您擁有以下套件：
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tqdm
```
*(注意：若執行時出現 `ModuleNotFoundError: No module named 'xgboost'`，請確保已安裝 xgboost。)*

---

## 腳本執行指令

### 一次跑出標準混合版本與左右手 Stacked 版本
```bash
python xgb_exp/xgb_loocv_eval.py --csv_path "pd_features_with_medication(1).csv" --dataset_source all --mode both
```

### 只跑左右手分離特徵版 (Stacked Mode)
若是只想跑先分左右手獨立訓練 XGBoost 再用 Linear Regression 結合分數的版本，可以加上 `--mode stacked`：
```bash
python xgb_exp/xgb_loocv_eval.py --csv_path "pd_features_with_medication(1).csv" --mode stacked
```

### 啟用 Youden's J Index 最佳化閾值
加上 `--use_youden` 參數，模型會在繪製 ROC 曲線時，自動根據 Youden's J 找出最佳閾值，並以此閾值計算 Accuracy、F1-score 與 Confusion Matrix。
```bash
python xgb_exp/xgb_loocv_eval.py --csv_path "pd_features_with_medication(1).csv" \
    --dataset_source horizontal \
    --use_youden
```

---

### 參數清單說明
* `--csv_path`: (必填) 特徵 CSV 檔案路徑。
* `--k_features`: (選用) 設定要篩選的 Top K 特徵數量，預設為 `10`。
* `--use_youden`: (選用) 是否使用 Youden's J 尋找最佳機率閾值。不加此參數則預設使用 `0.5` 進行決策。
* `--dataset_source`: (選用) 指定要跑的資料來源。可選 `horizontal`, `old`, `all`。預設為 `all`。
* `--mode`: (選用) 控制 LOOCV 計算模式。
  * `standard`: 一般混合雙手所有特徵。
  * `stacked`: 將 `L_` 開頭特徵與 `R_` 開頭特徵分離，各自由 XGBoost 預測後經由 Linear Regression 進行 Stack 堆疊。
  * `both`: 同時依序執行上列兩種模式並產出圖表 (預設)。 
* `--save_dir`: (選用) 圖表與評估指標 CSV 的儲存目錄徑。預設為 `xgb_exp/results`。
* `--no_show`: (選用) 加上此參數將不會彈出 pyplot 視窗，適合在背景或伺服器上執行。

---

## 輸出結果
1. **Console 輸出**: LOOCV 過程進度條、評估指標報表以及由各方法挑選出來的 Top K 特徵。
2. **圖表儲存**: 會在 `--save_dir` (預設為 `xgb_exp/results/`) 下儲存不同實驗模式的 ROC Curve 與 Confusion Matrix。
3. **數據儲存**: 將模型評估的 Metrics 數據存為對應實驗模式名稱的 CSV (`metrics_Standard_LOOCV_horizontal.csv` 等)。
