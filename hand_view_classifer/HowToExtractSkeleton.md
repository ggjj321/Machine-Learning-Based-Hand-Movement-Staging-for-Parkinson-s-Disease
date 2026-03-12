# 如何萃取手部骨架序列 (How to Extract Skeleton sequences)

這份文件說明了如何從原始病患手部旋轉影片，轉換成可用於機器學習訓練的 `.pt` 檔案張量資料（包含每幀手部的 3D 關鍵點），並自動處理分類。

## 工作流程與步驟說明

整個處理流程分為兩個主要步驟：

### 1. 視角自動分類 (`classify_hand_view.py`)
病患的手部旋轉影片主要有兩種拍攝視角：「由上往下拍 (top_down)」跟「從側邊水平拍 (horizontal)」。
此腳本會載入 `right_hand_files_...` (或是其他輸入的資料夾) 的影片，透過 MediaPipe 取前幾禎的手部關鍵點：
- 如果中指 Y 座標在手心下方（手背朝上），分類為 **Top Down View**
- 如果中指 Y 座標在手心上方（手心朝上），分類為 **Horizontal View**

**如何執行：**
直接執行腳本，程式中已預設了要掃描的日期資料夾與輸出至 `classified_hand_videos` 中。

```bash
python classify_hand_view.py
```

### 2. 骨架萃取與病歷綁定 (`process_videos_to_skeleton.py`)
主要的處理邏輯被統一在這支腳本裡，支援不同的執行模式來過濾並打包病患資料。它會尋找前面分類好的 `classified_hand_videos` 資料夾，讀取每一隻影片並萃取出 21 點 3D 骨架的 Tensor，再比對 `收案_CAREs...csv` 這個病歷表，找出每個影片所屬的 PD 期數分級。

這支程式支援三種運行模式 (透過命令列參數)：

#### A. 處理所有可配對的資料 (預設)
會解析收案表裡的所有人，將骨架抽出來放進期數對應的資料夾 `stage_1`, `stage_2` 等。如果期數無效或為空，則會歸類為 `stage_0`。

```bash
python process_videos_to_skeleton.py --mode all --csv "收案_CAREs 20251009-加密 - deID.csv" --input_dir classified_hand_videos --output_dir skeleton_sequences
```

#### B. 僅挑出未分期個案 (stage0_only)
如果只想專門將那些在 CSV 中「沒有填寫 PD 分期（空白或-）」的影片獨立挑出來做健康對照組或未分類組，請指派此模式，腳本會自動忽略其他有分期資料的病人。

```bash
python process_videos_to_skeleton.py --mode stage0_only --csv "收案_CAREs 20251009-加密 - deID.csv" --input_dir classified_hand_videos --output_dir skeleton_sequences_0
```

#### C. 獨立個人影片處理 (individual)
若有拿來測試用的一批單獨影片（例如放在 `Individual_handling_video` 資料夾中）且不想依賴 CSV 病歷表綁定，請使用這模式。程式會忽略載入 CSV，強制將影片的期數標註為 `-1` 並直接轉出成 `.pt` 供你檢視或測試。

```bash
python process_videos_to_skeleton.py --mode individual --individual_in Individual_handling_video --individual_out Individual_handling_video_output
```

## 輸出檔案的結構與讀取方式
最終輸出的 `.pt` 是一份 Dictionary，可使用 `torch.load()` 讀取，它包含了以下欄位：
- `date`：收案日期
- `case_number`：收案號
- `pd_stage`：PD 期數（0~4，-1 代表使用 individual 無匹配模式）
- `skeleton_sequence`：最關鍵的張量特徵。維度為 `(影片總幀數, 21, 3)` 的 PyTorch Tensor。每一幀包含 MediaPipe 21 個手部關鍵點的 `(x, y, z)` 歸一化座標向量。若中間有少數幾幀未偵測到手部，程式會以全零向量自動補齊來維持影片的時間連續長度。

### 3. 多維度時間序列特徵抽取 (`extract_features.py`)
當骨架的 `.pt` 檔案儲存完畢後，這支腳本負責將病患的 Metadata (`csv`) 與他們對應的 `_L.pt` 和 `_R.pt` 進行綁定，接著計算特徵。
目前的特徵演算法基於手部每個關節三維座標的自相關 (autocorrelation)，計算出週期運動的**清晰度 (Clarity)**，其被定義為「自相關的第一個高峰值除以延遲為 0 時的自相關值」。

腳本將會自動比對資料並計算每個患者左右手總共 21 個關節，各 `x, y, z` 軸的清晰度 (共 126 個特徵)，最後將其與 PD Stage 標籤匯出成一張結構化的 CSV 檔案，以方便後續餵入機器學習模型 (如 LightGBM、XGBoost、LDA 等)。

**如何執行：**
傳入你的 Metadata CSV 與骨架 .pt 檔案的根目錄，並指定輸出的 CSV 檔案名稱。
```bash
python extract_features.py --csv "收案_CAREs 20251009-加密 - deID.csv" --pt_dir skeleton_sequences/horizontal_view --output extracted_features.csv
```

