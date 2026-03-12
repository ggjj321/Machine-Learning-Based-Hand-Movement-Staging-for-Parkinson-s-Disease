# Hand Movement Staging for Parkinson's Disease - Automated Pipeline

This repository contains a complete end-to-end automated pipeline to process raw hand movement videos, extract 3D structural kinematics, and evaluate the ML classification performance using XGBoost.

## 🚀 Quick Start

You can run the entire pipeline with a single command from the project root via the `run_pipeline.sh` script.

### Prerequisites
Make sure you have installed the required Python packages and system utilities:
```bash
# Core dependencies for processing and evaluating
pip install mediapipe pandas numpy scipy scikit-learn xgboost matplotlib seaborn
# Required to download the zip file from Google Drive
pip install gdown
```

Ensure the metadata CSV is located in the project root:
- `收案_CAREs 20251009-加密 - deID.csv`

### Usage
Run the script by passing the Google Drive URL or File ID of the zipped video folder directly:

```bash
# Provide execute permission to the script (only need to do this once)
chmod +x run_pipeline.sh

# Run the pipeline with a Google Drive Link
./run_pipeline.sh "https://drive.google.com/file/d/1-CpQWyx66SuOzbZuUD5ED0vcMCt4X4FE/view?usp=sharing"
```

## 🧩 Pipeline Steps Breakdown

The `run_pipeline.sh` automates the following steps systematically:

1. **Download Data**: Uses `gdown` to download the zip file to `hand_view_classifer/right_hand.zip`.
2. **Extract Files**: Unzips the contents into a `hand_view_classifer/raw_hand_videos/` directory.
3. **View Classification** (`hand_view_classifer/classify_hand_view.py`):
   - Scans the raw videos using MediaPipe to detect the hand orientation in the first few frames.
   - Separates videos into two folders inside `hand_view_classifer/classified_hand_videos/`: 
     - `top_down_view/` (Palm facing down)
     - `horizontal_view/` (Palm facing up/sideways)
4. **Skeleton Extraction** (`hand_view_classifer/process_videos_to_skeleton.py`):
   - Reads the categorized `classified_hand_videos/`.
   - Iterates through the videos, parsing a full sequence of 21 3D landmarks for every frame via MediaPipe.
   - Aligns the subject with their PD Stage from the `收案_CAREs...csv` file.
   - Saves the resulting PyTorch Tensors (`.pt`) to `hand_view_classifer/skeleton_sequences/`.
5. **Feature Extraction** (`hand_view_classifer/extract_features.py`):
   - Analyzes the skeleton kinematics (currently from `horizontal_view`).
   - Computes advanced time-series features like **Autocorrelation Clarity**, **Frequency**, and Rolling Maximum/Minimum Amplitudes for all (xyz) coordinate axes across the 21 joints.
   - Saves a compiled dataset to `hand_view_classifer/extracted_features.csv`.
6. **ML Staging / Evaluation** (`xgb_exp/xgb_loocv_eval.py`):
   - Takes the `extracted_features.csv`.
   - Runs a highly tuned XGBoost algorithm utilizing **Leave-One-Out Cross-Validation (LOOCV)**.
   - Selects the top `k=10` prominent features dynamically.
   - Calculates predictive probabilities and evaluates Threshold Optimization (Youden's J Index).
   - Generates Confusion Matrices, ROC Curves, and exports a numerical metrics report directly to `xgb_exp/results/`.

## 📊 Viewing the Results

Once the script completes successfully, navigate to the `xgb_exp/results/` directory:
- `metrics_...csv`: A detailed text-based file outlining metrics like Accuracy, Precision, Recall, F1, and AUROC.
- `roc_...png`: The Receiver Operating Characteristic (ROC) curve plotting FPR vs TPR.
- `cm_...png`: Confusion matrices illustrating healthy vs PD classification distribution.
