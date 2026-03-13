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

The script accepts two different modes depending on whether you want to process raw videos from scratch or start from pre-extracted skeletons.

```bash
# Provide execute permission to the script (only need to do this once)
chmod +x run_pipeline.sh

# 1. Full Pipeline (Starts from Raw MP4 Videos)
./run_pipeline.sh --video-url "https://drive.google.com/file/d/1-CpQWyx66SuOzbZuUD5ED0vcMCt4X4FE/view?usp=sharing"

# 2. Short Pipeline (Starts from Extracted `.pt` Skeletons, bypasses video extraction)
./run_pipeline.sh --skeleton-url "https://drive.google.com/file/d/1gXNraemdykV03ITTreTBWPZ1tWQJnXbP/view?usp=sharing"
```

## 🧩 Pipeline Steps Breakdown

Depending on the mode chosen, the script automates the following steps:

### When using `--video-url`:
1. **Download Data**: Uses `gdown` to download the raw video zip file.
2. **Extract Files**: Unzips the contents into a `hand_view_classifer/raw_hand_videos/` directory.
3. **View Classification** (`hand_view_classifer/classify_hand_view.py`):
   - Scans the raw videos using MediaPipe to detect hand orientation.
   - Separates videos into `top_down_view/` and `horizontal_view/`.
4. **Skeleton Extraction** (`hand_view_classifer/process_videos_to_skeleton.py`):
   - Generates a full sequence of 21 3D landmarks for every frame.
   - Saves the PyTorch Tensors (`.pt`) to `hand_view_classifer/skeleton_sequences/`.

### When using `--skeleton-url`:
1. **Skip Video Processing**: The script bypasses steps 1-4.
2. **Download & Extract Skeletons**: Automatically fetches and extracts the Zip file containing pre-processed PyTorch Tensors (`.pt`) directly into `hand_view_classifer/skeleton_sequences/`.

### Shared ML Steps (Both Modes):
5. **Feature Extraction** (`hand_view_classifer/extract_features.py`):
   - Analyzes kinematics from the generated/downloaded skeletons in `horizontal_view`.
   - Computes advanced time-series features like Autocorrelation Clarity, Frequency, and Amplitude arrays.
   - Compiles a dataset to `hand_view_classifer/extracted_features.csv`.
6. **ML Staging / Evaluation** (`xgb_exp/xgb_loocv_eval.py`):
   - Takes the extracted CSV and runs Leave-One-Out Cross-Validation on an XGBoost model.
   - Selects Top 10 prominent metrics dynamically.
   - Calculates predictive probabilities and Youden's J Index.
   - Generates and saves the metrics to `xgb_exp/results/`.

## 📊 Viewing the Results

Once the script completes successfully, navigate to the `xgb_exp/results/` directory:
- `metrics_...csv`: A detailed text-based file outlining metrics like Accuracy, Precision, Recall, F1, and AUROC.
- `roc_...png`: The Receiver Operating Characteristic (ROC) curve plotting FPR vs TPR.
- `cm_...png`: Confusion matrices illustrating healthy vs PD classification distribution.
