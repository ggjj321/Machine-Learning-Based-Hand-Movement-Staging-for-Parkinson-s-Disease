#!/bin/bash

# ==============================================================================
# Hand Movement Staging for Parkinson's Disease - Automation Pipeline
# ==============================================================================
# This script automates the end-to-end process of:
# 1. Downloading video data from Google Drive
# 2. Extracting the videos
# 3. Classifying the hand view (Top-Down vs Horizontal)
# 4. Extracting the 3D skeleton sequences (.pt)
# 5. Extracting multi-dimensional clarity and amplitude features (.csv)
# 6. Running XGBoost LOOCV to output the final evaluation metrics
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# -------------- Configuration --------------
# Default paths and configurations (relative to the project root)
BASE_DIR="$(pwd)"
HAND_VIEW_DIR="hand_view_classifer"
RAW_ZIP="${HAND_VIEW_DIR}/right_hand.zip"
RAW_EXTRACT_DIR="${HAND_VIEW_DIR}/raw_hand_videos"
CLASSIFIED_DIR="${HAND_VIEW_DIR}/classified_hand_videos"
SKELETON_DIR="${HAND_VIEW_DIR}/skeleton_sequences"
FEATURE_OUTPUT="${HAND_VIEW_DIR}/extracted_features.csv"
CSV_METADATA="收案_CAREs 20251009-加密 - deID.csv"
XGB_EVAL_SCRIPT="xgb_exp/xgb_loocv_eval.py"

# Function to display usage
usage() {
    echo "Usage: $0 [GOOGLE_DRIVE_URL_OR_ID]"
    echo ""
    echo "Arguments:"
    echo "  GOOGLE_DRIVE_URL_OR_ID    Google Drive shareable link or file ID to the .zip containing the videos."
    echo ""
    echo "Example:"
    echo "  $0 https://drive.google.com/file/d/1-CpQWyx66SuOzbZuUD5ED0vcMCt4X4FE/view?usp=sharing"
    exit 1
}

# Check if URL/ID argument is provided
if [ $# -eq 0 ]; then
    usage
fi

GDRIVE_INPUT="$1"

# Automatically extract ID from URL if a full URL is provided
if [[ "$GDRIVE_INPUT" == *"drive.google.com"* ]]; then
    # Extract the ID which is usually between /d/ and /view
    FILE_ID=$(echo "$GDRIVE_INPUT" | grep -o '/d/[a-zA-Z0-9_-]*' | sed 's/\/d\///')
    if [ -z "$FILE_ID" ]; then
        # Try another variation where ID is in the id= parameter
        FILE_ID=$(echo "$GDRIVE_INPUT" | grep -o 'id=[a-zA-Z0-9_-]*' | sed 's/id=//')
    fi
else
    FILE_ID="$GDRIVE_INPUT"
fi

echo "========================================"
echo " Starting Automation Pipeline"
echo "========================================"

# -------------- Step 1: Download --------------
echo -e "\n---> Step 1: Downloading from Google Drive (ID: $FILE_ID)..."
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please run 'pip install gdown'"
    exit 1
fi

mkdir -p "$HAND_VIEW_DIR"
gdown --id "$FILE_ID" -O "$RAW_ZIP"


# -------------- Step 2: Extraction --------------
echo -e "\n---> Step 2: Extracting $RAW_ZIP..."
mkdir -p "$RAW_EXTRACT_DIR"
# Unzip quietly, overwrite without prompting
unzip -q -o "$RAW_ZIP" -d "$RAW_EXTRACT_DIR"
echo "Extracted to $RAW_EXTRACT_DIR"

# Try to find the actual directory inside that matches the date pattern (e.g. right_hand_files_2025...)
# Assuming the zip extracts into a subfolder, we want to find it to pass to the classifier
# Find directories under RAW_EXTRACT_DIR
EXTRACTED_SUBDIR=$(find "$RAW_EXTRACT_DIR" -maxdepth 1 -mindepth 1 -type d | head -n 1)

if [ -z "$EXTRACTED_SUBDIR" ]; then
    # If the zip didn't contain a root folder and just dumped files
    EXTRACTED_SUBDIR="$RAW_EXTRACT_DIR"
fi
echo "Using input directory: $EXTRACTED_SUBDIR"


# -------------- Step 3: View Classification --------------
echo -e "\n---> Step 3: Classifying hand views (Top-Down vs Horizontal)..."
cat << 'EOF' > temp_classify.py
import sys
import os
# Add hand_view_classifer to the python path so we can import HandViewClassifier
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_view_classifer'))
from classify_hand_view import HandViewClassifier

def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    classifier = HandViewClassifier(input_folder, output_folder)
    classifier.process_videos()

if __name__ == "__main__":
    main()
EOF

python3 temp_classify.py "$EXTRACTED_SUBDIR" "$CLASSIFIED_DIR"
rm temp_classify.py


# -------------- Step 4: Skeleton Extraction --------------
echo -e "\n---> Step 4: Extracting 3D Skeleton Sequences..."
if [ ! -f "$CSV_METADATA" ]; then
    echo "Warning: Metadata CSV ($CSV_METADATA) not found at $(pwd)!"
    echo "Please ensure the CSV is placed at the specified location."
    exit 1
fi

python3 "${HAND_VIEW_DIR}/process_videos_to_skeleton.py" \
    --mode all \
    --csv "$CSV_METADATA" \
    --input_dir "$CLASSIFIED_DIR" \
    --output_dir "$SKELETON_DIR"


# -------------- Step 5: Feature Extraction --------------
echo -e "\n---> Step 5: Extracting time-series features (Clarity & Amplitude)..."
# Only feature extraction on the horizontal view is supported in the ML pipeline.
# We will extract features from skeleton_sequences/horizontal_view
HORIZONTAL_SKELETON_DIR="${SKELETON_DIR}/horizontal_view"

if [ ! -d "$HORIZONTAL_SKELETON_DIR" ]; then
    echo "No horizontal_view directory found in $SKELETON_DIR. Cannot proceed to feature extraction."
    exit 1
fi

python3 "${HAND_VIEW_DIR}/extract_features.py" \
    --csv "$CSV_METADATA" \
    --pt_dir "$HORIZONTAL_SKELETON_DIR" \
    --output "$FEATURE_OUTPUT"


# -------------- Step 6: ML Evaluation --------------
echo -e "\n---> Step 6: Running XGBoost LOOCV ML Evaluation..."
if [ ! -f "$FEATURE_OUTPUT" ]; then
    echo "Features CSV ($FEATURE_OUTPUT) was not generated."
    exit 1
fi

if [ ! -f "$XGB_EVAL_SCRIPT" ]; then
     echo "Error: ML Evaluation script not found at $XGB_EVAL_SCRIPT"
     exit 1
fi

echo "Running XGBoost Evaluation..."
python3 "$XGB_EVAL_SCRIPT" \
    --csv_path "$FEATURE_OUTPUT" \
    --k_features 10 \
    --use_youden \
    --dataset_source all \
    --mode both

echo -e "\n========================================"
echo " Pipeline Execution Completed Successfully!"
echo " ML Metrics and Plots are saved in: xgb_exp/results/"
echo "========================================"
