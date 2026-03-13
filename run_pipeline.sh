#!/bin/bash

# ==============================================================================
# Hand Movement Staging for Parkinson's Disease - Automation Pipeline
# ==============================================================================
# This script automates the end-to-end process of:
# 1. Downloading video/skeleton data from Google Drive
# 2. Extracting the data
# 3. Classifying the hand view [If Video] (Top-Down vs Horizontal)
# 4. Extracting the 3D skeleton sequences [If Video] (.pt)
# 5. Extracting multi-dimensional clarity and amplitude features (.csv)
# 6. Running XGBoost LOOCV to output the final evaluation metrics
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# -------------- Configuration --------------
# Default paths and configurations (relative to the project root)
BASE_DIR="$(pwd)"
HAND_VIEW_DIR="hand_view_classifer"

# Video Workflow Paths
RAW_VIDEO_ZIP="${HAND_VIEW_DIR}/right_hand.zip"
RAW_EXTRACT_DIR="${HAND_VIEW_DIR}/raw_hand_videos"
CLASSIFIED_DIR="${HAND_VIEW_DIR}/classified_hand_videos"

# Skeleton / Feature Paths
RAW_SKELETON_ZIP="${HAND_VIEW_DIR}/skeleton.zip"
SKELETON_DIR="${HAND_VIEW_DIR}/skeleton_sequences"
HORIZONTAL_SKELETON_DIR="${SKELETON_DIR}/horizontal_view"
FEATURE_OUTPUT="extracted_features.csv"

# Global Paths
CSV_METADATA="收案_CAREs 20251009-加密 - deID.csv"
XGB_EVAL_SCRIPT="xgb_exp/xgb_loocv_eval.py"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --video-url <URL>      Google Drive link to a .zip containing Raw Hand Videos."
    echo "                         (Executes full pipeline: Classify -> Skeleton -> ML)"
    echo "  --skeleton-url <URL>   Google Drive link to a .zip containing Extracted Skeletons (.pt)."
    echo "                         (Executes shortened pipeline: Skeleton -> ML)"
    echo ""
    echo "Examples:"
    echo "  $0 --video-url https://drive.google.com/file/d/1-CpQWyx66Su.../view"
    echo "  $0 --skeleton-url https://drive.google.com/file/d/1gXNraemdy.../view"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

PIPELINE_MODE=""
GDRIVE_INPUT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video-url)
            PIPELINE_MODE="VIDEO"
            GDRIVE_INPUT="$2"
            shift 2
            ;;
        --skeleton-url)
            PIPELINE_MODE="SKELETON"
            GDRIVE_INPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

if [ -z "$GDRIVE_INPUT" ]; then
    echo "Error: URL cannot be empty."
    usage
fi

# Automatically extract ID from URL if a full URL is provided
if [[ "$GDRIVE_INPUT" == *"drive.google.com"* ]]; then
    FILE_ID=$(echo "$GDRIVE_INPUT" | grep -o '/d/[a-zA-Z0-9_-]*' | sed 's/\/d\///')
    if [ -z "$FILE_ID" ]; then
        FILE_ID=$(echo "$GDRIVE_INPUT" | grep -o 'id=[a-zA-Z0-9_-]*' | sed 's/id=//')
    fi
else
    FILE_ID="$GDRIVE_INPUT"
fi

if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please run 'pip install gdown'"
    exit 1
fi

echo "========================================"
echo " Starting Automation Pipeline (Mode: $PIPELINE_MODE)"
echo "========================================"

mkdir -p "$HAND_VIEW_DIR"

if [ "$PIPELINE_MODE" == "VIDEO" ]; then
    # ==============================
    # VIDEO WORKFLOW
    # ==============================
    echo -e "\n---> Step 1: Downloading Video Zip from Google Drive (ID: $FILE_ID)..."
    gdown --id "$FILE_ID" -O "$RAW_VIDEO_ZIP"

    echo -e "\n---> Step 2: Extracting $RAW_VIDEO_ZIP..."
    mkdir -p "$RAW_EXTRACT_DIR"
    unzip -q -o "$RAW_VIDEO_ZIP" -d "$RAW_EXTRACT_DIR"
    echo "Extracted to $RAW_EXTRACT_DIR"

    EXTRACTED_SUBDIR=$(find "$RAW_EXTRACT_DIR" -maxdepth 1 -mindepth 1 -type d | head -n 1)
    if [ -z "$EXTRACTED_SUBDIR" ]; then
        EXTRACTED_SUBDIR="$RAW_EXTRACT_DIR"
    fi
    echo "Using input directory: $EXTRACTED_SUBDIR"

    echo -e "\n---> Step 3: Classifying hand views (Top-Down vs Horizontal)..."
    cat << 'EOF' > temp_classify.py
import sys
import os
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

    python temp_classify.py "$EXTRACTED_SUBDIR" "$CLASSIFIED_DIR"
    rm temp_classify.py

    echo -e "\n---> Step 4: Extracting 3D Skeleton Sequences..."
    if [ ! -f "$CSV_METADATA" ]; then
        echo "Error: Metadata CSV ($CSV_METADATA) not found at $(pwd)!"
        exit 1
    fi

    python "${HAND_VIEW_DIR}/process_videos_to_skeleton.py" \
        --mode all \
        --csv "$CSV_METADATA" \
        --input_dir "$CLASSIFIED_DIR" \
        --output_dir "$SKELETON_DIR"

elif [ "$PIPELINE_MODE" == "SKELETON" ]; then
    # ==============================
    # SKELETON WORKFLOW
    # ==============================
    echo -e "\n---> Step 1 & 2: Downloading & Extracting Skeleton Data (ID: $FILE_ID)..."
    gdown --id "$FILE_ID" -O "$RAW_SKELETON_ZIP"
    
    # Clean old skeleton dir to prevent overlap bugs
    rm -rf "$SKELETON_DIR"
    mkdir -p "$SKELETON_DIR"
    unzip -q -o "$RAW_SKELETON_ZIP" -d "$SKELETON_DIR"
    echo "Extracted Skeletons to $SKELETON_DIR"

    # Some zip structures might put everything inside a nested folder (e.g. skeleton_sequences/horizontal_view/...)
    # Check if horizontal_view exists immediately inside SKELETON_DIR, if not, find it.
    if [ ! -d "$HORIZONTAL_SKELETON_DIR" ]; then
        FOUND_HORIZ=$(find "$SKELETON_DIR" -type d -name "horizontal_view" | head -n 1)
        if [ ! -z "$FOUND_HORIZ" ] && [ "$FOUND_HORIZ" != "$HORIZONTAL_SKELETON_DIR" ]; then
            echo "Relocating nested horizontal_view from $FOUND_HORIZ to $HORIZONTAL_SKELETON_DIR"
            mv "$FOUND_HORIZ" "$HORIZONTAL_SKELETON_DIR"
        fi
    fi
fi

# ==============================
# COMMON WORKFLOW
# ==============================
echo -e "\n---> Step 5: Extracting time-series features (Clarity & Amplitude)..."

if [ ! -d "$HORIZONTAL_SKELETON_DIR" ]; then
    echo "Error: No horizontal_view directory found in $SKELETON_DIR. Cannot proceed to feature extraction."
    exit 1
fi

if [ ! -f "$CSV_METADATA" ]; then
    echo "Error: Metadata CSV ($CSV_METADATA) not found at $(pwd)!"
    exit 1
fi

python "${HAND_VIEW_DIR}/extract_features.py" \
    --csv "$CSV_METADATA" \
    --pt_dir "./hand_view_classifer/skeleton_sequences/skeleton_sequences_4_to_8/horizontal_view" \
    --output "$FEATURE_OUTPUT"


echo -e "\n---> Step 6: Running XGBoost LOOCV ML Evaluation..."
if [ ! -f "$FEATURE_OUTPUT" ]; then
    echo "Error: Features CSV ($FEATURE_OUTPUT) was not generated."
    exit 1
fi

if [ ! -f "$XGB_EVAL_SCRIPT" ]; then
     echo "Error: ML Evaluation script not found at $XGB_EVAL_SCRIPT"
     exit 1
fi

python "$XGB_EVAL_SCRIPT" \
    --csv_path "$FEATURE_OUTPUT" \
    --k_features 10 \
    --use_youden \
    --dataset_source all \
    --mode both

echo -e "\n========================================"
echo " Pipeline Execution Completed Successfully!"
echo " ML Metrics and Plots are saved in: xgb_exp/results/"
echo "========================================"
