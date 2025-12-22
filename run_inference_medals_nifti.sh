#!/bin/bash
#
# Medal-S Inference Script
#
# This script runs Medal-S inference in Stage 1 + Stage 2 mode:
#   - Stage 1 + Stage 2: Accurate two-stage inference with ROI refinement
#
# Usage:
#   bash run_inference_medals_nifti.sh
#   bash run_inference_medals_nifti.sh [input_path] [output_dir] [device] [checkpoints_path]
#
# Configuration Files:
#   - CT images: Use config_CT.json (supports multi-window types: soft_tissue, bone, lung)
#     * Multiple window types: Each window type will be processed separately and merged
#     * Single window type: Uses the corresponding window settings
#   - Non-CT images: Use config_nonCT.json (MRI, US, PET, microscopy)
#     * Uses normalization_settings for percentile-based normalization
#
# To control verbose output, edit VERBOSE variable in Configuration section:
#   VERBOSE=""              # Default: verbose disabled
#   VERBOSE="--verbose"    # Explicitly enable verbose output
#
# Output files will be automatically named with mode suffix:
#   - *_stage1+stage2.nii.gz

# ============================================================================
# Configuration
# ============================================================================
IMAGE_PATH="./inputs/CT_chest_and_abdomen_large_depth_row_8_0000.nii.gz" #Totalsegmentator_s0059_0000.nii.gz"
OUTPUT_DIR="./outputs"
DEVICE="cuda:1"
CHECKPOINTS_PATH="./checkpoints"
CONFIG_FILE="./config_CT.json"  # Use config_CT.json for CT, config_nonCT.json for non-CT
VERBOSE="--verbose" # Set to "--verbose" to explicitly enable verbose output, empty for default (disabled)
# ============================================================================
# Setup
# ============================================================================

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please check the CONFIG_FILE path in the script configuration section."
    exit 1
fi

# Get output filename (without extension)
OUTPUT_FILENAME=$(basename "$IMAGE_PATH")
OUTPUT_BASE_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"

# Print configuration
echo "=========================================="
echo "Medal-S Inference - Stage 1 + Stage 2"
echo "=========================================="
echo "Input: $IMAGE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Device: $DEVICE"
echo "Checkpoints: $CHECKPOINTS_PATH"
if [ -n "$VERBOSE" ]; then
    echo "Verbose: $VERBOSE"
else
    echo "Verbose: default (disabled)"
fi
echo "=========================================="
echo ""
echo "Note:"
echo "  - CT images with multiple window types will be processed separately"
echo "  - Each window type uses its corresponding window settings"
echo "  - Results from all window types will be merged automatically"
echo ""

# ============================================================================
# Stage 1 + Stage 2 Inference
# ============================================================================
echo "=========================================="
echo "Stage 1 + Stage 2 Inference"
echo "=========================================="
echo "Running Stage 1 + Stage 2 inference..."
echo ""

python inference_medals_nifti.py \
    --input "$IMAGE_PATH" \
    --output "$OUTPUT_BASE_PATH" \
    --config "$CONFIG_FILE" \
    --mode stage1+stage2 \
    --device "$DEVICE" \
    --checkpoints "$CHECKPOINTS_PATH" \
    $VERBOSE

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Stage 1 + Stage 2 inference completed successfully!"
    echo ""
else
    echo ""
    echo "✗ Error: Stage 1 + Stage 2 inference failed!"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "Inference completed successfully!"
echo "=========================================="
echo ""
echo "Output file:"
# Handle .nii.gz extension properly
if [[ "$OUTPUT_FILENAME" == *.nii.gz ]]; then
    BASE_NAME="${OUTPUT_FILENAME%.nii.gz}"
    echo "  - $OUTPUT_DIR/${BASE_NAME}_stage1+stage2.nii.gz"
elif [[ "$OUTPUT_FILENAME" == *.nii ]]; then
    BASE_NAME="${OUTPUT_FILENAME%.nii}"
    echo "  - $OUTPUT_DIR/${BASE_NAME}_stage1+stage2.nii"
else
    echo "  - $OUTPUT_DIR/${OUTPUT_FILENAME}_stage1+stage2"
fi
echo ""

