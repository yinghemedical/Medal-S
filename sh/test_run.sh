#!/bin/bash

# Define paths
SOURCE_INPUTS="/media/shipc/hhd_8T/spc/code/CVPR2025_Text_guided_seg_submission/inputs"
SOURCE_GTS="/media/shipc/hhd_8T/spc/code/CVPR2025_Text_guided_seg_submission/gts"
WORK_INPUTS="/media/shipc/hhd_8T/spc/code/CVPR2025_Text_guided_seg_submission/workspace_teamx/inputs"
WORK_GTS="/media/shipc/hhd_8T/spc/code/CVPR2025_Text_guided_seg_submission/workspace_teamx/gts"
TEAM_NAME="teamx"

# Ensure working directories exist
mkdir -p "$WORK_INPUTS"
mkdir -p "$WORK_GTS"

# Get list of input files
INPUT_FILES=("$SOURCE_INPUTS"/*)
TOTAL_FILES=${#INPUT_FILES[@]}
PROCESSED=0

echo "Starting processing of $TOTAL_FILES files..."

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    # Extract filename
    FILENAME=$(basename "$INPUT_FILE")
    
    echo "========================================"
    echo "Processing file $((PROCESSED+1))/$TOTAL_FILES: $FILENAME"
    
    # Clean working directories
    echo "Cleaning working directories..."
    rm -f "$WORK_INPUTS"/*
    rm -f "$WORK_GTS"/*
    
    # Copy current input file and corresponding GT file
    echo "Copying files to working directories..."
    cp "$INPUT_FILE" "$WORK_INPUTS/"
    GT_FILE="$SOURCE_GTS/$FILENAME"
    if [ -f "$GT_FILE" ]; then
        cp "$GT_FILE" "$WORK_GTS/"
    else
        echo "Warning: Corresponding GT file $GT_FILE not found"
    fi
    
    # Run Docker container for inference with time measurement
    echo "Starting Docker container for inference..."
    START_TIME=$(date +%s.%N)
    
    docker container run \
        --gpus "device=0" \
        -m 32G \
        --name ${TEAM_NAME} \
        --rm \
        -v "$WORK_INPUTS":/workspace/inputs/ \
        -v "$PWD/outputs/":/workspace/outputs/ \
        ${TEAM_NAME}:latest \
        /bin/bash -c "sh predict.sh"
    
    # Calculate inference time
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Check if inference succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for file $FILENAME (Time: ${ELAPSED}s)"
        exit 1
    fi
    
    echo "Inference completed in ${ELAPSED} seconds"
    
    # Run evaluation script (not timed)
    echo "Running evaluation script..."
    python evaluate_results.py
    
    # Clean working directories for next file
    echo "Cleaning working directories..."
    rm -f "$WORK_INPUTS"/*
    rm -f "$WORK_GTS"/*
    
    PROCESSED=$((PROCESSED+1))
    echo "Completed $PROCESSED/$TOTAL_FILES files"
    echo "========================================"
    echo ""
done

echo "All files processed successfully!"
