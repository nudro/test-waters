#!/bin/bash

# YOLOv8 Vessel Detection Training Script
# Optimized for DGX Spark with 128GB VRAM
# Uses AMP, FP16, and PyTorch optimizations

set -e  # Exit on error

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    # Try to activate pytorch_cuda environment
    if conda env list | grep -q "pytorch_cuda"; then
        echo "Activating conda environment: pytorch_cuda"
        eval "$(conda shell.bash hook)"
        conda activate pytorch_cuda
    fi
fi

echo "=========================================="
echo "YOLOv8 Vessel Detection Training"
echo "=========================================="

# Configuration
MODEL="yolov8n"  # Using nano model for better memory efficiency
DATA_YAML="datasets/combined/data.yaml"  # Full dataset for training
EPOCHS=150
IMG_SIZE=512 # 640 is too large for 128GB VRAM
BATCH_SIZE=64  # Batch size for training
WORKERS=16  # Reduced for stability
DEVICE=""  # Auto-detect (will use CUDA if available, else CPU)
PROJECT="runs/detect"
NAME="vessel_detection_yolov8n"

# Check if data is prepared
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: Data YAML not found: $DATA_YAML"
    echo "Please run: python prepare_vessel_datasets.py"
    exit 1
fi

# Check if model exists
if [ ! -f "models/${MODEL}.pt" ]; then
    echo "Warning: models/${MODEL}.pt not found"
    echo "The script will download the pretrained model automatically"
fi

# Check for CUDA availability
CUDA_CHECK=$(python -c "import torch; print('CUDA_AVAILABLE' if torch.cuda.is_available() else 'CUDA_NOT_AVAILABLE')" 2>/dev/null || echo "CUDA_NOT_AVAILABLE")

if [ "$CUDA_CHECK" = "CUDA_AVAILABLE" ]; then
    echo "CUDA detected - using GPU"
    export CUDA_VISIBLE_DEVICES=0  # Use first GPU, adjust if needed
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management
    export CUDA_LAUNCH_BLOCKING=0  # Async CUDA operations
    
    # Set CUDA environment variables for GB10 (sm_121) support
    # Check for CUDA 12.8 first, fall back to CUDA 13.0 if available
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME=/usr/local/cuda-12.8
        echo "Using CUDA 12.8 from $CUDA_HOME"
    elif [ -d "/usr/local/cuda-13.0" ]; then
        export CUDA_HOME=/usr/local/cuda-13.0
        echo "Using CUDA 13.0 from $CUDA_HOME"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Using CUDA from $CUDA_HOME"
    fi
    
    # Add CUDA to PATH and LD_LIBRARY_PATH
    if [ -n "$CUDA_HOME" ]; then
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    fi
    
    # Set architecture to sm_12.1 (GB10) - target the actual GPU architecture
    export TORCH_CUDA_ARCH_LIST="12.1"
    
    # Disable JIT compilation - PyTorch doesn't fully support sm_12.1 yet
    # The Python patch will handle box_iou operations in eager mode
    export TORCH_COMPILE_DISABLE=1  # Disable torch.compile
    export PYTORCH_DISABLE_JIT=1    # Disable JIT compilation
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    
    # Additional CUDA workarounds for GB10 compatibility
    export CUDA_LAUNCH_BLOCKING=0  # Keep async for performance
    export TORCH_USE_CUDA_DSA=0    # Disable device-side assertions (can cause issues)
    
    DEVICE="cuda"
    # Note: BATCH_SIZE is set at top of script - don't override it here
    echo "Note: Using sm_12.1 architecture (GB10) - TORCH_CUDA_ARCH_LIST=12.1"
    echo "      JIT compilation disabled - Python patch handles box_iou in eager mode"
else
    echo "ERROR: CUDA not available"
    echo "Please ensure PyTorch with CUDA support is installed"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_YAML"
echo "  Epochs: $EPOCHS"
echo "  Image size: $IMG_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Workers: $WORKERS"
echo "  Device: $DEVICE"
echo "  Project: $PROJECT"
echo "  Name: $NAME"
echo ""
echo "Optimizations enabled:"
if [ "$DEVICE" = "cuda" ]; then
    echo "  - AMP (Automatic Mixed Precision)"
    echo "  - FP16 precision"
fi
echo "  - AdamW optimizer"
echo "  - Data augmentation"
echo ""

# Start GPU monitoring in background
MONITOR_LOG="logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
echo "Starting GPU monitoring..."
echo "  Log file: $MONITOR_LOG"
python monitor_gpu.py --log "$MONITOR_LOG" --interval 2 &
MONITOR_PID=$!
echo "  Monitor PID: $MONITOR_PID"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null
    echo "GPU monitor stopped. Log saved to: $MONITOR_LOG"
}
trap cleanup EXIT

# Run training
python train_yolov8_vessel.py \
    --model "$MODEL" \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --imgsz "$IMG_SIZE" \
    --batch "$BATCH_SIZE" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --patience 50 \
    --project "$PROJECT" \
    --name "$NAME"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Best model: $PROJECT/$NAME/weights/best.pt"
echo "Output model: models/${MODEL}_vessel.pt"
echo "=========================================="

