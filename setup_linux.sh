#!/bin/bash

# MuseTalk Linux Setup Script
# This script sets up the complete environment for MuseTalk on Linux

set -e  # Exit on any error

echo "==============================================="
echo "    MuseTalk Linux Setup Script"
echo "==============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if conda is installed
if ! command_exists conda; then
    print_error "Conda is not installed. Please install Miniconda or Anaconda first."
    print_error "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found. Proceeding with environment setup..."

# Create conda environment
ENV_NAME="MuseTalk"
print_status "Creating conda environment: $ENV_NAME with Python 3.10..."

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=3.10 -y
print_status "Conda environment '$ENV_NAME' created successfully."

# Activate environment
print_status "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Verify Python version
python_version=$(python --version 2>&1)
print_status "Using Python version: $python_version"

# Install PyTorch 2.0.1 with CUDA 11.8 support
print_status "Installing PyTorch 2.0.1 with CUDA 11.8 support..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install main dependencies
print_status "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install MMLab packages
print_status "Installing MMLab ecosystem packages..."
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Check for FFmpeg
print_status "Checking FFmpeg installation..."
if ! command_exists ffmpeg; then
    print_warning "FFmpeg not found. Installing FFmpeg..."
    
    # Try different package managers
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command_exists yum; then
        sudo yum install -y ffmpeg
    elif command_exists dnf; then
        sudo dnf install -y ffmpeg
    elif command_exists pacman; then
        sudo pacman -S ffmpeg
    elif command_exists zypper; then
        sudo zypper install ffmpeg
    else
        print_warning "Could not install FFmpeg automatically. Please install it manually:"
        print_warning "Ubuntu/Debian: sudo apt-get install ffmpeg"
        print_warning "CentOS/RHEL: sudo yum install ffmpeg"
        print_warning "Or download static builds from: https://github.com/BtbN/FFmpeg-Builds/releases"
    fi
else
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n 1)
    print_status "FFmpeg found: $ffmpeg_version"
fi

# Set up FFmpeg path environment variable
print_status "Setting up FFmpeg environment variable..."
FFMPEG_PATH=$(which ffmpeg)
if [ -n "$FFMPEG_PATH" ]; then
    print_status "FFmpeg path: $FFMPEG_PATH"
    echo "export FFMPEG_PATH=$FFMPEG_PATH" >> ~/.bashrc
    export FFMPEG_PATH=$FFMPEG_PATH
else
    print_warning "FFmpeg not found in PATH. You may need to set FFMPEG_PATH manually."
fi

# Download model weights
print_status "Downloading model weights..."
if [ -f "download_weights.sh" ]; then
    chmod +x download_weights.sh
    bash download_weights.sh
else
    print_error "download_weights.sh not found. Please download model weights manually."
    print_error "See README.md for manual download instructions."
fi

# Verify installation
print_status "Verifying installation..."

# Check if key directories exist
if [ -d "models/musetalk" ] && [ -d "models/musetalkV15" ]; then
    print_status "Model directories found."
else
    print_warning "Model directories not found. Weights may not have downloaded correctly."
fi

# Test basic imports
print_status "Testing basic Python imports..."
python -c "
import torch
import torchvision
import cv2
import numpy as np
import diffusers
import transformers
print('All basic imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"

print_status "Creating activation script..."
cat > activate_musetalk.sh << 'EOF'
#!/bin/bash
# MuseTalk Environment Activation Script

source $(conda info --base)/etc/profile.d/conda.sh
conda activate MuseTalk

# Set FFmpeg path if not already set
if [ -z "$FFMPEG_PATH" ]; then
    export FFMPEG_PATH=$(which ffmpeg)
fi

echo "MuseTalk environment activated!"
echo "Python: $(which python)"
echo "FFmpeg: $FFMPEG_PATH"
echo ""
echo "Usage examples:"
echo "  Normal inference:     python -m scripts.inference --inference_config configs/inference/test.yaml"
echo "  Real-time inference:  python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml"
echo "  Gradio demo:          python app.py --use_float16"
echo ""
EOF

chmod +x activate_musetalk.sh

print_status "Creating inference helper script..."
cat > run_inference.sh << 'EOF'
#!/bin/bash
# MuseTalk Inference Helper Script

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MuseTalk

# Set FFmpeg path
export FFMPEG_PATH=$(which ffmpeg)

# Default parameters
VERSION="v1.5"
MODE="normal"
CONFIG="configs/inference/test.yaml"
RESULT_DIR="results"
USE_FLOAT16=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        v1.0|v1.5)
            VERSION="$1"
            shift
            ;;
        normal|realtime)
            MODE="$1"
            shift
            ;;
        --use_float16)
            USE_FLOAT16="--use_float16"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [v1.0|v1.5] [normal|realtime] [--use_float16]"
            exit 1
            ;;
    esac
done

echo "Running MuseTalk $VERSION in $MODE mode..."

if [ "$VERSION" = "v1.5" ]; then
    MODEL_PATH="models/musetalkV15/unet.pth"
    CONFIG_PATH="models/musetalkV15/musetalk.json"
    VERSION_FLAG="v15"
else
    MODEL_PATH="models/musetalk/pytorch_model.bin"
    CONFIG_PATH="models/musetalk/musetalk.json"
    VERSION_FLAG="v1"
fi

if [ "$MODE" = "realtime" ]; then
    CONFIG="configs/inference/realtime.yaml"
    RESULT_DIR="results/realtime"
    python -m scripts.realtime_inference \
        --inference_config $CONFIG \
        --result_dir $RESULT_DIR \
        --unet_model_path $MODEL_PATH \
        --unet_config $CONFIG_PATH \
        --version $VERSION_FLAG \
        --fps 25 \
        $USE_FLOAT16
else
    CONFIG="configs/inference/test.yaml"
    RESULT_DIR="results/test"
    python -m scripts.inference \
        --inference_config $CONFIG \
        --result_dir $RESULT_DIR \
        --unet_model_path $MODEL_PATH \
        --unet_config $CONFIG_PATH \
        --version $VERSION_FLAG \
        $USE_FLOAT16
fi
EOF

chmod +x run_inference.sh

print_status "Creating Gradio demo launcher..."
cat > run_gradio.sh << 'EOF'
#!/bin/bash
# MuseTalk Gradio Demo Launcher

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MuseTalk

# Set FFmpeg path
export FFMPEG_PATH=$(which ffmpeg)

# Run Gradio demo with float16 for better performance
python app.py --use_float16
EOF

chmod +x run_gradio.sh

echo ""
echo "==============================================="
echo "    MuseTalk Setup Complete!"
echo "==============================================="
echo ""
print_status "Setup completed successfully!"
echo ""
echo "Quick Start:"
echo "  1. Activate environment:     source activate_musetalk.sh"
echo "  2. Run inference:            ./run_inference.sh v1.5 normal"
echo "  3. Run real-time inference:  ./run_inference.sh v1.5 realtime"
echo "  4. Launch Gradio demo:       ./run_gradio.sh"
echo ""
echo "Manual activation:"
echo "  conda activate MuseTalk"
echo "  export FFMPEG_PATH=\$(which ffmpeg)"
echo ""
print_status "Environment: $ENV_NAME"
print_status "Python: $(python --version)"
print_status "FFmpeg: $FFMPEG_PATH"
echo ""
echo "For more details, check the README.md file."
echo "==============================================="
