#!/bin/bash

# Depth Anything 3 + Gaussian Splatting Installation Script
# Run with: bash install.sh

set -e  # Exit on any error

echo "🚀 Depth Anything 3 + Gaussian Splatting Installation Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found! Please install Miniconda or Anaconda first."
    print_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA availability
print_info "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "NVIDIA drivers detected"
else
    print_warning "nvidia-smi not found. GPU acceleration may not work."
fi

# Environment name
ENV_NAME="depth_anything_3"
PROJECT_DIR="$HOME/Desktop/project/depth_anything"

# Create project directory
print_info "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    print_warning "Environment '$ENV_NAME' already exists!"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_info "Using existing environment"
        conda activate "$ENV_NAME"
    fi
else
    # Create new conda environment
    print_info "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate environment
print_info "Activating environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Clone repository if not exists
if [ ! -d "Depth-Anything-3" ]; then
    print_info "Cloning Depth Anything 3 repository..."
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
else
    print_status "Repository already exists"
fi

cd Depth-Anything-3

# Install PyTorch with CUDA support
print_info "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
print_info "Installing requirements..."
pip install -r requirements.txt

# Install Depth Anything 3
print_info "Installing Depth Anything 3..."
pip install -e .

# Install Gaussian Splatting dependencies
print_info "Installing Gaussian Splatting support..."
pip install ninja pybind11
pip install gsplat

# Verify installation
print_info "Verifying installation..."
python -c "
import torch
from depth_anything_3.api import DepthAnything3
print('✅ Depth Anything 3: OK')
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))

try:
    import gsplat
    print('✅ GSplat version:', gsplat.__version__)
except ImportError:
    print('❌ GSplat not available')
"

# Return to project root
cd "$PROJECT_DIR"

print_status "Installation completed successfully! 🎉"
echo
print_info "Next steps:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Run test: python test.py"
echo "  3. Try Gaussian Splatting: python gaussian_splatting_test.py"
echo "  4. Launch viewer: python gaussian_splat_viewer.py"
echo
print_info "Project location: $PROJECT_DIR"
print_info "See README.md for detailed usage instructions"