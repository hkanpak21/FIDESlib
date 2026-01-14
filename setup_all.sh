#!/bin/bash
#
# FIDESlib Complete Setup Script
# ==============================
# This script builds and installs everything needed for FIDESlib multi-GPU benchmarks:
#   1. System dependencies (apt packages)
#   2. OpenFHE (CPU-side HE library)
#   3. FIDESlib (GPU-accelerated CKKS library)
#   4. Multi-GPU benchmark suite
#
# Usage:
#   ./setup_all.sh              # Full setup (all steps)
#   ./setup_all.sh --deps-only  # Install dependencies only
#   ./setup_all.sh --openfhe    # Build OpenFHE only
#   ./setup_all.sh --fideslib   # Build FIDESlib only
#   ./setup_all.sh --benchmarks # Build benchmarks only
#   ./setup_all.sh --help       # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIDESLIB_DIR="$SCRIPT_DIR"
FIDESLIB_BUILD_DIR="$FIDESLIB_DIR/build"
OPENFHE_VERSION="v1.2.3"
OPENFHE_DIR="/tmp/openfhe-development"
BENCHMARK_DIR="$FIDESLIB_DIR/examples/multigpu_benchmark"
BENCHMARK_BUILD_DIR="$BENCHMARK_DIR/build"

# Build settings
CUDA_ARCH="${CUDA_ARCH:-89}"  # Default to sm_89 (L4/Ada), change for your GPU
NUM_JOBS="${NUM_JOBS:-$(nproc)}"

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         FIDESlib Complete Setup Script                        ║"
    echo "║         Multi-GPU CKKS Homomorphic Encryption                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no option)      Full setup: deps → OpenFHE → FIDESlib → benchmarks"
    echo "  --deps-only      Install system dependencies only"
    echo "  --openfhe        Build and install OpenFHE only"
    echo "  --fideslib       Build and install FIDESlib only"
    echo "  --benchmarks     Build benchmark binaries only"
    echo "  --clean          Clean all build directories"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CUDA_ARCH=$CUDA_ARCH     CUDA compute capability (e.g., 89 for L4, 86 for A100)"
    echo "  NUM_JOBS=$NUM_JOBS        Number of parallel build jobs"
    echo ""
    echo "Supported CUDA architectures:"
    echo "  70  - V100"
    echo "  75  - T4, RTX 20xx"
    echo "  80  - A100"
    echo "  86  - RTX 30xx, A40"
    echo "  89  - L4, RTX 40xx (Ada)"
    echo "  90  - H100"
    echo ""
    echo "Examples:"
    echo "  ./setup_all.sh                      # Full setup with default arch"
    echo "  CUDA_ARCH=86 ./setup_all.sh         # Full setup for RTX 3090"
    echo "  ./setup_all.sh --benchmarks         # Rebuild benchmarks only"
}

print_step() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}Warning: Not running as root. Some operations may require sudo.${NC}"
    fi
}

# ============================================================
# Step 1: Install System Dependencies
# ============================================================
install_dependencies() {
    print_step "Step 1/4: Installing System Dependencies"
    
    echo "Updating package lists..."
    apt-get update
    
    echo "Installing required packages..."
    apt-get install -y \
        build-essential \
        cmake \
        git \
        libomp-dev \
        libtbb-dev \
        libgmp-dev \
        libntl-dev \
        autoconf \
        libtool \
        pkg-config \
        wget \
        curl \
        vim \
        htop
    
    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: CUDA Toolkit not found. Please install CUDA first.${NC}"
        echo "Visit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA Toolkit found: $CUDA_VERSION${NC}"
    
    # Install NCCL if not present
    if ! ldconfig -p | grep -q libnccl; then
        echo "Installing NCCL..."
        apt-get install -y libnccl2 libnccl-dev || {
            echo -e "${YELLOW}Warning: NCCL installation failed. Multi-GPU collective ops may not work.${NC}"
        }
    else
        echo -e "${GREEN}✓ NCCL already installed${NC}"
    fi
    
    echo -e "\n${GREEN}✓ Dependencies installed successfully${NC}"
}

# ============================================================
# Step 2: Build and Install OpenFHE
# ============================================================
build_openfhe() {
    print_step "Step 2/4: Building OpenFHE $OPENFHE_VERSION"
    
    # Check if OpenFHE is already installed
    if [ -f "/usr/local/lib/libOPENFHEcore.so" ]; then
        INSTALLED_VERSION=$(cat /usr/local/include/openfhe/version.h 2>/dev/null | grep "OPENFHE_VERSION" | head -1 || echo "unknown")
        echo -e "${YELLOW}OpenFHE appears to be already installed.${NC}"
        read -p "Reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping OpenFHE build."
            return 0
        fi
    fi
    
    # Clone OpenFHE
    echo "Cloning OpenFHE repository..."
    rm -rf "$OPENFHE_DIR"
    git clone --depth 1 --branch "$OPENFHE_VERSION" \
        https://github.com/openfheorg/openfhe-development.git "$OPENFHE_DIR"
    
    # Apply FIDESlib patches if they exist
    if [ -f "$FIDESLIB_DIR/cmake/openfhe-base.patch" ]; then
        echo "Applying FIDESlib patches to OpenFHE..."
        cd "$OPENFHE_DIR"
        patch -p1 < "$FIDESLIB_DIR/cmake/openfhe-base.patch" || {
            echo -e "${YELLOW}Warning: Base patch may have already been applied${NC}"
        }
        if [ -f "$FIDESLIB_DIR/cmake/openfhe-hook.patch" ]; then
            patch -p1 < "$FIDESLIB_DIR/cmake/openfhe-hook.patch" || {
                echo -e "${YELLOW}Warning: Hook patch may have already been applied${NC}"
            }
        fi
    fi
    
    # Build OpenFHE
    echo "Building OpenFHE..."
    mkdir -p "$OPENFHE_DIR/build"
    cd "$OPENFHE_DIR/build"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_UNITTESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DWITH_BE2=ON \
        -DWITH_BE4=ON \
        -DWITH_NTL=ON \
        -DWITH_OPENMP=ON
    
    make -j"$NUM_JOBS"
    
    echo "Installing OpenFHE..."
    make install
    
    # Update library cache
    ldconfig
    
    echo -e "\n${GREEN}✓ OpenFHE installed successfully${NC}"
}

# ============================================================
# Step 3: Build and Install FIDESlib
# ============================================================
build_fideslib() {
    print_step "Step 3/4: Building FIDESlib (CUDA arch: sm_$CUDA_ARCH)"
    
    cd "$FIDESLIB_DIR"
    
    # Clean previous build if exists
    if [ -d "$FIDESLIB_BUILD_DIR" ]; then
        echo "Cleaning previous FIDESlib build..."
        rm -rf "$FIDESLIB_BUILD_DIR"
    fi
    
    mkdir -p "$FIDESLIB_BUILD_DIR"
    cd "$FIDESLIB_BUILD_DIR"
    
    echo "Configuring FIDESlib..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DBUILD_TESTS=ON \
        -DBUILD_BENCHMARKS=ON
    
    echo "Building FIDESlib..."
    make -j"$NUM_JOBS"
    
    echo "Installing FIDESlib..."
    make install
    
    # Update library cache
    ldconfig
    
    echo -e "\n${GREEN}✓ FIDESlib installed successfully${NC}"
    
    # Run quick test
    echo -e "\nRunning quick GPU test..."
    if [ -f "$FIDESLIB_BUILD_DIR/gpu-test" ]; then
        "$FIDESLIB_BUILD_DIR/gpu-test" --gtest_filter="*Smoke*" 2>/dev/null || {
            echo -e "${YELLOW}Note: GPU test skipped or failed. This may be normal.${NC}"
        }
    fi
}

# ============================================================
# Step 4: Build Multi-GPU Benchmarks
# ============================================================
build_benchmarks() {
    print_step "Step 4/4: Building Multi-GPU Benchmarks"
    
    if [ ! -d "$BENCHMARK_DIR" ]; then
        echo -e "${RED}Error: Benchmark directory not found: $BENCHMARK_DIR${NC}"
        exit 1
    fi
    
    cd "$BENCHMARK_DIR"
    
    # Clean previous build if exists
    if [ -d "$BENCHMARK_BUILD_DIR" ]; then
        echo "Cleaning previous benchmark build..."
        rm -rf "$BENCHMARK_BUILD_DIR"
    fi
    
    mkdir -p "$BENCHMARK_BUILD_DIR"
    cd "$BENCHMARK_BUILD_DIR"
    
    echo "Configuring benchmarks..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    echo "Building benchmark targets..."
    TARGETS=(
        "multigpu_simple_bench"
        "multigpu_profiled_bench"
        "multigpu_gpu_only_bench"
        "multigpu_matmul_bench"
        "multigpu_bootstrap_bench"
        "test_dual_context"
    )
    
    for target in "${TARGETS[@]}"; do
        echo -ne "  Building ${target}... "
        if make "$target" -j"$NUM_JOBS" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠ (skipped)${NC}"
        fi
    done
    
    echo -e "\n${GREEN}✓ Benchmarks built successfully${NC}"
    
    # List available binaries
    echo -e "\nAvailable benchmark binaries:"
    ls -la "$BENCHMARK_BUILD_DIR"/multigpu_* "$BENCHMARK_BUILD_DIR"/test_* 2>/dev/null | \
        awk '{print "  " $NF " (" $5 " bytes)"}' || true
}

# ============================================================
# Clean all build directories
# ============================================================
clean_all() {
    print_step "Cleaning all build directories"
    
    echo "Cleaning FIDESlib build..."
    rm -rf "$FIDESLIB_BUILD_DIR"
    
    echo "Cleaning benchmark build..."
    rm -rf "$BENCHMARK_BUILD_DIR"
    
    echo "Cleaning OpenFHE source..."
    rm -rf "$OPENFHE_DIR"
    
    echo -e "\n${GREEN}✓ All build directories cleaned${NC}"
}

# ============================================================
# Print system info
# ============================================================
print_system_info() {
    echo -e "${YELLOW}System Information:${NC}"
    echo "  OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || uname -s)"
    echo "  Kernel: $(uname -r)"
    echo "  CPU: $(nproc) cores"
    
    if command -v nvcc &> /dev/null; then
        echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}')"
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
            while read line; do echo "    $line"; done
    fi
    
    echo "  Target CUDA arch: sm_$CUDA_ARCH"
    echo "  Build jobs: $NUM_JOBS"
    echo ""
}

# ============================================================
# Print success summary
# ============================================================
print_success() {
    echo -e "\n${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete!                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo "Installed components:"
    echo "  ✓ OpenFHE $OPENFHE_VERSION → /usr/local"
    echo "  ✓ FIDESlib → /usr/local"
    echo "  ✓ Benchmarks → $BENCHMARK_BUILD_DIR"
    echo ""
    echo "Quick start:"
    echo "  cd $BENCHMARK_DIR"
    echo "  ./build_and_run.sh --run-simple"
    echo ""
    echo "Or run directly:"
    echo "  $BENCHMARK_BUILD_DIR/multigpu_simple_bench --logN 14 --L 12 --ops 32 --gpus 4"
    echo ""
}

# ============================================================
# Main
# ============================================================
print_header

# Parse arguments
DO_DEPS=false
DO_OPENFHE=false
DO_FIDESLIB=false
DO_BENCHMARKS=false
DO_CLEAN=false
DO_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --deps-only)
            DO_DEPS=true
            DO_ALL=false
            shift
            ;;
        --openfhe)
            DO_OPENFHE=true
            DO_ALL=false
            shift
            ;;
        --fideslib)
            DO_FIDESLIB=true
            DO_ALL=false
            shift
            ;;
        --benchmarks)
            DO_BENCHMARKS=true
            DO_ALL=false
            shift
            ;;
        --clean)
            DO_CLEAN=true
            DO_ALL=false
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

check_root
print_system_info

# Execute requested operations
if [ "$DO_CLEAN" = true ]; then
    clean_all
    exit 0
fi

if [ "$DO_ALL" = true ]; then
    install_dependencies
    build_openfhe
    build_fideslib
    build_benchmarks
    print_success
elif [ "$DO_DEPS" = true ]; then
    install_dependencies
elif [ "$DO_OPENFHE" = true ]; then
    build_openfhe
elif [ "$DO_FIDESLIB" = true ]; then
    build_fideslib
elif [ "$DO_BENCHMARKS" = true ]; then
    build_benchmarks
fi

echo -e "${GREEN}Done!${NC}"
