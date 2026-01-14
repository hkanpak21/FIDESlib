#!/bin/bash
# =============================================================================
# FIDESlib Multi-GPU Bootstrap Script
# =============================================================================
# This script rebuilds the modified FIDESlib with multi-GPU support from scratch
# Run this after a fresh clone or server restart
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "FIDESlib Multi-GPU Bootstrap"
echo "=========================================="

# Configuration
FIDES_ROOT="/FIDES"
FIDESLIB_DIR="${FIDES_ROOT}/FIDESlib"
EXAMPLES_DIR="${FIDESLIB_DIR}/examples/he_matmul"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================
log_info "Checking prerequisites..."

if ! command -v nvcc &> /dev/null; then
    log_error "CUDA not found. Please install CUDA toolkit."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    log_error "CMake not found. Please install CMake."
    exit 1
fi

# Check for GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 1 ]; then
    log_error "No GPUs detected."
    exit 1
fi
log_info "Found ${GPU_COUNT} GPU(s)"

# =============================================================================
# Step 2: Build FIDESlib
# =============================================================================
log_info "Building FIDESlib..."

cd "${FIDESLIB_DIR}"

# Create build directory
mkdir -p build
cd build

# Configure with correct CUDA architecture
# L4 GPUs use sm_89
CUDA_ARCH="89"
log_info "Configuring for CUDA architecture sm_${CUDA_ARCH}"

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release

# Build
log_info "Compiling FIDESlib (this may take 5-10 minutes)..."
make -j$(nproc)

# Install
log_info "Installing FIDESlib..."
make install

log_info "FIDESlib build complete!"

# =============================================================================
# Step 3: Build Examples
# =============================================================================
log_info "Building example programs..."

cd "${EXAMPLES_DIR}"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

log_info "Examples build complete!"

# =============================================================================
# Step 4: Run Tests
# =============================================================================
log_info "Running dual context test..."

if ./test_dual_context; then
    log_info "Dual context test PASSED!"
else
    log_error "Dual context test FAILED!"
    exit 1
fi

# =============================================================================
# Step 5: Run Benchmark (optional)
# =============================================================================
echo ""
read -p "Run full multi-GPU benchmark? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Running multi-GPU benchmark..."
    ./he_matmul_multigpu
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================="
echo "Bootstrap Complete!"
echo "=========================================="
echo ""
echo "Available executables:"
echo "  ${EXAMPLES_DIR}/build/test_dual_context"
echo "  ${EXAMPLES_DIR}/build/he_matmul_multigpu"
echo ""
echo "To re-run tests:"
echo "  cd ${EXAMPLES_DIR}/build"
echo "  ./test_dual_context"
echo "  ./he_matmul_multigpu"
echo ""
