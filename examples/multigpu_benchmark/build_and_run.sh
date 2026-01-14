#!/bin/bash
#
# FIDESlib Multi-GPU Benchmark Build & Run Script
# ================================================
# This script builds all benchmark binaries and optionally runs them.
#
# Usage:
#   ./build_and_run.sh          # Build only
#   ./build_and_run.sh --run    # Build and run all benchmarks
#   ./build_and_run.sh --help   # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default parameters for benchmarks
LOGN=14
L=12
OPS=32
ITERS=5
GPUS=4

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     FIDESlib Multi-GPU Benchmark Build & Run Script           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only     Build all binaries (default)"
    echo "  --run            Build and run all benchmarks"
    echo "  --run-simple     Build and run only the simple benchmark"
    echo "  --run-profiled   Build and run the profiled benchmark"
    echo "  --clean          Clean build directory before building"
    echo "  --help           Show this help message"
    echo ""
    echo "Benchmark Parameters (environment variables):"
    echo "  LOGN=$LOGN        Ring dimension (2^LOGN)"
    echo "  L=$L           Multiplicative depth"
    echo "  OPS=$OPS          Number of HE operations"
    echo "  ITERS=$ITERS          Number of iterations"
    echo "  GPUS=$GPUS           Number of GPUs to use"
    echo ""
    echo "Example:"
    echo "  LOGN=15 L=14 OPS=64 GPUS=4 ./build_and_run.sh --run"
}

check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: nvcc not found. Please install CUDA Toolkit.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} CUDA: $(nvcc --version | grep release | awk '{print $6}')"
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}Error: cmake not found. Please install CMake.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} CMake: $(cmake --version | head -1 | awk '{print $3}')"
    
    # Check FIDESlib installation
    if [ ! -f "/usr/local/lib/libFIDESlib.so" ] && [ ! -f "/usr/local/lib/libFIDESlib.a" ]; then
        echo -e "${YELLOW}  ⚠ FIDESlib not found in /usr/local/lib. You may need to install it first.${NC}"
        echo -e "    Run: cd /root/FIDESlib/build && make install"
    else
        echo -e "  ${GREEN}✓${NC} FIDESlib installed"
    fi
    
    # Check GPUs
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo -e "  ${GREEN}✓${NC} GPUs available: $GPU_COUNT"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
            echo -e "      GPU $line"
        done
    else
        echo -e "${RED}Error: nvidia-smi not found. Cannot detect GPUs.${NC}"
        exit 1
    fi
    
    echo ""
}

build_benchmarks() {
    echo -e "${YELLOW}Building benchmark binaries...${NC}"
    
    # Create build directory if needed
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    # Run CMake if needed
    if [ ! -f "Makefile" ]; then
        echo -e "  Running CMake..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi
    
    # Build all targets
    echo -e "  Building targets..."
    
    TARGETS=(
        "multigpu_simple_bench"
        "multigpu_profiled_bench"
        "multigpu_gpu_only_bench"
        "test_dual_context"
    )
    
    for target in "${TARGETS[@]}"; do
        echo -ne "    Building ${target}... "
        if make "$target" -j$(nproc) > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠ (may not exist)${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "Available binaries in ${BUILD_DIR}:"
    ls -la "$BUILD_DIR"/*.out "$BUILD_DIR"/multigpu_* "$BUILD_DIR"/test_* 2>/dev/null | awk '{print "  " $NF}' || true
    echo ""
}

clean_build() {
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}  ✓ Build directory cleaned${NC}"
    else
        echo -e "  Build directory does not exist"
    fi
    echo ""
}

run_simple_benchmark() {
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running Simple Benchmark (Pure GPU Compute Timing)"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    if [ ! -f "$BUILD_DIR/multigpu_simple_bench" ]; then
        echo -e "${RED}Error: multigpu_simple_bench not found. Build first.${NC}"
        return 1
    fi
    
    echo "Parameters: logN=$LOGN, L=$L, ops=$OPS, iters=$ITERS, gpus=$GPUS"
    echo ""
    
    "$BUILD_DIR/multigpu_simple_bench" \
        --logN "$LOGN" \
        --L "$L" \
        --ops "$OPS" \
        --iters "$ITERS" \
        --gpus "$GPUS"
}

run_profiled_benchmark() {
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running Profiled Benchmark (Full Phase Breakdown)"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    if [ ! -f "$BUILD_DIR/multigpu_profiled_bench" ]; then
        echo -e "${RED}Error: multigpu_profiled_bench not found. Build first.${NC}"
        return 1
    fi
    
    echo "Parameters: logN=$LOGN, L=$L, ops=$OPS, iters=$ITERS, gpus=$GPUS"
    echo ""
    
    "$BUILD_DIR/multigpu_profiled_bench" \
        --logN "$LOGN" \
        --L "$L" \
        --ops "$OPS" \
        --iters "$ITERS" \
        --gpus "$GPUS"
}

run_gpu_only_benchmark() {
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running GPU-Only Benchmark (Isolated GPU Timing)"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    if [ ! -f "$BUILD_DIR/multigpu_gpu_only_bench" ]; then
        echo -e "${RED}Error: multigpu_gpu_only_bench not found. Build first.${NC}"
        return 1
    fi
    
    echo "Parameters: logN=$LOGN, L=$L, ops=$OPS, iters=$ITERS, gpus=$GPUS"
    echo ""
    
    "$BUILD_DIR/multigpu_gpu_only_bench" \
        --logN "$LOGN" \
        --L "$L" \
        --ops "$OPS" \
        --iters "$ITERS" \
        --gpus "$GPUS"
}

run_all_benchmarks() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              Running All Benchmarks                           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    run_simple_benchmark
    echo ""
    echo ""
    
    # Optionally run profiled benchmark (takes longer due to CPU encryption)
    if [ "$RUN_PROFILED" = "true" ]; then
        run_profiled_benchmark
        echo ""
        echo ""
    fi
}

# Main script
print_header

# Parse arguments
DO_BUILD=true
DO_RUN=false
DO_CLEAN=false
RUN_SIMPLE=false
RUN_PROFILED=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            DO_BUILD=true
            DO_RUN=false
            shift
            ;;
        --run)
            DO_BUILD=true
            DO_RUN=true
            RUN_ALL=true
            shift
            ;;
        --run-simple)
            DO_BUILD=true
            DO_RUN=true
            RUN_SIMPLE=true
            shift
            ;;
        --run-profiled)
            DO_BUILD=true
            DO_RUN=true
            RUN_PROFILED=true
            shift
            ;;
        --clean)
            DO_CLEAN=true
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

# Override defaults from environment
LOGN=${LOGN:-14}
L=${L:-12}
OPS=${OPS:-32}
ITERS=${ITERS:-5}
GPUS=${GPUS:-4}

# Execute
check_prerequisites

if [ "$DO_CLEAN" = true ]; then
    clean_build
fi

if [ "$DO_BUILD" = true ]; then
    build_benchmarks
fi

if [ "$DO_RUN" = true ]; then
    if [ "$RUN_ALL" = true ]; then
        run_all_benchmarks
    fi
    if [ "$RUN_SIMPLE" = true ]; then
        run_simple_benchmark
    fi
    if [ "$RUN_PROFILED" = true ]; then
        run_profiled_benchmark
    fi
fi

echo -e "${GREEN}Done!${NC}"
