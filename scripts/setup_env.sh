#!/bin/bash
################################################################################
# VALAR Environment Setup Script
#
# This script sets up a consistent, aligned toolchain for FIDESlib development.
# Run with: source setup_env.sh
#
# Aligned Toolchain:
#   CUDA 12.9.1 + NCCL 2.28.9-cuda12.9 + GCC 12.5.0 + CMake 3.29.2
#
# @author hkanpak21
# @date 2025-12-29
################################################################################

echo "=== Setting up FIDES Development Environment ==="

# Purge existing modules to start fresh
module purge

# Load aligned toolchain
module load gnu12/12.5.0
module load cmake/3.29.2
module load cuda/12.9.1
module load nccl/2.28.9-cuda12.9
module load git/2.9.5

# Verify environment
echo ""
echo "=== Environment Verification ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "NCCL_ROOT: $NCCL_ROOT"
echo ""
echo "GCC: $(g++ --version | head -n 1)"
echo "NVCC: $(nvcc --version | grep release)"
echo "CMake: $(cmake --version | head -n 1)"
echo "Git: $(git --version)"
echo ""

# Set project paths
export FIDES_ROOT=/scratch/hkanpak21/fides_matmul_step1/FIDESlib
export FIDES_BUILD=$FIDES_ROOT/build
export OpenFHE_DIR=/scratch/hkanpak21/fides_matmul_step1/local/lib/OpenFHE

echo "=== FIDES Paths ==="
echo "FIDES_ROOT: $FIDES_ROOT"
echo "FIDES_BUILD: $FIDES_BUILD"
echo "OpenFHE_DIR: $OpenFHE_DIR"
echo ""

echo "=== Environment Ready ==="
echo ""
echo "Quick commands:"
echo "  Build:     cd \$FIDES_ROOT && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=\$OpenFHE_DIR && make -j8"
echo "  Rebuild:   cd \$FIDES_BUILD && make -j8"
echo "  Test:      cd \$FIDES_BUILD && ctest --output-on-failure"
echo ""
