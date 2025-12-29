#!/bin/bash
#SBATCH --job-name=comm_bench
#SBATCH --partition=ai
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --output=comm_bench_%j.out
#SBATCH --error=comm_bench_%j.err

################################################################################
# Communication Microbenchmarks on VALAR
# 
# This script runs NCCL and P2P bandwidth tests to verify multi-GPU
# communication before touching CKKS internals.
#
# Usage: sbatch comm_bench.slurm
################################################################################

echo "=========================================="
echo " VALAR Communication Benchmarks"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Load modules
module purge
module load gnu12/12.5.0
module load cmake/3.29.2
module load cuda/12.9.1
module load nccl/2.28.9-cuda12.9

# Verify environment
echo "=== Environment ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "NCCL_ROOT: $NCCL_ROOT"
nvcc --version | head -4
echo ""

# Check GPU availability
echo "=== GPU Availability ==="
nvidia-smi -L
echo ""

echo "=== GPU Topology ==="
nvidia-smi topo -m
echo ""

# Navigate to project
cd /scratch/hkanpak21/fides_matmul_step1/FIDESlib

# Build comm tests if not already built
if [ ! -d "build/comm" ]; then
    echo "=== Building Communication Tests ==="
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_COMPILER=$(which g++) \
             -DCMAKE_CUDA_COMPILER=$(which nvcc)
    make -j8 nccl_allreduce p2p_bw
    cd ..
fi

# Run NCCL All-Reduce Test
echo ""
echo "=========================================="
echo " NCCL All-Reduce Test"
echo "=========================================="
if [ -f "build/comm/nccl_allreduce" ]; then
    ./build/comm/nccl_allreduce 8
else
    echo "ERROR: nccl_allreduce binary not found"
fi

# Run P2P Bandwidth Test
echo ""
echo "=========================================="
echo " P2P Bandwidth Test"
echo "=========================================="
if [ -f "build/comm/p2p_bw" ]; then
    ./build/comm/p2p_bw
else
    echo "ERROR: p2p_bw binary not found"
fi

echo ""
echo "=========================================="
echo " Benchmark Complete"
echo "=========================================="
echo "End time: $(date)"
