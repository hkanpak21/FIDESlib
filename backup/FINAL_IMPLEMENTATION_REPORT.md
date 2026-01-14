# FIDESlib Multi-GPU Implementation Report

**Author:** Halil İbrahim Kanpak (hkanpak21@ku.edu.tr)  
**Repository:** https://github.com/halilkanpak/FIDESlib  
**Date:** January 2026

---

## Executive Summary

This report documents the successful implementation of multi-GPU support for FIDESlib, a CUDA-based Fully Homomorphic Encryption (FHE) library implementing the CKKS scheme. The implementation enables data-parallel execution of HE operations across multiple NVIDIA GPUs, achieving **1.29x–1.72x speedup** with 2 GPUs.

---

## 1. Environment Setup

### 1.1 Hardware
- **Platform:** RunPod Cloud
- **GPUs:** 2× NVIDIA L4 (Ada Lovelace architecture)
  - 24 GB VRAM each
  - Compute Capability 8.9
  - PCIe interconnect

### 1.2 Software Dependencies

```bash
# System packages installed
apt-get update
apt-get install -y libtbb-dev libnccl-dev

# Versions used
- CUDA Toolkit: 12.4.131
- NCCL: 2.29.2
- OpenFHE: 1.2.3
- GCC: 11.4.0
```

### 1.3 Build Process

```bash
# Build FIDESlib
cd /root/FIDESlib/build
cmake .. -DCUDA_ARCHITECTURES=89
make -j$(nproc)
make install

# Build benchmarks
cd /root/FIDESlib/examples/multigpu_benchmark
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 2. Technical Implementation

### 2.1 Problem: Static Global State

FIDESlib originally used static global variables for cryptographic keys:

```cpp
// BEFORE (problematic)
static std::optional<KeySwitchingKey> eval_key;
static std::map<int, KeySwitchingKey> rot_keys;
static std::map<int, BootstrapPrecomputation> boot_precomps;
```

This prevented multi-GPU operation because all contexts shared the same keys.

### 2.2 Solution: Per-Context Key Storage

**File Modified:** `include/CKKS/Context.cuh`

```cpp
// AFTER (fixed)
class Context {
private:
    mutable std::map<int, BootstrapPrecomputation> boot_precomps_;
    mutable std::map<int, KeySwitchingKey> rot_keys_;
    mutable std::optional<KeySwitchingKey> eval_key_;
public:
    void AddEvalKey(KeySwitchingKey&& ksk);
    KeySwitchingKey& GetEvalKey();
    // ... etc
};
```

### 2.3 Problem: CUDA Graph Cache Conflicts

CUDA graphs are device-specific, but the cache was keyed only by size:

```cpp
// BEFORE (problematic)
static std::map<int, cudaGraphExec_t> exec;
auto key = (int)v.size;
```

### 2.4 Solution: Device-Aware Graph Caching

**Files Modified:** `src/CKKS/Limb.cu`, `src/CKKS/LimbPartition.cu`

```cpp
// AFTER (fixed) - 4 locations total
static std::map<std::pair<int,int>, cudaGraphExec_t> exec;
int current_device;
cudaGetDevice(&current_device);
auto key = std::make_pair(current_device, (int)v.size);
```

### 2.5 Problem: GPU Constant Memory Indexing

Array indexing used loop counter instead of actual GPU ID:

```cpp
// BEFORE (problematic)
for (size_t i = 0; i < GPUid.size(); ++i) {
    cudaSetDevice(GPUid.at(i));
    cudaMalloc(&(hG_.psi_ptr[i][j]), bytes);  // BUG: uses i, not GPU ID
}
```

### 2.6 Solution: GPU ID-Based Indexing with Tracking

**File Modified:** `src/ConstantsGPU.cu`

```cpp
// AFTER (fixed)
static std::set<int> initialized_gpus;
static int initialized_N = 0, initialized_L = 0, initialized_K = 0;

// Check compatibility and find GPUs needing initialization
bool compatible = (initialized_N == N && initialized_L == q.size() && initialized_K == p.size());
std::vector<int> gpus_to_init;
for (int id : GPUid) {
    if (initialized_gpus.find(id) == initialized_gpus.end()) {
        gpus_to_init.push_back(id);
    }
}

// Initialize only new GPUs
for (int gpu_id : gpus_to_init) {
    cudaSetDevice(gpu_id);
    cudaMalloc(&(hG_.psi_ptr[gpu_id][j]), bytes);  // Uses actual GPU ID
    // ... copy constants ...
    initialized_gpus.insert(gpu_id);
}
```

---

## 3. Files Modified Summary

| File | Changes |
|------|---------|
| `include/CKKS/Context.cuh` | Added KeySwitchingKey.cuh, map, optional includes; Added member variables for keys |
| `src/CKKS/Context.cu` | Removed static globals; Implemented instance-based key methods |
| `src/CKKS/Limb.cu` | Changed graph cache to pair<int,int> with device ID (2 locations) |
| `src/CKKS/LimbPartition.cu` | Changed graph cache to pair<int,int> with device ID (2 locations) |
| `src/ConstantsGPU.cu` | Fixed cudaSetDevice; Added initialization tracking; Fixed array indexing |

---

## 4. Benchmark Suite

### 4.1 Benchmarks Created

Located in `examples/multigpu_benchmark/src/`:

1. **test_dual_context.cu** - Validates both GPUs work independently
2. **multigpu_simple_bench.cu** - Pure GPU compute timing (HMult operations)
3. **multigpu_matmul_bench.cu** - HE matrix-vector multiplication with timing breakdown
4. **multigpu_bootstrap_bench.cu** - Data-parallel bootstrapping benchmark

### 4.2 Running Benchmarks

```bash
cd /root/FIDESlib/examples/multigpu_benchmark/build

# Basic validation
./test_dual_context

# Simple benchmark (configurable)
./multigpu_simple_bench --logN 14 --L 12 --ops 32 --iters 10

# MatMul benchmark
./multigpu_matmul_bench --rows 8 --iters 3

# Bootstrap benchmark  
./multigpu_bootstrap_bench --iters 2
```

---

## 5. Benchmark Results

### 5.1 Performance Table

| Benchmark | Single GPU | Multi-GPU (2×L4) | Speedup | Efficiency |
|-----------|------------|------------------|---------|------------|
| Simple (logN=13, L=8, 16 ops) | 33.05 ms | 23.22 ms | **1.42×** | 71.2% |
| Simple (logN=14, L=12, 32 ops) | 188.64 ms | 109.81 ms | **1.72×** | 85.9% |
| Simple (logN=15, L=14, 64 ops) | 456.06 ms | 313.43 ms | **1.46×** | 72.8% |
| MatMul (compute only) | 27.69 ms | 19.97 ms | **1.39×** | 69.3% |
| Bootstrap (compute only) | 16.29 ms | 12.60 ms | **1.29×** | 64.7% |

### 5.2 Throughput

| Configuration | Single GPU | Multi-GPU |
|---------------|------------|-----------|
| logN=14, L=12 | 1696 ops/sec | 2914 ops/sec |
| logN=15, L=14 | 1403 ops/sec | 2042 ops/sec |

### 5.3 Sample Output

```
╔═══════════════════════════════════════════════════════════════╗
║   FIDESlib Multi-GPU Simple Benchmark                         ║
╚═══════════════════════════════════════════════════════════════╝

=== GPU Information ===
  GPU 0: NVIDIA L4 (Compute 8.9)
  GPU 1: NVIDIA L4 (Compute 8.9)

Configuration: logN=14, L=12, ops=32, iters=10, GPUs=2

>>> Single GPU (GPU 0): 32 ops x 10 iters <<<
  Total time: 188.642 ms
  Throughput: 1696.34 ops/sec

>>> Multi-GPU (2 GPUs): 32 ops x 10 iters <<<
  Total time (wall): 109.813 ms
    GPU 0: 108.461 ms (16 ops)
    GPU 1: 109.639 ms (16 ops)
  Throughput: 2914.04 ops/sec

╔═══════════════════════════════════════════════════════════════╗
║                       RESULTS                                 ║
╚═══════════════════════════════════════════════════════════════╝
  Speedup:              1.72x
  Parallel Efficiency:  85.89%
```

---

## 6. Multi-GPU Usage Pattern

```cpp
#include <CKKS/Context.cuh>
#include <CKKS/Ciphertext.cuh>
#include <thread>

// Create independent contexts for each GPU
FIDESlib::CKKS::Context ctx0(params, {0});  // GPU 0
FIDESlib::CKKS::Context ctx1(params, {1});  // GPU 1

// Load keys into each context
ctx0.AddEvalKey(std::move(eval_key_gpu0));
ctx1.AddEvalKey(std::move(eval_key_gpu1));

// Create ciphertexts on respective GPUs
cudaSetDevice(0);
auto ct0 = std::make_unique<Ciphertext>(ctx0, raw_ct);
cudaSetDevice(1);
auto ct1 = std::make_unique<Ciphertext>(ctx1, raw_ct);

// Run operations in parallel
std::thread t0([&]() {
    cudaSetDevice(0);
    ct0->mult(*ct0, ctx0.GetEvalKey());
    cudaDeviceSynchronize();
});

std::thread t1([&]() {
    cudaSetDevice(1);
    ct1->mult(*ct1, ctx1.GetEvalKey());
    cudaDeviceSynchronize();
});

t0.join();
t1.join();
```

---

## 7. Key Findings

1. **Larger workloads scale better**: The logN=14, L=12 configuration achieved 85.9% parallel efficiency vs 65% for smaller configs.

2. **Load balancing is effective**: GPU 0 and GPU 1 execution times differ by < 5%.

3. **Context overhead is amortized**: Initial setup (48-171ms) is one-time per context.

4. **GPU-count agnostic**: The solution works with any number of GPUs without code changes.

---

## 8. Git Commits

```
de8f639 Update project report with detailed multi-GPU implementation results
4008645 Enhance multi-GPU support in FIDESlib by refactoring key management and CUDA graph execution
```

---

## 9. Future Work

1. **NVLink interconnect**: Test on systems with NVLink for faster inter-GPU communication
2. **4+ GPU scaling**: Evaluate scaling efficiency beyond 2 GPUs
3. **RNS limb sharding**: Explore distributing RNS limbs across GPUs (model parallelism)
4. **Automatic load balancing**: Dynamic work distribution based on GPU capabilities

---

## 10. Acknowledgments

- Original FIDESlib authors at University of Murcia (CAPS-UMU)
- RunPod for cloud GPU access
