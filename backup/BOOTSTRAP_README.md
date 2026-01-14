# FIDESlib Multi-GPU Modifications - Quick Start Guide

## Overview

This document provides instructions to restore the multi-GPU modifications to FIDESlib after a server restart or fresh environment.

## Prerequisites

- NVIDIA GPUs (tested on L4, should work on any Compute Capability 8.0+)
- CUDA 12.0+
- CMake 3.18+
- OpenFHE installed
- NCCL installed

## Quick Start

```bash
cd /FIDES
chmod +x bootstrap.sh
./bootstrap.sh
```

## Manual Steps (if bootstrap fails)

### 1. Build FIDESlib

```bash
cd /FIDES/FIDESlib
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
make install
```

### 2. Build Examples

```bash
cd /FIDES/FIDESlib/examples/he_matmul
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
```

### 3. Run Tests

```bash
# Verify dual context works
./test_dual_context

# Run benchmark
./he_matmul_multigpu
```

## Key Modified Files

If you need to verify or re-apply modifications, check these files:

| File | Purpose |
|------|---------|
| `include/CKKS/Context.cuh` | Per-context key storage (member variables) |
| `src/CKKS/Context.cu` | Instance methods for key access |
| `src/CKKS/Limb.cu` | Per-device graph cache (2 locations) |
| `src/CKKS/LimbPartition.cu` | Per-device graph cache (2 locations) |
| `src/ConstantsGPU.cu` | Multi-context constants initialization |

## Expected Results

### Dual Context Test
```
=== Testing Multiple FIDESlib Contexts ===
...
=== SUCCESS! Both contexts work independently! ===
```

### Benchmark Results (2x L4 GPUs)
```
Single GPU:    ~118 ms (270 ops/sec)
Multi-GPU:     ~47 ms (679 ops/sec)
Speedup:       ~2.5x
```

## Troubleshooting

### "illegal memory access" errors

1. Check that all 4 graph cache modifications are in place
2. Verify ConstantsGPU.cu has the GPU ID indexing fix
3. Ensure Context.cu uses instance members, not static globals

### Build errors about KeySwitchingKey

Make sure Context.cuh has:
```cpp
#include "KeySwitchingKey.cuh"
#include <map>
#include <optional>
```

### "NCCL not found" errors

```bash
# Install NCCL if missing
apt-get install libnccl2 libnccl-dev
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenFHE Context                          │
│                 (CPU-side encryption)                       │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   FIDESlib Context 0    │     │   FIDESlib Context 1    │
│       (GPU 0)           │     │       (GPU 1)           │
├─────────────────────────┤     ├─────────────────────────┤
│ • eval_key_ (own copy)  │     │ • eval_key_ (own copy)  │
│ • rot_keys_ (own copy)  │     │ • rot_keys_ (own copy)  │
│ • GPU 0 memory          │     │ • GPU 1 memory          │
│ • GPU 0 streams         │     │ • GPU 1 streams         │
└─────────────────────────┘     └─────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                    ┌───────────────┐
                    │     NCCL      │
                    │ (aggregation) │
                    └───────────────┘
```

## Contact

See `/FIDES/IMPLEMENTATION_REPORT.md` for detailed technical documentation.
