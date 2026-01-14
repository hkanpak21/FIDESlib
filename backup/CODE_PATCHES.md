# FIDESlib Multi-GPU Modifications - Code Patches

This file contains the exact code changes needed to enable multi-GPU support.
Use this as a reference to manually re-apply changes if needed.

---

## Patch 1: Context.cuh - Add Per-Context Key Storage

**File:** `/FIDES/FIDESlib/include/CKKS/Context.cuh`

### Add includes after existing includes:
```cpp
#include "KeySwitchingKey.cuh"
#include <map>
#include <optional>
```

### Add member variables in Context class (private section):
```cpp
// Per-context key storage (replaces static globals)
mutable std::map<int, BootstrapPrecomputation> boot_precomps_;
mutable std::map<int, KeySwitchingKey> rot_keys_;
mutable std::optional<KeySwitchingKey> eval_key_;
```

### Change method signatures (remove 'static' keyword):
```cpp
// Change FROM static TO instance methods:
void AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) const;
BootstrapPrecomputation& GetBootPrecomputation(int slots) const;
void AddRotationKey(int index, KeySwitchingKey&& ksk);
KeySwitchingKey& GetRotationKey(int index);
bool HasRotationKey(int index);
void AddEvalKey(KeySwitchingKey&& ksk);
KeySwitchingKey& GetEvalKey();
```

---

## Patch 2: Context.cu - Use Instance Members

**File:** `/FIDES/FIDESlib/src/CKKS/Context.cu`

### Remove these global variables (if present):
```cpp
// DELETE THESE LINES:
// static std::optional<KeySwitchingKey> eval_key;
// static std::map<int, KeySwitchingKey> rot_keys;
// static std::map<int, BootstrapPrecomputation> boot_precomps;
```

### Update method implementations:
```cpp
void Context::AddEvalKey(KeySwitchingKey&& ksk) {
    eval_key_ = std::move(ksk);  // Use member eval_key_
}

KeySwitchingKey& Context::GetEvalKey() {
    if (!eval_key_.has_value()) {
        throw std::runtime_error("Eval key not set for this context");
    }
    return eval_key_.value();
}

void Context::AddRotationKey(int index, KeySwitchingKey&& ksk) {
    rot_keys_[index] = std::move(ksk);  // Use member rot_keys_
}

KeySwitchingKey& Context::GetRotationKey(int index) {
    return rot_keys_.at(index);
}

bool Context::HasRotationKey(int index) {
    return rot_keys_.find(index) != rot_keys_.end();
}

void Context::AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) const {
    boot_precomps_[slots] = std::move(precomp);  // Use member boot_precomps_
}

BootstrapPrecomputation& Context::GetBootPrecomputation(int slots) const {
    return boot_precomps_.at(slots);
}
```

---

## Patch 3: Limb.cu - Per-Device Graph Caches

**File:** `/FIDES/FIDESlib/src/CKKS/Limb.cu`

### Location 1 (~line 345, NTT function):

Find:
```cpp
static std::map<int, cudaGraphExec_t> exec;
```

Replace with:
```cpp
static std::map<std::pair<int,int>, cudaGraphExec_t> exec;
int current_device;
cudaGetDevice(&current_device);
```

And change the key creation from:
```cpp
auto key = (int)v.size;
```
To:
```cpp
auto key = std::make_pair(current_device, (int)v.size);
```

### Location 2 (~line 480, INTT function):

Same pattern - find static map with int key, change to pair<int,int> with device ID.

---

## Patch 4: LimbPartition.cu - Per-Device Graph Caches

**File:** `/FIDES/FIDESlib/src/CKKS/LimbPartition.cu`

### Location 1 (~line 342, batch NTT):

Find:
```cpp
static std::map<int, cudaGraphExec_t> execs;
```

Replace with:
```cpp
static std::map<std::pair<int,int>, cudaGraphExec_t> execs;
int current_device;
cudaGetDevice(&current_device);
auto key = std::make_pair(current_device, (int)limb.size());
```

### Location 2 (~line 551, multPt function):

Same pattern.

---

## Patch 5: ConstantsGPU.cu - Multi-Context Support

**File:** `/FIDES/FIDESlib/src/ConstantsGPU.cu`

### Add after includes:
```cpp
#include <set>

namespace FIDESlib {

// Track which GPUs have been initialized
static std::set<int> initialized_gpus;
static int initialized_N = 0;
static int initialized_L = 0;
static int initialized_K = 0;
```

### Fix hardcoded device (around line 351):

Find:
```cpp
cudaSetDevice(0/*GPUid.at(i)*/);
```

Replace with:
```cpp
cudaSetDevice(GPUid.at(i));
```

### Add compatibility check at start of SetupConstants():
```cpp
// Check if we need to re-initialize or can reuse existing setup
bool compatible = (initialized_N == N && initialized_L == (int)q.size() && initialized_K == (int)p.size());

// Find which GPUs need initialization
std::vector<int> gpus_to_init;
for (int id : GPUid) {
    if (initialized_gpus.find(id) == initialized_gpus.end()) {
        gpus_to_init.push_back(id);
    }
}

// If compatible and all GPUs already initialized, we can skip entirely
if (compatible && gpus_to_init.empty()) {
    for (int id : GPUid) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(constants, &host_constants, sizeof(Constants), 0, cudaMemcpyHostToDevice);
    }
    return;
}

// If not compatible with previous setup, clean up and start fresh
if (!compatible && !initialized_gpus.empty()) {
    cleanUpPrevious();
    initialized_gpus.clear();
    gpus_to_init = std::vector<int>(GPUid.begin(), GPUid.end());
}

// Track what we're initializing
initialized_N = N;
initialized_L = q.size();
initialized_K = p.size();
```

### Change GPU initialization loop indexing:

Find loop like:
```cpp
for (size_t i = 0; i < GPUid.size(); ++i) {
    cudaSetDevice(GPUid.at(i));
    // ... uses hG_.psi_ptr[i][j] ...
}
```

Replace with:
```cpp
for (int gpu_id : gpus_to_init) {
    cudaSetDevice(gpu_id);
    // ... uses hG_.psi_ptr[gpu_id][j] ...  (note: gpu_id, not i)
    
    // At end of loop:
    initialized_gpus.insert(gpu_id);
}
```

---

## Verification

After applying all patches:

1. Rebuild FIDESlib:
```bash
cd /FIDES/FIDESlib/build
make -j$(nproc)
make install
```

2. Rebuild and run test:
```bash
cd /FIDES/FIDESlib/examples/he_matmul/build
make -j$(nproc) test_dual_context
./test_dual_context
```

Expected output:
```
=== SUCCESS! Both contexts work independently! ===
```

---

## Common Errors After Patching

### Error: "no member named 'eval_key_' in Context"
- Make sure you added the member variables to Context.cuh

### Error: "cannot call member function without object"
- Make sure you removed 'static' from method declarations

### Error: "illegal memory access" at runtime
- Check that ALL 4 graph cache locations were updated
- Verify ConstantsGPU.cu uses gpu_id instead of loop index i

### Error: "undefined reference to SetupConstants"
- Rebuild FIDESlib completely (make clean && make)
