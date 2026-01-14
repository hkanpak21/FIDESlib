/**
 * @file multigpu_gpu_only_bench.cu
 * @brief Multi-GPU Benchmark - Pure GPU Compute Performance Analysis
 * 
 * This benchmark focuses ONLY on GPU compute time, excluding:
 * - CPU-side encryption (which dominates in full benchmarks)
 * - Context setup overhead
 * - Key generation
 * 
 * This shows the true multi-GPU speedup for HE operations.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cuda_runtime.h>

#include <openfhe.h>
#include <CKKS/Context.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

#include "PerformanceMonitor.cuh"

// Prime records
std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
    {.p = 2251799787601921},    {.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
    {.p = 2251799807131649},    {.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
    {.p = 2251799809884161},    {.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
    {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

int getGPUCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void printGPUInfo() {
    int deviceCount = getGPUCount();
    std::cout << "\n=== GPU Information ===" << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (SM: " << prop.multiProcessorCount << ")"
                  << " - " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB"
                  << std::endl;
    }
    std::cout << std::endl;
}

size_t estimateCiphertextSize(int logN, int L) {
    return 2ULL * (L + 1) * (1ULL << logN) * sizeof(uint64_t);
}

// ============================================================================
// Benchmark Results Structure
// ============================================================================
struct BenchResult {
    double totalTimeMs;
    double avgIterTimeMs;
    double opsPerSec;
    int totalOps;
    int iterations;
    
    void print(const std::string& label) const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << label << ":" << std::endl;
        std::cout << "    Total GPU time:    " << totalTimeMs << " ms" << std::endl;
        std::cout << "    Avg per iteration: " << avgIterTimeMs << " ms" << std::endl;
        std::cout << "    Throughput:        " << opsPerSec << " ops/sec" << std::endl;
    }
};

// ============================================================================
// Single GPU - GPU-only timing
// ============================================================================
BenchResult benchGPUOnly_Single(int numOps, int numIters, int logN, int L) {
    std::cout << "\n>>> Setting up Single GPU benchmark (GPU 0)..." << std::endl;
    
    // Setup (not timed)
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    params.SetMultiplicativeDepth(L);
    params.SetScalingModSize(51);
    params.SetRingDim(1 << logN);
    params.SetBatchSize(1 << (logN - 1));
    params.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    params.SetFirstModSize(60);
    
    auto cc = lbcrypto::GenCryptoContext(params);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    cudaSetDevice(0);
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    FIDESlib::CKKS::Context gpu_cc(p, {0});
    
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(gpu_cc);
    eval_key_gpu.Initialize(gpu_cc, eval_key_raw);
    gpu_cc.AddEvalKey(std::move(eval_key_gpu));
    
    int slots = 1 << (logN - 1);
    std::vector<double> test_data(slots, 1.5);
    auto pt = cc->MakeCKKSPackedPlaintext(test_data);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> gpu_cts;
    for (int i = 0; i < numOps; ++i) {
        auto ct = cc->Encrypt(keys.publicKey, pt);
        auto ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
        gpu_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(gpu_cc, ct_raw));
    }
    
    cudaDeviceSynchronize();
    
    // Warmup with fresh copy
    std::cout << "  Warming up..." << std::endl;
    {
        auto ct_warmup = cc->Encrypt(keys.publicKey, pt);
        auto ct_warmup_raw = FIDESlib::CKKS::GetRawCipherText(cc, ct_warmup);
        FIDESlib::CKKS::Ciphertext gpu_ct_warmup(gpu_cc, ct_warmup_raw);
        for (int w = 0; w < 3; ++w) {
            gpu_ct_warmup.mult(gpu_ct_warmup, gpu_cc.GetEvalKey());
        }
        cudaDeviceSynchronize();
    }
    
    // =========================================================================
    // TIMED REGION - GPU COMPUTE ONLY
    // Each iteration creates fresh ciphertexts to avoid level exhaustion
    // =========================================================================
    std::cout << "  Running " << numOps << " ops x " << numIters << " iterations..." << std::endl;
    std::cout << "  (Each iteration uses fresh ciphertexts)" << std::endl;
    
    size_t ctSize = estimateCiphertextSize(logN, L);
    
    // Pre-create raw ciphertexts on CPU
    std::vector<FIDESlib::CKKS::RawCipherText> raw_cts;
    for (int i = 0; i < numOps; ++i) {
        auto ct = cc->Encrypt(keys.publicKey, pt);
        raw_cts.push_back(FIDESlib::CKKS::GetRawCipherText(cc, ct));
    }
    
    // Use CUDA events for timing
    PerfMon::CudaEventTimer timer(0);
    double totalMs = 0;
    
    for (int iter = 0; iter < numIters; ++iter) {
        // Create fresh GPU ciphertexts for this iteration
        std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> iter_cts;
        for (int i = 0; i < numOps; ++i) {
            iter_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(gpu_cc, raw_cts[i]));
        }
        cudaDeviceSynchronize();
        
        // Time just the multiplications
        timer.start();
        for (int i = 0; i < numOps; ++i) {
            iter_cts[i]->mult(*iter_cts[i], gpu_cc.GetEvalKey());
        }
        cudaDeviceSynchronize();
        timer.stop();
        
        totalMs += timer.elapsedMs();
    }
    
    // Get results
    BenchResult result;
    result.totalTimeMs = totalMs;
    result.avgIterTimeMs = totalMs / numIters;
    result.totalOps = numOps * numIters;
    result.iterations = numIters;
    result.opsPerSec = (result.totalOps * 1000.0) / totalMs;
    
    cc->ClearEvalMultKeys();
    return result;
}

// ============================================================================
// Multi GPU - GPU-only timing
// ============================================================================
BenchResult benchGPUOnly_Multi(int numOps, int numIters, int logN, int L, int numGPUs,
                                std::vector<double>& perGpuTimesOut) {
    std::cout << "\n>>> Setting up Multi-GPU benchmark (" << numGPUs << " GPUs)..." << std::endl;
    
    // Setup (not timed)
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    params.SetMultiplicativeDepth(L);
    params.SetScalingModSize(51);
    params.SetRingDim(1 << logN);
    params.SetBatchSize(1 << (logN - 1));
    params.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    params.SetFirstModSize(60);
    
    auto cc = lbcrypto::GenCryptoContext(params);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Context>> gpu_contexts;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        gpu_contexts.push_back(std::make_unique<FIDESlib::CKKS::Context>(p, std::vector<int>{g}));
        cudaDeviceSynchronize();
    }
    
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(*gpu_contexts[g]);
        eval_key_gpu.Initialize(*gpu_contexts[g], eval_key_raw);
        gpu_contexts[g]->AddEvalKey(std::move(eval_key_gpu));
        cudaDeviceSynchronize();
    }
    
    int slots = 1 << (logN - 1);
    std::vector<double> test_data(slots, 1.5);
    auto pt = cc->MakeCKKSPackedPlaintext(test_data);
    
    int opsPerGPU = numOps / numGPUs;
    
    // Pre-create raw ciphertexts for each GPU
    std::vector<std::vector<FIDESlib::CKKS::RawCipherText>> raw_cts(numGPUs);
    for (int g = 0; g < numGPUs; ++g) {
        int myOps = (g == numGPUs - 1) ? (numOps - g * opsPerGPU) : opsPerGPU;
        for (int i = 0; i < myOps; ++i) {
            auto ct = cc->Encrypt(keys.publicKey, pt);
            raw_cts[g].push_back(FIDESlib::CKKS::GetRawCipherText(cc, ct));
        }
    }
    
    // Warmup with fresh ciphertexts
    std::cout << "  Warming up all GPUs..." << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        if (!raw_cts[g].empty()) {
            FIDESlib::CKKS::Ciphertext warmup_ct(*gpu_contexts[g], raw_cts[g][0]);
            for (int w = 0; w < 3; ++w) {
                warmup_ct.mult(warmup_ct, gpu_contexts[g]->GetEvalKey());
            }
        }
        cudaDeviceSynchronize();
    }
    
    // Sync all GPUs
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    
    // =========================================================================
    // TIMED REGION - GPU COMPUTE ONLY (PARALLEL)
    // Each iteration uses fresh ciphertexts to avoid level exhaustion
    // =========================================================================
    std::cout << "  Running " << numOps << " ops x " << numIters << " iterations in parallel..." << std::endl;
    std::cout << "  (Each iteration uses fresh ciphertexts)" << std::endl;
    
    size_t ctSize = estimateCiphertextSize(logN, L);
    std::vector<double> perGpuTimes(numGPUs, 0);
    
    // Wall-clock timer for total parallel time
    auto wallStart = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int g = 0; g < numGPUs; ++g) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            int myOps = raw_cts[g].size();
            
            PerfMon::CudaEventTimer timer(g);
            double gpuTotalMs = 0;
            
            for (int iter = 0; iter < numIters; ++iter) {
                // Create fresh GPU ciphertexts
                std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> iter_cts;
                for (int i = 0; i < myOps; ++i) {
                    iter_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(
                        *gpu_contexts[g], raw_cts[g][i]));
                }
                cudaDeviceSynchronize();
                
                // Time just the multiplications
                timer.start();
                for (auto& ct : iter_cts) {
                    ct->mult(*ct, gpu_contexts[g]->GetEvalKey());
                }
                cudaDeviceSynchronize();
                timer.stop();
                
                gpuTotalMs += timer.elapsedMs();
            }
            
            perGpuTimes[g] = gpuTotalMs;
        });
    }
    
    for (auto& t : threads) t.join();
    
    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
    
    // Store per-GPU times for caller
    perGpuTimesOut = perGpuTimes;
    
    // Print per-GPU breakdown
    std::cout << "\n  Per-GPU Compute Times:" << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        std::cout << "    GPU " << g << ": " << std::fixed << std::setprecision(2) 
                  << perGpuTimes[g] << " ms (" << raw_cts[g].size() << " ops/iter)" << std::endl;
    }
    
    // The total time is the wall clock (max of parallel execution)
    BenchResult result;
    result.totalTimeMs = wallMs;
    result.avgIterTimeMs = wallMs / numIters;
    result.totalOps = numOps * numIters;
    result.iterations = numIters;
    result.opsPerSec = (result.totalOps * 1000.0) / wallMs;
    
    cc->ClearEvalMultKeys();
    return result;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   FIDESlib Multi-GPU Benchmark - GPU Compute Only                         ║" << std::endl;
    std::cout << "║   (Excludes CPU encryption/setup overhead)                                ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    printGPUInfo();
    
    int numGPUs = getGPUCount();
    int logN = 14;
    int L = 12;
    int numOps = 32;
    int numIters = 10;
    bool exportCSV = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--logN" && i+1 < argc) logN = std::stoi(argv[++i]);
        else if (arg == "--L" && i+1 < argc) L = std::stoi(argv[++i]);
        else if (arg == "--ops" && i+1 < argc) numOps = std::stoi(argv[++i]);
        else if (arg == "--iters" && i+1 < argc) numIters = std::stoi(argv[++i]);
        else if (arg == "--gpus" && i+1 < argc) numGPUs = std::min(std::stoi(argv[++i]), getGPUCount());
        else if (arg == "--csv") exportCSV = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --logN <n>     Ring dimension log (default: 14)\n"
                      << "  --L <n>        Multiplicative depth (default: 12)\n"
                      << "  --ops <n>      Number of operations (default: 32)\n"
                      << "  --iters <n>    Number of iterations (default: 10)\n"
                      << "  --gpus <n>     Number of GPUs (default: all)\n"
                      << "  --csv          Export results to CSV\n";
            return 0;
        }
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  logN = " << logN << " (Ring dim = " << (1 << logN) << ")" << std::endl;
    std::cout << "  L = " << L << " (Multiplicative depth)" << std::endl;
    std::cout << "  Operations = " << numOps << std::endl;
    std::cout << "  Iterations = " << numIters << std::endl;
    std::cout << "  GPUs = " << numGPUs << std::endl;
    std::cout << "  Est. CT size = " << (estimateCiphertextSize(logN, L) / (1024.0*1024.0)) << " MB" << std::endl;
    
    // Run benchmarks
    auto singleResult = benchGPUOnly_Single(numOps, numIters, logN, L);
    BenchResult multiResult = {0,0,0,0,0};
    std::vector<double> perGpuTimes;
    
    if (numGPUs > 1) {
        multiResult = benchGPUOnly_Multi(numOps, numIters, logN, L, numGPUs, perGpuTimes);
    }
    
    // =========================================================================
    // Results
    // =========================================================================
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                       GPU COMPUTE RESULTS                                 ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Single GPU (GPU 0)                                                          │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  Total GPU Compute:    " << std::setw(10) << singleResult.totalTimeMs << " ms                                      │" << std::endl;
    std::cout << "│  Per Iteration:        " << std::setw(10) << singleResult.avgIterTimeMs << " ms                                      │" << std::endl;
    std::cout << "│  Throughput:           " << std::setw(10) << singleResult.opsPerSec << " HE-mults/sec                             │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    if (numGPUs > 1) {
        double speedup = singleResult.totalTimeMs / multiResult.totalTimeMs;
        double efficiency = (speedup / numGPUs) * 100.0;
        
        std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ Multi-GPU (" << numGPUs << " GPUs)                                                           │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│  Total GPU Compute:    " << std::setw(10) << multiResult.totalTimeMs << " ms                                      │" << std::endl;
        std::cout << "│  Per Iteration:        " << std::setw(10) << multiResult.avgIterTimeMs << " ms                                      │" << std::endl;
        std::cout << "│  Throughput:           " << std::setw(10) << multiResult.opsPerSec << " HE-mults/sec                             │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│  SPEEDUP:              " << std::setw(10) << speedup << "x                                         │" << std::endl;
        std::cout << "│  Parallel Efficiency:  " << std::setw(10) << efficiency << "%                                        │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        // Speedup visualization
        std::cout << "\n  Speedup: [";
        int bars = (int)(speedup * 10);
        for (int i = 0; i < 40; ++i) {
            if (i < bars) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] " << speedup << "x" << std::endl;
        
        std::cout << "\n  Efficiency: [";
        int effBars = (int)(efficiency / 2.5);
        for (int i = 0; i < 40; ++i) {
            if (i < effBars) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] " << efficiency << "%" << std::endl;
    }
    
    // Print GPU utilization summary
    if (numGPUs > 1 && !perGpuTimes.empty()) {
        std::cout << "\n  Per-GPU Analysis:" << std::endl;
        double maxTime = *std::max_element(perGpuTimes.begin(), perGpuTimes.end());
        double minTime = *std::min_element(perGpuTimes.begin(), perGpuTimes.end());
        double avgTime = 0;
        for (double t : perGpuTimes) avgTime += t;
        avgTime /= perGpuTimes.size();
        
        std::cout << "    Max GPU time:   " << maxTime << " ms" << std::endl;
        std::cout << "    Min GPU time:   " << minTime << " ms" << std::endl;
        std::cout << "    Avg GPU time:   " << avgTime << " ms" << std::endl;
        std::cout << "    Load imbalance: " << ((maxTime - minTime) / avgTime * 100.0) << "%" << std::endl;
    }
    
    std::cout << "\n✓ Benchmark complete!" << std::endl;
    return 0;
}
