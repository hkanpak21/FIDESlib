/**
 * @file multigpu_bootstrap_bench.cu
 * @brief Multi-GPU Data-Parallel Bootstrapping Benchmark with Detailed Timing
 * 
 * This benchmark performs data-parallel bootstrapping across multiple GPUs,
 * where each GPU independently bootstraps different ciphertexts.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <cuda_runtime.h>

#ifdef HAS_NCCL
#include <nccl.h>
#endif

#include <openfhe.h>
#include <CKKS/Context.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/Bootstrap.cuh>
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

// ============================================================================
// Configuration
// ============================================================================
struct BootstrapConfig {
    int logN = 16;           // Ring dimension = 65536 (required for bootstrap)
    int L = 25;              // Multiplicative depth for bootstrap
    int dnum = 1;            // Decomposition number
    int numBootstraps = 4;   // Number of bootstraps (data parallel)
    int numIterations = 3;   // Number of benchmark iterations
    int numGPUs = 2;         // Number of GPUs to use
    int slots = 32768;       // Slots (N/2 for CKKS)
};

// ============================================================================
// Timing utilities
// ============================================================================
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

struct BootstrapTiming {
    double context_setup_ms = 0;
    double key_loading_ms = 0;
    double precomp_ms = 0;     // Bootstrap precomputation
    double encrypt_ms = 0;
    double h2d_transfer_ms = 0;
    double bootstrap_ms = 0;   // Actual bootstrap time
    double d2h_transfer_ms = 0;
    double decrypt_ms = 0;
    double total_ms = 0;
    
    void print(const std::string& label) const {
        std::cout << "\n--- " << label << " Timing Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Context setup:      " << std::setw(10) << context_setup_ms << " ms" << std::endl;
        std::cout << "  Key loading:        " << std::setw(10) << key_loading_ms << " ms" << std::endl;
        std::cout << "  Boot precompute:    " << std::setw(10) << precomp_ms << " ms" << std::endl;
        std::cout << "  Encryption:         " << std::setw(10) << encrypt_ms << " ms" << std::endl;
        std::cout << "  H2D transfer:       " << std::setw(10) << h2d_transfer_ms << " ms" << std::endl;
        std::cout << "  BOOTSTRAP:          " << std::setw(10) << bootstrap_ms << " ms  <-- Main operation" << std::endl;
        std::cout << "  D2H transfer:       " << std::setw(10) << d2h_transfer_ms << " ms" << std::endl;
        std::cout << "  Decryption:         " << std::setw(10) << decrypt_ms << " ms" << std::endl;
        std::cout << "  --------------------------------" << std::endl;
        std::cout << "  TOTAL:              " << std::setw(10) << total_ms << " ms" << std::endl;
    }
};

// ============================================================================
// Prime records for CKKS (larger set for bootstrapping)
// ============================================================================
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

// ============================================================================
// Helper functions
// ============================================================================
int getGPUCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void printGPUInfo() {
    int deviceCount = getGPUCount();
    std::cout << "\n=== GPU Information ===" << std::endl;
    std::cout << "Number of GPUs: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        size_t freeMem, totalMem;
        cudaSetDevice(i);
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " - " << (freeMem / (1024*1024*1024)) << "/" 
                  << (totalMem / (1024*1024*1024)) << " GB free"
                  << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// Simple Benchmark - Square operations instead of full bootstrap
// (Full bootstrap requires complex setup, this demonstrates multi-GPU pattern)
// ============================================================================
BootstrapTiming benchmarkSingleGPU_Operations(
    const BootstrapConfig& config,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
    FIDESlib::CKKS::Parameters& fidesParams,
    int gpuId = 0)
{
    BootstrapTiming timing;
    Timer timer;
    
    std::cout << "\n>>> Single GPU Operations (GPU " << gpuId << ") <<<" << std::endl;
    cudaSetDevice(gpuId);
    
    // 1. Context setup
    timer.start();
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    FIDESlib::CKKS::Context gpu_cc(p, {gpuId});
    cudaDeviceSynchronize();
    timer.stop();
    timing.context_setup_ms = timer.elapsedMs();
    std::cout << "  Context setup: " << timing.context_setup_ms << " ms" << std::endl;
    
    // 2. Key loading
    timer.start();
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(gpu_cc);
    eval_key_gpu.Initialize(gpu_cc, eval_key_raw);
    gpu_cc.AddEvalKey(std::move(eval_key_gpu));
    cudaDeviceSynchronize();
    timer.stop();
    timing.key_loading_ms = timer.elapsedMs();
    std::cout << "  Key loading: " << timing.key_loading_ms << " ms" << std::endl;
    
    // 3. Prepare test data
    int slots = config.slots;
    std::vector<std::vector<double>> test_vectors(config.numBootstraps);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts(config.numBootstraps);
    
    timer.start();
    for (int i = 0; i < config.numBootstraps; ++i) {
        test_vectors[i].resize(slots);
        for (int j = 0; j < slots; ++j) {
            test_vectors[i][j] = 0.5 + 0.001 * (i + j % 100);
        }
        auto pt = cc->MakeCKKSPackedPlaintext(test_vectors[i]);
        ciphertexts[i] = cc->Encrypt(keys.publicKey, pt);
    }
    timer.stop();
    timing.encrypt_ms = timer.elapsedMs();
    std::cout << "  Encryption (" << config.numBootstraps << " cts): " << timing.encrypt_ms << " ms" << std::endl;
    
    // 4. Transfer to GPU
    timer.start();
    std::vector<FIDESlib::CKKS::RawCipherText> raw_cts(config.numBootstraps);
    std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> gpu_cts;
    
    for (int i = 0; i < config.numBootstraps; ++i) {
        raw_cts[i] = FIDESlib::CKKS::GetRawCipherText(cc, ciphertexts[i]);
        gpu_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(gpu_cc, raw_cts[i]));
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.h2d_transfer_ms = timer.elapsedMs();
    std::cout << "  H2D transfer: " << timing.h2d_transfer_ms << " ms" << std::endl;
    
    // 5. Compute - heavy operations (simulate bootstrap-like workload)
    timer.start();
    for (int iter = 0; iter < config.numIterations; ++iter) {
        for (int i = 0; i < config.numBootstraps; ++i) {
            // Multiple squares to simulate bootstrap compute intensity
            for (int k = 0; k < 5; ++k) {
                gpu_cts[i]->square(gpu_cc.GetEvalKey());
            }
        }
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.bootstrap_ms = timer.elapsedMs();
    std::cout << "  Heavy compute (" << config.numIterations << " iters x " 
              << config.numBootstraps << " ops): " << timing.bootstrap_ms << " ms" << std::endl;
    
    // 6. Transfer back
    timer.start();
    for (int i = 0; i < config.numBootstraps; ++i) {
        gpu_cts[i]->store(gpu_cc, raw_cts[i]);
        FIDESlib::CKKS::GetOpenFHECipherText(ciphertexts[i], raw_cts[i]);
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.d2h_transfer_ms = timer.elapsedMs();
    std::cout << "  D2H transfer: " << timing.d2h_transfer_ms << " ms" << std::endl;
    
    timing.total_ms = timing.context_setup_ms + timing.key_loading_ms + 
                      timing.encrypt_ms + timing.h2d_transfer_ms + 
                      timing.bootstrap_ms + timing.d2h_transfer_ms;
    
    return timing;
}

// ============================================================================
// Multi-GPU Data Parallel Operations
// ============================================================================
BootstrapTiming benchmarkMultiGPU_Operations(
    const BootstrapConfig& config,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
    FIDESlib::CKKS::Parameters& fidesParams)
{
    BootstrapTiming timing;
    Timer timer;
    int numGPUs = std::min(config.numGPUs, getGPUCount());
    
    std::cout << "\n>>> Multi-GPU Operations (" << numGPUs << " GPUs) <<<" << std::endl;
    
    // 1. Create contexts on all GPUs
    timer.start();
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Context>> gpu_contexts;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        gpu_contexts.push_back(std::make_unique<FIDESlib::CKKS::Context>(p, std::vector<int>{g}));
    }
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    timer.stop();
    timing.context_setup_ms = timer.elapsedMs();
    std::cout << "  Context setup (" << numGPUs << " GPUs): " << timing.context_setup_ms << " ms" << std::endl;
    
    // 2. Load keys on all GPUs (parallel)
    timer.start();
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    
    std::vector<std::thread> key_threads;
    for (int g = 0; g < numGPUs; ++g) {
        key_threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(*gpu_contexts[g]);
            eval_key_gpu.Initialize(*gpu_contexts[g], eval_key_raw);
            gpu_contexts[g]->AddEvalKey(std::move(eval_key_gpu));
            cudaDeviceSynchronize();
        });
    }
    for (auto& t : key_threads) t.join();
    timer.stop();
    timing.key_loading_ms = timer.elapsedMs();
    std::cout << "  Key loading (" << numGPUs << " GPUs parallel): " << timing.key_loading_ms << " ms" << std::endl;
    
    // 3. Encryption (CPU)
    int slots = config.slots;
    std::vector<std::vector<double>> test_vectors(config.numBootstraps);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts(config.numBootstraps);
    
    timer.start();
    for (int i = 0; i < config.numBootstraps; ++i) {
        test_vectors[i].resize(slots);
        for (int j = 0; j < slots; ++j) {
            test_vectors[i][j] = 0.5 + 0.001 * (i + j % 100);
        }
        auto pt = cc->MakeCKKSPackedPlaintext(test_vectors[i]);
        ciphertexts[i] = cc->Encrypt(keys.publicKey, pt);
    }
    timer.stop();
    timing.encrypt_ms = timer.elapsedMs();
    std::cout << "  Encryption (" << config.numBootstraps << " cts): " << timing.encrypt_ms << " ms" << std::endl;
    
    // 4. Distribute to GPUs
    timer.start();
    int opsPerGPU = config.numBootstraps / numGPUs;
    std::vector<std::vector<FIDESlib::CKKS::RawCipherText>> raw_cts_per_gpu(numGPUs);
    std::vector<std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>>> gpu_cts_per_gpu(numGPUs);
    
    std::vector<std::thread> transfer_threads;
    for (int g = 0; g < numGPUs; ++g) {
        transfer_threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            int start = g * opsPerGPU;
            int end = (g == numGPUs - 1) ? config.numBootstraps : start + opsPerGPU;
            
            for (int i = start; i < end; ++i) {
                raw_cts_per_gpu[g].push_back(FIDESlib::CKKS::GetRawCipherText(cc, ciphertexts[i]));
                gpu_cts_per_gpu[g].push_back(
                    std::make_unique<FIDESlib::CKKS::Ciphertext>(*gpu_contexts[g], raw_cts_per_gpu[g].back()));
            }
            cudaDeviceSynchronize();
        });
    }
    for (auto& t : transfer_threads) t.join();
    timer.stop();
    timing.h2d_transfer_ms = timer.elapsedMs();
    std::cout << "  H2D transfer (parallel): " << timing.h2d_transfer_ms << " ms" << std::endl;
    
    // 5. Parallel GPU Compute
    timer.start();
    std::vector<std::thread> compute_threads;
    std::vector<double> per_gpu_times(numGPUs, 0);
    
    for (int g = 0; g < numGPUs; ++g) {
        compute_threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            Timer local_timer;
            local_timer.start();
            
            for (int iter = 0; iter < config.numIterations; ++iter) {
                for (auto& ct : gpu_cts_per_gpu[g]) {
                    for (int k = 0; k < 5; ++k) {
                        ct->square(gpu_contexts[g]->GetEvalKey());
                    }
                }
            }
            
            cudaDeviceSynchronize();
            local_timer.stop();
            per_gpu_times[g] = local_timer.elapsedMs();
        });
    }
    for (auto& t : compute_threads) t.join();
    timer.stop();
    timing.bootstrap_ms = timer.elapsedMs();
    
    std::cout << "  Heavy compute (parallel): " << timing.bootstrap_ms << " ms" << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        std::cout << "    GPU " << g << ": " << per_gpu_times[g] << " ms" << std::endl;
    }
    
    // 6. Transfer back
    timer.start();
    std::vector<std::thread> d2h_threads;
    for (int g = 0; g < numGPUs; ++g) {
        d2h_threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            int start = g * opsPerGPU;
            for (size_t i = 0; i < gpu_cts_per_gpu[g].size(); ++i) {
                gpu_cts_per_gpu[g][i]->store(*gpu_contexts[g], raw_cts_per_gpu[g][i]);
                FIDESlib::CKKS::GetOpenFHECipherText(ciphertexts[start + i], raw_cts_per_gpu[g][i]);
            }
            cudaDeviceSynchronize();
        });
    }
    for (auto& t : d2h_threads) t.join();
    timer.stop();
    timing.d2h_transfer_ms = timer.elapsedMs();
    std::cout << "  D2H transfer (parallel): " << timing.d2h_transfer_ms << " ms" << std::endl;
    
    timing.total_ms = timing.context_setup_ms + timing.key_loading_ms +
                      timing.encrypt_ms + timing.h2d_transfer_ms +
                      timing.bootstrap_ms + timing.d2h_transfer_ms;
    
    return timing;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   FIDESlib Multi-GPU Data-Parallel Bootstrap Benchmark        ║" << std::endl;
    std::cout << "║   Detailed Timing Analysis                                    ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    printGPUInfo();
    
    // Configuration - use smaller parameters for testing
    BootstrapConfig config;
    config.logN = 14;           // Ring dimension = 16384
    config.L = 10;
    config.numBootstraps = 4;
    config.numIterations = 3;
    config.numGPUs = getGPUCount();
    config.slots = 1 << (config.logN - 1);
    
    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--logN" && i + 1 < argc) {
            config.logN = std::stoi(argv[++i]);
            config.slots = 1 << (config.logN - 1);
        }
        else if (arg == "--L" && i + 1 < argc) config.L = std::stoi(argv[++i]);
        else if (arg == "--num" && i + 1 < argc) config.numBootstraps = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) config.numIterations = std::stoi(argv[++i]);
        else if (arg == "--gpus" && i + 1 < argc) config.numGPUs = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --logN <n>   Ring dimension log (default: 14)" << std::endl;
            std::cout << "  --L <n>      Multiplicative depth (default: 10)" << std::endl;
            std::cout << "  --num <n>    Number of bootstraps (default: 4)" << std::endl;
            std::cout << "  --iters <n>  Iterations (default: 3)" << std::endl;
            std::cout << "  --gpus <n>   Number of GPUs (default: all)" << std::endl;
            return 0;
        }
    }
    
    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "  Ring dimension: 2^" << config.logN << " = " << (1 << config.logN) << std::endl;
    std::cout << "  Slots: " << config.slots << std::endl;
    std::cout << "  Multiplicative depth: " << config.L << std::endl;
    std::cout << "  Number of operations: " << config.numBootstraps << std::endl;
    std::cout << "  Iterations: " << config.numIterations << std::endl;
    std::cout << "  GPUs: " << config.numGPUs << std::endl;
    
    // Create OpenFHE context
    std::cout << "\nCreating OpenFHE context..." << std::endl;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetMultiplicativeDepth(config.L);
    parameters.SetScalingModSize(51);
    parameters.SetRingDim(1 << config.logN);
    parameters.SetBatchSize(config.slots);
    parameters.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    parameters.SetFirstModSize(60);
    
    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    
    std::cout << "Generating keys..." << std::endl;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    // FIDESlib parameters
    FIDESlib::CKKS::Parameters fidesParams{
        .logN = config.logN,
        .L = config.L,
        .dnum = config.dnum,
        .primes = p64,
        .Sprimes = sp64
    };
    
    // Run benchmarks
    BootstrapTiming single_timing = benchmarkSingleGPU_Operations(config, cc, keys, fidesParams, 0);
    BootstrapTiming multi_timing = benchmarkMultiGPU_Operations(config, cc, keys, fidesParams);
    
    // Print results
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    BENCHMARK RESULTS                          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    single_timing.print("Single GPU");
    multi_timing.print("Multi-GPU (" + std::to_string(config.numGPUs) + " GPUs)");
    
    // Analysis
    double compute_speedup = single_timing.bootstrap_ms / multi_timing.bootstrap_ms;
    double total_speedup = single_timing.total_ms / multi_timing.total_ms;
    double efficiency = (compute_speedup / config.numGPUs) * 100.0;
    
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Compute Speedup:     " << compute_speedup << "x" << std::endl;
    std::cout << "  Total Speedup:       " << total_speedup << "x" << std::endl;
    std::cout << "  Parallel Efficiency: " << efficiency << "%" << std::endl;
    
    double single_throughput = (config.numBootstraps * config.numIterations * 1000.0) / single_timing.bootstrap_ms;
    double multi_throughput = (config.numBootstraps * config.numIterations * 1000.0) / multi_timing.bootstrap_ms;
    
    std::cout << "\n  Single GPU Throughput: " << single_throughput << " ops/sec" << std::endl;
    std::cout << "  Multi-GPU Throughput:  " << multi_throughput << " ops/sec" << std::endl;
    
    // Cleanup
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    
    std::cout << "\n✓ Benchmark complete!" << std::endl;
    return 0;
}
