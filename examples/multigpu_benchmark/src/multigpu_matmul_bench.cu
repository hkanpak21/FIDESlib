/**
 * @file multigpu_matmul_bench.cu
 * @brief Multi-GPU HE Matrix-Vector Multiplication Benchmark with Detailed Timing
 * 
 * This benchmark measures:
 * 1. Single GPU baseline performance
 * 2. Multi-GPU (data parallel) performance
 * 3. Detailed timing breakdown for each phase
 * 4. Speedup and efficiency analysis
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
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

// ============================================================================
// Configuration
// ============================================================================
struct BenchConfig {
    int logN = 14;           // Ring dimension = 16384
    int L = 10;              // Multiplicative depth
    int dnum = 1;            // Decomposition number
    int matrixRows = 4096;   // Matrix dimension  
    int matrixCols = 64;     // Matrix columns
    int numOperations = 16;  // Number of operations per batch
    int numIterations = 5;   // Number of benchmark iterations
    int numGPUs = 2;         // Number of GPUs to use
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
    
    double elapsedUs() const {
        return std::chrono::duration<double, std::micro>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

class CudaTimer {
public:
    CudaTimer(int device = 0) : device_(device) {
        cudaSetDevice(device);
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaSetDevice(device_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaSetDevice(device_);
        cudaEventRecord(start_, stream);
    }
    
    void stop(cudaStream_t stream = 0) {
        cudaSetDevice(device_);
        cudaEventRecord(stop_, stream);
    }
    
    float elapsedMs() {
        cudaSetDevice(device_);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    int device_;
    cudaEvent_t start_, stop_;
};

// Timing breakdown structure
struct TimingBreakdown {
    double context_setup_ms = 0;
    double key_loading_ms = 0;
    double encrypt_ms = 0;
    double h2d_transfer_ms = 0;  // Host to device
    double compute_ms = 0;
    double d2h_transfer_ms = 0;  // Device to host
    double decrypt_ms = 0;
    double total_ms = 0;
    
    void print(const std::string& label) const {
        std::cout << "\n--- " << label << " Timing Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Context setup:    " << std::setw(10) << context_setup_ms << " ms" << std::endl;
        std::cout << "  Key loading:      " << std::setw(10) << key_loading_ms << " ms" << std::endl;
        std::cout << "  Encryption:       " << std::setw(10) << encrypt_ms << " ms" << std::endl;
        std::cout << "  H2D transfer:     " << std::setw(10) << h2d_transfer_ms << " ms" << std::endl;
        std::cout << "  GPU Compute:      " << std::setw(10) << compute_ms << " ms" << std::endl;
        std::cout << "  D2H transfer:     " << std::setw(10) << d2h_transfer_ms << " ms" << std::endl;
        std::cout << "  Decryption:       " << std::setw(10) << decrypt_ms << " ms" << std::endl;
        std::cout << "  --------------------------------" << std::endl;
        std::cout << "  TOTAL:            " << std::setw(10) << total_ms << " ms" << std::endl;
    }
};

// ============================================================================
// Prime records for CKKS
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
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")"
                  << " - " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB VRAM"
                  << std::endl;
    }
    std::cout << std::endl;
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createOpenFHEContext(const BenchConfig& config) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetMultiplicativeDepth(config.L);
    parameters.SetScalingModSize(51);
    parameters.SetRingDim(1 << config.logN);
    parameters.SetBatchSize(1 << (config.logN - 1));
    parameters.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    parameters.SetFirstModSize(60);
    
    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    
    return cc;
}

// ============================================================================
// Single GPU Benchmark
// ============================================================================
TimingBreakdown benchmarkSingleGPU(
    const BenchConfig& config,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
    FIDESlib::CKKS::Parameters& fidesParams,
    int gpuId = 0) 
{
    TimingBreakdown timing;
    Timer timer;
    
    std::cout << "\n>>> Single GPU Benchmark (GPU " << gpuId << ") <<<" << std::endl;
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
    int slots = 1 << (config.logN - 1);
    std::vector<std::vector<double>> test_vectors(config.numOperations);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts(config.numOperations);
    
    timer.start();
    for (int i = 0; i < config.numOperations; ++i) {
        test_vectors[i].resize(slots);
        for (int j = 0; j < slots; ++j) {
            test_vectors[i][j] = 1.0 + 0.001 * i + 0.0001 * j;
        }
        auto pt = cc->MakeCKKSPackedPlaintext(test_vectors[i]);
        ciphertexts[i] = cc->Encrypt(keys.publicKey, pt);
    }
    timer.stop();
    timing.encrypt_ms = timer.elapsedMs();
    std::cout << "  Encryption (" << config.numOperations << " cts): " << timing.encrypt_ms << " ms" << std::endl;
    
    // 4. Transfer to GPU
    timer.start();
    std::vector<FIDESlib::CKKS::RawCipherText> raw_cts(config.numOperations);
    std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> gpu_cts;
    gpu_cts.reserve(config.numOperations);
    
    for (int i = 0; i < config.numOperations; ++i) {
        raw_cts[i] = FIDESlib::CKKS::GetRawCipherText(cc, ciphertexts[i]);
        gpu_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(gpu_cc, raw_cts[i]));
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.h2d_transfer_ms = timer.elapsedMs();
    std::cout << "  H2D transfer: " << timing.h2d_transfer_ms << " ms" << std::endl;
    
    // 5. GPU Compute - simulate matmul operations (mult + add)
    CudaTimer cuda_timer(gpuId);
    cuda_timer.start();
    
    // Perform operations: square each ciphertext (simulates matmul inner product)
    for (int iter = 0; iter < config.numIterations; ++iter) {
        for (int i = 0; i < config.numOperations; ++i) {
            gpu_cts[i]->square(gpu_cc.GetEvalKey());
        }
    }
    
    cuda_timer.stop();
    timing.compute_ms = cuda_timer.elapsedMs();
    std::cout << "  GPU Compute (" << config.numIterations << " iters x " 
              << config.numOperations << " ops): " << timing.compute_ms << " ms" << std::endl;
    
    // 6. Transfer back to host
    timer.start();
    for (int i = 0; i < config.numOperations; ++i) {
        gpu_cts[i]->store(gpu_cc, raw_cts[i]);
        FIDESlib::CKKS::GetOpenFHECipherText(ciphertexts[i], raw_cts[i]);
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.d2h_transfer_ms = timer.elapsedMs();
    std::cout << "  D2H transfer: " << timing.d2h_transfer_ms << " ms" << std::endl;
    
    // 7. Decryption (optional, for verification)
    timer.start();
    lbcrypto::Plaintext result_pt;
    cc->Decrypt(keys.secretKey, ciphertexts[0], &result_pt);
    timer.stop();
    timing.decrypt_ms = timer.elapsedMs();
    std::cout << "  Decryption (1 ct): " << timing.decrypt_ms << " ms" << std::endl;
    
    timing.total_ms = timing.context_setup_ms + timing.key_loading_ms + timing.encrypt_ms +
                      timing.h2d_transfer_ms + timing.compute_ms + timing.d2h_transfer_ms;
    
    return timing;
}

// ============================================================================
// Multi-GPU Benchmark (Data Parallel)
// ============================================================================
TimingBreakdown benchmarkMultiGPU(
    const BenchConfig& config,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
    FIDESlib::CKKS::Parameters& fidesParams)
{
    TimingBreakdown timing;
    Timer timer;
    int numGPUs = std::min(config.numGPUs, getGPUCount());
    
    std::cout << "\n>>> Multi-GPU Benchmark (" << numGPUs << " GPUs) <<<" << std::endl;
    
    // 1. Create contexts on all GPUs
    timer.start();
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Context>> gpu_contexts;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        gpu_contexts.push_back(std::make_unique<FIDESlib::CKKS::Context>(p, std::vector<int>{g}));
    }
    
    // Sync all GPUs
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    timer.stop();
    timing.context_setup_ms = timer.elapsedMs();
    std::cout << "  Context setup (" << numGPUs << " GPUs): " << timing.context_setup_ms << " ms" << std::endl;
    
    // 2. Load keys on all GPUs
    timer.start();
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(*gpu_contexts[g]);
        eval_key_gpu.Initialize(*gpu_contexts[g], eval_key_raw);
        gpu_contexts[g]->AddEvalKey(std::move(eval_key_gpu));
    }
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    timer.stop();
    timing.key_loading_ms = timer.elapsedMs();
    std::cout << "  Key loading (" << numGPUs << " GPUs): " << timing.key_loading_ms << " ms" << std::endl;
    
    // 3. Prepare test data (same as single GPU)
    int slots = 1 << (config.logN - 1);
    std::vector<std::vector<double>> test_vectors(config.numOperations);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts(config.numOperations);
    
    timer.start();
    for (int i = 0; i < config.numOperations; ++i) {
        test_vectors[i].resize(slots);
        for (int j = 0; j < slots; ++j) {
            test_vectors[i][j] = 1.0 + 0.001 * i + 0.0001 * j;
        }
        auto pt = cc->MakeCKKSPackedPlaintext(test_vectors[i]);
        ciphertexts[i] = cc->Encrypt(keys.publicKey, pt);
    }
    timer.stop();
    timing.encrypt_ms = timer.elapsedMs();
    std::cout << "  Encryption (" << config.numOperations << " cts): " << timing.encrypt_ms << " ms" << std::endl;
    
    // 4. Distribute and transfer to GPUs
    timer.start();
    int opsPerGPU = config.numOperations / numGPUs;
    std::vector<std::vector<FIDESlib::CKKS::RawCipherText>> raw_cts_per_gpu(numGPUs);
    std::vector<std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>>> gpu_cts_per_gpu(numGPUs);
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        int start = g * opsPerGPU;
        int end = (g == numGPUs - 1) ? config.numOperations : start + opsPerGPU;
        
        for (int i = start; i < end; ++i) {
            raw_cts_per_gpu[g].push_back(FIDESlib::CKKS::GetRawCipherText(cc, ciphertexts[i]));
            gpu_cts_per_gpu[g].push_back(
                std::make_unique<FIDESlib::CKKS::Ciphertext>(*gpu_contexts[g], raw_cts_per_gpu[g].back()));
        }
    }
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    timer.stop();
    timing.h2d_transfer_ms = timer.elapsedMs();
    std::cout << "  H2D transfer (distributed): " << timing.h2d_transfer_ms << " ms" << std::endl;
    
    // 5. Parallel GPU Compute using threads
    std::vector<CudaTimer> cuda_timers;
    for (int g = 0; g < numGPUs; ++g) {
        cuda_timers.emplace_back(g);
    }
    
    Timer overall_compute;
    overall_compute.start();
    
    std::vector<std::thread> threads;
    std::atomic<int> completed{0};
    
    for (int g = 0; g < numGPUs; ++g) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            
            for (int iter = 0; iter < config.numIterations; ++iter) {
                for (auto& ct : gpu_cts_per_gpu[g]) {
                    ct->square(gpu_contexts[g]->GetEvalKey());
                }
            }
            
            cudaDeviceSynchronize();
            completed++;
        });
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    overall_compute.stop();
    timing.compute_ms = overall_compute.elapsedMs();
    std::cout << "  GPU Compute (parallel, " << config.numIterations << " iters x " 
              << config.numOperations << " ops): " << timing.compute_ms << " ms" << std::endl;
    
    // 6. Transfer back to host
    timer.start();
    int idx = 0;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        for (size_t i = 0; i < gpu_cts_per_gpu[g].size(); ++i) {
            gpu_cts_per_gpu[g][i]->store(*gpu_contexts[g], raw_cts_per_gpu[g][i]);
            FIDESlib::CKKS::GetOpenFHECipherText(ciphertexts[idx++], raw_cts_per_gpu[g][i]);
        }
    }
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    timer.stop();
    timing.d2h_transfer_ms = timer.elapsedMs();
    std::cout << "  D2H transfer: " << timing.d2h_transfer_ms << " ms" << std::endl;
    
    timing.total_ms = timing.context_setup_ms + timing.key_loading_ms + timing.encrypt_ms +
                      timing.h2d_transfer_ms + timing.compute_ms + timing.d2h_transfer_ms;
    
    return timing;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   FIDESlib Multi-GPU HE MatMul Benchmark                      ║" << std::endl;
    std::cout << "║   Detailed Timing Analysis                                    ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    printGPUInfo();
    
    // Configuration
    BenchConfig config;
    config.numGPUs = getGPUCount();
    
    // Parse command line args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--logN" && i + 1 < argc) config.logN = std::stoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc) config.L = std::stoi(argv[++i]);
        else if (arg == "--ops" && i + 1 < argc) config.numOperations = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) config.numIterations = std::stoi(argv[++i]);
        else if (arg == "--gpus" && i + 1 < argc) config.numGPUs = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --logN <n>   Ring dimension log (default: 14)" << std::endl;
            std::cout << "  --L <n>      Multiplicative depth (default: 10)" << std::endl;
            std::cout << "  --ops <n>    Operations per batch (default: 16)" << std::endl;
            std::cout << "  --iters <n>  Benchmark iterations (default: 5)" << std::endl;
            std::cout << "  --gpus <n>   Number of GPUs (default: all)" << std::endl;
            return 0;
        }
    }
    
    std::cout << "\n=== Benchmark Configuration ===" << std::endl;
    std::cout << "  Ring dimension: 2^" << config.logN << " = " << (1 << config.logN) << std::endl;
    std::cout << "  Multiplicative depth: " << config.L << std::endl;
    std::cout << "  Operations per batch: " << config.numOperations << std::endl;
    std::cout << "  Iterations: " << config.numIterations << std::endl;
    std::cout << "  GPUs: " << config.numGPUs << std::endl;
    
    // Create OpenFHE context
    std::cout << "\nCreating OpenFHE context..." << std::endl;
    auto cc = createOpenFHEContext(config);
    
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
    TimingBreakdown single_gpu_timing = benchmarkSingleGPU(config, cc, keys, fidesParams, 0);
    TimingBreakdown multi_gpu_timing = benchmarkMultiGPU(config, cc, keys, fidesParams);
    
    // Print summary
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    BENCHMARK RESULTS                          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    single_gpu_timing.print("Single GPU");
    multi_gpu_timing.print("Multi-GPU (" + std::to_string(config.numGPUs) + " GPUs)");
    
    // Calculate speedup
    double compute_speedup = single_gpu_timing.compute_ms / multi_gpu_timing.compute_ms;
    double total_speedup = single_gpu_timing.total_ms / multi_gpu_timing.total_ms;
    double efficiency = (compute_speedup / config.numGPUs) * 100.0;
    
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Compute Speedup:     " << compute_speedup << "x" << std::endl;
    std::cout << "  Total Speedup:       " << total_speedup << "x" << std::endl;
    std::cout << "  Parallel Efficiency: " << efficiency << "%" << std::endl;
    
    double single_throughput = (config.numOperations * config.numIterations * 1000.0) / single_gpu_timing.compute_ms;
    double multi_throughput = (config.numOperations * config.numIterations * 1000.0) / multi_gpu_timing.compute_ms;
    
    std::cout << "\n  Single GPU Throughput: " << single_throughput << " ops/sec" << std::endl;
    std::cout << "  Multi-GPU Throughput:  " << multi_throughput << " ops/sec" << std::endl;
    
    // Cleanup
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    
    std::cout << "\n✓ Benchmark complete!" << std::endl;
    return 0;
}
