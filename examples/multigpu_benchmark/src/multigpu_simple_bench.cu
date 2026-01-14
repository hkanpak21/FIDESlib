/**
 * @file multigpu_simple_bench.cu
 * @brief Simplified Multi-GPU Benchmark - Pure GPU Compute Timing
 * 
 * This benchmark focuses on measuring actual GPU compute speedup
 * by measuring only the HE operations on GPU, not CPU overhead.
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

// Prime records for CKKS
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
                  << " (Compute " << prop.major << "." << prop.minor << ")"
                  << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// Single GPU: Run N operations sequentially on one GPU
// ============================================================================
double benchSingleGPU(int numOps, int numIters, int logN, int L) {
    std::cout << "\n>>> Single GPU (GPU 0): " << numOps << " ops x " << numIters << " iters <<<" << std::endl;
    
    // Setup OpenFHE
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
    
    // Setup FIDESlib
    cudaSetDevice(0);
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    FIDESlib::CKKS::Context gpu_cc(p, {0});
    
    // Load eval key
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(gpu_cc);
    eval_key_gpu.Initialize(gpu_cc, eval_key_raw);
    gpu_cc.AddEvalKey(std::move(eval_key_gpu));
    
    // Create ciphertexts
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
    
    // WARMUP: Run one operation to compile CUDA graphs
    std::cout << "  Warming up..." << std::endl;
    gpu_cts[0]->mult(*gpu_cts[0], gpu_cc.GetEvalKey());
    cudaDeviceSynchronize();
    
    // Benchmark - pure GPU compute
    std::cout << "  Running benchmark..." << std::endl;
    Timer timer;
    timer.start();
    
    for (int iter = 0; iter < numIters; ++iter) {
        for (int i = 0; i < numOps; ++i) {
            gpu_cts[i]->mult(*gpu_cts[i], gpu_cc.GetEvalKey());  // Self-mult (square)
        }
    }
    
    cudaDeviceSynchronize();
    timer.stop();
    
    double totalMs = timer.elapsedMs();
    double opsPerSec = (numOps * numIters * 1000.0) / totalMs;
    
    std::cout << "  Total time: " << totalMs << " ms" << std::endl;
    std::cout << "  Throughput: " << opsPerSec << " ops/sec" << std::endl;
    
    cc->ClearEvalMultKeys();
    return totalMs;
}

// ============================================================================
// Multi GPU: Distribute N operations across GPUs (data parallel)
// ============================================================================
double benchMultiGPU(int numOps, int numIters, int logN, int L, int numGPUs) {
    std::cout << "\n>>> Multi-GPU (" << numGPUs << " GPUs): " << numOps << " ops x " << numIters << " iters <<<" << std::endl;
    
    // Setup OpenFHE (shared)
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
    
    // Setup FIDESlib on each GPU - SEQUENTIALLY to avoid conflicts
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Context>> gpu_contexts;
    
    // Create contexts one at a time (important!)
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        gpu_contexts.push_back(std::make_unique<FIDESlib::CKKS::Context>(p, std::vector<int>{g}));
        cudaDeviceSynchronize();
    }
    
    // Load eval keys on each GPU - SEQUENTIALLY
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(*gpu_contexts[g]);
        eval_key_gpu.Initialize(*gpu_contexts[g], eval_key_raw);
        gpu_contexts[g]->AddEvalKey(std::move(eval_key_gpu));
        cudaDeviceSynchronize();
    }
    
    // Distribute ciphertexts to GPUs
    int slots = 1 << (logN - 1);
    std::vector<double> test_data(slots, 1.5);
    auto pt = cc->MakeCKKSPackedPlaintext(test_data);
    
    int opsPerGPU = numOps / numGPUs;
    std::vector<std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>>> gpu_cts(numGPUs);
    
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        int myOps = (g == numGPUs - 1) ? (numOps - g * opsPerGPU) : opsPerGPU;
        
        for (int i = 0; i < myOps; ++i) {
            auto ct = cc->Encrypt(keys.publicKey, pt);
            auto ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
            gpu_cts[g].push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(*gpu_contexts[g], ct_raw));
        }
        cudaDeviceSynchronize();
    }
    
    // WARMUP: Run one operation on each GPU to ensure CUDA graphs are compiled
    std::cout << "  Warming up GPUs..." << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        if (!gpu_cts[g].empty()) {
            gpu_cts[g][0]->mult(*gpu_cts[g][0], gpu_contexts[g]->GetEvalKey());
        }
        cudaDeviceSynchronize();
    }
    
    // Sync all GPUs before benchmark
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    
    // Benchmark - parallel GPU compute
    Timer timer;
    std::vector<double> perGpuTime(numGPUs);
    
    std::cout << "  Running parallel benchmark..." << std::endl;
    timer.start();
    
    std::vector<std::thread> threads;
    for (int g = 0; g < numGPUs; ++g) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            Timer localTimer;
            localTimer.start();
            
            for (int iter = 0; iter < numIters; ++iter) {
                for (auto& ct : gpu_cts[g]) {
                    ct->mult(*ct, gpu_contexts[g]->GetEvalKey());  // Self-mult
                }
            }
            
            cudaDeviceSynchronize();
            localTimer.stop();
            perGpuTime[g] = localTimer.elapsedMs();
        });
    }
    
    for (auto& t : threads) t.join();
    
    timer.stop();
    double totalMs = timer.elapsedMs();
    double opsPerSec = (numOps * numIters * 1000.0) / totalMs;
    
    std::cout << "  Total time (wall): " << totalMs << " ms" << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        std::cout << "    GPU " << g << ": " << perGpuTime[g] << " ms (" 
                  << gpu_cts[g].size() << " ops)" << std::endl;
    }
    std::cout << "  Throughput: " << opsPerSec << " ops/sec" << std::endl;
    
    cc->ClearEvalMultKeys();
    return totalMs;
}

int main(int argc, char* argv[]) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   FIDESlib Multi-GPU Simple Benchmark                         ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    printGPUInfo();
    
    int numGPUs = getGPUCount();
    int logN = 13;     // Ring dim = 8192 (faster for testing)
    int L = 8;         // Multiplicative depth
    int numOps = 16;   // Total operations
    int numIters = 5;  // Iterations
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--logN" && i+1 < argc) logN = std::stoi(argv[++i]);
        else if (arg == "--L" && i+1 < argc) L = std::stoi(argv[++i]);
        else if (arg == "--ops" && i+1 < argc) numOps = std::stoi(argv[++i]);
        else if (arg == "--iters" && i+1 < argc) numIters = std::stoi(argv[++i]);
        else if (arg == "--gpus" && i+1 < argc) numGPUs = std::min(std::stoi(argv[++i]), getGPUCount());
    }
    
    std::cout << "Configuration: logN=" << logN << ", L=" << L 
              << ", ops=" << numOps << ", iters=" << numIters 
              << ", GPUs=" << numGPUs << std::endl;
    
    // Run benchmarks
    double singleTime = benchSingleGPU(numOps, numIters, logN, L);
    double multiTime = benchMultiGPU(numOps, numIters, logN, L, numGPUs);
    
    // Results
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                       RESULTS                                 ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    double speedup = singleTime / multiTime;
    double efficiency = (speedup / numGPUs) * 100.0;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Single GPU time:      " << singleTime << " ms" << std::endl;
    std::cout << "  Multi-GPU time:       " << multiTime << " ms" << std::endl;
    std::cout << "  Speedup:              " << speedup << "x" << std::endl;
    std::cout << "  Parallel Efficiency:  " << efficiency << "%" << std::endl;
    
    double singleThroughput = (numOps * numIters * 1000.0) / singleTime;
    double multiThroughput = (numOps * numIters * 1000.0) / multiTime;
    
    std::cout << "\n  Single GPU Throughput: " << singleThroughput << " ops/sec" << std::endl;
    std::cout << "  Multi-GPU Throughput:  " << multiThroughput << " ops/sec" << std::endl;
    
    std::cout << "\n✓ Done!" << std::endl;
    return 0;
}
