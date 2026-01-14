/**
 * @file multigpu_profiled_bench.cu
 * @brief Multi-GPU Benchmark with Comprehensive Performance Monitoring
 * 
 * This benchmark includes:
 * - Detailed per-phase timing breakdown
 * - NVTX markers for Nsight Systems profiling
 * - Memory bandwidth tracking
 * - GPU utilization monitoring
 * - Latency bottleneck analysis
 * - CSV export for further analysis
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
                  << " - " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB VRAM"
                  << ", SM Count: " << prop.multiProcessorCount
                  << std::endl;
    }
    std::cout << std::endl;
}

// Calculate estimated ciphertext size for bandwidth tracking
size_t estimateCiphertextSize(int logN, int L) {
    // Each ciphertext has 2 polynomials, each with (L+1) limbs, each limb has N coefficients of 64 bits
    return 2ULL * (L + 1) * (1ULL << logN) * sizeof(uint64_t);
}

// ============================================================================
// Single GPU Benchmark with Performance Monitoring
// ============================================================================
void benchSingleGPU(int numOps, int numIters, int logN, int L, 
                    PerfMon::PerformanceMonitor& monitor) {
    NVTX_SCOPE_COLOR("Single GPU Benchmark", PerfMon::NVTXColors::Blue);
    
    std::cout << "\n>>> Single GPU (GPU 0): " << numOps << " ops x " << numIters << " iters <<<" << std::endl;
    
    // Track ciphertext size for bandwidth calculations
    size_t ctSize = estimateCiphertextSize(logN, L);
    
    // -------------------------------------------------------------------------
    // Phase 1: OpenFHE Context Setup
    // -------------------------------------------------------------------------
    {
        NVTX_SCOPE_COLOR("OpenFHE Setup", PerfMon::NVTXColors::Yellow);
        monitor.startPhase(PerfMon::Phase::ContextSetup, 0);
    }
    
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
    
    monitor.endPhase(PerfMon::Phase::ContextSetup, 0);
    
    // -------------------------------------------------------------------------
    // Phase 2: Key Generation
    // -------------------------------------------------------------------------
    monitor.startPhase(PerfMon::Phase::KeyGeneration, 0);
    
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    monitor.endPhase(PerfMon::Phase::KeyGeneration, 0);
    
    // -------------------------------------------------------------------------
    // Phase 3: FIDESlib Context Setup
    // -------------------------------------------------------------------------
    monitor.startPhase(PerfMon::Phase::ContextSetup, 0);
    
    cudaSetDevice(0);
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    FIDESlib::CKKS::Context gpu_cc(p, {0});
    
    monitor.endPhase(PerfMon::Phase::ContextSetup, 0);
    
    // -------------------------------------------------------------------------
    // Phase 4: Key Loading to GPU
    // -------------------------------------------------------------------------
    monitor.startPhase(PerfMon::Phase::KeyLoading, 0);
    
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(gpu_cc);
    eval_key_gpu.Initialize(gpu_cc, eval_key_raw);
    gpu_cc.AddEvalKey(std::move(eval_key_gpu));
    
    cudaDeviceSynchronize();
    monitor.endPhase(PerfMon::Phase::KeyLoading, 0);
    
    // Capture memory after setup
    monitor.captureMemory(0);
    
    // -------------------------------------------------------------------------
    // Phase 5: Encryption and H2D Transfer
    // -------------------------------------------------------------------------
    int slots = 1 << (logN - 1);
    std::vector<double> test_data(slots, 1.5);
    auto pt = cc->MakeCKKSPackedPlaintext(test_data);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> gpu_cts;
    
    monitor.startPhase(PerfMon::Phase::Encryption, 0);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> cpu_cts;
    for (int i = 0; i < numOps; ++i) {
        cpu_cts.push_back(cc->Encrypt(keys.publicKey, pt));
    }
    monitor.endPhase(PerfMon::Phase::Encryption, 0, 0, numOps * ctSize, numOps);
    
    monitor.startPhase(PerfMon::Phase::H2DTransfer, 0);
    for (int i = 0; i < numOps; ++i) {
        auto ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, cpu_cts[i]);
        gpu_cts.push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(gpu_cc, ct_raw));
    }
    cudaDeviceSynchronize();
    monitor.endPhase(PerfMon::Phase::H2DTransfer, 0, 0, numOps * ctSize, numOps);
    
    // -------------------------------------------------------------------------
    // Warmup
    // -------------------------------------------------------------------------
    std::cout << "  Warming up..." << std::endl;
    gpu_cts[0]->mult(*gpu_cts[0], gpu_cc.GetEvalKey());
    cudaDeviceSynchronize();
    
    // -------------------------------------------------------------------------
    // Phase 6: GPU Compute (HE Multiplications)
    // -------------------------------------------------------------------------
    std::cout << "  Running benchmark..." << std::endl;
    
    // Time each iteration separately for variance analysis
    for (int iter = 0; iter < numIters; ++iter) {
        monitor.startPhase(PerfMon::Phase::HEMultiply, 0);
        
        for (int i = 0; i < numOps; ++i) {
            gpu_cts[i]->mult(*gpu_cts[i], gpu_cc.GetEvalKey());
        }
        
        cudaDeviceSynchronize();
        monitor.endPhase(PerfMon::Phase::HEMultiply, 0, 0, numOps * ctSize * 2, numOps);
    }
    
    // Synchronize and collect timing data
    monitor.synchronize();
    
    cc->ClearEvalMultKeys();
}

// ============================================================================
// Multi-GPU Benchmark with Performance Monitoring
// ============================================================================
void benchMultiGPU(int numOps, int numIters, int logN, int L, int numGPUs,
                   PerfMon::PerformanceMonitor& monitor) {
    NVTX_SCOPE_COLOR("Multi-GPU Benchmark", PerfMon::NVTXColors::Green);
    
    std::cout << "\n>>> Multi-GPU (" << numGPUs << " GPUs): " << numOps << " ops x " << numIters << " iters <<<" << std::endl;
    
    size_t ctSize = estimateCiphertextSize(logN, L);
    
    // -------------------------------------------------------------------------
    // Phase 1: OpenFHE Context Setup (Shared)
    // -------------------------------------------------------------------------
    monitor.startPhase(PerfMon::Phase::ContextSetup, 0);
    
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
    
    monitor.endPhase(PerfMon::Phase::ContextSetup, 0);
    
    // -------------------------------------------------------------------------
    // Phase 2: Key Generation
    // -------------------------------------------------------------------------
    monitor.startPhase(PerfMon::Phase::KeyGeneration, 0);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    monitor.endPhase(PerfMon::Phase::KeyGeneration, 0);
    
    // -------------------------------------------------------------------------
    // Phase 3: FIDESlib Context Setup (Per-GPU)
    // -------------------------------------------------------------------------
    FIDESlib::CKKS::Parameters fidesParams{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto p = fidesParams.adaptTo(raw_params);
    
    std::vector<std::unique_ptr<FIDESlib::CKKS::Context>> gpu_contexts;
    
    for (int g = 0; g < numGPUs; ++g) {
        NVTX_SCOPE_COLOR(("Setup GPU " + std::to_string(g)).c_str(), PerfMon::NVTXColors::Yellow);
        
        monitor.startPhase(PerfMon::Phase::ContextSetup, g);
        cudaSetDevice(g);
        gpu_contexts.push_back(std::make_unique<FIDESlib::CKKS::Context>(p, std::vector<int>{g}));
        cudaDeviceSynchronize();
        monitor.endPhase(PerfMon::Phase::ContextSetup, g);
    }
    
    // -------------------------------------------------------------------------
    // Phase 4: Key Loading (Per-GPU)
    // -------------------------------------------------------------------------
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    
    for (int g = 0; g < numGPUs; ++g) {
        monitor.startPhase(PerfMon::Phase::KeyLoading, g);
        cudaSetDevice(g);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(*gpu_contexts[g]);
        eval_key_gpu.Initialize(*gpu_contexts[g], eval_key_raw);
        gpu_contexts[g]->AddEvalKey(std::move(eval_key_gpu));
        cudaDeviceSynchronize();
        monitor.endPhase(PerfMon::Phase::KeyLoading, g);
    }
    
    // Capture memory after setup
    monitor.captureAllGPUMemory(numGPUs);
    
    // -------------------------------------------------------------------------
    // Phase 5: Encryption and Distribution
    // -------------------------------------------------------------------------
    int slots = 1 << (logN - 1);
    std::vector<double> test_data(slots, 1.5);
    auto pt = cc->MakeCKKSPackedPlaintext(test_data);
    
    int opsPerGPU = numOps / numGPUs;
    std::vector<std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>>> gpu_cts(numGPUs);
    
    // Encrypt on CPU
    monitor.startPhase(PerfMon::Phase::Encryption, 0);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> cpu_cts;
    for (int i = 0; i < numOps; ++i) {
        cpu_cts.push_back(cc->Encrypt(keys.publicKey, pt));
    }
    monitor.endPhase(PerfMon::Phase::Encryption, 0, 0, numOps * ctSize, numOps);
    
    // Transfer to each GPU
    int ctIdx = 0;
    for (int g = 0; g < numGPUs; ++g) {
        monitor.startPhase(PerfMon::Phase::H2DTransfer, g);
        cudaSetDevice(g);
        int myOps = (g == numGPUs - 1) ? (numOps - g * opsPerGPU) : opsPerGPU;
        
        for (int i = 0; i < myOps; ++i) {
            auto ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, cpu_cts[ctIdx++]);
            gpu_cts[g].push_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(*gpu_contexts[g], ct_raw));
        }
        cudaDeviceSynchronize();
        monitor.endPhase(PerfMon::Phase::H2DTransfer, g, 0, myOps * ctSize, myOps);
    }
    
    // -------------------------------------------------------------------------
    // Warmup
    // -------------------------------------------------------------------------
    std::cout << "  Warming up GPUs..." << std::endl;
    for (int g = 0; g < numGPUs; ++g) {
        cudaSetDevice(g);
        if (!gpu_cts[g].empty()) {
            gpu_cts[g][0]->mult(*gpu_cts[g][0], gpu_contexts[g]->GetEvalKey());
        }
        cudaDeviceSynchronize();
    }
    
    // Barrier sync before benchmark
    PerfMon::SyncAnalyzer syncAnalyzer(numGPUs);
    syncAnalyzer.barrierSync();
    
    // -------------------------------------------------------------------------
    // Phase 6: Parallel GPU Compute
    // -------------------------------------------------------------------------
    std::cout << "  Running parallel benchmark..." << std::endl;
    
    // Track synchronization overhead
    monitor.startPhase(PerfMon::Phase::Synchronization, 0);
    
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<PerfMon::PerformanceMonitor>> gpuMonitors;
    
    for (int g = 0; g < numGPUs; ++g) {
        gpuMonitors.push_back(std::make_unique<PerfMon::PerformanceMonitor>("GPU " + std::to_string(g)));
    }
    
    for (int g = 0; g < numGPUs; ++g) {
        threads.emplace_back([&, g]() {
            NVTX_SCOPE_COLOR(("Compute GPU " + std::to_string(g)).c_str(), 
                             g == 0 ? PerfMon::NVTXColors::Green : 
                             g == 1 ? PerfMon::NVTXColors::Blue : 
                             g == 2 ? PerfMon::NVTXColors::Cyan :
                                      PerfMon::NVTXColors::Magenta);
            
            cudaSetDevice(g);
            int myOps = gpu_cts[g].size();
            
            for (int iter = 0; iter < numIters; ++iter) {
                gpuMonitors[g]->startPhase(PerfMon::Phase::HEMultiply, g);
                
                for (auto& ct : gpu_cts[g]) {
                    ct->mult(*ct, gpu_contexts[g]->GetEvalKey());
                }
                
                cudaDeviceSynchronize();
                gpuMonitors[g]->endPhase(PerfMon::Phase::HEMultiply, g, 0, 
                                        myOps * ctSize * 2, myOps);
            }
            
            gpuMonitors[g]->synchronize();
        });
    }
    
    for (auto& t : threads) t.join();
    
    monitor.endPhase(PerfMon::Phase::Synchronization, 0);
    
    // Aggregate GPU monitors into main monitor
    for (int g = 0; g < numGPUs; ++g) {
        auto gpuStats = gpuMonitors[g]->getPhaseStatistics();
        for (const auto& [phase, stats] : gpuStats) {
            monitor.addRecord(phase, g, stats.totalMs, stats.totalBytes, stats.count);
        }
    }
    
    monitor.synchronize();
    syncAnalyzer.print();
    
    cc->ClearEvalMultKeys();
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   FIDESlib Multi-GPU Profiled Benchmark                                   ║" << std::endl;
    std::cout << "║   With Comprehensive Performance Monitoring                               ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    printGPUInfo();
    
    int numGPUs = getGPUCount();
    int logN = 14;
    int L = 10;
    int numOps = 32;
    int numIters = 5;
    bool exportCSV = false;
    std::string csvFile = "perf_results.csv";
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--logN" && i+1 < argc) logN = std::stoi(argv[++i]);
        else if (arg == "--L" && i+1 < argc) L = std::stoi(argv[++i]);
        else if (arg == "--ops" && i+1 < argc) numOps = std::stoi(argv[++i]);
        else if (arg == "--iters" && i+1 < argc) numIters = std::stoi(argv[++i]);
        else if (arg == "--gpus" && i+1 < argc) numGPUs = std::min(std::stoi(argv[++i]), getGPUCount());
        else if (arg == "--csv") { exportCSV = true; if (i+1 < argc && argv[i+1][0] != '-') csvFile = argv[++i]; }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --logN <n>     Ring dimension log (default: 14)\n"
                      << "  --L <n>        Multiplicative depth (default: 10)\n"
                      << "  --ops <n>      Number of operations (default: 32)\n"
                      << "  --iters <n>    Number of iterations (default: 5)\n"
                      << "  --gpus <n>     Number of GPUs (default: all)\n"
                      << "  --csv [file]   Export results to CSV\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  logN = " << logN << " (Ring dim = " << (1 << logN) << ")" << std::endl;
    std::cout << "  L = " << L << " (Multiplicative depth)" << std::endl;
    std::cout << "  Operations = " << numOps << std::endl;
    std::cout << "  Iterations = " << numIters << std::endl;
    std::cout << "  GPUs = " << numGPUs << std::endl;
    std::cout << "  Estimated CT size = " << std::fixed << std::setprecision(2) 
              << (estimateCiphertextSize(logN, L) / (1024.0*1024.0)) << " MB" << std::endl;
    
    // =========================================================================
    // Run Single GPU Benchmark
    // =========================================================================
    PerfMon::PerformanceMonitor singleMonitor("Single GPU Benchmark");
    benchSingleGPU(numOps, numIters, logN, L, singleMonitor);
    
    std::cout << "\n";
    singleMonitor.printSummary();
    singleMonitor.printLatencyAnalysis();
    
    if (exportCSV) {
        singleMonitor.exportToCSV("single_gpu_" + csvFile);
    }
    
    // =========================================================================
    // Run Multi-GPU Benchmark
    // =========================================================================
    if (numGPUs > 1) {
        PerfMon::PerformanceMonitor multiMonitor("Multi-GPU Benchmark (" + std::to_string(numGPUs) + " GPUs)");
        benchMultiGPU(numOps, numIters, logN, L, numGPUs, multiMonitor);
        
        std::cout << "\n";
        multiMonitor.printSummary();
        multiMonitor.printPerGPUBreakdown();
        multiMonitor.printLatencyAnalysis();
        
        if (exportCSV) {
            multiMonitor.exportToCSV("multi_gpu_" + csvFile);
        }
    }
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Benchmark Complete!                                                     ║" << std::endl;
    std::cout << "║                                                                           ║" << std::endl;
    std::cout << "║   For detailed profiling with Nsight Systems:                             ║" << std::endl;
    std::cout << "║     nsys profile --trace=cuda,nvtx ./multigpu_profiled_bench              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    return 0;
}
