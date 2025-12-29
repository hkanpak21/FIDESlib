/**
 * @file bootstrap_batch_mgpu.cu
 * @brief Multi-GPU Batch Bootstrapping Benchmark
 * 
 * Purpose: Scale throughput by distributing independent ciphertext jobs 
 * across GPUs WITHOUT modifying FIDESlib bootstrapping implementation.
 * 
 * This implements "Mode A" from the project proposal:
 * - Embarrassingly parallel across ciphertexts
 * - N bootstraps on 1 GPU vs N bootstraps on p GPUs
 * 
 * The scheduler:
 * 1. Splits list of ciphertexts across devices
 * 2. Sets cudaSetDevice(gpu) 
 * 3. Runs Bootstrap(ct) on each GPU (one stream per GPU initially)
 * 4. Gathers output and measures time
 * 
 * Output:
 * - Total time
 * - Bootstraps/sec
 * - Per-GPU work distribution
 * 
 * No NCCL required - this is purely task parallel.
 * 
 * @author hkanpak21
 * @date 2025-12-29
 */

#include <cuda_runtime.h>
#include <openfhe/pke/openfhe.h>

#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call)                                                                                    \
    do {                                                                                                    \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess) {                                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) \
                      << std::endl;                                                                         \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

using namespace lbcrypto;
using namespace FIDESlib::CKKS;

/**
 * @brief Configuration for batch bootstrap benchmark
 */
struct BatchBootstrapConfig {
    int numCiphertexts;  // Total number of ciphertexts to bootstrap
    int numGPUs;         // Number of GPUs to use
    int slots;           // Number of slots
    int numIterations;   // Number of iterations for timing
    bool verbose;        // Print detailed output
};

/**
 * @brief Results from batch bootstrap benchmark
 */
struct BatchBootstrapResult {
    double totalTimeMs;                // Total time in milliseconds
    double bootstrapsPerSec;           // Throughput: bootstraps per second
    std::vector<int> perGPUWork;       // Number of ciphertexts per GPU
    std::vector<double> perGPUTimeMs;  // Time per GPU in milliseconds
};

/**
 * @brief Run batch bootstrap on multiple GPUs
 * 
 * @param config Benchmark configuration
 * @param cc OpenFHE crypto context
 * @param keys OpenFHE key pair
 * @return Benchmark results
 */
BatchBootstrapResult runBatchBootstrapMGPU(const BatchBootstrapConfig& config, CryptoContext<DCRTPoly>& cc,
                                           const KeyPair<DCRTPoly>& keys) {
    BatchBootstrapResult result;
    result.perGPUWork.resize(config.numGPUs);
    result.perGPUTimeMs.resize(config.numGPUs, 0.0);

    // Distribute work across GPUs
    int baseWork = config.numCiphertexts / config.numGPUs;
    int remainder = config.numCiphertexts % config.numGPUs;

    for (int g = 0; g < config.numGPUs; g++) {
        result.perGPUWork[g] = baseWork + (g < remainder ? 1 : 0);
    }

    if (config.verbose) {
        std::cout << "Work distribution: ";
        for (int g = 0; g < config.numGPUs; g++) {
            std::cout << "GPU" << g << "=" << result.perGPUWork[g] << " ";
        }
        std::cout << std::endl;
    }

    // Prepare OpenFHE bootstrap keys
    cc->EvalBootstrapSetup({2, 2}, {0, 0}, config.slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, config.slots);

    // Create plaintext and encrypt test ciphertexts
    std::vector<double> testData = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(testData);
    auto baseCt = cc->Encrypt(keys.publicKey, ptxt);

    // Create FIDESlib contexts and ciphertexts for each GPU
    // Each GPU needs its own context and set of ciphertexts
    std::vector<std::unique_ptr<Context>> gpuContexts;
    std::vector<std::vector<std::unique_ptr<Ciphertext>>> gpuCiphertexts;

    RawParams rawParams = GetRawParams(cc);
    Parameters fideslibParams;
    fideslibParams.batch = 1;

    for (int g = 0; g < config.numGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        // Create context for this GPU
        std::vector<int> gpuList = {g};
        gpuContexts.push_back(std::make_unique<Context>(fideslibParams.adaptTo(rawParams), gpuList));

        // Add bootstrap precomputation for this context
        AddBootstrapPrecomputation(cc, keys, config.slots, *gpuContexts[g]);

        // Add eval key
        {
            KeySwitchingKey kskEval(*gpuContexts[g]);
            RawKeySwitchKey rawKskEval = GetEvalKeySwitchKey(keys);
            kskEval.Initialize(*gpuContexts[g], rawKskEval);
            gpuContexts[g]->AddEvalKey(std::move(kskEval));
        }

        // Create ciphertexts for this GPU
        gpuCiphertexts.emplace_back();
        RawCipherText rawCt = GetRawCipherText(cc, baseCt);

        for (int i = 0; i < result.perGPUWork[g]; i++) {
            gpuCiphertexts[g].push_back(std::make_unique<Ciphertext>(*gpuContexts[g], rawCt));
        }
    }

    // Synchronize all GPUs before timing
    for (int g = 0; g < config.numGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run benchmark
    std::mutex mtx;
    auto startTotal = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(config.numGPUs)
    for (int g = 0; g < config.numGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(g));

        auto startGPU = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < config.numIterations; iter++) {
            for (int i = 0; i < result.perGPUWork[g]; i++) {
                Bootstrap(*gpuCiphertexts[g][i], config.slots);

                // Restore ciphertext level for next iteration
                int initLevel = gpuContexts[g]->GetBootPrecomputation(config.slots).StC.at(0).A.at(0).c0.getLevel();
                gpuCiphertexts[g][i]->c0.grow(initLevel);
                gpuCiphertexts[g][i]->c1.grow(initLevel);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto endGPU = std::chrono::high_resolution_clock::now();
        double gpuTime = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();

        std::lock_guard<std::mutex> lock(mtx);
        result.perGPUTimeMs[g] = gpuTime;
    }

    auto endTotal = std::chrono::high_resolution_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTotal - startTotal).count();

    int totalBootstraps = config.numCiphertexts * config.numIterations;
    result.bootstrapsPerSec = (totalBootstraps * 1000.0) / result.totalTimeMs;

    // Cleanup OpenFHE rotation keys
    cc->GetEvalAutomorphismKeyMap(keys.publicKey->GetKeyTag()).clear();

    return result;
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -n, --num-ciphertexts N   Number of ciphertexts (default: 8)\n"
              << "  -g, --gpus G              Number of GPUs to use (default: all)\n"
              << "  -s, --slots S             Number of slots (default: 64)\n"
              << "  -i, --iterations I        Number of iterations (default: 1)\n"
              << "  -v, --verbose             Verbose output\n"
              << "  -h, --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    BatchBootstrapConfig config;
    config.numCiphertexts = 8;
    config.slots = 64;
    config.numIterations = 1;
    config.verbose = false;

    // Get available GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    config.numGPUs = deviceCount;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--num-ciphertexts") && i + 1 < argc) {
            config.numCiphertexts = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpus") && i + 1 < argc) {
            config.numGPUs = std::stoi(argv[++i]);
            if (config.numGPUs > deviceCount)
                config.numGPUs = deviceCount;
        } else if ((arg == "-s" || arg == "--slots") && i + 1 < argc) {
            config.slots = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iterations") && i + 1 < argc) {
            config.numIterations = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::cout << "===== Multi-GPU Batch Bootstrap Benchmark =====\n";
    std::cout << "Available GPUs: " << deviceCount << "\n";
    std::cout << "Using GPUs: " << config.numGPUs << "\n";
    std::cout << "Ciphertexts: " << config.numCiphertexts << "\n";
    std::cout << "Slots: " << config.slots << "\n";
    std::cout << "Iterations: " << config.numIterations << "\n\n";

    // Print GPU info
    for (int i = 0; i < config.numGPUs; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  GPU " << i << ": " << prop.name << " (" << prop.totalGlobalMem / (1024 * 1024) << " MB)\n";
    }
    std::cout << std::endl;

    // Setup OpenFHE crypto context with bootstrap parameters
    std::cout << "Initializing OpenFHE context..." << std::endl;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(29);
    params.SetScalingModSize(59);
    params.SetFirstModSize(60);
    params.SetRingDim(1 << 16);  // N = 65536
    params.SetBatchSize(config.slots);
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetScalingTechnique(FLEXIBLEAUTOEXT);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::cout << "Context initialized.\n\n";

    // Run benchmarks for different GPU counts
    std::cout << "=== Running Benchmarks ===\n\n";

    std::vector<std::pair<int, BatchBootstrapResult>> results;

    for (int numGPUs = 1; numGPUs <= config.numGPUs; numGPUs *= 2) {
        BatchBootstrapConfig runConfig = config;
        runConfig.numGPUs = numGPUs;

        std::cout << "Testing with " << numGPUs << " GPU(s)..." << std::endl;

        auto result = runBatchBootstrapMGPU(runConfig, cc, keys);
        results.emplace_back(numGPUs, result);

        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.totalTimeMs << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << result.bootstrapsPerSec
                  << " bootstraps/sec\n";

        if (config.verbose) {
            for (int g = 0; g < numGPUs; g++) {
                std::cout << "    GPU " << g << ": " << result.perGPUWork[g] << " cts, " << result.perGPUTimeMs[g]
                          << " ms\n";
            }
        }
        std::cout << std::endl;
    }
    // Also test with max GPUs if not a power of 2
    if (config.numGPUs > 1 && (config.numGPUs & (config.numGPUs - 1)) != 0) {
        std::cout << "Testing with " << config.numGPUs << " GPU(s)..." << std::endl;
        auto result = runBatchBootstrapMGPU(config, cc, keys);
        results.emplace_back(config.numGPUs, result);

        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.totalTimeMs << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << result.bootstrapsPerSec
                  << " bootstraps/sec\n\n";
    }

    // Summary table
    std::cout << "=== Summary ===\n\n";
    std::cout << std::setw(8) << "GPUs" << std::setw(15) << "Time (ms)" << std::setw(15) << "Boot/sec" << std::setw(12)
              << "Speedup"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    double baseTime = results[0].second.totalTimeMs;
    for (const auto& [numGPUs, result] : results) {
        double speedup = baseTime / result.totalTimeMs;
        std::cout << std::setw(8) << numGPUs << std::setw(15) << std::fixed << std::setprecision(2)
                  << result.totalTimeMs << std::setw(15) << std::fixed << std::setprecision(2)
                  << result.bootstrapsPerSec << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";
    }
    std::cout << std::endl;

    std::cout << "===== Benchmark Complete =====\n";
    return 0;
}
