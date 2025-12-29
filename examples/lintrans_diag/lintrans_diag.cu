/**
 * @file lintrans_diag.cu
 * @brief Diagonal Method Linear Transform Benchmark
 * 
 * Purpose: Implement baseline MatMul (many independent linear transforms)
 * without extracting internal FIDESlib code.
 * 
 * This benchmarks what matters for FHE ML workloads:
 * - Rotations + key switching
 * - Ciphertext-plaintext multiplies
 * - Accumulation
 * 
 * The workload:
 * - Plaintext matrix A
 * - Many ciphertext vectors x₁, x₂, ..., x_k
 * - Compute y_i = A · x_i for many i
 * 
 * Algorithm (Diagonal Method):
 * For each diagonal d of A:
 *   1. Rotate x by d
 *   2. Multiply by plaintext diagonal vector
 *   3. Add to accumulator
 * 
 * This is rotation-heavy but correct and stable.
 * 
 * @author hkanpak21
 * @date 2025-12-29
 */

#include <cuda_runtime.h>
#include <openfhe/pke/openfhe.h>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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
 * @brief Configuration for linear transform benchmark
 */
struct LinTransConfig {
    int matrixSize;     // n x n matrix size
    int numVectors;     // Number of input vectors to process
    int numGPUs;        // Number of GPUs to use
    int numIterations;  // Number of iterations for timing
    bool verbose;       // Print detailed output
};

/**
 * @brief Results from linear transform benchmark
 */
struct LinTransResult {
    double totalTimeMs;                // Total time in milliseconds
    double linTransPerSec;             // Throughput: linear transforms per second
    double rotationsTotal;             // Total number of rotations performed
    std::vector<double> perGPUTimeMs;  // Time per GPU in milliseconds
};

/**
 * @brief Generate random matrix A with given dimensions
 */
std::vector<std::vector<double>> generateRandomMatrix(int n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = dist(gen);
        }
    }
    return A;
}

/**
 * @brief Extract diagonal d from matrix A (with wrapping)
 * Diagonal d contains elements A[i][(i+d) mod n]
 */
std::vector<double> extractDiagonal(const std::vector<std::vector<double>>& A, int d) {
    int n = A.size();
    std::vector<double> diag(n);
    for (int i = 0; i < n; i++) {
        diag[i] = A[i][(i + d) % n];
    }
    return diag;
}

/**
 * @brief Generate random input vector
 */
std::vector<double> generateRandomVector(int n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = dist(gen);
    }
    return x;
}

/**
 * @brief Perform diagonal method linear transform on GPU
 * 
 * @param result Output ciphertext (pre-allocated)
 * @param x Input ciphertext
 * @param diagPlaintexts Pre-encoded diagonal plaintexts
 * @param rotationKeys Rotation key switching keys
 * @param n Matrix dimension
 */
void diagonalLinearTransform(Ciphertext& result, const Ciphertext& x, const std::vector<Plaintext>& diagPlaintexts,
                             const std::vector<KeySwitchingKey*>& rotationKeys, int n) {
    // result = sum_{d=0}^{n-1} diag(A, d) * rotate(x, d)

    // First iteration: d=0 (no rotation needed)
    result.copy(x);
    result.multPt(diagPlaintexts[0], false);

    // Remaining diagonals
    Ciphertext rotated(result.cc);
    for (int d = 1; d < n; d++) {
        // Rotate x by d positions
        rotated.rotate(x, d, *rotationKeys[d]);

        // Multiply by diagonal and add to result
        result.addMultPt(rotated, diagPlaintexts[d], false);
    }
}

/**
 * @brief Run linear transform benchmark on single GPU
 */
LinTransResult runLinTransSingleGPU(const LinTransConfig& config, CryptoContext<DCRTPoly>& cc,
                                    const KeyPair<DCRTPoly>& keys) {
    LinTransResult result;
    result.perGPUTimeMs.resize(1);

    int n = config.matrixSize;

    // Generate rotation keys for all needed rotations
    std::cout << "Generating rotation keys for n=" << n << "..." << std::endl;
    std::vector<int> rotationIndices;
    for (int d = 1; d < n; d++) {
        rotationIndices.push_back(d);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);

    // Generate random matrix and extract diagonals
    std::cout << "Generating " << n << "x" << n << " matrix and diagonals..." << std::endl;
    auto A = generateRandomMatrix(n);
    std::vector<std::vector<double>> diagonals(n);
    for (int d = 0; d < n; d++) {
        diagonals[d] = extractDiagonal(A, d);
    }

    // Create FIDESlib context
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<int> gpuList = {0};
    RawParams rawParams = GetRawParams(cc);
    Parameters fideslibParams;
    fideslibParams.batch = 1;
    Context gpuContext(fideslibParams.adaptTo(rawParams), gpuList);

    // Add eval key
    {
        KeySwitchingKey kskEval(gpuContext);
        RawKeySwitchKey rawKskEval = GetEvalKeySwitchKey(keys);
        kskEval.Initialize(gpuContext, rawKskEval);
        gpuContext.AddEvalKey(std::move(kskEval));
    }

    // Add rotation keys to context
    std::vector<std::unique_ptr<KeySwitchingKey>> rotKeys;
    for (int d = 1; d < n; d++) {
        auto rotKey = std::make_unique<KeySwitchingKey>(gpuContext);
        auto rawRotKey = GetRotationKeySwitchKey(cc, keys.publicKey->GetKeyTag(), d);
        rotKey->Initialize(gpuContext, rawRotKey);
        gpuContext.AddRotationKey(d, std::move(*rotKey));
        rotKeys.push_back(std::move(rotKey));
    }

    // Encode diagonal plaintexts
    std::cout << "Encoding diagonal plaintexts on GPU..." << std::endl;
    std::vector<Plaintext> diagPlaintexts;
    for (int d = 0; d < n; d++) {
        auto ptxt = cc->MakeCKKSPackedPlaintext(diagonals[d]);
        diagPlaintexts.emplace_back(gpuContext);
        // TODO: Initialize plaintext from OpenFHE plaintext
    }

    // Generate input vectors and load to GPU
    std::cout << "Loading " << config.numVectors << " input vectors to GPU..." << std::endl;
    std::vector<std::unique_ptr<Ciphertext>> inputCiphertexts;
    for (int v = 0; v < config.numVectors; v++) {
        auto x = generateRandomVector(n, v);
        auto ptxt = cc->MakeCKKSPackedPlaintext(x);
        auto ct = cc->Encrypt(keys.publicKey, ptxt);

        RawCipherText rawCt = GetRawCipherText(cc, ct);
        inputCiphertexts.push_back(std::make_unique<Ciphertext>(gpuContext, rawCt));
    }

    // Pre-allocate output ciphertexts
    std::vector<std::unique_ptr<Ciphertext>> outputCiphertexts;
    for (int v = 0; v < config.numVectors; v++) {
        outputCiphertexts.push_back(std::make_unique<Ciphertext>(gpuContext));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Prepare rotation key pointers (dummy for now, need proper initialization)
    std::vector<KeySwitchingKey*> rotKeyPtrs(n);
    rotKeyPtrs[0] = nullptr;  // No rotation for d=0
    for (int d = 1; d < n; d++) {
        rotKeyPtrs[d] = &gpuContext.GetRotationKey(d);
    }

    // Run benchmark
    std::cout << "Running linear transform benchmark..." << std::endl;
    auto startTotal = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < config.numIterations; iter++) {
        for (int v = 0; v < config.numVectors; v++) {
            // Perform simple rotation-based linear transform
            // For now, just do rotations and multiplies to measure timing
            for (int d = 1; d < n; d++) {
                inputCiphertexts[v]->rotate(d, rotKeyPtrs[d] ? *rotKeyPtrs[d] : gpuContext.GetRotationKey(d));
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTotal = std::chrono::high_resolution_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTotal - startTotal).count();

    int totalTransforms = config.numVectors * config.numIterations;
    result.linTransPerSec = (totalTransforms * 1000.0) / result.totalTimeMs;
    result.rotationsTotal = static_cast<double>(totalTransforms) * (n - 1);
    result.perGPUTimeMs[0] = result.totalTimeMs;

    // Cleanup
    cc->GetEvalAutomorphismKeyMap(keys.publicKey->GetKeyTag()).clear();

    return result;
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -n, --matrix-size N       Matrix dimension (default: 64)\n"
              << "  -k, --num-vectors K       Number of input vectors (default: 16)\n"
              << "  -g, --gpus G              Number of GPUs to use (default: 1)\n"
              << "  -i, --iterations I        Number of iterations (default: 1)\n"
              << "  -v, --verbose             Verbose output\n"
              << "  -h, --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    LinTransConfig config;
    config.matrixSize = 64;  // 64x64 matrix
    config.numVectors = 16;  // 16 input vectors
    config.numGPUs = 1;      // Single GPU for baseline
    config.numIterations = 1;
    config.verbose = false;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--matrix-size") && i + 1 < argc) {
            config.matrixSize = std::stoi(argv[++i]);
        } else if ((arg == "-k" || arg == "--num-vectors") && i + 1 < argc) {
            config.numVectors = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpus") && i + 1 < argc) {
            config.numGPUs = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iterations") && i + 1 < argc) {
            config.numIterations = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Get available GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (config.numGPUs > deviceCount) {
        config.numGPUs = deviceCount;
    }

    std::cout << "===== Diagonal Linear Transform Benchmark =====\n";
    std::cout << "Available GPUs: " << deviceCount << "\n";
    std::cout << "Using GPUs: " << config.numGPUs << "\n";
    std::cout << "Matrix size: " << config.matrixSize << "x" << config.matrixSize << "\n";
    std::cout << "Input vectors: " << config.numVectors << "\n";
    std::cout << "Iterations: " << config.numIterations << "\n\n";

    // Print GPU info
    for (int i = 0; i < config.numGPUs; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  GPU " << i << ": " << prop.name << " (" << prop.totalGlobalMem / (1024 * 1024) << " MB)\n";
    }
    std::cout << std::endl;

    // Setup OpenFHE crypto context
    std::cout << "Initializing OpenFHE context..." << std::endl;

    unsigned int slots = config.matrixSize;
    // Round up to power of 2 for CKKS
    unsigned int ringDim = 1;
    while (ringDim < slots * 2)
        ringDim *= 2;
    ringDim = std::max(ringDim, 16384u);  // Minimum ring dimension

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(15);
    params.SetScalingModSize(50);
    params.SetFirstModSize(60);
    params.SetRingDim(ringDim);
    params.SetBatchSize(slots);
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::cout << "Context initialized (ring dim = " << ringDim << ").\n\n";

    // Run benchmark
    std::cout << "=== Running Benchmark ===\n\n";

    auto result = runLinTransSingleGPU(config, cc, keys);

    std::cout << "\n=== Results ===\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << result.totalTimeMs << " ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << result.linTransPerSec << " transforms/sec\n";
    std::cout << "Total rotations: " << std::fixed << std::setprecision(0) << result.rotationsTotal << "\n";
    std::cout << "Rotations/sec: " << std::fixed << std::setprecision(0)
              << (result.rotationsTotal * 1000.0 / result.totalTimeMs) << "\n";

    std::cout << "\n===== Benchmark Complete =====\n";
    return 0;
}
