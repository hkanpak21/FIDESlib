/**
 * @file nccl_allreduce.cu
 * @brief NCCL All-Reduce Smoke Test
 * 
 * Purpose: Verify that NCCL works correctly on VALAR HPC cluster.
 * This establishes that multi-GPU communication works before touching CKKS internals.
 * 
 * Test procedure:
 * 1. Initialize NCCL communicator
 * 2. Allocate float buffer on each GPU
 * 3. Perform all-reduce operation (sum)
 * 4. Verify results match expected sum
 * 
 * @author hkanpak21
 * @date 2025-12-29
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

#define NCCL_CHECK(call)                                                                               \
    do {                                                                                               \
        ncclResult_t res = call;                                                                       \
        if (res != ncclSuccess) {                                                                      \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

/**
 * @brief Run NCCL all-reduce test with specified number of GPUs
 * @param numGPUs Number of GPUs to use
 * @param bufferSize Number of float elements per GPU
 * @return true if test passed, false otherwise
 */
bool runNcclAllReduceTest(int numGPUs, size_t bufferSize) {
    printf("\n=== NCCL All-Reduce Test ===\n");
    printf("GPUs: %d, Buffer size: %zu floats (%zu MB)\n", numGPUs, bufferSize,
           bufferSize * sizeof(float) / (1024 * 1024));

    // Initialize NCCL
    std::vector<ncclComm_t> comms(numGPUs);
    std::vector<cudaStream_t> streams(numGPUs);
    std::vector<float*> d_send(numGPUs);
    std::vector<float*> d_recv(numGPUs);
    std::vector<float*> h_send(numGPUs);
    std::vector<float*> h_recv(numGPUs);

    // Create NCCL communicator
    std::vector<int> devs(numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        devs[i] = i;
    }

    NCCL_CHECK(ncclCommInitAll(comms.data(), numGPUs, devs.data()));
    printf("NCCL communicator initialized successfully\n");

    // Allocate buffers and streams for each GPU
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));

        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_send[i], bufferSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recv[i], bufferSize * sizeof(float)));

        h_send[i] = new float[bufferSize];
        h_recv[i] = new float[bufferSize];

        // Initialize send buffer: GPU i fills with value (i + 1)
        for (size_t j = 0; j < bufferSize; j++) {
            h_send[i][j] = static_cast<float>(i + 1);
        }

        CUDA_CHECK(cudaMemcpy(d_send[i], h_send[i], bufferSize * sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("Buffers allocated and initialized\n");

    // Time the all-reduce operation
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    // All-reduce: sum across all GPUs
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < numGPUs; i++) {
        NCCL_CHECK(ncclAllReduce(d_send[i], d_recv[i], bufferSize, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Synchronize all streams
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double bandwidth_gb_s =
        (bufferSize * sizeof(float) * numGPUs * 2.0) / (elapsed_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

    printf("All-reduce completed in %.3f ms (theoretical bandwidth: %.2f GB/s)\n", elapsed_ms, bandwidth_gb_s);

    // Verify results
    // Expected sum = 1 + 2 + ... + numGPUs = numGPUs * (numGPUs + 1) / 2
    float expected = static_cast<float>(numGPUs * (numGPUs + 1) / 2);
    bool passed = true;

    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(h_recv[i], d_recv[i], bufferSize * sizeof(float), cudaMemcpyDeviceToHost));

        // Check first and last elements
        if (h_recv[i][0] != expected || h_recv[i][bufferSize - 1] != expected) {
            printf("GPU %d: FAILED - got %f, expected %f\n", i, h_recv[i][0], expected);
            passed = false;
        }
    }

    if (passed) {
        printf("✅ Verification PASSED - All GPUs have correct result: %f\n", expected);
    } else {
        printf("❌ Verification FAILED\n");
    }

    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_send[i]));
        CUDA_CHECK(cudaFree(d_recv[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        delete[] h_send[i];
        delete[] h_recv[i];
    }
    for (int i = 0; i < numGPUs; i++) {
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }
    printf("Cleanup complete\n");

    return passed;
}

int main(int argc, char* argv[]) {
    printf("===== NCCL All-Reduce Smoke Test =====\n");
    printf("This tests multi-GPU communication on VALAR HPC\n");

    // Get number of available GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n", deviceCount);

    if (deviceCount < 1) {
        printf("No CUDA-capable devices found!\n");
        return EXIT_FAILURE;
    }

    // Print GPU info
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (Compute %d.%d, %zu MB)\n", i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024));
    }

    // Parse command line for number of GPUs to test
    int maxGPUs = deviceCount;
    if (argc > 1) {
        maxGPUs = atoi(argv[1]);
        if (maxGPUs > deviceCount)
            maxGPUs = deviceCount;
    }

    // Buffer size: 1M floats = 4 MB per GPU
    size_t bufferSize = 1024 * 1024;

    // Run tests with increasing GPU counts
    printf("\n=== Running tests with GPU counts: 1");
    for (int g = 2; g <= maxGPUs; g *= 2) {
        printf(", %d", g);
    }
    printf(" ===\n");

    bool allPassed = true;
    for (int numGPUs = 1; numGPUs <= maxGPUs; numGPUs *= 2) {
        if (numGPUs > deviceCount)
            break;
        if (!runNcclAllReduceTest(numGPUs, bufferSize)) {
            allPassed = false;
        }
    }
    // Also test with max GPUs if not a power of 2
    if (maxGPUs > 1 && (maxGPUs & (maxGPUs - 1)) != 0) {
        if (!runNcclAllReduceTest(maxGPUs, bufferSize)) {
            allPassed = false;
        }
    }

    printf("\n===== Test Summary =====\n");
    if (allPassed) {
        printf("✅ All NCCL tests PASSED\n");
        return EXIT_SUCCESS;
    } else {
        printf("❌ Some NCCL tests FAILED\n");
        return EXIT_FAILURE;
    }
}
