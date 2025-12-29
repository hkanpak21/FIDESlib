/**
 * @file p2p_bw.cu
 * @brief CUDA Peer-to-Peer Bandwidth Test
 * 
 * Purpose: Measure P2P bandwidth between GPUs to set realistic expectations
 * for any cross-GPU design in FHE workloads.
 * 
 * Test procedure:
 * 1. Enable peer access between GPU pairs
 * 2. Perform cudaMemcpyPeerAsync between GPUs
 * 3. Measure bandwidth in GB/s for various buffer sizes
 * 
 * @author hkanpak21
 * @date 2025-12-29
 */

#include <cuda_runtime.h>
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

/**
 * @brief Test P2P bandwidth between two GPUs
 * @param srcGPU Source GPU ID
 * @param dstGPU Destination GPU ID
 * @param bufferSize Buffer size in bytes
 * @param numIterations Number of iterations for timing
 * @return Bandwidth in GB/s
 */
double testP2PBandwidth(int srcGPU, int dstGPU, size_t bufferSize, int numIterations = 100) {
    float* d_src = nullptr;
    float* d_dst = nullptr;
    cudaStream_t stream;

    // Allocate source buffer on srcGPU
    CUDA_CHECK(cudaSetDevice(srcGPU));
    CUDA_CHECK(cudaMalloc(&d_src, bufferSize));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate destination buffer on dstGPU
    CUDA_CHECK(cudaSetDevice(dstGPU));
    CUDA_CHECK(cudaMalloc(&d_dst, bufferSize));

    // Enable P2P access if possible
    int canAccessPeer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, dstGPU, srcGPU));

    if (canAccessPeer) {
        CUDA_CHECK(cudaSetDevice(dstGPU));
        cudaDeviceEnablePeerAccess(srcGPU, 0);  // Ignore error if already enabled
        CUDA_CHECK(cudaSetDevice(srcGPU));
        cudaDeviceEnablePeerAccess(dstGPU, 0);  // Ignore error if already enabled
    }

    CUDA_CHECK(cudaSetDevice(srcGPU));

    // Warm-up
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dstGPU, d_src, srcGPU, bufferSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; i++) {
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dstGPU, d_src, srcGPU, bufferSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    double bandwidth_gb_s = (static_cast<double>(bufferSize) * numIterations) / (elapsed_s * 1024.0 * 1024.0 * 1024.0);

    // Cleanup
    CUDA_CHECK(cudaSetDevice(srcGPU));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaSetDevice(dstGPU));
    CUDA_CHECK(cudaFree(d_dst));

    return bandwidth_gb_s;
}

/**
 * @brief Test bidirectional P2P bandwidth between two GPUs
 */
double testP2PBidirectionalBandwidth(int gpu0, int gpu1, size_t bufferSize, int numIterations = 100) {
    float *d_buf0, *d_buf1;
    cudaStream_t stream0, stream1;

    // Allocate on GPU 0
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMalloc(&d_buf0, bufferSize));
    CUDA_CHECK(cudaStreamCreate(&stream0));

    // Allocate on GPU 1
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaMalloc(&d_buf1, bufferSize));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    // Enable P2P if possible
    int canAccess01 = 0, canAccess10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess01, gpu0, gpu1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess10, gpu1, gpu0));

    if (canAccess01) {
        CUDA_CHECK(cudaSetDevice(gpu0));
        cudaDeviceEnablePeerAccess(gpu1, 0);
    }
    if (canAccess10) {
        CUDA_CHECK(cudaSetDevice(gpu1));
        cudaDeviceEnablePeerAccess(gpu0, 0);
    }

    // Warm-up
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaSetDevice(gpu0));
        CUDA_CHECK(cudaMemcpyPeerAsync(d_buf1, gpu1, d_buf0, gpu0, bufferSize, stream0));
        CUDA_CHECK(cudaSetDevice(gpu1));
        CUDA_CHECK(cudaMemcpyPeerAsync(d_buf0, gpu0, d_buf1, gpu1, bufferSize, stream1));
    }
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Timed iterations (bidirectional)
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; i++) {
        CUDA_CHECK(cudaSetDevice(gpu0));
        CUDA_CHECK(cudaMemcpyPeerAsync(d_buf1, gpu1, d_buf0, gpu0, bufferSize, stream0));
        CUDA_CHECK(cudaSetDevice(gpu1));
        CUDA_CHECK(cudaMemcpyPeerAsync(d_buf0, gpu0, d_buf1, gpu1, bufferSize, stream1));
    }
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    // Total data moved in both directions
    double bandwidth_gb_s =
        (static_cast<double>(bufferSize) * numIterations * 2.0) / (elapsed_s * 1024.0 * 1024.0 * 1024.0);

    // Cleanup
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaFree(d_buf0));
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    return bandwidth_gb_s;
}

int main(int argc, char* argv[]) {
    printf("===== CUDA Peer-to-Peer Bandwidth Test =====\n");
    printf("This measures P2P transfer speeds between GPU pairs\n\n");

    // Get number of available GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n\n", deviceCount);

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
    printf("\n");

    // Check P2P access matrix
    printf("=== P2P Access Matrix ===\n");
    printf("        ");
    for (int j = 0; j < deviceCount; j++) {
        printf("GPU%-3d  ", j);
    }
    printf("\n");

    for (int i = 0; i < deviceCount; i++) {
        printf("GPU %d:  ", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) {
                printf("-       ");
            } else {
                int canAccess = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
                printf("%s    ", canAccess ? "Yes" : "No ");
            }
        }
        printf("\n");
    }
    printf("\n");

    // Test buffer sizes
    std::vector<size_t> bufferSizes = {
        1 * 1024 * 1024,   // 1 MB
        4 * 1024 * 1024,   // 4 MB
        16 * 1024 * 1024,  // 16 MB
        64 * 1024 * 1024,  // 64 MB
        256 * 1024 * 1024  // 256 MB
    };

    if (deviceCount >= 2) {
        printf("=== Unidirectional P2P Bandwidth (GPU 0 -> GPU 1) ===\n");
        printf("Buffer Size     Bandwidth\n");
        printf("-----------     ---------\n");

        for (size_t size : bufferSizes) {
            double bw = testP2PBandwidth(0, 1, size, 100);
            printf("%6zu MB       %6.2f GB/s\n", size / (1024 * 1024), bw);
        }
        printf("\n");

        printf("=== Bidirectional P2P Bandwidth (GPU 0 <-> GPU 1) ===\n");
        printf("Buffer Size     Bandwidth\n");
        printf("-----------     ---------\n");

        for (size_t size : bufferSizes) {
            double bw = testP2PBidirectionalBandwidth(0, 1, size, 100);
            printf("%6zu MB       %6.2f GB/s\n", size / (1024 * 1024), bw);
        }
        printf("\n");

        // Test all GPU pairs for 64MB buffer
        if (deviceCount > 2) {
            printf("=== P2P Bandwidth Matrix (64 MB buffer) ===\n");
            printf("Source\\Dest  ");
            for (int j = 0; j < deviceCount; j++) {
                printf("GPU%-5d", j);
            }
            printf("\n");

            for (int i = 0; i < deviceCount; i++) {
                printf("GPU %d:       ", i);
                for (int j = 0; j < deviceCount; j++) {
                    if (i == j) {
                        printf("-       ");
                    } else {
                        double bw = testP2PBandwidth(i, j, 64 * 1024 * 1024, 50);
                        printf("%5.1f   ", bw);
                    }
                }
                printf("\n");
            }
        }
    } else {
        printf("Only 1 GPU available - P2P tests require at least 2 GPUs\n");
    }

    printf("\n===== Test Complete =====\n");
    printf("Note: PCIe systems typically achieve 10-20 GB/s\n");
    printf("      NVLink systems can achieve 50-300+ GB/s\n");

    return EXIT_SUCCESS;
}
