/**
 * @file PerformanceMonitor.cuh
 * @brief Comprehensive Performance Monitoring for Multi-GPU HE Operations
 * 
 * Features:
 * - CUDA Event-based precise GPU timing
 * - NVTX markers for Nsight Systems profiling  
 * - Memory bandwidth tracking
 * - GPU utilization monitoring
 * - Per-phase breakdown analysis
 * - Multi-GPU coordination tracking
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

// NVTX for Nsight Systems profiling
#ifdef __NVCC__
#include <nvtx3/nvToolsExt.h>
#endif

namespace PerfMon {

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)

// ============================================================================
// NVTX Scoped Range (for Nsight Systems profiling)
// ============================================================================
class NVTXRange {
public:
    NVTXRange(const char* name, uint32_t color = 0xFF00FF00) {
#ifdef __NVCC__
        nvtxEventAttributes_t attrib = {0};
        attrib.version = NVTX_VERSION;
        attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attrib.colorType = NVTX_COLOR_ARGB;
        attrib.color = color;
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attrib.message.ascii = name;
        nvtxRangePushEx(&attrib);
#endif
        (void)name; (void)color;  // Suppress warnings if NVTX not available
    }
    
    ~NVTXRange() {
#ifdef __NVCC__
        nvtxRangePop();
#endif
    }
};

// NVTX color palette
namespace NVTXColors {
    constexpr uint32_t Green    = 0xFF00FF00;
    constexpr uint32_t Blue     = 0xFF0000FF;
    constexpr uint32_t Red      = 0xFFFF0000;
    constexpr uint32_t Yellow   = 0xFFFFFF00;
    constexpr uint32_t Cyan     = 0xFF00FFFF;
    constexpr uint32_t Magenta  = 0xFFFF00FF;
    constexpr uint32_t Orange   = 0xFFFFA500;
    constexpr uint32_t Purple   = 0xFF800080;
}

// Macros for easy NVTX usage
#define NVTX_SCOPE(name) PerfMon::NVTXRange nvtx_##__LINE__(name)
#define NVTX_SCOPE_COLOR(name, color) PerfMon::NVTXRange nvtx_##__LINE__(name, color)

// ============================================================================
// CUDA Event Timer - Precise GPU timing
// ============================================================================
class CudaEventTimer {
public:
    CudaEventTimer(int deviceId = 0) : deviceId_(deviceId) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaEventTimer() {
        cudaSetDevice(deviceId_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }
    
    float elapsedMs() {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
    void synchronize() {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

private:
    int deviceId_;
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Phase Timing - Track individual operation phases
// ============================================================================
enum class Phase {
    ContextSetup,
    KeyGeneration,
    KeyLoading,
    Encryption,
    H2DTransfer,
    NTTForward,
    NTTInverse,
    ModMult,
    KeySwitch,
    Rescale,
    HEMultiply,
    HEAdd,
    HERotate,
    D2HTransfer,
    Decryption,
    Synchronization,
    Total
};

const char* phaseToString(Phase phase) {
    switch (phase) {
        case Phase::ContextSetup:    return "Context Setup";
        case Phase::KeyGeneration:   return "Key Generation";
        case Phase::KeyLoading:      return "Key Loading";
        case Phase::Encryption:      return "Encryption";
        case Phase::H2DTransfer:     return "H2D Transfer";
        case Phase::NTTForward:      return "NTT Forward";
        case Phase::NTTInverse:      return "NTT Inverse";
        case Phase::ModMult:         return "Modular Mult";
        case Phase::KeySwitch:       return "Key Switching";
        case Phase::Rescale:         return "Rescale";
        case Phase::HEMultiply:      return "HE Multiply";
        case Phase::HEAdd:           return "HE Add";
        case Phase::HERotate:        return "HE Rotate";
        case Phase::D2HTransfer:     return "D2H Transfer";
        case Phase::Decryption:      return "Decryption";
        case Phase::Synchronization: return "Synchronization";
        case Phase::Total:           return "TOTAL";
        default:                     return "Unknown";
    }
}

// ============================================================================
// Timing Record - Single timing measurement
// ============================================================================
struct TimingRecord {
    Phase phase;
    int gpuId;
    double durationMs;
    size_t bytesProcessed;  // For bandwidth calculation
    int operationCount;
    
    double bandwidthGBps() const {
        if (durationMs > 0 && bytesProcessed > 0) {
            return (bytesProcessed / (1024.0 * 1024.0 * 1024.0)) / (durationMs / 1000.0);
        }
        return 0;
    }
};

// ============================================================================
// GPU Memory Stats
// ============================================================================
struct GPUMemoryStats {
    int deviceId;
    size_t totalMemory;
    size_t freeMemory;
    size_t usedMemory;
    double utilizationPercent;
    
    static GPUMemoryStats capture(int deviceId) {
        GPUMemoryStats stats;
        stats.deviceId = deviceId;
        cudaSetDevice(deviceId);
        cudaMemGetInfo(&stats.freeMemory, &stats.totalMemory);
        stats.usedMemory = stats.totalMemory - stats.freeMemory;
        stats.utilizationPercent = (double)stats.usedMemory / stats.totalMemory * 100.0;
        return stats;
    }
    
    void print() const {
        std::cout << "  GPU " << deviceId << " Memory: "
                  << std::fixed << std::setprecision(1)
                  << (usedMemory / (1024.0*1024.0*1024.0)) << " / "
                  << (totalMemory / (1024.0*1024.0*1024.0)) << " GB ("
                  << utilizationPercent << "% used)" << std::endl;
    }
};

// ============================================================================
// Performance Monitor - Main class for tracking all metrics
// ============================================================================
class PerformanceMonitor {
public:
    PerformanceMonitor(const std::string& benchmarkName = "Benchmark") 
        : benchmarkName_(benchmarkName), enabled_(true) {
        wallClockStart_ = std::chrono::high_resolution_clock::now();
    }
    
    void enable(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    // -------------------------------------------------------------------------
    // Phase Timing
    // -------------------------------------------------------------------------
    void startPhase(Phase phase, int gpuId = 0, cudaStream_t stream = 0) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        PhaseKey key{phase, gpuId};
        auto& timer = phaseTimers_[key];
        if (!timer) {
            timer = std::make_unique<CudaEventTimer>(gpuId);
        }
        timer->start(stream);
        
        // NVTX marker
        std::string name = std::string(phaseToString(phase)) + " (GPU " + std::to_string(gpuId) + ")";
        nvtxRangePush(name.c_str());
    }
    
    void endPhase(Phase phase, int gpuId = 0, cudaStream_t stream = 0, 
                  size_t bytesProcessed = 0, int opCount = 1) {
        if (!enabled_) return;
        
        nvtxRangePop();
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        PhaseKey key{phase, gpuId};
        auto it = phaseTimers_.find(key);
        if (it != phaseTimers_.end() && it->second) {
            it->second->stop(stream);
        }
        
        // Store pending record to be finalized later
        PendingRecord pending{phase, gpuId, bytesProcessed, opCount};
        pendingRecords_.push_back(pending);
    }
    
    // -------------------------------------------------------------------------
    // Synchronize and collect all timing data
    // -------------------------------------------------------------------------
    void synchronize() {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Collect all pending timing records
        for (auto& pending : pendingRecords_) {
            PhaseKey key{pending.phase, pending.gpuId};
            auto it = phaseTimers_.find(key);
            if (it != phaseTimers_.end() && it->second) {
                TimingRecord record;
                record.phase = pending.phase;
                record.gpuId = pending.gpuId;
                record.durationMs = it->second->elapsedMs();
                record.bytesProcessed = pending.bytesProcessed;
                record.operationCount = pending.opCount;
                records_.push_back(record);
            }
        }
        pendingRecords_.clear();
    }
    
    // -------------------------------------------------------------------------
    // Add manual timing record (for CPU-side measurements)
    // -------------------------------------------------------------------------
    void addRecord(Phase phase, int gpuId, double durationMs, 
                   size_t bytesProcessed = 0, int opCount = 1) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        TimingRecord record{phase, gpuId, durationMs, bytesProcessed, opCount};
        records_.push_back(record);
    }
    
    // -------------------------------------------------------------------------
    // GPU Memory Tracking
    // -------------------------------------------------------------------------
    void captureMemory(int deviceId) {
        if (!enabled_) return;
        memoryStats_.push_back(GPUMemoryStats::capture(deviceId));
    }
    
    void captureAllGPUMemory(int numGPUs) {
        for (int i = 0; i < numGPUs; ++i) {
            captureMemory(i);
        }
    }
    
    // -------------------------------------------------------------------------
    // Analysis & Reporting
    // -------------------------------------------------------------------------
    struct PhaseStats {
        double totalMs = 0;
        double minMs = std::numeric_limits<double>::max();
        double maxMs = 0;
        double avgMs = 0;
        int count = 0;
        size_t totalBytes = 0;
        double avgBandwidthGBps = 0;
    };
    
    std::map<Phase, PhaseStats> getPhaseStatistics() const {
        std::map<Phase, PhaseStats> stats;
        
        for (const auto& record : records_) {
            auto& s = stats[record.phase];
            s.totalMs += record.durationMs;
            s.minMs = std::min(s.minMs, record.durationMs);
            s.maxMs = std::max(s.maxMs, record.durationMs);
            s.count++;
            s.totalBytes += record.bytesProcessed;
        }
        
        for (auto& [phase, s] : stats) {
            s.avgMs = s.count > 0 ? s.totalMs / s.count : 0;
            if (s.totalMs > 0 && s.totalBytes > 0) {
                s.avgBandwidthGBps = (s.totalBytes / (1024.0*1024.0*1024.0)) / (s.totalMs / 1000.0);
            }
            if (s.count == 0) s.minMs = 0;
        }
        
        return stats;
    }
    
    std::map<int, std::map<Phase, PhaseStats>> getPerGPUStatistics() const {
        std::map<int, std::map<Phase, PhaseStats>> gpuStats;
        
        for (const auto& record : records_) {
            auto& s = gpuStats[record.gpuId][record.phase];
            s.totalMs += record.durationMs;
            s.minMs = std::min(s.minMs, record.durationMs);
            s.maxMs = std::max(s.maxMs, record.durationMs);
            s.count++;
            s.totalBytes += record.bytesProcessed;
        }
        
        for (auto& [gpu, phases] : gpuStats) {
            for (auto& [phase, s] : phases) {
                s.avgMs = s.count > 0 ? s.totalMs / s.count : 0;
                if (s.count == 0) s.minMs = 0;
            }
        }
        
        return gpuStats;
    }
    
    void printSummary() const {
        auto wallClockEnd = std::chrono::high_resolution_clock::now();
        double wallClockMs = std::chrono::duration<double, std::milli>(
            wallClockEnd - wallClockStart_).count();
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    PERFORMANCE MONITOR SUMMARY                            ║" << std::endl;
        std::cout << "║  " << std::left << std::setw(72) << benchmarkName_ << " ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
        
        // Wall clock time
        std::cout << "\n  Wall Clock Time: " << std::fixed << std::setprecision(2) 
                  << wallClockMs << " ms" << std::endl;
        
        // Phase breakdown
        auto stats = getPhaseStatistics();
        double totalTrackedMs = 0;
        
        std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ Phase Breakdown                                                             │" << std::endl;
        std::cout << "├──────────────────┬───────────┬───────────┬───────────┬───────────┬──────────┤" << std::endl;
        std::cout << "│ Phase            │  Total(ms)│    Avg(ms)│    Min(ms)│    Max(ms)│    Count │" << std::endl;
        std::cout << "├──────────────────┼───────────┼───────────┼───────────┼───────────┼──────────┤" << std::endl;
        
        for (const auto& [phase, s] : stats) {
            std::cout << "│ " << std::left << std::setw(16) << phaseToString(phase) << " │"
                      << std::right << std::setw(10) << std::fixed << std::setprecision(2) << s.totalMs << " │"
                      << std::setw(10) << s.avgMs << " │"
                      << std::setw(10) << s.minMs << " │"
                      << std::setw(10) << s.maxMs << " │"
                      << std::setw(9) << s.count << " │" << std::endl;
            totalTrackedMs += s.totalMs;
        }
        
        std::cout << "├──────────────────┼───────────┼───────────┼───────────┼───────────┼──────────┤" << std::endl;
        std::cout << "│ " << std::left << std::setw(16) << "TRACKED TOTAL" << " │"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << totalTrackedMs << " │"
                  << std::setw(10) << "-" << " │"
                  << std::setw(10) << "-" << " │"
                  << std::setw(10) << "-" << " │"
                  << std::setw(9) << records_.size() << " │" << std::endl;
        std::cout << "└──────────────────┴───────────┴───────────┴───────────┴───────────┴──────────┘" << std::endl;
        
        // Overhead analysis
        double overhead = wallClockMs - totalTrackedMs;
        double overheadPercent = (overhead / wallClockMs) * 100.0;
        std::cout << "\n  Untracked Overhead: " << overhead << " ms (" << overheadPercent << "%)" << std::endl;
        
        // Memory stats
        if (!memoryStats_.empty()) {
            std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
            std::cout << "│ GPU Memory Usage                                                            │" << std::endl;
            std::cout << "└─────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            for (const auto& mem : memoryStats_) {
                mem.print();
            }
        }
    }
    
    void printPerGPUBreakdown() const {
        auto gpuStats = getPerGPUStatistics();
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                       PER-GPU TIMING BREAKDOWN                            ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
        
        for (const auto& [gpuId, phases] : gpuStats) {
            std::cout << "\n  GPU " << gpuId << ":" << std::endl;
            std::cout << "  ─────────────────────────────────────────────" << std::endl;
            
            double gpuTotal = 0;
            for (const auto& [phase, s] : phases) {
                std::cout << "    " << std::left << std::setw(20) << phaseToString(phase) 
                          << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
                          << s.totalMs << " ms";
                
                if (s.count > 1) {
                    std::cout << " (avg: " << s.avgMs << " ms x " << s.count << ")";
                }
                std::cout << std::endl;
                gpuTotal += s.totalMs;
            }
            std::cout << "    " << std::string(45, '-') << std::endl;
            std::cout << "    " << std::left << std::setw(20) << "GPU Total:" 
                      << std::right << std::setw(10) << gpuTotal << " ms" << std::endl;
        }
    }
    
    void printLatencyAnalysis() const {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                       LATENCY BOTTLENECK ANALYSIS                         ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;
        
        auto stats = getPhaseStatistics();
        
        // Sort phases by total time
        std::vector<std::pair<Phase, PhaseStats>> sortedPhases(stats.begin(), stats.end());
        std::sort(sortedPhases.begin(), sortedPhases.end(), 
                  [](const auto& a, const auto& b) { return a.second.totalMs > b.second.totalMs; });
        
        // Calculate total
        double totalMs = 0;
        for (const auto& [phase, s] : sortedPhases) {
            totalMs += s.totalMs;
        }
        
        std::cout << "\n  Top Latency Contributors:" << std::endl;
        std::cout << "  ─────────────────────────────────────────────────────────" << std::endl;
        
        double cumulative = 0;
        for (size_t i = 0; i < std::min(sortedPhases.size(), size_t(10)); ++i) {
            const auto& [phase, s] = sortedPhases[i];
            double percent = (s.totalMs / totalMs) * 100.0;
            cumulative += percent;
            
            // Progress bar (using ASCII characters)
            int barLen = (int)(percent / 2);
            std::string bar(barLen, '#');
            bar += std::string(50 - barLen, '-');
            
            std::cout << "  " << std::setw(3) << (i+1) << ". " 
                      << std::left << std::setw(16) << phaseToString(phase)
                      << " │" << bar << "│ "
                      << std::right << std::setw(6) << std::fixed << std::setprecision(1) 
                      << percent << "% ("
                      << std::setw(8) << std::setprecision(2) << s.totalMs << " ms)"
                      << std::endl;
        }
        
        std::cout << "\n  Recommendations:" << std::endl;
        std::cout << "  ─────────────────────────────────────────────────────────" << std::endl;
        
        // Generate recommendations based on analysis
        if (!sortedPhases.empty()) {
            const auto& topPhase = sortedPhases[0].first;
            switch (topPhase) {
                case Phase::KeySwitch:
                    std::cout << "  → Key Switching is the bottleneck. Consider:" << std::endl;
                    std::cout << "    - Using larger dnum for digit decomposition" << std::endl;
                    std::cout << "    - Enabling key switching kernel optimizations" << std::endl;
                    break;
                case Phase::H2DTransfer:
                case Phase::D2HTransfer:
                    std::cout << "  → Memory transfers are the bottleneck. Consider:" << std::endl;
                    std::cout << "    - Using pinned memory for faster transfers" << std::endl;
                    std::cout << "    - Overlapping transfers with computation" << std::endl;
                    std::cout << "    - Reducing transfer frequency (batch operations)" << std::endl;
                    break;
                case Phase::NTTForward:
                case Phase::NTTInverse:
                    std::cout << "  → NTT operations are the bottleneck. Consider:" << std::endl;
                    std::cout << "    - Using larger polynomial ring (more parallelism)" << std::endl;
                    std::cout << "    - Fusing NTT operations where possible" << std::endl;
                    break;
                case Phase::HEMultiply:
                    std::cout << "  → HE Multiplication dominates (expected for CKKS)." << std::endl;
                    std::cout << "    - Current implementation is compute-bound" << std::endl;
                    std::cout << "    - Multi-GPU parallelism is effective here" << std::endl;
                    break;
                case Phase::Synchronization:
                    std::cout << "  → Synchronization overhead is high. Consider:" << std::endl;
                    std::cout << "    - Using async operations with streams" << std::endl;
                    std::cout << "    - Reducing cross-GPU synchronization points" << std::endl;
                    break;
                default:
                    std::cout << "  → Analyze " << phaseToString(topPhase) << " for optimization opportunities." << std::endl;
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // Export to CSV for further analysis
    // -------------------------------------------------------------------------
    void exportToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        file << "Phase,GPU,Duration_ms,Bytes,OpCount,Bandwidth_GBps\n";
        for (const auto& r : records_) {
            file << phaseToString(r.phase) << ","
                 << r.gpuId << ","
                 << std::fixed << std::setprecision(4) << r.durationMs << ","
                 << r.bytesProcessed << ","
                 << r.operationCount << ","
                 << r.bandwidthGBps() << "\n";
        }
        
        std::cout << "  Performance data exported to: " << filename << std::endl;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        pendingRecords_.clear();
        phaseTimers_.clear();
        memoryStats_.clear();
        wallClockStart_ = std::chrono::high_resolution_clock::now();
    }

private:
    struct PhaseKey {
        Phase phase;
        int gpuId;
        
        bool operator<(const PhaseKey& other) const {
            if (phase != other.phase) return phase < other.phase;
            return gpuId < other.gpuId;
        }
    };
    
    struct PendingRecord {
        Phase phase;
        int gpuId;
        size_t bytesProcessed;
        int opCount;
    };
    
    std::string benchmarkName_;
    bool enabled_;
    mutable std::mutex mutex_;
    
    std::map<PhaseKey, std::unique_ptr<CudaEventTimer>> phaseTimers_;
    std::vector<TimingRecord> records_;
    std::vector<PendingRecord> pendingRecords_;
    std::vector<GPUMemoryStats> memoryStats_;
    
    std::chrono::high_resolution_clock::time_point wallClockStart_;
};

// ============================================================================
// Scoped Phase Timer - RAII-style phase timing
// ============================================================================
class ScopedPhase {
public:
    ScopedPhase(PerformanceMonitor& monitor, Phase phase, int gpuId = 0, 
                cudaStream_t stream = 0, size_t bytesProcessed = 0, int opCount = 1)
        : monitor_(monitor), phase_(phase), gpuId_(gpuId), stream_(stream),
          bytesProcessed_(bytesProcessed), opCount_(opCount) {
        monitor_.startPhase(phase_, gpuId_, stream_);
    }
    
    ~ScopedPhase() {
        monitor_.endPhase(phase_, gpuId_, stream_, bytesProcessed_, opCount_);
    }

private:
    PerformanceMonitor& monitor_;
    Phase phase_;
    int gpuId_;
    cudaStream_t stream_;
    size_t bytesProcessed_;
    int opCount_;
};

// ============================================================================
// Multi-GPU Sync Analyzer - Track synchronization overhead
// ============================================================================
class SyncAnalyzer {
public:
    SyncAnalyzer(int numGPUs) : numGPUs_(numGPUs), syncPoints_(0), totalSyncMs_(0) {}
    
    void recordSyncStart() {
        syncStart_ = std::chrono::high_resolution_clock::now();
    }
    
    void recordSyncEnd() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - syncStart_).count();
        totalSyncMs_ += ms;
        syncPoints_++;
    }
    
    void barrierSync() {
        recordSyncStart();
        for (int g = 0; g < numGPUs_; ++g) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        recordSyncEnd();
    }
    
    void print() const {
        std::cout << "\n  Synchronization Analysis:" << std::endl;
        std::cout << "    Sync points: " << syncPoints_ << std::endl;
        std::cout << "    Total sync time: " << totalSyncMs_ << " ms" << std::endl;
        if (syncPoints_ > 0) {
            std::cout << "    Avg sync time: " << (totalSyncMs_ / syncPoints_) << " ms" << std::endl;
        }
    }

private:
    int numGPUs_;
    int syncPoints_;
    double totalSyncMs_;
    std::chrono::high_resolution_clock::time_point syncStart_;
};

// ============================================================================
// Throughput Calculator
// ============================================================================
class ThroughputCalculator {
public:
    void addDataPoint(double timeMs, int operations) {
        times_.push_back(timeMs);
        ops_.push_back(operations);
    }
    
    double getAverageOpsPerSec() const {
        if (times_.empty()) return 0;
        double totalOps = 0, totalTime = 0;
        for (size_t i = 0; i < times_.size(); ++i) {
            totalTime += times_[i];
            totalOps += ops_[i];
        }
        return (totalOps / totalTime) * 1000.0;
    }
    
    double getStdDevOpsPerSec() const {
        if (times_.size() < 2) return 0;
        
        double avg = getAverageOpsPerSec();
        double sumSq = 0;
        
        for (size_t i = 0; i < times_.size(); ++i) {
            double opsPerSec = (ops_[i] / times_[i]) * 1000.0;
            sumSq += (opsPerSec - avg) * (opsPerSec - avg);
        }
        
        return std::sqrt(sumSq / (times_.size() - 1));
    }
    
    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Throughput: " << getAverageOpsPerSec() << " ops/sec"
                  << " (±" << getStdDevOpsPerSec() << ")" << std::endl;
    }

private:
    std::vector<double> times_;
    std::vector<int> ops_;
};

} // namespace PerfMon
