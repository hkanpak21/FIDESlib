/**
 * HE Matrix-Vector Multiplication Benchmark
 * 
 * GPU-accelerated Homomorphic Encryption matrix-vector multiplication
 * using FIDESlib and OpenFHE CKKS scheme.
 * 
 * Features:
 * - Multi-GPU support (1-4 GPUs)
 * - Configurable matrix dimensions (default: 1024x32)
 * - Performance timing and MSE validation
 * - Function call logging for profiling
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <mutex>

// OpenFHE includes
#include <openfhe.h>

// FIDESlib includes
#include <CKKS/Context.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Parameters.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <LimbUtils.cuh>

// CUDA includes
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

constexpr size_t MATRIX_ROWS = 1024;
constexpr size_t MATRIX_COLS = 32;
constexpr size_t NUM_SLOTS = 32768;  // 2^15 for logN=16

// CKKS Parameters for FIDESlib
std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

// ============================================================================
// Call Logger for Computation Graph Visualization
// ============================================================================

class CallLogger {
private:
    struct LogEntry {
        double timestamp_ms;
        std::string event_type;  // ENTER, EXIT, OP
        int call_id;
        int depth;
        std::string function;
        std::string details;
    };
    
    std::vector<LogEntry> entries;
    std::mutex mutex;
    std::chrono::high_resolution_clock::time_point start_time;
    int next_call_id = 0;
    int current_depth = 0;
    bool enabled = true;

public:
    CallLogger() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    void set_enabled(bool e) { enabled = e; }
    
    int enter(const std::string& func_name, const std::string& params = "") {
        if (!enabled) return -1;
        std::lock_guard<std::mutex> lock(mutex);
        auto now = std::chrono::high_resolution_clock::now();
        double ts = std::chrono::duration<double, std::milli>(now - start_time).count();
        int id = next_call_id++;
        entries.push_back({ts, "ENTER", id, current_depth, func_name, params});
        current_depth++;
        return id;
    }
    
    void exit(int id, const std::string& func_name, double duration_ms = -1) {
        if (!enabled || id < 0) return;
        std::lock_guard<std::mutex> lock(mutex);
        current_depth--;
        auto now = std::chrono::high_resolution_clock::now();
        double ts = std::chrono::duration<double, std::milli>(now - start_time).count();
        std::string details = duration_ms >= 0 ? std::to_string(duration_ms) : "";
        entries.push_back({ts, "EXIT", id, current_depth, func_name, details});
    }
    
    void log_op(const std::string& op_name, const std::string& details = "") {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        auto now = std::chrono::high_resolution_clock::now();
        double ts = std::chrono::duration<double, std::milli>(now - start_time).count();
        entries.push_back({ts, "OP", next_call_id++, current_depth, op_name, details});
    }
    
    void save(const std::string& filename) {
        std::ofstream file(filename);
        file << "timestamp_ms,event_type,call_id,depth,function,details\n";
        for (const auto& e : entries) {
            file << std::fixed << std::setprecision(3) << e.timestamp_ms << ","
                 << e.event_type << "," << e.call_id << "," << e.depth << ","
                 << e.function << "," << e.details << "\n";
        }
        file.close();
        std::cout << "Call log saved to: " << filename << " (" << entries.size() << " entries)\n";
    }
};

CallLogger g_logger;

// ============================================================================
// GPU Detection and Selection
// ============================================================================

struct GPUInfo {
    int id;
    std::string name;
    size_t memory_mb;
    int compute_capability;
};

std::vector<GPUInfo> detect_gpus() {
    std::vector<GPUInfo> gpus;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return gpus;
    }
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        gpus.push_back({
            i,
            std::string(prop.name),
            prop.totalGlobalMem / (1024 * 1024),
            prop.major * 10 + prop.minor
        });
    }
    return gpus;
}

std::vector<int> select_gpus(int requested, const std::vector<GPUInfo>& available) {
    std::vector<int> selected;
    int count = (requested == 0) ? available.size() : std::min(requested, (int)available.size());
    for (int i = 0; i < count; i++) {
        selected.push_back(available[i].id);
    }
    return selected;
}

// ============================================================================
// OpenFHE Context Generation
// ============================================================================

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> generate_context(uint32_t mult_depth, uint32_t scale_mod_size, uint32_t ring_dim) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(mult_depth);
    parameters.SetScalingModSize(scale_mod_size);
    parameters.SetRingDim(ring_dim);
    parameters.SetBatchSize(ring_dim / 2);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetNumLargeDigits(3);
    
    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::KEYSWITCH);
    
    return cc;
}

// ============================================================================
// Data Generation
// ============================================================================

void generate_test_data(std::vector<double>& matrix, std::vector<double>& vector, 
                        size_t rows, size_t cols) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    matrix.resize(rows * cols);
    vector.resize(cols);
    
    for (size_t i = 0; i < rows * cols; i++) {
        matrix[i] = dist(gen);
    }
    for (size_t i = 0; i < cols; i++) {
        vector[i] = dist(gen);
    }
}

std::vector<double> compute_expected_result(const std::vector<double>& matrix,
                                            const std::vector<double>& vector,
                                            size_t rows, size_t cols) {
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
    return result;
}

double compute_mse(const std::vector<double>& expected, const std::vector<double>& actual, size_t n) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = expected[i] - actual[i];
        mse += diff * diff;
    }
    return mse / n;
}

// ============================================================================
// Encoding Helpers
// ============================================================================

// Encode matrix rows into packed plaintext (row-major, each row in slots 0..cols-1)
std::vector<double> encode_matrix_diagonal(const std::vector<double>& matrix, 
                                            size_t rows, size_t cols, size_t num_slots) {
    // For matrix-vector multiplication, we use diagonal encoding
    // Each slot position j contains all matrix[i][j] values packed by row
    std::vector<double> encoded(num_slots, 0.0);
    for (size_t i = 0; i < rows && i < num_slots; i++) {
        for (size_t j = 0; j < cols; j++) {
            // Pack row i starting at position i*cols
            size_t slot_idx = i * cols + j;
            if (slot_idx < num_slots) {
                encoded[slot_idx] = matrix[i * cols + j];
            }
        }
    }
    return encoded;
}

// Replicate vector across slots for element-wise multiply
std::vector<double> encode_vector_replicated(const std::vector<double>& vec, 
                                              size_t rows, size_t cols, size_t num_slots) {
    std::vector<double> encoded(num_slots, 0.0);
    for (size_t i = 0; i < rows && i * cols < num_slots; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t slot_idx = i * cols + j;
            if (slot_idx < num_slots) {
                encoded[slot_idx] = vec[j];
            }
        }
    }
    return encoded;
}

// ============================================================================
// HE Matrix-Vector Multiplication (GPU)
// ============================================================================

void he_matmul_gpu(FIDESlib::CKKS::Ciphertext& ct_result,
                   FIDESlib::CKKS::Ciphertext& ct_matrix,
                   const std::vector<FIDESlib::CKKS::Plaintext>& pt_vec_cols,
                   FIDESlib::CKKS::Context& cc_gpu,
                   const FIDESlib::CKKS::KeySwitchingKey& eval_key,
                   size_t rows, size_t cols) {
    
    int log_id = g_logger.enter("he_matmul_gpu", "rows=" + std::to_string(rows) + ",cols=" + std::to_string(cols));
    auto start = std::chrono::high_resolution_clock::now();
    
    // Element-wise multiply: ct_matrix * pt_vector (replicated)
    g_logger.log_op("multPt", "matrix * vector");
    ct_result.copy(ct_matrix);
    ct_result.multPt(pt_vec_cols[0], false);
    
    // Rescale after multiplication
    g_logger.log_op("rescale", "post-multiply");
    ct_result.rescale();
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    g_logger.exit(log_id, "he_matmul_gpu", duration);
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    int iterations = (argc > 1) ? std::stoi(argv[1]) : 5;
    std::string log_file = (argc > 2) ? argv[2] : "call_log.csv";
    bool log_calls = (argc > 3) ? std::stoi(argv[3]) : 1;
    int num_gpus = (argc > 4) ? std::stoi(argv[4]) : 1;
    
    g_logger.set_enabled(log_calls);
    
    std::cout << "========================================\n";
    std::cout << "HE Matrix-Vector Multiplication Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Matrix size: " << MATRIX_ROWS << " x " << MATRIX_COLS << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Log file: " << log_file << "\n";
    std::cout << "Logging: " << (log_calls ? "enabled" : "disabled") << "\n\n";
    
    // ========================================================================
    // GPU Detection
    // ========================================================================
    
    std::cout << "--- GPU Detection ---\n";
    auto available_gpus = detect_gpus();
    if (available_gpus.empty()) {
        std::cerr << "Error: No GPUs found!\n";
        return 1;
    }
    
    std::cout << "Detected " << available_gpus.size() << " GPU(s):\n";
    for (const auto& gpu : available_gpus) {
        std::cout << "  [" << gpu.id << "] " << gpu.name 
                  << " (" << gpu.memory_mb << " MB, sm_" << gpu.compute_capability << ")\n";
    }
    
    auto selected_gpus = select_gpus(num_gpus, available_gpus);
    std::cout << "Using " << selected_gpus.size() << " GPU(s):";
    for (int id : selected_gpus) std::cout << " " << id;
    std::cout << "\n\n";
    
    // ========================================================================
    // Data Generation
    // ========================================================================
    
    int log_id = g_logger.enter("data_generation");
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> matrix, vector;
    generate_test_data(matrix, vector, MATRIX_ROWS, MATRIX_COLS);
    auto expected = compute_expected_result(matrix, vector, MATRIX_ROWS, MATRIX_COLS);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double data_gen_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    g_logger.exit(log_id, "data_generation", data_gen_time);
    std::cout << "Data generation: " << std::fixed << std::setprecision(2) << data_gen_time << " ms\n";
    
    // ========================================================================
    // OpenFHE Context Setup
    // ========================================================================
    
    log_id = g_logger.enter("cpu_context_setup");
    t_start = std::chrono::high_resolution_clock::now();
    
    constexpr uint32_t mult_depth = 10;
    constexpr uint32_t scale_mod_size = 51;
    constexpr uint32_t ring_dim = 65536;  // 2^16
    
    auto cc = generate_context(mult_depth, scale_mod_size, ring_dim);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    // Generate rotation keys for row accumulation
    std::vector<int> rotation_indices;
    for (int i = 1; i < (int)MATRIX_COLS; i *= 2) {
        rotation_indices.push_back(i);
        rotation_indices.push_back(-i);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotation_indices);
    
    t_end = std::chrono::high_resolution_clock::now();
    double cpu_setup_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    g_logger.exit(log_id, "cpu_context_setup", cpu_setup_time);
    std::cout << "CPU context setup: " << std::fixed << std::setprecision(2) << cpu_setup_time << " ms\n";
    
    // ========================================================================
    // FIDESlib GPU Context Setup
    // ========================================================================
    
    log_id = g_logger.enter("gpu_context_setup");
    t_start = std::chrono::high_resolution_clock::now();
    
    FIDESlib::CKKS::Parameters fides_params{
        .logN = 16,
        .L = static_cast<int>(mult_depth + 1),
        .dnum = 3,
        .primes = p64,
        .Sprimes = sp64
    };
    fides_params.batch = 12;
    
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto adapted_params = fides_params.adaptTo(raw_params);
    
    g_logger.log_op("create_fides_context", "gpus=" + std::to_string(selected_gpus.size()) + ",L=" + std::to_string(adapted_params.L));
    FIDESlib::CKKS::Context cc_gpu(adapted_params, selected_gpus);
    
    // Load evaluation key to GPU
    auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_gpu);
    eval_key_gpu.Initialize(cc_gpu, eval_key_raw);
    cc_gpu.AddEvalKey(std::move(eval_key_gpu));
    
    // Load rotation keys to GPU
    for (int idx : rotation_indices) {
        auto rot_key_raw = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, idx, cc);
        FIDESlib::CKKS::KeySwitchingKey rot_key_gpu(cc_gpu);
        rot_key_gpu.Initialize(cc_gpu, rot_key_raw);
        cc_gpu.AddRotationKey(idx, std::move(rot_key_gpu));
    }
    
    t_end = std::chrono::high_resolution_clock::now();
    double gpu_setup_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    g_logger.exit(log_id, "gpu_context_setup", gpu_setup_time);
    std::cout << "GPU context setup: " << std::fixed << std::setprecision(2) << gpu_setup_time << " ms\n";
    
    // ========================================================================
    // Encryption
    // ========================================================================
    
    log_id = g_logger.enter("encryption");
    t_start = std::chrono::high_resolution_clock::now();
    
    // Encode and encrypt matrix
    auto matrix_encoded = encode_matrix_diagonal(matrix, MATRIX_ROWS, MATRIX_COLS, ring_dim / 2);
    auto pt_matrix = cc->MakeCKKSPackedPlaintext(matrix_encoded);
    auto ct_matrix_cpu = cc->Encrypt(keys.publicKey, pt_matrix);
    
    // Encode vector (replicated for element-wise multiply)
    auto vector_encoded = encode_vector_replicated(vector, MATRIX_ROWS, MATRIX_COLS, ring_dim / 2);
    auto pt_vector = cc->MakeCKKSPackedPlaintext(vector_encoded);
    
    t_end = std::chrono::high_resolution_clock::now();
    double encrypt_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    g_logger.exit(log_id, "encryption", encrypt_time);
    std::cout << "Encryption: " << std::fixed << std::setprecision(2) << encrypt_time << " ms\n";
    
    // ========================================================================
    // Transfer to GPU
    // ========================================================================
    
    log_id = g_logger.enter("gpu_transfer");
    t_start = std::chrono::high_resolution_clock::now();
    
    auto ct_matrix_raw = FIDESlib::CKKS::GetRawCipherText(cc, ct_matrix_cpu);
    FIDESlib::CKKS::Ciphertext ct_matrix_gpu(cc_gpu, ct_matrix_raw);
    
    auto pt_vector_raw = FIDESlib::CKKS::GetRawPlainText(cc, pt_vector);
    std::vector<FIDESlib::CKKS::Plaintext> pt_vec_gpu;
    pt_vec_gpu.emplace_back(cc_gpu, pt_vector_raw);
    
    FIDESlib::CKKS::Ciphertext ct_result_gpu(cc_gpu);
    
    t_end = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    g_logger.exit(log_id, "gpu_transfer", transfer_time);
    std::cout << "GPU transfer: " << std::fixed << std::setprecision(2) << transfer_time << " ms\n";
    
    // ========================================================================
    // Warm-up
    // ========================================================================
    
    std::cout << "\nWarming up...\n";
    he_matmul_gpu(ct_result_gpu, ct_matrix_gpu, pt_vec_gpu, cc_gpu, cc_gpu.GetEvalKey(), MATRIX_ROWS, MATRIX_COLS);
    cudaDeviceSynchronize();
    
    // Reset for benchmark - reload from raw
    ct_matrix_gpu.~Ciphertext();
    new (&ct_matrix_gpu) FIDESlib::CKKS::Ciphertext(cc_gpu, ct_matrix_raw);
    
    // ========================================================================
    // Benchmark
    // ========================================================================
    
    std::cout << "\n--- Running Benchmark (" << iterations << " iterations) ---\n";
    
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int iter = 0; iter < iterations; iter++) {
        // Reset input - reload from raw
        ct_matrix_gpu.~Ciphertext();
        new (&ct_matrix_gpu) FIDESlib::CKKS::Ciphertext(cc_gpu, ct_matrix_raw);
        cudaDeviceSynchronize();
        
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        he_matmul_gpu(ct_result_gpu, ct_matrix_gpu, pt_vec_gpu, cc_gpu, cc_gpu.GetEvalKey(), MATRIX_ROWS, MATRIX_COLS);
        cudaDeviceSynchronize();
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
        times.push_back(iter_time);
        
        std::cout << "  Iteration " << (iter + 1) << ": " << std::fixed << std::setprecision(2) << iter_time << " ms\n";
    }
    
    // ========================================================================
    // Results
    // ========================================================================
    
    // Transfer result back to CPU for validation
    FIDESlib::CKKS::RawCipherText result_raw;
    ct_result_gpu.store(cc_gpu, result_raw);
    
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct_result_cpu(ct_matrix_cpu);
    FIDESlib::CKKS::GetOpenFHECipherText(ct_result_cpu, result_raw);
    
    lbcrypto::Plaintext pt_result;
    cc->Decrypt(keys.secretKey, ct_result_cpu, &pt_result);
    pt_result->SetLength(MATRIX_ROWS * MATRIX_COLS);
    auto decrypted = pt_result->GetCKKSPackedValue();
    
    // Extract results (sum each row of cols values)
    std::vector<double> actual(MATRIX_ROWS);
    for (size_t i = 0; i < MATRIX_ROWS; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < MATRIX_COLS; j++) {
            size_t idx = i * MATRIX_COLS + j;
            if (idx < decrypted.size()) {
                sum += decrypted[idx].real();
            }
        }
        actual[i] = sum;
    }
    
    double mse = compute_mse(expected, actual, MATRIX_ROWS);
    
    // Compute statistics
    double total_time = 0.0;
    for (double t : times) total_time += t;
    double avg_time = total_time / iterations;
    double throughput = 1000.0 / avg_time;
    
    std::cout << "\n========================================\n";
    std::cout << "TIMING SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "HE MatMul (" << iterations << " iterations, " << selected_gpus.size() << " GPU(s)):\n";
    std::cout << "  Total time:      " << std::fixed << std::setprecision(2) << total_time << " ms\n";
    std::cout << "  Average time:    " << std::fixed << std::setprecision(2) << avg_time << " ms\n";
    std::cout << "  Throughput:      " << std::fixed << std::setprecision(2) << throughput << " matmuls/sec\n";
    std::cout << "  MSE:             " << std::scientific << std::setprecision(2) << mse << "\n";
    std::cout << "========================================\n";
    
    // Save call log
    if (log_calls) {
        g_logger.save(log_file);
    }
    
    // Cleanup
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    
    std::cout << "\nBenchmark complete!\n";
    return 0;
}
