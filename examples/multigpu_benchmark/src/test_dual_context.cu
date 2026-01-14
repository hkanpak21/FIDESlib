/**
 * @file test_dual_context.cu
 * @brief Test that validates multi-GPU support in FIDESlib
 * 
 * This test creates two independent FIDESlib contexts on different GPUs
 * and verifies they can operate without interference.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include <openfhe.h>
#include <CKKS/Context.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

// Prime records for CKKS - standard 64-bit primes
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
    std::cout << "Number of GPUs: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")"
                  << " - " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB"
                  << std::endl;
    }
    std::cout << std::endl;
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createOpenFHEContext(int logN, int L) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetMultiplicativeDepth(L);
    parameters.SetScalingModSize(51);
    parameters.SetRingDim(1 << logN);
    parameters.SetBatchSize(1 << (logN - 1));
    parameters.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    parameters.SetFirstModSize(60);
    
    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    
    return cc;
}

bool testSingleGPU(int gpuId, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, 
                   lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
                   FIDESlib::CKKS::Parameters& params) {
    std::cout << "  Testing GPU " << gpuId << "..." << std::endl;
    
    try {
        cudaSetDevice(gpuId);
        
        // Get raw params from OpenFHE
        auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
        auto p = params.adaptTo(raw_params);
        
        // Create FIDESlib context on this specific GPU
        FIDESlib::CKKS::Context gpu_cc(p, {gpuId});
        
        // Load evaluation key
        auto eval_key_raw = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(gpu_cc);
        eval_key_gpu.Initialize(gpu_cc, eval_key_raw);
        gpu_cc.AddEvalKey(std::move(eval_key_gpu));
        
        // Create test data
        std::vector<double> test_data(1 << (params.logN - 1), 1.5);
        
        // Encrypt
        auto pt = cc->MakeCKKSPackedPlaintext(test_data);
        auto ct = cc->Encrypt(keys.publicKey, pt);
        
        // Convert to FIDESlib format
        auto ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
        FIDESlib::CKKS::Ciphertext ct_gpu(gpu_cc, ct_raw);
        
        // Perform operation: square
        ct_gpu.square(gpu_cc.GetEvalKey());
        
        // Store back
        ct_gpu.store(gpu_cc, ct_raw);
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct(ct);
        FIDESlib::CKKS::GetOpenFHECipherText(result_ct, ct_raw);
        
        // Decrypt and verify
        lbcrypto::Plaintext result_pt;
        cc->Decrypt(keys.secretKey, result_ct, &result_pt);
        result_pt->SetLength(8);  // Check first few values
        auto result = result_pt->GetCKKSPackedValue();
        
        // Verify: 1.5^2 = 2.25
        double expected = 2.25;
        bool success = true;
        for (int i = 0; i < 8 && success; ++i) {
            double err = std::abs(result[i].real() - expected);
            if (err > 0.01) {
                std::cerr << "    ERROR: Value at " << i << " = " << result[i].real() 
                          << ", expected " << expected << std::endl;
                success = false;
            }
        }
        
        if (success) {
            std::cout << "    GPU " << gpuId << " PASSED: Square operation correct (1.5² ≈ 2.25)" << std::endl;
        }
        
        cudaDeviceSynchronize();
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "    ERROR on GPU " << gpuId << ": " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  FIDESlib Multi-GPU Dual Context Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    printGPUInfo();
    
    int numGPUs = getGPUCount();
    if (numGPUs < 2) {
        std::cerr << "ERROR: This test requires at least 2 GPUs." << std::endl;
        std::cerr << "Found only " << numGPUs << " GPU(s)." << std::endl;
        return 1;
    }
    
    // Parameters
    int logN = 13;  // Ring dimension = 8192
    int L = 5;      // Multiplicative depth
    
    std::cout << "Creating OpenFHE context (logN=" << logN << ", L=" << L << ")..." << std::endl;
    auto cc = createOpenFHEContext(logN, L);
    
    std::cout << "Generating keys..." << std::endl;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    // FIDESlib parameters
    FIDESlib::CKKS::Parameters params{.logN = logN, .L = L, .dnum = 1, .primes = p64, .Sprimes = sp64};
    
    std::cout << "\n--- Testing Independent Contexts ---" << std::endl;
    
    // Test GPU 0
    bool gpu0_ok = testSingleGPU(0, cc, keys, params);
    
    // Test GPU 1  
    bool gpu1_ok = testSingleGPU(1, cc, keys, params);
    
    std::cout << "\n========================================" << std::endl;
    if (gpu0_ok && gpu1_ok) {
        std::cout << "  SUCCESS! Both GPUs work independently!" << std::endl;
        std::cout << "  Multi-GPU support is functional." << std::endl;
    } else {
        std::cout << "  FAILED: One or more GPUs had errors." << std::endl;
        std::cout << "  Check the patches were applied correctly." << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    // Cleanup
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    
    return (gpu0_ok && gpu1_ok) ? 0 : 1;
}
