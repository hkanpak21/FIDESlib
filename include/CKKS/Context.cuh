//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_CKKS_CONTEXT_CUH
#define FIDESLIB_CKKS_CONTEXT_CUH

#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "Parameters.cuh"
#include "RNSPoly.cuh"
#include "KeySwitchingKey.cuh"

#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <optional>

namespace FIDESlib::CKKS {

class RNSPoly;

class Context {
    static constexpr const char* loc{"Context"};
    CudaNvtxRange my_range;

   public:
    enum RESCALE_TECHNIQUE { NO_RESCALE, FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT };

    Parameters param;
    const int logN, N, slots;
    const RESCALE_TECHNIQUE rescaleTechnique;
    const int L, logQ;
    int batch;
    const std::vector<int> GPUid;
    const int dnum;

    std::vector<std::vector<int>> digitGPUid;
    const std::vector<PrimeRecord> prime;

    std::vector<std::vector<LimbRecord>> meta;
    const std::vector<int> logQ_d;

    const int K, logP;
    const std::vector<PrimeRecord> specialPrime;

    std::vector<LimbRecord> specialMeta;                           // Make const maybe
    std::vector<std::vector<std::vector<LimbRecord>>> decompMeta;  // Make const maybe
    std::vector<std::vector<std::vector<LimbRecord>>> digitMeta;   // Make const maybe

    const std::vector<dim3> limbGPUid;
    const std::vector<int> GPUrank;

    std::unique_ptr<RNSPoly> key_switch_aux = nullptr;
    std::unique_ptr<RNSPoly> key_switch_aux2 = nullptr;
    std::unique_ptr<RNSPoly> moddown_aux = nullptr;
    //      std::array<Stream, 8> blockingStream;
    //      std::vector<std::vector<Stream>> asyncStream;
    RNSPoly& getKeySwitchAux();
    RNSPoly& getKeySwitchAux2();
    RNSPoly& getModdownAux();

    bool isValidPrimeId(const int i) const;

    Context(Parameters param, const std::vector<int>& devs, const int secBits = 0);
    ~Context();

   private:
    static int computeLogQ(const int L, std::vector<PrimeRecord>& primes);

    static int validateDnum(const std::vector<int>& GPUid, const int dnum);

    static std::vector<std::vector<LimbRecord>> generateMeta(const std::vector<int>& GPUid, const int dnum,
                                                             const std::vector<std::vector<int>> digitGPUid,
                                                             const std::vector<PrimeRecord>& prime,
                                                             const Parameters& param);

    static std::vector<int> computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta,
                                          const std::vector<PrimeRecord>& prime);

    static int computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, const Parameters& param);

    static std::vector<LimbRecord> generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                                       const std::vector<PrimeRecord>& specialPrime, const int ID0);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDecompMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> dnum);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDigitMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<LimbRecord>& specialMeta,
        const std::vector<std::vector<int>> dnum);

    static std::vector<dim3> generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L);

    static std::vector<std::vector<int>> generateDigitGPUid(const int dnum, const std::vector<int>& devs);

   public:
    std::vector<uint64_t> ElemForEvalMult(int level, const double operand);
    std::vector<uint64_t> ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg);
    std::vector<double>& GetCoeffsChebyshev();
    int GetDoubleAngleIts();
    void AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) const;
    BootstrapPrecomputation& GetBootPrecomputation(int slots) const;
    void AddRotationKey(int index, KeySwitchingKey&& ksk);
    KeySwitchingKey& GetRotationKey(int index);
    bool HasRotationKey(int index);
    void AddEvalKey(KeySwitchingKey&& ksk);
    KeySwitchingKey& GetEvalKey();
    int GetBootK();

    // Per-context key storage (replaces static globals for multi-GPU support)
    mutable std::map<int, BootstrapPrecomputation> boot_precomps_;
    mutable std::map<int, KeySwitchingKey> rot_keys_;
    mutable std::optional<KeySwitchingKey> eval_key_;
    int GetBootCorrectionFactor();
    static RESCALE_TECHNIQUE translateRescalingTechnique(lbcrypto::ScalingTechnique technique);
};
}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_CONTEXT_CUH