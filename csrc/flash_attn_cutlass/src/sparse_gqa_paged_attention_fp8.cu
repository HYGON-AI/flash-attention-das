// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Paged family FP8 entry and cache codecs; shared attention/dispatch is in the family header.
// FP8 KV storage, BF16 Q/P MMAC and output, FP32 accumulation.
// Scales remain device pointers, including during graph replay.
#include "sparse_gqa_paged_attention.h"
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <hip/hip_fp16.h>

namespace sparse_gqa_paged_detail {

template<bool E5M2>
struct Fp8Cache {
    using Element = uint8_t;
    using KElement = Element;
    using VElement = Element;
    using KFragment = sparse_attention_detail::mmac::Words2;
    using VFragment = uint32_t;
    static constexpr bool fp8 = true;
    static constexpr bool packed_nan = false;
    static constexpr bool shared_offsets = false;
    static constexpr bool e5 = E5M2;
    static constexpr bool mixed_predecoded = false;

    __device__ static __forceinline__ float decode(uint8_t byte) {
        if constexpr (E5M2) {
            return float(c10::Float8_e5m2(byte, c10::Float8_e5m2::from_bits()));
        } else {
            return float(c10::Float8_e4m3fn(byte, c10::Float8_e4m3fn::from_bits()));
        }
    }

    __device__ static __forceinline__ void decode_words4(uint32_t bits, uint32_t *words) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
        using Float2 = __attribute__((ext_vector_type(2))) float;
        Float2 lo, hi;
        if constexpr (E5M2) {
            lo = __builtin_amdgcn_cvt_pk_f32_bf8(bits, false);
            hi = __builtin_amdgcn_cvt_pk_f32_bf8(bits, true);
        } else {
            lo = __builtin_amdgcn_cvt_pk_f32_fp8(bits, false);
            hi = __builtin_amdgcn_cvt_pk_f32_fp8(bits, true);
        }
        __builtin_memcpy(words, &lo, 8);
        __builtin_memcpy(words + 2, &hi, 8);
        // Exhaustive gfx938 probe: finite/Inf codes are exact and every NaN
        // code yields a quiet NaN with a zero low halfword, so BF16 packing
        // keeps NaN without the per-byte canonicalization below.
#else
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = static_cast<uint8_t>(bits >> (8 * i));
            if constexpr (E5M2) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
                // Exact E5M2 -> FP32 mapping, including subnormals and signed zero.
                // Multiplying by 2^112 adjusts the bit-constructed exponent bias.
                const uint32_t magnitude = byte & 0x7fu;
                const uint32_t encoded = magnitude << 21;
                float value;
                __builtin_memcpy(&value, &encoded, sizeof(value));
                value *= 0x1p112f;
                __builtin_memcpy(words + i, &value, sizeof(value));
                words[i] |= (uint32_t(byte) & 0x80u) << 24;
                if (magnitude >= 0x7cu)
                    words[i] = magnitude == 0x7cu
                                   ? (0x7f800000u | ((uint32_t(byte) & 0x80u) << 24))
                                   : 0x7fc00000u;
#else
                const float value = decode(byte);
                __builtin_memcpy(words + i, &value, sizeof(value));
#endif
            } else {
                // Region operator's exact software E4M3FN mapping (signed zero/NaN).
                const uint32_t magnitude = byte & 0x7fu;
                const uint32_t encoded = magnitude << 20;
                float value;
                __builtin_memcpy(&value, &encoded, sizeof(value));
                value *= 0x1p120f;
                __builtin_memcpy(words + i, &value, sizeof(value));
                words[i] |= (uint32_t(byte) & 0x80u) << 24;
                if (magnitude == 0x7fu) words[i] = 0x7fc00000u;
            }
        }
        // Finite FP8 values are exactly representable in BF16. Preserve NaNs
        // even when the FP32 decoder places their payload in the lower bits.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = static_cast<uint8_t>(bits >> (8 * i));
            if (E5M2 ? ((byte & 0x7fu) > 0x7cu) : ((byte & 0x7fu) == 0x7fu))
                words[i] = 0x7fc00000u;
        }
#endif
    }

    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 decode4(uint32_t bits) {
        uint32_t words[4];
        decode_words4(bits, words);
        // Exact widening and canonical NaNs leave the low FP32 halfwords zero.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0] >> 16) | words[1], (words[2] >> 16) | words[3]};
#else
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0] >> 16) | (words[1] & 0xffff0000u),
            (words[2] >> 16) | (words[3] & 0xffff0000u)};
#endif
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &packed, sizeof(result));
        return result;
    }

    __device__ static __forceinline__ KFragment load_k(const Element *p) {
        KFragment result;
        __builtin_memcpy(&result, p, sizeof(result));
        return result;
    }
    __device__ static __forceinline__ VFragment load_v(const Element *p) {
        VFragment result;
        __builtin_memcpy(&result, p, sizeof(result));
        return result;
    }
    __device__ static __forceinline__ sparse_attention_detail::mmac::Vec8 shuffle_k(KFragment k, int lane) {
        return {decode4(__shfl(k[0], lane, 64)), decode4(__shfl(k[1], lane, 64))};
    }
    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 transpose_v(
        VFragment bits, int g, int r) {
        return decode4(sparse_attention_detail::mmac::transpose_v_fp8(bits, g, r));
    }
    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 lane_v(
        const VFragment *values, int d) {
        const int chunk = d / 4;
        // Decode directly into token pairs for the raw-P MMAC operand.
        // Preserve signed zero/Inf and canonicalize every NaN before packing.
        uint32_t words[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            decode_words4(values[4 * j + chunk], words[j]);
        const int c = d % 4;
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0][c] >> 16) | words[1][c],
            (words[2][c] >> 16) | words[3][c]};
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &packed, 8);
        return result;
    }
};

// E5M2 bytes embed exactly in binary16 as byte << 8.
// Canonicalize NaNs from the original FP8 bytes, not the half conversion payload.
template<bool PackedNan = false>
__device__ __forceinline__ void paged_e5_half_words4(uint32_t bits, uint32_t *words) {
    static_assert(sizeof(__half2) == 4, "two packed binary16 values required");
    static_assert(sizeof(float2) == 8, "two packed FP32 values required");
    if constexpr (PackedNan) {
        // Each magnitude is <=127, so adding 3 never carries across a byte.
        // Only E5 NaNs (125..127) set the high bit. Canonicalize all four
        // bytes before exact half widening, rather than four FP32 selects.
        const uint32_t nan_high = ((bits & 0x7f7f7f7fu) + 0x03030303u) & 0x80808080u;
        const uint32_t nan_mask = (nan_high >> 7) * 0xffu;
        bits = (bits & ~nan_mask) | (0x7e7e7e7eu & nan_mask);
    }
    // V_PERM(src0, src1): selectors 4..7 name src0, 0..3 name src1.
    // Produce [0,b0,0,b1] and [0,b2,0,b3] in little-endian byte order.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx936__) || defined(__gfx938__))
    const uint32_t lo_bits = __builtin_amdgcn_perm(bits, 0u, 0x05000400u);
    const uint32_t hi_bits = __builtin_amdgcn_perm(bits, 0u, 0x07000600u);
#else
    const uint32_t lo_bits = ((bits & 0xffu) << 8) | ((bits & 0xff00u) << 16);
    const uint32_t hi_bits = ((bits >> 8) & 0xff00u) | (bits & 0xff000000u);
#endif
    __half2 lo_half, hi_half;
    __builtin_memcpy(&lo_half, &lo_bits, 4);
    __builtin_memcpy(&hi_half, &hi_bits, 4);
    const float2 lo = __half22float2(lo_half);
    const float2 hi = __half22float2(hi_half);
    __builtin_memcpy(words, &lo, 8);
    __builtin_memcpy(words + 2, &hi, 8);
    if constexpr (!PackedNan) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t magnitude = (bits >> (8 * i)) & 0x7fu;
            if (magnitude > 0x7cu) words[i] = 0x7fc00000u;
        }
    }
}

// Exact E4M3FN -> binary16 -> FP32 mapping.
// Magnitude bits embed at half bit 7; scaling by 2^8 restores the E4 bias.
// Bytes 01..07/81..87 become half subnormals: exhaustive device tests are required.
__device__ __forceinline__ void paged_e4_half_words4(uint32_t bits, uint32_t *words) {
    static_assert(sizeof(__half2) == 4, "two packed binary16 values required");
    static_assert(sizeof(float2) == 8, "two packed FP32 values required");
    using Float2 = __attribute__((ext_vector_type(2))) float;
    // For each FP8 byte x, the half bytes are [(x&1)<<7, ((x>>1)&0x3f)|(x&0x80)].
    // Build all four byte bit-planes together before the two lane-local V_PERMs.
    const uint32_t low_bytes = (bits << 7) & 0x80808080u;
    const uint32_t high_bytes = ((bits >> 1) & 0x3f3f3f3fu) | (bits & 0x80808080u);
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx936__) || defined(__gfx938__))
    // V_PERM selectors 4..7 name src0, 0..3 name src1.
    const uint32_t lo_bits = __builtin_amdgcn_perm(high_bytes, low_bytes, 0x05010400u);
    const uint32_t hi_bits = __builtin_amdgcn_perm(high_bytes, low_bytes, 0x07030602u);
#else
    const uint32_t lo_bits = (low_bytes & 0xffu) | ((high_bytes & 0xffu) << 8) |
                             ((low_bytes & 0xff00u) << 8) | ((high_bytes & 0xff00u) << 16);
    const uint32_t hi_bits = ((low_bytes >> 16) & 0xffu) | ((high_bytes >> 8) & 0xff00u) |
                             ((low_bytes >> 8) & 0xff0000u) | (high_bytes & 0xff000000u);
#endif
    __half2 lo_half, hi_half;
    __builtin_memcpy(&lo_half, &lo_bits, 4);
    __builtin_memcpy(&hi_half, &hi_bits, 4);
    const float2 lo_decoded = __half22float2(lo_half);
    const float2 hi_decoded = __half22float2(hi_half);
    Float2 lo, hi;
    __builtin_memcpy(&lo, &lo_decoded, 8);
    __builtin_memcpy(&hi, &hi_decoded, 8);
    const Float2 scale = {0x1p8f, 0x1p8f};
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx936__) || defined(__gfx938__))
    lo = __builtin_hcu_pk_mul_f32(lo, scale);
    hi = __builtin_hcu_pk_mul_f32(hi, scale);
#else
    lo *= scale;
    hi *= scale;
#endif
    __builtin_memcpy(words, &lo, 8);
    __builtin_memcpy(words + 2, &hi, 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // E4 NaN embeds as finite half +/-1.875, not as half NaN.
        const uint32_t magnitude = (bits >> (8 * i)) & 0x7fu;
        if (magnitude == 0x7fu) words[i] = 0x7fc00000u;
    }
}

// Exact half-based decoding on gfx936; other targets retain their decoder.
template<bool E5M2, bool PackedNan = false, bool SharedOffsets = false>
struct Fp8CacheViaHalf : Fp8Cache<E5M2> {
    static constexpr bool packed_nan = PackedNan;
    static constexpr bool shared_offsets = SharedOffsets;
    using Element = typename Fp8Cache<E5M2>::Element;
    using KFragment = typename Fp8Cache<E5M2>::KFragment;
    using VFragment = typename Fp8Cache<E5M2>::VFragment;
    __device__ static __forceinline__ void decode_words4(uint32_t bits, uint32_t *words) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
        if constexpr (E5M2) paged_e5_half_words4<PackedNan>(bits, words);
        else paged_e4_half_words4(bits, words);
#else
        Fp8Cache<E5M2>::decode_words4(bits, words);
#endif
    }

    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 decode4(uint32_t bits) {
        uint32_t words[4];
        decode_words4(bits, words);
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0] >> 16) | words[1], (words[2] >> 16) | words[3]};
#else
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0] >> 16) | (words[1] & 0xffff0000u),
            (words[2] >> 16) | (words[3] & 0xffff0000u)};
#endif
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &packed, sizeof(result));
        return result;
    }

    // Private Mixed K only: the producer already mapped all NaNs to 0x7e.
    // E5 finite/Inf/zero embed exactly in half as byte<<8; canonical half
    // 0x7e00 widens to FP32 0x7fc00000, hence BF16 0x7fc0. No reclassification.
    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 decode4_canonical(
        uint32_t bits) {
        static_assert(E5M2, "precanonicalized K is an E5-only private policy");
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx936__) || defined(__gfx938__))
        const uint32_t lo_bits = __builtin_amdgcn_perm(bits, 0u, 0x05000400u);
        const uint32_t hi_bits = __builtin_amdgcn_perm(bits, 0u, 0x07000600u);
#else
        const uint32_t lo_bits = ((bits & 0xffu) << 8) | ((bits & 0xff00u) << 16);
        const uint32_t hi_bits = ((bits >> 8) & 0xff00u) | (bits & 0xff000000u);
#endif
        __half2 lo_half, hi_half;
        __builtin_memcpy(&lo_half, &lo_bits, 4);
        __builtin_memcpy(&hi_half, &hi_bits, 4);
        const float2 lo = __half22float2(lo_half);
        const float2 hi = __half22float2(hi_half);
        uint32_t words[4];
        __builtin_memcpy(words, &lo, 8);
        __builtin_memcpy(words + 2, &hi, 8);
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0] >> 16) | words[1], (words[2] >> 16) | words[3]};
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &packed, sizeof(result));
        return result;
    }

    __device__ static __forceinline__ sparse_attention_detail::mmac::Vec8 shuffle_k(KFragment k, int lane) {
        return {decode4(__shfl(k[0], lane, 64)), decode4(__shfl(k[1], lane, 64))};
    }

    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 transpose_v(
        VFragment bits, int g, int r) {
        return decode4(sparse_attention_detail::mmac::transpose_v_fp8(bits, g, r));
    }

    __device__ static __forceinline__ sparse_attention_detail::mmac::Half4 lane_v(
        const VFragment *values, int d) {
        const int chunk = d / 4;
        uint32_t words[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            decode_words4(values[4 * j + chunk], words[j]);
        const int c = d % 4;
        const sparse_attention_detail::mmac::Words2 packed = {
            (words[0][c] >> 16) | words[1][c],
            (words[2][c] >> 16) | words[3][c]};
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &packed, 8);
        return result;
    }
};

const float *checked_scale(const c10::optional<at::Tensor> &scale,
                           const at::Tensor &q, const char *name) {
    TORCH_CHECK(scale.has_value(), name, " must be a device FP32 tensor (use a tensor containing 1 for unit scale)");
    sparse_attention_detail::check_device(*scale, q, name);
    TORCH_CHECK(scale->scalar_type() == at::kFloat && scale->is_contiguous() && scale->numel() == 1,
                name, " must be contiguous FP32 with numel == 1");
    // Value validation belongs to the writer/model; no D2H or host scalar copy.
    return scale->data_ptr<float>();
}
} // namespace sparse_gqa_paged_detail

// ---- Public FP8 entry: format/scale contract and cache-policy selection ----
at::Tensor sparse_gqa_paged_attention_fp8(
    const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &logical_indices, const at::Tensor &block_table,
    const at::Tensor &token_to_req, c10::optional<at::Tensor> out,
    c10::optional<at::Tensor> k_scale, c10::optional<at::Tensor> v_scale,
    c10::optional<std::string> kv_cache_dtype) {
    using namespace sparse_gqa_paged_detail;
    TORCH_CHECK(q.is_cuda(), "Q must be on the accelerator");
    TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type(), "K/V must have the same dtype");
    const auto dtype = k_cache.scalar_type();
    int format = 0;
    if (dtype == at::ScalarType::Float8_e4m3fn) format = 1;
    else if (dtype == at::ScalarType::Float8_e5m2) format = 2;
    else TORCH_CHECK(dtype == at::kByte, "K/V must be E4M3FN, E5M2 or explicitly formatted uint8");
    if (kv_cache_dtype.has_value()) {
        const auto &name = *kv_cache_dtype;
        const int requested = (name == "fp8" || name == "fp8_e4m3") ? 1 : (name == "fp8_e5m2" ? 2 : 0);
        TORCH_CHECK(requested != 0, "kv_cache_dtype must be fp8_e4m3 or fp8_e5m2");
        TORCH_CHECK(format == 0 || format == requested, "kv_cache_dtype disagrees with native K/V dtype");
        format = requested;
    }
    TORCH_CHECK(format != 0, "uint8 K/V require explicit kv_cache_dtype; FP8 format cannot be inferred from bytes");
    const float *ks = checked_scale(k_scale, q, "k_scale");
    const float *vs = checked_scale(v_scale, q, "v_scale");
    // Query architecture once. run_paged_attention<> validates the common group12 contract;
    // cache policy and workload select tuning, never a particular page size.
    hipDeviceProp_t props{};
    C10_HIP_CHECK(hipGetDeviceProperties(&props, q.get_device()));
    if (format == 2) {
        if (std::string(props.gcnArchName).compare(0, 6, "gfx936") == 0) {
            const bool shared_offsets = k_cache.dim() == 4 && v_cache.dim() == 4 &&
                k_cache.stride(0) == v_cache.stride(0) &&
                k_cache.stride(1) == v_cache.stride(1) &&
                k_cache.stride(2) == v_cache.stride(2);
            if (shared_offsets)
                return run_paged_attention<Fp8CacheViaHalf<true, true, true>>(q, k_cache, v_cache, logical_indices,
                                                            block_table, token_to_req, out, ks, vs, &props);
            return run_paged_attention<Fp8CacheViaHalf<true, true>>(q, k_cache, v_cache, logical_indices,
                                                  block_table, token_to_req, out, ks, vs, &props);
        }
        return run_paged_attention<Fp8CacheViaHalf<true>>(q, k_cache, v_cache, logical_indices,
                                         block_table, token_to_req, out, ks, vs, &props);
    }
    // E4 half conversion helps populated grids; tiny grids retain the decoder.
    if (q.dim() == 3 && k_cache.dim() == 4 && k_cache.size(2) > 0 &&
        q.size(0) >= 1 + 127 / k_cache.size(2))
        return run_paged_attention<Fp8CacheViaHalf<false>>(q, k_cache, v_cache, logical_indices,
                                          block_table, token_to_req, out, ks, vs, &props);
    return run_paged_attention<Fp8Cache<false>>(q, k_cache, v_cache, logical_indices,
                               block_table, token_to_req, out, ks, vs, &props);
}
