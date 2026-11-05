// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Family FP8 implementation. Shared geometry and merges live in sparse_gqa_attention.h.
// Token-sparse GQA with FP8 K/V cache. The compute/output dtype follows Q
// (BF16 or FP16), matching op_opt3/src/qsa/sparse_attn.py.
// FP8 values are decoded without scaling before MMAC. k_scale multiplies the
// FP32 QK result; v_scale multiplies each FP32 PV contribution.
// One wave per query, V16/K4 prefetch, packed K shuffle and packed V transpose.
// Raw E4M3FN/E5M2 G12 on gfx938/gfx936 instead assembles V within each lane
// and keeps P in QK order. Other formats/targets and predecode retain fallback.
// G12 low-query workloads may split token tiles and merge FP32/log2 states.
// This changes compute scheduling, not the contiguous packed KV interface.
// This is FP8 storage with mixed-precision compute, not FP8-quantized Q/P MMAC.
// On gfx936, sufficiently reused E4M3FN K/V may be decoded once into a bounded
// temporary 16-bit buffer. Large BF16-Q/E5M2/G12 prefills may instead widen
// only V, retain raw K, and pack four independent query waves per block.
// Both preparations run every call/replay, respect the workspace budget,
// and keep all Q/P rounding and scale locations. Other inputs fall back.
// The original FP8 cache and continuous-KV interface are never changed.

#include <torch/python.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hytlass/numeric_conversion.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2fnuz.h>


#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>

#include "static_switch.h"

#include "sparse_gqa_attention.h"

namespace {

using namespace sparse_gqa_detail;

// ---- Private FP8 storage and conversion helpers ----
template <typename Element, bool Predecoded>
using KvElement = std::conditional_t<Predecoded, Element, uint8_t>;

template <bool Predecoded>
using KFragment = std::conditional_t<Predecoded, Vec8, Words2>;

template <bool Predecoded>
using VFragment = std::conditional_t<Predecoded, Half4, uint32_t>;

template <typename Element, bool Predecoded>
__device__ __forceinline__ KFragment<Predecoded> load_k(
    const KvElement<Element, Predecoded>* ptr) {
    if constexpr (Predecoded) return load8(ptr);
    else return *reinterpret_cast<const Words2*>(ptr);
}

template <typename Element, bool Predecoded>
__device__ __forceinline__ VFragment<Predecoded> load_v(
    const KvElement<Element, Predecoded>* ptr) {
    return *reinterpret_cast<const VFragment<Predecoded>*>(ptr);
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
// E5M2-only widening used by this FP8 implementation. Preserve the existing
// half bits and float upper bits exactly, including zero/Inf/NaN encodings.
// Keep this private: the other formats and architecture paths use their own codecs.
template <typename Storage>
__device__ __forceinline__ Storage decode_e5m2_via_half(const uint8_t& input) {
    union HalfBits { uint16_t as_bits; _Float16 as_value; };
    union FloatBits { uint32_t as_bits; float as_value; };
    HalfBits half;
    half.as_bits = static_cast<uint16_t>(input) << 8;
    if constexpr (std::is_same<Storage, _Float16>::value) {
        return half.as_value;
    } else {
        FloatBits value;
        value.as_value = static_cast<float>(half.as_value);
        return value.as_bits >> 16;
    }
}
#endif

enum class Fp8Format { E4M3FN, E5M2, E4M3FNUZ, E5M2FNUZ };
using Float2 = __attribute__((ext_vector_type(2))) float;

template <Fp8Format Format>
__device__ __forceinline__ float decode_fp8_byte(uint8_t bits) {
    if constexpr (Format == Fp8Format::E4M3FN)
        return static_cast<float>(c10::Float8_e4m3fn(bits, c10::Float8_e4m3fn::from_bits()));
    else if constexpr (Format == Fp8Format::E5M2)
        return static_cast<float>(c10::Float8_e5m2(bits, c10::Float8_e5m2::from_bits()));
    else if constexpr (Format == Fp8Format::E4M3FNUZ)
        return static_cast<float>(c10::Float8_e4m3fnuz(bits, c10::Float8_e4m3fnuz::from_bits()));
    else
        return static_cast<float>(c10::Float8_e5m2fnuz(bits, c10::Float8_e5m2fnuz::from_bits()));
}

template <Fp8Format Format, bool High>
__device__ __forceinline__ Float2 decode_fp8_pair(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    if constexpr (Format == Fp8Format::E4M3FN)
        return __builtin_amdgcn_cvt_pk_f32_fp8(bits, High);
    else if constexpr (Format == Fp8Format::E5M2)
        return __builtin_amdgcn_cvt_pk_f32_bf8(bits, High);
#endif
    constexpr int offset = High ? 16 : 0;
    return {decode_fp8_byte<Format>(static_cast<uint8_t>(bits >> offset)),
            decode_fp8_byte<Format>(static_cast<uint8_t>(bits >> (offset + 8)))};
}

template <typename Element, Fp8Format Format>
__device__ __forceinline__ Half4 decode_fp8x4_fallback(uint32_t bits) {
    const Float2 a = decode_fp8_pair<Format, false>(bits);
    const Float2 b = decode_fp8_pair<Format, true>(bits);
    Half4 result;
    if constexpr (std::is_same<Element, hytlass::bfloat16_t>::value) {
        // Every finite FP8 value is exactly representable in BF16; no rounding
        // is needed. Preserve float32's upper bits instead of adding a scale
        // before the conversion, which would change the model's contract.
        uint32_t words[4];
        __builtin_memcpy(words, &a, 8);
        __builtin_memcpy(words + 2, &b, 8);
        // FNUZ's software float decoder may return a NaN whose payload lies
        // only in low float32 bits. Canonicalize it before dropping those bits.
        if constexpr (Format == Fp8Format::E4M3FNUZ || Format == Fp8Format::E5M2FNUZ) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (static_cast<uint8_t>(bits >> (8 * i)) == 0x80) words[i] = 0x7fc00000u;
        }
        const Words2 packed = {(words[0] >> 16) | (words[1] & 0xffff0000u),
                               (words[2] >> 16) | (words[3] & 0xffff0000u)};
        __builtin_memcpy(&result, &packed, 8);
    } else {
#if defined(__HIP_DEVICE_COMPILE__)
        const auto lo = __builtin_amdgcn_cvt_pkrtz(a[0], a[1]);
        const auto hi = __builtin_amdgcn_cvt_pkrtz(b[0], b[1]);
        __builtin_memcpy(&result, &lo, 4);
        __builtin_memcpy(reinterpret_cast<char*>(&result) + 4, &hi, 4);
#else
        auto* p = reinterpret_cast<Element*>(&result);
        p[0] = as_element<Element>(a[0]); p[1] = as_element<Element>(a[1]);
        p[2] = as_element<Element>(b[0]); p[3] = as_element<Element>(b[1]);
#endif
    }
    return result;
}

// Preserve the old scalar half->float upper bits, including NaN payload/sign.
// Only byte-to-half construction is paired; no c10/RNE canonicalization is used.
__device__ __forceinline__ Half4 decode_e5_bf16_g12(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    const uint32_t lo_bits = __builtin_amdgcn_perm(bits, 0u, 0x05000400u);
    const uint32_t hi_bits = __builtin_amdgcn_perm(bits, 0u, 0x07000600u);
#else
    const uint32_t lo_bits = ((bits & 0xffu) << 8) | ((bits & 0xff00u) << 16);
    const uint32_t hi_bits = ((bits >> 8) & 0xff00u) | (bits & 0xff000000u);
#endif
    __half2 lo_half, hi_half;
    __builtin_memcpy(&lo_half, &lo_bits, sizeof(lo_half));
    __builtin_memcpy(&hi_half, &hi_bits, sizeof(hi_half));
    const float2 lo = __half22float2(lo_half);
    const float2 hi = __half22float2(hi_half);
    uint32_t words[4];
    __builtin_memcpy(words, &lo, sizeof(lo));
    __builtin_memcpy(words + 2, &hi, sizeof(hi));
    // Mask the upper partner explicitly: do not rely on NaN low-half payloads.
    const Words2 packed = {(words[0] >> 16) | (words[1] & 0xffff0000u),
                           (words[2] >> 16) | (words[3] & 0xffff0000u)};
    Half4 result;
    __builtin_memcpy(&result, &packed, sizeof(result));
    return result;
}

template <typename Element, Fp8Format Format, bool G12Codec = false>
__device__ __forceinline__ Half4 decode_fp8x4(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    if constexpr (G12Codec && Format == Fp8Format::E5M2 &&
                  std::is_same<Element, hytlass::bfloat16_t>::value)
        return decode_e5_bf16_g12(bits);
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    if constexpr (Format == Fp8Format::E4M3FN) {
        // Exact power-of-two decode, validated for all 256 byte patterns on
        // gfx936. Unlike uint82half's E4M3 software path, preserve signed zero
        // and map both 0x7f/0xff encodings to NaN, not +/-480.
        uint32_t words[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t byte = static_cast<uint8_t>(bits >> (8 * i));
            const uint32_t magnitude = byte & 0x7fu;
            const float value = __uint_as_float(magnitude << 20) * 0x1p120f;
            uint32_t raw = __float_as_uint(value) | ((byte & 0x80u) << 24);
            if (magnitude == 0x7fu) raw = 0x7fc00000u;
            words[i] = raw;
        }
        Half4 result;
        if constexpr (std::is_same<Element, hytlass::bfloat16_t>::value) {
            const Words2 packed = {(words[0] >> 16) | (words[1] & 0xffff0000u),
                                   (words[2] >> 16) | (words[3] & 0xffff0000u)};
            __builtin_memcpy(&result, &packed, 8);
        } else {
            const auto lo = __builtin_amdgcn_cvt_pkrtz(
                __uint_as_float(words[0]), __uint_as_float(words[1]));
            const auto hi = __builtin_amdgcn_cvt_pkrtz(
                __uint_as_float(words[2]), __uint_as_float(words[3]));
            __builtin_memcpy(&result, &lo, 4);
            __builtin_memcpy(reinterpret_cast<char*>(&result) + 4, &hi, 4);
        }
        return result;
    } else if constexpr (Format == Fp8Format::E5M2) {
        // E5M2 has the same exponent bias as FP16. The existing helper embeds
        // the byte in FP16, preserving zero/Inf/NaN, then widens if needed.
        uint16_t words[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = static_cast<uint8_t>(bits >> (8 * i));
            if constexpr (std::is_same<Element, hytlass::half_t>::value) {
                const _Float16 value = decode_e5m2_via_half<_Float16>(byte);
                __builtin_memcpy(words + i, &value, 2);
            } else {
                words[i] = decode_e5m2_via_half<uint16_t>(byte);
            }
        }
        Half4 result;
        __builtin_memcpy(&result, words, 8);
        return result;
    }
#endif
    // Keep gfx938's native packed conversion, other targets' C10 conversion,
    // and both FNUZ formats unchanged. No FP8 format is reinterpreted as another.
    return decode_fp8x4_fallback<Element, Format>(bits);
}


// Same FP8 widening and upper-bit semantics as decode_fp8x4_fallback, but
// arrange token pairs directly instead of building/transposing four Half4s.
template <Fp8Format Format>
__device__ __forceinline__ void paired_value_operands(
    uint32_t a, uint32_t b, uint32_t c, uint32_t d, Half4 (&output)[4]) {
    const uint32_t raw[4] = {a, b, c, d};
    uint32_t words[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const Float2 lo = decode_fp8_pair<Format, false>(raw[j]);
        const Float2 hi = decode_fp8_pair<Format, true>(raw[j]);
        __builtin_memcpy(words[j], &lo, 8);
        __builtin_memcpy(words[j] + 2, &hi, 8);
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
        const Words2 packed = {
            (words[0][column] >> 16) | (words[1][column] & 0xffff0000u),
            (words[2][column] >> 16) | (words[3][column] & 0xffff0000u)};
        __builtin_memcpy(&output[column], &packed, 8);
    }
}

// Four FP8 values occupy one word. Transpose while still byte-packed, then
// widen to the MMAC operand. This halves initial shuffle traffic versus BF16.
template <bool UseSwizzle>
__device__ __forceinline__ uint32_t transpose_v_fp8(uint32_t bits, int g, int r) {
    uint32_t x = __shfl(bits, g * 16 + (r & 3) * 4 + (r >> 2), kWaveSize);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    if constexpr (UseSwizzle) {
        // Fixed quad XOR1/XOR2 exchanges. Keep the initial irregular shuffle.
        uint32_t other = __builtin_amdgcn_ds_swizzle(x, 0x80b1);
        x = __builtin_amdgcn_perm(other, x, (r & 1) ? 0x03070105u : 0x06020400u);
        other = __builtin_amdgcn_ds_swizzle(x, 0x804e);
        return __builtin_amdgcn_perm(other, x, (r & 2) ? 0x03020706u : 0x05040100u);
    }
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    uint32_t other = __builtin_amdgcn_mov_dpp(x, 0xb1, 15, 15, false);
    x = __builtin_amdgcn_perm(other, x, (r & 1) ? 0x03070105u : 0x06020400u);
    other = __builtin_amdgcn_mov_dpp(x, 0x4e, 15, 15, false);
    return __builtin_amdgcn_perm(other, x, (r & 2) ? 0x03020706u : 0x05040100u);
#else
    uint32_t other = __shfl_xor(x, 1, kWaveSize);
    x = (r & 1) ? ((x & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                : ((x & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
    other = __shfl_xor(x, 2, kWaveSize);
    return (r & 2) ? ((x & 0xffff0000u) | (other >> 16))
                   : ((x & 0xffffu) | (other << 16));
#endif
}

// The 16-bit V layout used by the predecoded path is the same as the
// FP16/BF16 kernel. These are bitwise moves, not arithmetic conversions.
__device__ __forceinline__ Half4 transpose_v_16(Half4 value, int g, int r) {
    return sparse_attention_detail::mmac::transpose_v_dpp(value, g, r);
}

// Repack the QK accumulator's token=g+4e layout for the PV MMAC operand.
template <bool UseSwizzle>
__device__ __forceinline__ float shuffle_xor16(float value) {
    return sparse_gqa_detail::shuffle_xor16(value);
}

template <bool UseSwizzle>
__device__ __forceinline__ Float4 transpose_probability(Float4 p, int g) {
    return sparse_attention_detail::mmac::transpose_p(p, g);
}

// ---- Main attention kernel: retain numerical order and launch options ----
template <typename Element, Fp8Format Format,
          bool Predecoded = false, bool DirectPV = false, bool EnableG12FastPath = false, bool SplitMode = false,
          bool VPredecoded = false, int QueryRows = 1, int CooperativeWaves = 1, int PrefixMode = 0>
__global__ __launch_bounds__(kWaveSize * QueryRows * CooperativeWaves) void sparse_gqa_attention_fp8_kernel(
    const Element* __restrict__ q, const KvElement<Element, Predecoded>* __restrict__ k,
    const KvElement<Element, (Predecoded || VPredecoded)>* __restrict__ v, Element* __restrict__ out,
    const int32_t* __restrict__ indices, const int32_t* __restrict__ cu_q,
    const int32_t* __restrict__ cu_k, const int32_t* __restrict__ kv_lens,
    int hq, int hkv, int topk, float k_scale, float v_scale,
    int64_t index_query_stride, int64_t index_head_stride, int reference_slot_block_size,
    float* __restrict__ partial, float* __restrict__ lse, int splits, int total_q,
    const int32_t* __restrict__ prefix_lengths = nullptr, int prefix_capacity = 0) {
    static_assert(QueryRows == 1 || QueryRows == 4, "validated independent query counts only");
    static_assert(QueryRows == 1 || (VPredecoded && !Predecoded && EnableG12FastPath && !SplitMode),
                  "independent query waves are limited to unsplit Mixed-V G12");
    static_assert(CooperativeWaves == 1 || PrefixMode == 0,
                  "cooperative decode and active-prefix consumers are disjoint");
    static_assert(PrefixMode >= 0 && PrefixMode <= 2, "0=original, 1=prefix raw fallback, 2=prefix Mixed-V");
    static_assert(PrefixMode == 0 || (!Predecoded && !SplitMode && EnableG12FastPath),
                  "prefix cache is restricted to the non-split G12 route");
    static_assert(PrefixMode != 2 || VPredecoded, "prefix Mixed-V must consume decoded V");
    static_assert(CooperativeWaves == 1 || (QueryRows == 1 &&
                  (CooperativeWaves == 2 || CooperativeWaves == 4) &&
                  !Predecoded && !VPredecoded && EnableG12FastPath &&
                  Format == Fp8Format::E5M2 &&
                  std::is_same<Element, hytlass::bfloat16_t>::value),
                  "cooperative decode W2/W4 path is BF16-Q/E5/G12 only");
    const int wave = CooperativeWaves == 1 ? 0 : static_cast<int>(threadIdx.x) / kWaveSize;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    // Enable only for the validated E4M3FN/G12 path (BF16 or FP16 Q).
    // Other formats, group sizes and architectures retain their prior path.
    constexpr bool use_swizzle = Format == Fp8Format::E4M3FN && EnableG12FastPath;
#else
    constexpr bool use_swizzle = false;
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx938__) && !defined(__gfx936__)
    constexpr bool inlane_v = false;
#else
    // Raw E4M3FN/E5M2 G12 caches benefit on both validated targets.
    // A gfx936 predecoded cache keeps the original 16-bit V layout.
    constexpr bool inlane_v = EnableG12FastPath
        && (Format == Fp8Format::E4M3FN || Format == Fp8Format::E5M2)
        && !Predecoded;
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool pair_v = inlane_v && !VPredecoded
        && std::is_same<Element, hytlass::bfloat16_t>::value;
#else
    constexpr bool pair_v = false;
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool late_v = inlane_v && !SplitMode && Format == Fp8Format::E5M2
        && std::is_same<Element, hytlass::bfloat16_t>::value;
#endif
    const int batch = static_cast<int>(blockIdx.y) / hkv;
    const int kv_head = static_cast<int>(blockIdx.y) % hkv;
    if constexpr (PrefixMode != 0) {
        // Summary and producer precede both consumers on this same stream.
        // Each batch/head uses exactly one consumer, including graph replays.
        const bool prefix_eligible = prefix_lengths[int64_t(batch) * hkv + kv_head] <= prefix_capacity;
        if constexpr (PrefixMode == 1) { if (prefix_eligible) return; }
        else { if (!prefix_eligible) return; }
    }
    const int qr = static_cast<int>(blockIdx.x) * QueryRows +
        (QueryRows == 1 ? 0 : static_cast<int>(threadIdx.x) / kWaveSize);
    const int qs = cu_q[batch], qlen = cu_q[batch + 1] - qs;
    if (qr >= qlen) return;
    const int query = qs + qr;
    const int ks = cu_k[batch], klen = kv_lens[batch];
    // Only V is packed into the bounded per-request prefix workspace. K keeps
    // the original cu_k base; never substitute this base in QK address math.
    const int64_t v_token_base = PrefixMode == 2 ? int64_t(batch) * prefix_capacity : int64_t(ks);
    const int group = CooperativeWaves > 1 ? 12 : hq / hkv;
    const int head_start = kv_head * group;
    const int active = CooperativeWaves > 1 ? 12 : min(16, group);
    constexpr int KWindow = CooperativeWaves > 1 ? 8 : kKPrefetchSlices;
    // The target rounds the consumed slot window to its reference BLOCK_N.
    // For conventional -1 padding this agrees with the visible prefix, while
    // preserving the target's behavior if those rounded slots contain tokens.
    const int64_t visible = static_cast<int64_t>(klen) - qlen + qr + 1;
    const int64_t row_topk = min(static_cast<int64_t>(topk), max(int64_t{0}, visible));
    const int reference_mask = reference_slot_block_size - 1; // reference sizes are powers of two
    const int count = static_cast<int>(min(static_cast<int64_t>(topk),
        (row_topk + reference_mask) & ~static_cast<int64_t>(reference_mask)));
    const int lane = QueryRows == 1 && CooperativeWaves == 1
        ? static_cast<int>(threadIdx.x) : static_cast<int>(threadIdx.x) % kWaveSize;
    const int r = lane % 16, g = lane / 16;
    const int64_t qb = (static_cast<int64_t>(query) * hq + head_start) * kHeadDim;
    const int32_t* ir = indices + static_cast<int64_t>(query) * index_query_stride + kv_head * index_head_stride;
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    constexpr float softmax_scale = 0.0625f*kLog2E;
//Q*K_SCALE
    // Quantize the scaled Q once, matching the existing MMAC/softmax math.
    Vec8 qv[kQkSlices];
#pragma unroll
    for (int i = 0; i < kQkSlices; ++i) {
        qv[i] = r < active ? load8(q + qb + r * kHeadDim + i * 32 + g * 8) : zero;
        Element* elements = reinterpret_cast<Element*>(&qv[i]);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            elements[e] = as_element<Element>(as_float(elements[e]) * softmax_scale);
    }
    Float4 pv[kVFragments];
#pragma unroll
    for (int d = 0; d < kVFragments; ++d) pv[d] = {0, 0, 0, 0};
    float row_max = kNegativeInfinity, row_sum = 0;
    bool has_seen_valid = false;
    bool empty_first_reference = count == 0;

    const int split = SplitMode ? static_cast<int>(blockIdx.z) : 0;
    bool leading_empty = false;
    if constexpr (SplitMode || CooperativeWaves > 1) {
        const int value = lane < min(count, reference_slot_block_size) ? ir[lane] : -1;
        leading_empty = __ballot(value >= 0 && value < klen) == 0;
    }
    const int tiles = static_cast<int>((int64_t(count) + 15) / 16);
    const int first = CooperativeWaves > 1
        ? static_cast<int>(int64_t(split * CooperativeWaves + wave) * tiles / (splits * CooperativeWaves) * 16)
        : (SplitMode ? static_cast<int>(int64_t(split) * tiles / splits * 16) : 0);
    const int last = CooperativeWaves > 1
        ? static_cast<int>(min(int64_t(count), int64_t(split * CooperativeWaves + wave + 1) * tiles / (splits * CooperativeWaves) * 16))
        : (SplitMode ? static_cast<int>(min(int64_t(count), int64_t(split + 1) * tiles / splits * 16)) : count);
    for (int start = first; start < last && !leading_empty; start += kTileTokens) {
        int token = -1;
        if (g == 0 && start + r < count) {
            const int value = ir[start + r];
            token = value >= 0 && value < klen ? value : -1;
        }
        token = __shfl(token, r, kWaveSize);
        const bool tile_has_valid = __ballot(token >= 0) != 0;
        if constexpr (SplitMode || CooperativeWaves > 1) {
            if (!tile_has_valid) continue;
        } else if (!has_seen_valid && !tile_has_valid) {
            // A reference block can span 2 or 4 hardware tiles. Do not run
            // softmax on an empty prefix (-inf - -inf); only a fully empty
            // first reference block poisons the row, matching Triton.
            if (start + kTileTokens >= count || ((start + kTileTokens) & reference_mask) == 0) {
                empty_first_reference = true;
                break;
            }
            continue;
        }
        has_seen_valid = has_seen_valid || tile_has_valid;

        // V16: issue all V loads before QK/softmax to separate load and use.
        VFragment<(Predecoded || VPredecoded)> v_prefetch[kVFragments];
        VFragment<(Predecoded || VPredecoded)> lane_values[4][4];

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
        if constexpr (!late_v) {
#endif
        if constexpr (inlane_v) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int vt = __shfl(token, g + 4 * j, kWaveSize);
                const bool live = vt >= 0;
                const int64_t value_token = PrefixMode == 2
                    ? v_token_base + (live ? vt : 0)
                    : static_cast<int64_t>(ks + (live ? vt : 0));
                const int64_t base = (value_token * hkv + kv_head) * kHeadDim;
                __builtin_memcpy(lane_values[j], __builtin_assume_aligned(v + base + r * 16, 16), sizeof(lane_values[j]));
                if (!live) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) lane_values[j][c] = VFragment<(Predecoded || VPredecoded)>{};
                }
            }
        } else {
        const int pref_token = __shfl(token, lane / 4, kWaveSize);
        const bool pref_valid = pref_token >= 0 && pref_token < klen;
        const int64_t value_token = PrefixMode == 2
            ? v_token_base + (pref_valid ? pref_token : 0)
            : static_cast<int64_t>(ks + (pref_valid ? pref_token : 0));
        const int64_t pref_base = (value_token * hkv + kv_head) * kHeadDim;
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
            v_prefetch[d] = load_v<Element, (Predecoded || VPredecoded)>(
            v + pref_base + d * 16 + (lane % 4) * 4);
        }

    if (!pref_valid) {
        for (int d = 0; d < kVFragments; ++d) {
            v_prefetch[d] = VFragment<(Predecoded || VPredecoded)>{};
        }
    }
        }

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
        }
#endif

// K4: 根据当前 tile 的索引有效性，选择全有效快路径或通用路径。
Float4 score = {0, 0, 0, 0};

const int natural_token = __shfl(token, lane / 4, kWaveSize);
const bool natural_valid =
    natural_token >= 0 && natural_token < klen;

// 所有 lane 都参与投票；没有无效 lane 时，整个 wave 进入快路径。
if (__ballot(!natural_valid) == 0ULL) {
    // ---------- 快路径：当前 16 个索引槽位全部有效 ----------
    const int64_t natural_base =
        (static_cast<int64_t>(ks + natural_token) * hkv + kv_head)
        * kHeadDim;

    KFragment<Predecoded> k_window[KWindow];

    // 初始窗口直接加载，不需要有效性选择或清零。
#pragma unroll
    for (int p = 0; p < KWindow; ++p) {
        k_window[p] = load_k<Element, Predecoded>(
            k + natural_base + p * 32 + (lane % 4) * 8);
    }

#pragma unroll
    for (int i = 0; i < kQkSlices; ++i) {
        const KFragment<Predecoded> natural_k =
            k_window[i % KWindow];

        // 后续片段同样直接加载，只保留维度循环的收尾判断。
        if (i + KWindow < kQkSlices) {
            k_window[i % KWindow] =
                load_k<Element, Predecoded>(
                    k + natural_base
                      + (i + KWindow) * 32
                      + (lane % 4) * 8);
        }

        Vec8 kv;

        if constexpr (Predecoded) {
            // 已预解码：输入已是 BF16/FP16，只需重排。
            kv = shuffle8(natural_k, r * 4 + g);
        } else {
            // 原始 FP8：先搬运打包字节，再解码成 MMAC 输入。
            const uint32_t lo =
                __shfl(natural_k[0], r * 4 + g, kWaveSize);
            const uint32_t hi =
                __shfl(natural_k[1], r * 4 + g, kWaveSize);

            kv = {
                decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(lo),
                decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(hi)
            };
        }

        mmac<Element>(qv[i].x, kv.x, score);
        mmac<Element>(qv[i].y, kv.y, score);
    }
} else {
    // ---------- 通用路径：包含无效索引或不足 16 项的尾块 ----------
    // 保持当前版本的占位地址策略：无效 token 使用 request 内 token0。
    const int64_t natural_base =
        (static_cast<int64_t>(
            ks + (natural_valid ? natural_token : 0))
         * hkv + kv_head) * kHeadDim;

    KFragment<Predecoded> k_window[KWindow];

    // 保持你已修改的“初始统一加载，再对无效 lane 清零”。
#pragma unroll
    for (int p = 0; p < KWindow; ++p) {
        k_window[p] = load_k<Element, Predecoded>(
            k + natural_base + p * 32 + (lane % 4) * 8);
    }

    if (!natural_valid) {
        for (int p = 0; p < KWindow; ++p) {
            k_window[p] = KFragment<Predecoded>{};
        }
    }

#pragma unroll
    for (int i = 0; i < kQkSlices; ++i) {
        const KFragment<Predecoded> natural_k =
            k_window[i % KWindow];

        // 通用路径保留后续加载的有效性判断。
        if (i + KWindow < kQkSlices) {
            k_window[i % KWindow] = natural_valid
                ? load_k<Element, Predecoded>(
                    k + natural_base
                      + (i + KWindow) * 32
                      + (lane % 4) * 8)
                : KFragment<Predecoded>{};
        }

        Vec8 kv;

        if constexpr (Predecoded) {
            kv = shuffle8(natural_k, r * 4 + g);
        } else {
            const uint32_t lo =
                __shfl(natural_k[0], r * 4 + g, kWaveSize);
            const uint32_t hi =
                __shfl(natural_k[1], r * 4 + g, kWaveSize);

            kv = {
                decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(lo),
                decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(hi)
            };
        }

        mmac<Element>(qv[i].x, kv.x, score);
        mmac<Element>(qv[i].y, kv.y, score);
    }
}

// 原来的 Online softmax 从这里继续，保持不变。


#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
        // gfx936 unsplit G12: K4/QK finish before V16 becomes live.
        // SplitMode retains the original earlier V prefetch.
        if constexpr (late_v) {
        if constexpr (inlane_v) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int vt = __shfl(token, g + 4 * j, kWaveSize);
                const bool live = vt >= 0;
                const int64_t value_token = PrefixMode == 2
                    ? v_token_base + (live ? vt : 0)
                    : static_cast<int64_t>(ks + (live ? vt : 0));
                const int64_t base = (value_token * hkv + kv_head) * kHeadDim;
                __builtin_memcpy(lane_values[j], __builtin_assume_aligned(v + base + r * 16, 16), sizeof(lane_values[j]));
                if (!live) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) lane_values[j][c] = VFragment<(Predecoded || VPredecoded)>{};
                }
            }
        } else {
        const int pref_token = __shfl(token, lane / 4, kWaveSize);
        const bool pref_valid = pref_token >= 0 && pref_token < klen;
        const int64_t value_token = PrefixMode == 2
            ? v_token_base + (pref_valid ? pref_token : 0)
            : static_cast<int64_t>(ks + (pref_valid ? pref_token : 0));
        const int64_t pref_base = (value_token * hkv + kv_head) * kHeadDim;
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
            v_prefetch[d] = load_v<Element, (Predecoded || VPredecoded)>(
            v + pref_base + d * 16 + (lane % 4) * 4);
        }

    if (!pref_valid) {
        for (int d = 0; d < kVFragments; ++d) {
            v_prefetch[d] = VFragment<(Predecoded || VPredecoded)>{};
        }
    }
        }
        }
#endif
        // Online softmax: every lane's four accumulators belong to one head.
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = __shfl(token, g + e * 4, kWaveSize);
            score[e] = r < active && t >= 0 ? score[e] * k_scale : kNegativeInfinity;
        }
        float m = maximum(maximum(score[0], score[1]), maximum(score[2], score[3]));
        m = maximum(m, shuffle_xor16<use_swizzle>(m));
        m = maximum(m, __shfl_xor(m, 32, kWaveSize));
        m = maximum(row_max, m);
        const float alpha = SplitMode && row_sum == 0 ? 0.0f : exp2_fast(row_max - m);
        Float4 prob;
#pragma unroll
        for (int e = 0; e < 4; ++e){
            prob[e] = (!SplitMode || (r < active && __shfl(token, g + e * 4, kWaveSize) >= 0))
                ? exp2_fast(score[e] - m) : 0.0f;
        } 
        float sum = (prob[0] + prob[1]) + (prob[2] + prob[3]);
        sum += shuffle_xor16<use_swizzle>(sum);
        sum += __shfl_xor(sum, 32, kWaveSize);
        row_sum = row_sum * alpha + sum;
        row_max = m;
        // Keep the validated per-head policy for gfx936 E4/G12: a wave-wide
        // ballot rescaled unchanged heads and regressed long prefill there.
        // Other paths skip only a uniform, exact multiply-by-one. NaN alpha
        // still takes the arithmetic path rather than being hidden.
        const bool rescale = use_swizzle
            ? (r < active && alpha != 1.0f)
            : (!has_seen_valid || __ballot(r < active && alpha != 1.0f) != 0);
        if (rescale) {
#pragma unroll
            for (int d = 0; d < kVFragments; ++d) {
#pragma unroll
                for (int e = 0; e < 4; ++e) pv[d][e] *= alpha;
            }
        }

        Half4 pvec = zero.x;
        Element* pe = reinterpret_cast<Element*>(&pvec);
        const Float4 transposed = inlane_v ? prob : transpose_probability<use_swizzle>(prob, g);
#pragma unroll
        for (int e = 0; e < 4; ++e) pe[e] = as_element<Element>(transposed[e]);
        if constexpr (pair_v) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                Half4 operands[4];
                paired_value_operands<Format>(lane_values[0][c], lane_values[1][c],
                    lane_values[2][c], lane_values[3][c], operands);
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    if constexpr (DirectPV) mmac<Element>(pvec, operands[b], pv[4 * c + b]);
                    else {
                        Float4 contribution = {0, 0, 0, 0};
                        mmac<Element>(pvec, operands[b], contribution);
#pragma unroll
                        for (int e = 0; e < 4; ++e) pv[4 * c + b][e] += contribution[e] * v_scale;
                    }
                }
            }
        } else if constexpr (inlane_v) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                Half4 rows[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (Predecoded || VPredecoded) rows[j] = lane_values[j][c];
                    else rows[j] = decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(lane_values[j][c]);
                }
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const Half4 vv = assemble_inlane_value_operand(rows[0], rows[1], rows[2], rows[3], b);
                    if constexpr (DirectPV) mmac<Element>(pvec, vv, pv[4 * c + b]);
                    else {
                        Float4 contribution = {0, 0, 0, 0};
                        mmac<Element>(pvec, vv, contribution);
#pragma unroll
                        for (int e = 0; e < 4; ++e) pv[4 * c + b][e] += contribution[e] * v_scale;
                    }
                }
            }
        } else {
        const int vt = __shfl(token, lane / 4, kWaveSize);
        VFragment<(Predecoded || VPredecoded)> natural = vt >= 0 ? v_prefetch[0] : VFragment<(Predecoded || VPredecoded)>{};
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
            VFragment<(Predecoded || VPredecoded)> next{};
            if (d + 1 < kVFragments && vt >= 0) next = v_prefetch[d + 1];
            Half4 vv;
            if constexpr (Predecoded || VPredecoded) {
                vv = transpose_v_16(natural, g, r);
            } else {
                const uint32_t packed_v = transpose_v_fp8<use_swizzle>(natural, g, r);
                vv = decode_fp8x4<Element, Format, (EnableG12FastPath && (VPredecoded || CooperativeWaves > 1))>(packed_v);
            }
            if constexpr (DirectPV) {
                // Host dispatch selects this only when v_scale is exactly 1.
                mmac<Element>(pvec, vv, pv[d]);
            } else {
                Float4 contribution = {0, 0, 0, 0};
                mmac<Element>(pvec, vv, contribution);
#pragma unroll
                for (int e = 0; e < 4; ++e) pv[d][e] += contribution[e] * v_scale;
            }
            natural = next;
        }
        }
        // The reference produces NaN if its leading reference block is empty.
        // Do not poison a row merely because one of our smaller 16-token
        // subtiles is empty, or because an empty tail follows valid tokens.
        if ((start + kTileTokens >= count || ((start + kTileTokens) & reference_mask) == 0)
                && !SplitMode && CooperativeWaves == 1 && !has_seen_valid)
        {
            empty_first_reference = true;
            break;  // an empty leading reference block poisons the entire row
        }
    }

    // Two/four waves collaborate on ONE query. Keep wave0's numerator in its
    // registers and expose only waves1..3 through LDS, as in the Paged path.
    // Raw numerator/max/sum preserve GQA's Q/P rounding and V-scale placement.
    if constexpr (CooperativeWaves > 1) {
        constexpr int SharedStride = 260;
        constexpr int Heads = 12;
        extern __shared__ float shared[];
        float* shared_max = shared + (CooperativeWaves - 1) * Heads * SharedStride;
        float* shared_sum = shared_max + CooperativeWaves * Heads;
        const bool poison = leading_empty || isnan(row_sum) || isnan(row_max);
        if (r < Heads) {
#pragma unroll
            for (int d = 0; d < kVFragments; ++d) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (wave != 0)
                        shared[((wave - 1) * Heads + r) * SharedStride +
                            value_output_column(d, g, e, inlane_v)] =
                            row_sum > 0 ? pv[d][e] : 0.0f;
                }
            }
            if (g == 0) {
                shared_max[wave * Heads + r] = poison
                    ? __int_as_float(0x7fc00000) : (row_sum > 0 ? row_max : -INFINITY);
                shared_sum[wave * Heads + r] = row_sum > 0 ? row_sum : 0.0f;
            }
        }
        __syncthreads();
        if (wave != 0) return;
        if (r < Heads) {
            float maximum = -INFINITY;
            bool merged_poison = false;
            float weights[CooperativeWaves];
#pragma unroll
            for (int w = 0; w < CooperativeWaves; ++w) {
                const float maximum_w = shared_max[w * Heads + r];
                merged_poison = merged_poison || isnan(maximum_w);
                maximum = fmaxf(maximum, maximum_w);
            }
            float denominator = 0;
#pragma unroll
            for (int w = 0; w < CooperativeWaves; ++w) {
                weights[w] = maximum > -INFINITY
                    ? exp2_fast(shared_max[w * Heads + r] - maximum) : 0.0f;
                denominator += weights[w] * shared_sum[w * Heads + r];
            }
#pragma unroll
            for (int d = 0; d < kVFragments; ++d) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float value = 0;
#pragma unroll
                    for (int w = 0; w < CooperativeWaves; ++w) {
                        const float numerator = w == 0
                            ? (row_sum > 0 ? pv[d][e] : 0.0f)
                            : shared[((w - 1) * Heads + r) * SharedStride +
                                value_output_column(d, g, e, inlane_v)];
                        value += numerator * weights[w];
                    }
                    pv[d][e] = merged_poison ? __int_as_float(0x7fc00000) : value;
                }
            }
            row_sum = merged_poison ? __int_as_float(0x7fc00000) : denominator;
            row_max = merged_poison ? __int_as_float(0x7fc00000) : maximum;
        }
        empty_first_reference = empty_first_reference || leading_empty;
    }
    // Four adjacent columns in the G12 in-lane layout share one vector store.
    // Fresh contiguous out/partial allocation and D256 guarantee 8/16B stores.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool vector_store = inlane_v && std::is_same<Element, hytlass::bfloat16_t>::value;
#else
    constexpr bool vector_store = false;
#endif
    if constexpr (vector_store) {
        if constexpr (SplitMode) {
            const int64_t state = (int64_t(split) * total_q + query) * hq + head_start + r;
            const bool poison = leading_empty || isnan(row_sum) || isnan(row_max);
            const float inverse = row_sum > 0 ? 1.0f / row_sum : 0;
            if (r < active) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        Float4 packed;
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            packed[b] = poison ? __int_as_float(0x7fc00000)
                                               : pv[4 * c + b][e] * inverse;
                        const int column = 16 * (g + 4 * e) + 4 * c;
                        *reinterpret_cast<Float4*>(partial + state * kHeadDim + column) = packed;
                    }
                }
            }
            if (r < active && g == 0)
                lse[state] = poison ? 0 : (row_sum > 0 ? row_max + log2f(row_sum) : -INFINITY);
        } else {
            if (r < active) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        Half4 packed;
                        Element* elements = reinterpret_cast<Element*>(&packed);
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            elements[b] = empty_first_reference ? canonical_nan<Element>()
                                : as_element<Element>(pv[4 * c + b][e] / row_sum);
                        const int column = 16 * (g + 4 * e) + 4 * c;
                        *reinterpret_cast<Half4*>(out + qb + r * kHeadDim + column) = packed;
                    }
                }
            }
        }
    } else {
    if constexpr (SplitMode) {
        const int64_t state = (int64_t(split) * total_q + query) * hq + head_start + r;
        const bool poison = leading_empty || isnan(row_sum) || isnan(row_max);
        const float inverse = row_sum > 0 ? 1.0f / row_sum : 0;
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (r < active)
                    partial[state * kHeadDim + value_output_column(d, g, e, inlane_v)] =
                        poison ? __int_as_float(0x7fc00000) : pv[d][e] * inverse;
            }
        }
        if (r < active && g == 0)
            lse[state] = poison ? 0 : (row_sum > 0 ? row_max + log2f(row_sum) : -INFINITY);
    } else {
    #pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
    #pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (r < active)
                    out[qb + r * kHeadDim + value_output_column(d, g, e, inlane_v)] =
                        empty_first_reference ? canonical_nan<Element>()
                        : as_element<Element>(pv[d][e] / row_sum);
            }
        }
    }
    }
}


// ---- Decode-specific merge: only the existing FP8 dispatch uses this variant ----
// Private GQA variant of the Paged tiny32 layout.  Its inputs retain GQA's
// normalized FP32 partials and log2 LSE; no cross-family launcher is referenced.
template <typename Element, int Parts>
__global__ __launch_bounds__(64) void sparse_gqa_decode_merge_columns32(
    const float* partial, const float* lse, Element* out, int64_t rows) {
    const int lane = threadIdx.x;
    const int column_lane = lane % 32, partition_lane = lane / 32;
    const int64_t row = blockIdx.x;
    const int column = blockIdx.y * 32 + column_lane;
    constexpr int PerLane = Parts / 2;
    float logs[PerLane], values[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const int part = partition_lane + i * 2;
        logs[i] = lse[int64_t(part) * rows + row];
        values[i] = partial[(int64_t(part) * rows + row) * kHeadDim + column];
    }
    float peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) peak = fmaxf(peak, logs[i]);
    peak = fmaxf(peak, __shfl_xor(peak, 32, 64));
    float denominator = 0, value = 0;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const float weight = peak > -INFINITY ? exp2_fast(logs[i] - peak) : 0;
        denominator += weight;
        value += values[i] * weight;
    }
    denominator += __shfl_xor(denominator, 32, 64);
    value += __shfl_xor(value, 32, 64);
    if (partition_lane == 0)
        out[row * kHeadDim + column] =
            as_element<Element>(denominator > 0 ? value / denominator : 0);
}

template <typename Element>
void launch_gqa_decode_merge(const at::Tensor& partial, const at::Tensor& lse,
                             at::Tensor& out, int splits, hipStream_t stream) {
    const int64_t rows = out.size(0) * out.size(1);
    const dim3 grid(static_cast<unsigned int>(rows), 8);
#define GQA_DECODE_MERGE(P) \
    sparse_gqa_decode_merge_columns32<Element, P><<<grid, 64, 0, stream>>>( \
        partial.data_ptr<float>(), lse.data_ptr<float>(), \
        reinterpret_cast<Element*>(out.data_ptr()), rows)
    switch (splits) {
      case 16: { GQA_DECODE_MERGE(16); break; }
      case 32: { GQA_DECODE_MERGE(32); break; }
      default: { launch_split_merge<Element>(partial, lse, out, splits, stream); return; }
    }
#undef GQA_DECODE_MERGE
    C10_HIP_CHECK(hipGetLastError());
}


// ---- Per-call KV preparation: workspace owners stay in the public invocation ----
template <typename Element, Fp8Format Format>
__global__ void decode_kv_cache(const uint32_t* k, const uint32_t* v,
                               Element* decoded_k, Element* decoded_v, int64_t words) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) return;
    const Half4 a = decode_fp8x4<Element, Format>(k[i]);
    const Half4 b = decode_fp8x4<Element, Format>(v[i]);
    *reinterpret_cast<Half4*>(decoded_k + 4 * i) = a;
    *reinterpret_cast<Half4*>(decoded_v + 4 * i) = b;
}

// Every invocation/replay widens the current V without applying its scale.
// Use exactly the existing decoder, including signed-zero and NaN upper bits.
template <typename Element>
__global__ void sparse_gqa_decode_v_e5_kernel(
    const uint32_t* input, Element* output, int64_t words) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) return;
    const Half4 value = decode_fp8x4<Element, Fp8Format::E5M2>(input[i]);
    *reinterpret_cast<Half4*>(output + 4 * i) = value;
}

// Bounded active-prefix summary. This scans precisely the slot window the
// attention kernel consumes, not all TOPK slots or all physically allocated KV.
// No deduplication/reordering and no CPU readback are involved. A capped length
// > prefix_capacity means this batch/head must use the original raw consumer.
__global__ void sparse_gqa_v_prefix_summary_kernel(
    const int32_t* indices, const int32_t* cu_q, const int32_t* kv_lens,
    int32_t* prefix_lengths, int hkv, int topk, int reference_slot_block_size,
    int64_t index_query_stride, int64_t index_head_stride, int prefix_capacity) {
    constexpr int kQueriesPerBlock = 8;
    const int batch = int(blockIdx.y) / hkv;
    const int head = int(blockIdx.y) % hkv;
    const int64_t bh = int64_t(batch) * hkv + head;
    __shared__ int scratch[4];
    __shared__ int rejected;
    // The state is monotonically increasing. Once rejected, later chunks can
    // skip their index reads without changing the final routing decision.
    if (threadIdx.x == 0) rejected = atomicAdd(prefix_lengths + bh, 0) > prefix_capacity;
    __syncthreads();
    if (rejected) return;
    const int qs = cu_q[batch];
    const int qlen = cu_q[batch + 1] - qs;
    const int klen = kv_lens[batch];
    const int qr = int(blockIdx.x) * kQueriesPerBlock + int(threadIdx.x) / 32;
    const int col_lane = int(threadIdx.x) % 32;
    int length = klen <= 0 ? prefix_capacity + 1 : 0;
    if (qr < qlen && klen > 0) {
        const int64_t visible = int64_t(klen) - qlen + qr + 1;
        const int64_t row_topk = min(int64_t(topk), max(int64_t{0}, visible));
        const int64_t mask = reference_slot_block_size - 1;
        const int count = int(min(int64_t(topk), (row_topk + mask) & ~mask));
        const int32_t* row = indices + int64_t(qs + qr) * index_query_stride
            + int64_t(head) * index_head_stride;
        for (int slot = col_lane; slot < count; slot += 32) {
            const int token = row[slot];
            if (token >= 0 && token < klen)
                length = max(length, min(token + 1, prefix_capacity + 1));
            // Rejection is final: no need to read the remaining slots of a
            // lane that already proved this head exceeds the bounded prefix.
            // Every thread still reaches the reductions and CTA barrier below.
            if (length > prefix_capacity) break;
        }
    }
    // All 256 threads reach the barriers, including tail queries. A wave owns
    // two 32-thread query groups, but only the capped maximum is shared.
    for (int offset = 32; offset > 0; offset /= 2)
        length = max(length, __shfl_xor(length, offset, kWaveSize));
    if ((int(threadIdx.x) % kWaveSize) == 0) scratch[int(threadIdx.x) / kWaveSize] = length;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int block_length = max(max(scratch[0], scratch[1]), max(scratch[2], scratch[3]));
        atomicMax(prefix_lengths + bh, block_length);
    }
}

// Store V as [batch, prefix_capacity, Hkv, 256], while reading each request
// from its true packed cu_k base. Per-head lengths do not change KV semantics.
template <typename Element>
__global__ void sparse_gqa_decode_v_e5_prefix_kernel(
    const uint8_t* input, Element* output, const int32_t* cu_k,
    const int32_t* prefix_lengths, int hkv, int prefix_capacity) {
    const int batch = int(blockIdx.y) / hkv;
    const int head = int(blockIdx.y) % hkv;
    const int length = prefix_lengths[int64_t(batch) * hkv + head];
    if (length > prefix_capacity) return;
    // A dead invalid-token load in the original kernel uses token zero before
    // zeroing its fragment. Initialize that token even for an all-invalid row.
    const int live_tokens = max(1, length);
    const int64_t words = int64_t(live_tokens) * (kHeadDim / 4);
    for (int64_t word = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         word < words; word += int64_t(gridDim.x) * blockDim.x) {
        const int token = int(word / (kHeadDim / 4));
        const int dim_word = int(word % (kHeadDim / 4));
        const int64_t src = ((int64_t(cu_k[batch]) + token) * hkv + head) * kHeadDim
            + dim_word * 4;
        const int64_t dst = ((int64_t(batch) * prefix_capacity + token) * hkv + head) * kHeadDim
            + dim_word * 4;
        const Half4 value = decode_fp8x4<Element, Fp8Format::E5M2>(
            *reinterpret_cast<const uint32_t*>(input + src));
        *reinterpret_cast<Half4*>(output + dst) = value;
    }
}

int64_t predecode_workspace_limit() {
    // This bounds temporary decoded K/V (E4) or V (E5), not allocator bookkeeping or output.
    const char* value = std::getenv("FA_SPARSE_GQA_FP8_PREDECODE_MAX_MB");
    if (value == nullptr) return 256LL * 1024 * 1024;
    char* end = nullptr;
    const long mb = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && mb >= 0 && mb <= 4096,
                "FA_SPARSE_GQA_FP8_PREDECODE_MAX_MB must be an integer in [0, 4096]");
    return static_cast<int64_t>(mb) * 1024 * 1024;
}

// ---- Host contract checks ----


} // namespace

// ---- Public FP8 entry and dispatch: contract, safety, profitability, fallback ----
at::Tensor sparse_gqa_attention_fp8(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& indices, const at::Tensor& cu_q,
    const at::Tensor& cu_k, const at::Tensor& kv_lens,
    double scale, double k_scale, double v_scale, int64_t max_seqlen_q) {
    check_contiguous_accelerator_tensor(q, "q"); check_contiguous_accelerator_tensor(k, "k"); check_contiguous_accelerator_tensor(v, "v");
    check_contiguous_accelerator_tensor(indices, "indices"); check_contiguous_accelerator_tensor(cu_q, "cu_q");
    check_contiguous_accelerator_tensor(cu_k, "cu_k"); check_contiguous_accelerator_tensor(kv_lens, "kv_lens");
    const auto dtype = q.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "Q must be FP16 or BF16");
    TORCH_CHECK(k.scalar_type() == v.scalar_type(), "FP8 K and V must have the same dtype");
    Fp8Format format;
    switch (k.scalar_type()) {
        case at::ScalarType::Float8_e4m3fn: format = Fp8Format::E4M3FN; break;
        case at::ScalarType::Float8_e5m2: format = Fp8Format::E5M2; break;
        case at::ScalarType::Float8_e4m3fnuz: format = Fp8Format::E4M3FNUZ; break;
        case at::ScalarType::Float8_e5m2fnuz: format = Fp8Format::E5M2FNUZ; break;
        default: TORCH_CHECK(false, "K/V must use a supported FP8 dtype");
    }
    TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim, "Q must have shape [total_q, Hq, 256]");
    TORCH_CHECK(k.dim() == 3 && k.size(2) == kHeadDim && v.sizes() == k.sizes(),
                "K/V must have shape [total_k, Hkv, 256]");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0, "Hq must be divisible by Hkv");
    const int64_t group = q.size(1) / k.size(1);
    TORCH_CHECK(group >= 1 && group <= 16, "GQA group_size must be in [1, 16]");
    TORCH_CHECK((indices.dim() == 2 || indices.dim() == 3) && indices.size(0) == q.size(0),
                "indices must have shape [total_q, topk] or [total_q, Hkv, topk]");
    TORCH_CHECK(indices.dim() == 2 || indices.size(1) == k.size(1), "indices Hkv dimension does not match K/V");
    const int64_t topk = indices.size(-1);
    const int64_t batch = kv_lens.numel();
    TORCH_CHECK(topk > 0 && topk <= INT32_MAX, "topk must be positive and fit in int32");
    TORCH_CHECK(indices.scalar_type() == at::kInt && cu_q.scalar_type() == at::kInt
                    && cu_k.scalar_type() == at::kInt && kv_lens.scalar_type() == at::kInt,
                "indices and sequence metadata must be int32");
    TORCH_CHECK(kv_lens.dim() == 1 && cu_q.dim() == 1 && cu_k.dim() == 1
                    && cu_q.numel() == batch + 1 && cu_k.numel() == batch + 1,
                "sequence metadata has incompatible dimensions");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == indices.device()
                    && q.device() == cu_q.device() && q.device() == cu_k.device()
                    && q.device() == kv_lens.device(), "all inputs must be on the same device");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_q <= INT32_MAX && q.size(0) <= INT32_MAX
                    && k.size(0) <= INT32_MAX && q.size(1) <= INT32_MAX && batch * k.size(1) <= INT32_MAX,
                "token/head counts and grid dimensions must fit in int32");
    TORCH_CHECK(std::isfinite(static_cast<float>(scale))
                    && std::isfinite(static_cast<float>(k_scale)) && std::isfinite(static_cast<float>(v_scale)),
                "attention/K/V scales must be finite float32 values");
    auto out = torch::empty_like(q);
    if (q.numel() == 0 || batch == 0) return out;
    // HCU uses op_opt3's non-H20 configuration table. This parameter preserves
    // its slot-window rounding; the compute tile itself remains 16 tokens.
    const int reference_slot_block_size = q.size(0) <= 32 ? 32 : (q.size(0) <= 128 ? 64 : (q.size(0) <= 512 ? 32 : 16));
    at::hip::HIPGuardMasqueradingAsCUDA device_guard{static_cast<char>(q.get_device())};
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipDeviceProp_t properties{};
    C10_HIP_CHECK(hipGetDeviceProperties(&properties, q.get_device()));
    const bool gfx936 = std::strncmp(properties.gcnArchName, "gfx936", 6) == 0;
    const int64_t decode_programs = q.size(0) * k.size(1);
    const int64_t decode_tiles = (topk + 15) / 16;
    const bool decode_tuned = gfx936 && format == Fp8Format::E5M2 &&
        dtype == at::kBFloat16 && group == 12 && max_seqlen_q == 1 &&
        q.size(0) <= 32 && decode_programs <= std::max(1, properties.multiProcessorCount) &&
        decode_tiles >= 128 && decode_tiles <= 256;
    int splits = choose_splits(q.size(0), k.size(1), group, topk, q.get_device());
    int decode_waves = 4;
    if (decode_tuned) {
        if (decode_programs <= 2) {
            decode_waves = 4;
            splits = 32;
        } else if (decode_programs <= std::max(2, properties.multiProcessorCount / 5)) {
            // Preserve the measured small-grid W4 schedule instead of forcing
            // W2 onto query grids where it regresses the tuned parent.
            decode_waves = 4;
            splits = 1;
            const int64_t cap = std::min<int64_t>(32, std::max<int64_t>(1, decode_tiles / 4));
            while (splits * 2 <= cap &&
                    decode_programs <= int64_t(properties.multiProcessorCount) / (splits * 2))
                splits *= 2;
        } else {
            decode_waves = 2;
            splits = 1;
            const int64_t cap = std::min<int64_t>(32, std::max<int64_t>(1, decode_tiles / 2));
            const int64_t target = 4LL * std::max(1, properties.multiProcessorCount);
            while (splits * 2 <= cap && decode_programs <= target / (splits * 2))
                splits *= 2;
        }
    }
    at::Tensor partial, lse;
    if (splits > 1) {
        partial = at::empty({splits, q.size(0), q.size(1), kHeadDim}, q.options().dtype(at::kFloat));
        lse = at::empty({splits, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    }
    const dim3 grid(static_cast<unsigned int>(max_seqlen_q), static_cast<unsigned int>(batch * k.size(1)), splits);

    // Decode-only narrow gate; no prefill, other dtype/format, or target
    // reaches the new kernel.  Public signatures and contiguous KV stay intact.
    if (decode_tuned) {
        auto launch_decode = [&](auto direct_tag, auto wave_tag) {
            constexpr int Waves = decltype(wave_tag)::value;
            constexpr int SharedBytes = ((Waves - 1) * 12 * 260 + Waves * 12 * 2) * sizeof(float);
            sparse_gqa_attention_fp8_kernel<hytlass::bfloat16_t, Fp8Format::E5M2,
                false, decltype(direct_tag)::value, true, true,
                false, 1, Waves, 0><<<grid, kWaveSize * Waves, SharedBytes, stream>>>(
                    reinterpret_cast<const hytlass::bfloat16_t*>(q.data_ptr()),
                    reinterpret_cast<const uint8_t*>(k.data_ptr()),
                    reinterpret_cast<const uint8_t*>(v.data_ptr()),
                    reinterpret_cast<hytlass::bfloat16_t*>(out.data_ptr()),
                    indices.data_ptr<int32_t>(), cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                    static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(topk),
                    static_cast<float>(k_scale), static_cast<float>(v_scale),
                    indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0,
                    reference_slot_block_size, partial.defined() ? partial.data_ptr<float>() : nullptr,
                    lse.defined() ? lse.data_ptr<float>() : nullptr, splits,
                    static_cast<int>(q.size(0)));
            C10_HIP_CHECK(hipGetLastError());
        };
        // The selected gate has programs<=CU and >=128 tiles. Both
        // non-tiny rules select >=4 external splits; tiny queries use32.
        // No cooperative unsplit kernel instance or branch is necessary.
        auto launch_waves = [&](auto wave_tag) {
            if (static_cast<float>(v_scale) == 1.0f)
                launch_decode(std::true_type{}, wave_tag);
            else
                launch_decode(std::false_type{}, wave_tag);
        };
        if (decode_waves == 2) launch_waves(std::integral_constant<int, 2>{});
        else launch_waves(std::integral_constant<int, 4>{});
        launch_gqa_decode_merge<hytlass::bfloat16_t>(partial, lse, out, splits, stream);
        return out;
    }
    bool predecode = false;
    at::Tensor decoded_k, decoded_v;
    if (gfx936 && format == Fp8Format::E4M3FN && group == 12 && q.size(0) >= 256 && topk >= 128
            && k.numel() > 0 && q.size(0) * std::min<int64_t>(topk, k.size(0)) >= k.size(0) * 16) {
        // Decode without applying scales. Preserve the reference's post-QK and
        // post-PV scale locations, including for non-unit or negative scales.
        const int64_t limit = predecode_workspace_limit();
        if (k.numel() <= limit / 4) {
            try {
                decoded_k = torch::empty(k.sizes(), q.options());
                decoded_v = torch::empty(v.sizes(), q.options());
                predecode = true;
            } catch (const c10::OutOfMemoryError&) {
                // Allocation occurs before any kernel launch. Keep the original
                // no-workspace route available when the allocator cannot fit it.
                decoded_k = at::Tensor();
                decoded_v = at::Tensor();
            }
        }
    }
    // Reuse one widened V cache across queries; K remains byte-packed.
    // Large-M/non-split gating avoids extra preparation for small workloads.
    bool v_only_predecode = false;
    at::Tensor v_only_decoded;
    if (gfx936 && format == Fp8Format::E5M2 && dtype == at::kBFloat16 &&
        group == 12 && q.size(0) >= 4096 && topk >= 128 && splits == 1 &&
        v.numel() > 0 && q.size(0) * std::min<int64_t>(topk, k.size(0)) >= k.size(0) * 16) {
        // A workspace fitting in memory is not sufficient: large/random caches
        // regress when widened V traffic outweighs saved conversion work. Keep
        // this route within the validated 16-MiB decoded-V reuse range. Larger
        // caches retain the original raw path, even with a larger user budget.
        const int64_t limit = std::min<int64_t>(predecode_workspace_limit(), 16LL * 1024 * 1024);
        // Division first avoids overflow and budgets only the new owning V.
        if (v.numel() <= limit / sizeof(at::BFloat16)) {
            try {
                v_only_decoded = torch::empty(v.sizes(), q.options());
                v_only_predecode = true;
            } catch (const c10::OutOfMemoryError&) {
                // No producer has launched, so the original raw path is safe.
                v_only_decoded = at::Tensor();
            }
        }
    }
    bool prefix_predecode = false;
    int prefix_capacity = 0;
    at::Tensor prefix_v, prefix_lengths;
    if (!v_only_predecode && gfx936 && format == Fp8Format::E5M2 && dtype == at::kBFloat16 &&
        group == 12 && q.size(0) >= 4096 && topk >= 128 && splits == 1 &&
        v.numel() > 0 && q.size(0) * std::min<int64_t>(topk, k.size(0)) >= k.size(0) * 16) {
        const int64_t limit = std::min<int64_t>(predecode_workspace_limit(), 16LL * 1024 * 1024);
        const int64_t metadata_bytes = batch * k.size(1) * sizeof(int32_t);
        const int64_t token_bytes = batch * k.size(1) * kHeadDim * sizeof(at::BFloat16);
        if (limit > metadata_bytes && token_bytes > 0) {
            prefix_capacity = static_cast<int>(std::min<int64_t>(16384, (limit - metadata_bytes) / token_bytes));
            if (prefix_capacity >= 128) {
                try {
                    prefix_v = torch::empty({batch, prefix_capacity, k.size(1), kHeadDim}, q.options());
                    prefix_lengths = torch::empty({batch, k.size(1)}, cu_k.options());
                    prefix_predecode = true;
                } catch (const c10::OutOfMemoryError&) {
                    prefix_v = at::Tensor();
                    prefix_lengths = at::Tensor();
                    prefix_capacity = 0;
                }
            }
        }
    }
    FP16_SWITCH(dtype == at::kHalf, [&] {
        auto launch_format = [&](auto format_tag) {
            constexpr Fp8Format fp8_format = decltype(format_tag)::value;
            auto launch_group = [&](auto group_tag) {
                constexpr int static_group = decltype(group_tag)::value;
                auto launch = [&](auto decoded_tag, auto direct_tag) {
                    constexpr bool cached = decltype(decoded_tag)::value;
                    constexpr bool direct = decltype(direct_tag)::value;
                    using CacheElement = KvElement<elem_type, cached>;
                    const void* key_ptr = cached ? decoded_k.data_ptr() : k.data_ptr();
                    const void* value_ptr = cached ? decoded_v.data_ptr() : v.data_ptr();
                    auto run = [&](auto split_tag) {
                        sparse_gqa_attention_fp8_kernel<elem_type, fp8_format, cached, direct, (static_group == 12), decltype(split_tag)::value><<<grid, kWaveSize, 0, stream>>>(
                        reinterpret_cast<const elem_type*>(q.data_ptr()),
                        reinterpret_cast<const CacheElement*>(key_ptr), reinterpret_cast<const CacheElement*>(value_ptr),
                        reinterpret_cast<elem_type*>(out.data_ptr()), indices.data_ptr<int32_t>(),
                        cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                        static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(topk),
                        static_cast<float>(k_scale), static_cast<float>(v_scale),
                        indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0, reference_slot_block_size,
                        partial.defined() ? partial.data_ptr<float>() : nullptr,
                        lse.defined() ? lse.data_ptr<float>() : nullptr, splits, static_cast<int>(q.size(0)));
                    };
                    if constexpr (static_group == 12) {
                        if (splits > 1) run(std::true_type{});
                        else run(std::false_type{});
                    } else {
                        run(std::false_type{});
                    }
                    C10_HIP_CHECK(hipGetLastError());
                };
                if constexpr (static_group == 12 && fp8_format == Fp8Format::E5M2 &&
                              std::is_same<elem_type, hytlass::bfloat16_t>::value) {
                    if (prefix_predecode) {
                        C10_HIP_CHECK(hipMemsetAsync(prefix_lengths.data_ptr(), 0,
                            batch * k.size(1) * sizeof(int32_t), stream));
                        const dim3 summary_grid((int64_t(grid.x) + 7) / 8, grid.y, 1);
                        sparse_gqa_v_prefix_summary_kernel<<<summary_grid, 256, 0, stream>>>(
                            indices.data_ptr<int32_t>(), cu_q.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                            prefix_lengths.data_ptr<int32_t>(), static_cast<int>(k.size(1)), static_cast<int>(topk),
                            reference_slot_block_size, indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0,
                            prefix_capacity);
                        C10_HIP_CHECK(hipGetLastError());
                        const dim3 prefix_grid(256, grid.y, 1);
                        sparse_gqa_decode_v_e5_prefix_kernel<elem_type><<<prefix_grid, 256, 0, stream>>>(
                            reinterpret_cast<const uint8_t*>(v.data_ptr()),
                            reinterpret_cast<elem_type*>(prefix_v.data_ptr()), cu_k.data_ptr<int32_t>(),
                            prefix_lengths.data_ptr<int32_t>(), static_cast<int>(k.size(1)), prefix_capacity);
                        C10_HIP_CHECK(hipGetLastError());
                        const dim3 prefix_query_grid((int64_t(grid.x) + 3) / 4, grid.y, 1);
                        auto run_prefix = [&](auto direct_tag) {
                            // Rejected heads retain raw FP8, one wave per query.
                            sparse_gqa_attention_fp8_kernel<elem_type, fp8_format, false,
                                decltype(direct_tag)::value, true, false, false, 1, 1, 1>
                                <<<grid, kWaveSize, 0, stream>>>(
                                    reinterpret_cast<const elem_type*>(q.data_ptr()),
                                    reinterpret_cast<const uint8_t*>(k.data_ptr()),
                                    reinterpret_cast<const uint8_t*>(v.data_ptr()),
                                    reinterpret_cast<elem_type*>(out.data_ptr()), indices.data_ptr<int32_t>(),
                                    cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                                    static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(topk),
                                    static_cast<float>(k_scale), static_cast<float>(v_scale),
                                    indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0, reference_slot_block_size,
                                    nullptr, nullptr, 1, static_cast<int>(q.size(0)),
                                    prefix_lengths.data_ptr<int32_t>(), prefix_capacity);
                            C10_HIP_CHECK(hipGetLastError());
                            // Eligible heads use independently indexed bounded V.
                            sparse_gqa_attention_fp8_kernel<elem_type, fp8_format, false,
                                decltype(direct_tag)::value, true, false, true, 4, 1, 2>
                                <<<prefix_query_grid, kWaveSize * 4, 0, stream>>>(
                                    reinterpret_cast<const elem_type*>(q.data_ptr()),
                                    reinterpret_cast<const uint8_t*>(k.data_ptr()),
                                    reinterpret_cast<const elem_type*>(prefix_v.data_ptr()),
                                    reinterpret_cast<elem_type*>(out.data_ptr()), indices.data_ptr<int32_t>(),
                                    cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                                    static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(topk),
                                    static_cast<float>(k_scale), static_cast<float>(v_scale),
                                    indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0, reference_slot_block_size,
                                    nullptr, nullptr, 1, static_cast<int>(q.size(0)),
                                    prefix_lengths.data_ptr<int32_t>(), prefix_capacity);
                            C10_HIP_CHECK(hipGetLastError());
                        };
                        if (static_cast<float>(v_scale) == 1.0f) run_prefix(std::true_type{});
                        else run_prefix(std::false_type{});
                        return;
                    }
                    if (v_only_predecode) {
                        const int64_t words = v.numel() / 4;
                        sparse_gqa_decode_v_e5_kernel<elem_type>
                            <<<static_cast<unsigned int>((words + 255) / 256), 256, 0, stream>>>(
                                reinterpret_cast<const uint32_t*>(v.data_ptr()),
                                reinterpret_cast<elem_type*>(v_only_decoded.data_ptr()), words);
                        C10_HIP_CHECK(hipGetLastError());
                        // Four independent query waves, not four-wave collaboration.
                        const dim3 v_only_grid((int64_t(grid.x) + 3) / 4, grid.y, 1);
                        auto run_v_only = [&](auto direct_tag) {
                            sparse_gqa_attention_fp8_kernel<elem_type, fp8_format, false,
                                decltype(direct_tag)::value, true, false, true, 4>
                                <<<v_only_grid, kWaveSize * 4, 0, stream>>>(
                                    reinterpret_cast<const elem_type*>(q.data_ptr()),
                                    reinterpret_cast<const uint8_t*>(k.data_ptr()),
                                    reinterpret_cast<const elem_type*>(v_only_decoded.data_ptr()),
                                    reinterpret_cast<elem_type*>(out.data_ptr()), indices.data_ptr<int32_t>(),
                                    cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
                                    static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(topk),
                                    static_cast<float>(k_scale), static_cast<float>(v_scale),
                                    indices.stride(0), indices.dim() == 3 ? indices.stride(1) : 0, reference_slot_block_size,
                                    nullptr, nullptr, 1, static_cast<int>(q.size(0)));
                            C10_HIP_CHECK(hipGetLastError());
                        };
                        if (static_cast<float>(v_scale) == 1.0f) run_v_only(std::true_type{});
                        else run_v_only(std::false_type{});
                        return;
                    }
                }
                if constexpr (static_group == 12 && fp8_format == Fp8Format::E4M3FN) {
                    if (predecode) {
                        const int64_t words = k.numel() / 4;
                        decode_kv_cache<elem_type, fp8_format><<<static_cast<unsigned int>((words + 255) / 256), 256, 0, stream>>>(
                            reinterpret_cast<const uint32_t*>(k.data_ptr()), reinterpret_cast<const uint32_t*>(v.data_ptr()),
                            reinterpret_cast<elem_type*>(decoded_k.data_ptr()), reinterpret_cast<elem_type*>(decoded_v.data_ptr()), words);
                        if (static_cast<float>(v_scale) == 1.0f) launch(std::true_type{}, std::true_type{});
                        else launch(std::true_type{}, std::false_type{});
                        return;
                    }
                }
                if constexpr (static_group == 12 && (fp8_format == Fp8Format::E4M3FN || fp8_format == Fp8Format::E5M2)) {
                    if ((gfx936 || std::strncmp(properties.gcnArchName, "gfx938", 6) == 0)
                            && static_cast<float>(v_scale) == 1.0f) {
                        launch(std::false_type{}, std::true_type{});
                        return;
                    }
                }
                launch(std::false_type{}, std::false_type{});
            };
            if (group == 12) launch_group(std::integral_constant<int, 12>{});
            else launch_group(std::integral_constant<int, 0>{});
        };
        switch (format) {
            case Fp8Format::E4M3FN: launch_format(std::integral_constant<Fp8Format, Fp8Format::E4M3FN>{}); break;
            case Fp8Format::E5M2: launch_format(std::integral_constant<Fp8Format, Fp8Format::E5M2>{}); break;
            case Fp8Format::E4M3FNUZ: launch_format(std::integral_constant<Fp8Format, Fp8Format::E4M3FNUZ>{}); break;
            case Fp8Format::E5M2FNUZ: launch_format(std::integral_constant<Fp8Format, Fp8Format::E5M2FNUZ>{}); break;
        }
        if (splits > 1) launch_split_merge<elem_type>(partial, lse, out, splits, stream);
    });
    return out;
}
