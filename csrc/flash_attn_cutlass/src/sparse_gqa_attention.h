#pragma once
// Shared token-sparse GQA geometry and precision-preserving device helpers.
#include "sparse_attention_mmac.h"
#include <cutlass/numeric_conversion.h>
#include <type_traits>

namespace sparse_gqa_detail {

constexpr int kHeadDim = 256;
constexpr int kWaveSize = 64;
constexpr int kTileTokens = 16;
constexpr int kQkSlices = kHeadDim / 32;
constexpr int kKPrefetchSlices = 4;
constexpr int kVFragments = kHeadDim / 16;
// A finite sentinel would collide with a valid QK result at -FLT_MAX.
constexpr float kNegativeInfinity = -INFINITY;
constexpr float kLog2E = 1.4426950408889634F;

// G12: retain the QK accumulator's token=g+4e order for PV and assemble
// the corresponding V operand within each lane. The 16-bit path benefits
// on gfx938 and gfx936; other targets retain the DPP/shuffle layout.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx938__) && !defined(__gfx936__)
constexpr bool kBfInLaneV = false;
#else
constexpr bool kBfInLaneV = true;
#endif

using sparse_attention_detail::mmac::Half4;
using sparse_attention_detail::mmac::Bf4;
using sparse_attention_detail::mmac::Float4;
using sparse_attention_detail::mmac::Words2;
using sparse_attention_detail::mmac::Vec8;
using sparse_attention_detail::mmac::load8;
using sparse_attention_detail::mmac::shuffle8;

// Each lane owns four token rows g+4j, and 16 contiguous dimensions 16r+b.
// MMAC operand b consists of those four tokens at column 16r+b. All selects
// are within a lane; no DS/DPP operation or FP arithmetic is required.
__device__ __forceinline__ Half4 inlane_value_operand(
    Half4 a, Half4 b, Half4 c, Half4 d, int component) {
    Words2 words[4];
    __builtin_memcpy(&words[0], &a, 8);
    __builtin_memcpy(&words[1], &b, 8);
    __builtin_memcpy(&words[2], &c, 8);
    __builtin_memcpy(&words[3], &d, 8);
    const int half = component / 2;
    Words2 packed;
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    const uint32_t selector = (component & 1) ? 0x07060302u : 0x05040100u;
    packed = {__builtin_amdgcn_perm(words[1][half], words[0][half], selector),
              __builtin_amdgcn_perm(words[3][half], words[2][half], selector)};
#else
    packed = (component & 1)
        ? Words2{(words[0][half] >> 16) | (words[1][half] & 0xffff0000u),
                 (words[2][half] >> 16) | (words[3][half] & 0xffff0000u)}
        : Words2{(words[0][half] & 0xffffu) | (words[1][half] << 16),
                 (words[2][half] & 0xffffu) | (words[3][half] << 16)};
#endif
    Half4 result;
    __builtin_memcpy(&result, &packed, 8);
    return result;
}

__device__ __forceinline__ int value_output_column(int d, int g, int e, bool enabled) {
    return enabled ? 16 * (g + 4 * e) + d : d * 16 + g + 4 * e;
}

template <typename Element>
__device__ __forceinline__ float as_float(Element value) {
    return cutlass::NumericConverter<float, Element>()(value);
}

template <typename Element>
__device__ __forceinline__ Element as_element(float value) {
    return cutlass::NumericConverter<
        Element, float, cutlass::FloatRoundStyle::round_half_ulp_truncate>()(value);
}


template <typename Element>
__device__ __forceinline__ void mmac(Half4 a, Half4 b, Float4& accumulator) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<Element, cutlass::half_t>::value) {
        accumulator = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, accumulator);
    } else {
        accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(
            *reinterpret_cast<const Bf4*>(&a),
            *reinterpret_cast<const Bf4*>(&b), accumulator);
    }
#endif
}

__device__ __forceinline__ float exp2_fast(float value) {
    return __builtin_amdgcn_exp2f(value);
}

__device__ __forceinline__ float maximum(float a, float b) {
    return a > b ? a : b;
}

// Convert the natural V load into the MMAC B fragment using bitwise moves.
// The initial permutation and the two butterfly stages stay inside a 16-lane
// row. DPP/V_PERM is enabled on gfx938 and gfx936; other build targets
// retain the equivalent shuffle network at compile time.
__device__ __forceinline__ Half4 transpose_v(Half4 value, int g, int r) {
    return sparse_attention_detail::mmac::transpose_v_dpp(value, g, r);
}

__device__ __forceinline__ float shuffle_xor16(float value) {
    return __shfl_xor(value, 16, kWaveSize);
}

// Repack the QK accumulator's token=g+4e layout for the PV MMAC operand.
__device__ __forceinline__ Float4 transpose_probability(Float4 p, int g) {
    return sparse_attention_detail::mmac::transpose_p(p, g);
}

} // namespace sparse_gqa_detail
