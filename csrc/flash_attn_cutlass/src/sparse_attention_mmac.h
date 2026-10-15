#pragma once

#include <hip/hip_runtime.h>
#include <cmath>

#include <c10/util/BFloat16.h>

#include <cstdint>

namespace sparse_attention_detail::mmac {

using Half4 = __attribute__((vector_size(8))) _Float16;
using Bf4 = __attribute__((vector_size(8))) short;
using Float4 = __attribute__((vector_size(16))) float;
using Words2 = __attribute__((vector_size(8))) uint32_t;

struct Vec8 {
    Half4 x;
    Half4 y;
};

template<typename Element>
__device__ __forceinline__ Vec8 load8(const Element *p) {
    return {*reinterpret_cast<const Half4 *>(p),
            *reinterpret_cast<const Half4 *>(p + 4)};
}

__device__ __forceinline__ Vec8 shuffle8(Vec8 value, int lane) {
    Vec8 result;
    auto *dst = reinterpret_cast<uint32_t *>(&result);
    const auto *src = reinterpret_cast<const uint32_t *>(&value);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        dst[i] = __shfl(src[i], lane, 64);
    return result;
}

__device__ __forceinline__ void dot(Half4 a, Half4 b, Float4 &acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    acc = __builtin_hcu_mmac_f32_16x16x16_bf16(
        *reinterpret_cast<const Bf4 *>(&a), *reinterpret_cast<const Bf4 *>(&b), acc);
#endif
}

__device__ __forceinline__ Half4 transpose_v(Half4 value, int g, int r) {
    Words2 input;
    __builtin_memcpy(&input, &value, sizeof(value));
    const int source = g * 16 + (r & 3) * 4 + (r >> 2);
    uint32_t lo = __shfl(input[0], source, 64);
    uint32_t hi = __shfl(input[1], source, 64);
    const unsigned shift = ((r ^ 1) & 1) * 16;
    const uint32_t send = ((lo >> shift) & 0xffffU) | ((hi >> shift) << 16);
    const uint32_t recv = __shfl_xor(send, 1, 64);
    const uint32_t keep = ~(0xffffU << shift);
    lo = (lo & keep) | ((recv & 0xffffU) << shift);
    hi = (hi & keep) | ((recv >> 16) << shift);
    const uint32_t pair = __shfl_xor((r & 2) ? lo : hi, 2, 64);
    const Words2 result = {(r & 2) ? pair : lo, (r & 2) ? hi : pair};
    Half4 output;
    __builtin_memcpy(&output, &result, sizeof(output));
    return output;
}

__device__ __forceinline__ Float4 transpose_p(Float4 p, int g) {
    float a = (g & 1) ? p[0] : p[1];
    float b = (g & 1) ? p[2] : p[3];
    a = __shfl_xor(a, 16, 64);
    b = __shfl_xor(b, 16, 64);
    p[0] = (g & 1) ? a : p[0];
    p[1] = (g & 1) ? p[1] : a;
    p[2] = (g & 1) ? b : p[2];
    p[3] = (g & 1) ? p[3] : b;
    a = (g & 2) ? p[0] : p[2];
    b = (g & 2) ? p[1] : p[3];
    a = __shfl_xor(a, 32, 64);
    b = __shfl_xor(b, 32, 64);
    p[0] = (g & 2) ? a : p[0];
    p[1] = (g & 2) ? b : p[1];
    p[2] = (g & 2) ? p[2] : a;
    p[3] = (g & 2) ? p[3] : b;
    return p;
}

__device__ __forceinline__ Half4 bf16_prob(Float4 values) {
    Half4 result;
    auto *output = reinterpret_cast<c10::BFloat16 *>(&result);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        output[i] = c10::BFloat16(values[i]);
    return result;
}

__device__ __forceinline__ float exp2(float value) {
    return exp2f(value);
}

// DPP variant for callers that already use this instruction path.
__device__ __forceinline__ Half4 transpose_v_dpp(Half4 value, int g, int r) {
    Words2 input;
    __builtin_memcpy(&input, &value, sizeof(value));
    const int source = g * 16 + (r & 3) * 4 + (r >> 2);
    uint32_t lo = __shfl(input[0], source, 64);
    uint32_t hi = __shfl(input[1], source, 64);
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    // V_PERM selectors 0..3 read the second operand, 4..7 the first.
    const uint32_t send = __builtin_amdgcn_perm(
        hi, lo, (r & 1) ? 0x05040100u : 0x07060302u);
    const uint32_t recv = __builtin_amdgcn_mov_dpp(send, 0xb1, 15, 15, false);
    lo = __builtin_amdgcn_perm(recv, lo, (r & 1) ? 0x03020504u : 0x05040100u);
    hi = __builtin_amdgcn_perm(recv, hi, (r & 1) ? 0x03020706u : 0x07060100u);
    const uint32_t pair = __builtin_amdgcn_mov_dpp(
        (r & 2) ? lo : hi, 0x4e, 15, 15, false);
#else
    const unsigned shift = ((r ^ 1) & 1) * 16;
    const uint32_t send = ((lo >> shift) & 0xffffU) | ((hi >> shift) << 16);
    const uint32_t recv = __shfl_xor(send, 1, 64);
    const uint32_t keep = ~(0xffffU << shift);
    lo = (lo & keep) | ((recv & 0xffffU) << shift);
    hi = (hi & keep) | ((recv >> 16) << shift);
    const uint32_t pair = __shfl_xor((r & 2) ? lo : hi, 2, 64);
#endif
    const Words2 result = {(r & 2) ? pair : lo, (r & 2) ? hi : pair};
    Half4 output;
    __builtin_memcpy(&output, &result, sizeof(output));
    return output;
}

// Transpose packed FP8 bytes without changing their encoding.
__device__ __forceinline__ uint32_t transpose_v_fp8(uint32_t bits, int g, int r) {
    uint32_t word = __shfl(bits, g * 16 + (r & 3) * 4 + (r >> 2), 64);
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    uint32_t other = __builtin_amdgcn_mov_dpp(word, 0xb1, 15, 15, false);
    word = __builtin_amdgcn_perm(other, word, (r & 1) ? 0x03070105u : 0x06020400u);
    other = __builtin_amdgcn_mov_dpp(word, 0x4e, 15, 15, false);
    return __builtin_amdgcn_perm(other, word, (r & 2) ? 0x03020706u : 0x05040100u);
#else
    uint32_t other = __shfl_xor(word, 1, 64);
    word = (r & 1) ? ((word & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                   : ((word & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
    other = __shfl_xor(word, 2, 64);
    return (r & 2) ? ((word & 0xffff0000u) | (other >> 16))
                   : ((word & 0xffffu) | (other << 16));
#endif
}

}  // namespace sparse_attention_detail::mmac
