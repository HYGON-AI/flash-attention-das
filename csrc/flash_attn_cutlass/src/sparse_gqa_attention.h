// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
// GQA family contract: geometry, device helpers, split policy and template merges.
// Both precision translation units include this header; kernels/entries stay in their CUs.
// Split policy is host-only; merge templates preserve the existing output conversion.
#include "sparse_attention.h"
#include "fa_numeric_conversion.h"
#include <type_traits>
#include <ATen/ATen.h>
#include <algorithm>
#include <cstring>

namespace sparse_gqa_detail {

// ---- Geometry and architecture-dependent device options ----
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
using sparse_attention_detail::mmac::assemble_inlane_value_operand;

__device__ __forceinline__ int value_output_column(int d, int g, int e, bool enabled) {
    return enabled ? 16 * (g + 4 * e) + d : d * 16 + g + 4 * e;
}

// ---- Numerical helpers shared by both main kernels ----
template <typename Element>
__device__ __forceinline__ Element canonical_nan() {
    // Write the encoding directly; no fast-math arithmetic may erase NaN.
    const uint16_t bits = std::is_same<Element, hytlass::half_t>::value ? 0x7e00u : 0x7fc0u;
    Element value;
    __builtin_memcpy(&value, &bits, sizeof(bits));
    return value;
}

template <typename Element>
__device__ __forceinline__ float as_float(Element value) {
    return flash::NumericConverter<float, Element>()(value);
}

template <typename Element>
__device__ __forceinline__ Element as_element(float value) {
    return flash::NumericConverter<
        Element, float, hytlass::FloatRoundStyle::round_half_ulp_truncate>()(value);
}


template <typename Element>
__device__ __forceinline__ void mmac(Half4 a, Half4 b, Float4& accumulator) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<Element, hytlass::half_t>::value) {
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

// ---- Host split planning: no device metadata reads ----
inline void check_contiguous_accelerator_tensor(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be on the accelerator");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Shape/device-only policy. Small query grids can expose token-tile parallelism;
// larger workloads retain the original no-workspace path.
inline int choose_splits(int64_t rows, int64_t hkv, int64_t group, int64_t topk, int device) {
    if (group != 12 || rows > 256 || topk < 64) return 1;
    hipDeviceProp_t props{};
    C10_HIP_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strncmp(props.gcnArchName, "gfx936", 6) != 0 &&
        std::strncmp(props.gcnArchName, "gfx938", 6) != 0) return 1;
    // gfx936's 129..256-query grid benefits from additional token parallelism.
    // Preserve the measured gfx938 policy for this range.
    if (rows > 128 && std::strncmp(props.gcnArchName, "gfx936", 6) != 0) return 1;
    const int64_t programs = rows * hkv;
    const int64_t tiles = (topk + 15) / 16;
    const int cus = std::max(1, props.multiProcessorCount);
    const bool populated = programs >= cus;
    const int64_t cap = std::min<int64_t>(populated ? 8 : 64, std::max<int64_t>(1, tiles / 2));
    const int64_t target = (populated ? 32 : 8) * int64_t(cus);
    int splits = 1;
    while (splits * 2 <= cap && programs <= target / (splits * 2)) splits *= 2;
    return splits;
}

// ---- Shared merge kernels: normalized FP32 partials and log2 LSE ----
template <typename Element, int Parts>
__global__ __launch_bounds__(64) void merge_small(
    const float* partial, const float* lse, Element* out, int64_t rows) {
    const int lane = threadIdx.x;
    const int64_t row = blockIdx.x;
    float logs[Parts], values[Parts][4];
#pragma unroll
    for (int s = 0; s < Parts; ++s) {
        logs[s] = lse[int64_t(s) * rows + row];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            values[s][c] = partial[(int64_t(s) * rows + row) * 256 + lane + c * 64];
    }
    float peak = -INFINITY;
#pragma unroll
    for (int s = 0; s < Parts; ++s) peak = fmaxf(peak, logs[s]);
    float total = 0;
#pragma unroll
    for (int s = 0; s < Parts; ++s) {
        logs[s] = peak > -INFINITY ? exp2_fast(logs[s]-peak) : 0;
        total += logs[s];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float sum = 0;
#pragma unroll
        for (int s = 0; s < Parts; ++s) sum += values[s][c] * logs[s];
        out[row * 256 + lane + c * 64] = as_element<Element>(total > 0 ? sum / total : 0);
    }
}

template <typename Element, int Parts>
__global__ __launch_bounds__(64) void merge_tiled(
    const float* partial, const float* lse, Element* out, int64_t rows) {
    const int lane = threadIdx.x, r = lane % 16, g = lane / 16;
    const int64_t row = blockIdx.x;
    const int d = blockIdx.y * 16 + r;
    constexpr int PerLane = Parts / 4;
    float logs[PerLane], values[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const int s = g + i * 4;
        logs[i] = lse[int64_t(s) * rows + row];
        values[i] = partial[(int64_t(s) * rows + row) * 256 + d];
    }
    float peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) peak = fmaxf(peak, logs[i]);
    peak = fmaxf(peak, __shfl_xor(peak, 16, 64));
    peak = fmaxf(peak, __shfl_xor(peak, 32, 64));
    float total = 0, sum = 0;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const float weight = peak > -INFINITY ? exp2_fast(logs[i]-peak) : 0;
        total += weight;
        sum += values[i] * weight;
    }
    total += __shfl_xor(total, 16, 64);
    total += __shfl_xor(total, 32, 64);
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    if (g == 0) out[row * 256 + d] = as_element<Element>(total > 0 ? sum / total : 0);
}

// ---- Shared merge launcher: preserve split count and stream ----
template <typename Element>
inline void launch_split_merge(const at::Tensor& partial, const at::Tensor& lse,
                              at::Tensor& out, int splits, hipStream_t stream) {
    const int64_t rows = out.size(0) * out.size(1);
    auto* target = reinterpret_cast<Element*>(out.data_ptr());
#define SMALL(P) merge_small<Element,P><<<static_cast<unsigned int>(rows),64,0,stream>>>(partial.data_ptr<float>(),lse.data_ptr<float>(),target,rows)
#define TILED(P) merge_tiled<Element,P><<<dim3(static_cast<unsigned int>(rows),16),64,0,stream>>>(partial.data_ptr<float>(),lse.data_ptr<float>(),target,rows)
    switch (splits) {
      case 2: { SMALL(2); break; }
      case 4: { SMALL(4); break; }
      case 8: { SMALL(8); break; }
      case 16: { TILED(16); break; }
      case 32: { TILED(32); break; }
      case 64: { TILED(64); break; }
    }
#undef SMALL
#undef TILED
    C10_HIP_CHECK(hipGetLastError());
}

} // namespace sparse_gqa_detail
