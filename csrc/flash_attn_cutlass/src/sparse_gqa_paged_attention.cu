// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Paged family main implementation: BF16 entry plus merges shared with the FP8 CU.
// This file is not BF16-private. Merge definitions are linked exactly once.
#include "sparse_gqa_paged_attention.h"

namespace {
template <int Parts>
__global__ __launch_bounds__(64) void sparse_gqa_paged_attention_merge_small_kernel(
    const float *partial, const float *partial_lse, at::BFloat16 *out, int64_t state_rows, int dim,
    int64_t heads, int64_t os0, int64_t os1) {
    const int lane = threadIdx.x;
    const int64_t row = blockIdx.x;
    float logs[Parts], values[Parts][4];
#pragma unroll
    for (int s = 0; s < Parts; ++s) {
        logs[s] = partial_lse[int64_t(s) * state_rows + row];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int d = lane + c * 64;
            values[s][c] = d < dim ? partial[(int64_t(s) * state_rows + row) * dim + d] : 0;
        }
    }
    float peak = -INFINITY;
#pragma unroll
    for (int s = 0; s < Parts; ++s)
        peak = fmaxf(peak, logs[s]);
    float total = 0;
#pragma unroll
    for (int s = 0; s < Parts; ++s) {
        logs[s] = peak > -INFINITY ? sparse_attention_detail::mmac::exp2(logs[s] - peak) : 0;
        total += logs[s];
    }
    const float inverse = total > 0 ? 1.0f / total : 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float acc = 0;
#pragma unroll
        for (int s = 0; s < Parts; ++s)
            acc += values[s][c] * logs[s];
        const int d = lane + c * 64;
        if (d < dim)
            out[(row / heads) * os0 + (row % heads) * os1 + d] = at::BFloat16(acc * inverse);
    }
}

// Parallelize both output dimensions and split reduction for small state grids.
// Each wave handles 16 columns with four independent split accumulators, so
// large split counts do not serialize every column through all 128 states.
template <int Parts>
__global__ __launch_bounds__(64) void sparse_gqa_paged_attention_merge_tiled_kernel(
    const float *partial, const float *partial_lse, at::BFloat16 *out, int64_t state_rows, int dim,
    int64_t heads, int64_t os0, int64_t os1) {
    const int lane = threadIdx.x, r = lane % 16, g = lane / 16;
    const int64_t row = blockIdx.x;
    const int d = blockIdx.y * 16 + r;
    constexpr int per_lane = Parts / 4;
    float logs[per_lane], values[per_lane];
    // Bulk issue independent loads before consuming them. A serial load/exp/
    // accumulate loop is latency-bound for a small number of output heads.
#pragma unroll
    for (int i = 0; i < per_lane; ++i) {
        const int s = g + i * 4;
        logs[i] = partial_lse[int64_t(s) * state_rows + row];
        values[i] = d < dim ? partial[(int64_t(s) * state_rows + row) * dim + d] : 0;
    }
    float peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < per_lane; ++i)
        peak = fmaxf(peak, logs[i]);
    peak = fmaxf(peak, __shfl_xor(peak, 16, 64));
    peak = fmaxf(peak, __shfl_xor(peak, 32, 64));
    float total = 0, acc = 0;
#pragma unroll
    for (int i = 0; i < per_lane; ++i) {
        const float weight =
            peak > -INFINITY ? sparse_attention_detail::mmac::exp2(logs[i] - peak) : 0;
        total += weight;
        acc += values[i] * weight;
    }
    total += __shfl_xor(total, 16, 64);
    total += __shfl_xor(total, 32, 64);
    acc += __shfl_xor(acc, 16, 64);
    acc += __shfl_xor(acc, 32, 64);
    if (g == 0 && d < dim)
        out[(row / heads) * os0 + (row % heads) * os1 + d] =
            at::BFloat16(total > 0 ? acc / total : 0);
}

// Tiny gfx936 G12/E5 path has exactly32 partitions. One wave handles
// 32 output columns with two independent partition accumulators.
__global__ __launch_bounds__(64) void sparse_gqa_paged_attention_merge_tiny32_kernel(
    const float *partial, const float *partial_lse, at::BFloat16 *out,
    int64_t state_rows, int64_t heads, int64_t os0, int64_t os1) {
    const int lane = threadIdx.x, column_lane = lane % 32, group = lane / 32;
    const int64_t row = blockIdx.x;
    const int column = blockIdx.y * 32 + column_lane;
    float logs[16], values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int part = group + i * 2;
        logs[i] = partial_lse[int64_t(part) * state_rows + row];
        values[i] = partial[(int64_t(part) * state_rows + row) * 256 + column];
    }
    float peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) peak = fmaxf(peak, logs[i]);
    peak = fmaxf(peak, __shfl_xor(peak, 32, 64));
    float total = 0, value = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float weight = peak > -INFINITY ?
            sparse_attention_detail::mmac::exp2(logs[i] - peak) : 0;
        total += weight;
        value += values[i] * weight;
    }
    total += __shfl_xor(total, 32, 64);
    value += __shfl_xor(value, 32, 64);
    if (group == 0)
        out[(row / heads) * os0 + (row % heads) * os1 + column] =
            at::BFloat16(total > 0 ? value / total : 0);
}

} // namespace

// ---- Shared merge launchers used by both precision paths ----
void sparse_gqa_paged_detail::launch_merge(const float *partial, const float *lse,
                                           at::BFloat16 *out, int64_t rows, int64_t heads,
                                           int splits, int64_t os0, int64_t os1,
                                           hipStream_t stream) {
    using namespace sparse_attention_detail;
    if (splits < 16) {
#define SMALL(P)                                                                                   \
    sparse_gqa_paged_attention_merge_small_kernel<P><<<checked_grid_extent(rows *heads), 64, 0, stream>>>(   \
        partial, lse, out, rows * heads, 256, heads, os0, os1)
        switch (splits) {
        case 2: {
            SMALL(2);
            break;
        }
        case 4: {
            SMALL(4);
            break;
        }
        case 8: {
            SMALL(8);
            break;
        }
        }
#undef SMALL
    } else {
        const dim3 grid(checked_grid_extent(rows * heads), 16);
#define TILED(P)                                                                                   \
    sparse_gqa_paged_attention_merge_tiled_kernel<P>                                               \
        <<<grid, 64, 0, stream>>>(partial, lse, out, rows * heads, 256, heads, os0, os1)
        switch (splits) {
        case 16: {
            TILED(16);
            break;
        }
        case 32: {
            TILED(32);
            break;
        }
        case 64: {
            TILED(64);
            break;
        }
        }
#undef TILED
    }
}

void sparse_gqa_paged_detail::launch_merge_tiny32(
    const float *partial, const float *lse, at::BFloat16 *out,
    int64_t rows, int64_t heads, int64_t os0, int64_t os1, hipStream_t stream) {
    const dim3 grid(sparse_attention_detail::checked_grid_extent(rows * heads), 8);
    sparse_gqa_paged_attention_merge_tiny32_kernel<<<grid, 64, 0, stream>>>(
        partial, lse, out, rows * heads, heads, os0, os1);
}

// ---- Public BF16 entry ----
at::Tensor sparse_gqa_paged_attention(const at::Tensor &q, const at::Tensor &k_cache,
                                      const at::Tensor &v_cache, const at::Tensor &logical_indices,
                                      const at::Tensor &block_table, const at::Tensor &token_to_req,
                                      c10::optional<at::Tensor> out) {
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "paged sparse GQA requires BF16 K/V");
    return sparse_gqa_paged_detail::run_paged_attention<sparse_gqa_paged_detail::Bf16Cache>(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out);
}
