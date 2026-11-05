// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Region family main implementation: BF16 entries and shared internal launchers/merge.
// The FP8 CU calls these BF16 launchers after predecode and shares the final-state merge.
#include "sparse_region_attention.h"

namespace {
using namespace sparse_region_detail;

// Merge FP32 internal states with log2 LSE into final BF16 / natural-log LSE.
// Small internal split counts: preload independent states for all output
// columns before the reduction, rather than serialize a load/use loop for
// every split and every 64-column pass. All lanes share the same log weights.
template<int Parts>
__global__ __launch_bounds__(64) void region_internal_merge_small_kernel(
    const float *partial, const float *partial_lse, at::BFloat16 *out, float *lse,
    int64_t state_rows, int dim) {
    const int lane=threadIdx.x;
    const int64_t row=blockIdx.x;
    float logs[Parts], values[Parts][4];
#pragma unroll
    for (int s=0;s<Parts;++s) {
        logs[s]=partial_lse[int64_t(s)*state_rows+row];
#pragma unroll
        for (int c=0;c<4;++c) {
            const int d=lane+c*64;
            values[s][c]=d<dim ? partial[(int64_t(s)*state_rows+row)*dim+d] : 0;
        }
    }
    float peak=-INFINITY;
#pragma unroll
    for (int s=0;s<Parts;++s) peak=fmaxf(peak,logs[s]);
    float total=0;
#pragma unroll
    for (int s=0;s<Parts;++s) {
        logs[s]=peak>-INFINITY ? region_exp2(logs[s]-peak) : 0;
        total+=logs[s];
    }
    const float inverse=total>0 ? 1.0f/total : 0;
#pragma unroll
    for (int c=0;c<4;++c) {
        float acc=0;
#pragma unroll
        for (int s=0;s<Parts;++s) acc+=values[s][c]*logs[s];
        const int d=lane+c*64;
        if (d<dim) out[row*dim+d]=at::BFloat16(acc*inverse);
    }
    if (lane==0) lse[row]=total>0 ? (peak+log2f(total))*kLn2 : -INFINITY;
}

// Parallelize both output dimensions and split reduction for small state grids.
// Reuse each log weight across Columns/16 output values. Keep four independent
// split accumulators and the original FP32 reduction order for every column.
template<int Parts, int Columns = 16>
__global__ __launch_bounds__(64) void region_internal_merge_tiled_kernel(
    const float *partial, const float *partial_lse, at::BFloat16 *out, float *lse,
    int64_t state_rows, int dim) {
    const int lane=threadIdx.x,r=lane%16,g=lane/16;
    const int64_t row=blockIdx.x;
    constexpr int per_lane=Parts/4, columns_per_lane=Columns/16;
    float logs[per_lane],values[per_lane][columns_per_lane];
    // Bulk issue independent loads before consuming them. A serial load/exp/
    // accumulate loop is latency-bound for a small number of output heads.
#pragma unroll
    for (int i=0;i<per_lane;++i) {
        const int s=g+i*4;
        logs[i]=partial_lse[int64_t(s)*state_rows+row];
#pragma unroll
        for (int c=0;c<columns_per_lane;++c) {
            const int d=blockIdx.y*Columns+r+c*16;
            values[i][c]=d<dim ? partial[(int64_t(s)*state_rows+row)*dim+d] : 0;
        }
    }
    float peak=-INFINITY;
#pragma unroll
    for (int i=0;i<per_lane;++i) peak=fmaxf(peak,logs[i]);
    peak=fmaxf(peak,__shfl_xor(peak,16,64));
    peak=fmaxf(peak,__shfl_xor(peak,32,64));
    float total=0,acc[columns_per_lane]={};
#pragma unroll
    for (int i=0;i<per_lane;++i) {
        const float weight=peak>-INFINITY ? region_exp2(logs[i]-peak) : 0;
        total+=weight;
#pragma unroll
        for (int c=0;c<columns_per_lane;++c) acc[c]+=values[i][c]*weight;
    }
    total+=__shfl_xor(total,16,64);total+=__shfl_xor(total,32,64);
#pragma unroll
    for (int c=0;c<columns_per_lane;++c) {
        acc[c]+=__shfl_xor(acc[c],16,64);acc[c]+=__shfl_xor(acc[c],32,64);
        const int d=blockIdx.y*Columns+r+c*16;
        if (g==0 && d<dim) out[row*dim+d]=at::BFloat16(total>0 ? acc[c]/total : 0);
    }
    if (lane==0 && blockIdx.y==0) lse[row]=total>0 ? (peak+log2f(total))*kLn2 : -INFINITY;
}

template<int Parts>
void launch_tiled_merge(const at::Tensor &workspace, const at::Tensor &workspace_lse,
    at::Tensor &out, at::Tensor &lse, int64_t state_rows, int dim, hipStream_t stream) {
    // Small grids need enough CTAs; large Parts also limit register headroom.
    // This selects only the merge tile, never the attention split count.
    int columns=16;
    if constexpr (Parts==32) {
        if (state_rows>=512) columns=32;
        else if (state_rows>=128) columns=64;
    } else if constexpr (Parts==64 || Parts==128) {
        if (state_rows>=32 && state_rows<256) columns=32;
    }
#define TILED_MERGE(C) region_internal_merge_tiled_kernel<Parts,C><<< \
        dim3(sparse_attention_detail::checked_grid_extent(state_rows),(dim+C-1)/C),64,0,stream>>>( \
        workspace.data_ptr<float>(),workspace_lse.data_ptr<float>(),out.data_ptr<at::BFloat16>(), \
        lse.data_ptr<float>(),state_rows,dim)
    if constexpr (Parts==32) {
        if (columns==64) { TILED_MERGE(64); return; }
    }
    if constexpr (Parts!=16) {
        if (columns==32) { TILED_MERGE(32); return; }
    }
    TILED_MERGE(16);
#undef TILED_MERGE
}

// ---- BF16 address-safety guard and shared internal compute launch ----
// Offset32 needs u24 token/byte-stride products and 32-bit byte offsets
// (one 384-byte row past the last token start).
bool region_bf16_offset32_fits(const RegionArgs &a) {
    constexpr int64_t kU24 = int64_t{1} << 24, kU32 = int64_t{1} << 32;
    return a.kv_capacity_tokens < kU24 && a.ks0 >= 0 && a.ks0 * 2 < kU24 && a.vs0 >= 0 && a.vs0 * 2 < kU24 &&
           a.kv_capacity_tokens * a.ks0 * 2 + 384 <= kU32 && a.kv_capacity_tokens * a.vs0 * 2 + 384 <= kU32;
}

template<bool Decode>
void launch_bf16_opt(RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    // Out-of-range strides fall back to the same options without Offset32.
    if (!region_bf16_offset32_fits(a)) {
        launch_region_kernel<Decode, false, kBf16Opt & ~kFp8Offset32>(a, grid, stream, parts, dim);
        return;
    }
    launch_region_kernel<Decode, false, kBf16Opt>(a, grid, stream, parts, dim);
}

} // namespace

void sparse_region_detail::launch_decode_bf16(RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    launch_bf16_opt<true>(a, grid, stream, parts, dim);
}

// ---- Shared merge definition: both public precision paths use this function ----
// Shared internal-merge entry point for both attention paths.
void sparse_region_detail::merge_internal_states(
    const at::Tensor &workspace, const at::Tensor &workspace_lse,
    at::Tensor &out, at::Tensor &lse, int64_t state_rows, int dim, int parts, hipStream_t stream,
    int /*device_cus*/, bool /*decode*/) {
    using namespace sparse_attention_detail;
        if (parts > 1) {
            if (parts>=16) {
#define MERGE(P) launch_tiled_merge<P>(workspace,workspace_lse,out,lse,state_rows,dim,stream)
                switch(parts) {
                    case 16: { MERGE(16);break; }
                    case 32: { MERGE(32);break; }
                    case 64: { MERGE(64);break; }
                    case 128: { MERGE(128);break; }
                }
#undef MERGE
            } else {
#define SMALL_MERGE(P) region_internal_merge_small_kernel<P><<<checked_grid_extent(state_rows),64,0,stream>>>( \
                    workspace.data_ptr<float>(),workspace_lse.data_ptr<float>(),out.data_ptr<at::BFloat16>(), \
                    lse.data_ptr<float>(),state_rows,dim)
                switch(parts) {
                    case 2: { SMALL_MERGE(2);break; }
                    case 4: { SMALL_MERGE(4);break; }
                    case 8: { SMALL_MERGE(8);break; }
                }
#undef SMALL_MERGE
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
}

// ---- Public BF16 entries and shared prefill launcher ----
std::vector<at::Tensor> fa_sparse_attention_decode(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &packed, const at::Tensor &counts, const at::Tensor &seqlens,
    int64_t groups, int64_t region, c10::optional<double> scale, int64_t /*num_splits*/) {
    using namespace sparse_region_detail;
    // num_splits is retained for call compatibility; only final states are exposed.
    return run_region_attention<false>(q, k, v, packed, counts, seqlens, groups, region,
        scale, 1.0, 1.0, launch_decode_bf16, region_decode_parts);
}

void sparse_region_detail::launch_prefill_bf16(
    RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    launch_bf16_opt<false>(a, grid, stream, parts, dim);
}

std::vector<at::Tensor> fa_sparse_attention_prefill(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &packed, const at::Tensor &counts, int64_t groups,
    int64_t region, c10::optional<double> scale) {
    using namespace sparse_region_detail;
    return run_region_attention<false>(q, k, v, packed, counts, c10::nullopt, groups, region,
        scale, 1.0, 1.0, launch_prefill_bf16, region_default_parts);
}
