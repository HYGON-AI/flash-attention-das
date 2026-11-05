// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// GQA family main implementation: FP16/BF16 entry and kernel.
// Shared split/merge templates are declared and defined in the family header.
// Token-sparse GQA chunk-prefill, head_dim=256, FP16/BF16.
// One wave64 computes one query and one KV head (GQA group_size <= 16).
// Low-query G12 workloads can split token tiles across CTAs and merge FP32
// states, retaining the same packed-KV interface and request-relative indices.
// Q stays in registers; K uses a four-slice rolling window; V prefetches all
// sixteen output fragments. QK, online softmax and PV are fused without LDS.
// gfx938/gfx936 G12 keeps probabilities in QK order and forms V operands
// within each lane; other targets/groups retain the original shuffle layout.
// Indices are request-relative and consumed in their original order, including
// duplicates. Match the target's block-rounded slot window and return NaN for
// empty rows / empty leading reference blocks. Support shared or per-KV-head
// indices; negative/out-of-range token indices are ignored.

#include <torch/python.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hytlass/numeric_conversion.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "static_switch.h"

#include "sparse_gqa_attention.h"

namespace {
using namespace sparse_gqa_detail;

// ---- Main attention kernel ----
template <typename Element, int StaticGroupSize, bool SplitMode = false>
__global__ __launch_bounds__(kWaveSize) void sparse_gqa_attention_kernel(
    const Element* __restrict__ q, const Element* __restrict__ k,
    const Element* __restrict__ v, Element* __restrict__ out,
    const int32_t* __restrict__ indices, const int32_t* __restrict__ cu_q,
    const int32_t* __restrict__ cu_k, const int32_t* __restrict__ kv_lens,
    int hq, int hkv, int topk,
    int64_t index_query_stride, int64_t index_head_stride, int reference_slot_block_size,
    float* __restrict__ partial, float* __restrict__ lse, int splits, int total_q) {
    constexpr bool inlane_v = StaticGroupSize == 12 && kBfInLaneV;
    const int batch = static_cast<int>(blockIdx.y) / hkv;
    const int kv_head = static_cast<int>(blockIdx.y) % hkv;
    const int qr = static_cast<int>(blockIdx.x);
    const int qs = cu_q[batch], qlen = cu_q[batch + 1] - qs;
    if (qr >= qlen) return;
    const int query = qs + qr;
    const int ks = cu_k[batch], klen = kv_lens[batch];
    const int group = StaticGroupSize > 0 ? StaticGroupSize : hq / hkv;
    const int head_start = kv_head * group;
    const int active = min(16, group);
    const int64_t visible = static_cast<int64_t>(klen) - qlen + qr + 1;
    const int64_t row_topk = min(static_cast<int64_t>(topk), max(int64_t{0}, visible));
    const int reference_mask = reference_slot_block_size - 1;
    const int count = static_cast<int>(min(static_cast<int64_t>(topk),
        (row_topk + reference_mask) & ~static_cast<int64_t>(reference_mask)));
    const int lane = threadIdx.x, r = lane % 16, g = lane / 16;
    const int64_t qb = (static_cast<int64_t>(query) * hq + head_start) * kHeadDim;
    const int32_t* ir = indices + static_cast<int64_t>(query) * index_query_stride + kv_head * index_head_stride;
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    constexpr float softmax_scale = 0.0625f*kLog2E;
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
    if constexpr (SplitMode) {
        const int value = lane < min(count, reference_slot_block_size) ? ir[lane] : -1;
        leading_empty = __ballot(value >= 0 && value < klen) == 0;
    }
    const int tiles = static_cast<int>((int64_t(count) + 15) / 16);
    const int first = SplitMode ? static_cast<int>(int64_t(split) * tiles / splits * 16) : 0;
    const int last = SplitMode ? static_cast<int>(min(int64_t(count), int64_t(split + 1) * tiles / splits * 16)) : count;
    for (int start = first; start < last && !leading_empty; start += kTileTokens) {
        int token = -1;
        if (g == 0 && start + r < count) {
            const int value = ir[start + r];
            token = value >= 0 && value < klen ? value : -1;
        }
        token = __shfl(token, r, kWaveSize);
        const bool tile_has_valid = __ballot(token >= 0) != 0;
        if constexpr (SplitMode) {
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
        Half4 v_prefetch[kVFragments];
        Half4 lane_values[4][4];
        if constexpr (inlane_v) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int vt = __shfl(token, g + 4 * j, kWaveSize);
                const bool live = vt >= 0;
                const int64_t base = (static_cast<int64_t>(ks + (live ? vt : 0)) * hkv + kv_head) * kHeadDim;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const Vec8 row = load8(v + base + 16 * r + 8 * c);
                    lane_values[j][2 * c] = live ? row.x : zero.x;
                    lane_values[j][2 * c + 1] = live ? row.y : zero.x;
                }
            }
        } else {
        const int pref_token = __shfl(token, lane / 4, kWaveSize);
        const bool pref_valid = pref_token >= 0 && pref_token < klen;
        const int64_t pref_base =
            (static_cast<int64_t>(ks + (pref_valid ? pref_token : 0)) * hkv + kv_head) * kHeadDim;
#pragma unroll
        for (int d = 0; d < kVFragments; ++d){
            v_prefetch[d]=*reinterpret_cast<const Half4*>(v + pref_base + d * 16 + (lane % 4) * 4);
        }
        if(!pref_valid){
            for (int d = 0; d < kVFragments; ++d){
                v_prefetch[d]=zero.x;
            }
        }
        }
                // K4: naturally coalesced loads, then shuffle into the MMAC layout.
        Float4 score = {0, 0, 0, 0};
        const int natural_token = __shfl(token, lane / 4, kWaveSize);
        const bool natural_valid = natural_token >= 0 && natural_token < klen;

         if(__ballot(!natural_valid)==0){
            const int64_t natural_base =
                (static_cast<int64_t>(ks +  natural_token) * hkv + kv_head) * kHeadDim;
            Vec8 k_window[kKPrefetchSlices];
#pragma unroll
            for (int p = 0; p < kKPrefetchSlices; ++p)
                k_window[p] = load8(k + natural_base + p * 32 + (lane % 4) * 8);
#pragma unroll
            for (int i = 0; i < kQkSlices; ++i) {
                const Vec8 natural_k = k_window[i % kKPrefetchSlices];
                if (i + kKPrefetchSlices < kQkSlices)
                    k_window[i % kKPrefetchSlices] = load8(k + natural_base + (i + kKPrefetchSlices) * 32 + (lane % 4) * 8);
                const Vec8 kv = shuffle8(natural_k, r * 4 + g);
                mmac<Element>(qv[i].x, kv.x, score);
                mmac<Element>(qv[i].y, kv.y, score);
            }
        }
        else{
            const int64_t natural_base =
                (static_cast<int64_t>(ks + (natural_valid ? natural_token : 0)) * hkv + kv_head) * kHeadDim;
            Vec8 k_window[kKPrefetchSlices];
#pragma unroll
            for (int p = 0; p < kKPrefetchSlices; ++p)
                k_window[p] = load8(k + natural_base + p * 32 + (lane % 4) * 8);
            if(!natural_valid){
                for (int p = 0; p < kKPrefetchSlices; ++p)
                    k_window[p] = zero;
            }
#pragma unroll
            for (int i = 0; i < kQkSlices; ++i) {
                const Vec8 natural_k = k_window[i % kKPrefetchSlices];
                if (i + kKPrefetchSlices < kQkSlices)
                    k_window[i % kKPrefetchSlices] = natural_valid
                        ? load8(k + natural_base + (i + kKPrefetchSlices) * 32 + (lane % 4) * 8) : zero;
                const Vec8 kv = shuffle8(natural_k, r * 4 + g);
                mmac<Element>(qv[i].x, kv.x, score);
                mmac<Element>(qv[i].y, kv.y, score);
            }
        }
        
        // Online softmax: every lane's four accumulators belong to one head.
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = __shfl(token, g + e * 4, kWaveSize);
            score[e] = r < active && t >= 0 ? score[e] : kNegativeInfinity;
        }
        float m = maximum(maximum(score[0], score[1]), maximum(score[2], score[3]));
        m = maximum(m, shuffle_xor16(m));
        m = maximum(m, __shfl_xor(m, 32, kWaveSize));
        m = maximum(row_max, m);
        const float alpha = SplitMode && row_sum == 0 ? 0.0f : exp2_fast(row_max - m);
        Float4 prob;
#pragma unroll
        for (int e = 0; e < 4; ++e)
            prob[e] = (!SplitMode || (r < active && __shfl(token, g + e * 4, kWaveSize) >= 0))
                ? exp2_fast(score[e] - m) : 0.0f;
        float sum = (prob[0] + prob[1]) + (prob[2] + prob[3]);
        sum += shuffle_xor16(sum);
        sum += __shfl_xor(sum, 32, kWaveSize);
        row_sum = row_sum * alpha + sum;
        row_max = m;
        // Uniform vote skips an exact identity rescale for the whole wave.
        if (!has_seen_valid || __ballot(r < active && alpha != 1.0f) != 0) {
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
#pragma unroll
            for (int e = 0; e < 4; ++e) pv[d][e] *= alpha;
        }
        }

        Half4 pvec = zero.x;
        Element* pe = reinterpret_cast<Element*>(&pvec);
        const Float4 transposed = inlane_v ? prob : transpose_probability(prob, g);
#pragma unroll
        for (int e = 0; e < 4; ++e) pe[e] = as_element<Element>(transposed[e]);
        if constexpr (inlane_v) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const Half4 vv = assemble_inlane_value_operand(lane_values[0][c], lane_values[1][c],
                        lane_values[2][c], lane_values[3][c], b);
                    mmac<Element>(pvec, vv, pv[4 * c + b]);
                }
            }
        } else {
        const int vt = __shfl(token, lane / 4, kWaveSize);
        Half4 natural = vt >= 0 ? v_prefetch[0] : zero.x;
#pragma unroll
        for (int d = 0; d < kVFragments; ++d) {
            Half4 next = zero.x;
            if (d + 1 < kVFragments && vt >= 0) next = v_prefetch[d + 1];
            const Half4 vv = transpose_v(natural, g, r);
            mmac<Element>(pvec, vv, pv[d]);
            natural = next;
        }
        }
        // A wholly empty leading reference block poisons the target's online
        // softmax. An empty 16-token subtile inside a valid larger block does not.
        if ((start + kTileTokens >= count || ((start + kTileTokens) & reference_mask) == 0)
                && !SplitMode && !has_seen_valid)
        {
            empty_first_reference = true;
            break;  // an empty leading reference block poisons the entire row
        }
    }
    // Four adjacent columns in the G12 in-lane layout share one vector store.
    // Fresh contiguous out/partial allocation and D256 guarantee 8/16B stores.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool vector_store = inlane_v;
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

// ---- Host contract checks ----


} // namespace

// ---- Public ordinary-precision entry and dispatch ----
at::Tensor sparse_gqa_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& indices, const at::Tensor& cu_q,
    const at::Tensor& cu_k, const at::Tensor& kv_lens,
    double scale, int64_t max_seqlen_q) {
    check_contiguous_accelerator_tensor(q, "q");
    check_contiguous_accelerator_tensor(k, "k");
    check_contiguous_accelerator_tensor(v, "v");
    check_contiguous_accelerator_tensor(indices, "indices");
    check_contiguous_accelerator_tensor(cu_q, "cu_q");
    check_contiguous_accelerator_tensor(cu_k, "cu_k");
    check_contiguous_accelerator_tensor(kv_lens, "kv_lens");

    const auto dtype = q.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "q must be torch.float16 or torch.bfloat16");
    TORCH_CHECK(k.scalar_type() == dtype && v.scalar_type() == dtype,
                "q, k and v must have the same dtype");
    TORCH_CHECK(indices.scalar_type() == at::kInt && cu_q.scalar_type() == at::kInt
                    && cu_k.scalar_type() == at::kInt && kv_lens.scalar_type() == at::kInt,
                "indices, cu_q, cu_k and kv_lens must be torch.int32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim,
                "q must have shape [total_q, num_q_heads, 256]");
    TORCH_CHECK(k.dim() == 3 && k.size(2) == kHeadDim,
                "k must have shape [total_k, num_kv_heads, 256]");
    TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
    TORCH_CHECK((indices.dim() == 2 || indices.dim() == 3) && indices.size(0) == q.size(0)
                    && indices.size(-1) > 0,
                "indices must have shape [total_q, topk] or [total_q, Hkv, topk] with positive topk");
    TORCH_CHECK(indices.dim() == 2 || indices.size(1) == k.size(1),
                "indices Hkv dimension must match K/V");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0,
                "num_q_heads must be divisible by num_kv_heads");
    const int64_t group_size = q.size(1) / k.size(1);
    TORCH_CHECK(group_size >= 1 && group_size <= 16, "GQA group_size must be in [1, 16]");
    const int64_t batch = kv_lens.numel();
    TORCH_CHECK(kv_lens.dim() == 1, "kv_lens must be one-dimensional");
    TORCH_CHECK(cu_q.dim() == 1 && cu_q.numel() == batch + 1,
                "cu_q must have batch + 1 elements");
    TORCH_CHECK(cu_k.dim() == 1 && cu_k.numel() == batch + 1,
                "cu_k must have batch + 1 elements");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device()
                    && q.device() == indices.device() && q.device() == cu_q.device()
                    && q.device() == cu_k.device() && q.device() == kv_lens.device(),
                "all inputs must be on the same device");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_q <= INT32_MAX,
                "max_seqlen_q must be positive and fit in int32");
    TORCH_CHECK(q.size(0) <= INT32_MAX && k.size(0) <= INT32_MAX
                    && q.size(1) <= INT32_MAX && indices.size(-1) <= INT32_MAX
                    && batch * k.size(1) <= INT32_MAX,
                "token counts, head counts, topk and batch * num_kv_heads must fit in int32");

    auto out = torch::empty_like(q);
    if (q.numel() == 0 || batch == 0) return out;
    at::hip::HIPGuardMasqueradingAsCUDA device_guard{static_cast<char>(q.get_device())};
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipDeviceProp_t properties{};
    bool static_g12 = false;
    if (group_size == 12) {
        C10_HIP_CHECK(hipGetDeviceProperties(&properties, q.get_device()));
        static_g12 = std::strncmp(properties.gcnArchName, "gfx938", 6) == 0
            || std::strncmp(properties.gcnArchName, "gfx936", 6) == 0;
    }
    const int reference_slot_block_size = q.size(0) <= 32 ? 32 : (q.size(0) <= 128 ? 64 : (q.size(0) <= 512 ? 32 : 16));
    const int splits = choose_splits(q.size(0), k.size(1), group_size, indices.size(-1), q.get_device());
    at::Tensor partial, lse;
    if (splits > 1) {
        partial = at::empty({splits, q.size(0), q.size(1), kHeadDim}, q.options().dtype(at::kFloat));
        lse = at::empty({splits, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    }
    const dim3 grid(static_cast<unsigned int>(max_seqlen_q),
                    static_cast<unsigned int>(batch * k.size(1)), splits);
    FP16_SWITCH(dtype == at::kHalf, [&] {
        auto launch = [&](auto group_tag, auto split_tag) {
            sparse_gqa_attention_kernel<elem_type, decltype(group_tag)::value, decltype(split_tag)::value><<<grid, kWaveSize, 0, stream>>>(
                reinterpret_cast<const elem_type*>(q.data_ptr()),
                reinterpret_cast<const elem_type*>(k.data_ptr()),
                reinterpret_cast<const elem_type*>(v.data_ptr()),
                reinterpret_cast<elem_type*>(out.data_ptr()), indices.data_ptr<int32_t>(),
                cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(),
                kv_lens.data_ptr<int32_t>(), static_cast<int>(q.size(1)),
                static_cast<int>(k.size(1)), static_cast<int>(indices.size(-1)),
                indices.stride(0),
                indices.dim() == 3 ? indices.stride(1) : 0, reference_slot_block_size,
                partial.defined() ? partial.data_ptr<float>() : nullptr,
                lse.defined() ? lse.data_ptr<float>() : nullptr, splits, static_cast<int>(q.size(0)));
        };
        if (group_size == 12) {
            if (splits > 1) launch(std::integral_constant<int,12>{}, std::true_type{});
            else if (static_g12) launch(std::integral_constant<int,12>{}, std::false_type{});
            else launch(std::integral_constant<int,0>{}, std::false_type{});
        } else {
            launch(std::integral_constant<int,0>{}, std::false_type{});
        }
        C10_HIP_CHECK(hipGetLastError());
        if (splits > 1) launch_split_merge<elem_type>(partial, lse, out, splits, stream);
    });
    return out;
}
