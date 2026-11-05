// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Paged, weight-free multi-head query / single-head key scoring.
// One-wave MMAC path for H=4/D=128; strided/generic inputs retain a scalar path.
#include "sparse_attention.h"

#include <vector>

namespace {

using sparse_attention_detail::CacheStrides;

// One row per CTA. Four waves reuse Q while covering 64 or 256 columns.
// M=1 keeps the narrower grid so a long prefix still exposes enough CTAs.
__global__ __launch_bounds__(256) void mqa_paged_score_mmac_kernel(
    const at::BFloat16 *q, const at::BFloat16 *k, const int32_t *table, const int32_t *requests,
    const int64_t *positions, const int32_t *lengths, float *logits, int32_t *visible, int64_t rows,
    int64_t columns, int64_t pages, int64_t page_size, int64_t table_width, int64_t num_requests,
    int64_t qs0, int64_t qs1, CacheStrides ks, int64_t ts0, int64_t ts1, int64_t ratio,
    float divisor, int tile_columns) {
    using namespace sparse_attention_detail::mmac;
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    const int r = lane % 16, g = lane / 16;
    const int64_t tile_start = static_cast<int64_t>(blockIdx.y) * tile_columns;
    const int request = requests[row];
    const bool valid_request = request >= 0 && request < num_requests;
    const int64_t count = min((positions[row] + 1) / ratio,
                              valid_request ? lengths[request] / ratio : int64_t(0));
    if (blockIdx.y == 0 && threadIdx.x == 0)
        visible[row] = static_cast<int32_t>(count);
    if (!valid_request || tile_start >= count || tile_start >= columns ||
        tile_start >= table_width * page_size) {
        const int64_t column = tile_start + threadIdx.x;
        if (column < columns && column < tile_start + tile_columns)
            logits[row * columns + column] = -INFINITY;
        return;
    }
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    Vec8 query[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        query[i] = r < 4 ? load8(q + row * qs0 + r * qs1 + i * 32 + g * 8) : zero;
#pragma unroll
    for (int subtile = 0; subtile < 4; ++subtile) {
        if (subtile * 64 >= tile_columns)
            break;
        const int64_t start = tile_start + wave * 16 + subtile * 64;
        if (start >= count || start >= columns || start >= table_width * page_size) {
            if (lane < 16 && start + lane < columns)
                logits[row * columns + start + lane] = -INFINITY;
            continue;
        }
        int page = -1;
        const int64_t column = start + r;
        if (g == 0 && column < count && column < columns && column / page_size < table_width)
            page = table[request * ts0 + (column / page_size) * ts1];
        page = __shfl(page, r, 64);
        const int natural_page = __shfl(page, lane / 4, 64);
        const bool key_valid = natural_page >= 0 && natural_page < pages;
        const int64_t kb = key_valid ? static_cast<int64_t>(natural_page) * ks.block +
                                           ((start + lane / 4) % page_size) * ks.token
                                     : 0;
        Float4 scores = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const Vec8 natural = key_valid ? load8(k + kb + i * 32 + (lane % 4) * 8) : zero;
            const Vec8 key = shuffle8(natural, r * 4 + g);
            dot(query[i].x, key.x, scores);
            dot(query[i].y, key.y, scores);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float sum = fmaxf(scores[e], 0.0f);
            sum += __shfl_xor(sum, 1, 64);
            sum += __shfl_xor(sum, 2, 64);
            const int64_t output_column = start + g + e * 4;
            const int output_page = __shfl(page, g + e * 4, 64);
            if (r == 0 && output_column < columns)
                logits[row * columns + output_column] =
                    output_page >= 0 && output_page < pages ? sum / divisor : -INFINITY;
        }
    }
}

__global__ void mqa_paged_score_generic_kernel(
    const at::BFloat16 *q, const at::BFloat16 *k, const int32_t *table, const int32_t *requests,
    const int64_t *positions, const int32_t *lengths, float *logits, int32_t *visible,
    int64_t columns, int64_t heads, int64_t dim, int64_t pages, int64_t page_size,
    int64_t table_width, int64_t num_requests, int64_t qs0, int64_t qs1, int64_t qs2,
    CacheStrides ks, int64_t ts0, int64_t ts1, int64_t ratio, float divisor) {
    const int64_t tiles = (columns + 63) / 64;
    const int64_t row = blockIdx.x / tiles;
    const int64_t column = (blockIdx.x % tiles) * 64 + threadIdx.x;
    const int64_t request = requests[row];
    const bool valid_request = request >= 0 && request < num_requests;
    const int64_t seq_len = valid_request ? lengths[request] : 0;
    const int64_t count = min((positions[row] + 1) / ratio, seq_len / ratio);
    if (column == 0)
        visible[row] = static_cast<int32_t>(count);
    if (column >= columns)
        return;
    const int64_t logical_page = column / page_size;
    const bool valid = valid_request && column < count && logical_page < table_width;
    const int64_t page = valid ? table[request * ts0 + logical_page * ts1] : -1;
    float score = 0.0f;
    if (page >= 0 && page < pages) {
        const int64_t key_base = page * ks.block + (column % page_size) * ks.token;
        for (int64_t head = 0; head < heads; ++head) {
            float dot = 0.0f;
            for (int64_t d = 0; d < dim; ++d) {
                dot += static_cast<float>(q[row * qs0 + head * qs1 + d * qs2]) *
                       static_cast<float>(k[key_base + d * ks.dim]);
            }
            score += fmaxf(dot, 0.0f);
        }
        score /= divisor;
    } else {
        score = -INFINITY;
    }
    logits[row * columns + column] = score;
}

} // namespace

std::vector<at::Tensor>
mqa_paged_score(const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &page_table,
                const at::Tensor &token_to_req, const at::Tensor &query_positions,
                const at::Tensor &sequence_lengths, int64_t compress_ratio,
                c10::optional<int64_t> num_columns, c10::optional<double> score_scale) {
    using namespace sparse_attention_detail;
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) > 0 && q.size(2) > 0,
                "q must be [rows, heads, head_dim] on the accelerator");
    check_device(k_cache, q, "k_cache");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16,
                "mqa_paged_score requires BF16 q and k_cache");
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1 && k_cache.size(3) == q.size(2),
                "k_cache must be [pages, page_size, 1, head_dim]");
    check_i32(page_table, q, "page_table");
    check_i32(token_to_req, q, "token_to_req");
    check_i32(sequence_lengths, q, "sequence_lengths");
    check_device(query_positions, q, "query_positions");
    TORCH_CHECK(query_positions.scalar_type() == at::kLong, "query_positions must be int64");
    TORCH_CHECK(page_table.dim() == 2, "page_table must be two-dimensional");
    TORCH_CHECK(token_to_req.dim() == 1 && token_to_req.numel() == q.size(0) &&
                    token_to_req.is_contiguous(),
                "token_to_req must be contiguous [rows]");
    TORCH_CHECK(query_positions.dim() == 1 && query_positions.numel() == q.size(0) &&
                    query_positions.is_contiguous(),
                "query_positions must be contiguous [rows]");
    TORCH_CHECK(sequence_lengths.dim() == 1 && sequence_lengths.numel() == page_table.size(0) &&
                    sequence_lengths.is_contiguous(),
                "sequence_lengths must be contiguous [requests]");
    TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
    const double divisor = score_scale.value_or(std::sqrt(static_cast<double>(q.size(2))));
    TORCH_CHECK(std::isfinite(divisor) && divisor > 0, "score_scale must be finite and positive");
    const int64_t columns = num_columns.value_or(page_table.size(1) * k_cache.size(1));
    TORCH_CHECK(columns >= 0, "num_columns must be non-negative");
    TORCH_CHECK(q.size(0) == 0 || (k_cache.size(0) > 0 && k_cache.size(1) > 0 &&
                                   page_table.size(0) > 0 && page_table.size(1) > 0),
                "nonempty queries require nonempty cache and page table");
    at::hip::HIPGuardMasqueradingAsCUDA guard{static_cast<char>(q.get_device())};
    auto logits = at::empty({q.size(0), columns}, q.options().dtype(at::kFloat));
    auto visible = at::empty({q.size(0)}, q.options().dtype(at::kInt));
    if (q.size(0) == 0)
        return {logits, visible};
    if (columns == 0) {
        // The source leaves this output uninitialized; make that corner safe.
        visible.zero_();
        return {logits, visible};
    }
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (q.size(1) == 4 && q.size(2) == 128 && q.stride(2) == 1 && k_cache.stride(3) == 1 &&
        q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && k_cache.stride(0) % 8 == 0 &&
        k_cache.stride(1) % 8 == 0 && is_pointer_aligned_8(q) && is_pointer_aligned_8(k_cache)) {
        const int tile_columns = q.size(0) == 1 ? 64 : 256;
        const dim3 grid(checked_grid_extent(q.size(0)), checked_grid_extent((columns + tile_columns - 1) / tile_columns));
        mqa_paged_score_mmac_kernel<<<grid, 256, 0, stream>>>(
            q.data_ptr<at::BFloat16>(), k_cache.data_ptr<at::BFloat16>(),
            page_table.data_ptr<int32_t>(), token_to_req.data_ptr<int32_t>(),
            query_positions.data_ptr<int64_t>(), sequence_lengths.data_ptr<int32_t>(),
            logits.data_ptr<float>(), visible.data_ptr<int32_t>(), q.size(0), columns,
            k_cache.size(0), k_cache.size(1), page_table.size(1), page_table.size(0), q.stride(0),
            q.stride(1), cache_strides(k_cache), page_table.stride(0), page_table.stride(1),
            compress_ratio, static_cast<float>(divisor), tile_columns);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {logits, visible};
    }
    mqa_paged_score_generic_kernel<<<checked_grid_extent(q.size(0) * ((columns + 63) / 64)), 64, 0, stream>>>(
        q.data_ptr<at::BFloat16>(), k_cache.data_ptr<at::BFloat16>(),
        page_table.data_ptr<int32_t>(), token_to_req.data_ptr<int32_t>(),
        query_positions.data_ptr<int64_t>(), sequence_lengths.data_ptr<int32_t>(),
        logits.data_ptr<float>(), visible.data_ptr<int32_t>(), columns, q.size(1), q.size(2),
        k_cache.size(0), k_cache.size(1), page_table.size(1), page_table.size(0), q.stride(0),
        q.stride(1), q.stride(2), cache_strides(k_cache), page_table.stride(0),
        page_table.stride(1), compress_ratio, static_cast<float>(divisor));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {logits, visible};
}
