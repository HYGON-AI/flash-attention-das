// BF16 public entry and shared FP32/log2-LSE merge. FP8 has its own translation unit.
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

} // namespace

void sparse_gqa_paged_detail::launch_merge(const float *partial, const float *lse,
                                           at::BFloat16 *out, int64_t rows, int64_t heads,
                                           int splits, int64_t os0, int64_t os1,
                                           hipStream_t stream) {
    using namespace sparse_attention_detail;
    if (splits < 16) {
#define SMALL(P)                                                                                   \
    sparse_gqa_paged_attention_merge_small_kernel<P><<<grid_size(rows *heads), 64, 0, stream>>>(   \
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
        const dim3 grid(grid_size(rows * heads), 16);
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

at::Tensor sparse_gqa_paged_attention(const at::Tensor &q, const at::Tensor &k_cache,
                                      const at::Tensor &v_cache, const at::Tensor &logical_indices,
                                      const at::Tensor &block_table, const at::Tensor &token_to_req,
                                      c10::optional<at::Tensor> out) {
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "paged sparse GQA requires BF16 K/V");
    // Small grids benefit from lane-local V and compile-time split metadata;
    // the same layout regresses large prefill grids, which retain Bf16Cache.
    if (q.is_cuda() && q.dim() == 3 && k_cache.dim() == 4 && q.size(2) == 256 &&
        k_cache.size(1) == 64 && k_cache.size(2) > 0 &&
        q.size(1) == 6 * k_cache.size(2)) {
        hipDeviceProp_t props{};
        C10_HIP_CHECK(hipGetDeviceProperties(&props, q.get_device()));
        if (std::string(props.gcnArchName).compare(0, 6, "gfx936") == 0 &&
            q.size(0) * k_cache.size(2) <= 2 * int64_t(std::max(1, props.multiProcessorCount)))
            return sparse_gqa_paged_detail::run<sparse_gqa_paged_detail::Bf16CacheSmall936>(
                q, k_cache, v_cache, logical_indices, block_table, token_to_req, out);
    }
    return sparse_gqa_paged_detail::run<sparse_gqa_paged_detail::Bf16Cache>(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out);
}
