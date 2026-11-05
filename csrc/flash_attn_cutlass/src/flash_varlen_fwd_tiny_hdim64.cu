/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

constexpr int kWaveSize = 64;
constexpr int kPairsPerBlock = 4;
constexpr int kThreadsPerBlock = kWaveSize * kPairsPerBlock;
static const bool print_param = get_env_("FLASH_ATTENTION_PRINT_PARAM");

template<typename storage_t>
static __device__ inline void from_float(storage_t &out, float value) {
    if constexpr (std::is_same_v<storage_t, _Float16>) {
        out = static_cast<_Float16>(value);
    } else {
        union {
            uint32_t int32;
            float fp32;
        } u = {0};
        u.fp32 = value;
        uint32_t bits = u.int32;
        bits += 0x8000;
        out = static_cast<uint16_t>(bits >> 16);
    }
}

template<typename storage_t>
static __device__ inline float to_float(storage_t value) {
    if constexpr (std::is_same_v<storage_t, _Float16>) {
        return static_cast<float>(value);
    } else {
        union {
            uint32_t int32;
            float fp32;
        } u = {static_cast<uint32_t>(value) << 16};
        return u.fp32;
    }
}

static __device__ inline float wave_allreduce_sum(float value) {
    for (int mask = kWaveSize / 2; mask >= 1; mask /= 2) {
        value += __shfl_xor(value, mask);
    }
    return value;
}

template<typename storage_t>
__global__ __launch_bounds__(kThreadsPerBlock)
void flash_varlen_fwd_tiny_hdim64_kernel(const Flash_fwd_params params) {
    const int wave_idx = threadIdx.x / kWaveSize;
    const int lane = threadIdx.x % kWaveSize;
    const int pair_idx = blockIdx.x * kPairsPerBlock + wave_idx;
    const int total_pairs = params.b * params.h;
    if (pair_idx >= total_pairs) {
        return;
    }

    const int batch_idx = pair_idx / params.h;
    const int q_head_idx = pair_idx % params.h;
    const int kv_head_idx = q_head_idx / params.h_h_k_ratio;

    const int q_start = params.cu_seqlens_q[batch_idx];
    const int q_end = params.cu_seqlens_q[batch_idx + 1];
    const int q_len = q_end - q_start;
    if (q_len <= 0) {
        return;
    }

    const int k_start = params.cu_seqlens_k[batch_idx];
    int k_len = params.is_seqlens_k_cumulative
        ? params.cu_seqlens_k[batch_idx + 1] - k_start
        : params.cu_seqlens_k[batch_idx];
    if (params.seqused_k != nullptr) {
        k_len = params.seqused_k[batch_idx];
    }
    k_len = k_len < 0 ? 0 : k_len;

    auto *out_ptr = reinterpret_cast<storage_t *>(params.o_ptr);
    auto *lse_ptr = reinterpret_cast<float *>(params.softmax_lse_ptr);
    const auto *q_ptr = reinterpret_cast<const storage_t *>(params.q_ptr);
    const auto *k_ptr = reinterpret_cast<const storage_t *>(params.k_ptr);
    const auto *v_ptr = reinterpret_cast<const storage_t *>(params.v_ptr);

    if (k_len == 0) {
        storage_t zero_value;
        from_float(zero_value, 0.f);
        const int64_t lse_base = static_cast<int64_t>(q_head_idx) * params.total_q;
        for (int row = 0; row < 4; ++row) {
            if (row >= q_len) {
                break;
            }
            const int64_t out_offset =
                static_cast<int64_t>(q_start + row) * params.o_row_stride
                + static_cast<int64_t>(q_head_idx) * params.o_head_stride
                + lane;
            out_ptr[out_offset] = zero_value;
            if (lane == 0) {
                lse_ptr[lse_base + q_start + row] = std::numeric_limits<float>::infinity();
            }
        }
        return;
    }

    float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float k_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float v_reg[4] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row < q_len) {
            const int64_t q_offset =
                static_cast<int64_t>(q_start + row) * params.q_row_stride
                + static_cast<int64_t>(q_head_idx) * params.q_head_stride
                + lane;
            q_reg[row] = to_float(q_ptr[q_offset]);
        }
        if (row < k_len) {
            const int64_t k_offset =
                static_cast<int64_t>(k_start + row) * params.k_row_stride
                + static_cast<int64_t>(kv_head_idx) * params.k_head_stride
                + lane;
            const int64_t v_offset =
                static_cast<int64_t>(k_start + row) * params.v_row_stride
                + static_cast<int64_t>(kv_head_idx) * params.v_head_stride
                + lane;
            k_reg[row] = to_float(k_ptr[k_offset]);
            v_reg[row] = to_float(v_ptr[v_offset]);
        }
    }

    float probs[4][4];
    float row_lse[4];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            float score = q_reg[row] * k_reg[col] * params.scale_softmax;
            score = wave_allreduce_sum(score);
            probs[row][col] = score;
        }
    }

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        const int causal_limit = row + k_len - q_len;
        float row_max = -std::numeric_limits<float>::infinity();
        bool has_valid_key = false;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const bool valid = row < q_len && col < k_len && col <= causal_limit;
            if (!valid) {
                probs[row][col] = -std::numeric_limits<float>::infinity();
                continue;
            }
            has_valid_key = true;
            row_max = fmaxf(row_max, probs[row][col]);
        }
        if (!has_valid_key) {
            row_lse[row] = std::numeric_limits<float>::infinity();
            #pragma unroll
            for (int col = 0; col < 4; ++col) {
                probs[row][col] = 0.f;
            }
            continue;
        }

        float row_sum = 0.f;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const float score = probs[row][col];
            if (score == -std::numeric_limits<float>::infinity()) {
                probs[row][col] = 0.f;
                continue;
            }
            const float prob = expf(score - row_max);
            probs[row][col] = prob;
            row_sum += prob;
        }

        const float inv_row_sum = 1.f / row_sum;
        row_lse[row] = row_max + logf(row_sum);
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            probs[row][col] *= inv_row_sum;
        }
    }

    float out_accum[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            out_accum[row] += probs[row][col] * v_reg[col];
        }
    }

    const int64_t lse_base = static_cast<int64_t>(q_head_idx) * params.total_q;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if (row >= q_len) {
            break;
        }
        storage_t out_value;
        from_float(out_value, out_accum[row]);
        const int64_t out_offset =
            static_cast<int64_t>(q_start + row) * params.o_row_stride
            + static_cast<int64_t>(q_head_idx) * params.o_head_stride
            + lane;
        out_ptr[out_offset] = out_value;
        if (lane == 0) {
            lse_ptr[lse_base + q_start + row] = row_lse[row];
        }
    }
}

template<typename storage_t, int kSeqLen, int Row>
static __device__ __forceinline__ void process_equal_len_row(
        const Flash_fwd_params &params,
        const int lane,
        const int q_start,
        const int q_head_idx,
        const int64_t lse_base,
        const float (&k_reg)[kSeqLen],
        const float (&v_reg)[kSeqLen]) {
    static_assert(kSeqLen >= 2 && kSeqLen <= 3, "kSeqLen must be 2 or 3");
    static_assert(Row < kSeqLen, "Row must be inside kSeqLen");

    // 固定等长 causal 场景下第 Row 行只需要看 K[0..Row]，避免原通用 kernel 的 4x4 无效 score。
    // 强制内联有助于 k_reg/v_reg 标量化，降低 private memory/scratch 风险。
    auto *out_ptr = reinterpret_cast<storage_t *>(params.o_ptr);
    auto *lse_ptr = reinterpret_cast<float *>(params.softmax_lse_ptr);
    const auto *q_ptr = reinterpret_cast<const storage_t *>(params.q_ptr);

    const int64_t q_offset =
        static_cast<int64_t>(q_start + Row) * params.q_row_stride
        + static_cast<int64_t>(q_head_idx) * params.q_head_stride
        + lane;
    const float q_value = to_float(q_ptr[q_offset]);

    float scores[Row + 1];
    #pragma unroll
    for (int col = 0; col <= Row; ++col) {
        // 保持与原 kernel 相同的乘法与 wave 规约顺序，避免引入额外数值差异。
        float score = q_value * k_reg[col] * params.scale_softmax;
        score = wave_allreduce_sum(score);
        scores[col] = score;
    }

    float out_accum = 0.f;
    float row_lse = scores[0];
    if constexpr (Row == 0) {
        // 第 0 行 causal 下只有一个合法 key，softmax 概率必为 1
        out_accum = v_reg[0];
    } else {
        float row_max = scores[0];
        #pragma unroll
        for (int col = 1; col <= Row; ++col) {
            row_max = fmaxf(row_max, scores[col]);
        }

        float exp_scores[Row + 1];
        float row_sum = 0.f;
        #pragma unroll
        for (int col = 0; col <= Row; ++col) {
            const float prob = expf(scores[col] - row_max);
            exp_scores[col] = prob;
            row_sum += prob;
        }

        const float inv_row_sum = 1.f / row_sum;
        row_lse = row_max + logf(row_sum);

        #pragma unroll
        for (int col = 0; col <= Row; ++col) {
            exp_scores[col] *= inv_row_sum;
        }
        #pragma unroll
        for (int col = 0; col <= Row; ++col) {
            out_accum += exp_scores[col] * v_reg[col];
        }
    }

    storage_t out_value;
    from_float(out_value, out_accum);
    const int64_t out_offset =
        static_cast<int64_t>(q_start + Row) * params.o_row_stride
        + static_cast<int64_t>(q_head_idx) * params.o_head_stride
        + lane;
    out_ptr[out_offset] = out_value;
    if (lane == 0) {
        lse_ptr[lse_base + q_start + Row] = row_lse;
    }
}

template<typename storage_t, int kSeqLen>
__global__ __launch_bounds__(kThreadsPerBlock)
void flash_varlen_fwd_tiny_hdim64_equal_len_kernel(const Flash_fwd_params params) {
    static_assert(kSeqLen >= 2 && kSeqLen <= 3, "kSeqLen must be 2 or 3");

    const int wave_idx = threadIdx.x / kWaveSize;
    const int lane = threadIdx.x % kWaveSize;
    const int pair_idx = blockIdx.x * kPairsPerBlock + wave_idx;
    const int total_pairs = params.b * params.h;
    if (pair_idx >= total_pairs) {
        return;
    }

    const int batch_idx = pair_idx / params.h;
    const int q_head_idx = pair_idx % params.h;
    const int kv_head_idx = q_head_idx;

    const int q_start = params.cu_seqlens_q[batch_idx];
    const int k_start = params.cu_seqlens_k[batch_idx];

    const auto *k_ptr = reinterpret_cast<const storage_t *>(params.k_ptr);
    const auto *v_ptr = reinterpret_cast<const storage_t *>(params.v_ptr);

    float k_reg[kSeqLen];
    float v_reg[kSeqLen];
    #pragma unroll
    for (int row = 0; row < kSeqLen; ++row) {
        // 只加载真实长度 kSeqLen 的 K/V，避免通用路径中为长度 4 预留的无效寄存器和加载分支。
        const int64_t k_offset =
            static_cast<int64_t>(k_start + row) * params.k_row_stride
            + static_cast<int64_t>(kv_head_idx) * params.k_head_stride
            + lane;
        const int64_t v_offset =
            static_cast<int64_t>(k_start + row) * params.v_row_stride
            + static_cast<int64_t>(kv_head_idx) * params.v_head_stride
            + lane;
        k_reg[row] = to_float(k_ptr[k_offset]);
        v_reg[row] = to_float(v_ptr[v_offset]);
    }

    const int64_t lse_base = static_cast<int64_t>(q_head_idx) * params.total_q;
    // 编译期展开有效行：L=2 只生成 Row0/1，L=3 只生成 Row0/1/2。
    process_equal_len_row<storage_t, kSeqLen, 0>(params, lane, q_start, q_head_idx, lse_base, k_reg, v_reg);
    if constexpr (kSeqLen >= 2) {
        process_equal_len_row<storage_t, kSeqLen, 1>(params, lane, q_start, q_head_idx, lse_base, k_reg, v_reg);
    }
    if constexpr (kSeqLen >= 3) {
        process_equal_len_row<storage_t, kSeqLen, 2>(params, lane, q_start, q_head_idx, lse_base, k_reg, v_reg);
    }
}

static inline bool can_use_equal_len_specialization(const Flash_fwd_params &params, const int64_t total_k) {
    const int seq_len = params.seqlen_q;
    // L=1 在上层会被规范化为 non-causal，保持默认路径；这里只特化真实命中 tiny causal 的 L=2/3。
    if (seq_len < 2 || seq_len > 3 || params.seqlen_k != seq_len) {
        return false;
    }
    const int64_t expected_total = static_cast<int64_t>(params.b) * seq_len;
    // 通过 max_seqlen 与 total_q/total_k 同时匹配证明所有 batch 都是统一等长，无需读取 GPU 上的 cu_seqlens。
    return params.b > 0
        && params.h == params.h_k
        && params.d == 64
        && params.d_value == 64
        && params.is_seqlens_k_cumulative
        && params.is_causal
        && static_cast<int64_t>(params.total_q) == expected_total
        && static_cast<int64_t>(total_k) == expected_total;
}

template<typename storage_t, int kSeqLen>
void run_mha_varlen_tiny_fwd_dim64_equal_len_(Flash_fwd_params &params, cudaStream_t stream) {
    const int total_pairs = params.b * params.h;
    if (total_pairs == 0) {
        return;
    }
    const dim3 grid((total_pairs + kPairsPerBlock - 1) / kPairsPerBlock);
    const dim3 block(kThreadsPerBlock);
    flash_varlen_fwd_tiny_hdim64_equal_len_kernel<storage_t, kSeqLen><<<grid, block, 0, stream>>>(params);
}

template<typename storage_t>
void run_mha_varlen_tiny_fwd_dim64_(Flash_fwd_params &params, cudaStream_t stream, const int64_t total_k) {
    if (can_use_equal_len_specialization(params, total_k)) {
        if (print_param) {
            printf("VARLEN_TINY_DIM64_EQUAL_LEN_%d PATH\n", params.seqlen_q);
        }
        switch (params.seqlen_q) {
            case 2: run_mha_varlen_tiny_fwd_dim64_equal_len_<storage_t, 2>(params, stream); return;
            case 3: run_mha_varlen_tiny_fwd_dim64_equal_len_<storage_t, 3>(params, stream); return;
            default: break;
        }
    }

    // 保留原通用 tiny kernel 作为 fallback，覆盖混合长度、L=4、BF16、GQA 等所有非专用场景。
    const int total_pairs = params.b * params.h;
    if (total_pairs == 0) {
        return;
    }
    const dim3 grid((total_pairs + kPairsPerBlock - 1) / kPairsPerBlock);
    const dim3 block(kThreadsPerBlock);
    flash_varlen_fwd_tiny_hdim64_kernel<storage_t><<<grid, block, 0, stream>>>(params);
}

}  // namespace

void run_mha_varlen_tiny_fwd_dim64(Flash_fwd_params &params, cudaStream_t stream, const int64_t total_k) {
    if (params.is_bf16) {
        run_mha_varlen_tiny_fwd_dim64_<uint16_t>(params, stream, total_k);
    } else {
        run_mha_varlen_tiny_fwd_dim64_<_Float16>(params, stream, total_k);
    }
}
