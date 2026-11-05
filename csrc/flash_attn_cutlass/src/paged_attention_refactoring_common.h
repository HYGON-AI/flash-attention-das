// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "hip/hip_runtime.h"
#include "paged_attention.h"

struct PagedAttentionOptPolicy {
    static constexpr int stages = 2;
    static constexpr int block_size = 64;
    static constexpr int num_threads = 256;
    static constexpr int q_lds_pad = 8;
    static constexpr int reduce_lds_pad = 4;
    static constexpr int max_q_rows_per_tile = 32;
};

__forceinline__ int get_nums_simd()
{
    int device = 0;
    TORCH_CHECK(0 == hipGetDevice(&device), "hipGetDevice fail!");
    hipDeviceProp_t prop;
    TORCH_CHECK(0 == hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties fail!");
    return 4 * prop.multiProcessorCount;
}

__device__ __forceinline__ int get_valid_num_partitions(int valid_num_block_pages, int balanced_seq_len_tile)
{
    constexpr int MAIN_KERNEL_NUM_WARPS = 4;
    constexpr int MIN_PAGES_PER_PARTITION = MAIN_KERNEL_NUM_WARPS;
    int max_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(valid_num_block_pages, MIN_PAGES_PER_PARTITION));
    int valid_num_partition = MAX_T(1, MIN_T(balanced_seq_len_tile, max_seq_len_tile));
    return valid_num_partition;
}

template <typename scalarx8_t>
static __device__ __forceinline__ void inline_global_load_dwordx4_no_wait(scalarx8_t &data, const uint64_t *addr)
{
    __builtin_amdgcn_sched_barrier(0);
    // 不使用 glc slc
    asm volatile("global_load_dwordx4 %0 %1 off\n"
                : "=v" (data)
                : "v" (addr));
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template <typename scalarx4_t>
static __device__ __forceinline__ void inline_global_load_dwordx2_no_wait(scalarx4_t &data, const uint64_t *addr)
{
    __builtin_amdgcn_sched_barrier(0);
    // 不使用 glc slc
    asm volatile("global_load_dwordx2 %0 %1 off\n"
                : "=v" (data)
                : "v" (addr));
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template <typename scalarx2_t>
static __device__ __forceinline__ void inline_global_load_dword_no_wait(scalarx2_t &data, const uint64_t *addr)
{
    __builtin_amdgcn_sched_barrier(0);
    // 不使用 glc slc
    asm volatile("global_load_dword %0 %1 off\n"
                : "=v" (data)
                : "v" (addr));
    __builtin_amdgcn_sched_barrier(0);
    return;
}

static __device__ __forceinline__ void inline_wait_vmcnt(const int wait_count=0) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)\n":: "n"(wait_count));
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template<typename T>
static __forceinline__ __device__ T __shfl_xor_tmp(T x, const int lane_mask) {
    int lane_id = threadIdx.x & 63;
    // 当lane_mask = 32, lane_id ^ lane_mask的作用是 lane 0~31  → lane 32~63, lane 32~63 → lane 0~31
    // << 2 的目的是 __builtin_amdgcn_ds_bpermute 的第一个参数是 字节偏移, sizeof(float) = 4
    // 此函数的功能是交换t0 t32, t1 t33...线程数据
    int index   = (lane_id ^ lane_mask) << 2;
    int res = __builtin_amdgcn_ds_bpermute(index, *(int*)&x); // attention, __builtin only support int
    return *(T*)&res;
}


template<typename T>
static __forceinline__ __device__ T __shfl_swap16(T x) {
    // 含义为 result[lane_id] = x[lane_id ^ 16];
    // 作用是交换laneid 0-15/16-31的数据  并交换32-47/48-63的数据
    int result = __builtin_amdgcn_ds_swizzle(*(int*)&x, 0x401F);
    return *(T*)&result;
}

template<typename scalarx4_t, bool is_half>
static __forceinline__ __device__ void builtin_amdgcn_mmac(const scalarx4_t& reg_a, const scalarx4_t& reg_b, float4_t& reg_c)
{
  if constexpr (is_half){
    reg_c=__builtin_hcu_mmac_f32_16x16x16_f16(reg_a,reg_b,reg_c);
  }else{
    reg_c=__builtin_hcu_mmac_f32_16x16x16_bf16(*(v4bh*)&reg_a,*(v4bh*)&reg_b,reg_c);
  }
}

template<typename scalar_t> 
static __device__ __forceinline__ scalar_t from_float(float f){
    scalar_t out;
    if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
        out=f;
    }
    else{
        uint32_t u = *(uint32_t*)(&f);
        // u += 0x7fff + ((u >> 16) & 1); 
        u += 0x8000; 
        out = u>>16;
    }
    return out;
}

template <typename scalarx8_t, typename scalar_t, bool is_e4m3>
static __forceinline__ __device__ scalarx8_t decode_fp8x8_to_b16x8(const uint8x8_t& raw)
{
    scalarx8_t out;
    for(int i = 0;i< 8;i++){
        out[i] = uint82half<scalar_t, is_e4m3>(raw[i]);
    }
    return out;
}

template <typename scalarx4_t, typename scalar_t, bool is_e4m3>
static __forceinline__ __device__ scalarx4_t decode_fp8x4_to_b16x4(const uint8x4_t& raw)
{
    scalarx4_t out;
    for(int i = 0;i < 4;i++){
        out[i] = uint82half<scalar_t, is_e4m3>(raw[i]);
    }
    return out;
}

template <int Mloop, int HEAD_SIZE_V>
__forceinline__ __device__ void attention_initialize(
                              float scores_max[Mloop],
                              float scores_sum[Mloop],
                              float4_t acc_o[Mloop][HEAD_SIZE_V / 16]
                            )
{
    for(int m = 0;m < Mloop;m++){
        scores_max[m] = -FLT_MAX;
        scores_sum[m] = 0.0f;
    }
    for(int m = 0;m < Mloop;m++){
        for(int i = 0;i < (HEAD_SIZE_V / 16);i++){
        (acc_o[m][i])[0] = 0.0f;
        (acc_o[m][i])[1] = 0.0f;
        (acc_o[m][i])[2] = 0.0f;
        (acc_o[m][i])[3] = 0.0f;
        }
    }
    return;
}

template <int Mloop, int BLOCK_SIZE>
__forceinline__ __device__ void reduce_max(float4_t scores[Mloop][BLOCK_SIZE / 16],
                                            float scores_max[Mloop],
                                            float scores_max_cur[Mloop]
                                          )
{
    for(int m = 0;m < Mloop;m++){
        float max = -INFINITY;
        // 线程内 max
        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
            for(int j = 0;j < 4;j++){
                max = MAX_T(max, (scores[m][i])[j]);
            }
        }

        // 线程间 max, 如 0 16 32 48为一组
        // 假设t0 t16 t32 t48中的值分别为0 1 2 3
        // __shfl_xor_tmp(max, 32) 之后, 结果为2 3 2 3
        // __shfl_swap16(max) 之后, 结果为3 3 3 3
        max = MAX_T(max, __shfl_xor_tmp(max, 32));
        max = MAX_T(max, __shfl_swap16(max));

        scores_max_cur[m] = MAX_T(max, scores_max[m]);
    }

    return;
}

template <int Mloop, int BLOCK_SIZE>
__forceinline__ __device__ void scale_apply_exp2(float4_t scores[Mloop][BLOCK_SIZE / 16],
                                                  float scores_max_cur[Mloop],
                                                  float softmax_scale_log2
                                                )
{
    for(int m = 0;m < Mloop;m++){
        const float max_scaled = scores_max_cur[m] == -INFINITY ? 0.0f : scores_max_cur[m] * softmax_scale_log2;
        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
            (scores[m][i])[0] = __builtin_amdgcn_exp2f(((scores[m][i])[0] * softmax_scale_log2) - max_scaled);
            (scores[m][i])[1] = __builtin_amdgcn_exp2f(((scores[m][i])[1] * softmax_scale_log2) - max_scaled);
            (scores[m][i])[2] = __builtin_amdgcn_exp2f(((scores[m][i])[2] * softmax_scale_log2) - max_scaled);
            (scores[m][i])[3] = __builtin_amdgcn_exp2f(((scores[m][i])[3] * softmax_scale_log2) - max_scaled);
        }
    }
    return;
}

template <int Mloop, int BLOCK_SIZE>
__forceinline__ __device__ void reduce_sum(float4_t scores[Mloop][BLOCK_SIZE / 16],
                                            float scores_sum_cur[Mloop]                                        
                                          )
{
    for(int m = 0;m < Mloop;m++){
        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
            for(int j = 0;j < 4;j++){
                scores_sum_cur[m] += (scores[m][i])[j];
            }
        }

        scores_sum_cur[m] += __shfl_xor_tmp(scores_sum_cur[m], 32);
        scores_sum_cur[m] += __shfl_xor_tmp(scores_sum_cur[m], 16);
    }
    return;
}

template <int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE>
__forceinline__ __device__ void softmax_rescale_o(float4_t scores[Mloop][BLOCK_SIZE / 16],
                                                  float scores_max[Mloop],
                                                  float scores_sum[Mloop],
                                                  float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                  float softmax_scale_log2
                                                )
{
    float scores_max_cur[Mloop];

    // scores_max: old_max, scores_max_cur: new_max
    reduce_max<Mloop, BLOCK_SIZE>(scores, scores_max, scores_max_cur);

    for(int m = 0;m < Mloop;m++){
        float scores_max_cur_reg = (scores_max_cur[m] == -INFINITY) ? 0.0f : scores_max_cur[m];
        // scores_scale = 2^((max_old - max_new) * softmax_scale_log2) = e^((max_old - max_new) * softmax_scale)
        float scores_scale = __builtin_amdgcn_exp2f((scores_max[m] - scores_max_cur_reg) * softmax_scale_log2);
        // scores_sum = scores_sum * scores_scale, 对应公式d(i-1) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        scores_sum[m] *= scores_scale;

        // 对应 公式 O(i-1) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        for(int i = 0;i < (HEAD_SIZE_V / 16);i++){
            (acc_o[m][i])[0] *= scores_scale;
            (acc_o[m][i])[1] *= scores_scale;
            (acc_o[m][i])[2] *= scores_scale;
            (acc_o[m][i])[3] *= scores_scale;
        }
    }
    // 对 scores 计算 指数, 对应公式 e^(xi[j] - m(i))
    // S = 2^(S * softmax_scale_log2 - max_current * softmax_scale_log2) = e^(S * softmax_scale - max_current * softmax_scale)
    scale_apply_exp2<Mloop, BLOCK_SIZE>(scores, scores_max_cur, softmax_scale_log2);

    float scores_sum_cur[Mloop];
    for(int m = 0;m < Mloop;m++){
      scores_sum_cur[m] = 0.0f;
    }

    // scores_sum: old_sum, scores_sum_cur: new_sum
    // 每个线程的 scores_sum_cur 结果为当前行的和
    reduce_sum<Mloop, BLOCK_SIZE>(scores, scores_sum_cur);

    // 更新 sum max
    for(int m = 0;m < Mloop;m++){
        // 对应公式 d(i) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale) +  row_sum(e^((S - max_current) * softmax_scale))
        scores_sum[m] += scores_sum_cur[m];

        scores_max[m] = scores_max_cur[m];
    }
    
    return;
}

template <typename scalar_t, int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void apply_attention_sink(
    float scores_max[Mloop],
    float scores_sum[Mloop],
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
    const scalar_t* __restrict__ s_aux_ptr,
    float scale_softmax,
    float scale_softmax_log2,
    int warp_id,
    int partition_idx,
    int mtp,
    int num_heads,
    int num_kv_heads,
    int kv_head_idx)
{
    // Sink 在整个序列中只能出现一次，并且只能由一个 warp 添加。
    if (s_aux_ptr == nullptr ||
        partition_idx != 0 ||
        warp_id != 0) {
        return;
    }

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_in_tile = lane_id & 15;

    // num_heads = mtp * num_q_heads
    const int num_q_heads = num_heads / mtp;
    const int q_heads_per_kv_head =
        num_q_heads / num_kv_heads;
    const int rows_per_kv_head =
        num_heads / num_kv_heads;
    const int q_head_begin =
        kv_head_idx * q_heads_per_kv_head;

    for (int m = 0; m < Mloop; ++m) {
        // row 为当前处理的 global q 的行号, 包含 seqlen_q 和 num_q_heads 维度
        const int row = m * 16 + row_in_tile;

        if (row >= rows_per_kv_head) {
            continue;
        }
        // q_token_idx 为 当前 row 对应的 seqlen_q 的索引
        const int q_token_idx =
            row / q_heads_per_kv_head;
        // local_q_head 为当前 row 对应的局部 head 索引
        const int local_q_head =
            row % q_heads_per_kv_head;
        // global_q_head 为当前 row 对应的 全局 head 索引
        const int global_q_head =
            q_head_begin + local_q_head;
        // s_aux_ptr 形状为 [seqlen_q * num_q_heads]
        // sink_idx 即为 当前 row 对应的 s_aux_ptr 数据的偏移
        const int sink_idx =
            q_token_idx * num_q_heads +
            global_q_head;

        const float sink =
            to_float(s_aux_ptr[sink_idx]);

        // scores_max 当前保存 raw QK max：
        // 参考 paged_attention_kernel, 可知 sink = sink_raw * scale_softmax
        // sink_raw = sink / scale_softmax, 与 当前 scores_max 的 scale 域 一致
        const float sink_raw =
            sink / scale_softmax;

        if (sink_raw == -INFINITY) {
            continue;
        }

        // 对 sink_raw 的理解:
        //      sink_raw 相当于人为 在 S 矩阵的每一行 增加一个元素
        //      下方的逻辑为 引入 sink_raw 之后, 对 max sum acc_o 进行 rescale
        const float old_max = scores_max[m];
        const float new_max = MAX_T(old_max, sink_raw);

        // old_scale = e ^(old_max * scale_softmax - new_max * scale_softmax)
        const float old_scale =
            (old_max == -INFINITY)
                ? 0.0f
                : __builtin_amdgcn_exp2f(
                      (old_max - new_max) *
                      scale_softmax_log2);

        // Sink 自身在 new_max 基准下的 softmax 权重。
        // sink_weight = e^(sink_raw * scale_softmax - new_max * scale_softmax)
        const float sink_weight =
            __builtin_amdgcn_exp2f(
                (sink_raw - new_max) *
                scale_softmax_log2);

        const float old_sum = scores_sum[m];
        const float new_sum = old_sum * old_scale + sink_weight;
        const float old_output_scale =
            (old_sum + 1e-6f) * old_scale / (new_sum + 1e-6f);

        // acc_o is normalized before P is converted to b16. Sink has V=0,
        // so only the previous normalized output contributes to the new row.
        for (int i = 0; i < HEAD_SIZE_V / 16; ++i) {
            (acc_o[m][i])[0] *= old_output_scale;
            (acc_o[m][i])[1] *= old_output_scale;
            (acc_o[m][i])[2] *= old_output_scale;
            (acc_o[m][i])[3] *= old_output_scale;
        }
        scores_sum[m] = new_sum;
        // 更新 scores_max
        scores_max[m] = new_max;
    }
}

template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS>
__device__ __forceinline__  void epilogue_rescale_acco(
                                float scores_max[Mloop],
                                float scores_sum[Mloop],
                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                float scores_max_new[Mloop],
                                float scores_sum_new[Mloop],
                                float *smem_max,
                                float *smem_sum,
                                float scale_softmax_log2,
                                int warp_id,
                                int num_heads,
                                int num_kv_heads
                            )
{
    int lane_id = threadIdx.x % WARP_SIZE;
    // warp 间 online
    // max/sum register -> lds
    int scale_row = lane_id & 15;
    int scale_col = lane_id >> 4;
    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
    for(int m = 0;m < Mloop;m++){
        int row = m * 16 + scale_row;
        if(row < seqlen_q_per_kv_head){
            // row 在有效范围内时
            // smem_max 形状为(REUSE_KV_TIMES, NUM_WARPS), 行主序
            // smem_sum 形状为(REUSE_KV_TIMES, NUM_WARPS), 行主序
            if(0 == scale_col){
                smem_max[row * NUM_WARPS + warp_id] = scores_max[m];
                smem_sum[row * NUM_WARPS + warp_id] = scores_sum[m];
            }
        }
    }
    __syncthreads();
    
    for(int m = 0;m < Mloop;m++){
        scores_max_new[m] = -INFINITY;
    }
    for(int m = 0;m < Mloop;m++){
        int row = m * 16 + scale_row;
        if(row < seqlen_q_per_kv_head){
            for(int i = 0;i < NUM_WARPS;i++){
                scores_max_new[m] = MAX_T(scores_max_new[m], smem_max[row * NUM_WARPS + i]);
            }
        }
    }
    __syncthreads();
    // 当前 scores_max_new 为 PARTION_SIZE 行的最大值
    
    for(int m = 0;m < Mloop;m++){
        scores_sum_new[m] = 0.0f;
    }
    for(int m = 0;m < Mloop;m++){
        int row = m * 16 + scale_row;
        if(row < seqlen_q_per_kv_head){
            for(int i = 0;i < NUM_WARPS;i++){
                // smem_sum 和 smem_max 对应
                // 因为 smem_max 和 scores_max_new 为 raw max, 所以此处 __builtin_amdgcn_exp2f 需要乘 scale_softmax_log2
                scores_sum_new[m] += smem_sum[row * NUM_WARPS + i] * __builtin_amdgcn_exp2f((smem_max[row * NUM_WARPS + i] - scores_max_new[m]) * scale_softmax_log2);
            }
        }
    }
    __syncthreads();
    // 当前 scores_sum_new 为 PARTION_SIZE 行的 expsum

    // rescale acc_o
    for(int m = 0;m < Mloop;m++){
        int row = m * 16 + scale_row;
        float acc_rescale = 1.0f / (scores_sum_new[m] + 1e-6f);
        if(row < seqlen_q_per_kv_head){
            for(int i = 0;i < HEAD_SIZE_V / 16;i++){
              // 因为 smem_max 和 scores_max_new 为 raw max, 所以此处 __builtin_amdgcn_exp2f 需要乘 scale_softmax_log2
                acc_o[m][i] *=  __builtin_amdgcn_exp2f((scores_max[m] - scores_max_new[m]) * scale_softmax_log2) * acc_rescale;
            }          
        }
    }
    return;
}

template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int Q_LDS_PAD = 0>
__forceinline__ __device__ void prefetch_q_to_lds(const scalar_t *q,
                                                   scalar_t *smem_q,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int mtp,
                                                   int kv_head_idx,
                                                   int batch_idx,
                                                   int q_stride)
{
    int tid = threadIdx.x;
    int num_q_heads = num_heads / mtp;
    int num_q_heads_per_kv_head = num_q_heads / num_kv_heads;
    int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
    const scalar_t *q_ptr = q + batch_idx * q_stride + q_head_idx * HEAD_SIZE;
    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
    int q_seqlen_stride = num_q_heads * HEAD_SIZE;
    int q_seqlen_data_num = num_q_heads_per_kv_head * HEAD_SIZE;

    for(int i = tid * 8;i < seqlen_q_per_kv_head * HEAD_SIZE;i += NUM_THREADS * 8){
        int q_seqlen_idx = i / q_seqlen_data_num;
        int q_offset = q_seqlen_idx * q_seqlen_stride + i % q_seqlen_data_num;
        const int q_row = i / HEAD_SIZE;
        const int q_col = i % HEAD_SIZE;
        *reinterpret_cast<scalarx8_t *>(smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + q_col) =
            *reinterpret_cast<const scalarx8_t *>(q_ptr + q_offset);
    }
    __syncthreads();
}

template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int Q_LDS_PAD = 0, int NUM_Q_TILES>
__forceinline__ __device__ void prefetch_q_to_lds_qtile(const scalar_t *q,
                                                   scalar_t *smem_q,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int mtp,
                                                   int kv_head_idx,
                                                   int batch_idx,
                                                   int q_stride,
                                                   int q_tile_idx
                                                )
{
    // Q [num_seqs, seqlen_q, num_q_heads, head_size]
    int tid = threadIdx.x;
    int num_q_heads = num_heads / mtp;
    int num_q_heads_per_kv_head = num_q_heads / num_kv_heads;
    int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
    const scalar_t *q_ptr = q + batch_idx * q_stride + q_head_idx * HEAD_SIZE;
    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
    int q_seqlen_stride = num_q_heads * HEAD_SIZE;
    int q_seqlen_data_num = num_q_heads_per_kv_head * HEAD_SIZE;

    int rows_per_tile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
    int start_row = q_tile_idx * rows_per_tile;
    int tile_rows = MIN_T(rows_per_tile, seqlen_q_per_kv_head - start_row);

    int num_q_start = start_row * HEAD_SIZE;
    int num_q_end = num_q_start + tile_rows * HEAD_SIZE;

    for(int i = (tid * 8 + num_q_start);i < num_q_end;i += NUM_THREADS * 8){
        int q_seqlen_idx = i / q_seqlen_data_num;
        int q_offset = q_seqlen_idx * q_seqlen_stride + i % q_seqlen_data_num;
        const int local_i = i - num_q_start;
        const int q_row = local_i / HEAD_SIZE;
        const int q_col = local_i % HEAD_SIZE;
        *reinterpret_cast<scalarx8_t *>(smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + q_col) =
            *reinterpret_cast<const scalarx8_t *>(q_ptr + q_offset);
    }
    __syncthreads();

}

template <typename scalar_t, int Mloop, int HEAD_SIZE_V, int NUM_Q_TILES>
__device__ __forceinline__ void apply_attention_sink_q_tile(
    float scores_max[Mloop],
    float scores_sum[Mloop],
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
    const scalar_t* __restrict__ s_aux_ptr,
    float scale_softmax,
    float scale_softmax_log2,
    int warp_id,
    int partition_idx,
    int mtp,
    int num_heads,
    int num_kv_heads,
    int kv_head_idx,
    int q_tile_idx
)
{
    // Sink 在整个序列中只能出现一次，并且只能由一个 warp 添加。
    if (s_aux_ptr == nullptr ||
        partition_idx != 0 ||
        warp_id != 0) {
        return;
    }

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_in_tile = lane_id & 15;

    // num_heads = mtp * num_q_heads
    const int num_q_heads = num_heads / mtp;
    const int q_heads_per_kv_head =
        num_q_heads / num_kv_heads;
    const int rows_per_kv_head =
        num_heads / num_kv_heads;
    const int q_head_begin =
        kv_head_idx * q_heads_per_kv_head;

    int rows_per_qtile = DIVIDE_ROUND_UP(rows_per_kv_head, NUM_Q_TILES);
    int tile_start = q_tile_idx * rows_per_qtile;
    int tile_rows = MIN_T(rows_per_qtile, rows_per_kv_head - tile_start);

    for (int m = 0; m < Mloop; ++m) {
        // row 为当前处理的 global q 的行号, 包含 seqlen_q 和 num_q_heads 维度
        int local_row  = m * 16 + (lane_id & 15);
        if (local_row >= tile_rows) {
            continue;
        }

        int global_row = tile_start + local_row;
        // q_token_idx 为 当前 row 对应的 seqlen_q 的索引
        const int q_token_idx =
            global_row / q_heads_per_kv_head;
        // local_q_head 为当前 row 对应的局部 head 索引
        const int local_q_head =
            global_row % q_heads_per_kv_head;
        // global_q_head 为当前 row 对应的 全局 head 索引
        const int global_q_head =
            q_head_begin + local_q_head;
        // s_aux_ptr 形状为 [seqlen_q * num_q_heads]
        // sink_idx 即为 当前 row 对应的 s_aux_ptr 数据的偏移
        const int sink_idx =
            q_token_idx * num_q_heads +
            global_q_head;

        const float sink =
            to_float(s_aux_ptr[sink_idx]);

        // scores_max 当前保存 raw QK max：
        // 参考 paged_attention_kernel, 可知 sink = sink_raw * scale_softmax
        // sink_raw = sink / scale_softmax, 与 当前 scores_max 的 scale 域 一致
        const float sink_raw =
            sink / scale_softmax;

        if (sink_raw == -INFINITY) {
            continue;
        }

        // 对 sink_raw 的理解:
        //      sink_raw 相当于人为 在 S 矩阵的每一行 增加一个元素
        //      下方的逻辑为 引入 sink_raw 之后, 对 max sum acc_o 进行 rescale
        const float old_max = scores_max[m];
        const float new_max = MAX_T(old_max, sink_raw);

        // old_scale = e ^(old_max * scale_softmax - new_max * scale_softmax)
        const float old_scale =
            (old_max == -INFINITY)
                ? 0.0f
                : __builtin_amdgcn_exp2f(
                      (old_max - new_max) *
                      scale_softmax_log2);

        // Sink 自身在 new_max 基准下的 softmax 权重。
        // sink_weight = e^(sink_raw * scale_softmax - new_max * scale_softmax)
        const float sink_weight =
            __builtin_amdgcn_exp2f(
                (sink_raw - new_max) *
                scale_softmax_log2);

        // 更新 scores_sum
        // 对应 公式 d(i) = d(i-1) * e^(m(i - 1) - m(i)) + row_sum(e^(xi(j) - mi))
        scores_sum[m] =
            scores_sum[m] * old_scale +
            sink_weight;

        // Sink 的 V 为 0，因此 不添加 sink_weight * V
        // 更新 acc_o 对应公式: o(i - 1) = o(i - 1) * e^(m(i - 1) - m(i))
        for (int i = 0; i < HEAD_SIZE_V / 16; ++i) {
            (acc_o[m][i])[0] *= old_scale;
            (acc_o[m][i])[1] *= old_scale;
            (acc_o[m][i])[2] *= old_scale;
            (acc_o[m][i])[3] *= old_scale;
        }
        // 更新 scores_max
        scores_max[m] = new_max;
    }
}

template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int BLOCK_SIZE>
__global__ __launch_bounds__(NUM_THREADS, 1) void paged_attention_combine_opt(
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
    scalar_t* out_tmp,                      // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const int* __restrict__ seq_lens,       // [num_seqs]
    const int balanced_seq_len_tile,
    int num_heads) {
      // grid_size (num_heads_q * seqlen_q, batch)
      extern __shared__ char shared_mem[];
      int head_idx = blockIdx.x;
      int batch_idx = blockIdx.y;

        // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
        int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
        int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
        int valid_num_partition = get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

        if(1 == valid_num_partition){
            return;
        }

      // block_size = 64
      const int tid = threadIdx.x;
      float *smem_exp_sum = (float *)shared_mem;
      float *smem_max = smem_exp_sum + valid_num_partition;
      float global_exp_sum = 0.0f;
      float max_logit = -INFINITY;
      // max, sum 形状为 (batch, seqlen_q, num_q_heads, max_num_partitions)
      // num_heads = seqlen_q * num_q_heads;
      int offset = batch_idx * num_heads * balanced_seq_len_tile  + head_idx * balanced_seq_len_tile ;
      const float *exp_sum = (float *)out_tmp;
      const float *max = (float *)(out_tmp + max_tmp_offset);
      const float* max_logits_ptr = max + offset;
      const float* exp_sums_ptr = exp_sum + offset;

      // out 形状为(batch, seqlen_q, num_q_heads, head_dim), scalar_t
      const scalar_t *out_ptr = out + batch_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
      // temp out 形状为(batch, seqlen_q, num_q_heads, max_num_partitions, head_dim), fp32
      const scalar_t *tmp_out_ptr = out_tmp + out_tmp_offset + offset * HEAD_SIZE;

      // max
      // 线程内 加载数据 的最大值
      for(int i = tid;i< valid_num_partition;i += WARP_SIZE)
      {
          const float l = max_logits_ptr[i];
          smem_max[i] = l;
          // 同一循环中将 exp_sum 保存到 lds
          smem_exp_sum[i] = exp_sums_ptr[i];
          max_logit = MAX_T(max_logit, l);
      }
      // warp内, 即线程块内 最大值, 即 valid_num_partition 范围的最大值
      for(int mask = WARP_SIZE / 2; mask > 0;mask /= 2){
          max_logit = MAX_T(max_logit, __shfl_xor(max_logit, mask));
      }

      __syncthreads();

      // sum
      for(int i = tid;i < valid_num_partition;i += WARP_SIZE){
          // smem_exp_sum[i] 的目的是 减少 global 访存
          const float partition_exp_sum = smem_exp_sum[i];
          if(partition_exp_sum > 0.0f){
              float rescaled_exp_sum = partition_exp_sum * __builtin_amdgcn_exp2f(smem_max[i] - max_logit);
              global_exp_sum += rescaled_exp_sum;
              smem_exp_sum[i] = rescaled_exp_sum;            
          }else{
              smem_exp_sum[i] = 0.0f;
          }
      }
      __syncthreads();
      for(int mask = WARP_SIZE / 2; mask > 0;mask /= 2){
          global_exp_sum += __shfl_xor(global_exp_sum, mask);
      }
      // 1 / di
      const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

      constexpr int vec_size_o = HEAD_SIZE / 64;
      constexpr int vec_size = (vec_size_o == 3) ? 4 : vec_size_o;
      using half_vec= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
      using float_vec= __attribute__( (__vector_size__(vec_size * sizeof(float)) )) float;
      float_vec acc = {0.0f};
      half_vec acc_half;
      if(tid < (HEAD_SIZE / vec_size)){
          for(int j = 0;j < valid_num_partition;j++){
              half_vec temp_acc = *((half_vec *)(tmp_out_ptr + j * HEAD_SIZE + tid * vec_size));
              float temp_sum = smem_exp_sum[j] * inv_global_exp_sum;
              for(int i = 0;i < vec_size;i++){
                  acc[i] += to_float(temp_acc[i]) * temp_sum;
              }
          }
          for(int i = 0;i < vec_size;i++){
            scalar_t temp;
            from_float(temp, acc[i]);
            acc_half[i] = temp;
          }
          *((half_vec *)(out_ptr + tid * vec_size)) = acc_half;
      }
}

// gfx938: combine 中 FP16 -> FP32 用单独的 v_cvt_f32_f16。否则编译器把 "取打包 FP16 的高半字并转换" 与乘加合并成
// 带 op_sel 的 v_fma_mix_f32; 用 --save-temps 编译时 (先输出 .s 再汇编) 这条指令的 op_sel 会丢失, 高半字被当作低半字读取,
// FP16 输出的奇数列等于前一个偶数列。其他架构与 BF16 仍用 to_float。
template <typename T>
__device__ __forceinline__ float combine_parallel_to_float(T x)
{
#if defined(__gfx938__)
    if constexpr (std::is_same<T, _Float16>::value) {
        float f;
        asm("v_cvt_f32_f16 %0, %1" : "=v"(f) : "v"(x));
        return f;
    }
#endif
    return to_float(x);
}

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
__global__ __launch_bounds__(256, 1) void paged_attention_combine_parallel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ out_tmp,
    const int* __restrict__ seq_lens,
    int balanced_seq_len_tile,
    int num_heads)
{
    extern __shared__ char shared_mem[];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;

    const int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    const int valid_num_partition =
        get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);
    if (valid_num_partition == 1) {
        return;
    }

    float* smem_weight = reinterpret_cast<float*>(shared_mem);
    float* smem_max = smem_weight + balanced_seq_len_tile;
    float* smem_partial = smem_max + balanced_seq_len_tile;
    float* smem_scalar = smem_partial + 4 * HEAD_SIZE;
    const int offset = (batch_idx * num_heads + head_idx) * balanced_seq_len_tile;
    const float* exp_sums = reinterpret_cast<const float*>(out_tmp) + offset;
    const float* max_logits =
        reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + offset;
    const scalar_t* tmp_out = out_tmp + out_tmp_offset + offset * HEAD_SIZE;

    if (HEAD_SIZE != 128 || valid_num_partition < 256) {
    // One warp prepares the same normalized partition weights as the original
    // combine kernel. The remaining warps join for the output reduction.
    if (warp == 0) {
        float local_max = -INFINITY;
        for (int i = lane; i < valid_num_partition; i += WARP_SIZE) {
            const float value = max_logits[i];
            smem_max[i] = value;
            smem_weight[i] = exp_sums[i];
            local_max = MAX_T(local_max, value);
        }
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
        }
        if (lane == 0) {
            smem_scalar[0] = local_max;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float global_max = smem_scalar[0];
        float local_sum = 0.0f;
        for (int i = lane; i < valid_num_partition; i += WARP_SIZE) {
            const float exp_sum = smem_weight[i];
            const float weight = exp_sum > 0.0f
                ? exp_sum * __builtin_amdgcn_exp2f(smem_max[i] - global_max)
                : 0.0f;
            smem_weight[i] = weight;
            local_sum += weight;
        }
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            local_sum += __shfl_xor(local_sum, mask);
        }
        if (lane == 0) {
            smem_scalar[0] = __fdividef(1.0f, local_sum + 1e-6f);
        }
    }
    __syncthreads();
    } else {
        // Large partitions: spread the max and weight reductions over all four warps.
        constexpr int NUM_WARPS = 256 / WARP_SIZE;
        float local_max = -INFINITY;
        for (int i = tid; i < valid_num_partition; i += 256) {
            const float value = max_logits[i];
            smem_max[i] = value;
            smem_weight[i] = exp_sums[i];
            local_max = MAX_T(local_max, value);
        }
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
        }
        if (lane == 0) smem_scalar[warp] = local_max;
        __syncthreads();

        if (warp == 0) {
            float value = lane < NUM_WARPS ? smem_scalar[lane] : -INFINITY;
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                value = MAX_T(value, __shfl_xor(value, mask));
            }
            if (lane == 0) smem_scalar[0] = value;
        }
        __syncthreads();

        const float global_max = smem_scalar[0];
        float local_sum = 0.0f;
        for (int i = tid; i < valid_num_partition; i += 256) {
            const float exp_sum = smem_weight[i];
            const float weight = exp_sum > 0.0f
                ? exp_sum * __builtin_amdgcn_exp2f(smem_max[i] - global_max)
                : 0.0f;
            smem_weight[i] = weight;
            local_sum += weight;
        }
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            local_sum += __shfl_xor(local_sum, mask);
        }
        if (lane == 0) smem_scalar[warp] = local_sum;
        __syncthreads();

        if (warp == 0) {
            float value = lane < NUM_WARPS ? smem_scalar[lane] : 0.0f;
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                value += __shfl_xor(value, mask);
            }
            if (lane == 0) smem_scalar[0] = __fdividef(1.0f, value + 1e-6f);
        }
        __syncthreads();
    }

    constexpr int vec_size_o = HEAD_SIZE / 64;
    constexpr int vec_size = (vec_size_o == 3) ? 4 : vec_size_o;
    using scalar_vec = __attribute__((__vector_size__(vec_size * sizeof(scalar_t)))) scalar_t;
    using float_vec = __attribute__((__vector_size__(vec_size * sizeof(float)))) float;
    const float inv_sum = smem_scalar[0];
    float_vec acc0 = {0.0f};
    float_vec acc1 = {0.0f};
    float_vec acc2 = {0.0f};
    float_vec acc3 = {0.0f};
    if (lane < HEAD_SIZE / vec_size) {
        int j = warp;
        for (; j + 12 < valid_num_partition; j += 16) {
            const scalar_vec value0 =
                *reinterpret_cast<const scalar_vec*>(tmp_out + j * HEAD_SIZE + lane * vec_size);
            const scalar_vec value1 =
                *reinterpret_cast<const scalar_vec*>(tmp_out + (j + 4) * HEAD_SIZE + lane * vec_size);
            const scalar_vec value2 =
                *reinterpret_cast<const scalar_vec*>(tmp_out + (j + 8) * HEAD_SIZE + lane * vec_size);
            const scalar_vec value3 =
                *reinterpret_cast<const scalar_vec*>(tmp_out + (j + 12) * HEAD_SIZE + lane * vec_size);
            const float weight0 = smem_weight[j] * inv_sum;
            const float weight1 = smem_weight[j + 4] * inv_sum;
            const float weight2 = smem_weight[j + 8] * inv_sum;
            const float weight3 = smem_weight[j + 12] * inv_sum;
            for (int i = 0; i < vec_size; ++i) {
                acc0[i] += combine_parallel_to_float(value0[i]) * weight0;
                acc1[i] += combine_parallel_to_float(value1[i]) * weight1;
                acc2[i] += combine_parallel_to_float(value2[i]) * weight2;
                acc3[i] += combine_parallel_to_float(value3[i]) * weight3;
            }
        }
        for (; j < valid_num_partition; j += 4) {
            const scalar_vec value =
                *reinterpret_cast<const scalar_vec*>(tmp_out + j * HEAD_SIZE + lane * vec_size);
            const float weight = smem_weight[j] * inv_sum;
            for (int i = 0; i < vec_size; ++i) {
                acc0[i] += combine_parallel_to_float(value[i]) * weight;
            }
        }
        for (int i = 0; i < vec_size; ++i) {
            smem_partial[warp * HEAD_SIZE + lane * vec_size + i] =
                (acc0[i] + acc1[i]) + (acc2[i] + acc3[i]);
        }
    }
    __syncthreads();

    if (warp == 0 && lane < HEAD_SIZE / vec_size) {
        scalar_vec result;
        for (int i = 0; i < vec_size; ++i) {
            const int col = lane * vec_size + i;
            const float sum = smem_partial[col]
                + smem_partial[HEAD_SIZE + col]
                + smem_partial[2 * HEAD_SIZE + col]
                + smem_partial[3 * HEAD_SIZE + col];
            scalar_t converted;
            from_float(converted, sum);
            result[i] = converted;
        }
        scalar_t* out_head = out + (batch_idx * num_heads + head_idx) * HEAD_SIZE;
        *reinterpret_cast<scalar_vec*>(out_head + lane * vec_size) = result;
    }
}