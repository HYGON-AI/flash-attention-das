// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// ===========================================================================
// gfx92a 16 位 / E5M2 KV PA 实现（BF16/FP16 Q，同类型 16 位 KV 或 E5M2 KV），layout=0 与 BSHD 共用。
// 由 BHSD 的 gfx92a 实现（paged_attention_refactoring_bhsd.cu 在 gfx92a 设备编译中生效的代码）改写而来：
// 直接加载路径、Q tile 拆分、行切分与长序列路径、partition 合并两种布局相同；
// 差别只在 K/V 寻址与 V 的读取方式（layout=0 的 V 已转置，每个通道的 token 连续）。
//
// 本文件不是普通头文件：没有 include guard，由各布局的 .cu 在自己的命名空间里包含，每包含一次生成一套实现：
//   paged_attention_refactoring_bshd.cu      namespace bshd_gfx92a
//   paged_attention_refactoring.cu           namespace l0_gfx92a
// kernel 名与原来相同，每个 kernel 的指令与合并前逐条相同。
//
// 包含前需定义（包含结束时本文件会 #undef）：
//   PA_B16_92A_LAYOUT      PA_LAYOUT_L0 / PA_LAYOUT_BSHD
// 布局之间的差别用 #if PA_B16_92A_LAYOUT == ... 区分。
// ===========================================================================

#ifndef PA_LAYOUT_L0
#define PA_LAYOUT_L0 0      // layout=0（custom）：V 转置
#define PA_LAYOUT_BHSD 1    // layout=1
#define PA_LAYOUT_BSHD 2    // layout=2
#endif

#if !defined(PA_B16_92A_LAYOUT)
#error "paged_attention_refactoring_b16_gfx92a.h: 包含前需定义 PA_B16_92A_LAYOUT"
#endif

// 寻址: BSHD 的 K/V cache 为 [num_blocks, block_size, num_kv_heads, head_dim], 同一 KV head 相邻 token 相隔
// num_kv_heads * head_dim, head 起点相隔 head_dim; layout=0 的 K 为 [num_blocks, num_kv_heads, block_size, head_dim],
// V 为 [num_blocks, num_kv_heads, head_dim, block_size] (已转置)。其余算法、调度与 BHSD 92a 实现相同;
// 行切分与长序列出口 (按 BHSD 连续整块加载) 两种布局都不使用。

// Host-only tuning state: read once per process, before launching this TU's kernels.
// Use a new process when comparing configurations; defaults reproduce candidate 04.
namespace {
struct BhsdRuntimeTuning {
    int q_tiles;
    int partitions;
    int combine;
};

int read_bhsd_tuning_integer(const char *name, int maximum)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr) return 0;
    TORCH_CHECK(raw[0] != '\0', name, " must be a nonempty decimal integer in [0, ", maximum, "]");
    int value = 0;
    for (const char *p = raw; *p != '\0'; ++p) {
        TORCH_CHECK(*p >= '0' && *p <= '9', name,
                    " must contain decimal digits only; received '", raw, "'");
        // value <= maximum after each preceding digit, so this cannot overflow.
        value = value * 10 + (*p - '0');
        TORCH_CHECK(value <= maximum, name, " must be in [0, ", maximum,
                    "]; received '", raw, "'");
    }
    return value;
}

const BhsdRuntimeTuning &bhsd_runtime_tuning()
{
    static const BhsdRuntimeTuning tuning = []() {
        BhsdRuntimeTuning value = {
            read_bhsd_tuning_integer("PA_BHSD_Q_TILES", 4),
            read_bhsd_tuning_integer("PA_BHSD_PARTITIONS", 512),
            read_bhsd_tuning_integer("PA_BHSD_COMBINE", 1)
        };
        TORCH_CHECK(value.q_tiles == 0 || value.q_tiles == 2 || value.q_tiles == 4,
                    "PA_BHSD_Q_TILES must be 0, 2 or 4; received ", value.q_tiles);
        return value;
    }();
    return tuning;
}

void check_bhsd_partition_workspace(int batch, int num_heads, int head_dim_v, int partitions)
{
    // paged_attention.cu allocates temp_out_size + 10000000 bytes: at least
    // 200000000 + 10000000 (gfx938 allocates more). Keep this conservative
    // bound without changing the existing allocation or cache lifetime.
    constexpr int64_t workspace_capacity_bytes = 210000000;
    constexpr int64_t scalar_bytes = sizeof(uint16_t);
    constexpr int64_t max_begin = static_cast<int64_t>(max_tmp_offset) * scalar_bytes;
    constexpr int64_t output_begin = static_cast<int64_t>(out_tmp_offset) * scalar_bytes;
    const int64_t query_rows = static_cast<int64_t>(batch) * num_heads;
    const int64_t bytes_per_query_row = static_cast<int64_t>(partitions) * head_dim_v * scalar_bytes;
    TORCH_CHECK(query_rows >= 0 && bytes_per_query_row > 0 &&
                query_rows <= (std::numeric_limits<int64_t>::max() - output_begin) / bytes_per_query_row,
                "PA_BHSD_PARTITIONS workspace size overflows int64; reduce PA_BHSD_PARTITIONS");
    const int64_t partition_rows = query_rows * partitions;
    const int64_t stats_bytes = partition_rows * static_cast<int64_t>(sizeof(float));
    const int64_t max_end = max_begin + stats_bytes;
    const int64_t output_end = output_begin + query_rows * bytes_per_query_row;
    TORCH_CHECK(stats_bytes <= max_begin && max_end <= output_begin && output_end <= workspace_capacity_bytes,
                "PA_BHSD_PARTITIONS=", partitions, " requires sum bytes=", stats_bytes,
                " (limit ", max_begin, "), max end=", max_end, " (limit ", output_begin,
                "), output end=", output_end, " (workspace limit ", workspace_capacity_bytes,
                "); reduce PA_BHSD_PARTITIONS");
}
} // namespace

// The uploaded BHSD implementation also passes a first-page optimization hint.
// This project's common header has the original five-argument helper; forwarding
// preserves its computation (the first-page accumulator is initialized to zero).
// The direct E5M2 path below has its own softmax and does not call this overload.
using ::softmax_rescale_o;
template <int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE>
__device__ __forceinline__ void softmax_rescale_o(
    float4_t scores[Mloop][BLOCK_SIZE / 16], float scores_max[Mloop],
    float scores_sum[Mloop], float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
    float softmax_scale_log2, bool /*has_previous_page*/)
{
    softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(
        scores, scores_max, scores_sum, acc_o, softmax_scale_log2);
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, typename cache_t, int Mloop, int HEAD_SIZE, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, bool is_query_fp16, int STAGES, bool is_kv_fp8, bool is_e4m3, int Q_LDS_PAD = 0, int NUM_Q_TILES = 1>
__forceinline__ __device__ void qk_gemm_func_single_buffer_bhsd(const cache_t *k_ptr,
                                                  const scalar_t *smem_q,
                                                  int seqlen_q_per_kv_head,
                                                  float4_t s_reg[Mloop][BLOCK_SIZE / 16],
                                                  int block_idx,
                                                  int valid_seqlen_kv
                                                )
{
    // 每个 warp 处理一个 page (只支持 BLOCK_SIZE = 64): s_reg 每个线程持有一行 4 个连续元素 (s_reg[m][0..3])[j];
    // k_cache 每个 warp 一次加载 32 维 x 64 token, lane 0-3 读同一 token 的连续 32 字节, 再经 permute 转成 mmac 布局
    using k_stage_t = std::conditional_t<is_kv_fp8, uint8x8_t, scalarx8_t>;
    k_stage_t k_reg[(NUM_KV_TOKENS_PER_LOAD / 16)];
    int lane_id = threadIdx.x % WARP_SIZE;
    // 0-3, 4-7 线程连续 global->register, 下方通过 permute 转成 mmac 需要的布局
    int k_g2r_row = lane_id & 3;        // 0, 1, 2, 3
    int k_g2r_col = lane_id >> 2;       // 0-15
    int q_s2r_row = lane_id & 15;
    int q_s2r_col = lane_id >> 4;

    #pragma unroll
    for(int k_tile = 0;k_tile < (HEAD_SIZE / 32);k_tile++){

        const cache_t *k_tile_ptr = k_ptr + k_tile * 32;
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
            const cache_t *src = k_tile_ptr + k_g2r_row * 8 + (k_g2r_col * (NUM_KV_TOKENS_PER_LOAD / 16) + i) * HEAD_SIZE;
#else
            const cache_t *src = k_tile_ptr + k_g2r_row * 8 + (k_g2r_col * (NUM_KV_TOKENS_PER_LOAD / 16) + i) * HEAD_SIZE * (gridDim.x / NUM_Q_TILES);
#endif
            if constexpr (is_kv_fp8){
                k_reg[i] = *((uint8x8_t *)(src));
            }else{
                k_reg[i] = *((scalarx8_t *)(src));
            }
        }
        scalarx8_t q_fragment[Mloop];
        for(int m = 0;m < Mloop;m++){
            int q_row = m * 16 + q_s2r_row;
            if(q_row < seqlen_q_per_kv_head){
                q_fragment[m] = *reinterpret_cast<const scalarx8_t *>(
                    smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + (k_tile) * 32 + q_s2r_col * 8);
            }else{
                q_fragment[m] = {0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
        inline_wait_vmcnt(0);

        // k E5M2->b16
        scalarx8_t k_fragment[NUM_KV_TOKENS_PER_LOAD / 16];
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            if constexpr(is_kv_fp8){
                k_fragment[i] = decode_fp8x8_to_b16x8<scalarx8_t, scalar_t, is_e4m3>(
                    k_reg[i]);
            } else {
                k_fragment[i] = k_reg[i];
            }
        }

        {
            // permute
            for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
                int *temp_ptr = reinterpret_cast<int *>(&(k_fragment[i]));
                for(int j = 0;j < 4;j++){
                    int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                    int res = __builtin_amdgcn_ds_permute(index, temp_ptr[j]);
                    temp_ptr[j] = res;
                }
            }
        }

        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
                const scalarx4_t *q_v4 = reinterpret_cast<const scalarx4_t *>(&(q_fragment[m]));
                const scalarx4_t *k_v4 = reinterpret_cast<const scalarx4_t *>(&(k_fragment[i]));
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[0], k_v4[0], s_reg[m][i]);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[1], k_v4[1], s_reg[m][i]);
            }
        }
    }

    if ((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv) {
        for(int m = 0; m < Mloop; ++m){
            for(int i = 0; i < (NUM_KV_TOKENS_PER_LOAD / 16); ++i){
                const int token_idx = block_idx * BLOCK_SIZE + i + q_s2r_col * 4;
                for(int j = 0; j < 4; ++j){
                    if((token_idx + j * 16) >= valid_seqlen_kv){
                        (s_reg[m][i])[j] = -INFINITY;
                    }
                }
            }
        }
    }
}

template <typename scalarx4_t, typename scalar_t, typename cache_t, int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, bool is_query_fp16, int STAGES,
          bool is_kv_fp8=false, bool is_e4m3=false, int NUM_Q_TILES = 1>
__device__ __forceinline__  void pv_gemm_func_single_buffer_bhsd(const cache_t *v_ptr,
                                                    scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                    int block_idx,
                                                    int valid_seqlen_kv
                                                  )
{
    // 每个 warp 处理一个 page, v_cache 每次加载 64 token x 32 维 (v_reg[4][2]); p 每个线程持有 4 个连续元素, 对应 v 同一列的 4 个
    using v_stage_t = std::conditional_t<is_kv_fp8, uint8x4_t, scalarx4_t>;
    v_stage_t v_reg[NUM_KV_TOKENS_PER_LOAD / 16][2];
    int v_tile = 0;
    int lane_id = threadIdx.x % WARP_SIZE;
    // 0-3, 4-7 线程连续 global->register, 下方通过bpermute 转成 mmac 需要的布局
    int v_g2r_row = lane_id & 3;       // 0, 1, 2, 3
    int v_g2r_col = lane_id >> 2;       // 0-15

    // v_cache [num_blocks, block_size, num_kv_heads, head_dim_v]

    // Keep accumulator and ping-pong buffer indices constant after inlining.
    #pragma unroll
    for(v_tile = 0; v_tile < (HEAD_SIZE_V / 32);v_tile++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
        const cache_t *v_tile_ptr = v_ptr + v_tile * 32 * BLOCK_SIZE;
        // 取下一份数据
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const cache_t *cur_v_ptr =v_tile_ptr + (j * 16 + v_g2r_col) * BLOCK_SIZE + i * 16 + v_g2r_row * 4;
                v_reg[i][j] = *reinterpret_cast<const v_stage_t *>(cur_v_ptr);
#else
        const cache_t *v_tile_ptr = v_ptr + v_tile * 32;
        // 取下一份数据
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const cache_t *cur_v_ptr =v_tile_ptr + (i * 16 * HEAD_SIZE_V + v_g2r_row * 4 * HEAD_SIZE_V) * (gridDim.x / NUM_Q_TILES) + (j * 16 + v_g2r_col);
                if constexpr(is_kv_fp8){
                    (v_reg[i][j])[0] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 0);
                    (v_reg[i][j])[1] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 1 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                    (v_reg[i][j])[2] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 2 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                    (v_reg[i][j])[3] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 3 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                } else {
                    (v_reg[i][j])[0] = *reinterpret_cast<const scalar_t *>(cur_v_ptr + 0);
                    (v_reg[i][j])[1] = *reinterpret_cast<const scalar_t *>(cur_v_ptr + 1 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                    (v_reg[i][j])[2] = *reinterpret_cast<const scalar_t *>(cur_v_ptr + 2 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                    (v_reg[i][j])[3] = *reinterpret_cast<const scalar_t *>(cur_v_ptr + 3 * HEAD_SIZE_V * (gridDim.x / NUM_Q_TILES));
                }
#endif
            }
        }

        inline_wait_vmcnt(0);
        // 已加载 v 的 64 行, 对应 P 的 64 列 (BLOCK_SIZE = 64); 每个 V 片段解码、permute 一次后供各 Q 行块复用
        for(int j = 0;j < (NUM_KV_TOKENS_PER_LOAD / 16);j++){
            scalarx4_t v_fragment0;
            scalarx4_t v_fragment1;
            if constexpr(is_kv_fp8){
                v_fragment0 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][0]);
                v_fragment1 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][1]);
            } else {
                v_fragment0 = v_reg[j][0];
                v_fragment1 = v_reg[j][1];
            }

            // v 边界处理
            int token_idx = block_idx * BLOCK_SIZE + j * 16 + v_g2r_row * 4;
            for(int i = 0;i < 4;i++){
                if((token_idx + i) >= valid_seqlen_kv){
                    v_fragment0[i] = 0;
                    v_fragment1[i] = 0;
                }
            }

            {
                // permute
                int *temp_ptr0 = reinterpret_cast<int *>(&(v_fragment0));
                int *temp_ptr1 = reinterpret_cast<int *>(&(v_fragment1));
                for(int j = 0;j < 2;j++){
                    int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                    int res0 = __builtin_amdgcn_ds_permute(index, temp_ptr0[j]);
                    int res1 = __builtin_amdgcn_ds_permute(index, temp_ptr1[j]);
                    temp_ptr0[j] = res0;
                    temp_ptr1[j] = res1;
                }
            }

            #pragma unroll
            for(int m = 0;m < Mloop;m++){
                // p 在需要的时候再构造, 减少 temp_p_reg 的生存时间
                scalarx4_t temp_p_reg = {(p_reg[m][0])[j], (p_reg[m][1])[j], (p_reg[m][2])[j], (p_reg[m][3])[j]};
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment0, acc_o[m][(v_tile) * 2 + 0]);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment1, acc_o[m][(v_tile) * 2 + 1]);
            }
        }
    }

}

// Coalesced FP8 V loads for BF16 / E5M2 Q tiles. Each lane loads
// four adjacent channels; bpermute gathers four tokens of one channel into
// the MFMA operand. A 64x32 V tile uses eight dword loads per lane instead
// of thirty-two byte loads, without a shared-memory transpose.
template <typename scalarx4_t, typename scalar_t, typename cache_t, int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, bool is_query_fp16, int STAGES,
          bool is_kv_fp8=false, bool is_e4m3=false, int NUM_Q_TILES = 1>
__device__ __forceinline__  void pv_gemm_func_single_buffer_fp8_bhsd(
                                                    cache_t *smem_v,
                                                    const cache_t *v_ptr,
                                                    scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                    int block_idx,
                                                    int valid_seqlen_kv,
                                                    int num_kv_heads
                                                  )
{
    // 每个 warp 处理一个 block_page
    // v_cache 每个 warp 一次加载 16x128 (对应两次 matrix_load_b8 指令, 每个 matrix_load_b8 加载8x128行主序的数据)
    // p 矩阵每个线程持有 4 个连续的元素, 对应的 v 每个线程 持有同一列的4个连续数据

    // v_cache [num_blocks, num_kv_heads, block_size, head_dim_v]
    #pragma unroll
    for(int v_col_tile = 0;v_col_tile < (HEAD_SIZE_V / 128);v_col_tile++){
        for(int v_row_tile = 0;v_row_tile < (BLOCK_SIZE / 16);v_row_tile++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
            const cache_t *v_tile_ptr0 = v_ptr + v_col_tile * 128 + v_row_tile * 16 * HEAD_SIZE_V;
            const cache_t *v_tile_ptr1 = v_tile_ptr0 + 8 * HEAD_SIZE_V;

            uintx4 vdata; // 指令格式要求, 无实际作用
            uint64_t global_addr0 = reinterpret_cast<uint64_t>(v_tile_ptr0);
            uint64_t global_addr1 = reinterpret_cast<uint64_t>(v_tile_ptr1);
            uint32_t global_stride = HEAD_SIZE_V;
#else
            const cache_t *v_tile_ptr0 = v_ptr + v_col_tile * 128 + v_row_tile * 16 * HEAD_SIZE_V * num_kv_heads;
            const cache_t *v_tile_ptr1 = v_tile_ptr0 + 8 * HEAD_SIZE_V * num_kv_heads;

            uintx4 vdata; // 指令格式要求, 无实际作用
            uint64_t global_addr0 = reinterpret_cast<uint64_t>(v_tile_ptr0);
            uint64_t global_addr1 = reinterpret_cast<uint64_t>(v_tile_ptr1);
            uint32_t global_stride = HEAD_SIZE_V * num_kv_heads;
#endif
            const int lds_base0 = reinterpret_cast<size_t>(smem_v);
            const int lds_base1 = reinterpret_cast<size_t>(smem_v) + 1024;

            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "s_mov_b32 m0, %3\n\t"
                "s_nop 0\n\t"
                "matrix_load_b8 %0, %1, %2 t lds\n\t"
                : "=v"(vdata)
                : "s"(global_addr0),
                "s"(global_stride),
                "s"(lds_base0)
                : "memory");
            asm volatile(
                "s_mov_b32 m0, %3\n\t"
                "s_nop 0\n\t"
                "matrix_load_b8 %0, %1, %2 t lds\n\t"
                : "=v"(vdata)
                : "s"(global_addr1),
                "s"(global_stride),
                "s"(lds_base1)
                : "memory");

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);

            // [1][8], [1] 代表加载 16 行, [8] 代表加载 128 列
            uint8x4_t v_reg[1][8];
            int lane_id = threadIdx.x % WARP_SIZE;
            // s2r 线程排布
            //  0   1   2...
            //  16  17  18...
            //  32  33  34...
            //  48  49  50...
            int v_s2r_row = lane_id >> 4;       // 0, 1, 2, 3
            int v_s2r_col = lane_id & 15;       // 0-15

            #pragma unroll
            for(int i = 0;i < 8;i++){
                const int row = v_s2r_row * 4;
                const int col = i * 16 + v_s2r_col;
                (v_reg[0][i])[0] = *(smem_v + (row + 0) * 128 + col);
                (v_reg[0][i])[1] = *(smem_v + (row + 1) * 128 + col);
                (v_reg[0][i])[2] = *(smem_v + (row + 2) * 128 + col);
                (v_reg[0][i])[3] = *(smem_v + (row + 3) * 128 + col);
            }

            // v 边界处理
            if((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv){
                int token_idx = block_idx * BLOCK_SIZE + v_row_tile * 16 + v_s2r_row * 4;
                for(int i = 0;i < 4;i++){
                    if((token_idx + i) >= valid_seqlen_kv){
                        (v_reg[0][0])[i] = 0;
                        (v_reg[0][1])[i] = 0;
                        (v_reg[0][2])[i] = 0;
                        (v_reg[0][3])[i] = 0;
                        (v_reg[0][4])[i] = 0;
                        (v_reg[0][5])[i] = 0;
                        (v_reg[0][6])[i] = 0;
                        (v_reg[0][7])[i] = 0;
                    }
                }
            }

            #pragma unroll
            for(int i = 0;i < 8;i++){
                scalarx4_t v_fragment;
                v_fragment = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[0][i]);
                for(int m = 0;m < Mloop;m++){
                    scalarx4_t temp_p_reg = {(p_reg[m][0])[v_row_tile], (p_reg[m][1])[v_row_tile], (p_reg[m][2])[v_row_tile], (p_reg[m][3])[v_row_tile]};
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment, acc_o[m][v_col_tile * 8 + i]);
                }
            }
        }
    }
}

// FP8 KV 直接加载路径 (HEAD_SIZE/HEAD_SIZE_V 为 64/128/256)
//  K: QK 的 B 操作数中 lane (n = lane & 15, g = lane >> 4) 需要 token 4n+i 的连续通道,
//     每个 lane 直接加载该 token 在每个 64 通道分组中的第 g 个 16 字节, 不需要 ds_permute。
//     一个 64 通道分组 (4 token x 16 字节) 称为一个 K chunk。
//  V: lane n 持有 CH = HEAD_SIZE_V / 16 个连续通道, 每 16 个 token 的 tile 中持有 4 个连续 token,
//     正好是 PV B 操作数的 token 顺序, 不需要经过 LDS 转置。acc_o 第 q 个 tile 的第 n 列对应
//     通道 CH * n + q, epilogue 写 LDS 时还原通道顺序。
//  E5M2 是截断的 FP16, 字节放到 FP16 高 8 位即为精确值, 因此 PV 使用 FP16 mmac (延迟重缩放时 P 不超过 256)。
//  K chunk / V tile 分别用 K_AHEAD / V_AHEAD 个寄存器槽位做环形预取: 一个槽位消费后立即发射
//  后续数据, 超出当前 page 时预取同一 warp 的下一个 page, 隐藏 page 之间的访存延迟。
// ---------------------------------------------------------------------------
template <int HEAD_SIZE, int HEAD_SIZE_V, int Mloop>
struct DirectFp8Config {
    static constexpr bool enabled = (HEAD_SIZE == 64 || HEAD_SIZE == 128 || HEAD_SIZE == 192 || HEAD_SIZE == 256) &&
                                    (HEAD_SIZE_V == 64 || HEAD_SIZE_V == 128 || HEAD_SIZE_V == 192 || HEAD_SIZE_V == 256);
    static constexpr int K_CHUNKS = HEAD_SIZE / 64;
    static constexpr int V_CH = HEAD_SIZE_V / 16;
    static constexpr int V_NW = V_CH / 4;
    // 寄存器预算: acc_o 占 Mloop * HEAD_SIZE_V / 4 个 VGPR, 规模越大预取槽位越少
    static constexpr int ACC_VGPRS = Mloop * HEAD_SIZE_V / 4;
    static constexpr int K_AHEAD = ACC_VGPRS >= 128 ? 1 : (ACC_VGPRS >= 64 ? (K_CHUNKS % 2 == 0 ? 2 : 1) : K_CHUNKS);
    static constexpr int V_AHEAD = ACC_VGPRS >= 128 ? 1 : (ACC_VGPRS >= 64 ? 2 : 4);
    static_assert(!enabled || (K_CHUNKS % K_AHEAD == 0 && 4 % V_AHEAD == 0), "ring slots must repeat per page");
    // 每个 CU 的线程块数: 每页 MMA 量 (Mloop * HEAD_SIZE) 较大时两个线程块共驻, 使 MMA 与另一线程块的访存/解码重叠
    static constexpr int CTAS_PER_CU = (Mloop * HEAD_SIZE / 128) < 1 ? 1 : ((Mloop * HEAD_SIZE / 128) > 2 ? 2 : (Mloop * HEAD_SIZE / 128));
};

// 直接加载路径的工作划分 (主 kernel 与合并 kernel 共用, 只依赖 page 数与 partition 上限):
//  page 数不足以让每个 warp 分到独立 page 时 (pages < 4 * P) 使用页内切分模式: 线程块的 4 个 warp
//  共同处理同一 page, 每个线程块至少 1 个 page, 并行线程块数最多为常规模式的 4 倍;
//  否则每个 warp 处理独立 page, 先确定每个 warp 的 page 数, 再只保留达到该 page 数所需的 partition,
//  关键路径不变时减少线程块、partial 输出与合并开销。
//  balanced_seq_len_tile = CTAS_PER_CU * CU 数 / (batch * kv_heads * q_tiles), 因此
//  slice_cap = balanced_seq_len_tile / CTAS_PER_CU 个 partition 时线程块数不超过 CU 数。
//  页内切分只在线程块数不超过 CU 数且每个线程块不超过 2 个 page 时使用: 更多 page 时
//  常规模式的每 warp 独立 page 流水更高效。
template <int CTAS_PER_CU, bool ONE_PAGE_SLICE = false>
__device__ __forceinline__ bool use_direct_slice_mode(int valid_num_block_pages, int balanced_seq_len_tile)
{
    const int slice_cap = MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU);
#if defined(__gfx92a__)
    if constexpr (ONE_PAGE_SLICE) return valid_num_block_pages <= slice_cap;
#endif
    return valid_num_block_pages <= 2 * slice_cap;
}

//  LONG_SINGLE_STREAM: 每个 CU 两路 KV 读取流且每页计算量不大时, 若 partition 数减半 (每个 CU 一路流) 后每个 warp
//  仍有 4 个以上 page, 只使用一半 partition: page 间预取已能隐藏访存延迟, 第二路流只增加 DRAM 上的并发流数。
//  主 kernel 与合并 kernel 使用相同的模板参数, 得到相同的 partition 数。
template <int CTAS_PER_CU, bool LONG_SINGLE_STREAM = false, bool RESIDENT_CAP = false, bool ONE_PAGE_SLICE = false>
__device__ __forceinline__ int get_direct_num_partitions(int valid_num_block_pages, int balanced_seq_len_tile)
{
    constexpr int MAIN_KERNEL_NUM_WARPS = 4;
    if (use_direct_slice_mode<CTAS_PER_CU, ONE_PAGE_SLICE>(valid_num_block_pages, balanced_seq_len_tile)) {
        return MAX_T(1, MIN_T(MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU), valid_num_block_pages));
    }
    // Derive the resident limit from the ORIGINAL budget, before LONG_SINGLE_STREAM.
    const int resident_seq_len_tile = MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU);
    if constexpr (LONG_SINGLE_STREAM) {
        const int single_stream_tile = MAX_T(1, balanced_seq_len_tile / 2);
        if (DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * single_stream_tile) > 4) {
            balanced_seq_len_tile = single_stream_tile;
        }
    }
#if defined(__gfx92a__)
    if constexpr (RESIDENT_CAP) {
        balanced_seq_len_tile = MIN_T(balanced_seq_len_tile, resident_seq_len_tile);
    }
#endif
    const int max_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS));
    if (max_seq_len_tile <= balanced_seq_len_tile) {
        return max_seq_len_tile;
    }
    const int pages_per_warp = DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * balanced_seq_len_tile);
    return DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * pages_per_warp);
}

template <int NW> struct dword_vec;
template <> struct dword_vec<1> { using type = uint32_t; };
template <> struct dword_vec<2> { using type = uint32_t __attribute__((ext_vector_type(2))); };
template <> struct dword_vec<4> { using type = uint32_t __attribute__((ext_vector_type(4))); };

using half4_v = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;

// K/V 使用 buffer load: page 基址放在 SGPR 资源描述符中, 每个 lane 只保留一个与 page 无关的 32 位偏移
using buffer_rsrc_t = int __attribute__((ext_vector_type(4)));
__device__ typename dword_vec<4>::type raw_buffer_load_b128(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ typename dword_vec<2>::type raw_buffer_load_b64(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ uint32_t raw_buffer_load_b32(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

template <int NW>
__device__ __forceinline__ typename dword_vec<NW>::type raw_buffer_load_dwords(buffer_rsrc_t rsrc, int voffset, int soffset)
{
    if constexpr (NW == 4) return raw_buffer_load_b128(rsrc, voffset, soffset, 0);
    else if constexpr (NW == 2) return raw_buffer_load_b64(rsrc, voffset, soffset, 0);
    else return raw_buffer_load_b32(rsrc, voffset, soffset, 0);
}

__device__ __forceinline__ buffer_rsrc_t make_page_rsrc(const uint8_t *page)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(page);
    buffer_rsrc_t rsrc;
    rsrc[0] = static_cast<int>(addr & 0xffffffffu);
    rsrc[1] = static_cast<int>(addr >> 32);   // stride = 0
    rsrc[2] = static_cast<int>(0x80000000u);  // num_records
    rsrc[3] = 0x00020000;                      // 32 位数据格式
    return rsrc;
}

// lane 偏移: token 4 * (lane & 15) 的第 lane >> 4 个 16 字节
__device__ __forceinline__ int k_fp8_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * 4 * token_stride + (lane_id >> 4) * 16;
}

// K 槽位 slot 的第 j 次加载对应的 token 组 i (token 4 * (lane & 15) + i) 与 64 字节 chunk:
//  配对槽位覆盖 2 个 token 组 x 相邻 2 个 chunk, 同一条 128 字节 cache line 的两半由相邻指令读取
//  (两半间隔数条指令读取会降低 DRAM 效率)。HEAD_SIZE == 128 总是配对; HEAD_SIZE == 256 在 Mloop == 1 时配对
//  (每个 S 累加器的 mmac 顺序不变), Mloop >= 2 保留未配对的 Q 半片段路径以控制寄存器;
//  其他情况槽位为 1 个 chunk x 4 个 token 组。每页的槽位数均为 K_CHUNKS。
template <int HEAD_SIZE, int Mloop = 1>
struct KSlotMap {
    static constexpr bool PAIRED = HEAD_SIZE == 128 || (HEAD_SIZE == 256 && Mloop == 1);
    static constexpr __device__ __forceinline__ int token(int slot, int j) { return PAIRED ? 2 * (slot & 1) + (j >> 1) : j; }
    static constexpr __device__ __forceinline__ int chunk(int slot, int j) { return PAIRED ? 2 * (slot >> 1) + (j & 1) : slot; }
};

template <int HEAD_SIZE, int Mloop>
__device__ __forceinline__ void load_k_fp8_chunk(buffer_rsrc_t k_page, int lane_offset, int token_stride, int slot,
                                                 uint32_t raw[4][4])
{
    using Map = KSlotMap<HEAD_SIZE, Mloop>;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const typename dword_vec<4>::type value =
            raw_buffer_load_b128(k_page, lane_offset + Map::chunk(slot, j) * 64, Map::token(slot, j) * token_stride, 0);
        #pragma unroll
        for (int w = 0; w < 4; ++w) raw[j][w] = value[w];
    }
}

// lane 偏移: token 4 * (lane >> 4) 的通道 CH * (lane & 15) 起
template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_fp8_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id >> 4) * 4 * token_stride + (lane_id & 15) * (HEAD_SIZE_V == 192 ? 4 : HEAD_SIZE_V / 16);
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_v_fp8_tile(buffer_rsrc_t v_page, int lane_offset, int token_stride, int tile,
                                                uint32_t raw[4][HEAD_SIZE_V / 64])
{
    constexpr int NW = HEAD_SIZE_V / 64;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (HEAD_SIZE_V == 192) {
            // D192: 3 段各 4 个通道 (偏移 64s)
            #pragma unroll
            for (int s = 0; s < 3; ++s) raw[k][s] = raw_buffer_load_b32(v_page, lane_offset + 64 * s, (tile * 16 + k) * token_stride, 0);
            continue;
        } else {
        const typename dword_vec<NW>::type value = raw_buffer_load_dwords<NW>(v_page, lane_offset, (tile * 16 + k) * token_stride);
        if constexpr (NW == 1) {
            raw[k][0] = value;
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) raw[k][w] = value[w];
        }
        }
    }
}

// 4 个 E5M2 解码为 4 个 b16: FP16 只需移字节; BF16 由精确的 FP16 经 FP32 截取高 16 位 (同样精确)
template <typename scalarx4_t, bool is_query_fp16>
__device__ __forceinline__ scalarx4_t decode_e5m2x4_b16(uint32_t raw)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    using half2_v = _Float16 __attribute__((ext_vector_type(2)));
    const uint32_t h01 = __builtin_amdgcn_perm(raw, raw, 0x010C000Cu);
    const uint32_t h23 = __builtin_amdgcn_perm(raw, raw, 0x030C020Cu);
    if constexpr (is_query_fp16) {
        return __builtin_bit_cast(scalarx4_t, (u32x2){h01, h23});
    } else {
#if defined(__gfx92a__)
        using f32x2 = float __attribute__((ext_vector_type(2)));
        const f32x2 a = __builtin_amdgcn_cvt_pk_f32_bf8(raw, false);
        const f32x2 b = __builtin_amdgcn_cvt_pk_f32_bf8(raw, true);
        // Materialize scalar floats before bit_cast to avoid the clang18
        // ext-vector element miscompile exposed by mixed E5M2 bytes.
        const float f0 = a[0], f1 = a[1], f2 = b[0], f3 = b[1];
        const uint32_t p01 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, f1),
                                                   __builtin_bit_cast(uint32_t, f0), 0x07060302u);
        const uint32_t p23 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, f3),
                                                   __builtin_bit_cast(uint32_t, f2), 0x07060302u);
#else
        const half2_v a = __builtin_bit_cast(half2_v, h01);
        const half2_v b = __builtin_bit_cast(half2_v, h23);
        const uint32_t p01 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, (float)a[1]),
                                                   __builtin_bit_cast(uint32_t, (float)a[0]), 0x07060302u);
        const uint32_t p23 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, (float)b[1]),
                                                   __builtin_bit_cast(uint32_t, (float)b[0]), 0x07060302u);
#endif
        return __builtin_bit_cast(scalarx4_t, (u32x2){p01, p23});
    }
}

// 直接加载路径的 online softmax:
//  每个 lane 只累计自己持有的 S 元素的行和 (行内 4 个 lane 组的部分和在 page 循环结束后再归约);
//  行 max 的增量不超过 2^8 (log2 域) 时沿用旧的参考 max, 不缩放 acc_o 与行和, P <= 2^8 仍在 FP16 范围内,
//  累加为 FP32, 在实数运算上等价于每页缩放, 浮点舍入可能不同。参考 max 只增不减, 全部被掩码的行保持 -FLT_MAX, 其 P 为 0。
constexpr float DIRECT_SOFTMAX_RESCALE_LOG2 = 8.0f;

// Preserve the subtraction before multiplication. Under fast-math the compiler
// may turn (a-b)*scale into a fused multiply-add; equal -FLT_MAX sentinels
// then leave a large rounding residual and exp2 can overflow.
__device__ __forceinline__ float direct_exp_difference(float a, float b, float scale)
{
    float difference;
    asm("v_sub_f32 %0, %1, %2" : "=v"(difference) : "v"(a), "v"(b));
    return __builtin_amdgcn_exp2f(difference * scale);
}

// FAST (行切分主循环): 行最大值用 v_max_f32, exp2 参数用 fma
template <int Mloop, int HEAD_SIZE_V, int NT, bool FAST = false>
__device__ __forceinline__ void direct_softmax(float4_t s_reg[Mloop][NT], float scores_max[Mloop], float scores_sum[Mloop],
                                               float4_t acc_o[Mloop][HEAD_SIZE_V / 16], float scale_log2, bool first_page)
{
    auto vmax = [](float a, float b) {
        if constexpr (FAST) return __builtin_fmaxf(a, b);
        else return MAX_T(a, b);
    };
    float row_max[Mloop];
    bool rescale = false;
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        float4_t partial = s_reg[m][0];
        #pragma unroll
        for (int i = 1; i < NT; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) partial[j] = vmax(partial[j], (s_reg[m][i])[j]);
        }
        float value = vmax(vmax(partial[0], partial[1]), vmax(partial[2], partial[3]));
        value = vmax(value, __shfl_xor_tmp(value, 32));
        value = vmax(value, __shfl_swap16(value));
        row_max[m] = vmax(value, scores_max[m]);
        rescale |= (row_max[m] - scores_max[m]) * scale_log2 > DIRECT_SOFTMAX_RESCALE_LOG2;
    }
    if (__any(rescale)) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const float factor = direct_exp_difference(scores_max[m], row_max[m], scale_log2);
            scores_sum[m] *= factor;
            if (!first_page) {
                #pragma unroll
                for (int i = 0; i < HEAD_SIZE_V / 16; ++i) acc_o[m][i] *= factor;
            }
            scores_max[m] = row_max[m];
        }
    }
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const float max_scaled = scores_max[m] * scale_log2;
        float4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < NT; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (FAST) (s_reg[m][i])[j] = __builtin_amdgcn_exp2f(__builtin_fmaf((s_reg[m][i])[j], scale_log2, -max_scaled));
                else (s_reg[m][i])[j] = __builtin_amdgcn_exp2f((s_reg[m][i])[j] * scale_log2 - max_scaled);
            }
            sum += s_reg[m][i];
        }
        scores_sum[m] += (sum[0] + sum[1]) + (sum[2] + sum[3]);
    }
}

// page 循环结束后归约行和 (同一行由 lane n, n + 16, n + 32, n + 48 持有)
template <int Mloop>
__device__ __forceinline__ void direct_softmax_finish(float scores_sum[Mloop])
{
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        scores_sum[m] += __shfl_xor_tmp(scores_sum[m], 32);
        scores_sum[m] += __shfl_swap16(scores_sum[m]);
    }
}

// 读取一个 64 维 chunk 的 Q 分片
template <typename scalarx8_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD>
__device__ __forceinline__ void load_q_fp8_chunk(const scalar_t *smem_q, int chunk, int seqlen_q_per_kv_head,
                                                 scalarx8_t q_fragment[Mloop][2])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int q_row = m * 16 + q_s2r_row;
        const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + chunk * 64 + q_s2r_col * 16;
        if (q_row < seqlen_q_per_kv_head) {
            q_fragment[m][0] = *reinterpret_cast<const scalarx8_t *>(q_src);
            q_fragment[m][1] = *reinterpret_cast<const scalarx8_t *>(q_src + 8);
        } else {
            q_fragment[m][0] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            q_fragment[m][1] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
}

// 槽位 slot 的 QK: 按 chunk 分组, 组内各 token 组的 mmac 写不同累加器。
// 配对槽位的 Q 分片由调用者按 chunk 对读取 (相邻两个槽位共用), 每页 Q 的 LDS 读取量与未配对时相同
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD, bool is_query_fp16>
__device__ __forceinline__ void qk_fp8_chunk(const uint32_t raw[4][4], const scalar_t *smem_q, int slot,
                                             int seqlen_q_per_kv_head, float4_t s_reg[Mloop][4],
                                             const scalarx8_t (*q_pair)[Mloop][2] = nullptr)
{
    using Map = KSlotMap<HEAD_SIZE, Mloop>;
    constexpr int GROUPS = Map::PAIRED ? 2 : 1;      // 槽位内的 chunk 数
    constexpr int PER_GROUP = 4 / GROUPS;            // 每个 chunk 的 token 组数
    // Non-paired slots consume Q in two 16-byte halves. Keep one half live
    // per query row while preserving the original sub/t/m MMAC order.
    // D128 keeps its paired-slot Q reuse below unchanged.
    if constexpr (!Map::PAIRED) {
        const int lane_id = threadIdx.x % WARP_SIZE;
        const int q_s2r_row = lane_id & 15;
        const int q_s2r_col = lane_id >> 4;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            scalarx8_t q_half[Mloop];
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                const int q_row = m * 16 + q_s2r_row;
                const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD)
                    + Map::chunk(slot, 0) * 64 + q_s2r_col * 16 + half * 8;
                if (q_row < seqlen_q_per_kv_head) {
                    q_half[m] = *reinterpret_cast<const scalarx8_t *>(q_src);
                } else {
                    q_half[m] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const scalarx4_t k_fragment = decode_e5m2x4_b16<scalarx4_t, is_query_fp16>(raw[t][half * 2 + sub]);
                    #pragma unroll
                    for (int m = 0; m < Mloop; ++m) {
                        const scalarx8_t q8 = q_half[m];
                        const scalarx4_t q_v4 = sub == 0 ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)
                                                        : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);
                        builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, s_reg[m][t]);
                    }
                }
            }
        }
        return;
    }
    #pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
        scalarx8_t q_fragment[Mloop][2];
        if constexpr (Map::PAIRED) {
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                q_fragment[m][0] = q_pair[g][m][0];
                q_fragment[m][1] = q_pair[g][m][1];
            }
        } else {
            load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, Map::chunk(slot, g), seqlen_q_per_kv_head, q_fragment);
        }
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            #pragma unroll
            for (int t = 0; t < PER_GROUP; ++t) {
                const int j = Map::PAIRED ? 2 * t + g : t;
                const scalarx4_t k_fragment = decode_e5m2x4_b16<scalarx4_t, is_query_fp16>(raw[j][sub]);
                #pragma unroll
                for (int m = 0; m < Mloop; ++m) {
                    const scalarx8_t q8 = q_fragment[m][sub / 2];
                    const scalarx4_t q_v4 = (sub % 2 == 0) ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)
                                                           : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, s_reg[m][Map::token(slot, j)]);
                }
            }
        }
    }
}

template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void pv_fp8_tile_f16(const uint32_t raw[4][HEAD_SIZE_V / 64],
                                                const half4_v p[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16])
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    constexpr int CH = HEAD_SIZE_V / 16;
    #pragma unroll
    for (int q = 0; q < CH; ++q) {
        const int w = q / 4;
        constexpr uint32_t zero = 0x0C;
        const uint32_t b = q % 4;
        // 结果字节为 {0, token0, 0, token1}, 即两个 FP16
        const uint32_t sel = zero | (b << 8) | (zero << 16) | ((4 + b) << 24);
        const u32x2 packed = {__builtin_amdgcn_perm(raw[1][w], raw[0][w], sel),
                              __builtin_amdgcn_perm(raw[3][w], raw[2][w], sel)};
        const half4_v v_fragment = __builtin_bit_cast(half4_v, packed);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            builtin_amdgcn_mmac<half4_v, true>(p[m], v_fragment, acc_o[m][q]);
        }
    }
}

// 每个 lane 写出 16 个 float, 将 acc_o 的交织通道还原到 smem_reduce 的 64 列分组
template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void store_direct_v_acc_group(float *row_ptr, const float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                         int tile_row, int tile_col, int acc_col)
{
    if constexpr (HEAD_SIZE_V == 192) {
        // 分段通道顺序: 第 tile_col 段 (64 通道) 为 acc_o[m][4 tile_col + q], 第 j 个元素为通道 64 tile_col + 4 (4j + acc_col) + q
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const float4_t value = {(acc_o[tile_row][4 * tile_col + 0])[jj], (acc_o[tile_row][4 * tile_col + 1])[jj],
                                    (acc_o[tile_row][4 * tile_col + 2])[jj], (acc_o[tile_row][4 * tile_col + 3])[jj]};
            *reinterpret_cast<float4_t *>(row_ptr + 4 * acc_col + 16 * jj) = value;
        }
        return;
    }
    constexpr int CH = HEAD_SIZE_V / 16;
    constexpr int NJ = 16 / CH;
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        const int j = tile_col * NJ + jj;
        #pragma unroll
        for (int q4 = 0; q4 < CH / 4; ++q4) {
            const float4_t value = {(acc_o[tile_row][q4 * 4 + 0])[j], (acc_o[tile_row][q4 * 4 + 1])[j],
                                    (acc_o[tile_row][q4 * 4 + 2])[j], (acc_o[tile_row][q4 * 4 + 3])[j]};
            *reinterpret_cast<float4_t *>(row_ptr + CH * acc_col + 4 * CH * jj + 4 * q4) = value;
        }
    }
}

#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
// 自然通道顺序 (acc_o[m][q] 第 i 个元素为通道 16q + 4i + acc_col) 的写回: 64 列分块 tile_col 对应 q = 4 tile_col .. +3
template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void store_direct_v_acc_group_natural(float *row_ptr, const float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                                 int tile_row, int tile_col, int acc_col)
{
    #pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) row_ptr[qq * 16 + 4 * i + acc_col] = (acc_o[tile_row][tile_col * 4 + qq])[i];
    }
}

#else
#endif
// 线程块内只通过 LDS 交换数据时使用: __syncthreads() 的 workgroup release 会等待所有向量访存
// (包括仍在预取的 K/V 与已发射的 global store), 此处只需等待本 wave 的 LDS 操作完成
__device__ __forceinline__ void lds_barrier()
{
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Q 预取拆分为发射与写 LDS 两步, 两步之间发射首个 page 的 K/V 加载
template <typename scalarx8_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS>
struct QStage {
    static constexpr int ITERS = DIVIDE_ROUND_UP(MAX_ROWS * HEAD_SIZE, NUM_THREADS * 8);
    scalarx8_t data[ITERS];
};

// rows_begin/rows_count 为当前线程块处理的 Q 行 (行号包含 seqlen_q 与 q head 维度)
template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS>
__device__ __forceinline__ void issue_q_stage(QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                              const scalar_t *q, int num_heads, int num_kv_heads, int mtp,
                                              int kv_head_idx, int batch_idx, int q_stride,
                                              int rows_begin, int rows_count)
{
    const int num_q_heads = num_heads / mtp;
    const int num_q_heads_per_kv_head = num_q_heads / num_kv_heads;
    const scalar_t *q_ptr = q + batch_idx * q_stride + num_q_heads_per_kv_head * kv_head_idx * HEAD_SIZE;
    const int q_seqlen_stride = num_q_heads * HEAD_SIZE;
    const int q_seqlen_data_num = num_q_heads_per_kv_head * HEAD_SIZE;
    // 加载不放在分支内, 编译器才能只等待 Q 而不等待随后发射的 K/V; 多余线程重复读取最后 8 个元素
    const int q_elems = rows_count * HEAD_SIZE;
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = MIN_T((it * NUM_THREADS + static_cast<int>(threadIdx.x)) * 8, q_elems - 8);
        const int i = rows_begin * HEAD_SIZE + local_i;
        const int q_offset = (i / q_seqlen_data_num) * q_seqlen_stride + i % q_seqlen_data_num;
        stage.data[it] = *reinterpret_cast<const scalarx8_t *>(q_ptr + q_offset);
    }
}

template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS, int Q_LDS_PAD>
__device__ __forceinline__ void commit_q_stage(const QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                               scalar_t *smem_q, int rows_count)
{
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            *reinterpret_cast<scalarx8_t *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) =
                stage.data[it];
        }
    }
    lds_barrier();
}

// BF16 Q 以 FP16 写入 LDS, QK 使用 FP16 mmac, K 的 E5M2 解码只需移字节 (BF16 需经 FP32 转换, 每元素约多 1.5 条 VALU)。
// 按线程块内 max|Q| 选取 2 的幂缩放 2^-s 使最大值落在 [2^14, 2^15): BF16 的 8 位有效位在 FP16 的 11 位内,
// 转换前检查指数跨度; 过宽的动态范围保留 BF16 QK, 避免小分量下溢。返回 2^s, 调用者据此还原 softmax 缩放。
// reduce 为 Q 区之外的 LDS (每个 warp 一个 int); OVERLAP 时 reduce 可与 Q 区重叠 (读完归约结果后多一次屏障,
// 用于 LDS 紧凑的行切分 kernel)。
template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS, int Q_LDS_PAD, bool OVERLAP = false>
__device__ __forceinline__ float commit_q_stage_fp16(const QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                                     _Float16 *smem_q, int rows_count, uint32_t *reduce)
{
    // BF16 values are stored as uint16_t here. A scalar bit_cast of a vector
    // element is redundant and miscompiled by this DTK Clang frontend.
    static_assert(std::is_same<scalar_t, uint16_t>::value,
                  "BF16 Q range analysis requires uint16_t storage");
    using half8_v = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    uint32_t max_bits = 0, min_bits = 0x7fffu;
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t bits = static_cast<uint32_t>(stage.data[it][e]) & 0x7fffu;
                max_bits = MAX_T(max_bits, bits);
                if (bits != 0) min_bits = MIN_T(min_bits, bits);
            }
        }
    }
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        max_bits = MAX_T(max_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(max_bits), mask)));
        min_bits = MIN_T(min_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(min_bits), mask)));
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        reduce[threadIdx.x / WARP_SIZE] = max_bits;
        reduce[NUM_WARPS + threadIdx.x / WARP_SIZE] = min_bits;
    }
    lds_barrier();
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        max_bits = MAX_T(max_bits, reduce[w]);
        min_bits = MIN_T(min_bits, reduce[NUM_WARPS + w]);
    }
    // 归约区与 Q 区重叠时: 所有 warp 读完归约结果后才写 Q
    if constexpr (OVERLAP) lds_barrier();
    if (max_bits != 0 && ((max_bits >> 7) == 255 || (max_bits >> 7) - (min_bits >> 7) > 31)) {
        #pragma unroll
        for (int it = 0; it < stage.ITERS; ++it) {
            const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
            if (local_i < rows_count * HEAD_SIZE) {
                *reinterpret_cast<scalarx8_t *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) = stage.data[it];
            }
        }
        lds_barrier();
        return -1.0f; // BF16 QK inside the same direct optimized kernel.
    }
    // BF16 指数 (含偏置 127); 全零 Q 不缩放
    const int exponent = static_cast<int>(max_bits >> 7);
    const int shift = max_bits == 0 ? 0 : MIN_T(MAX_T(exponent - 127 - 14, -126), 113);
    const float q_mul = __builtin_bit_cast(float, static_cast<uint32_t>(127 - shift) << 23);
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            half8_v h;
            #pragma unroll
            for (int e = 0; e < 8; ++e) h[e] = static_cast<_Float16>(to_float(stage.data[it][e]) * q_mul);
            *reinterpret_cast<half8_v *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) = h;
        }
    }
    lds_barrier();
    return __builtin_bit_cast(float, static_cast<uint32_t>(127 + shift) << 23);
}

// ---- 页内切分模式: warp w 负责 page 内 token [16w, 16w + 16) ----
//  QK 的 B 操作数第 n 列对应 token 16w + n, 因此 S 的第 j 个元素对应 token 16w + g + 4j (g = lane >> 4);
//  PV 中 lane 组 g 的 4 个 K 元素即这 4 个 token, V 按相同顺序加载后仍用同一转置。
__device__ __forceinline__ int k_fp8_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (16 * warp_id + (lane_id & 15)) * token_stride + (lane_id >> 4) * 16;
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_fp8_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (16 * warp_id + (lane_id >> 4)) * token_stride + (lane_id & 15) * (HEAD_SIZE_V == 192 ? 4 : HEAD_SIZE_V / 16);
}

template <int HEAD_SIZE>
__device__ __forceinline__ void load_k_fp8_slice(buffer_rsrc_t k_page, int lane_offset, uint32_t raw[HEAD_SIZE / 64][4])
{
    #pragma unroll
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {
        const typename dword_vec<4>::type value = raw_buffer_load_b128(k_page, lane_offset + h * 64, 0, 0);
        #pragma unroll
        for (int w = 0; w < 4; ++w) raw[h][w] = value[w];
    }
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_v_fp8_slice(buffer_rsrc_t v_page, int lane_offset, int token_stride,
                                                 uint32_t raw[4][HEAD_SIZE_V / 64])
{
    constexpr int NW = HEAD_SIZE_V / 64;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (HEAD_SIZE_V == 192) {
            #pragma unroll
            for (int s = 0; s < 3; ++s) raw[j][s] = raw_buffer_load_b32(v_page, lane_offset + 64 * s, 4 * j * token_stride, 0);
            continue;
        } else {
        const typename dword_vec<NW>::type value = raw_buffer_load_dwords<NW>(v_page, lane_offset, 4 * j * token_stride);
        if constexpr (NW == 1) {
            raw[j][0] = value;
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) raw[j][w] = value[w];
        }
        }
    }
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD, bool is_query_fp16>
__device__ __forceinline__ void qk_fp8_slice(const uint32_t raw[HEAD_SIZE / 64][4], const scalar_t *smem_q,
                                             int seqlen_q_per_kv_head, float4_t s_reg[Mloop][1])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    // Mloop == 1 时同一 tile 的 mmac 交替写两个累加器缩短依赖链; Mloop >= 2 时各 m 已是独立链
    constexpr int NACC = Mloop == 1 ? 2 : 1;
    float4_t partial[Mloop][2];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        partial[m][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        partial[m][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {
        scalarx8_t q_fragment[Mloop][2];
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int q_row = m * 16 + q_s2r_row;
            const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + h * 64 + q_s2r_col * 16;
            if (q_row < seqlen_q_per_kv_head) {
                q_fragment[m][0] = *reinterpret_cast<const scalarx8_t *>(q_src);
                q_fragment[m][1] = *reinterpret_cast<const scalarx8_t *>(q_src + 8);
            } else {
                q_fragment[m][0] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
                q_fragment[m][1] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const scalarx4_t k_fragment = decode_e5m2x4_b16<scalarx4_t, is_query_fp16>(raw[h][sub]);
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                const scalarx8_t q8 = q_fragment[m][sub / 2];
                const scalarx4_t q_v4 = (sub % 2 == 0) ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)
                                                       : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, partial[m][sub % NACC]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) s_reg[m][0] = NACC == 2 ? partial[m][0] + partial[m][1] : partial[m][0];
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_slice(
                                                 uint32_t k_raw[HEAD_SIZE / 64][4], uint32_t v_raw[4][HEAD_SIZE_V / 64],
                                                 int block_idx, bool first_page, int warp_id, int v_token_stride,
                                                 const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                 int valid_seqlen_kv, float scale_softmax_log2,
                                                 float scores_max[Mloop], float scores_sum[Mloop],
                                                 float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    constexpr int NW = HEAD_SIZE_V / 64;
    const int lane_id = threadIdx.x % WARP_SIZE;
    float4_t s_reg[Mloop][1];
    qk_fp8_slice<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
        k_raw, smem_q, seqlen_q_per_kv_head, s_reg);

    const int token_base = block_idx * BLOCK_SIZE + 16 * warp_id;
    const bool partial_page = token_base + 15 >= valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 1, 4);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        p_reg[m] = {(_Float16)(s_reg[m][0])[0], (_Float16)(s_reg[m][0])[1],
                    (_Float16)(s_reg[m][0])[2], (_Float16)(s_reg[m][0])[3]};
    }
    if (partial_page) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) {
                #pragma unroll
                for (int w = 0; w < NW; ++w) v_raw[j][w] = 0;
            }
        }
    }
    pv_fp8_tile_f16<Mloop, HEAD_SIZE_V>(v_raw, p_reg, acc_o);
}

// 处理一个 page; PREFETCH 为 true 时在消费槽位后发射 next 页的数据, 所有加载都不在分支内,
// 编译器插入的 s_waitcnt 才能精确等待对应的槽位
template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                int k_lane_offset, int v_lane_offset,
                                                uint32_t k_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][4],
                                                uint32_t v_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::V_AHEAD][4][HEAD_SIZE_V / 64],
                                                int block_idx, bool first_page, int k_token_stride, int v_token_stride,
                                                const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                int valid_seqlen_kv, float scale_softmax_log2,
                                                float scores_max[Mloop], float scores_sum[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
    constexpr int K_CHUNKS = Cfg::K_CHUNKS;
    constexpr int V_TILES = BLOCK_SIZE / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;

    float4_t s_reg[Mloop][V_TILES];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int i = 0; i < V_TILES; ++i) s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    scalarx8_t q_pair[2][Mloop][2];
    #pragma unroll
    for (int c = 0; c < K_CHUNKS; ++c) {
        if constexpr (KSlotMap<HEAD_SIZE, Mloop>::PAIRED) {
            if (c % 2 == 0) {
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2), seqlen_q_per_kv_head, q_pair[0]);
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2) + 1, seqlen_q_per_kv_head, q_pair[1]);
            }
        }
        qk_fp8_chunk<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_slot[c % KA], smem_q, c, seqlen_q_per_kv_head, s_reg, q_pair);
        if (c + KA < K_CHUNKS) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c + KA, k_slot[c % KA]);
        } else if constexpr (PREFETCH) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_next, k_lane_offset, k_token_stride, c + KA - K_CHUNKS, k_slot[c % KA]);
        }
    }

    const bool partial_page = (block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < V_TILES; ++i) {
                const int token_idx = block_idx * BLOCK_SIZE + i + (lane_id >> 4) * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (token_idx + j * 16 >= valid_seqlen_kv) (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 4, 16);

    direct_softmax<Mloop, HEAD_SIZE_V, V_TILES>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[V_TILES][Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int t = 0; t < V_TILES; ++t) {
            p_reg[t][m] = {(_Float16)(s_reg[m][0])[t], (_Float16)(s_reg[m][1])[t],
                           (_Float16)(s_reg[m][2])[t], (_Float16)(s_reg[m][3])[t]};
        }
    }

    #pragma unroll
    for (int t = 0; t < V_TILES; ++t) {
        uint32_t (&raw)[4][Cfg::V_NW] = v_slot[t % VA];
        if (partial_page) {
            // 超出有效长度的 token 的 V 可能为任意字节, 置零避免 0 * NaN
            const int token_idx = block_idx * BLOCK_SIZE + t * 16 + (lane_id >> 4) * 4;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (token_idx + k >= valid_seqlen_kv) {
                    #pragma unroll
                    for (int w = 0; w < Cfg::V_NW; ++w) raw[k][w] = 0;
                }
            }
        }
        pv_fp8_tile_f16<Mloop, HEAD_SIZE_V>(raw, p_reg[t], acc_o);
        if (t + VA < V_TILES) {
            load_v_fp8_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t + VA, raw);
        } else if constexpr (PREFETCH) {
            load_v_fp8_tile<HEAD_SIZE_V>(v_next, v_lane_offset, v_token_stride, t + VA - V_TILES, raw);
        }
    }
}

// 常规模式: 每个 warp 依次处理 partition_idx + warp_id * P, 再每次前进 4P 的 page;
// 页内切分模式: 4 个 warp 依次处理 partition_idx, partition_idx + P, ... 的同一 page 的不同 token。
// q_commit 在首个 page 的加载发射后写 Q 到 LDS 并同步, 返回 LDS 中 Q 相对原值的缩放倒数 (softmax 缩放乘以该值);
// score_mask 处理 alibi/causal 掩码。
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename QCommit, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page_loop(const uint8_t *k_base, const uint8_t *v_base,
                                                     int k_block_stride, int v_block_stride,
                                                     const int *block_table, int partition_idx, int num_partitions,
                                                     int warp_id, bool slice_mode, int end_block, int num_kv_heads,
                                                     const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                     int valid_seqlen_kv, float scale_softmax_log2,
                                                     float scores_max[Mloop], float scores_sum[Mloop],
                                                     float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                     QCommit q_commit, ScoreMask score_mask)
{
    static_assert(BLOCK_SIZE == 64, "direct FP8 path assumes 64-token pages");
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
    const int k_token_stride = HEAD_SIZE;
    const int v_token_stride = HEAD_SIZE_V;
#else
    const int k_token_stride = HEAD_SIZE * num_kv_heads;
    const int v_token_stride = HEAD_SIZE_V * num_kv_heads;
#endif
    const int first_block = slice_mode ? partition_idx : partition_idx + warp_id * num_partitions;
    const int page_stride = slice_mode ? num_partitions : 4 * num_partitions;
    const int num_pages = first_block < end_block ? DIVIDE_ROUND_UP(end_block - first_block, page_stride) : 0;

    // 没有 page 的 warp 读取物理块 0 (总是有效), 使首个 page 的加载不在分支内
    const int first_entry = block_table[MAX_T(MIN_T(first_block, end_block - 1), 0)];
    const int64_t first_physical = num_pages > 0 ? static_cast<int64_t>(first_entry) : 0;
    buffer_rsrc_t k_page = make_page_rsrc(k_base + first_physical * k_block_stride);
    buffer_rsrc_t v_page = make_page_rsrc(v_base + first_physical * v_block_stride);

    // 页内切分模式每个线程块最多处理 2 个 page, 不做跨 page 预取, 只保留一份循环主体以控制代码体积
    if (__builtin_expect(slice_mode, 0)) {
        const int k_lane_offset = k_fp8_slice_lane_offset(k_token_stride, warp_id);
        const int v_lane_offset = v_fp8_slice_lane_offset<HEAD_SIZE_V>(v_token_stride, warp_id);
        uint32_t k_raw[Cfg::K_CHUNKS][4];
        uint32_t v_raw[4][Cfg::V_NW];
        load_k_fp8_slice<HEAD_SIZE>(k_page, k_lane_offset, k_raw);
        load_v_fp8_slice<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, v_raw);
        const float committed_q_scale = q_commit();
        const bool q_bf16 = committed_q_scale < 0.0f;
        scale_softmax_log2 *= fabsf(committed_q_scale);
        auto run_pages = [&](auto query_scalar) {
            using page_scalar_t = decltype(query_scalar);
            using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
            using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
            int block_idx = first_block;
        for (int p = 0; p < num_pages; ++p) {
            if (p > 0) {
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                load_k_fp8_slice<HEAD_SIZE>(make_page_rsrc(k_base + physical_block_number * k_block_stride), k_lane_offset, k_raw);
                load_v_fp8_slice<HEAD_SIZE_V>(make_page_rsrc(v_base + physical_block_number * v_block_stride), v_lane_offset,
                                              v_token_stride, v_raw);
            }
            direct_fp8_slice<page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
                k_raw, v_raw, block_idx, p == 0, warp_id, v_token_stride, reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv,
                scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
            block_idx += page_stride;
        }
        };
        if (q_bf16) run_pages(uint16_t{});
        else run_pages(scalar_t{});
        direct_softmax_finish<Mloop>(scores_sum);
        return;
    }

    const int k_lane_offset = k_fp8_lane_offset(k_token_stride);
    const int v_lane_offset = v_fp8_lane_offset<HEAD_SIZE_V>(v_token_stride);
    uint32_t k_slot[KA][4][4];
    uint32_t v_slot[VA][4][Cfg::V_NW];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int t = 0; t < VA; ++t) load_v_fp8_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t, v_slot[t]);
    const float committed_q_scale = q_commit();
    const bool q_bf16 = committed_q_scale < 0.0f;
    scale_softmax_log2 *= fabsf(committed_q_scale);

    auto run_pages = [&](auto query_scalar) {
        using page_scalar_t = decltype(query_scalar);
        using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
        using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
        int block_idx = first_block;
    for (int p = 0; p + 1 < num_pages; ++p) {
        const int64_t next_physical = static_cast<int64_t>(block_table[block_idx + page_stride]);
        const buffer_rsrc_t k_next = make_page_rsrc(k_base + next_physical * k_block_stride);
        const buffer_rsrc_t v_next = make_page_rsrc(v_base + next_physical * v_block_stride);
        direct_fp8_page<true, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
            k_page, v_page, k_next, v_next, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, p == 0, k_token_stride, v_token_stride,
            reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
        k_page = k_next;
        v_page = v_next;
        block_idx += page_stride;
    }
    if (num_pages > 0) {
        direct_fp8_page<false, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
            k_page, v_page, k_page, v_page, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, num_pages == 1, k_token_stride, v_token_stride,
            reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
    }
    };
    if (q_bf16) run_pages(uint16_t{});
    else run_pages(scalar_t{});
    direct_softmax_finish<Mloop>(scores_sum);
}

// ---------------------------------------------------------------------------
// 16 位 KV (BF16/FP16, 与 Q 同类型) 的直接加载路径: lane 映射、页流水、页内切分、online softmax、尾部与合并
// 都与上面的 E5M2 直接加载路径相同, 只是 K/V 不需要解码, 每个 token 的字节数加倍。
//  K: 每个 64 维 chunk 的 token 行为 128 字节。lane (r = lane & 15, g = lane >> 4) 对 token 组 j (token 4r + j)
//     发两条相邻的 16 字节加载 (chunk 内偏移 16g 与 64 + 16g), 一条 128 字节 cache line 由相邻两条指令读完。
//     第 h 条加载的 4 个 dword 为维度 32h + 8g + 0..7; 第 sub 次 mmac 用其中第 2sub、2sub + 1 个 dword
//     (维度 32h + 8g + 4sub + 0..3), Q 片段按相同维度从 LDS 读取。S 的布局与 E5M2 路径相同 (s_reg[m][j] 为 token 4r + j)。
//  V: lane (c = lane & 15, g = lane >> 4) 持有 token 16t + 4g + k 的通道 c * CH .. c * CH + CH - 1 (CH = HEAD_SIZE_V / 16,
//     2CH 字节); 通道 q 的 4 个 token 由两条 perm 组成 mmac 的 B 片段。acc_o 的通道交织与 E5M2 路径相同, 尾部共用。
//  P 用 from_float 转为 Q 的类型 (与原 16 位路径相同), BF16 KV 用 BF16 mmac。
// ---------------------------------------------------------------------------
template <int HEAD_SIZE, int HEAD_SIZE_V, int Mloop>
struct DirectB16Config {
    static constexpr bool enabled = (HEAD_SIZE == 64 || HEAD_SIZE == 128 || HEAD_SIZE == 192 || HEAD_SIZE == 256) && HEAD_SIZE_V == HEAD_SIZE;
    static constexpr int K_CHUNKS = HEAD_SIZE / 64;
    static constexpr int V_NW = HEAD_SIZE_V / 32;                 // 每个 lane 每个 token 的 V dword 数
    // 每个槽位的字节数是 E5M2 路径的两倍: 槽位数减半时在途字节数相同
    static constexpr int ACC_VGPRS = Mloop * HEAD_SIZE_V / 4;
    // 槽位数须整除 K_CHUNKS; D192 (3 个 chunk) 取 1 (每个槽位 32 个 VGPR, 3 个槽位会溢出寄存器)
    static constexpr int K_AHEAD = ACC_VGPRS >= 64 ? 1 : (K_CHUNKS % 2 == 0 ? 2 : 1);
    static constexpr int V_AHEAD = ACC_VGPRS >= 128 ? 1 : 2;
    static_assert(!enabled || (K_CHUNKS % K_AHEAD == 0 && 4 % V_AHEAD == 0), "ring slots must repeat per page");
};

// 直接加载路径 (E5M2 或 16 位) 是否用于 (cache_t, HEAD_SIZE, HEAD_SIZE_V)
template <typename cache_t, int HEAD_SIZE, int HEAD_SIZE_V>
struct DirectKV {
    static constexpr bool fp8 = std::is_same<cache_t, uint8_t>::value && DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, 1>::enabled;
    static constexpr bool b16 = !std::is_same<cache_t, uint8_t>::value && DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, 1>::enabled;
    static constexpr bool any = fp8 || b16;
};

template <int NW>
__device__ __forceinline__ void load_b16_dwords(buffer_rsrc_t rsrc, int voffset, int soffset, uint32_t *raw)
{
    if constexpr (NW <= 4) {
        const typename dword_vec<NW>::type value = raw_buffer_load_dwords<NW>(rsrc, voffset, soffset);
        if constexpr (NW == 1) {
            raw[0] = value;
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) raw[w] = value[w];
        }
    } else {
        #pragma unroll
        for (int h = 0; h < NW / 4; ++h) {
            const typename dword_vec<4>::type value = raw_buffer_load_b128(rsrc, voffset + 16 * h, soffset, 0);
            #pragma unroll
            for (int w = 0; w < 4; ++w) raw[4 * h + w] = value[w];
        }
    }
}

// token_stride 均为字节
__device__ __forceinline__ int k_b16_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * 4 * token_stride + (lane_id >> 4) * 16;
}

__device__ __forceinline__ void load_k_b16_chunk(buffer_rsrc_t k_page, int lane_offset, int token_stride, int chunk,
                                                 uint32_t raw[4][8])
{
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const typename dword_vec<4>::type value = raw_buffer_load_b128(k_page, lane_offset + chunk * 128 + h * 64, j * token_stride, 0);
            #pragma unroll
            for (int w = 0; w < 4; ++w) raw[j][4 * h + w] = value[w];
        }
    }
}

// D192: V 通道按 64 通道分段, lane c 在第 s 段持有通道 64s + 4c + i (与 D64 的交织相同), 每段一条 8 字节加载 (偏移 128s)
template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_b16_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id >> 4) * 4 * token_stride + (lane_id & 15) * (HEAD_SIZE_V == 192 ? 8 : HEAD_SIZE_V / 8);
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_v_b16_token(buffer_rsrc_t v_page, int lane_offset, int soffset, uint32_t *raw)
{
    if constexpr (HEAD_SIZE_V == 192) {
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            const typename dword_vec<2>::type value = raw_buffer_load_b64(v_page, lane_offset + 128 * s, soffset, 0);
            raw[2 * s] = value[0];
            raw[2 * s + 1] = value[1];
        }
    } else {
        load_b16_dwords<HEAD_SIZE_V / 32>(v_page, lane_offset, soffset, raw);
    }
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_v_b16_tile(buffer_rsrc_t v_page, int lane_offset, int token_stride, int tile,
                                                uint32_t raw[4][HEAD_SIZE_V / 32])
{
    #pragma unroll
    for (int k = 0; k < 4; ++k) load_v_b16_token<HEAD_SIZE_V>(v_page, lane_offset, (tile * 16 + k) * token_stride, raw[k]);
}

// 第 h 条加载的第 2sub、2sub + 1 个 dword 为 mmac 的 K 片段
template <typename scalarx4_t>
__device__ __forceinline__ scalarx4_t k_b16_fragment(const uint32_t raw[8], int h, int sub)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    return __builtin_bit_cast(scalarx4_t, (u32x2){raw[4 * h + 2 * sub], raw[4 * h + 2 * sub + 1]});
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD, bool is_query_fp16>
__device__ __forceinline__ void qk_b16_chunk(const uint32_t raw[4][8], const scalar_t *smem_q, int chunk,
                                             int seqlen_q_per_kv_head, float4_t s_reg[Mloop][4])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        scalarx8_t q8[Mloop];
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int q_row = m * 16 + q_s2r_row;
            const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + chunk * 64 + h * 32 + q_s2r_col * 8;
            q8[m] = q_row < seqlen_q_per_kv_head ? *reinterpret_cast<const scalarx8_t *>(q_src) : scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const scalarx4_t k_fragment = k_b16_fragment<scalarx4_t>(raw[t], h, sub);
                #pragma unroll
                for (int m = 0; m < Mloop; ++m) {
                    const scalarx4_t q_v4 = sub == 0 ? __builtin_shufflevector(q8[m], q8[m], 0, 1, 2, 3)
                                                     : __builtin_shufflevector(q8[m], q8[m], 4, 5, 6, 7);
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, s_reg[m][t]);
                }
            }
        }
    }
}

template <typename scalarx4_t, int Mloop, int HEAD_SIZE_V, bool is_query_fp16>
__device__ __forceinline__ void pv_b16_tile(const uint32_t raw[4][HEAD_SIZE_V / 32], const scalarx4_t p[Mloop],
                                            float4_t acc_o[Mloop][HEAD_SIZE_V / 16])
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    constexpr int CH = HEAD_SIZE_V / 16;
    #pragma unroll
    for (int q = 0; q < CH; ++q) {
        const int w = q / 2;
        // 结果为 {token0, token1} 两个 16 位值 (通道 q 在 dword w 的低/高半)
        const uint32_t sel = q % 2 == 0 ? 0x05040100u : 0x07060302u;
        const u32x2 packed = {__builtin_amdgcn_perm(raw[1][w], raw[0][w], sel),
                              __builtin_amdgcn_perm(raw[3][w], raw[2][w], sel)};
        const scalarx4_t v_fragment = __builtin_bit_cast(scalarx4_t, packed);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(p[m], v_fragment, acc_o[m][q]);
        }
    }
}

template <typename scalarx4_t, typename scalar_t>
__device__ __forceinline__ scalarx4_t p_to_b16(const float4_t a, const float4_t b, const float4_t c, const float4_t d, int t)
{
    scalarx4_t p;
    scalar_t v;
    from_float(v, a[t]); p[0] = v;
    from_float(v, b[t]); p[1] = v;
    from_float(v, c[t]); p[2] = v;
    from_float(v, d[t]); p[3] = v;
    return p;
}

// 页内切分模式: 每个 warp 处理本 page 的 16 个 token (16 * warp_id + lane & 15)
__device__ __forceinline__ int k_b16_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (16 * warp_id + (lane_id & 15)) * token_stride + (lane_id >> 4) * 16;
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_b16_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (16 * warp_id + (lane_id >> 4)) * token_stride + (lane_id & 15) * (HEAD_SIZE_V == 192 ? 8 : HEAD_SIZE_V / 8);
}

template <int HEAD_SIZE>
__device__ __forceinline__ void load_k_b16_slice(buffer_rsrc_t k_page, int lane_offset, uint32_t raw[HEAD_SIZE / 64][8])
{
    #pragma unroll
    for (int c = 0; c < HEAD_SIZE / 64; ++c) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const typename dword_vec<4>::type value = raw_buffer_load_b128(k_page, lane_offset + c * 128 + h * 64, 0, 0);
            #pragma unroll
            for (int w = 0; w < 4; ++w) raw[c][4 * h + w] = value[w];
        }
    }
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_v_b16_slice(buffer_rsrc_t v_page, int lane_offset, int token_stride,
                                                 uint32_t raw[4][HEAD_SIZE_V / 32])
{
    #pragma unroll
    for (int j = 0; j < 4; ++j) load_v_b16_token<HEAD_SIZE_V>(v_page, lane_offset, 4 * j * token_stride, raw[j]);
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_b16_slice(uint32_t k_raw[HEAD_SIZE / 64][8], uint32_t v_raw[4][HEAD_SIZE_V / 32],
                                                 int block_idx, bool first_page, int warp_id,
                                                 const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                 int valid_seqlen_kv, float scale_softmax_log2,
                                                 float scores_max[Mloop], float scores_sum[Mloop],
                                                 float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    constexpr int NW = HEAD_SIZE_V / 32;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    // Mloop == 1 时交替写两个累加器缩短依赖链
    constexpr int NACC = Mloop == 1 ? 2 : 1;
    float4_t partial[Mloop][2];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        partial[m][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        partial[m][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int c = 0; c < HEAD_SIZE / 64; ++c) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            scalarx8_t q8[Mloop];
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                const int q_row = m * 16 + q_s2r_row;
                const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + c * 64 + h * 32 + q_s2r_col * 8;
                q8[m] = q_row < seqlen_q_per_kv_head ? *reinterpret_cast<const scalarx8_t *>(q_src) : scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const scalarx4_t k_fragment = k_b16_fragment<scalarx4_t>(k_raw[c], h, sub);
                #pragma unroll
                for (int m = 0; m < Mloop; ++m) {
                    const scalarx4_t q_v4 = sub == 0 ? __builtin_shufflevector(q8[m], q8[m], 0, 1, 2, 3)
                                                     : __builtin_shufflevector(q8[m], q8[m], 4, 5, 6, 7);
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, partial[m][sub % NACC]);
                }
            }
        }
    }
    float4_t s_reg[Mloop][1];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) s_reg[m][0] = NACC == 2 ? partial[m][0] + partial[m][1] : partial[m][0];

    const int token_base = block_idx * BLOCK_SIZE + 16 * warp_id;
    const bool partial_page = token_base + 15 >= valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 1, 4);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    scalarx4_t p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            scalar_t v;
            from_float(v, (s_reg[m][0])[j]);
            p_reg[m][j] = v;
        }
    }
    if (partial_page) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) {
                #pragma unroll
                for (int w = 0; w < NW; ++w) v_raw[j][w] = 0;
            }
        }
    }
    pv_b16_tile<scalarx4_t, Mloop, HEAD_SIZE_V, is_query_fp16>(v_raw, p_reg, acc_o);
}

// 处理一个 page; PREFETCH 为 true 时在消费槽位后发射 next 页的数据 (与 direct_fp8_page 相同的槽位轮转)
template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_b16_page(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                int k_lane_offset, int v_lane_offset,
                                                uint32_t k_slot[DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][8],
                                                uint32_t v_slot[DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::V_AHEAD][4][HEAD_SIZE_V / 32],
                                                int block_idx, bool first_page, int k_token_stride, int v_token_stride,
                                                const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                int valid_seqlen_kv, float scale_softmax_log2,
                                                float scores_max[Mloop], float scores_sum[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using Cfg = DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
    constexpr int K_CHUNKS = Cfg::K_CHUNKS;
    constexpr int V_TILES = BLOCK_SIZE / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;

    float4_t s_reg[Mloop][V_TILES];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int i = 0; i < V_TILES; ++i) s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int c = 0; c < K_CHUNKS; ++c) {
        qk_b16_chunk<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_slot[c % KA], smem_q, c, seqlen_q_per_kv_head, s_reg);
        if (c + KA < K_CHUNKS) {
            load_k_b16_chunk(k_page, k_lane_offset, k_token_stride, c + KA, k_slot[c % KA]);
        } else if constexpr (PREFETCH) {
            load_k_b16_chunk(k_next, k_lane_offset, k_token_stride, c + KA - K_CHUNKS, k_slot[c % KA]);
        }
    }

    const bool partial_page = (block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < V_TILES; ++i) {
                const int token_idx = block_idx * BLOCK_SIZE + i + (lane_id >> 4) * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (token_idx + j * 16 >= valid_seqlen_kv) (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 4, 16);

    direct_softmax<Mloop, HEAD_SIZE_V, V_TILES>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    scalarx4_t p_reg[V_TILES][Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int t = 0; t < V_TILES; ++t) {
            p_reg[t][m] = p_to_b16<scalarx4_t, scalar_t>(s_reg[m][0], s_reg[m][1], s_reg[m][2], s_reg[m][3], t);
        }
    }

    #pragma unroll
    for (int t = 0; t < V_TILES; ++t) {
        uint32_t (&raw)[4][Cfg::V_NW] = v_slot[t % VA];
        if (partial_page) {
            // 超出有效长度的 token 的 V 可能为任意值, 置零避免 0 * NaN
            const int token_idx = block_idx * BLOCK_SIZE + t * 16 + (lane_id >> 4) * 4;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (token_idx + k >= valid_seqlen_kv) {
                    #pragma unroll
                    for (int w = 0; w < Cfg::V_NW; ++w) raw[k][w] = 0;
                }
            }
        }
        pv_b16_tile<scalarx4_t, Mloop, HEAD_SIZE_V, is_query_fp16>(raw, p_reg[t], acc_o);
        if (t + VA < V_TILES) {
            load_v_b16_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t + VA, raw);
        } else if constexpr (PREFETCH) {
            load_v_b16_tile<HEAD_SIZE_V>(v_next, v_lane_offset, v_token_stride, t + VA - V_TILES, raw);
        }
    }
}

// 页循环 (与 direct_fp8_page_loop 相同的工作划分); k/v_base 为字节指针, block stride 为元素数, num_kv_heads 决定 token 跨度
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename QCommit, typename ScoreMask>
__device__ __forceinline__ void direct_b16_page_loop(const uint8_t *k_base, const uint8_t *v_base,
                                                     int k_block_stride, int v_block_stride,
                                                     const int *block_table, int partition_idx, int num_partitions,
                                                     int warp_id, bool slice_mode, int end_block, int num_kv_heads,
                                                     const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                     int valid_seqlen_kv, float scale_softmax_log2,
                                                     float scores_max[Mloop], float scores_sum[Mloop],
                                                     float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                     QCommit q_commit, ScoreMask score_mask)
{
    static_assert(BLOCK_SIZE == 64, "direct 16-bit path assumes 64-token pages");
    using Cfg = DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
    constexpr int ELEM = 2;
    const int64_t k_block_bytes = static_cast<int64_t>(k_block_stride) * ELEM;
    const int64_t v_block_bytes = static_cast<int64_t>(v_block_stride) * ELEM;
    const int k_token_stride = HEAD_SIZE * ELEM * num_kv_heads;
    const int v_token_stride = HEAD_SIZE_V * ELEM * num_kv_heads;
    const int first_block = slice_mode ? partition_idx : partition_idx + warp_id * num_partitions;
    const int page_stride = slice_mode ? num_partitions : 4 * num_partitions;
    const int num_pages = first_block < end_block ? DIVIDE_ROUND_UP(end_block - first_block, page_stride) : 0;

    // 没有 page 的 warp 读取物理块 0 (总是有效), 使首个 page 的加载不在分支内
    const int first_entry = block_table[MAX_T(MIN_T(first_block, end_block - 1), 0)];
    const int64_t first_physical = num_pages > 0 ? static_cast<int64_t>(first_entry) : 0;
    buffer_rsrc_t k_page = make_page_rsrc(k_base + first_physical * k_block_bytes);
    buffer_rsrc_t v_page = make_page_rsrc(v_base + first_physical * v_block_bytes);

    if (__builtin_expect(slice_mode, 0)) {
        const int k_lane_offset = k_b16_slice_lane_offset(k_token_stride, warp_id);
        const int v_lane_offset = v_b16_slice_lane_offset<HEAD_SIZE_V>(v_token_stride, warp_id);
        uint32_t k_raw[Cfg::K_CHUNKS][8];
        uint32_t v_raw[4][Cfg::V_NW];
        load_k_b16_slice<HEAD_SIZE>(k_page, k_lane_offset, k_raw);
        load_v_b16_slice<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, v_raw);
        scale_softmax_log2 *= fabsf(q_commit());
        int block_idx = first_block;
        for (int p = 0; p < num_pages; ++p) {
            if (p > 0) {
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                load_k_b16_slice<HEAD_SIZE>(make_page_rsrc(k_base + physical_block_number * k_block_bytes), k_lane_offset, k_raw);
                load_v_b16_slice<HEAD_SIZE_V>(make_page_rsrc(v_base + physical_block_number * v_block_bytes), v_lane_offset,
                                              v_token_stride, v_raw);
            }
            direct_b16_slice<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
                k_raw, v_raw, block_idx, p == 0, warp_id, smem_q, seqlen_q_per_kv_head, valid_seqlen_kv,
                scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
            block_idx += page_stride;
        }
        direct_softmax_finish<Mloop>(scores_sum);
        return;
    }

    const int k_lane_offset = k_b16_lane_offset(k_token_stride);
    const int v_lane_offset = v_b16_lane_offset<HEAD_SIZE_V>(v_token_stride);
    uint32_t k_slot[KA][4][8];
    uint32_t v_slot[VA][4][Cfg::V_NW];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_b16_chunk(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int t = 0; t < VA; ++t) load_v_b16_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t, v_slot[t]);
    scale_softmax_log2 *= fabsf(q_commit());

    int block_idx = first_block;
    for (int p = 0; p + 1 < num_pages; ++p) {
        const int64_t next_physical = static_cast<int64_t>(block_table[block_idx + page_stride]);
        const buffer_rsrc_t k_next = make_page_rsrc(k_base + next_physical * k_block_bytes);
        const buffer_rsrc_t v_next = make_page_rsrc(v_base + next_physical * v_block_bytes);
        direct_b16_page<true, scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_page, v_page, k_next, v_next, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, p == 0, k_token_stride, v_token_stride,
            smem_q, seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
        k_page = k_next;
        v_page = v_next;
        block_idx += page_stride;
    }
    if (num_pages > 0) {
        direct_b16_page<false, scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_page, v_page, k_page, v_page, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, num_pages == 1, k_token_stride, v_token_stride,
            smem_q, seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
    }
    direct_softmax_finish<Mloop>(scores_sum);
}

#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
// layout=0: V 字节放到 16 位高字节后的解码 (E5M2 即为 FP16; E4M3 见 e4m3_high_bytes_to_f16x2)
__device__ __forceinline__ uint32_t vt_high_bytes_to_f16x2(uint32_t x) { return x; }
// ---------------------------------------------------------------------------
// layout=0 (V 为 [num_blocks, num_kv_heads, head_dim_v, block_size], 每个通道的 64 个 token 连续) 的 FP8 直接加载路径。
//  K 与 BHSD 相同; 只改 K 的 lane -> token 对应, 使 S/P 中 lane 组 g 持有 token [16g, 16g + 16):
//   常规模式: lane r = lane & 15 读取 token 16 (r & 3) + 4 (r >> 2) + j (j 为 token 组), 于是 S 第 j 组第 e 个元素
//             对应 token 16g + 4e + j; PV 第 t 个 16-token tile 的 P 片段第 jj 个元素对应 token 16g + 4t + jj。
//   页内切分: lane r 读取 token 16w + 4 (r & 3) + (r >> 2), S 第 e 个元素对应 token 16w + 4g + e。
//  V: lane (c = lane & 15, g = lane >> 4) 的通道 c * CH + q 与原路径相同 (acc_o 的通道交织与尾部不变);
//   常规模式每个通道一条 16 字节加载取得 token 16g .. 16g + 15 (4 个 tile 的 B 片段各 1 个 dword),
//   页内切分每个通道一个 dword (token 16w + 4g .. +3)。字节放到 16 位高字节后由 vt_high_bytes_to_f16x2 转为 FP16。
//  通道按 G 个一组加载 (组环形预取), 计算时 tile 在外、组内通道在内, 各通道的累加器相互独立。
// ---------------------------------------------------------------------------
template <int HEAD_SIZE_V, int Mloop, int ELEM>
struct DirectVtConfig {
    static constexpr int CH = HEAD_SIZE_V / 16;                    // 每个 lane 的通道数
    static constexpr int G = 4;                                    // 每组通道数
    static constexpr int GROUPS = CH / G;
    static constexpr int NW = 16 * ELEM / 4;                       // 每个通道每 lane 的 dword 数 (16 个 token)
    static constexpr int ACC_VGPRS = Mloop * HEAD_SIZE_V / 4;
    static constexpr int VA0 = ACC_VGPRS >= 128 ? 1 : 2;
    // 组槽位数 (每页从槽位 0 开始, 须整除组数): D192 为 3 组, 累加器较少时 3 个槽位, 否则 1 个
    static constexpr int VA = GROUPS % VA0 == 0 ? VA0 : (ACC_VGPRS < 64 ? GROUPS : 1);
    static_assert(CH % G == 0 && GROUPS % VA == 0, "vt ring slots must repeat per page");
};

__device__ __forceinline__ int k_fp8_vt_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int r = lane_id & 15;
    return (16 * (r & 3) + 4 * (r >> 2)) * token_stride + (lane_id >> 4) * 16;
}

__device__ __forceinline__ int k_fp8_vt_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int r = lane_id & 15;
    return (16 * warp_id + 4 * (r & 3) + (r >> 2)) * token_stride + (lane_id >> 4) * 16;
}

// 通道行为 64 * ELEM 字节; lane 偏移指向本 lane 第 0 个通道的 token 16g (常规) 或 16w + 4g (页内切分)
// D192 的通道按 64 通道分段: lane c 的第 q 个通道为 64 (q / 4) + 4c + q % 4 (每段与 D64 相同), 其他 D 为 c * CH + q
template <int HEAD_SIZE_V>
__device__ __forceinline__ constexpr int vt_lane_channels() { return HEAD_SIZE_V == 192 ? 4 : HEAD_SIZE_V / 16; }
template <int HEAD_SIZE_V>
__device__ __forceinline__ constexpr int vt_channel_offset(int q) { return HEAD_SIZE_V == 192 ? 64 * (q / 4) + q % 4 : q; }

template <int HEAD_SIZE_V, int ELEM>
__device__ __forceinline__ int vt_lane_offset()
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * vt_lane_channels<HEAD_SIZE_V>() * 64 * ELEM + (lane_id >> 4) * 16 * ELEM;
}

template <int HEAD_SIZE_V, int ELEM>
__device__ __forceinline__ int vt_slice_lane_offset(int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * vt_lane_channels<HEAD_SIZE_V>() * 64 * ELEM + (16 * warp_id + 4 * (lane_id >> 4)) * ELEM;
}

// 第 gi 组的 G 个通道, 每个通道 NW 个 dword
template <int HEAD_SIZE_V, int Mloop, int ELEM>
__device__ __forceinline__ void load_vt_group(buffer_rsrc_t v_page, int lane_offset, int gi,
                                              uint32_t raw[DirectVtConfig<HEAD_SIZE_V, Mloop, ELEM>::G][DirectVtConfig<HEAD_SIZE_V, Mloop, ELEM>::NW])
{
    using Cfg = DirectVtConfig<HEAD_SIZE_V, Mloop, ELEM>;
    #pragma unroll
    for (int u = 0; u < Cfg::G; ++u) {
        #pragma unroll
        for (int h = 0; h < Cfg::NW / 4; ++h) {
            const typename dword_vec<4>::type value =
                raw_buffer_load_b128(v_page, lane_offset + vt_channel_offset<HEAD_SIZE_V>(gi * Cfg::G + u) * 64 * ELEM + 16 * h, 0, 0);
            #pragma unroll
            for (int w = 0; w < 4; ++w) raw[u][4 * h + w] = value[w];
        }
    }
}

// 有效 token 数 vc (本 lane 的 16 个 token 中前 vc 个有效): 第 w 个 dword 的保留掩码 (每个 token ELEM 字节)
template <int ELEM>
__device__ __forceinline__ uint32_t vt_keep_mask(int vc, int w)
{
    constexpr int TPW = 4 / ELEM;                                   // 每个 dword 的 token 数
    const int keep = vc - w * TPW;
    return keep >= TPW ? 0xffffffffu : (keep <= 0 ? 0u : (0xffffffffu >> (32 - 8 * ELEM * keep)));
}

template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void pv_fp8_vt_group(const uint32_t raw[4][4], const half4_v p[4][Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], int q0)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        half4_v v_fragment[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            // 一个 dword 为同一通道的 4 个 token: 各放到一个 16 位的高字节
            const uint32_t v01 = __builtin_amdgcn_perm(raw[u][t], raw[u][t], 0x010C000Cu);
            const uint32_t v23 = __builtin_amdgcn_perm(raw[u][t], raw[u][t], 0x030C020Cu);
            v_fragment[u] = __builtin_bit_cast(half4_v, (u32x2){vt_high_bytes_to_f16x2(v01), vt_high_bytes_to_f16x2(v23)});
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                builtin_amdgcn_mmac<half4_v, true>(p[t][m], v_fragment[u], acc_o[m][q0 + u]);
            }
        }
    }
}

template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page_vt(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                   buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                   int k_lane_offset, int v_lane_offset,
                                                   uint32_t k_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][4],
                                                   uint32_t v_slot[DirectVtConfig<HEAD_SIZE_V, Mloop, 1>::VA][4][4],
                                                   int block_idx, bool first_page, int k_token_stride,
                                                   const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                   int valid_seqlen_kv, float scale_softmax_log2,
                                                   float scores_max[Mloop], float scores_sum[Mloop],
                                                   float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    using Vt = DirectVtConfig<HEAD_SIZE_V, Mloop, 1>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Vt::VA;
    constexpr int K_CHUNKS = Cfg::K_CHUNKS;
    constexpr int V_TILES = BLOCK_SIZE / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;

    float4_t s_reg[Mloop][V_TILES];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int i = 0; i < V_TILES; ++i) s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    scalarx8_t q_pair[2][Mloop][2];
    #pragma unroll
    for (int c = 0; c < K_CHUNKS; ++c) {
        if constexpr (KSlotMap<HEAD_SIZE, Mloop>::PAIRED) {
            if (c % 2 == 0) {
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2), seqlen_q_per_kv_head, q_pair[0]);
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2) + 1, seqlen_q_per_kv_head, q_pair[1]);
            }
        }
        qk_fp8_chunk<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_slot[c % KA], smem_q, c, seqlen_q_per_kv_head, s_reg, q_pair);
        if (c + KA < K_CHUNKS) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c + KA, k_slot[c % KA]);
        } else if constexpr (PREFETCH) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_next, k_lane_offset, k_token_stride, c + KA - K_CHUNKS, k_slot[c % KA]);
        }
    }

    // S 第 i 组第 j 个元素对应 token 16g + 4j + i
    const bool partial_page = (block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < V_TILES; ++i) {
                const int token_idx = block_idx * BLOCK_SIZE + i + (lane_id >> 4) * 16;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (token_idx + j * 4 >= valid_seqlen_kv) (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 16, 4);

    direct_softmax<Mloop, HEAD_SIZE_V, V_TILES>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[V_TILES][Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int t = 0; t < V_TILES; ++t) {
            p_reg[t][m] = {(_Float16)(s_reg[m][0])[t], (_Float16)(s_reg[m][1])[t],
                           (_Float16)(s_reg[m][2])[t], (_Float16)(s_reg[m][3])[t]};
        }
    }

    // 超出有效长度的 token 的 V 可能为任意字节 (含 NaN 编码), 置零避免 0 * NaN
    const int vc = valid_seqlen_kv - (block_idx * BLOCK_SIZE + (lane_id >> 4) * 16);
    #pragma unroll
    for (int gi = 0; gi < Vt::GROUPS; ++gi) {
        uint32_t (&raw)[4][4] = v_slot[gi % VA];
        if (partial_page) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t keep = vt_keep_mask<1>(vc, w);
                #pragma unroll
                for (int u = 0; u < 4; ++u) raw[u][w] &= keep;
            }
        }
        pv_fp8_vt_group<Mloop, HEAD_SIZE_V>(raw, p_reg, acc_o, gi * 4);
        if (gi + VA < Vt::GROUPS) {
            load_vt_group<HEAD_SIZE_V, Mloop, 1>(v_page, v_lane_offset, gi + VA, raw);
        } else if constexpr (PREFETCH) {
            load_vt_group<HEAD_SIZE_V, Mloop, 1>(v_next, v_lane_offset, gi + VA - Vt::GROUPS, raw);
        }
    }
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_slice_vt(uint32_t k_raw[HEAD_SIZE / 64][4], uint32_t v_raw[HEAD_SIZE_V / 16],
                                                    int block_idx, bool first_page, int warp_id,
                                                    const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                    int valid_seqlen_kv, float scale_softmax_log2,
                                                    float scores_max[Mloop], float scores_sum[Mloop],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    constexpr int CH = HEAD_SIZE_V / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;
    float4_t s_reg[Mloop][1];
    qk_fp8_slice<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
        k_raw, smem_q, seqlen_q_per_kv_head, s_reg);

    // S 第 j 个元素对应 token 16w + 4g + j
    const int token_base = block_idx * BLOCK_SIZE + 16 * warp_id;
    const bool partial_page = token_base + 15 >= valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (token_base + 4 * (lane_id >> 4) + j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 4, 1);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        p_reg[m] = {(_Float16)(s_reg[m][0])[0], (_Float16)(s_reg[m][0])[1],
                    (_Float16)(s_reg[m][0])[2], (_Float16)(s_reg[m][0])[3]};
    }
    if (partial_page) {
        const uint32_t keep = vt_keep_mask<1>(valid_seqlen_kv - (token_base + 4 * (lane_id >> 4)), 0);
        #pragma unroll
        for (int q = 0; q < CH; ++q) v_raw[q] &= keep;
    }
    #pragma unroll
    for (int q0 = 0; q0 < CH; q0 += 2) {
        half4_v v_fragment[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            const uint32_t v01 = __builtin_amdgcn_perm(v_raw[q0 + u], v_raw[q0 + u], 0x010C000Cu);
            const uint32_t v23 = __builtin_amdgcn_perm(v_raw[q0 + u], v_raw[q0 + u], 0x030C020Cu);
            v_fragment[u] = __builtin_bit_cast(half4_v, (u32x2){vt_high_bytes_to_f16x2(v01), vt_high_bytes_to_f16x2(v23)});
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) builtin_amdgcn_mmac<half4_v, true>(p_reg[m], v_fragment[u], acc_o[m][q0 + u]);
        }
    }
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_vt_fp8_slice(buffer_rsrc_t v_page, int lane_offset, uint32_t raw[HEAD_SIZE_V / 16])
{
    #pragma unroll
    for (int q = 0; q < HEAD_SIZE_V / 16; ++q) raw[q] = raw_buffer_load_b32(v_page, lane_offset + vt_channel_offset<HEAD_SIZE_V>(q) * 64, 0, 0);
}

// 页循环 (与 direct_fp8_page_loop 相同的工作划分)
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename QCommit, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page_loop_vt(const uint8_t *k_base, const uint8_t *v_base,
                                                        int k_block_stride, int v_block_stride,
                                                        const int *block_table, int partition_idx, int num_partitions,
                                                        int warp_id, bool slice_mode, int end_block, int num_kv_heads,
                                                        const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                        int valid_seqlen_kv, float scale_softmax_log2,
                                                        float scores_max[Mloop], float scores_sum[Mloop],
                                                        float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                        QCommit q_commit, ScoreMask score_mask)
{
    static_assert(BLOCK_SIZE == 64, "direct FP8 path assumes 64-token pages");
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    using Vt = DirectVtConfig<HEAD_SIZE_V, Mloop, 1>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Vt::VA;
    const int k_token_stride = HEAD_SIZE;
    const int first_block = slice_mode ? partition_idx : partition_idx + warp_id * num_partitions;
    const int page_stride = slice_mode ? num_partitions : 4 * num_partitions;
    const int num_pages = first_block < end_block ? DIVIDE_ROUND_UP(end_block - first_block, page_stride) : 0;

    const int first_entry = block_table[MAX_T(MIN_T(first_block, end_block - 1), 0)];
    const int64_t first_physical = num_pages > 0 ? static_cast<int64_t>(first_entry) : 0;
    buffer_rsrc_t k_page = make_page_rsrc(k_base + first_physical * k_block_stride);
    buffer_rsrc_t v_page = make_page_rsrc(v_base + first_physical * v_block_stride);

    if (__builtin_expect(slice_mode, 0)) {
        const int k_lane_offset = k_fp8_vt_slice_lane_offset(k_token_stride, warp_id);
        const int v_lane_offset = vt_slice_lane_offset<HEAD_SIZE_V, 1>(warp_id);
        uint32_t k_raw[Cfg::K_CHUNKS][4];
        uint32_t v_raw[HEAD_SIZE_V / 16];
        load_k_fp8_slice<HEAD_SIZE>(k_page, k_lane_offset, k_raw);
        load_vt_fp8_slice<HEAD_SIZE_V>(v_page, v_lane_offset, v_raw);
        const float committed_q_scale = q_commit();
        const bool q_bf16 = committed_q_scale < 0.0f;
        scale_softmax_log2 *= fabsf(committed_q_scale);
        auto run_pages = [&](auto query_scalar) {
            using page_scalar_t = decltype(query_scalar);
            using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
            using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
            int block_idx = first_block;
            for (int p = 0; p < num_pages; ++p) {
                if (p > 0) {
                    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                    load_k_fp8_slice<HEAD_SIZE>(make_page_rsrc(k_base + physical_block_number * k_block_stride), k_lane_offset, k_raw);
                    load_vt_fp8_slice<HEAD_SIZE_V>(make_page_rsrc(v_base + physical_block_number * v_block_stride), v_lane_offset, v_raw);
                }
                direct_fp8_slice_vt<page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
                    k_raw, v_raw, block_idx, p == 0, warp_id, reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head,
                    valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
                block_idx += page_stride;
            }
        };
        if (q_bf16) run_pages(uint16_t{});
        else run_pages(scalar_t{});
        direct_softmax_finish<Mloop>(scores_sum);
        return;
    }

    const int k_lane_offset = k_fp8_vt_lane_offset(k_token_stride);
    const int v_lane_offset = vt_lane_offset<HEAD_SIZE_V, 1>();
    uint32_t k_slot[KA][4][4];
    uint32_t v_slot[VA][4][4];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int gi = 0; gi < VA; ++gi) load_vt_group<HEAD_SIZE_V, Mloop, 1>(v_page, v_lane_offset, gi, v_slot[gi]);
    const float committed_q_scale = q_commit();
    const bool q_bf16 = committed_q_scale < 0.0f;
    scale_softmax_log2 *= fabsf(committed_q_scale);

    auto run_pages = [&](auto query_scalar) {
        using page_scalar_t = decltype(query_scalar);
        using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
        using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
        int block_idx = first_block;
        for (int p = 0; p + 1 < num_pages; ++p) {
            const int64_t next_physical = static_cast<int64_t>(block_table[block_idx + page_stride]);
            const buffer_rsrc_t k_next = make_page_rsrc(k_base + next_physical * k_block_stride);
            const buffer_rsrc_t v_next = make_page_rsrc(v_base + next_physical * v_block_stride);
            direct_fp8_page_vt<true, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
                k_page, v_page, k_next, v_next, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, p == 0, k_token_stride,
                reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
            k_page = k_next;
            v_page = v_next;
            block_idx += page_stride;
        }
        if (num_pages > 0) {
            direct_fp8_page_vt<false, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
                k_page, v_page, k_page, v_page, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, num_pages == 1, k_token_stride,
                reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
        }
    };
    if (q_bf16) run_pages(uint16_t{});
    else run_pages(scalar_t{});
    direct_softmax_finish<Mloop>(scores_sum);
}
// ---------------------------------------------------------------------------
// layout=0 的 16 位 KV 直接加载路径 (每个元素 2 字节; 通道 16 个 token 的行为 128 字节):
//  通道顺序: lane (c = lane & 15, g = lane >> 4) 的第 q 个通道为 16q + c (同一条指令的 16 个 lane 读取相邻的 16 个通道行);
//   acc_o[m][q] 第 i 个元素为通道 16q + 4i + g, 尾部用 store_direct_v_acc_group_natural 写回。
//  常规模式: V 每个通道两条 16 字节加载, 第 h 条为 token 32h + 8g .. +7 (同一行的 4 个 lane 读取连续 64 字节,
//   两条相邻指令读完整行, 与 K 的读取方式相同)。K 的 lane r 读取 token 32 (r >> 3) + 8 (r & 3) + 4 ((r >> 2) & 1) + j,
//   于是 S 第 j 组第 e 个元素对应 token 32 (e >> 1) + 8g + 4 (e & 1) + j, PV 第 e 个 tile 的 B 片段为第 e >> 1 条加载的
//   第 2 (e & 1)、2 (e & 1) + 1 个 dword。score_mask 用 stride_j = -1 表示这一分段对应。
//   通道按 4 个一组加载 (组环形预取)。
//  页内切分: lane r 读取 token 16w + 4 (r & 3) + (r >> 2), S 第 e 个元素对应 token 16w + 4g + e;
//   V 每个通道一条 8 字节加载 (token 16w + 4g .. +3)。
// ---------------------------------------------------------------------------
__device__ __forceinline__ int k_b16_vt_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int r = lane_id & 15;
    return (32 * (r >> 3) + 8 * (r & 3) + 4 * ((r >> 2) & 1)) * token_stride + (lane_id >> 4) * 16;
}

__device__ __forceinline__ int k_b16_vt_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int r = lane_id & 15;
    return (16 * warp_id + 4 * (r & 3) + (r >> 2)) * token_stride + (lane_id >> 4) * 16;
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ int vt_b16_slice_lane_offset(int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * 128 + (16 * warp_id + 4 * (lane_id >> 4)) * 2;
}

__device__ __forceinline__ int vt_b16_lane_offset()
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * 128 + (lane_id >> 4) * 16;
}

// 第 gi 组的 4 个通道 (第 u 个为 16 (4 gi + u) + c), 每个通道两条 16 字节加载
__device__ __forceinline__ void load_vt_b16_group(buffer_rsrc_t v_page, int lane_offset, int gi, uint32_t raw[4][8])
{
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const typename dword_vec<4>::type value = raw_buffer_load_b128(v_page, lane_offset + (4 * gi + u) * 16 * 128 + 64 * h, 0, 0);
            #pragma unroll
            for (int w = 0; w < 4; ++w) raw[u][4 * h + w] = value[w];
        }
    }
}

template <typename scalarx4_t, int Mloop, int HEAD_SIZE_V, bool is_query_fp16>
__device__ __forceinline__ void pv_b16_vt_group(const uint32_t raw[4][8], const scalarx4_t p[4][Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], int q0)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int w = 4 * (e >> 1) + 2 * (e & 1);
            const scalarx4_t v_fragment = __builtin_bit_cast(scalarx4_t, (u32x2){raw[u][w], raw[u][w + 1]});
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(p[e][m], v_fragment, acc_o[m][q0 + u]);
            }
        }
    }
}

template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_b16_page_vt(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                   buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                   int k_lane_offset, int v_lane_offset,
                                                   uint32_t k_slot[DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][8],
                                                   uint32_t v_slot[DirectVtConfig<HEAD_SIZE_V, Mloop, 2>::VA][4][8],
                                                   int block_idx, bool first_page, int k_token_stride,
                                                   const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                   int valid_seqlen_kv, float scale_softmax_log2,
                                                   float scores_max[Mloop], float scores_sum[Mloop],
                                                   float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using Cfg = DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    using Vt = DirectVtConfig<HEAD_SIZE_V, Mloop, 2>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Vt::VA;
    constexpr int K_CHUNKS = Cfg::K_CHUNKS;
    constexpr int V_TILES = BLOCK_SIZE / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int g = lane_id >> 4;

    float4_t s_reg[Mloop][V_TILES];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int i = 0; i < V_TILES; ++i) s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int c = 0; c < K_CHUNKS; ++c) {
        qk_b16_chunk<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_slot[c % KA], smem_q, c, seqlen_q_per_kv_head, s_reg);
        if (c + KA < K_CHUNKS) {
            load_k_b16_chunk(k_page, k_lane_offset, k_token_stride, c + KA, k_slot[c % KA]);
        } else if constexpr (PREFETCH) {
            load_k_b16_chunk(k_next, k_lane_offset, k_token_stride, c + KA - K_CHUNKS, k_slot[c % KA]);
        }
    }

    // S 第 i 组第 e 个元素对应 token 32 (e >> 1) + 8g + 4 (e & 1) + i
    const bool partial_page = (block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < V_TILES; ++i) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int token_idx = block_idx * BLOCK_SIZE + 32 * (e >> 1) + 8 * g + 4 * (e & 1) + i;
                    if (token_idx >= valid_seqlen_kv) (s_reg[m][i])[e] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 8, -1);

    direct_softmax<Mloop, HEAD_SIZE_V, V_TILES>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    scalarx4_t p_reg[V_TILES][Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int e = 0; e < V_TILES; ++e) {
            p_reg[e][m] = p_to_b16<scalarx4_t, scalar_t>(s_reg[m][0], s_reg[m][1], s_reg[m][2], s_reg[m][3], e);
        }
    }

    #pragma unroll
    for (int gi = 0; gi < Vt::GROUPS; ++gi) {
        uint32_t (&raw)[4][8] = v_slot[gi % VA];
        if (partial_page) {
            // 超出有效长度的 token 的 V 可能为任意值, 置零避免 0 * NaN; 第 h 条加载为 token 32h + 8g .. +7
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int vc = valid_seqlen_kv - (block_idx * BLOCK_SIZE + 32 * h + 8 * g);
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t keep = vt_keep_mask<2>(vc, w);
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) raw[u][4 * h + w] &= keep;
                }
            }
        }
        pv_b16_vt_group<scalarx4_t, Mloop, HEAD_SIZE_V, is_query_fp16>(raw, p_reg, acc_o, gi * 4);
        if (gi + VA < Vt::GROUPS) {
            load_vt_b16_group(v_page, v_lane_offset, gi + VA, raw);
        } else if constexpr (PREFETCH) {
            load_vt_b16_group(v_next, v_lane_offset, gi + VA - Vt::GROUPS, raw);
        }
    }
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_b16_slice_vt(uint32_t k_raw[HEAD_SIZE / 64][8], uint32_t v_raw[HEAD_SIZE_V / 16][2],
                                                    int block_idx, bool first_page, int warp_id,
                                                    const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                    int valid_seqlen_kv, float scale_softmax_log2,
                                                    float scores_max[Mloop], float scores_sum[Mloop],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    constexpr int CH = HEAD_SIZE_V / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    constexpr int NACC = Mloop == 1 ? 2 : 1;
    float4_t partial[Mloop][2];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        partial[m][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        partial[m][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int c = 0; c < HEAD_SIZE / 64; ++c) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            scalarx8_t q8[Mloop];
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                const int q_row = m * 16 + q_s2r_row;
                const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + c * 64 + h * 32 + q_s2r_col * 8;
                q8[m] = q_row < seqlen_q_per_kv_head ? *reinterpret_cast<const scalarx8_t *>(q_src) : scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const scalarx4_t k_fragment = k_b16_fragment<scalarx4_t>(k_raw[c], h, sub);
                #pragma unroll
                for (int m = 0; m < Mloop; ++m) {
                    const scalarx4_t q_v4 = sub == 0 ? __builtin_shufflevector(q8[m], q8[m], 0, 1, 2, 3)
                                                     : __builtin_shufflevector(q8[m], q8[m], 4, 5, 6, 7);
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, partial[m][sub % NACC]);
                }
            }
        }
    }
    float4_t s_reg[Mloop][1];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) s_reg[m][0] = NACC == 2 ? partial[m][0] + partial[m][1] : partial[m][0];

    // S 第 j 个元素对应 token 16w + 4g + j
    const int token_base = block_idx * BLOCK_SIZE + 16 * warp_id;
    const bool partial_page = token_base + 15 >= valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (token_base + 4 * (lane_id >> 4) + j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 4, 1);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    scalarx4_t p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            scalar_t v;
            from_float(v, (s_reg[m][0])[j]);
            p_reg[m][j] = v;
        }
    }
    if (partial_page) {
        const int vc = valid_seqlen_kv - (token_base + 4 * (lane_id >> 4));
        const uint32_t keep0 = vt_keep_mask<2>(vc, 0);
        const uint32_t keep1 = vt_keep_mask<2>(vc, 1);
        #pragma unroll
        for (int q = 0; q < CH; ++q) {
            v_raw[q][0] &= keep0;
            v_raw[q][1] &= keep1;
        }
    }
    #pragma unroll
    for (int q = 0; q < CH; ++q) {
        const scalarx4_t v_fragment = __builtin_bit_cast(scalarx4_t, (u32x2){v_raw[q][0], v_raw[q][1]});
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(p_reg[m], v_fragment, acc_o[m][q]);
    }
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ void load_vt_b16_slice(buffer_rsrc_t v_page, int lane_offset, uint32_t raw[HEAD_SIZE_V / 16][2])
{
    #pragma unroll
    for (int q = 0; q < HEAD_SIZE_V / 16; ++q) {
        const typename dword_vec<2>::type value = raw_buffer_load_b64(v_page, lane_offset + q * 16 * 128, 0, 0);
        raw[q][0] = value[0];
        raw[q][1] = value[1];
    }
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename QCommit, typename ScoreMask>
__device__ __forceinline__ void direct_b16_page_loop_vt(const uint8_t *k_base, const uint8_t *v_base,
                                                        int k_block_stride, int v_block_stride,
                                                        const int *block_table, int partition_idx, int num_partitions,
                                                        int warp_id, bool slice_mode, int end_block, int num_kv_heads,
                                                        const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                        int valid_seqlen_kv, float scale_softmax_log2,
                                                        float scores_max[Mloop], float scores_sum[Mloop],
                                                        float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                        QCommit q_commit, ScoreMask score_mask)
{
    static_assert(BLOCK_SIZE == 64, "direct 16-bit path assumes 64-token pages");
    using Cfg = DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    using Vt = DirectVtConfig<HEAD_SIZE_V, Mloop, 2>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Vt::VA;
    constexpr int ELEM = 2;
    const int64_t k_block_bytes = static_cast<int64_t>(k_block_stride) * ELEM;
    const int64_t v_block_bytes = static_cast<int64_t>(v_block_stride) * ELEM;
    const int k_token_stride = HEAD_SIZE * ELEM;
    const int first_block = slice_mode ? partition_idx : partition_idx + warp_id * num_partitions;
    const int page_stride = slice_mode ? num_partitions : 4 * num_partitions;
    const int num_pages = first_block < end_block ? DIVIDE_ROUND_UP(end_block - first_block, page_stride) : 0;

    const int first_entry = block_table[MAX_T(MIN_T(first_block, end_block - 1), 0)];
    const int64_t first_physical = num_pages > 0 ? static_cast<int64_t>(first_entry) : 0;
    buffer_rsrc_t k_page = make_page_rsrc(k_base + first_physical * k_block_bytes);
    buffer_rsrc_t v_page = make_page_rsrc(v_base + first_physical * v_block_bytes);

    if (__builtin_expect(slice_mode, 0)) {
        const int k_lane_offset = k_b16_vt_slice_lane_offset(k_token_stride, warp_id);
        const int v_lane_offset = vt_b16_slice_lane_offset<HEAD_SIZE_V>(warp_id);
        uint32_t k_raw[Cfg::K_CHUNKS][8];
        uint32_t v_raw[HEAD_SIZE_V / 16][2];
        load_k_b16_slice<HEAD_SIZE>(k_page, k_lane_offset, k_raw);
        load_vt_b16_slice<HEAD_SIZE_V>(v_page, v_lane_offset, v_raw);
        scale_softmax_log2 *= fabsf(q_commit());
        int block_idx = first_block;
        for (int p = 0; p < num_pages; ++p) {
            if (p > 0) {
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                load_k_b16_slice<HEAD_SIZE>(make_page_rsrc(k_base + physical_block_number * k_block_bytes), k_lane_offset, k_raw);
                load_vt_b16_slice<HEAD_SIZE_V>(make_page_rsrc(v_base + physical_block_number * v_block_bytes), v_lane_offset, v_raw);
            }
            direct_b16_slice_vt<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
                k_raw, v_raw, block_idx, p == 0, warp_id, smem_q, seqlen_q_per_kv_head, valid_seqlen_kv,
                scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
            block_idx += page_stride;
        }
        direct_softmax_finish<Mloop>(scores_sum);
        return;
    }

    const int k_lane_offset = k_b16_vt_lane_offset(k_token_stride);
    const int v_lane_offset = vt_b16_lane_offset();
    uint32_t k_slot[KA][4][8];
    uint32_t v_slot[VA][4][8];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_b16_chunk(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int gi = 0; gi < VA; ++gi) load_vt_b16_group(v_page, v_lane_offset, gi, v_slot[gi]);
    scale_softmax_log2 *= fabsf(q_commit());

    int block_idx = first_block;
    for (int p = 0; p + 1 < num_pages; ++p) {
        const int64_t next_physical = static_cast<int64_t>(block_table[block_idx + page_stride]);
        const buffer_rsrc_t k_next = make_page_rsrc(k_base + next_physical * k_block_bytes);
        const buffer_rsrc_t v_next = make_page_rsrc(v_base + next_physical * v_block_bytes);
        direct_b16_page_vt<true, scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_page, v_page, k_next, v_next, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, p == 0, k_token_stride,
            smem_q, seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
        k_page = k_next;
        v_page = v_next;
        block_idx += page_stride;
    }
    if (num_pages > 0) {
        direct_b16_page_vt<false, scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_page, v_page, k_page, v_page, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, num_pages == 1, k_token_stride,
            smem_q, seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
    }
    direct_softmax_finish<Mloop>(scores_sum);
}

#else
#endif
// ---------------------------------------------------------------------------
// D256 大 Q 组 (每个 KV head 33..64 行, BF16 Q) 的行切分路径:
//  线程块对应 (kv head, partition); 4 个 warp 各负责 16 行 Q (Mloop = 1), Q 片段常驻寄存器。
//  partition p 依次处理 32-token chunk p, p + P, ... (全卡同时读取的 chunk 相邻, 保持紧凑读取窗口);
//  每个 chunk 的 K/V (各 8KB 连续字节) 由 4 个 warp 各发射 1 条 K、1 条 V 的 matrix_load_64x32_b8
//  (行跨度 64B, 每条 2KB 连续) 直接写入 LDS, 不占用 VGPR; 3 级 LDS 环形, 每个 chunk 一次 CTA 屏障。
//  K/V 每个 chunk 只读一次, 由 4 个 warp 共享 (原两个 Q tile 线程块各读一遍)。
//  lane 映射与页内切分模式相同: S 第 t 个 16-token tile 的第 e 个元素对应 token 16t + g + 4e (g = lane >> 4)。
//  每个 warp 持有完整的行, 无 warp 间归约; partial 输出/max/sum 格式与常规路径一致, 由同一合并 kernel 合并。
// ---------------------------------------------------------------------------
// 按 32 位 LDS 地址读取 (地址空间 3): 地址 = 每 lane 基址 + 编译期常量, 常量部分进入 ds_read 立即数
template <typename T>
__device__ __forceinline__ T lds_load(uint32_t address)
{
    return *reinterpret_cast<const __attribute__((address_space(3))) T *>(static_cast<size_t>(address));
}

namespace rowsplit_d256 {
constexpr int CHUNK = 32;
constexpr int HALF_BYTES = CHUNK * 256;            // 一个 chunk 的 K (或 V) 字节数
constexpr int STAGE_BYTES = 2 * HALF_BYTES;
constexpr int STAGES = 2;
constexpr int RING_BYTES = STAGES * STAGE_BYTES;    // 32KB: 每 CU 两个线程块
// partition 数: 每个 partition 至少 MIN_CHUNKS 个 chunk, 不超过预算 (线程块按 chunk 顺序处理整段)
constexpr int MIN_CHUNKS = 2;
// 预算按每 CU 两个线程块给出; 长度不足 LONG_CHUNKS 个 chunk 时只用一半 (每 CU 一个线程块): 中等长度下更多 partition
// 只增加合并开销, 长序列才用第二个线程块填补等待
constexpr int LONG_CHUNKS = 768;
__device__ __forceinline__ int num_partitions(int valid_seqlen_kv, int budget)
{
    const int chunks = DIVIDE_ROUND_UP(valid_seqlen_kv, CHUNK);
    const int cap = chunks >= LONG_CHUNKS ? budget : MAX_T(1, budget / 2);
    return MAX_T(1, MIN_T(cap, chunks / MIN_CHUNKS));
}
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
// matrix_load_64x32_b8 (行跨度 64B) 的 LDS 布局 (gfx92a 实测): 每 2KB 段内第 b 个 64B 块存源第
// r = b ^ ((b >> 3) & 1) 行, 块内第 s 个 16B 段存源段 s ^ ((r >> 1) & 3)。o 为 16B 对齐的源字节偏移。
__device__ __forceinline__ int lds_offset(int o)
{
    const int r = (o >> 6) & 31;
    return (o & ~2047) + ((r ^ ((r >> 3) & 1)) << 6) + ((((o >> 4) & 3) ^ ((r >> 1) & 3)) << 4);
}
__device__ __forceinline__ void issue_2kb(const uint8_t *src, int lds_address)
{
    const uint64_t a = reinterpret_cast<uint64_t>(src) & 0x0000ffffffffffffULL;
    u32x4 r;
    r[0] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(a));
    r[1] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(a >> 32));
    r[2] = 64;
    r[3] = 0;
    const int dst = __builtin_amdgcn_readfirstlane(lds_address);
    asm volatile("s_nop 4\n\tmatrix_load_64x32_b8 %0, %1, moffset:0 t lds" :: "s"(r), "s"(dst) : "memory");
}
}  // namespace rowsplit_d256

constexpr int DIRECT_COMBINE_MAX_GROUPS = 4096;
constexpr int DIRECT_COMBINE_COUNTER_STRIDE = 16;   // 每组计数器独占 64 字节
__device__ __forceinline__ void direct_combine_arrive(unsigned int *counter);
template <typename scalar_t, int HEAD_SIZE_V, bool REPLICATE_META = false>
__device__ __forceinline__ void direct_combine_head_in_kernel(unsigned int *counter, int num_partitions, int num_combiners,
                                                              scalar_t *out, scalar_t *out_tmp, int head, int alloc_partitions,
                                                              char *shared_mem);

// 行切分 kernel 在 partition 维追加的合并线程块 (计数器非空时才有; 本组第 partition_idx - balanced_seq_len_tile 行):
// 等本组的 partial 全部写出后合并一个 head 并返回。宏: 与原来写在各 kernel 中的代码相同
#define PA_ROWSPLIT_COMBINE_CTA(REPLICATE_META) \
if (combine_counter != nullptr && partition_idx >= balanced_seq_len_tile) {                                                             \
    if (num_partitions > 1) {                                                                                                           \
        const int rows_all = num_heads / num_kv_heads;                                                                                  \
        const int q_heads_per_kv_head = rows_all / mtp;                                                                                 \
        const int local_row = partition_idx - balanced_seq_len_tile;                                                                    \
        if (local_row >= rows_all) return;                                                                                              \
        const int head = batch_idx * num_heads + q_heads_per_kv_head * kv_head_idx                                                      \
                         + (local_row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + local_row % q_heads_per_kv_head;  \
        direct_combine_head_in_kernel<scalar_t, HEAD_SIZE, REPLICATE_META>(combine_counter, num_partitions, rows_all, out, out_tmp, head,               \
                                                           balanced_seq_len_tile, shared_mem);                                          \
    }                                                                                                                                   \
    return;                                                                                                                             \
}

// 行切分 kernel 每 lane 的 LDS 读取基址 k_base/v_base: RING (本轮 stage 的环形区地址) + K 每个 64 维段、V 每个 4-token 组的
// 偏移; 空 asm 使基址留在寄存器中, 循环内只加编译期常量 (16-token tile 偏移)。宏: 与原来写在各 kernel 中的代码相同
#define PA_ROWSPLIT_LDS_BASES(RING) \
{                                                                         \
    const uint32_t ring = RING;                                           \
    _Pragma("unroll")                                                     \
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {                            \
        k_base[h] = ring + lds_offset(n * HEAD_SIZE + h * 64 + g * 16);   \
        asm volatile("" : "+v"(k_base[h]));                               \
    }                                                                     \
    _Pragma("unroll")                                                     \
    for (int jj = 0; jj < 4; ++jj) {                                      \
        const int o = (g + 4 * jj) * HEAD_SIZE + n * (HEAD_SIZE / 16);    \
        v_base[jj] = ring + HALF_BYTES + lds_offset(o & ~15) + (o & 15);  \
        asm volatile("" : "+v"(v_base[jj]));                              \
    }                                                                     \
}

// 行切分 kernel 处理一个 chunk 的 NT_ 个 16-token tile (本 warp 16 行, Q 片段 QREG 在寄存器中; 第 t 个 tile 的第 e 个元素
// 对应 token token_base + 16t + g + 4e): QK、尾部 (超出有效长度的 token 的 V 可能为任意字节, 置零避免 0 * NaN) 与因果掩码
// (MTP_)、online softmax (FIRST 为首轮) 与 PV, 结束后 k_base/v_base 切换到另一级 stage (2 级环形)。
// 用宏而非函数: 展开后与原来写在各 kernel 中的代码相同 (改成函数后寄存器分配会变); 使用调用处的 token_base、k_base、
// v_base、stage_step、x8_t/x4_t/is_fp16、valid_seqlen_kv、mtp、row、rows_all、q_heads_per_kv_head、g、scale_log2、
// scores_max/scores_sum/acc_o
#define PA_ROWSPLIT_CHUNK(NT_, MTP_, QREG, FIRST) \
{                                                                                                                        \
    float4_t s_reg[1][NT_];                                                                                              \
    _Pragma("unroll")                                                                                                    \
    for (int t = 0; t < NT_; ++t) s_reg[0][t] = {0.0f, 0.0f, 0.0f, 0.0f};                                                \
    _Pragma("unroll")                                                                                                    \
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {                                                                           \
        u32x4 k_raw[NT_];                                                                                                \
        _Pragma("unroll")                                                                                                \
        for (int t = 0; t < NT_; ++t) {                                                                                  \
            k_raw[t] = lds_load<u32x4>(k_base[h] + (16 * t * HEAD_SIZE));                                                \
        }                                                                                                                \
        _Pragma("unroll")                                                                                                \
        for (int sub = 0; sub < 4; ++sub) {                                                                              \
            const x8_t q8 = __builtin_bit_cast(x8_t, QREG[h][sub / 2]);                                                  \
            const x4_t q_v4 = (sub % 2 == 0) ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)                               \
                                             : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);                              \
            _Pragma("unroll")                                                                                            \
            for (int t = 0; t < NT_; ++t) {                                                                              \
                builtin_amdgcn_mmac<x4_t, is_fp16>(q_v4, decode_e5m2x4_b16<x4_t, is_fp16>(k_raw[t][sub]), s_reg[0][t]);  \
            }                                                                                                            \
        }                                                                                                                \
    }                                                                                                                    \
    const bool tail = token_base + 16 * NT_ > valid_seqlen_kv;                                                           \
    if (tail || (MTP_ && token_base + 16 * NT_ - 1 + mtp > valid_seqlen_kv)) {                                           \
        const int causal_offset = mtp - row / q_heads_per_kv_head;                                                       \
        _Pragma("unroll")                                                                                                \
        for (int t = 0; t < NT_; ++t) {                                                                                  \
            _Pragma("unroll")                                                                                            \
            for (int e = 0; e < 4; ++e) {                                                                                \
                const int token = token_base + 16 * t + g + 4 * e;                                                       \
                if (token >= valid_seqlen_kv ||                                                                          \
                    (MTP_ && row < rows_all && token + causal_offset > valid_seqlen_kv)) {                               \
                    (s_reg[0][t])[e] = -INFINITY;                                                                        \
                }                                                                                                        \
            }                                                                                                            \
        }                                                                                                                \
    }                                                                                                                    \
    direct_softmax<1, HEAD_SIZE, NT_, true>(s_reg, scores_max, scores_sum, acc_o, scale_log2, FIRST);                    \
    _Pragma("unroll")                                                                                                    \
    for (int t = 0; t < NT_; ++t) {                                                                                      \
        const half4_v p_reg[1] = {{(_Float16)(s_reg[0][t])[0], (_Float16)(s_reg[0][t])[1],                               \
                                   (_Float16)(s_reg[0][t])[2], (_Float16)(s_reg[0][t])[3]}};                             \
        uint32_t v_raw[4][HEAD_SIZE / 64];                                                                               \
        _Pragma("unroll")                                                                                                \
        for (int j = 0; j < 4; ++j) {                                                                                    \
            const uint32_t src = v_base[j] + (16 * t * HEAD_SIZE);                                                       \
            if constexpr (HEAD_SIZE == 256) {                                                                            \
                const u32x4 value = lds_load<u32x4>(src);                                                                \
                _Pragma("unroll")                                                                                        \
                for (int w = 0; w < 4; ++w) v_raw[j][w] = value[w];                                                      \
            } else {                                                                                                     \
                using u32x2 = uint32_t __attribute__((ext_vector_type(2)));                                              \
                const u32x2 value = lds_load<u32x2>(src);                                                                \
                v_raw[j][0] = value[0];                                                                                  \
                v_raw[j][1] = value[1];                                                                                  \
            }                                                                                                            \
        }                                                                                                                \
        if (tail) {                                                                                                      \
            _Pragma("unroll")                                                                                            \
            for (int j = 0; j < 4; ++j) {                                                                                \
                if (token_base + 16 * t + g + 4 * j >= valid_seqlen_kv) {                                                \
                    _Pragma("unroll")                                                                                    \
                    for (int w = 0; w < HEAD_SIZE / 64; ++w) v_raw[j][w] = 0;                                            \
                }                                                                                                        \
            }                                                                                                            \
        }                                                                                                                \
        pv_fp8_tile_f16<1, HEAD_SIZE>(v_raw, p_reg, acc_o);                                                              \
    }                                                                                                                    \
    _Pragma("unroll")                                                                                                    \
    for (int h = 0; h < HEAD_SIZE / 64; ++h) k_base[h] += stage_step;                                                    \
    _Pragma("unroll")                                                                                                    \
    for (int jj = 0; jj < 4; ++jj) v_base[jj] += stage_step;                                                             \
    stage_step = -stage_step;                                                                                            \
}

// 行切分 kernel 写出本 lane 的一行 (第 row 行; 每个 lane 持有通道 CH * (g + 4e) + q, q = 0..CH-1, 乘 INV 归一化): 只有一个
// partition 时直接写 out, 否则写 partial 输出与 max/sum (max 乘 scale_log2, 含 Q 缩放修正)。宏: 与原来写在各 kernel 中的代码相同
#define PA_ROWSPLIT_STORE_ROW(INV, SUM, MAX) \
{                                                                                                                              \
    const int head_q_index = q_heads_per_kv_head * kv_head_idx;                                                                \
    const int out_row = (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;        \
    const int head = batch_idx * num_heads + head_q_index + out_row;                                                           \
    scalar_t *dst = split ? out_tmp + out_tmp_offset + head * (balanced_seq_len_tile * HEAD_SIZE) + partition_idx * HEAD_SIZE  \
                          : out + head * HEAD_SIZE;                                                                            \
    const float inv_sum_ = INV;                                                                                                \
    _Pragma("unroll")                                                                                                          \
    for (int e = 0; e < 4; ++e) {                                                                                              \
        _Pragma("unroll")                                                                                                      \
        for (int q4 = 0; q4 < HEAD_SIZE / 64; ++q4) {                                                                          \
            const scalarx4_t value = {from_float<scalar_t>((acc_o[0][4 * q4 + 0])[e] * inv_sum_),                              \
                                      from_float<scalar_t>((acc_o[0][4 * q4 + 1])[e] * inv_sum_),                              \
                                      from_float<scalar_t>((acc_o[0][4 * q4 + 2])[e] * inv_sum_),                              \
                                      from_float<scalar_t>((acc_o[0][4 * q4 + 3])[e] * inv_sum_)};                             \
            *reinterpret_cast<scalarx4_t *>(dst + (HEAD_SIZE / 16) * (g + 4 * e) + 4 * q4) = value;                            \
        }                                                                                                                      \
    }                                                                                                                          \
    if (split && g == 0) {                                                                                                     \
        const int offset = head * balanced_seq_len_tile + partition_idx;                                                       \
        reinterpret_cast<float *>(out_tmp)[offset] = SUM;                                                                      \
        reinterpret_cast<float *>(out_tmp + max_tmp_offset)[offset] = MAX * scale_log2;                                        \
    }                                                                                                                          \
}

// 行切分 kernel 的 Q 范围 (BF16 位型 |x| 的最大/最小非零值, 每个 lane 已扫描自己的 Q 片段到 max_bits/min_bits) 在线程块内
// 归约 (REDUCE 为 8 个 uint32 的 LDS 暂存; BARRIER_FIRST: 暂存可能与 Q 区重叠, 先等所有 warp 读完 Q), 动态范围过宽或含
// Inf/NaN 时 q_bf16 = true (保留 BF16 QK); 规则与 commit_q_stage_fp16 相同。以下三个宏与原来写在各 kernel 中的代码相同
#define PA_Q_RANGE_REDUCE(BARRIER_FIRST, REDUCE) \
{                                                                                                         \
    _Pragma("unroll")                                                                                     \
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {                                                 \
        max_bits = MAX_T(max_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(max_bits), mask)));  \
        min_bits = MIN_T(min_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(min_bits), mask)));  \
    }                                                                                                     \
    if (BARRIER_FIRST) lds_barrier();                                                                     \
    uint32_t *q_reduce = reinterpret_cast<uint32_t *>(REDUCE);                                            \
    if (lane_id == 0) {                                                                                   \
        q_reduce[warp_id] = max_bits;                                                                     \
        q_reduce[4 + warp_id] = min_bits;                                                                 \
    }                                                                                                     \
    lds_barrier();                                                                                        \
    _Pragma("unroll")                                                                                     \
    for (int w = 0; w < 4; ++w) {                                                                         \
        max_bits = MAX_T(max_bits, q_reduce[w]);                                                          \
        min_bits = MIN_T(min_bits, q_reduce[4 + w]);                                                      \
    }                                                                                                     \
    q_bf16 = max_bits != 0 && ((max_bits >> 7) == 255 || (max_bits >> 7) - (min_bits >> 7) > 31);         \
}
// 2 的幂缩放: 使 max|Q| 落在 [2^14, 2^15) (全零 Q 不缩放), q_mul 乘到 Q 上, q_scale 为其倒数 (softmax 缩放的修正倍数)
#define PA_Q_FP16_SCALE() \
const int exponent = static_cast<int>(max_bits >> 7);                                     \
const int shift = max_bits == 0 ? 0 : MIN_T(MAX_T(exponent - 127 - 14, -126), 113);       \
const float q_mul = __builtin_bit_cast(float, static_cast<uint32_t>(127 - shift) << 23);  \
q_scale = __builtin_bit_cast(float, static_cast<uint32_t>(127 + shift) << 23);
// 寄存器中的 BF16 Q 片段 QREG[HEAD_SIZE / 64][2] (u32x4) 乘 q_mul 后转为 FP16
#define PA_Q_REG_TO_FP16(QREG) \
_Pragma("unroll")                                                                                           \
for (int h = 0; h < HEAD_SIZE / 64; ++h) {                                                                  \
    _Pragma("unroll")                                                                                       \
    for (int k = 0; k < 2; ++k) {                                                                           \
        _Pragma("unroll")                                                                                   \
        for (int w = 0; w < 4; ++w) {                                                                       \
            const uint32_t v = QREG[h][k][w];                                                               \
            const _Float16 lo = static_cast<_Float16>(__builtin_bit_cast(float, v << 16) * q_mul);          \
            const _Float16 hi = static_cast<_Float16>(__builtin_bit_cast(float, v & 0xffff0000u) * q_mul);  \
            QREG[h][k][w] = static_cast<uint32_t>(__builtin_bit_cast(uint16_t, lo)) |                       \
                            (static_cast<uint32_t>(__builtin_bit_cast(uint16_t, hi)) << 16);                \
        }                                                                                                   \
    }                                                                                                       \
}

template <typename scalar_t, int ROWS, bool MTP_ENABLE>
__global__ __launch_bounds__(256, 2) void paged_attention_kernel_rowsplit_d256(
    scalar_t *__restrict__ out, scalar_t *__restrict__ out_tmp, const scalar_t *__restrict__ q,
    const uint8_t *__restrict__ k_cache, const uint8_t *__restrict__ v_cache,
    const int num_heads, const int num_kv_heads, const int *__restrict__ block_tables, const int *__restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int q_stride, const int k_block_stride, const int v_block_stride,
    const int balanced_seq_len_tile, const int mtp, unsigned int *__restrict__ combine_counters)
{
    // host 只对 BF16 Q 选择本路径; 其他实例为空函数体 (分发 lambda 中的 if constexpr 仍会实例化它们)
    if constexpr (!std::is_same<scalar_t, uint16_t>::value) {
        return;
    } else {
    using namespace rowsplit_d256;
    constexpr int HEAD_SIZE = 256, NUM_THREADS = 256, Q_LDS_PAD = 0, BLOCK_SIZE = 64;
    using scalarx8_t = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    using scalarx4_t = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    extern __shared__ char shared_mem[];
    const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const int num_partitions = rowsplit_d256::num_partitions(valid_seqlen_kv, balanced_seq_len_tile);
    unsigned int *combine_counter = combine_counters != nullptr
        ? combine_counters + (batch_idx * num_kv_heads + kv_head_idx) * DIRECT_COMBINE_COUNTER_STRIDE : nullptr;
    PA_ROWSPLIT_COMBINE_CTA(false);
    if (partition_idx >= num_partitions) return;

    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int n = lane_id & 15;
    const int g = lane_id >> 4;
    const int rows_all = num_heads / num_kv_heads;
    const int q_heads_per_kv_head = rows_all / mtp;
    const int row = warp_id * 16 + n;
    const bool warp_active = warp_id * 16 < rows_all;
    constexpr float scale_softmax_log2 = 0.0625f * M_LOG2E;

    // Q: global -> LDS (按线程块 max|Q| 缩放为 FP16, 动态范围过宽时保留 BF16), 再读入寄存器
    QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, ROWS> q_stage;
    issue_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, ROWS>(
        q_stage, q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, 0, rows_all);
    // Q 区 (无行填充, 64 行 32KB) 与环形区、归约区重叠, 总 LDS 32KB (每 CU 两个线程块)
    uint32_t *reduce = reinterpret_cast<uint32_t *>(shared_mem);
    const float committed_q_scale = commit_q_stage_fp16<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, ROWS, Q_LDS_PAD, true>(
        q_stage, reinterpret_cast<_Float16 *>(shared_mem), rows_all, reduce);
    const bool q_bf16 = committed_q_scale < 0.0f;
    const float scale_log2 = scale_softmax_log2 * fabsf(committed_q_scale);
    u32x4 q_reg[HEAD_SIZE / 64][2];
    #pragma unroll
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            q_reg[h][k] = row < rows_all
                ? *reinterpret_cast<const u32x4 *>(shared_mem + (row * (HEAD_SIZE + Q_LDS_PAD) + h * 64 + g * 16 + k * 8) * sizeof(scalar_t))
                : u32x4{0, 0, 0, 0};
        }
    }
    // 环形区与 Q 区重叠: 所有 warp 读完 Q 后才发射首批 K/V
    lds_barrier();

    const int chunks = DIVIDE_ROUND_UP(valid_seqlen_kv, CHUNK);
    const int my_chunks = partition_idx < chunks ? DIVIDE_ROUND_UP(chunks - partition_idx, num_partitions) : 0;
    const int *block_table = block_tables + batch_idx * max_num_blocks_per_seq;
    const uint8_t *k_head = k_cache + kv_head_idx * (BLOCK_SIZE * HEAD_SIZE);
    const uint8_t *v_head = v_cache + kv_head_idx * (BLOCK_SIZE * HEAD_SIZE);
    const int ring_address = static_cast<int>(reinterpret_cast<size_t>(shared_mem));
    // 本线程块的第 i 个 chunk 写入 stage i % STAGES; 每个 warp 发射 K、V 各 2KB
    auto issue_chunk = [&](int i) {
        const int c = partition_idx + i * num_partitions;
        const int64_t physical_block_number = static_cast<int64_t>(block_table[c >> 1]);
        const int half = (c & 1) * HALF_BYTES + warp_id * 2048;
        const int stage = ring_address + (i % STAGES) * STAGE_BYTES + warp_id * 2048;
        issue_2kb(k_head + physical_block_number * k_block_stride + half, stage);
        issue_2kb(v_head + physical_block_number * v_block_stride + half, stage + HALF_BYTES);
    };
    // 开头所有槽位都空闲: 预取 STAGES 个 chunk; 之后第 i 轮 (i >= 1) 在屏障后向第 i - 1 个 chunk 释放的槽位发射第 i + STAGES - 1 个
    #pragma unroll
    for (int i = 0; i < STAGES; ++i) {
        if (i < my_chunks) issue_chunk(i);
    }

    float scores_max[1];
    float scores_sum[1];
    float4_t acc_o[1][HEAD_SIZE / 16];
    attention_initialize<1, HEAD_SIZE>(scores_max, scores_sum, acc_o);

    auto run_chunks = [&](auto query_scalar) {
        using qs_t = decltype(query_scalar);
        constexpr bool is_fp16 = std::is_same<qs_t, _Float16>::value;
        using x8_t = __attribute__((__vector_size__(8 * sizeof(qs_t)))) qs_t;
        using x4_t = __attribute__((__vector_size__(4 * sizeof(qs_t)))) qs_t;
        // LDS 读取基址 (本轮 stage): 每 lane 的 K (每个 64 维段) 与 V (每个 4-token 组) 偏移 + 环形区基址;
        // 每轮结束加减 STAGE_BYTES, 循环内只加编译期常量 (16-token tile 偏移)
        static_assert(STAGES == 2, "base toggling assumes a two-stage ring");
        uint32_t k_base[HEAD_SIZE / 64], v_base[4];
        int stage_step = STAGE_BYTES;
        PA_ROWSPLIT_LDS_BASES(static_cast<uint32_t>(reinterpret_cast<size_t>(shared_mem)));
        for (int i = 0; i < my_chunks; ++i) {
            // 自己发射的第 i 个 chunk 已完成, 屏障后全体完成, 且所有 warp 已用完第 i - 1 个 chunk 所在的 stage
            // (第 i 个 chunk 之后已发射的 chunk 数: 第 0 轮为 min(STAGES - 1, my_chunks - 1), 之后为 min(STAGES - 2, my_chunks - 1 - i))
            static_assert(STAGES == 2, "wait counts below assume a two-stage ring");
            if (i == 0 && my_chunks > 1) {
                asm volatile("s_waitcnt vmcnt(2)\n\ts_barrier" ::: "memory");
            } else {
                asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier" ::: "memory");
            }
            if (i >= 1 && i + STAGES - 1 < my_chunks) issue_chunk(i + STAGES - 1);
            if (!warp_active) continue;
            const int token_base = (partition_idx + i * num_partitions) * CHUNK;
            PA_ROWSPLIT_CHUNK(2, MTP_ENABLE, q_reg, i == 0);
        }
    };
    if (q_bf16) run_chunks(uint16_t{});
    else run_chunks(_Float16{});
    direct_softmax_finish<1>(scores_sum);
    const bool split = num_partitions > 1;
    if (warp_active && row < rows_all) {
    PA_ROWSPLIT_STORE_ROW(1.0f / (scores_sum[0] + 1e-6f), scores_sum[0], scores_max[0]);
    }
    // 所有线程到达: partial 输出与 max/sum 写出后计数
    if (split && combine_counter != nullptr) direct_combine_arrive(combine_counter);
    }
}

// ---- 直接加载路径的 epilogue ----
//  各 warp 将行 max/sum 写入独立于 Q 与归约区的 LDS, 一次同步后各自求全局 max/sum 并缩放 acc_o
//  (同时确认所有 warp 已结束读取 Q); 随后按 16x64 分块在 LDS 中累加 4 个 warp 的结果。
template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS>
struct DirectEpilogueShape {
    static constexpr int TILE_FLOATS = 16 * (64 + 4);                  // 每个 warp 的 16x64 分块, 行填充 4
    static constexpr int RED_BYTES = NUM_WARPS * TILE_FLOATS * static_cast<int>(sizeof(float));
    static constexpr int meta_offset(int q_bytes)
    {
        return (((q_bytes > RED_BYTES ? q_bytes : RED_BYTES) + 15) / 16) * 16;
    }
    static constexpr int smem_bytes(int q_bytes, int max_rows)
    {
        return meta_offset(q_bytes) + max_rows * NUM_WARPS * 2 * static_cast<int>(sizeof(float));
    }
};

#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS, int Q_BYTES, bool NATURAL = false, typename StoreO, typename StoreMeta>
#else
template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS, int Q_BYTES, typename StoreO, typename StoreMeta>
#endif
__device__ __forceinline__ void direct_epilogue(float scores_max[Mloop], float scores_sum[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], char *shared_mem,
                                                int rows, int warp_id, float scale_softmax_log2,
                                                StoreO &store_o, StoreMeta &store_meta)
{
    using Shape = DirectEpilogueShape<Mloop, HEAD_SIZE_V, NUM_WARPS>;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int acc_row = lane_id & 15;
    const int acc_col = lane_id >> 4;
    float *meta = reinterpret_cast<float *>(shared_mem + Shape::meta_offset(Q_BYTES));
    float *red = reinterpret_cast<float *>(shared_mem);
    using float2_v = float __attribute__((ext_vector_type(2)));

    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int row = m * 16 + acc_row;
        if (row < rows && acc_col == 0) {
            *reinterpret_cast<float2_v *>(meta + (row * NUM_WARPS + warp_id) * 2) = {scores_max[m], scores_sum[m]};
        }
    }
    // 同时保证所有 warp 已结束对 Q 的读取, 归约区可以复用 Q 的 LDS
    lds_barrier();

    float row_max[Mloop];
    float row_sum[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int row = m * 16 + acc_row;
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
        if (row < rows) {
            float2_v values[NUM_WARPS];
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) values[w] = *reinterpret_cast<const float2_v *>(meta + (row * NUM_WARPS + w) * 2);
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) row_max[m] = MAX_T(row_max[m], values[w][0]);
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                row_sum[m] += values[w][1] * direct_exp_difference(values[w][0], row_max[m], scale_softmax_log2);
            }
            const float scale = direct_exp_difference(scores_max[m], row_max[m], scale_softmax_log2) / (row_sum[m] + 1e-6f);
            #pragma unroll
            for (int i = 0; i < HEAD_SIZE_V / 16; ++i) acc_o[m][i] *= scale;
        }
    }

    constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
    #pragma unroll
    for (int tile_idx = 0; tile_idx < num_16x64_tiles; ++tile_idx) {
        const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
        const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
        if (tile_idx > 0) lds_barrier();
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
        if constexpr (NATURAL) {
            store_direct_v_acc_group_natural<Mloop, HEAD_SIZE_V>(red + warp_id * Shape::TILE_FLOATS + acc_row * (64 + 4),
                                                                 acc_o, tile_idx_row, tile_idx_col, acc_col);
        } else {
            store_direct_v_acc_group<Mloop, HEAD_SIZE_V>(red + warp_id * Shape::TILE_FLOATS + acc_row * (64 + 4),
                                                         acc_o, tile_idx_row, tile_idx_col, acc_col);
        }
#else
        store_direct_v_acc_group<Mloop, HEAD_SIZE_V>(red + warp_id * Shape::TILE_FLOATS + acc_row * (64 + 4),
                                                     acc_o, tile_idx_row, tile_idx_col, acc_col);
#endif
        lds_barrier();
        const int out_r2g_row = threadIdx.x >> 4;
        const int out_r2g_col = threadIdx.x & 15;
        const int row = tile_idx_row * 16 + out_r2g_row;
        if (row < rows) {
            float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                sums += *reinterpret_cast<const float4_t *>(red + w * Shape::TILE_FLOATS + out_r2g_row * (64 + 4) + out_r2g_col * 4);
            }
            store_o(row, tile_idx_col * 64 + out_r2g_col * 4, sums);
        }
    }

    if (warp_id == 0 && acc_col == 0) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int row = m * 16 + acc_row;
            if (row < rows) store_meta(row, row_max[m], row_sum[m]);
        }
    }
}

// ---------------------------------------------------------------------------
// 直接加载路径的 partition 合并:
//  每个线程以 16 字节读取一个 partition 输出行的一段, 所需的全部 partial O 与 max/sum
//  一次性发射 (按加载数选择直线化实例, 不做冗余读取), 访存延迟只暴露一次。
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_SIZE>
struct CombineDirectShape {
    static constexpr int VEC = 8;                         // 每次加载 8 个 b16
    static constexpr int TPR = HEAD_SIZE / VEC;           // 每个 partition 行的线程数
    static constexpr int ROWS = 256 / TPR;                // 每轮覆盖的 partition 数
    static constexpr int MAX_LOADS = HEAD_SIZE > 128 ? 32 : 16;
    static constexpr int META_SLOTS = 2;                  // 每线程直线化处理的 max/sum 个数
    static constexpr int RED_FLOATS = ROWS * HEAD_SIZE;
};

template <typename scalar_t, int HEAD_SIZE, int NL, bool GROUPED = false>
__device__ __forceinline__ void combine_direct_body(scalar_t *__restrict__ out_head,
                                                    const scalar_t *__restrict__ tmp_out,
                                                    const float *__restrict__ exp_sums,
                                                    const float *__restrict__ max_logits,
                                                    int num_partitions, float *smem, bool write_out = true)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    constexpr int ROWS = Shape::ROWS;
    constexpr int TPR = Shape::TPR;
    constexpr int VEC = Shape::VEC;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const bool active = tid < ROWS * TPR;
    const int row = active ? tid / TPR : 0;
    const int col = active ? (tid % TPR) * VEC : 0;
    float *smem_red = smem;
    float *smem_scalar = smem_red + Shape::RED_FLOATS;
    float *smem_weight = smem_scalar + 8;

    // max/sum 与 partial O 同时发射; 越界项读取最后一个有效 partition, 随后以权重 0 屏蔽
    float part_max[Shape::META_SLOTS];
    float part_sum[Shape::META_SLOTS];
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        const int i = MIN_T(tid + j * 256, num_partitions - 1);
        part_max[j] = max_logits[i];
        part_sum[j] = exp_sums[i];
    }
    u32x4 values[NL];
    if constexpr (!GROUPED) {
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int p = MIN_T(row + k * ROWS, num_partitions - 1);
            values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
        }
    }

    float local_max = -INFINITY;
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        if (tid + j * 256 < num_partitions) local_max = MAX_T(local_max, part_max[j]);
    }
    for (int i = tid + Shape::META_SLOTS * 256; i < num_partitions; i += 256) local_max = MAX_T(local_max, max_logits[i]);
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
    if (lane == 0) smem_scalar[warp] = local_max;
    lds_barrier();
    const float global_max = MAX_T(MAX_T(smem_scalar[0], smem_scalar[1]), MAX_T(smem_scalar[2], smem_scalar[3]));

    float local_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        const int i = tid + j * 256;
        if (i < num_partitions) {
            const float weight = part_sum[j] > 0.0f ? part_sum[j] * __builtin_amdgcn_exp2f(part_max[j] - global_max) : 0.0f;
            smem_weight[i] = weight;
            local_sum += weight;
        }
    }
    for (int i = tid + Shape::META_SLOTS * 256; i < num_partitions; i += 256) {
        const float s = exp_sums[i];
        const float weight = s > 0.0f ? s * __builtin_amdgcn_exp2f(max_logits[i] - global_max) : 0.0f;
        smem_weight[i] = weight;
        local_sum += weight;
    }
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) local_sum += __shfl_xor(local_sum, mask);
    if (lane == 0) smem_scalar[4 + warp] = local_sum;
    lds_barrier();
    const float inv_sum = __fdividef(1.0f, (smem_scalar[4] + smem_scalar[6]) + (smem_scalar[5] + smem_scalar[7]) + 1e-6f);

    float acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.0f;
    // GROUPED: partition 数超过直线化实例覆盖范围时, 按 NL 次加载为一组循环处理
    const int groups = GROUPED ? DIVIDE_ROUND_UP(num_partitions, NL * ROWS) : 1;
    for (int group = 0; group < groups; ++group) {
        if constexpr (GROUPED) {
            #pragma unroll
            for (int k = 0; k < NL; ++k) {
                const int p = MIN_T(row + (group * NL + k) * ROWS, num_partitions - 1);
                values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
            }
        }
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int p = row + (group * NL + k) * ROWS;
            const float weight = (active && p < num_partitions) ? smem_weight[p] : 0.0f;
            const scalar_t *v = reinterpret_cast<const scalar_t *>(&values[k]);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) acc[e] += to_float(v[e]) * weight;
        }
    }
    if (active) {
        #pragma unroll
        for (int e = 0; e < VEC; e += 4) {
            *reinterpret_cast<float4_t *>(smem_red + row * HEAD_SIZE + col + e) = {acc[e], acc[e + 1], acc[e + 2], acc[e + 3]};
        }
    }
    lds_barrier();
    for (int c = tid; write_out && c < HEAD_SIZE; c += 256) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) sum += smem_red[r * HEAD_SIZE + c];
        scalar_t converted;
        from_float(converted, sum * inv_sum);
        out_head[c] = converted;
    }
}

// 按 partition 数选择直线化实例合并一个 head 的 partial; write_out 为 false 时只执行不写出 (用于预热指令缓存)
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void combine_direct_head(scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
                                                    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
                                                    int num_partitions, float *smem, bool write_out = true)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    const int loads = DIVIDE_ROUND_UP(num_partitions, Shape::ROWS);
    if (loads <= 1) {
        combine_direct_body<scalar_t, HEAD_SIZE, 1>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 2) {
        combine_direct_body<scalar_t, HEAD_SIZE, 2>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 4) {
        combine_direct_body<scalar_t, HEAD_SIZE, 4>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 8) {
        combine_direct_body<scalar_t, HEAD_SIZE, 8>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 16) {
        combine_direct_body<scalar_t, HEAD_SIZE, 16>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (Shape::MAX_LOADS == 32 && loads <= 32) {
        combine_direct_body<scalar_t, HEAD_SIZE, Shape::MAX_LOADS>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else {
        combine_direct_body<scalar_t, HEAD_SIZE, Shape::MAX_LOADS, true>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    }
}

// Each wave independently prepares the same <=64 partition weights. The
// original eight-row partial-output mapping and final LDS reduction stay intact.
template <typename scalar_t, int HEAD_SIZE, int NL>
__device__ __forceinline__ void combine_direct_replicated_body(
    scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
    int num_partitions, float *smem)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    constexpr int ROWS = Shape::ROWS, TPR = Shape::TPR, VEC = Shape::VEC;
    static_assert(HEAD_SIZE == 256 && NL * ROWS <= WARP_SIZE, "replicated metadata geometry");
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int row = tid / TPR;
    const int col = (tid % TPR) * VEC;

    const int i = MIN_T(lane, num_partitions - 1);
    const float part_max = max_logits[i];
    const float part_sum = exp_sums[i];
    u32x4 values[NL];
    #pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int p = MIN_T(row + k * ROWS, num_partitions - 1);
        values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
    }

    float local_max = lane < num_partitions ? MAX_T(-INFINITY, part_max) : -INFINITY;
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
    // Match the old metadata publication by lane 0, including the empty waves.
    const float wave_max = __builtin_bit_cast(float,
        __builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t, local_max)));
    const float global_max = MAX_T(MAX_T(wave_max, -INFINITY), MAX_T(-INFINITY, -INFINITY));
    float weight_lane = 0.0f;
    if (lane < num_partitions)
        weight_lane = part_sum > 0.0f ? part_sum * __builtin_amdgcn_exp2f(part_max - global_max) : 0.0f;
    float local_sum = weight_lane;
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        local_sum += __shfl_xor(local_sum, mask);
    float ordered_sum = __builtin_bit_cast(float,
        __builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t, local_sum)));
    // Preserve the original reduction/epsilon boundary under fast math.
    asm volatile("" : "+v"(ordered_sum));
    const float inv_sum = __fdividef(1.0f, (ordered_sum + 0.0f) + (0.0f + 0.0f) + 1e-6f);

    float acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.0f;
    #pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int p = row + k * ROWS;
        const float weight = __shfl(weight_lane, p);
        const scalar_t *v = reinterpret_cast<const scalar_t *>(&values[k]);
        #pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] += to_float(v[e]) * weight;
    }
    #pragma unroll
    for (int e = 0; e < VEC; e += 4)
        *reinterpret_cast<float4_t *>(smem + row * HEAD_SIZE + col + e) =
            {acc[e], acc[e + 1], acc[e + 2], acc[e + 3]};
    lds_barrier();
    for (int c = tid; c < HEAD_SIZE; c += 256) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) sum += smem[r * HEAD_SIZE + c];
        scalar_t converted;
        from_float(converted, sum * inv_sum);
        out_head[c] = converted;
    }
}

template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void combine_direct_replicated_head(
    scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
    int num_partitions, float *smem)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    const int loads = DIVIDE_ROUND_UP(num_partitions, Shape::ROWS);
    if (loads <= 1)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 1>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else if (loads <= 2)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 2>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else if (loads <= 4)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 4>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 8>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
}

// 合并 kernel 的一个 head (blockIdx = (head, batch)): partial 输出与 max/sum 按 (batch, head) 每 STRIDE 个 partition 存放。
// 宏: 与原来写在各合并 kernel 中的代码相同
#define PA_COMBINE_PARTIALS(T, HEAD, STRIDE) \
{                                                                                                                                  \
    const int offset = (batch_idx * num_heads + head_idx) * STRIDE;                                                                \
    const float *exp_sums = reinterpret_cast<const float *>(out_tmp) + offset;                                                     \
    const float *max_logits = reinterpret_cast<const float *>(out_tmp + max_tmp_offset) + offset;                                  \
    const T *tmp_out = out_tmp + out_tmp_offset + offset * HEAD;                                                                   \
    T *out_head = out + (batch_idx * num_heads + head_idx) * HEAD;                                                                 \
    combine_direct_head<T, HEAD>(out_head, tmp_out, exp_sums, max_logits, num_partitions, reinterpret_cast<float *>(shared_mem));  \
}

// ---------------------------------------------------------------------------
// 直接加载路径在同一次启动内合并 partition:
//  grid 的 partition 维在 balanced_seq_len_tile 个 partition 线程块之后追加每组每行一个合并线程块
//  (与合并 kernel 一样每个 head 一个线程块)。partition 线程块写完 partial 后对组计数器加一;
//  合并线程块排在同组全部 partition 线程块之后; 此融合路径依赖目标设备的 grid 派发次序。
//  合并线程块等到所有 partial 写出后才读取, 避免预读未初始化的临时输出。
//  省去合并 kernel 的启动开销与 kernel 间隔; 计数器在每次调用结束时由最后一个合并线程块复位。
// ---------------------------------------------------------------------------

// 每个设备、执行流各自持有计数器; 独立流之间不共享合并状态。
// 首次调用在流上清零, 此后的调用由最后一个合并线程块复位。
static unsigned int *direct_combine_counters(int device, cudaStream_t stream)
{
    using Key = std::pair<int, uintptr_t>;
    static std::mutex lock;
    static std::map<Key, unsigned int *> counters;
    struct LastCounter { int device; uintptr_t stream; unsigned int *ptr; };
    static thread_local LastCounter last = {-1, 0, nullptr};
    const Key key(device, reinterpret_cast<uintptr_t>(stream));
    // Always check capture before the cache: graphs use the stateless optimized
    // merge even after a normal invocation on the same stream.
    hipStreamCaptureStatus capture_status;
    if (hipStreamIsCapturing(stream, &capture_status) != hipSuccess || capture_status != hipStreamCaptureStatusNone) return nullptr;
    // Repeated calls on one thread/stream need no global lock or map lookup.
    // The map owns the allocation for the process lifetime; stream changes
    // still go through the synchronized lookup below.
    if (last.ptr != nullptr && last.device == device && last.stream == key.second) return last.ptr;
    std::lock_guard<std::mutex> guard(lock);
    const auto found = counters.find(key);
    if (found != counters.end()) {
        last = {device, key.second, found->second};
        return found->second;
    }
    constexpr size_t bytes = sizeof(unsigned int) * DIRECT_COMBINE_MAX_GROUPS * DIRECT_COMBINE_COUNTER_STRIDE;
    unsigned int *ptr = nullptr;
    if (hipMalloc(&ptr, bytes) != hipSuccess) return nullptr;
    if (hipMemsetAsync(ptr, 0, bytes, stream) != hipSuccess) {
        (void)hipFree(ptr);
        return nullptr;
    }
    counters.emplace(key, ptr);
    last = {device, key.second, ptr};
    return ptr;
}

// partition 线程块: partial 输出与 max/sum 写出后到达组计数器
__device__ __forceinline__ void direct_combine_arrive(unsigned int *counter)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// 合并线程块: 等待组内 num_partitions 个 partition 线程块到达后合并一个 head; num_combiners 个合并线程块都
// 完成计数后由最后一个复位计数器
template <typename scalar_t, int HEAD_SIZE_V, bool REPLICATE_META>
__device__ __forceinline__ void direct_combine_head_in_kernel(unsigned int *counter, int num_partitions, int num_combiners,
                                                              scalar_t *out, scalar_t *out_tmp, int head, int alloc_partitions,
                                                              char *shared_mem)
{
    float *smem = reinterpret_cast<float *>(shared_mem);
    int *flag = reinterpret_cast<int *>(smem + CombineDirectShape<scalar_t, HEAD_SIZE_V>::RED_FLOATS + 8 + alloc_partitions);
    const int offset = head * alloc_partitions;
    const float *exp_sums = reinterpret_cast<const float *>(out_tmp) + offset;
    const float *max_logits = reinterpret_cast<const float *>(out_tmp + max_tmp_offset) + offset;
    const scalar_t *tmp_out = out_tmp + out_tmp_offset + static_cast<int64_t>(offset) * HEAD_SIZE_V;
    scalar_t *out_head = out + static_cast<int64_t>(head) * HEAD_SIZE_V;
    const unsigned int expected = static_cast<unsigned int>(num_partitions);
    if (threadIdx.x == 0) *flag = __hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) >= expected;
    __syncthreads();
    if (*flag == 0) {
        if (threadIdx.x == 0) {
            while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < expected) __builtin_amdgcn_s_sleep(4);
        }
        __syncthreads();
    }
    __threadfence();
    if constexpr (REPLICATE_META) {
        if (num_partitions <= WARP_SIZE)
            combine_direct_replicated_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
        else
            combine_direct_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, true);
    } else {
        combine_direct_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, true);
    }
    if (threadIdx.x == 0) {
        if (atomicAdd(counter, 1u) == expected + static_cast<unsigned int>(num_combiners) - 1) atomicExch(counter, 0u);
    }
}

// 行切分路径的合并: partition 数按 rowsplit_d256::num_partitions 计算, 其余与常规合并 kernel 相同
template <typename scalar_t, int HEAD_SIZE>
__global__ __launch_bounds__(256, 1) void paged_attention_combine_rowsplit_d256(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ out_tmp, const int *__restrict__ seq_lens,
    int balanced_seq_len_tile, int num_heads)
{
    extern __shared__ char shared_mem[];
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const int num_partitions = rowsplit_d256::num_partitions(valid_seqlen_kv, balanced_seq_len_tile);
    if (num_partitions == 1) return;
    PA_COMBINE_PARTIALS(scalar_t, HEAD_SIZE, balanced_seq_len_tile);
}

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int CTAS_PER_CU, bool LONG_SINGLE_STREAM, bool RESIDENT_CAP, bool ONE_PAGE_SLICE>
__global__ __launch_bounds__(256, 1) void paged_attention_combine_direct(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ out_tmp,
    const int* __restrict__ seq_lens,
    int balanced_seq_len_tile,
    int num_heads)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    extern __shared__ char shared_mem[];
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    const int num_partitions = get_direct_num_partitions<CTAS_PER_CU, LONG_SINGLE_STREAM, RESIDENT_CAP, ONE_PAGE_SLICE>(valid_num_block_pages, balanced_seq_len_tile);
    if (num_partitions == 1) {
        return;
    }
    PA_COMBINE_PARTIALS(scalar_t, HEAD_SIZE, balanced_seq_len_tile);
}

// 启动合并 kernel: partition 上限超过直线化实例覆盖范围时使用通用合并
template <typename scalar_t, int HEAD_SIZE_V, int BLOCK_SIZE, bool DIRECT, int CTAS_PER_CU, bool LONG_SINGLE_STREAM, bool RESIDENT_CAP, bool ONE_PAGE_SLICE>
void launch_paged_attention_combine(scalar_t *out, scalar_t *out_tmp, int *seq_lens, int balanced_seq_len_tile,
                                    int num_heads, int batch, cudaStream_t stream)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE_V>;
    const dim3 reduce_grid_size(num_heads, batch);
    const dim3 reduce_block_size(256);
    if (DIRECT) {
        const int smem_size = (Shape::RED_FLOATS + 8 + balanced_seq_len_tile) * sizeof(float);
        paged_attention_combine_direct<scalar_t, HEAD_SIZE_V, BLOCK_SIZE, CTAS_PER_CU, LONG_SINGLE_STREAM, RESIDENT_CAP, ONE_PAGE_SLICE>
            <<<reduce_grid_size, reduce_block_size, smem_size, stream>>>(out, out_tmp, seq_lens, balanced_seq_len_tile, num_heads);
    } else {
        const int smem_size = (balanced_seq_len_tile * 2 + 4 * HEAD_SIZE_V + 4) * sizeof(float);
        paged_attention_combine_parallel<scalar_t, HEAD_SIZE_V, BLOCK_SIZE>
            <<<reduce_grid_size, reduce_block_size, smem_size, stream>>>(out, out_tmp, seq_lens, balanced_seq_len_tile, num_heads);
    }
}

// 长序列出口路径的独立合并 (不能在启动内合并时使用), 定义在文件末尾; 返回 false 时用原合并 kernel。stream 以 void* 传递。
inline bool bhsd_long_exit_combine(int, int, int, void *, void *, int *, int, int, int, void *) { return false; }

// D256 每组 17..32 行 (BF16 Q, E5M2 KV, 原两个 Q tile 的形状) 的行切分路径, 定义在文件末尾。stream 以 void* 传递。
void bhsd_rowsplit_d256q32_launch(void *out, void *out_tmp, const void *q, const void *k_cache, const void *v_cache,
                                  int num_heads, int num_kv_heads, int *block_tables, int *seq_lens, int max_num_blocks_per_seq,
                                  int q_stride, int k_block_stride, int v_block_stride, int budget, int mtp,
                                  int batch, int num_q_heads, int reuse_kv_times, bool print_param, void *stream,
                                  unsigned int *combine_counters);

// D128 每组 33..64 行 (BF16 Q, E5M2 KV) 的行切分路径, 定义在文件末尾。stream 以 void* 传递。
void bhsd_rowsplit_d128_launch(void *out, void *out_tmp, const void *q, const void *k_cache, const void *v_cache,
                               int num_heads, int num_kv_heads, int *block_tables, int *seq_lens, int max_num_blocks_per_seq,
                               int q_stride, int k_block_stride, int v_block_stride, int budget, int mtp,
                               int batch, int num_q_heads, int reuse_kv_times, bool print_param, void *stream,
                               unsigned int *combine_counters);

// 行切分路径的启动 (函数模板: 只为 BF16 Q、E5M2 KV、D256 且每组 33..64 行实例化 kernel)
template <int HEAD_SIZE, int REUSE_KV_TIMES, typename scalar_t, typename cache_t, bool is_e4m3, bool MTP_ENABLE, int Q_LDS_PAD, typename Stream>
void launch_rowsplit_d256(void *out_ptr, void *out_tmp, void *query_ptr, void *ker_cache_ptr, void *value_cache_ptr,
                          int num_heads, int num_kv_heads, int *block_tables_ptr, int *seq_lens_ptr, int max_num_blocks_per_seq,
                          int q_stride, int k_block_stride, int v_block_stride, int rowsplit_budget, int mtp,
                          int batch, int num_q_heads, bool print_param, Stream stream, unsigned int *combine_counters)
{
    if constexpr (HEAD_SIZE == 256 && (REUSE_KV_TIMES == 48 || REUSE_KV_TIMES == 64) && std::is_same<scalar_t, uint16_t>::value &&
                  std::is_same<cache_t, uint8_t>::value && !is_e4m3) {
        constexpr int q_bytes = REUSE_KV_TIMES * HEAD_SIZE * static_cast<int>(sizeof(scalar_t));  // 无行填充, 归约区与 Q 区重叠
        constexpr int smem = q_bytes > rowsplit_d256::RING_BYTES ? q_bytes : rowsplit_d256::RING_BYTES;
        if (print_param) {
            printf("rowsplit_d256 [1], HEAD_SIZE [%d], REUSE_KV_TIMES [%d], batch [%d], mtp [%d], num_q_heads [%d], num_kv_heads [%d], "
                   "balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d], smem [%d]\n",
                   HEAD_SIZE, REUSE_KV_TIMES, batch, mtp, num_q_heads, num_kv_heads, rowsplit_budget, num_kv_heads, rowsplit_budget, batch, smem);
        }
        // 计数器非空时在同一次启动内合并: grid 的 partition 维追加每组每行一个合并线程块
        const int combine_ctas = (combine_counters != nullptr && rowsplit_budget > 1) ? num_heads / num_kv_heads : 0;
        paged_attention_kernel_rowsplit_d256<scalar_t, REUSE_KV_TIMES, MTP_ENABLE>
            <<<dim3(num_kv_heads, rowsplit_budget + combine_ctas, batch), dim3(256), smem, stream>>>(
                reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<scalar_t *>(out_tmp), reinterpret_cast<const scalar_t *>(query_ptr),
                reinterpret_cast<const uint8_t *>(ker_cache_ptr), reinterpret_cast<const uint8_t *>(value_cache_ptr),
                num_heads, num_kv_heads, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,
                q_stride, k_block_stride, v_block_stride, rowsplit_budget, mtp, combine_ctas > 0 ? combine_counters : nullptr);
        if (rowsplit_budget > 1 && combine_ctas == 0) {
            using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
            paged_attention_combine_rowsplit_d256<scalar_t, HEAD_SIZE>
                <<<dim3(num_heads, batch), dim3(256), (Shape::RED_FLOATS + 8 + rowsplit_budget) * sizeof(float), stream>>>(
                    reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<const scalar_t *>(out_tmp), seq_lens_ptr,
                    rowsplit_budget, num_heads);
        }
    }
}

// 跳转汇编 (BhsdLongExit 与 W3 出口共用, 在 jump(x, y, z) / w3_exit(x, y, z) 中展开): 把 kernarg 指针、workgroup id
// (x, y, z) 与线程号写回初始寄存器 (s[4:5], s6..s8, v0) 后跳转到 SYM 的入口, 不返回; 可变参数为 clobber 列表
// (须以 "memory" 结尾; 需要时加 "s101" 把本 kernel 的 SGPR 分配撑到不小于目标 kernel)
#define PA_EXIT_JUMP(SYM, ...)                                                                   \
    asm volatile("v_mov_b32 v1, %1\n\t"                                                          \
                 "v_mov_b32 v2, %2\n\t"                                                          \
                 "v_mov_b32 v3, %3\n\t"                                                          \
                 "v_mov_b32 v0, %4\n\t"                                                          \
                 "s_mov_b64 s[4:5], %0\n\t"                                                      \
                 "v_readfirstlane_b32 s6, v1\n\t"                                                \
                 "v_readfirstlane_b32 s7, v2\n\t"                                                \
                 "v_readfirstlane_b32 s8, v3\n\t"                                                \
                 "s_getpc_b64 s[0:1]\n\t"                                                        \
                 "s_add_u32 s0, s0, " #SYM "@rel32@lo+4\n\t"                                     \
                 "s_addc_u32 s1, s1, " #SYM "@rel32@hi+12\n\t"                                   \
                 "s_setpc_b64 s[0:1]"                                                            \
                 :: "s"((uint64_t)__builtin_amdgcn_kernarg_segment_ptr()), "s"(x), "s"(y), "s"(z), \
                    "v"(static_cast<int>(threadIdx.x))                                           \
                 : "v1", "v2", "v3", __VA_ARGS__)

// ---------------------------------------------------------------------------
// 长序列出口: 部分形状 (BF16 Q, E5M2 KV) 的原 kernel 在读出本 batch 有效长度后, 长序列跳转到行切分 kernel 的入口。
// jump(x, y, z) 把 kernarg 指针、workgroup id (x, y, z) 与线程号写回初始寄存器 (s[4:5], s6..s8, v0) 后跳转, 不返回
// (中转用 v1..v3 与私有段寄存器 s[0:1], 不占 SGPR 临时寄存器; 目标 kernel 不用 scratch);
// 目标 kernel 的参数列表与原 kernel 相同 (共用 kernarg), 两者的 kernel 描述符布局一致 (原 kernel 的 SGPR 分配由 clobber
// 撑到不小于目标 kernel)。RING_BYTES 为行切分模式所需的 LDS, host 按两种模式的较大值申请。
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, bool is_e4m3, int HEAD_SIZE, int REUSE, int TILES>
struct BhsdLongExit {
    static constexpr bool enabled = false;
    static constexpr int RING_BYTES = 0;
    static __device__ __forceinline__ bool long_mode(int) { return false; }
    static __device__ __forceinline__ void jump(int, int, int) {}
};

// 特化: 形状 (HEAD, 每组 REUSE 行, 单 Q tile) 长度 >= MIN_TOKENS 时跳转到 SYM
#define PA_BHSD_LONG_EXIT(HEAD, REUSE, MIN_TOKENS, SYM, ...)                                                 \
    template <>                                                                                              \
    struct BhsdLongExit<uint16_t, uint8_t, false, HEAD, REUSE, 1> {                                          \
        static constexpr bool enabled = true;                                                                \
        static constexpr int RING_BYTES = 65536;                                                             \
        static __device__ __forceinline__ bool long_mode(int valid_seqlen_kv) { return valid_seqlen_kv >= MIN_TOKENS; } \
        static __device__ __forceinline__ void jump(int x, int y, int z)                                     \
        {                                                                                                    \
            PA_EXIT_JUMP(SYM, __VA_ARGS__);                                                                  \
            __builtin_unreachable();                                                                         \
        }                                                                                                    \
    };
#undef PA_BHSD_LONG_EXIT

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_bhsd_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_dim_qk]
    const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, block_size, head_dim_v]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const scalar_t* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp,
    unsigned int* __restrict__ combine_counters) // 非空时在同一次启动内合并 partition (见 direct_combine_head_in_kernel)
{
    extern __shared__ char shared_mem[];
    scalar_t *smem_q = (scalar_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);
    cache_t *smem_v = (cache_t *)(shared_mem + REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t));

    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;
    using scalarx8_t= __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;

    // grid_size: (num_kv_heads, max_partition, batch_size)
    const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    // 长序列出口 (见 BhsdLongExit): 行切分 kernel 处理本线程块 (含合并线程块与多余线程块)
    if constexpr (BhsdLongExit<scalar_t, cache_t, is_e4m3, HEAD_SIZE, REUSE_KV_TIMES, 1>::enabled) {
        if (BhsdLongExit<scalar_t, cache_t, is_e4m3, HEAD_SIZE, REUSE_KV_TIMES, 1>::long_mode(valid_seqlen_kv)) {
            BhsdLongExit<scalar_t, cache_t, is_e4m3, HEAD_SIZE, REUSE_KV_TIMES, 1>::jump(kv_head_idx, partition_idx, batch_idx);
        }
    }
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    constexpr bool direct_partitions = DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any;
    constexpr int direct_ctas_per_cu = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16)>::CTAS_PER_CU;
    constexpr bool long_single_stream = direct_ctas_per_cu == 2 && DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16) * HEAD_SIZE <= 256;
    constexpr bool resident_cap = direct_partitions && long_single_stream;
    int valid_num_partition = direct_partitions ? get_direct_num_partitions<direct_ctas_per_cu, long_single_stream, resident_cap>(valid_num_block_pages, balanced_seq_len_tile)
                                                : get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if constexpr (direct_partitions) {
        // partition 维追加的合并线程块: 第 partition_idx - balanced_seq_len_tile 行
        if (combine_counters != nullptr && partition_idx >= balanced_seq_len_tile) {
            if (valid_num_partition > 1) {
                const int rows = num_heads / num_kv_heads;
                const int q_heads_per_kv_head = rows / mtp;
                const int row = partition_idx - balanced_seq_len_tile;
                const int head = batch_idx * num_heads + q_heads_per_kv_head * kv_head_idx
                                 + (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                direct_combine_head_in_kernel<scalar_t, HEAD_SIZE_V>(
                    combine_counters + (batch_idx * num_kv_heads + kv_head_idx) * DIRECT_COMBINE_COUNTER_STRIDE,
                    valid_num_partition, rows, out, out_tmp, head, balanced_seq_len_tile, shared_mem);
            }
            return;
        }
    }

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_query_fp16 true, scalar_t = fp16; is_query_fp16 false, scalar_t = bf16;
    constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_kv_fp8 true, cache_t = e5m2; is_kv_fp8 false, cache_t = scalar_t
    constexpr bool is_kv_fp8 = std::is_same<cache_t, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    smem_v += warp_id * 2 * 8 * 128 * sizeof(uint8_t);

    // q global -> LDS; Q fragments are loaded on demand in QK GEMM.
    constexpr int Mloop = DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16);

    float alibi_slope[Mloop];
    // alibi slope
    if constexpr (ALIBI_ENABLE){
        int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
        int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
        for(int m = 0;m < Mloop;m++){
            int row = m * 16 + (lane_id & 15);
            if(row < seqlen_q_per_kv_head){
                int local_q_head = row % num_q_heads_per_kv_head;
                int slope_idx = q_head_idx + local_q_head;
                alibi_slope[m] = alibi_slopes[slope_idx] / scale_softmax;
            }
        }
    }

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    // Spread neighboring page indices across neighboring partitions.
    const int start_block_idx = partition_idx;
    const int end_block_idx = valid_num_block_pages;
    // Partition p owns pages p, p + P, ...; warps split that sequence.

    // k/v_cache 形状为 [num_blocks, num_kv_heads, block_size, head_dim]; k/v_ptr_base 不含 num_blocks 维度的偏移
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
#else
    constexpr int k_head_stride = HEAD_SIZE;
    constexpr int v_head_stride = HEAD_SIZE_V;
#endif
    const cache_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const cache_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;

    constexpr bool direct_fp8 = is_kv_fp8 && DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::enabled;
    constexpr bool direct_b16 = !is_kv_fp8 && DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::enabled;
    constexpr bool direct_any = direct_fp8 || direct_b16;
    // 直接加载路径: BF16 Q 以缩放后的 FP16 参与 QK (见 commit_q_stage_fp16); q_scale 为 softmax 缩放的修正倍数
    constexpr bool q_as_fp16 = direct_fp8 && !is_query_fp16;
    using qk_t = std::conditional_t<q_as_fp16, _Float16, scalar_t>;
    using qk_x8_t = __attribute__((__vector_size__(8 * sizeof(qk_t)))) qk_t;
    using qk_x4_t = __attribute__((__vector_size__(4 * sizeof(qk_t)))) qk_t;
    float q_scale = 1.0f;
    if constexpr (direct_any) {
        // Q 先发射, 首个 page 的 K/V 随后发射, 写 Q 到 LDS 时只需等待 Q
        const int q_rows_begin = 0;
        const int q_rows_count = num_heads / num_kv_heads;
        QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, REUSE_KV_TIMES> q_stage;
        issue_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, REUSE_KV_TIMES>(
            q_stage, q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, q_rows_begin, q_rows_count);
        auto q_commit = [&]() {
            if constexpr (q_as_fp16) {
                // The conversion/reduction cost is worthwhile when each page
                // has enough Q work or a warp processes multiple full pages.
                const bool amortize_fp16 = Mloop * HEAD_SIZE >= 512 ||
                    (!use_direct_slice_mode<direct_ctas_per_cu>(valid_num_block_pages, balanced_seq_len_tile) &&
                     valid_num_block_pages > NUM_WARPS * valid_num_partition);
                if (!amortize_fp16) {
                    commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, REUSE_KV_TIMES, Q_LDS_PAD>(
                        q_stage, smem_q, q_rows_count);
                    return -1.0f; // Same optimized kernel, BF16 QK.
                }
                uint32_t *reduce = reinterpret_cast<uint32_t *>(
                    shared_mem + DirectEpilogueShape<Mloop, HEAD_SIZE_V, NUM_WARPS>::meta_offset(REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)));
                q_scale = commit_q_stage_fp16<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, REUSE_KV_TIMES, Q_LDS_PAD>(
                    q_stage, reinterpret_cast<_Float16 *>(smem_q), q_rows_count, reduce);
                const bool q_bf16 = q_scale < 0.0f;
                q_scale = fabsf(q_scale);
                if constexpr (ALIBI_ENABLE) {
                    #pragma unroll
                    for (int m = 0; m < Mloop; ++m) alibi_slope[m] /= q_scale;
                }
                return q_bf16 ? -q_scale : q_scale;
            } else {
                commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, REUSE_KV_TIMES, Q_LDS_PAD>(q_stage, smem_q, q_rows_count);
            }
            return q_scale;
        };
        // token 下标为 token_base + i * stride_i + (lane >> 4) * stride_g + j * stride_j
        auto score_mask = [&](auto &s_reg, const int token_base, const int stride_i, const int stride_g, const int stride_j) {
            constexpr int NT = std::extent<std::remove_reference_t<decltype(s_reg)>, 1>::value;
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
            const int token_last = token_base + (NT - 1) * stride_i + 3 * stride_g + (stride_j < 0 ? 36 : 3 * stride_j);
#else
            const int token_last = token_base + (NT - 1) * stride_i + 3 * stride_g + 3 * stride_j;
#endif
            // alibi
            if constexpr (ALIBI_ENABLE){
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
                for(int m = 0;m < Mloop;m++){
                    int row = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(row < seqlen_q_per_kv_head){
                        int q_token_id = row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        int query_position = valid_seqlen_kv - casual_offset;
                        for(int i = 0;i < NT;i++){
                            for(int j = 0;j < 4;j++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                                int token_k_index = token_base + i * stride_i + col * stride_g + (stride_j < 0 ? 32 * (j >> 1) + 4 * (j & 1) : j * stride_j);
#else
                                int token_k_index = token_base + i * stride_i + col * stride_g + j * stride_j;
#endif
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    int relative_position = token_k_index - query_position;
                                    float alibi = alibi_slope[m] * relative_position;
                                    // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                    (s_reg[m][i])[j] += alibi;
                                }
                            }
                        }
                    }
                }
            }

            // mtp causal
            if constexpr (MTP_ENABLE){
                if (token_last + mtp > valid_seqlen_kv) {
                    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                    int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                    for(int m = 0;m < Mloop;m++){
                        int row = m * 16 + (lane_id & 15);
                        int col = lane_id >> 4;
                        if(row < seqlen_q_per_kv_head){
                            int q_token_id = row / num_q_heads_per_kv_head;
                            int casual_offset = mtp - q_token_id;
                            for(int i = 0;i < NT;i++){
                                for(int j = 0;j < 4;j++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                                    int token_k_index = token_base + i * stride_i + col * stride_g + (stride_j < 0 ? 32 * (j >> 1) + 4 * (j & 1) : j * stride_j);
#else
                                    int token_k_index = token_base + i * stride_i + col * stride_g + j * stride_j;
#endif
                                    if(token_k_index < valid_seqlen_kv){
                                        // 边界内
                                        if((token_k_index + casual_offset > valid_seqlen_kv)){
                                            (s_reg[m][i])[j] = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        };
        auto direct_page_loop = [&](auto &&...loop_args) {
            if constexpr (direct_b16) {
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                direct_b16_page_loop_vt<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(loop_args...);
            } else {
                direct_fp8_page_loop_vt<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16 || q_as_fp16>(loop_args...);
#else
                direct_b16_page_loop<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(loop_args...);
            } else {
                direct_fp8_page_loop<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16 || q_as_fp16>(loop_args...);
#endif
            }
        };
        direct_page_loop(
            reinterpret_cast<const uint8_t *>(k_ptr_base), reinterpret_cast<const uint8_t *>(v_ptr_base),
            k_block_stride, v_block_stride, block_table, partition_idx, valid_num_partition, warp_id,
            use_direct_slice_mode<direct_ctas_per_cu>(valid_num_block_pages, balanced_seq_len_tile), end_block_idx,
            num_kv_heads, reinterpret_cast<const qk_t *>(smem_q), num_heads / num_kv_heads, valid_seqlen_kv, scale_softmax_log2,
            scores_max, scores_sum, acc_o, q_commit, score_mask);
    } else {
        prefetch_q_to_lds<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD>(
            q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride);

        for(int block_idx = (start_block_idx + warp_id * valid_num_partition);block_idx < end_block_idx;block_idx += NUM_WARPS * valid_num_partition){
            // 1个warp处理一个block_page
            const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
            // 加上 num_blocks 维度的偏移
            const cache_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

            // v_cache 在 head_dim_v 维度连续, pv gemm 时 v 为行主序
            const cache_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;

            float4_t s_reg[Mloop][BLOCK_SIZE / 16];
            for(int m = 0;m < Mloop;m++){
                for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                    s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
                }
            }

            // 每个warp 处理 一个 kv block_page
            qk_gemm_func_single_buffer_bhsd<scalarx8_t, scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, Q_LDS_PAD>
                                        (k_ptr, smem_q, num_heads / num_kv_heads, s_reg, block_idx, valid_seqlen_kv);

            // alibi
            if constexpr (ALIBI_ENABLE){
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
                for(int m = 0;m < Mloop;m++){
                    int row = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(row < seqlen_q_per_kv_head){
                        int q_token_id = row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        int query_position = valid_seqlen_kv - casual_offset;
                        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                            for(int j = 0;j < 4;j++){
                                int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    int relative_position = token_k_index - query_position;
                                    float alibi = alibi_slope[m] * relative_position;
                                    // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                    (s_reg[m][i])[j] += alibi;
                                }
                            }
                        }
                    }
                }
            }

            // mtp causal
            if constexpr (MTP_ENABLE){
                if (block_idx * BLOCK_SIZE + BLOCK_SIZE - 1 + mtp > valid_seqlen_kv) {
                    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                    int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                    for(int m = 0;m < Mloop;m++){
                        int row = m * 16 + (lane_id & 15);
                        int col = lane_id >> 4;
                        if(row < seqlen_q_per_kv_head){
                            int q_token_id = row / num_q_heads_per_kv_head;
                            int casual_offset = mtp - q_token_id;
                            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                                for(int j = 0;j < 4;j++){
                                    int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                    if(token_k_index < valid_seqlen_kv){
                                        // 边界内
                                        if((token_k_index + casual_offset > valid_seqlen_kv)){
                                            (s_reg[m][i])[j] = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // online softmax: S <- exp(S - m(i)), d(i) = d(i-1) * exp(m(i-1) - m(i)) + rowsum, O 同乘 exp(m(i-1) - m(i)) (max 为未缩放的原值)
            softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, block_idx != start_block_idx + warp_id * valid_num_partition);

            scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16];
            // fp32 -> fp16
            for(int m = 0;m < Mloop;m++){
                for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                    scalar_t tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[0]);
                    (p_reg[m][i])[0] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[1]);
                    (p_reg[m][i])[1] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[2]);
                    (p_reg[m][i])[2] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[3]);
                    (p_reg[m][i])[3] = tmp_vale;
                }
            }

            // pv gemm
            // acc_o: O(i) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale) + gemm_pv
    #if (defined(__gfx936__)||defined(__gfx938__))
            // The matrix-load path processes full 128-channel V tiles.
            if constexpr (is_kv_fp8 && HEAD_SIZE_V % 128 == 0){
                pv_gemm_func_single_buffer_fp8_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3>(smem_v, v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv, num_kv_heads);
            }else{
                pv_gemm_func_single_buffer_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
            }
    #else
            pv_gemm_func_single_buffer_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
    #endif
        }
    }

    // Q remains live in LDS throughout the page loop. Prevent an early warp
    // from reusing the same LDS storage for max/sum while another warp reads Q.
    // 直接加载路径的 epilogue 在写入独立的 max/sum 区后同步, 此处无需同步
    if constexpr (!direct_any) {
        __syncthreads();
    }

    // sink
    if constexpr(SINK_ENABLE)
    {
        // sink 相当于 S 的每行多一个元素 sink_value (对应的 V 为 0): 按 max(sink_value, max) 更新 max/sum 并缩放 acc_o
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink<scalar_t, Mloop, HEAD_SIZE_V>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax * q_scale,
                scale_softmax_log2 * q_scale,
                warp_id,
                partition_idx,
                mtp,
                num_heads,
                num_kv_heads,
                kv_head_idx);
        }
    }

    if constexpr (direct_any) {
        const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        const int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
        const int head_q_index = q_heads_per_kv_head * kv_head_idx;
        const int epi_rows = seqlen_q_per_kv_head;
        auto out_row_of = [&](const int row) {
            return (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
        };
        // 1 == valid_num_partition 时直接写 out, 否则写 partial 输出与 max/sum
        const bool split = valid_num_partition > 1;
        const int out_head_stride = split ? balanced_seq_len_tile * HEAD_SIZE_V : HEAD_SIZE_V;
        scalar_t *out_ptr_base = split ? out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                                             + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V
                                       : out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
        auto store_o = [&](const int row, const int col, const float4_t sums) {
            const scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]),
                                         from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
            *reinterpret_cast<scalarx4_t *>(out_ptr_base + out_row_of(row) * out_head_stride + col) = out_sums;
        };
        auto store_meta = [&](const int row, const float max_new, const float sum_new) {
            if (split) {
                const int offset = batch_idx * num_heads * balanced_seq_len_tile
                                   + (head_q_index + out_row_of(row)) * balanced_seq_len_tile + partition_idx;
                reinterpret_cast<float *>(out_tmp)[offset] = sum_new;
                // 此处存到 global 的 max 经过 scale_softmax_log2
                reinterpret_cast<float *>(out_tmp + max_tmp_offset)[offset] = max_new * (scale_softmax_log2 * q_scale);
            }
        };
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
        direct_epilogue<Mloop, HEAD_SIZE_V, NUM_WARPS, REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t), direct_b16>(
#else
        direct_epilogue<Mloop, HEAD_SIZE_V, NUM_WARPS, REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)>(
#endif
            scores_max, scores_sum, acc_o, shared_mem, epi_rows, warp_id, scale_softmax_log2 * q_scale, store_o, store_meta);
        if (split && combine_counters != nullptr) {
            direct_combine_arrive(combine_counters + (batch_idx * num_kv_heads + kv_head_idx) * DIRECT_COMBINE_COUNTER_STRIDE);
        }
    } else {
        float scores_max_new[Mloop];
        float scores_sum_new[Mloop];

        epilogue_rescale_acco<Mloop, HEAD_SIZE_V, NUM_WARPS>(scores_max, scores_sum, acc_o, scores_max_new, scores_sum_new,
                                                            smem_max, smem_sum, scale_softmax_log2, warp_id, num_heads, num_kv_heads);

        // 当前 acc_o 为经过缩放结果, 需要将4个warp的 acc_o 加起来得到 PARTION_SIZE 的输出
        //  通过lds reduce实现
        {
            const int acc_row = lane_id & 15;
            const int acc_col = lane_id >> 4;
            // 每个kv head处理的token q数量
            const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
            // 每个kv head对应的q head数量
            int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
            // 当前block处理的q head起始索引, 即当前kv_head_id对应的q_head_id_start
            int head_q_index = q_heads_per_kv_head * kv_head_idx;
            // 若 1 == valid_num_partition, 对应的 out 的形状是 (batch, seqlen_q * num_q_heads, head_dim_v), scalar_t
            // 若 valid_num_partition > 1, 对应的 out_buffer 的形状是(batch, seqlen_q * num_q_heads, max_num_partitions, head_dim_v) scalar_t
            scalar_t *out_ptr_base = nullptr;
            int out_head_stride = 0;
            // num_heads = seqlen_q * num_q_heads
            // out_ptr_base 考虑了 batch 维度, head 维度起始head索引的偏移, 没有考虑 seqlen 维度的偏移
            if (1 == valid_num_partition){
                out_head_stride = HEAD_SIZE_V;
                out_ptr_base = out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
            }else{
                out_head_stride = balanced_seq_len_tile  * HEAD_SIZE_V;
                // out_tmp_offset 元素, 为预留的 max sum buffer
                out_ptr_base = out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                                + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V;
            }

            // lds reduce 每个迭代处理 NUM_WARPS * 16x16 *4 的数据, 保证每个线程 b128 的写出, 且每个warp需要写
            constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
            // lds 形状为(NUM_WARPS, WARP_SIZE * 4)
            float *smem_reduce = (float *)shared_mem;
            for(int tile_idx = 0;tile_idx < num_16x64_tiles;tile_idx++){
                const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
                const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
                for(int j = 0;j < 4;j++){
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 0) = (acc_o[tile_idx_row][tile_idx_col * 4 + 0])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 1) = (acc_o[tile_idx_row][tile_idx_col * 4 + 1])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 2) = (acc_o[tile_idx_row][tile_idx_col * 4 + 2])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 3) = (acc_o[tile_idx_row][tile_idx_col * 4 + 3])[j];
                }
                __syncthreads();

                int out_r2g_row = threadIdx.x >> 4;
                int out_r2g_col = threadIdx.x & 15;
                const int row = tile_idx_row * 16 + out_r2g_row;

                // 边界限制
                if(row < seqlen_q_per_kv_head){
                    // 组内第 row 行对应 token row / q_heads_per_kv_head 的第 head_q_index + row % q_heads_per_kv_head 个 q head
                    // (例: 16 个 q head、2 个 token、2 个 KV head 时, 每组 16 行: 0-7 为 token0、8-15 为 token1 的本组 8 个 head)
                    const int out_row = (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                    const int our_col = tile_idx_col * 64 + out_r2g_col * 4;
                    float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f};
                    for(int i = 0;i < NUM_WARPS;i++){
                        float4_t temp_value = *((float4_t *)(smem_reduce + i * 16 * (64 + REDUCE_LDS_PAD) + out_r2g_row * (64 + REDUCE_LDS_PAD) + out_r2g_col * 4));
                        sums[0] += temp_value[0];
                        sums[1] += temp_value[1];
                        sums[2] += temp_value[2];
                        sums[3] += temp_value[3];
                    }

                    scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]), from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
                    *((scalarx4_t *)(out_ptr_base + out_row * out_head_stride + our_col)) = out_sums;
                }
                 __syncthreads();
            }

            // 保存 max sum
            if(valid_num_partition > 1){
                // max, sum 形状为 (batch, seqlen_q, num_q_heads, max_num_partitions)
                if(warp_id == 0 && acc_col == 0){
                    for(int m = 0; m < Mloop;m++){
                        int row = m * 16 + acc_row;
                        if(row < seqlen_q_per_kv_head){
                            int out_row = (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                            int offset = batch_idx * num_heads * balanced_seq_len_tile
                                          + (head_q_index + out_row) * balanced_seq_len_tile
                                          + partition_idx;
                            float *exp_sums = reinterpret_cast<float*>(out_tmp);
                            // max_tmp_offset 元素单位
                            float *max = reinterpret_cast<float*>(out_tmp + max_tmp_offset);

                            exp_sums[offset] = scores_sum_new[m];
                            // 此处存到 global 的 max 经过 scale_softmax_log2
                            max[offset] = scores_max_new[m]  * scale_softmax_log2;
                        }
                    }
                }
            }
        }

    }

    return;
}

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE, int NUM_Q_TILES, int Q_TILE>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_qtile_bhsd_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size_v]
    const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, head_size_v, block_size]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const scalar_t* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp,
    unsigned int* __restrict__ combine_counters) // 非空时在同一次启动内合并 partition (见 direct_combine_head_in_kernel)
{
    extern __shared__ char shared_mem[];
    scalar_t *smem_q = (scalar_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);
    cache_t *smem_v = (cache_t *)(shared_mem + Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t));

    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;
    using scalarx8_t= __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;

    // 当前线程块对应的 Q 方向的线程块索引
    // blockIdx.x   q_tile_idx      kv_head_idx
    //      0           0           0           对应 q_token = 0, Q head[0, 31], K head 0
    //      1           1           0           对应 q_token = 1, Q head[0, 31], K head 0
    //      2           0           1           对应 q_token = 0, Q head[32, 63], K head 1
    //      3           1           1           对应 q_token = 1, Q head[32, 63], K head 1
    const int q_tile_idx = blockIdx.x % NUM_Q_TILES;
    // 当前线程块处理的 kv_head 的索引
    const int kv_head_idx = blockIdx.x / NUM_Q_TILES;
    // 将 num_heads 均分, 每个 Q 方向的线程块处理 相同数量的数据
    const int local_num_heads = num_heads / NUM_Q_TILES;

    // grid_size: (num_kv_heads, max_partition, batch_size)
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    constexpr bool direct_partitions = DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any;
    constexpr int direct_ctas_per_cu = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16)>::CTAS_PER_CU;
    constexpr bool resident_cap = direct_partitions && HEAD_SIZE == 256 && direct_ctas_per_cu == 2;
    constexpr bool one_page_slice = resident_cap && NUM_Q_TILES == 2 && Q_TILE >= 24;
    int valid_num_partition = direct_partitions ? get_direct_num_partitions<direct_ctas_per_cu, false, resident_cap, one_page_slice>(valid_num_block_pages, balanced_seq_len_tile)
                                                : get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if constexpr (direct_partitions) {
        // partition 维追加的合并线程块: 本 q tile 的第 partition_idx - balanced_seq_len_tile 行
        if (combine_counters != nullptr && partition_idx >= balanced_seq_len_tile) {
            if (valid_num_partition > 1) {
                const int rows_all = num_heads / num_kv_heads;
                const int q_heads_per_kv_head = rows_all / mtp;
                const int rows_per_qtile = DIVIDE_ROUND_UP(rows_all, NUM_Q_TILES);
                const int local_row = partition_idx - balanced_seq_len_tile;
                if (local_row >= MIN_T(rows_per_qtile, rows_all - q_tile_idx * rows_per_qtile)) return;
                const int row = q_tile_idx * rows_per_qtile + local_row;
                const int head = batch_idx * num_heads + q_heads_per_kv_head * kv_head_idx
                                 + (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                direct_combine_head_in_kernel<scalar_t, HEAD_SIZE_V>(
                    combine_counters + ((batch_idx * num_kv_heads + kv_head_idx) * NUM_Q_TILES + q_tile_idx) * DIRECT_COMBINE_COUNTER_STRIDE,
                    valid_num_partition, MIN_T(rows_per_qtile, rows_all - q_tile_idx * rows_per_qtile), out, out_tmp, head,
                    balanced_seq_len_tile, shared_mem);
            }
            return;
        }
    }

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_query_fp16 true, scalar_t = fp16; is_query_fp16 false, scalar_t = bf16;
    constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_kv_fp8 true, cache_t = e5m2; is_kv_fp8 false, cache_t = scalar_t
    constexpr bool is_kv_fp8 = std::is_same<cache_t, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    smem_v += warp_id * 2 * 8 * 128 * sizeof(uint8_t);

    // q global -> LDS; Q fragments are loaded on demand in QK GEMM.
    constexpr int Mloop = DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16);

    float alibi_slope[Mloop];
    // alibi slope
    if constexpr (ALIBI_ENABLE){
        int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
        int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;

        int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
        int tile_start = q_tile_idx * rows_per_qtile;
        int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

        for(int m = 0;m < Mloop;m++){
            int local_row  = m * 16 + (lane_id & 15);
            if(local_row < tile_rows){
                int global_row = tile_start + local_row;
                int local_q_head = global_row % num_q_heads_per_kv_head;
                int slope_idx = q_head_idx + local_q_head;
                alibi_slope[m] = alibi_slopes[slope_idx] / scale_softmax;
            }else{
                alibi_slope[m] = 0.0f;
            }
        }
    }

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    // Spread neighboring page indices across neighboring partitions.
    const int start_block_idx = partition_idx;
    const int end_block_idx = valid_num_block_pages;
    // Partition p owns pages p, p + P, ...; warps split that sequence.

    // k/v_cache 形状为 [num_blocks, num_kv_heads, block_size, head_dim]; k/v_ptr_base 不含 num_blocks 维度的偏移
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
#else
    constexpr int k_head_stride = HEAD_SIZE;
    constexpr int v_head_stride = HEAD_SIZE_V;
#endif
    const cache_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const cache_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;

    constexpr bool direct_fp8 = is_kv_fp8 && DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::enabled;
    constexpr bool direct_b16 = !is_kv_fp8 && DirectB16Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::enabled;
    constexpr bool direct_any = direct_fp8 || direct_b16;
    // 直接加载路径: BF16 Q 以缩放后的 FP16 参与 QK (见 commit_q_stage_fp16); q_scale 为 softmax 缩放的修正倍数
    constexpr bool q_as_fp16 = direct_fp8 && !is_query_fp16;
    using qk_t = std::conditional_t<q_as_fp16, _Float16, scalar_t>;
    using qk_x8_t = __attribute__((__vector_size__(8 * sizeof(qk_t)))) qk_t;
    using qk_x4_t = __attribute__((__vector_size__(4 * sizeof(qk_t)))) qk_t;
    float q_scale = 1.0f;
    if constexpr (direct_any) {
        // Q 先发射, 首个 page 的 K/V 随后发射, 写 Q 到 LDS 时只需等待 Q
        const int q_rows_all = num_heads / num_kv_heads;
        const int q_rows_per_tile = DIVIDE_ROUND_UP(q_rows_all, NUM_Q_TILES);
        const int q_rows_begin = q_tile_idx * q_rows_per_tile;
        const int q_rows_count = MIN_T(q_rows_per_tile, q_rows_all - q_rows_begin);
        QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, Q_TILE> q_stage;
        issue_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE>(
            q_stage, q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, q_rows_begin, q_rows_count);
        auto q_commit = [&]() {
            if constexpr (q_as_fp16) {
                // The conversion/reduction cost is worthwhile when each page
                // has enough Q work or a warp processes multiple full pages.
                const bool amortize_fp16 = Mloop * HEAD_SIZE >= 512 ||
                    (!use_direct_slice_mode<direct_ctas_per_cu, one_page_slice>(valid_num_block_pages, balanced_seq_len_tile) &&
                     valid_num_block_pages > NUM_WARPS * valid_num_partition);
                if (!amortize_fp16) {
                    commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(
                        q_stage, smem_q, q_rows_count);
                    return -1.0f; // Same optimized kernel, BF16 QK.
                }
                uint32_t *reduce = reinterpret_cast<uint32_t *>(
                    shared_mem + DirectEpilogueShape<Mloop, HEAD_SIZE_V, NUM_WARPS>::meta_offset(Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)));
                q_scale = commit_q_stage_fp16<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(
                    q_stage, reinterpret_cast<_Float16 *>(smem_q), q_rows_count, reduce);
                const bool q_bf16 = q_scale < 0.0f;
                q_scale = fabsf(q_scale);
                if constexpr (ALIBI_ENABLE) {
                    #pragma unroll
                    for (int m = 0; m < Mloop; ++m) alibi_slope[m] /= q_scale;
                }
                return q_bf16 ? -q_scale : q_scale;
            } else {
                commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(q_stage, smem_q, q_rows_count);
            }
            return q_scale;
        };
        // token 下标为 token_base + i * stride_i + (lane >> 4) * stride_g + j * stride_j
        auto score_mask = [&](auto &s_reg, const int token_base, const int stride_i, const int stride_g, const int stride_j) {
            constexpr int NT = std::extent<std::remove_reference_t<decltype(s_reg)>, 1>::value;
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
            const int token_last = token_base + (NT - 1) * stride_i + 3 * stride_g + (stride_j < 0 ? 36 : 3 * stride_j);
#else
            const int token_last = token_base + (NT - 1) * stride_i + 3 * stride_g + 3 * stride_j;
#endif
            // alibi
            if constexpr (ALIBI_ENABLE){
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;

                int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
                int tile_start = q_tile_idx * rows_per_qtile;
                const int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

                for(int m = 0;m < Mloop;m++){
                    int local_row  = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(local_row < tile_rows){
                        int global_row = tile_start + local_row;
                        int q_token_id = global_row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        int query_position = valid_seqlen_kv - casual_offset;
                        for(int i = 0;i < NT;i++){
                            for(int j = 0;j < 4;j++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                                int token_k_index = token_base + i * stride_i + col * stride_g + (stride_j < 0 ? 32 * (j >> 1) + 4 * (j & 1) : j * stride_j);
#else
                                int token_k_index = token_base + i * stride_i + col * stride_g + j * stride_j;
#endif
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    int relative_position = token_k_index - query_position;
                                    float alibi = alibi_slope[m] * relative_position;
                                    // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                    (s_reg[m][i])[j] += alibi;
                                }
                            }
                        }
                    }
                }
            }

            // mtp causal
            if constexpr (MTP_ENABLE){
                if (token_last + mtp > valid_seqlen_kv) {
                    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                    int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;

                    int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
                    int tile_start = q_tile_idx * rows_per_qtile;
                    int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

                    for(int m = 0;m < Mloop;m++){
                        int local_row  = m * 16 + (lane_id & 15);
                        int col = lane_id >> 4;
                        if(local_row < tile_rows){
                            int global_row = tile_start + local_row;
                            int q_token_id = global_row / num_q_heads_per_kv_head;
                            int casual_offset = mtp - q_token_id;
                            for(int i = 0;i < NT;i++){
                                for(int j = 0;j < 4;j++){
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                                    int token_k_index = token_base + i * stride_i + col * stride_g + (stride_j < 0 ? 32 * (j >> 1) + 4 * (j & 1) : j * stride_j);
#else
                                    int token_k_index = token_base + i * stride_i + col * stride_g + j * stride_j;
#endif
                                    if(token_k_index < valid_seqlen_kv){
                                        // 边界内
                                        if((token_k_index + casual_offset > valid_seqlen_kv)){
                                            (s_reg[m][i])[j] = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        };
        auto direct_page_loop = [&](auto &&...loop_args) {
            if constexpr (direct_b16) {
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
                direct_b16_page_loop_vt<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(loop_args...);
            } else {
                direct_fp8_page_loop_vt<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16 || q_as_fp16>(loop_args...);
#else
                direct_b16_page_loop<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16>(loop_args...);
            } else {
                direct_fp8_page_loop<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, is_query_fp16 || q_as_fp16>(loop_args...);
#endif
            }
        };
        direct_page_loop(
            reinterpret_cast<const uint8_t *>(k_ptr_base), reinterpret_cast<const uint8_t *>(v_ptr_base),
            k_block_stride, v_block_stride, block_table, partition_idx, valid_num_partition, warp_id,
            use_direct_slice_mode<direct_ctas_per_cu, one_page_slice>(valid_num_block_pages, balanced_seq_len_tile), end_block_idx,
            num_kv_heads, reinterpret_cast<const qk_t *>(smem_q), local_num_heads / num_kv_heads, valid_seqlen_kv, scale_softmax_log2,
            scores_max, scores_sum, acc_o, q_commit, score_mask);
    } else {
        // Q [num_seqs, seqlen_q, num_q_heads, head_size]
        prefetch_q_to_lds_qtile<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD, NUM_Q_TILES>(
            q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, q_tile_idx);

        for(int block_idx = (start_block_idx + warp_id * valid_num_partition);block_idx < end_block_idx;block_idx += NUM_WARPS * valid_num_partition){
            // 1个warp处理一个block_page
            const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
            // 加上 num_blocks 维度的偏移
            const cache_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

            // v_cache 在 head_dim_v 维度连续, pv gemm 时 v 为行主序
            const cache_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;

            float4_t s_reg[Mloop][BLOCK_SIZE / 16];
            for(int m = 0;m < Mloop;m++){
                for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                    s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
                }
            }

            // 每个warp 处理 一个 kv block_page
            qk_gemm_func_single_buffer_bhsd<scalarx8_t, scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, Q_LDS_PAD, NUM_Q_TILES>
                                        (k_ptr, smem_q, local_num_heads / num_kv_heads, s_reg, block_idx, valid_seqlen_kv);

            // alibi
            if constexpr (ALIBI_ENABLE){
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;

                int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
                int tile_start = q_tile_idx * rows_per_qtile;
                const int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

                for(int m = 0;m < Mloop;m++){
                    int local_row  = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(local_row < tile_rows){
                        int global_row = tile_start + local_row;
                        int q_token_id = global_row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        int query_position = valid_seqlen_kv - casual_offset;
                        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                            for(int j = 0;j < 4;j++){
                                int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    int relative_position = token_k_index - query_position;
                                    float alibi = alibi_slope[m] * relative_position;
                                    // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                    (s_reg[m][i])[j] += alibi;
                                }
                            }
                        }
                    }
                }
            }

            // mtp causal
            if constexpr (MTP_ENABLE){
                if (block_idx * BLOCK_SIZE + BLOCK_SIZE - 1 + mtp > valid_seqlen_kv) {
                    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                    int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;

                    int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
                    int tile_start = q_tile_idx * rows_per_qtile;
                    int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

                    for(int m = 0;m < Mloop;m++){
                        int local_row  = m * 16 + (lane_id & 15);
                        int col = lane_id >> 4;
                        if(local_row < tile_rows){
                            int global_row = tile_start + local_row;
                            int q_token_id = global_row / num_q_heads_per_kv_head;
                            int casual_offset = mtp - q_token_id;
                            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                                for(int j = 0;j < 4;j++){
                                    int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                    if(token_k_index < valid_seqlen_kv){
                                        // 边界内
                                        if((token_k_index + casual_offset > valid_seqlen_kv)){
                                            (s_reg[m][i])[j] = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // online softmax: S <- exp(S - m(i)), d(i) = d(i-1) * exp(m(i-1) - m(i)) + rowsum, O 同乘 exp(m(i-1) - m(i)) (max 为未缩放的原值)
            softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, block_idx != start_block_idx + warp_id * valid_num_partition);

            scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16];
            // fp32 -> fp16
            for(int m = 0;m < Mloop;m++){
                for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                    scalar_t tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[0]);
                    (p_reg[m][i])[0] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[1]);
                    (p_reg[m][i])[1] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[2]);
                    (p_reg[m][i])[2] = tmp_vale;
                    from_float(tmp_vale, (s_reg[m][i])[3]);
                    (p_reg[m][i])[3] = tmp_vale;
                }
            }

            // pv gemm
            // acc_o: O(i) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale) + gemm_pv
    #if (defined(__gfx936__)||defined(__gfx938__))
            // The matrix-load path processes full 128-channel V tiles.
            if constexpr (is_kv_fp8 && HEAD_SIZE_V % 128 == 0){
                pv_gemm_func_single_buffer_fp8_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, NUM_Q_TILES>(smem_v, v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv, num_kv_heads);
            }else{
                pv_gemm_func_single_buffer_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, NUM_Q_TILES>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
            }
    #else
            pv_gemm_func_single_buffer_bhsd<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, NUM_Q_TILES>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
    #endif
        }
    }

    // Q remains live in LDS throughout the page loop. Prevent an early warp
    // from reusing the same LDS storage for max/sum while another warp reads Q.
    // 直接加载路径的 epilogue 在写入独立的 max/sum 区后同步, 此处无需同步
    if constexpr (!direct_any) {
        __syncthreads();
    }

    // sink
    if constexpr(SINK_ENABLE)
    {
        // sink 相当于 S 的每行多一个元素 sink_value (对应的 V 为 0): 按 max(sink_value, max) 更新 max/sum 并缩放 acc_o
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink_q_tile<scalar_t, Mloop, HEAD_SIZE_V, NUM_Q_TILES>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax * q_scale,
                scale_softmax_log2 * q_scale,
                warp_id,
                partition_idx,
                mtp,
                num_heads,
                num_kv_heads,
                kv_head_idx,
                q_tile_idx
            );
        }
    }

    if constexpr (direct_any) {
        const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        const int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
        const int head_q_index = q_heads_per_kv_head * kv_head_idx;
        const int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
        const int tile_start = q_tile_idx * rows_per_qtile;
        const int epi_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);
        auto out_row_of = [&](const int local_row) {
            const int row = tile_start + local_row;
            return (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
        };
        // 1 == valid_num_partition 时直接写 out, 否则写 partial 输出与 max/sum
        const bool split = valid_num_partition > 1;
        const int out_head_stride = split ? balanced_seq_len_tile * HEAD_SIZE_V : HEAD_SIZE_V;
        scalar_t *out_ptr_base = split ? out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                                             + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V
                                       : out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
        auto store_o = [&](const int row, const int col, const float4_t sums) {
            const scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]),
                                         from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
            *reinterpret_cast<scalarx4_t *>(out_ptr_base + out_row_of(row) * out_head_stride + col) = out_sums;
        };
        auto store_meta = [&](const int row, const float max_new, const float sum_new) {
            if (split) {
                const int offset = batch_idx * num_heads * balanced_seq_len_tile
                                   + (head_q_index + out_row_of(row)) * balanced_seq_len_tile + partition_idx;
                reinterpret_cast<float *>(out_tmp)[offset] = sum_new;
                // 此处存到 global 的 max 经过 scale_softmax_log2
                reinterpret_cast<float *>(out_tmp + max_tmp_offset)[offset] = max_new * (scale_softmax_log2 * q_scale);
            }
        };
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
        direct_epilogue<Mloop, HEAD_SIZE_V, NUM_WARPS, Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t), direct_b16>(
#else
        direct_epilogue<Mloop, HEAD_SIZE_V, NUM_WARPS, Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)>(
#endif
            scores_max, scores_sum, acc_o, shared_mem, epi_rows, warp_id, scale_softmax_log2 * q_scale, store_o, store_meta);
        if (split && combine_counters != nullptr) {
            direct_combine_arrive(combine_counters + ((batch_idx * num_kv_heads + kv_head_idx) * NUM_Q_TILES + q_tile_idx)
                                  * DIRECT_COMBINE_COUNTER_STRIDE);
        }
    } else {
        float scores_max_new[Mloop];
        float scores_sum_new[Mloop];

        epilogue_rescale_acco<Mloop, HEAD_SIZE_V, NUM_WARPS>(scores_max, scores_sum, acc_o, scores_max_new, scores_sum_new,
                                                            smem_max, smem_sum, scale_softmax_log2, warp_id, local_num_heads, num_kv_heads);

        // 当前 acc_o 为经过缩放结果, 需要将4个warp的 acc_o 加起来得到 PARTION_SIZE 的输出
        //  通过lds reduce实现
        {
            const int acc_row = lane_id & 15;
            const int acc_col = lane_id >> 4;
            // 每个kv head处理的token q数量
            const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
            // 每个kv head对应的q head数量
            int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
            // 当前block处理的q head起始索引, 即当前kv_head_id对应的q_head_id_start
            int head_q_index = q_heads_per_kv_head * kv_head_idx;

            int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
            int tile_start = q_tile_idx * rows_per_qtile;
            int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

            // 若 1 == valid_num_partition, 对应的 out 的形状是 (batch, seqlen_q * num_q_heads, head_dim_v), scalar_t
            // 若 valid_num_partition > 1, 对应的 out_buffer 的形状是(batch, seqlen_q * num_q_heads, max_num_partitions, head_dim_v) scalar_t
            scalar_t *out_ptr_base = nullptr;
            int out_head_stride = 0;
            // num_heads = seqlen_q * num_q_heads
            // out_ptr_base 考虑了 batch 维度, head 维度起始head索引的偏移, 没有考虑 seqlen 维度的偏移
            if (1 == valid_num_partition){
                out_head_stride = HEAD_SIZE_V;
                out_ptr_base = out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
            }else{
                out_head_stride = balanced_seq_len_tile  * HEAD_SIZE_V;
                // out_tmp_offset 元素, 为预留的 max sum buffer
                out_ptr_base = out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                                + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V;
            }

            // lds reduce 每个迭代处理 NUM_WARPS * 16x16 *4 的数据, 保证每个线程 b128 的写出, 且每个warp需要写
            constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
            // lds 形状为(NUM_WARPS, WARP_SIZE * 4)
            float *smem_reduce = (float *)shared_mem;
            for(int tile_idx = 0;tile_idx < num_16x64_tiles;tile_idx++){
                const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
                const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
                for(int j = 0;j < 4;j++){
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 0) = (acc_o[tile_idx_row][tile_idx_col * 4 + 0])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 1) = (acc_o[tile_idx_row][tile_idx_col * 4 + 1])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 2) = (acc_o[tile_idx_row][tile_idx_col * 4 + 2])[j];
                    *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 3) = (acc_o[tile_idx_row][tile_idx_col * 4 + 3])[j];
                }
                __syncthreads();

                int out_r2g_row = threadIdx.x >> 4;
                int out_r2g_col = threadIdx.x & 15;
                int local_row = tile_idx_row * 16 + out_r2g_row;

                // 边界限制
                if(local_row < tile_rows){
                    // 组内第 row 行对应 token row / q_heads_per_kv_head 的第 head_q_index + row % q_heads_per_kv_head 个 q head
                    // (例: 16 个 q head、2 个 token、2 个 KV head 时, 每组 16 行: 0-7 为 token0、8-15 为 token1 的本组 8 个 head)
                    int global_row = tile_start + local_row;

                    const int out_row = (global_row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + global_row % q_heads_per_kv_head;
                    const int our_col = tile_idx_col * 64 + out_r2g_col * 4;
                    float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f,};
                    for(int i = 0;i < NUM_WARPS;i++){
                        float4_t temp_value = *((float4_t *)(smem_reduce + i * 16 * (64 + REDUCE_LDS_PAD) + out_r2g_row * (64 + REDUCE_LDS_PAD) + out_r2g_col * 4));
                        sums[0] += temp_value[0];
                        sums[1] += temp_value[1];
                        sums[2] += temp_value[2];
                        sums[3] += temp_value[3];
                    }

                    scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]), from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
                    *((scalarx4_t *)(out_ptr_base + out_row * out_head_stride + our_col)) = out_sums;
                }
                 __syncthreads();
            }

            // 保存 max sum
            if(valid_num_partition > 1){
                // max, sum 形状为 (batch, seqlen_q, num_q_heads, max_num_partitions)
                if(warp_id == 0 && acc_col == 0){
                    for(int m = 0; m < Mloop;m++){
                        int local_row  = m * 16 + acc_row;
                        if(local_row < tile_rows){
                            int global_row = tile_start + local_row;
                            int out_row = (global_row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + global_row % q_heads_per_kv_head;
                            int offset = batch_idx * num_heads * balanced_seq_len_tile
                                          + (head_q_index + out_row) * balanced_seq_len_tile
                                          + partition_idx;
                            float *exp_sums = reinterpret_cast<float*>(out_tmp);
                            // max_tmp_offset 元素单位
                            float *max = reinterpret_cast<float*>(out_tmp + max_tmp_offset);

                            exp_sums[offset] = scores_sum_new[m];
                            // 此处存到 global 的 max 经过 scale_softmax_log2
                            max[offset] = scores_max_new[m]  * scale_softmax_log2;
                        }
                    }
                }
            }
        }

    }

    return;
}

void launch(
                                        torch::Tensor& out,                                 // [num_seqs, seqlen_q, num_q_heads, head_dim_v]
                                        torch::Tensor& query,                               // [num_seqs, seqlen_q, num_q_heads, head_dim_qk]
                                        torch::Tensor& key_cache,                           // [num_blocks, num_kv_heads, block_size, head_dim_qk]
                                        torch::Tensor& value_cache,                         // [num_blocks, num_kv_heads, block_size, head_dim_v]
                                        double scale,
                                        torch::Tensor& block_tables,                        // [num_seqs, max_num_blocks_per_seq]
                                        torch::Tensor& seq_lens,                            // [num_seqs]
                                        const c10::optional<torch::Tensor>& alibi_slopes,   // [num_q_heads]
                                        const std::string& kv_cache_dtype,
                                        const c10::optional<torch::Tensor>& q_scale,
                                        const c10::optional<torch::Tensor>& k_scale,
                                        const c10::optional<torch::Tensor>& v_scale,
                                        int max_seq_len,
                                        const c10::optional<at::Tensor> &s_aux,             // [seqlen_q * num_q_heads]
                                        float *out_tmp,
                                        bool print_param
                                    )
{
    (void)scale; (void)kv_cache_dtype; (void)q_scale; (void)k_scale; (void)v_scale;

    int batch = query.size(0);
    int mtp = query.size(1);
    int num_q_heads = query.size(2);
    int head_dim_qk = query.size(3);

    int num_blocks = key_cache.size(0);
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
    int block_size = key_cache.size(2);
    int num_kv_heads = key_cache.size(1);

    int head_dim_v = value_cache.size(2);
#else
    int block_size = key_cache.size(1);
    int num_kv_heads = key_cache.size(2);

    int head_dim_v = value_cache.size(3);
#endif
    int max_num_blocks_per_seq = block_tables.size(1);
    int num_heads = num_q_heads * mtp;

    int real_reuse_times = num_heads / num_kv_heads;
    int reuse_kv=get_reusekv(num_heads, num_kv_heads);

    const BhsdRuntimeTuning &runtime_tuning = bhsd_runtime_tuning();
    // Limit partition overrides and four-Q-tile selection to the supported
    // BF16-Q/E5M2-KV direct path. Q_TILES=0 or 2 preserves candidate 04.
    const bool runtime_tuning_supported =
        query.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16 &&
        key_cache.scalar_type() == torch::kFloat8_e5m2 &&
        value_cache.scalar_type() == torch::kFloat8_e5m2 &&
        head_dim_qk == head_dim_v && block_size == 64 &&
        (head_dim_qk == 64 || head_dim_qk == 128 || head_dim_qk == 256);
    // The scheduling budget and actual launch share this exact decision.
    const bool use_three_q_tiles = runtime_tuning.q_tiles == 0 && key_cache.scalar_type() == query.scalar_type() &&
        head_dim_qk == head_dim_v && (head_dim_qk == 192 || head_dim_qk == 256) && reuse_kv == 48 && real_reuse_times % 3 == 0;
#if PA_B16_92A_LAYOUT == PA_LAYOUT_L0
    const bool use_four_q_tiles = (runtime_tuning.q_tiles == 4 && runtime_tuning_supported &&
        head_dim_qk == 256 && reuse_kv > 32 && real_reuse_times % 4 == 0) ||
        (runtime_tuning.q_tiles == 0 && key_cache.scalar_type() == query.scalar_type() &&
         head_dim_qk == 256 && head_dim_v == 256 && reuse_kv > 32 && real_reuse_times % 4 == 0);
#else
    const bool use_four_q_tiles = runtime_tuning.q_tiles == 4 && runtime_tuning_supported &&
        head_dim_qk == 256 && reuse_kv > 32 && real_reuse_times % 4 == 0;
#endif
    // D256 每组 33..64 行 (BF16 Q): 行切分 kernel, K/V 每个 chunk 只读一次; PA_BHSD_Q_TILES=2/4 保留原 Q tile 路径
    const bool rowsplit_d256_enabled = head_dim_qk == 256 && reuse_kv > 32 && query.scalar_type() == torch::kBFloat16 &&
        !(runtime_tuning_supported && runtime_tuning.q_tiles != 0);
    // D256 每组 17..32 行 (原两个 Q tile): 行切分路径; PA_BHSD_Q_TILES=2/4 保留原 Q tile 路径
    const bool rowsplit_d256q32_enabled = head_dim_qk == 256 && reuse_kv > 16 && reuse_kv <= 32 && real_reuse_times % 2 == 0 &&
        ((num_q_heads / num_kv_heads) * head_dim_qk * 2) % 2048 == 0 && query.stride(1) == num_q_heads * head_dim_qk &&
        query.scalar_type() == torch::kBFloat16 && !(runtime_tuning_supported && runtime_tuning.q_tiles != 0);
    // D128 每组 33..64 行: 行切分路径; PA_BHSD_Q_TILES=2/4 保留原 Q tile 路径
    const bool rowsplit_d128_enabled = head_dim_qk == 128 && reuse_kv > 32 && query.scalar_type() == torch::kBFloat16 &&
        ((num_q_heads / num_kv_heads) * head_dim_qk * 2) % 2048 == 0 && query.stride(1) == num_q_heads * head_dim_qk &&
        !(runtime_tuning_supported && runtime_tuning.q_tiles != 0);

    static int nums_simd = get_nums_simd();
    int target_blocks = nums_simd;
    int base_blocks = batch * num_kv_heads;
    int balanced_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(target_blocks, base_blocks));
    const bool direct_b16_kv = key_cache.scalar_type() == query.scalar_type() && value_cache.scalar_type() == query.scalar_type() &&
        head_dim_qk == head_dim_v && (head_dim_qk == 64 || head_dim_qk == 128 || head_dim_qk == 192 || head_dim_qk == 256);
    if (((query.scalar_type() == torch::kBFloat16 || query.scalar_type() == torch::kFloat16) &&
        key_cache.scalar_type() == torch::kFloat8_e5m2 &&
        value_cache.scalar_type() == torch::kFloat8_e5m2 &&
        head_dim_qk == head_dim_v) || direct_b16_kv) {
        // Scale the split budget with the device and batch/KV-head count.
        // Power-of-two splits balance the common power-of-two page counts.
        // Q tiles also produce CTAs; include them in the total scheduling budget.
        int num_q_tiles = (reuse_kv > 32 || (head_dim_qk >= 192 && reuse_kv > 16 && real_reuse_times % 2 == 0)) ? 2 : 1;
        // Keep the D=256 output accumulator within 64 VGPRs per thread.
        // Four Q tiles also count toward the shared CTA scheduling budget.
        if (use_four_q_tiles) {
            num_q_tiles = 4;
        }
        if (use_three_q_tiles) {
            num_q_tiles = 3;
        }
        if (head_dim_qk == 64 || head_dim_qk == 128 || head_dim_qk == 192 || head_dim_qk == 256) {
            // FP8 直接加载路径: 每个 warp 跨 page 流水, 每个 CU 一个线程块即可接近饱和;
            // 每页 MMA 量 (Mloop * head_dim) 较大时每个 CU 放两个线程块, 使 MMA 与另一线程块的访存/解码重叠。
            // 线程块数取 CU 数的整数倍避免尾部; 设备端再按每个 warp 的 page 数裁剪有效 partition。
            const int num_cus = MAX_T(1, nums_simd / 4);
            const int mloop = DIVIDE_ROUND_UP(DIVIDE_ROUND_UP(reuse_kv, num_q_tiles), 16);
            const int ctas_per_cu = MIN_T(2, MAX_T(1, mloop * head_dim_qk / 128));
            balanced_seq_len_tile = MAX_T(1, ctas_per_cu * num_cus / (base_blocks * num_q_tiles));
            // D256 单 Q tile 单 M 块: 1.5 倍 CU 数 (中等长度实测略快于 2 倍)
            if (head_dim_qk == 256 && num_q_tiles == 1 && mloop == 1) balanced_seq_len_tile = MAX_T(1, 3 * num_cus / (2 * base_blocks));
        } else {
            const int budget = MAX_T(1, DIVIDE_ROUND_UP(2 * nums_simd, base_blocks * num_q_tiles));
            balanced_seq_len_tile = 1;
            while (balanced_seq_len_tile <= budget / 2) {
                balanced_seq_len_tile *= 2;
            }
        }
    }

    if (runtime_tuning.partitions != 0 && runtime_tuning_supported) {
        // out_tmp comes from the dispatcher's existing allocation. Validate
        // every temporary region before any kernel can use the override.
        check_bhsd_partition_workspace(batch, num_heads, head_dim_v, runtime_tuning.partitions);
        balanced_seq_len_tile = runtime_tuning.partitions;
    }

    int q_stride = query.stride(0);
    int k_block_stride = key_cache.stride(0);
    int v_block_stride = value_cache.stride(0);

    TORCH_CHECK(0 == (head_dim_qk % 64) && head_dim_qk <= 256, "Page Attention head size must be 64, 128, 192 or 256");
    TORCH_CHECK(0 == (head_dim_v % 64) && head_dim_v <= 256, "Page Attention head size must be 64, 128, 192 or 256");
    TORCH_CHECK(num_heads <= (num_kv_heads * 64), "Page Attention qheads*mtp/kvheads must be smaller than 64");
    TORCH_CHECK(64 == block_size, "Page Attention opt only supprot block_size 64 for now!");

    void *query_ptr = query.data_ptr();
    void *ker_cache_ptr = key_cache.data_ptr();
    void *value_cache_ptr = value_cache.data_ptr();
    void *out_ptr = out.data_ptr();
    int *block_tables_ptr = block_tables.data_ptr<int>();
    int *seq_lens_ptr = seq_lens.data_ptr<int>();

    const void *alibi_slopes_ptr = nullptr;
    if(alibi_slopes.has_value()){
        auto alibi_slopes_value = alibi_slopes.value();
        TORCH_CHECK((torch::kFloat32 == alibi_slopes_value.dtype()), "alibi_slopes must be float!");
        CHECK_SHAPE(alibi_slopes_value, num_q_heads);
        CHECK_DEVICE(alibi_slopes_value);
        CHECK_CONTIGUOUS(alibi_slopes_value);
        alibi_slopes_ptr = alibi_slopes_value.data_ptr();
    }

    const void *s_aux_ptr = nullptr;
    if(s_aux.has_value()){
        auto s_aux_value = s_aux.value();
        TORCH_CHECK(s_aux_value.dtype() == query.dtype(),
                    "s_aux must have the same dtype as query. Got s_aux dtype: ", s_aux_value.dtype(),
                    ", query dtype: ", query.dtype());
        TORCH_CHECK(s_aux_value.dtype() == torch::kFloat16 || s_aux_value.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match query). Got: ", s_aux_value.dtype());
        TORCH_CHECK(num_heads <= 64,
                    "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux_value);
        CHECK_SHAPE(s_aux_value, num_heads);
        CHECK_CONTIGUOUS(s_aux_value);
        s_aux_ptr = s_aux_value.data_ptr();
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    HEADSIZE_SWITCH(head_dim_qk, [&] {
        Output_Type_SWITCH(out.dtype(), [&] {
            Cache_Type_SWITCH(scalar_t, key_cache.dtype(), [&] {
                REUSEKV_SWITCH(reuse_kv, [&] {

                    constexpr bool SINK_ENABLE = false;
                    constexpr bool ALIBI_ENABLE = false;
                    constexpr bool MTP_ENABLE = true;

                    constexpr int STAGES = 2;
                    constexpr int BLOCK_SIZE = 64;
                    // HEAD_SIZE scalar_t cache_t is_e4m3 REUSE_KV_TIMES 编译常量已经在宏中定义
                    constexpr static int NUM_THREADS = 256;
                    constexpr static int NUM_WARPS = NUM_THREADS / WARP_SIZE;
                    constexpr int HEAD_SIZE_V = HEAD_SIZE;

                    // Padding 缓解 bc
                    constexpr int Q_LDS_PAD = 8;
                    constexpr int REDUCE_LDS_PAD = 4;

                    TORCH_CHECK(2 == sizeof(scalar_t), "PA only support scalar_t b16 type for now!");
                    TORCH_CHECK(false == is_e4m3, "PA onlyt sopprt b8 is e5m2 for now!");

                    // 直接加载路径在同一次启动内合并 partition (grid 的 partition 维追加合并线程块), 省去合并 kernel;
                    // 组数超过计数器容量时退回合并 kernel
                    constexpr bool direct_path = DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any;
                    // Split the Q work when its output accumulator alone would occupy 128 VGPRs.
                    constexpr bool prefer_split_q = direct_path &&
                        DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16) * HEAD_SIZE_V >= 384;
                    const bool single_q_tile = REUSE_KV_TIMES <= 32 && !(prefer_split_q && real_reuse_times % 2 == 0);
                    constexpr bool prefer_four_q_tiles = prefer_split_q && !is_e4m3 && HEAD_SIZE_V == 256 && REUSE_KV_TIMES > 32;
                    constexpr bool three_q_tiles_ok = !std::is_same<cache_t, uint8_t>::value && REUSE_KV_TIMES == 48 &&
                        (HEAD_SIZE == 192 || HEAD_SIZE == 256) && HEAD_SIZE_V == HEAD_SIZE;
                    const int launch_q_tiles = single_q_tile ? 1 :
                        ((three_q_tiles_ok && use_three_q_tiles) ? 3 : ((prefer_four_q_tiles && use_four_q_tiles) ? 4 : 2));
                    unsigned int *combine_counters = nullptr;
                    if (runtime_tuning.combine == 0 && direct_path && balanced_seq_len_tile > 1 && batch * num_kv_heads * launch_q_tiles <= DIRECT_COMBINE_MAX_GROUPS) {
                        combine_counters = direct_combine_counters(query.get_device(), stream);
                    }
                    const int combine_ctas = combine_counters != nullptr ? DIVIDE_ROUND_UP(num_heads / num_kv_heads, launch_q_tiles) : 0;
                    if (print_param) {
                        printf("PA_BHSD_TUNING q_tiles=%d partitions=%d combine=%d supported=%d effective_q_tiles=%d partition_budget=%d fused_combine=%d\n",
                               runtime_tuning.q_tiles, runtime_tuning.partitions, runtime_tuning.combine,
                               runtime_tuning_supported, launch_q_tiles, balanced_seq_len_tile, combine_counters != nullptr);
                    }
                    // 行切分路径: partition 预算 (每 CU ctas 个线程块, 可由 PA_BHSD_PARTITIONS 覆盖) 与同一次启动内合并的计数器
                    // (Graph 捕获期间 direct_combine_counters 返回空, 退回合并 kernel)
                    auto rowsplit_budget_of = [&](int ctas) {
                        return (runtime_tuning.partitions != 0 && runtime_tuning_supported)
                            ? runtime_tuning.partitions : MAX_T(1, ctas * MAX_T(1, nums_simd / 4) / (batch * num_kv_heads));
                    };
                    auto rowsplit_counters_of = [&](int budget) {
                        return (runtime_tuning.combine == 0 && budget > 1 && batch * num_kv_heads <= DIRECT_COMBINE_MAX_GROUPS)
                            ? direct_combine_counters(query.get_device(), stream) : nullptr;
                    };
                    // BSHD: 行切分路径 (BHSD 连续 2KB 整块加载) 不使用
                    if constexpr (REUSE_KV_TIMES <= 32) {
                      if (single_q_tile) {
                        int smem_q_size = REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t);
                        // 每次处理 4 个 16x16 的 acc 结果
                        // smem_reduce 的形状是 [NUM_WARPS, 16 * 16 * 4]
                        int smem_reduce_size = NUM_WARPS * 16 * (16 * 4 + REDUCE_LDS_PAD) * sizeof(float);
                        int smem_v_size = (DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any) ? 0 : NUM_WARPS * 2 * 8 * 128 * sizeof(uint8_t);
                        // 两个 smem 空间复用, 取较大值
                        int smem_size = MAX_T(smem_q_size + smem_v_size, smem_reduce_size);
                        if constexpr (DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any) {
                            smem_size = DirectEpilogueShape<DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16), HEAD_SIZE_V, NUM_WARPS>::smem_bytes(smem_q_size, REUSE_KV_TIMES);
                        }
                        // 长序列出口的行切分模式需要的 LDS
                        smem_size = MAX_T(smem_size, BhsdLongExit<scalar_t, cache_t, is_e4m3, HEAD_SIZE, REUSE_KV_TIMES, 1>::RING_BYTES);

                        dim3 grid_size(num_kv_heads, balanced_seq_len_tile + combine_ctas, batch);
                        dim3 kernel_block_size(NUM_THREADS);
                        if(print_param){
                            constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
                            printf("is_e4m3 [%d], is_query_fp16 [%d], HEAD_SIZE [%d], BLOCK_SIZE [%d], REUSE_KV_TIMES [%d], batch [%d], mtp [%d], "
                                    "num_q_heads [%d], head_dim_qk [%d], num_blocks [%d], num_kv_heads [%d], head_dim_v [%d], "
                                    "max_num_blocks_per_seq [%d], max_seq_len [%d], sizeof(scalar_t) [%d], sizeof(cache_t) [%d], "
                                    " balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d]\n",
                                    is_e4m3, is_query_fp16, HEAD_SIZE, BLOCK_SIZE, REUSE_KV_TIMES, batch, mtp, num_q_heads, head_dim_qk,
                                    num_blocks, num_kv_heads, head_dim_v, max_num_blocks_per_seq, max_seq_len, sizeof(scalar_t),
                                    sizeof(cache_t), balanced_seq_len_tile, grid_size.x, grid_size.y, grid_size.z);
                        }
                        paged_attention_kernel_bhsd_opt<scalar_t, cache_t, is_e4m3, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, NUM_THREADS, REUSE_KV_TIMES, STAGES,
                                                    Q_LDS_PAD, REDUCE_LDS_PAD, SINK_ENABLE, ALIBI_ENABLE, MTP_ENABLE>
                            <<<grid_size, kernel_block_size, smem_size, stream>>>(
                                reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<scalar_t *>(out_tmp), reinterpret_cast<scalar_t *>(query_ptr),
                                reinterpret_cast<cache_t *>(ker_cache_ptr), reinterpret_cast<cache_t *>(value_cache_ptr), num_heads, num_kv_heads,
                                block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq, reinterpret_cast<const float*>(alibi_slopes_ptr), q_stride,
                                k_block_stride, v_block_stride, balanced_seq_len_tile, reinterpret_cast<const scalar_t*>(s_aux_ptr), mtp, combine_counters);
                        if(balanced_seq_len_tile > 1 && combine_counters == nullptr &&
                           !(BhsdLongExit<scalar_t, cache_t, is_e4m3, HEAD_SIZE, REUSE_KV_TIMES, 1>::enabled &&
                             bhsd_long_exit_combine(HEAD_SIZE, REUSE_KV_TIMES, 1, out_ptr, out_tmp, seq_lens_ptr, balanced_seq_len_tile,
                                                    num_heads, batch, reinterpret_cast<void *>(stream)))){
                            constexpr bool direct_combine = DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any;
                            constexpr int combine_ctas_per_cu = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16)>::CTAS_PER_CU;
                            constexpr bool combine_long_single_stream = combine_ctas_per_cu == 2 && DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16) * HEAD_SIZE <= 256;
                            constexpr bool combine_resident_cap = direct_combine && combine_long_single_stream;
                            launch_paged_attention_combine<scalar_t, HEAD_SIZE_V, BLOCK_SIZE, direct_combine, combine_ctas_per_cu, combine_long_single_stream, combine_resident_cap, false>(
                                reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<scalar_t *>(out_tmp), seq_lens_ptr,
                                balanced_seq_len_tile, num_heads, batch, stream);
                        }
                        return;
                      }
                    }
                    if constexpr (REUSE_KV_TIMES > 32 || prefer_split_q) {
                        auto launch_tiled = [&](auto tile_count) {
                        constexpr int NUM_Q_TILES = decltype(tile_count)::value;
                        // 每个Q维度的线程块处理的 REUSE_KV_TIMES
                        constexpr int Q_TILE = DIVIDE_ROUND_UP(REUSE_KV_TIMES, NUM_Q_TILES);

                        TORCH_CHECK(0 == (num_heads / num_kv_heads) % NUM_Q_TILES);

                        const dim3 grid_qtiles(num_kv_heads * NUM_Q_TILES, balanced_seq_len_tile + combine_ctas, batch);
                        const dim3 block_qtiles(NUM_THREADS);

                        int smem_q_size = Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t);
                        int smem_reduce_size = NUM_WARPS * 16 * (16 * 4 + REDUCE_LDS_PAD) * sizeof(float);
                        int smem_v_size = (DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any) ? 0 : NUM_WARPS * 2 * 8 * 128 * sizeof(uint8_t);
                        int tile_smem_size = MAX_T(smem_q_size + smem_v_size, smem_reduce_size);
                        if constexpr (DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any) {
                            tile_smem_size = DirectEpilogueShape<DIVIDE_ROUND_UP(Q_TILE, 16), HEAD_SIZE_V, NUM_WARPS>::smem_bytes(smem_q_size, Q_TILE);
                        }

                        if(print_param){
                            constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
                            printf("is_e4m3 [%d], is_query_fp16 [%d], HEAD_SIZE [%d], BLOCK_SIZE [%d], REUSE_KV_TIMES [%d], batch [%d], mtp [%d], "
                                    "num_q_heads [%d], head_dim_qk [%d], num_blocks [%d], num_kv_heads [%d], head_dim_v [%d], "
                                    "max_num_blocks_per_seq [%d], max_seq_len [%d], sizeof(scalar_t) [%d], sizeof(cache_t) [%d], "
                                    " balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d]\n",
                                    is_e4m3, is_query_fp16, HEAD_SIZE, BLOCK_SIZE, REUSE_KV_TIMES, batch, mtp, num_q_heads, head_dim_qk,
                                    num_blocks, num_kv_heads, head_dim_v, max_num_blocks_per_seq, max_seq_len, sizeof(scalar_t),
                                    sizeof(cache_t), balanced_seq_len_tile, grid_qtiles.x, grid_qtiles.y, grid_qtiles.z);
                        }

                        paged_attention_kernel_qtile_bhsd_opt<scalar_t, cache_t, is_e4m3, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, NUM_THREADS, Q_TILE, STAGES,
                                Q_LDS_PAD, REDUCE_LDS_PAD, SINK_ENABLE, ALIBI_ENABLE, MTP_ENABLE, NUM_Q_TILES, Q_TILE>
                            <<<grid_qtiles, block_qtiles, tile_smem_size, stream>>>(
                                reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<scalar_t *>(out_tmp), reinterpret_cast<scalar_t *>(query_ptr),
                                reinterpret_cast<cache_t *>(ker_cache_ptr), reinterpret_cast<cache_t *>(value_cache_ptr), num_heads, num_kv_heads,
                                block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq, reinterpret_cast<const float*>(alibi_slopes_ptr), q_stride,
                                k_block_stride, v_block_stride, balanced_seq_len_tile, reinterpret_cast<const scalar_t*>(s_aux_ptr), mtp, combine_counters);
                        if(balanced_seq_len_tile > 1 && combine_counters == nullptr){
                            constexpr bool direct_combine = DirectKV<cache_t, HEAD_SIZE, HEAD_SIZE_V>::any;
                            constexpr int combine_ctas_per_cu = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, DIVIDE_ROUND_UP(Q_TILE, 16)>::CTAS_PER_CU;
                            constexpr bool combine_long_single_stream = false;
                            constexpr bool combine_resident_cap = direct_combine && HEAD_SIZE == 256 && combine_ctas_per_cu == 2;
                            constexpr bool combine_one_page_slice = combine_resident_cap && NUM_Q_TILES == 2 && Q_TILE >= 24;
                            launch_paged_attention_combine<scalar_t, HEAD_SIZE_V, BLOCK_SIZE, direct_combine, combine_ctas_per_cu, combine_long_single_stream, combine_resident_cap, combine_one_page_slice>(
                                reinterpret_cast<scalar_t *>(out_ptr), reinterpret_cast<scalar_t *>(out_tmp), seq_lens_ptr,
                                balanced_seq_len_tile, num_heads, batch, stream);
                        }
                        };
                        if constexpr (three_q_tiles_ok) {
                            if (launch_q_tiles == 3) {
                                // 非模板函数中 if constexpr 的丢弃分支仍会实例化, 不需要时取已有的 2 (与 4 个 Q tile 的写法相同)
                                launch_tiled(std::integral_constant<int, three_q_tiles_ok ? 3 : 2>{});
                                return;
                            }
                        }
                        if constexpr (prefer_four_q_tiles) {
                            if (launch_q_tiles == 4) {
                                launch_tiled(std::integral_constant<int, prefer_four_q_tiles ? 4 : 2>{});
                            } else {
                                launch_tiled(std::integral_constant<int, 2>{});
                            }
                        } else {
                            launch_tiled(std::integral_constant<int, 2>{});
                        }
                    }
                });
            });
        });
    });

    return;
}


#undef PA_B16_92A_LAYOUT
// 本文件内定义的辅助宏
#undef PA_ROWSPLIT_COMBINE_CTA
#undef PA_ROWSPLIT_LDS_BASES
#undef PA_ROWSPLIT_CHUNK
#undef PA_ROWSPLIT_STORE_ROW
#undef PA_Q_RANGE_REDUCE
#undef PA_Q_FP16_SCALE
#undef PA_Q_REG_TO_FP16
#undef PA_COMBINE_PARTIALS
#undef PA_EXIT_JUMP
