// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// gfx938: teacher version (05). gfx92a: implementation at the end of the common header.
#include <unordered_map>
#include <cmath>
#include "paged_attention_refactoring_bhsd_fp8_common.h"
using namespace native_bhsd_fp8_opt;

// gfx936 handoff idea, adapted to native E4M3 operands: load the exact lane's
// K fragment directly, retaining FP8 MMAC and the original score/token mapping.
using bhsd_fp8_rsrc_t = int __attribute__((ext_vector_type(4)));
using bhsd_fp8_dwords_t = uint32_t __attribute__((ext_vector_type(4)));
__device__ bhsd_fp8_dwords_t bhsd_fp8_buffer_load(bhsd_fp8_rsrc_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

template <int HEAD_SIZE, int BLOCK_SIZE>
__device__ __forceinline__ bhsd_fp8_rsrc_t bhsd_fp8_page_rsrc(const uint8_t* page)
{
    const uint64_t address = reinterpret_cast<uint64_t>(page);
    return {static_cast<int>(address), static_cast<int>(address >> 32),
            BLOCK_SIZE * HEAD_SIZE, 0x00020000};
}

template <int Mloop, int HEAD_SIZE, int Q_LDS_PAD>
__device__ __forceinline__ void bhsd_fp8_cache_q(
    const uint8_t* smem_q, int q_rows,
    uint8x16_t (&cached_q)[Mloop][HEAD_SIZE / 64])
{
    // Keep Q live across the KV loop only for modest accumulator footprints.
    // Larger tiles retain the LDS path to protect occupancy and avoid spills.
    if constexpr (Mloop * HEAD_SIZE <= 256) {
        const int lane = threadIdx.x % WARP_SIZE;
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int row = m * 16 + (lane & 15);
            #pragma unroll
            for (int chunk = 0; chunk < HEAD_SIZE / 64; ++chunk) {
                cached_q[m][chunk] = row < q_rows
                    ? *reinterpret_cast<const uint8x16_t*>(smem_q + row * (HEAD_SIZE + Q_LDS_PAD) + chunk * 64 + (lane >> 4) * 16)
                    : uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }
}

template <int Mloop, int HEAD_SIZE, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD,
          bool is_e4m3, int Q_LDS_PAD>
__device__ __forceinline__ void qk_direct_lane_bhsd_fp8(
    const uint8_t* k_ptr, const uint8_t* smem_q, int q_rows,
    const uint8x16_t (&cached_q)[Mloop][HEAD_SIZE / 64],
    float4_t s_reg[Mloop][BLOCK_SIZE / 16], int block_idx, int valid_seqlen_kv)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane & 15;
    const int group = lane >> 4;
    const auto resource = bhsd_fp8_page_rsrc<HEAD_SIZE, BLOCK_SIZE>(k_ptr);
    constexpr int CHUNKS = HEAD_SIZE / 64;
    // Large output accumulator tiles retain the single-buffer register budget.
    // Use accumulator footprint rather than a fixed head-size rule.
    constexpr int AHEAD = Mloop * HEAD_SIZE >= 512 ? 1 : 2;
    uint8x16_t fragments[AHEAD][NUM_KV_TOKENS_PER_LOAD / 16];
    auto load_chunk = [&](int chunk, int slot) {
        #pragma unroll
        for (int i = 0; i < NUM_KV_TOKENS_PER_LOAD / 16; ++i) {
            const int offset = (4 * row + i) * HEAD_SIZE + group * 16 + chunk * 64;
            fragments[slot][i] = __builtin_bit_cast(uint8x16_t,
                bhsd_fp8_buffer_load(resource, offset, 0, 0));
        }
    };
    load_chunk(0, 0);
    #pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        const int slot = chunk % AHEAD;
        if constexpr (AHEAD == 2) {
            if (chunk + 1 < CHUNKS) load_chunk(chunk + 1, (chunk + 1) % AHEAD);
        }
        uint8x16_t q_fragment[Mloop];
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int q_row = m * 16 + row;
            if constexpr (Mloop * HEAD_SIZE <= 256) {
                q_fragment[m] = cached_q[m][chunk];
            } else {
                q_fragment[m] = q_row < q_rows
                    ? *reinterpret_cast<const uint8x16_t*>(smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + chunk * 64 + group * 16)
                    : uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
        if constexpr (AHEAD == 2) {
            if (chunk + 1 < CHUNKS) inline_wait_vmcnt(NUM_KV_TOKENS_PER_LOAD / 16);
            else inline_wait_vmcnt(0);
        } else inline_wait_vmcnt(0);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const intx2* q_vec = reinterpret_cast<const intx2*>(&q_fragment[m]);
            #pragma unroll
            for (int i = 0; i < NUM_KV_TOKENS_PER_LOAD / 16; ++i) {
                const intx2* k_vec = reinterpret_cast<const intx2*>(&fragments[slot][i]);
                builtin_amdgcn_mmac_fp8<is_e4m3>(q_vec[0], k_vec[0], s_reg[m][i]);
                builtin_amdgcn_mmac_fp8<is_e4m3>(q_vec[1], k_vec[1], s_reg[m][i]);
            }
        }
        if constexpr (AHEAD == 1) {
            if (chunk + 1 < CHUNKS) load_chunk(chunk + 1, 0);
        }
    }
    if ((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE / 16; ++i) {
                const int token = block_idx * BLOCK_SIZE + group * 4 + i;
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (token + j * 16 >= valid_seqlen_kv) s_reg[m][i][j] = -INFINITY;
            }
        }
    }
}


template <typename qtypex8_t, typename q_type, int HEAD_SIZE, int NUM_THREADS, int Q_LDS_PAD, bool is_e4m3>
__forceinline__ __device__ void prefetch_q_to_lds_fp8(const q_type *q,
                                                   uint8_t *smem_q,
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
    const q_type *q_ptr = q + batch_idx * q_stride + q_head_idx * HEAD_SIZE;
    int seqlen_q_per_kv_head = num_heads / num_kv_heads;
    int q_seqlen_stride = num_q_heads * HEAD_SIZE;
    int q_seqlen_data_num = num_q_heads_per_kv_head * HEAD_SIZE;

    for(int i = tid * 8;i < seqlen_q_per_kv_head * HEAD_SIZE;i += NUM_THREADS * 8){
        int q_seqlen_idx = i / q_seqlen_data_num;
        int q_offset = q_seqlen_idx * q_seqlen_stride + i % q_seqlen_data_num;
        const int q_row = i / HEAD_SIZE;
        const int q_col = i % HEAD_SIZE;

        intx2 packed;
        if constexpr (std::is_same<q_type, uint8_t>::value) {
            packed = *reinterpret_cast<const intx2*>(q_ptr + q_offset);
        }else{
            // qtype = b16
            const qtypex8_t v = *reinterpret_cast<const qtypex8_t*>(q_ptr + q_offset);
            packed[0] = to_f8_from_f32<is_e4m3>(to_float(v[0]), to_float(v[1]), to_float(v[2]), to_float(v[3]));
            packed[1] = to_f8_from_f32<is_e4m3>(to_float(v[4]), to_float(v[5]), to_float(v[6]), to_float(v[7]));
        }
        
        *reinterpret_cast<intx2*>(smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + q_col) = packed;
    }
    __syncthreads();
}

template <typename qtypex8_t, typename q_type, int HEAD_SIZE, int NUM_THREADS, int Q_LDS_PAD, bool is_e4m3, int NUM_Q_TILES>
__forceinline__ __device__ void prefetch_q_to_lds_fp8_qtile(const q_type *q,
                                                   uint8_t *smem_q,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int mtp,
                                                   int kv_head_idx,
                                                   int batch_idx,
                                                   int q_stride,
                                                   int q_tile_idx
                                                )
{
    int tid = threadIdx.x;
    int num_q_heads = num_heads / mtp;
    int num_q_heads_per_kv_head = num_q_heads / num_kv_heads;
    int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
    const q_type *q_ptr = q + batch_idx * q_stride + q_head_idx * HEAD_SIZE;
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

        intx2 packed;
        if constexpr (std::is_same<q_type, uint8_t>::value) {
            packed = *reinterpret_cast<const intx2*>(q_ptr + q_offset);
        }else{
            // qtype = b16
            const qtypex8_t v = *reinterpret_cast<const qtypex8_t*>(q_ptr + q_offset);
            packed[0] = to_f8_from_f32<is_e4m3>(to_float(v[0]), to_float(v[1]), to_float(v[2]), to_float(v[3]));
            packed[1] = to_f8_from_f32<is_e4m3>(to_float(v[4]), to_float(v[5]), to_float(v[6]), to_float(v[7]));
        }
        
        *reinterpret_cast<intx2*>(smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + q_col) = packed;
    }
    __syncthreads();
}

template <int Mloop, int HEAD_SIZE, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, int STAGES, bool is_e4m3, int Q_LDS_PAD = 0, int NUM_Q_TILES = 1>
__forceinline__ __device__ void qk_gemm_func_with_prefetch_k_bhsd_fp8(const uint8_t *k_ptr,
                                                  const uint8_t *smem_q,
                                                  int seqlen_q_per_kv_head,
                                                  const uint8x16_t (&cached_q)[Mloop][HEAD_SIZE / 64],
                                                  float4_t s_reg[Mloop][BLOCK_SIZE / 16],
                                                  int block_idx,
                                                  int valid_seqlen_kv
                                                )
{
    qk_direct_lane_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_e4m3, Q_LDS_PAD>(k_ptr, smem_q, seqlen_q_per_kv_head, cached_q, s_reg, block_idx, valid_seqlen_kv);
}

// Retain the legacy entry point; the shared helper now chooses K buffering
// from accumulator footprint, including one-stage loads for large Q tiles.
template <int Mloop, int HEAD_SIZE, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD,
          bool is_e4m3, int Q_LDS_PAD = 0, int NUM_Q_TILES = 1>
__forceinline__ __device__ void qk_gemm_func_single_buffer_k_bhsd_fp8(
    const uint8_t *k_ptr,
    const uint8_t *smem_q,
    int seqlen_q_per_kv_head,
    const uint8x16_t (&cached_q)[Mloop][HEAD_SIZE / 64],
    float4_t s_reg[Mloop][BLOCK_SIZE / 16],
    int block_idx,
    int valid_seqlen_kv)
{
    qk_direct_lane_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_e4m3, Q_LDS_PAD>(k_ptr, smem_q, seqlen_q_per_kv_head, cached_q, s_reg, block_idx, valid_seqlen_kv);
}

template <int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, int STAGES, bool is_e4m3=false, int NUM_Q_TILES = 1>
__device__ __forceinline__  void pv_gemm_func_with_prefetch_v_bhsd_fp8(const uint8_t *v_ptr,
                                                    intx2 p_reg[Mloop][BLOCK_SIZE / 32],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                    int block_idx,
                                                    int valid_seqlen_kv
                                                  )
{   
    // 每个 warp 处理一个 block_page
    // v_cache 每个 warp 一次加载 64x32
    // p 矩阵每个线程持有 4 个连续的元素, 对应的 v 每个线程 持有同一列的4个连续数据
    int stage_id = 0;
    // 假设 NUM_KV_TOKENS_PER_LOAD = 64, 则[4][2], [4] 代表加载 64 行, [2] 代表加载 32 列
    uint8x4_t v_reg[NUM_KV_TOKENS_PER_LOAD / 16][STAGES * 2];
    // 此处假设 STAGES = 2
    // v global -> lds 预取
    int v_tile = 0;
    int lane_id = threadIdx.x % WARP_SIZE;
    // 0-3, 4-7 线程连续 global->register, 下方通过bpermute 转成 mmac 需要的布局
    int v_g2r_row = lane_id & 3;       // 0, 1, 2, 3
    int v_g2r_col = lane_id >> 2;       // 0-15
    {
        const uint8_t *v_tile_ptr = v_ptr + v_tile * 32;
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const uint8_t *cur_v_ptr =v_tile_ptr + i * 16 * HEAD_SIZE_V + v_g2r_row * 4 * HEAD_SIZE_V + (j * 16 + v_g2r_col);
                (v_reg[i][stage_id * 2 + j])[0] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 0);
                (v_reg[i][stage_id * 2 + j])[1] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 1 * HEAD_SIZE_V);
                (v_reg[i][stage_id * 2 + j])[2] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 2 * HEAD_SIZE_V);
                (v_reg[i][stage_id * 2 + j])[3] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 3 * HEAD_SIZE_V);
            }
        }
    }

    // 切换到第二个buffer
    stage_id ^= 1;

    // Keep accumulator and ping-pong buffer indices constant after inlining.
    #pragma unroll
    for(v_tile = 1; v_tile < (HEAD_SIZE_V / 32);v_tile++){
        const uint8_t *v_tile_ptr = v_ptr + v_tile * 32;
        // 取下一份数据
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const uint8_t *cur_v_ptr =v_tile_ptr + i * 16 * HEAD_SIZE_V + v_g2r_row * 4 * HEAD_SIZE_V + (j * 16 + v_g2r_col);
                (v_reg[i][stage_id * 2 + j])[0] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 0);
                (v_reg[i][stage_id * 2 + j])[1] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 1 * HEAD_SIZE_V);
                (v_reg[i][stage_id * 2 + j])[2] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 2 * HEAD_SIZE_V);
                (v_reg[i][stage_id * 2 + j])[3] = *reinterpret_cast<const uint8_t *>(cur_v_ptr + 3 * HEAD_SIZE_V);
            }
        }
        // 此时有 2 * (NUM_KV_TOKENS_PER_LOAD / 16) * 2 buffer_loadx2 指令 on air
        // 等待 前 (NUM_KV_TOKENS_PER_LOAD / 16) * 2 个指令完成, 并进行 mmac

        // TODO 屏蔽后 sgpr溢出 待分析
        inline_wait_vmcnt((NUM_KV_TOKENS_PER_LOAD / 16) * 2);
        
        // 切换到 计算索引
        stage_id ^= 1;

        // 假设 NUM_KV_TOKENS_PER_LOAD = 64, 此时加载了 global v 的 64 行
        // 对应到 P 矩阵的 64 列, 当前 仅考虑 BLOCK_SIZE = 64 的场景
        // 若 BLOCK_SIZE = 128, 则 需要写另外的逻辑 加载 global v 的后64 行, 并与 P 的后 64 列对应进行mmac
        // Decode and permute each V fragment once, then reuse it across query tiles.
        for(int j = 0;j < (NUM_KV_TOKENS_PER_LOAD / 32);j++){
            uint8x8_t v_fragment0;
            uint8x8_t v_fragment1;
            // uint8x4_t v_reg[NUM_KV_TOKENS_PER_LOAD / 16][STAGES * 2];
            int *tmp_v0_ptr = reinterpret_cast<int *>(&(v_fragment0));
            int *tmp_v1_ptr = reinterpret_cast<int *>(&(v_fragment1));

            tmp_v0_ptr[0] = __builtin_bit_cast(int, v_reg[2 * j + 0][stage_id * 2 + 0]);
            tmp_v0_ptr[1] = __builtin_bit_cast(int, v_reg[2 * j + 1][stage_id * 2 + 0]);
            tmp_v1_ptr[0] = __builtin_bit_cast(int, v_reg[2 * j + 0][stage_id * 2 + 1]);
            tmp_v1_ptr[1] = __builtin_bit_cast(int, v_reg[2 * j + 1][stage_id * 2 + 1]);

            // v 边界处理
            if ((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv) {
                int token_idx = block_idx * BLOCK_SIZE + j * 32 + v_g2r_row * 4;
                for(int i = 0;i < 4;i++){
                    if((token_idx + i) >= valid_seqlen_kv){
                        v_fragment0[0 + i] = 0;
                        v_fragment1[0 + i] = 0;
                    }
                }
                token_idx += 16;
                for(int i = 0;i < 4;i++){
                    if((token_idx + i) >= valid_seqlen_kv){
                        v_fragment0[4 + i] = 0;
                        v_fragment1[4 + i] = 0;
                    }
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
                // p 已经按照 mmac 布局构造
                builtin_amdgcn_mmac_fp8<is_e4m3>(p_reg[m][j], __builtin_bit_cast(intx2, v_fragment0), acc_o[m][(v_tile - 1) * 2 + 0]);
                builtin_amdgcn_mmac_fp8<is_e4m3>(p_reg[m][j], __builtin_bit_cast(intx2, v_fragment1), acc_o[m][(v_tile - 1) * 2 + 1]);
            }
        }
    }
    // 最后待计算的 buffer id
    int last_stage_id = stage_id ^ 1;

    // TODO 屏蔽后 sgpr溢出 待分析
    inline_wait_vmcnt(0);
    
    // The final prefetched tile uses the same V-fragment reuse.
    for(int j = 0;j < (NUM_KV_TOKENS_PER_LOAD / 32);j++){
        uint8x8_t v_fragment0;
        uint8x8_t v_fragment1;
        // uint8x4_t v_reg[NUM_KV_TOKENS_PER_LOAD / 16][STAGES * 2];
        int *tmp_v0_ptr = reinterpret_cast<int *>(&(v_fragment0));
        int *tmp_v1_ptr = reinterpret_cast<int *>(&(v_fragment1));

        tmp_v0_ptr[0] = __builtin_bit_cast(int, v_reg[2 * j + 0][last_stage_id * 2 + 0]);
        tmp_v0_ptr[1] = __builtin_bit_cast(int, v_reg[2 * j + 1][last_stage_id * 2 + 0]);
        tmp_v1_ptr[0] = __builtin_bit_cast(int, v_reg[2 * j + 0][last_stage_id * 2 + 1]);
        tmp_v1_ptr[1] = __builtin_bit_cast(int, v_reg[2 * j + 1][last_stage_id * 2 + 1]);

        // v 边界处理
        if ((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv) {
            int token_idx = block_idx * BLOCK_SIZE + j * 32 + v_g2r_row * 4;
            for(int i = 0;i < 4;i++){
                if((token_idx + i) >= valid_seqlen_kv){
                    v_fragment0[0 + i] = 0;
                    v_fragment1[0 + i] = 0;
                }
            }
            token_idx += 16;
            for(int i = 0;i < 4;i++){
                if((token_idx + i) >= valid_seqlen_kv){
                    v_fragment0[4 + i] = 0;
                    v_fragment1[4 + i] = 0;
                }
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
            // p 已经按照 mmac 布局构造
            builtin_amdgcn_mmac_fp8<is_e4m3>(p_reg[m][j], __builtin_bit_cast(intx2, v_fragment0), acc_o[m][((HEAD_SIZE_V / 32) - 1) * 2 + 0]);
            builtin_amdgcn_mmac_fp8<is_e4m3>(p_reg[m][j], __builtin_bit_cast(intx2, v_fragment1), acc_o[m][((HEAD_SIZE_V / 32) - 1) * 2 + 1]);
        }
    }

    // TODO BLOCK_SIZE = 128 支持
    // half4_t temp_p_reg = {(p_reg[m][4])[j], (p_reg[m][5])[j], (p_reg[m][6])[j], (p_reg[m][7])[j]};
    return;
}

// A matrix-load tile is 32 rows by 64 FP8 columns. Keep two LDS slots per
// wave so a pending load never overwrites the tile currently being consumed.
constexpr int BHSD_FP8_V_TILE_BYTES = 32 * 64;
constexpr int BHSD_FP8_V_LDS_STAGES = 2;

template <int HEAD_SIZE_V, int NUM_KV_TOKENS_PER_LOAD>
__device__ __forceinline__ void bhsd_fp8_prefetch_v_matrix(
    const uint8_t* v_ptr, int tile, uint8_t* tile_lds)
{
    constexpr int ROW_TILES = NUM_KV_TOKENS_PER_LOAD / 32;
    const int col_tile = tile / ROW_TILES;
    const int row_tile = tile % ROW_TILES;
    const uint8_t* tile_ptr = v_ptr + col_tile * 64 + row_tile * 32 * HEAD_SIZE_V;
    uintx4 resource;
    *(uint64_t*)&resource = reinterpret_cast<uint64_t>(tile_ptr) & 0x0000ffffffffffffULL;
    resource[2] = HEAD_SIZE_V;
    resource[3] = 0;
    const int lds_base = reinterpret_cast<size_t>(tile_lds);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 4\n\t"
                 "matrix_load_64x32_b8 %0, %1, moffset:0 t lds\n"
                 :: "s"(resource), "s"(lds_base) : "memory");
    __builtin_amdgcn_sched_barrier(0);
}

template <int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD,
          int STAGES, bool is_e4m3=false, int NUM_Q_TILES = 1>
__device__ __forceinline__ void pv_gemm_func_pipelined_bhsd_fp8(
    uint8_t* smem_v, const uint8_t* v_ptr,
    intx2 p_reg[Mloop][BLOCK_SIZE / 32],
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
    int block_idx, int valid_seqlen_kv, int num_kv_heads)
{
    constexpr int ROW_TILES = NUM_KV_TOKENS_PER_LOAD / 32;
    constexpr int TILES = (HEAD_SIZE_V / 64) * ROW_TILES;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row_group = lane >> 4;
    const int column_lane = lane & 15;
    // Tile 0 is issued before QK/softmax by the caller. For subsequent tiles,
    // load the next slot before MMAC so memory latency overlaps current MMAC.
    #pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        const int col_tile = tile / ROW_TILES;
        const int row_tile = tile % ROW_TILES;
        const uint8_t* tile_lds = smem_v + (tile % BHSD_FP8_V_LDS_STAGES) * BHSD_FP8_V_TILE_BYTES;
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        uint8x8_t v_reg[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int column = i * 16 + column_lane;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row0 = row_group * 4 + j;
                const int row1 = row0 + 16;
                v_reg[i][j] = tile_lds[row0 * 64 + (column ^ ((row0 & 2) << 4))];
                v_reg[i][j + 4] = tile_lds[row1 * 64 + (column ^ ((row1 & 2) << 4))];
            }
        }
        if ((block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int token = block_idx * BLOCK_SIZE + row_tile * 32 + row_group * 4 + j;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (token >= valid_seqlen_kv) v_reg[i][j] = 0;
                    if (token + 16 >= valid_seqlen_kv) v_reg[i][j + 4] = 0;
                }
            }
        }
        if (tile + 1 < TILES) {
            bhsd_fp8_prefetch_v_matrix<HEAD_SIZE_V, NUM_KV_TOKENS_PER_LOAD>(
                v_ptr, tile + 1,
                smem_v + ((tile + 1) % BHSD_FP8_V_LDS_STAGES) * BHSD_FP8_V_TILE_BYTES);
        }
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                builtin_amdgcn_mmac_fp8<is_e4m3>(p_reg[m][row_tile],
                    __builtin_bit_cast(intx2, v_reg[i]), acc_o[m][col_tile * 4 + i]);
            }
        }
    }
}

template <typename scalar_t, typename q_type, bool is_e4m3 , int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE, int NUM_Q_TILES, int Q_TILE>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_bhsd_fp8_qtile_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const q_type* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const uint8_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size_v]
    const uint8_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, head_size_v, block_size]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,       
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const q_type* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp)
{
#if (defined(__gfx938__)||defined(__gfx92a__))
    extern __shared__ char shared_mem[];
    uint8_t *smem_q = (uint8_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);
    uint8_t *smem_v = (uint8_t *)(shared_mem + DIVIDE_ROUND_UP(Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(uint8_t), 128) * 128);
    
    using qtypex8_t= __attribute__( (__vector_size__(8 * sizeof(q_type)) )) q_type;
    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;

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
    // const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z; 

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    int valid_num_partition = get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_scalar_fp16 true, scalar_t = fp16; is_scalar_fp16 false, scalar_t = bf16; 
    constexpr bool is_scalar_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_query_fp8 true, q_type = fp8/e4m3; is_query_fp8 false, q_type = scalar_t
    constexpr bool is_query_fp8 = std::is_same<q_type, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    smem_v += warp_id * BHSD_FP8_V_LDS_STAGES * BHSD_FP8_V_TILE_BYTES;

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

    prefetch_q_to_lds_fp8_qtile<qtypex8_t, q_type, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD, is_e4m3, NUM_Q_TILES>(
        q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, q_tile_idx);

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    const int start_block_idx = static_cast<int>(static_cast<int64_t>(partition_idx) * valid_num_block_pages / valid_num_partition);
    const int end_block_idx  = static_cast<int>(static_cast<int64_t>(partition_idx + 1) * valid_num_block_pages / valid_num_partition);
    // 有效block_page_idx: [start_block_idx, end_block_idx)
    const int num_blocks = end_block_idx - start_block_idx;

    // k_cache 形状为 [num_blocks, num_kv_heads, block_size, head_dim_qk]
    // k_head_stride = BLOCK_SIZE * HEAD_DIM
    // k_ptr_base 没有考虑 num_blocks 维度的偏移
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
    const uint8_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const uint8_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;
    uint8x16_t cached_q[Mloop][HEAD_SIZE / 64];
    bhsd_fp8_cache_q<Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, local_num_heads / num_kv_heads, cached_q);

    
    for(int block_idx = (start_block_idx + warp_id);block_idx < end_block_idx;block_idx += NUM_WARPS){
        // 1个warp处理一个block_page
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
        // 加上 num_blocks 维度的偏移
        const uint8_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

        // v_cache 形状为 [num_blocks, num_heads, head_size, block_size]
        // v_cache 在 block_size 维度连续, pv gemm时, v 为列主序
        // 加上 num_blocks 维度的偏移
        const uint8_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;
        // Hide the first V load behind the independent QK and softmax work.
        bhsd_fp8_prefetch_v_matrix<HEAD_SIZE_V, NUM_KV_TOKENS_PER_LOAD>(v_ptr, 0, smem_v);


        float4_t s_reg[Mloop][BLOCK_SIZE / 16];
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        if constexpr (HEAD_SIZE == 256){
            qk_gemm_func_single_buffer_k_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_e4m3, Q_LDS_PAD, NUM_Q_TILES>
                (k_ptr, smem_q, local_num_heads / num_kv_heads, cached_q, s_reg, block_idx, valid_seqlen_kv);
        }else{
            qk_gemm_func_with_prefetch_k_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, STAGES, is_e4m3, Q_LDS_PAD, NUM_Q_TILES>
                (k_ptr, smem_q, local_num_heads / num_kv_heads, cached_q, s_reg, block_idx, valid_seqlen_kv);
        }
        
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
            // Only pages that can overlap the causal tail need per-element checks.
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
    
        // s_reg: S = 2^(S * softmax_scale_log2 - max_current * softmax_scale_log2) = e^(S * softmax_scale - max_current * softmax_scale)
        // scores_max = new_max, raw max, 没有 乘  scale_softmax_log2
        // scores_sum: 对应公式 d(i) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale) +  row_sum(e^((S - max_current) * softmax_scale))
        // acc_o: O(i-1) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, block_idx != start_block_idx + warp_id);

        intx2 p_reg[Mloop][BLOCK_SIZE / 32];
        // fp32 -> fp8
        // 和 fp32 -> b16 不同, 此处直接构造 fp8_mmac 布局数据
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 32);i++){
                (p_reg[m][i])[0] = to_f8_from_f32<is_e4m3>((s_reg[m][0])[i * 2 + 0],
                                    (s_reg[m][1])[i * 2 + 0],
                                    (s_reg[m][2])[i * 2 + 0],
                                    (s_reg[m][3])[i * 2 + 0]
                                    );
                (p_reg[m][i])[1] = to_f8_from_f32<is_e4m3>((s_reg[m][0])[i * 2 + 1],
                                    (s_reg[m][1])[i * 2 + 1],
                                    (s_reg[m][2])[i * 2 + 1],
                                    (s_reg[m][3])[i * 2 + 1]
                                    );                  
            }
        }

        pv_gemm_func_pipelined_bhsd_fp8<Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, STAGES, is_e4m3, NUM_Q_TILES>(smem_v, v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv, num_kv_heads);
    }

    __syncthreads();

    // sink
    if constexpr(SINK_ENABLE)
    {
        // 对 apply_attention_sink 的理解:
        //      sink 为 人为 在 S 矩阵的每一行增加一个元素, 假设为 sink_value
        //      则 new_max = MAX_T(sink_value, old_max), 根据new_max 更新 scores_max, scores_sum, acc_o
        //      包含一下公式:
        //              max = new_max
        //              d(i) = d(i-1) * e^(m(i - 1) - m(i)) + row_sum(e^(xi(j) - mi))
        //              o(i - 1) = o(i - 1) * e^(m(i - 1) - m(i)), sink 对应的 v 为0, 不考虑 pv 项
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink_q_tile<scalar_t, Mloop, HEAD_SIZE_V, NUM_Q_TILES>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax,
                scale_softmax_log2,
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

    float scores_max_new[Mloop];
    float scores_sum_new[Mloop];

    epilogue_rescale_acco<Mloop, HEAD_SIZE_V, NUM_WARPS>(scores_max, scores_sum, acc_o, scores_max_new, scores_sum_new,
                                                        smem_max, smem_sum, scale_softmax_log2, warp_id, local_num_heads, num_kv_heads);
    
                                                        // 当前 acc_o 为经过缩放结果, 需要将4个warp的 acc_o 加起来得到 PARTION_SIZE 的输出
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
                // row / q_heads_per_kv_head 表示当前行对应 out 的 token id
                //   例如: num_q_heads = 16, seqlen_q = 2, num_kv_heads = 2
                //         则 q_heads_per_kv_head = num_q_heads / num_kv_heads = 8
                //            seqlen_q_per_kv_head = num_q_heads * seqlen_q / num_kv_heads = 16
                //            kv_head_idx = 0时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head0-7, 8-15 对应 token1 的 head0-7
                //            kv_head_idx = 1时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head8-16, 8-15 对应 token1 的 head8-15
                
                //      num_kv_heads * q_heads_per_kv_head 为 num_q_heads
                //      (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) 含义是 当前 row 对应的 q_token_id 的 q_head 偏移
                //      row % q_heads_per_kv_head + head_q_index, 对应当前行的实际的 q_head 的索引
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

    return;
#endif
}

template <typename scalar_t, typename q_type, bool is_e4m3 , int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_bhsd_fp8_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const q_type* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const uint8_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size_v]
    const uint8_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, head_size_v, block_size]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,       
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const q_type* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp)
{
#if (defined(__gfx938__)||defined(__gfx92a__))
    extern __shared__ char shared_mem[];
    uint8_t *smem_q = (uint8_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);
    uint8_t *smem_v = (uint8_t *)(shared_mem + DIVIDE_ROUND_UP(REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(uint8_t), 128) * 128);
    
    using qtypex8_t= __attribute__( (__vector_size__(8 * sizeof(q_type)) )) q_type;
    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;

    // grid_size: (num_kv_heads, max_partition, batch_size)
    const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z; 

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    int valid_num_partition = get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_scalar_fp16 true, scalar_t = fp16; is_scalar_fp16 false, scalar_t = bf16; 
    constexpr bool is_scalar_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_query_fp8 true, q_type = fp8/e4m3; is_query_fp8 false, q_type = scalar_t
    constexpr bool is_query_fp8 = std::is_same<q_type, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;
    smem_v += warp_id * BHSD_FP8_V_LDS_STAGES * BHSD_FP8_V_TILE_BYTES;

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

    prefetch_q_to_lds_fp8<qtypex8_t, q_type, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD, is_e4m3>(
        q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride);

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    const int start_block_idx = static_cast<int>(static_cast<int64_t>(partition_idx) * valid_num_block_pages / valid_num_partition);
    const int end_block_idx  = static_cast<int>(static_cast<int64_t>(partition_idx + 1) * valid_num_block_pages / valid_num_partition);
    // 有效block_page_idx: [start_block_idx, end_block_idx)
    const int num_blocks = end_block_idx - start_block_idx;

    // k_cache 形状为 [num_blocks, num_kv_heads, block_size, head_dim_qk]
    // k_head_stride = BLOCK_SIZE * HEAD_DIM
    // k_ptr_base 没有考虑 num_blocks 维度的偏移
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
    const uint8_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const uint8_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;
    uint8x16_t cached_q[Mloop][HEAD_SIZE / 64];
    bhsd_fp8_cache_q<Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, num_heads / num_kv_heads, cached_q);

    
    for(int block_idx = (start_block_idx + warp_id);block_idx < end_block_idx;block_idx += NUM_WARPS){
        // 1个warp处理一个block_page
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
        // 加上 num_blocks 维度的偏移
        const uint8_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

        // v_cache 形状为 [num_blocks, num_heads, head_size, block_size]
        // v_cache 在 block_size 维度连续, pv gemm时, v 为列主序
        // 加上 num_blocks 维度的偏移
        const uint8_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;
        // Hide the first V load behind the independent QK and softmax work.
        bhsd_fp8_prefetch_v_matrix<HEAD_SIZE_V, NUM_KV_TOKENS_PER_LOAD>(v_ptr, 0, smem_v);


        float4_t s_reg[Mloop][BLOCK_SIZE / 16];
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        if constexpr (HEAD_SIZE == 256){
            qk_gemm_func_single_buffer_k_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_e4m3, Q_LDS_PAD>
                (k_ptr, smem_q, num_heads / num_kv_heads, cached_q, s_reg, block_idx, valid_seqlen_kv);
        }else{
            qk_gemm_func_with_prefetch_k_bhsd_fp8<Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, STAGES, is_e4m3, Q_LDS_PAD>
                (k_ptr, smem_q, num_heads / num_kv_heads, cached_q, s_reg, block_idx, valid_seqlen_kv);
        }
        
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
            // Only pages that can overlap the causal tail need per-element checks.
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
    
        // s_reg: S = 2^(S * softmax_scale_log2 - max_current * softmax_scale_log2) = e^(S * softmax_scale - max_current * softmax_scale)
        // scores_max = new_max, raw max, 没有 乘  scale_softmax_log2
        // scores_sum: 对应公式 d(i) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale) +  row_sum(e^((S - max_current) * softmax_scale))
        // acc_o: O(i-1) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, block_idx != start_block_idx + warp_id);

        intx2 p_reg[Mloop][BLOCK_SIZE / 32];
        // fp32 -> fp8
        // 和 fp32 -> b16 不同, 此处直接构造 fp8_mmac 布局数据
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 32);i++){
                (p_reg[m][i])[0] = to_f8_from_f32<is_e4m3>((s_reg[m][0])[i * 2 + 0],
                                    (s_reg[m][1])[i * 2 + 0],
                                    (s_reg[m][2])[i * 2 + 0],
                                    (s_reg[m][3])[i * 2 + 0]
                                    );
                (p_reg[m][i])[1] = to_f8_from_f32<is_e4m3>((s_reg[m][0])[i * 2 + 1],
                                    (s_reg[m][1])[i * 2 + 1],
                                    (s_reg[m][2])[i * 2 + 1],
                                    (s_reg[m][3])[i * 2 + 1]
                                    );                  
            }
        }

        pv_gemm_func_pipelined_bhsd_fp8<Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, STAGES, is_e4m3>(smem_v, v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv, num_kv_heads);
    }

    __syncthreads();

    // sink
    if constexpr(SINK_ENABLE)
    {
        // 对 apply_attention_sink 的理解:
        //      sink 为 人为 在 S 矩阵的每一行增加一个元素, 假设为 sink_value
        //      则 new_max = MAX_T(sink_value, old_max), 根据new_max 更新 scores_max, scores_sum, acc_o
        //      包含一下公式:
        //              max = new_max
        //              d(i) = d(i-1) * e^(m(i - 1) - m(i)) + row_sum(e^(xi(j) - mi))
        //              o(i - 1) = o(i - 1) * e^(m(i - 1) - m(i)), sink 对应的 v 为0, 不考虑 pv 项
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink<scalar_t, Mloop, HEAD_SIZE_V>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax,
                scale_softmax_log2,
                warp_id,
                partition_idx,
                mtp,
                num_heads,
                num_kv_heads,
                kv_head_idx);
        }
    }

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
                // row / q_heads_per_kv_head 表示当前行对应 out 的 token id
                //   例如: num_q_heads = 16, seqlen_q = 2, num_kv_heads = 2
                //         则 q_heads_per_kv_head = num_q_heads / num_kv_heads = 8
                //            seqlen_q_per_kv_head = num_q_heads * seqlen_q / num_kv_heads = 16
                //            kv_head_idx = 0时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head0-7, 8-15 对应 token1 的 head0-7
                //            kv_head_idx = 1时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head8-16, 8-15 对应 token1 的 head8-15
                
                //      num_kv_heads * q_heads_per_kv_head 为 num_q_heads
                //      (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) 含义是 当前 row 对应的 q_token_id 的 q_head 偏移
                //      row % q_heads_per_kv_head + head_q_index, 对应当前行的实际的 q_head 的索引
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

    return;
#endif
}

// ===========================================================================
// Pipelined native-E4M3 BHSD kernel (gfx938).
//  * Work unit is a 32-token chunk. Each wave streams chunks of its partition;
//    the K/V loads of the next chunk are issued while the current one computes,
//    so no wave waits on more than one memory latency per chunk.
//  * K is loaded straight into MMAC fragment layout. V is loaded row-major into
//    registers and transposed through a small LDS slot with ds_read_matrix.
//  * Q is loaded from global memory into fragment registers (no LDS).
//  * A CTA owns 16*MW query rows of one KV head; larger groups use more CTAs.
// ===========================================================================
__device__ uint32_t bhsd_fp8_buffer_load32(bhsd_fp8_rsrc_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
using bhsd_fp8_dwords2_t = uint32_t __attribute__((ext_vector_type(2)));
__device__ bhsd_fp8_dwords2_t bhsd_fp8_buffer_load64(bhsd_fp8_rsrc_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v2i32");
namespace bhsd_fp8_gfx938 {

// f16 -> f32 as a separate v_cvt_f32_f16 where FP16 partial outputs are accumulated. On gfx938 the compiler
// otherwise folds the conversion of the high half of a packed FP16 pair into v_fma_mix_f32 op_sel, which reads the
// low half: FP16 outputs merged across partitions then repeat every even column in the following odd one.
template <typename T>
__device__ __forceinline__ float combine_to_float(T x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<T, _Float16>::value) {
        float f;
        asm("v_cvt_f32_f16 %0, %1" : "=v"(f) : "v"(x));
        return f;
    }
#endif
    return to_float(x);
}


constexpr int CHUNK = 32;
constexpr int V_TILE_BYTES = 32 * 64;

__device__ __forceinline__ bhsd_fp8_rsrc_t make_rsrc(const uint8_t* base, int bytes)
{
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    return {static_cast<int>(a), static_cast<int>(a >> 32), bytes, 0x00020000};
}

__device__ __forceinline__ uint8x16_t load16(bhsd_fp8_rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(uint8x16_t, bhsd_fp8_buffer_load(r, voff, soff, 0));
}

// Large Q tiles stay in LDS so the accumulator, K and V buffers fit in VGPRs.
__host__ __device__ constexpr bool q_in_lds(int d, int mw) { return d * mw > 256; }
__host__ __device__ constexpr int q_lds_stride(int d) { return d + 16; }

// Epilogue merge width: one 16-row x RED_COLS block of every wave per round.
__host__ __device__ constexpr int red_cols(int waves, int d) { return (waves == 4 && d % 128 == 0) ? 128 : 64; }

__host__ __device__ constexpr int lds_bytes(int waves, int stage, int mw, int d, bool compact = false)
{
    // Main loop: V staging per wave (+ Q tile). Epilogue: max/sum exchange plus
    // the merge block; the fused combine reuses the same space for its weights.
    return (waves * stage * V_TILE_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0)) >
           (compact ? waves * 16 * red_cols(waves, d) * 4 : (2 * waves * 16 * mw * 4 + 64 + waves * 16 * (red_cols(waves, d) + 4) * 4))
        ? (waves * stage * V_TILE_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0))
        : (compact ? waves * 16 * red_cols(waves, d) * 4 : (2 * waves * 16 * mw * 4 + 64 + waves * 16 * (red_cols(waves, d) + 4) * 4));
}

// Partitions actually used: at most one CTA per CU (one_wave_parts) unless the
// sequence is long enough for the wider grid to pay for its extra partials.
__device__ __forceinline__ int valid_partitions(int chunks, int grid_parts, int one_wave_parts, int min_chunks)
{
    if (grid_parts > one_wave_parts && chunks >= 32 * grid_parts) return grid_parts;
    return MAX_T(1, MIN_T(one_wave_parts, DIVIDE_ROUND_UP(chunks, min_chunks)));
}

template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true, bool COMPACT = false>
__device__ __forceinline__ void pa_bhsd_fp8_body(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
{
#if defined(__gfx938__) || defined(__gfx92a__)
#if defined(__gfx92a__)
    // gfx92a 不启动本命名空间的 kernel (Q 全部由 bhsd_fp8_gfx92a 处理), 函数体只在 gfx938 上编译
    if constexpr (sizeof(q_type) == 0) {
#endif
    extern __shared__ char shared_mem[];
    constexpr int KC = D / 64;
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = blockIdx.x % row_groups;
    const int part = blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int chunks = DIVIDE_ROUND_UP(valid, CHUNK);
    const int nparts = valid_partitions(chunks, grid_parts, one_wave_parts, min_chunks);
    if (part >= nparts) return;

    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int r = lane & 15;
    const int g = lane >> 4;
    const int hpk = num_q_heads / num_kv_heads;
    const int R = hpk * mtp;
    const int row0 = rg * ROWS;
    const int num_heads = num_q_heads * mtp;

    const int c_begin = static_cast<int>(static_cast<int64_t>(part) * chunks / nparts);
    const int c_end = static_cast<int>(static_cast<int64_t>(part + 1) * chunks / nparts);

    const int* table = block_tables + static_cast<int64_t>(b) * max_num_blocks_per_seq;
    const uint8_t* k_head = k_cache + kv_head * (64 * D);
    const uint8_t* v_head = v_cache + kv_head * (64 * D);
    auto chunk_k = [&](int c, int phys) {
        return k_head + static_cast<int64_t>(phys) * k_block_stride + (c & 1) * (CHUNK * D);
    };
    auto chunk_v = [&](int c, int phys) {
        return v_head + static_cast<int64_t>(phys) * v_block_stride + (c & 1) * (CHUNK * D);
    };

    // K fragment rows: tile i column n holds token 16*i + 4*(n&3) + (n>>2), so the
    // P fragment of a 32-token chunk packs straight from the two score tiles.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * D + 16 * g;
    // V rows: load (T, h) covers rows 16h..16h+15 of 64-dim column tile T.
    const int v_row = lane >> 2;
    const int v_voff = v_row * D + 16 * (lane & 3);
    const int v_lds = v_row * 64 + ((16 * (lane & 3)) ^ ((v_row & 2) << 4));

    uint8x16_t kf[2][KC];
    uint8x16_t vf[KC][2];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_k(c, phys), CHUNK * D);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 16 * i * D + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_v(c, phys), CHUNK * D);
        #pragma unroll
        for (int t = 0; t < KC; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) vf[t][h] = load16(rs, v_voff, 16 * h * D + 64 * t);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[c >> 1]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * STAGE * V_TILE_BYTES;
    auto load_q16 = [&](int gr, int col) {
        uint8x16_t v = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            const q_type* qp = q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col;
            if constexpr (std::is_same<q_type, uint8_t>::value) {
                v = *reinterpret_cast<const uint8x16_t*>(qp);
            } else {
                using qx8 = __attribute__((__vector_size__(8 * sizeof(q_type)))) q_type;
                const qx8 a0 = *reinterpret_cast<const qx8*>(qp);
                const qx8 a1 = *reinterpret_cast<const qx8*>(qp + 8);
                intx4 w;
                w[0] = to_f8_from_f32<is_e4m3>(to_float(a0[0]), to_float(a0[1]), to_float(a0[2]), to_float(a0[3]));
                w[1] = to_f8_from_f32<is_e4m3>(to_float(a0[4]), to_float(a0[5]), to_float(a0[6]), to_float(a0[7]));
                w[2] = to_f8_from_f32<is_e4m3>(to_float(a1[0]), to_float(a1[1]), to_float(a1[2]), to_float(a1[3]));
                w[3] = to_f8_from_f32<is_e4m3>(to_float(a1[4]), to_float(a1[5]), to_float(a1[6]), to_float(a1[7]));
                v = __builtin_bit_cast(uint8x16_t, w);
            }
        }
        return v;
    };
    uint8x16_t qf[QL ? 1 : MW][QL ? 1 : KC];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 16); i += S * WARP_SIZE) {
            const int row = i / (D / 16), col = (i % (D / 16)) * 16;
            *reinterpret_cast<uint8x16_t*>(q_lds + row * QS + col) = load_q16(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) qf[m][kc] = load_q16(row0 + m * 16 + r, 64 * kc + 16 * g);
    }
    auto q_frag = [&](int m, int kc) {
        if constexpr (QL) return *reinterpret_cast<const uint8x16_t*>(q_lds + (m * 16 + r) * QS + 64 * kc + 16 * g);
        else return qf[m][kc];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
    float4_t acc[MW][D / 16];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_mem) + warp * STAGE * V_TILE_BYTES;

    for (; c < c_end; c += S) {
        const int next = c + S;
        const int next_phys = next < c_end ? table[next >> 1] : 0;
        const int base_tok = c * CHUNK;
        // ---- S = Q K^T over the 32-token chunk.
        float4_t s[MW][2];
        #pragma unroll
        for (int m = 0; m < MW; ++m) { s[m][0] = {0.f, 0.f, 0.f, 0.f}; s[m][1] = {0.f, 0.f, 0.f, 0.f}; }
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CHUNK - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = MAX_T(MAX_T(MAX_T(s[m][0][0], s[m][0][1]), MAX_T(s[m][0][2], s[m][0][3])),
                             MAX_T(MAX_T(s[m][1][0], s[m][1][1]), MAX_T(s[m][1][2], s[m][1][3])));
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            const float alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[m][i][j] = __builtin_amdgcn_exp2f(s[m][i][j] * scale_log2 - base);
                    sum += s[m][i][j];
                }
            sum += __shfl_xor_tmp(sum, 32);
            sum += __shfl_swap16(sum);
            row_sum[m] = row_sum[m] * alpha + sum;
            row_max[m] = new_max;
            #pragma unroll
            for (int i = 0; i < D / 16; ++i) acc[m][i] *= alpha;
            p[m][0] = to_f8_from_f32<is_e4m3>(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3]);
            p[m][1] = to_f8_from_f32<is_e4m3>(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3]);
        }

        // ---- V: zero tokens past the sequence end, stage, transpose-read, MMAC.
        if (base_tok + CHUNK > valid) {
            #pragma unroll
            for (int t = 0; t < KC; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    if (base_tok + 16 * h + v_row >= valid) vf[t][h] = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int t0 = 0; t0 < KC; t0 += STAGE) {
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                uint8_t* slot = stage + (t - t0) * V_TILE_BYTES;
                *reinterpret_cast<uint8x16_t*>(slot + v_lds) = vf[t][0];
                *reinterpret_cast<uint8x16_t*>(slot + 1024 + v_lds) = vf[t][1];
            }
            asm volatile("" ::: "memory");
            if (t0 + STAGE >= KC && next < c_end) issue_v(next, next_phys);
            intx4 frag[STAGE][2];
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                int8_t* slot = reinterpret_cast<int8_t*>(stage + (t - t0) * V_TILE_BYTES);
                frag[t - t0][0] = __builtin_hcu_ds_read_matrix_format_u8(slot, 0, 2, 2, 1);
                frag[t - t0][1] = __builtin_hcu_ds_read_matrix_format_u8(slot + 32, 0, 2, 2, 1);
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const intx2* fv = reinterpret_cast<const intx2*>(&frag[t - t0][half]);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[0], acc[m][t * 4 + half * 2 + 0]);
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[1], acc[m][t * 4 + half * 2 + 1]);
                    }
                }
        }
    }

    // ---- Merge the S waves of this CTA (same rows, disjoint chunks).
    __syncthreads();
    float* ms_max = reinterpret_cast<float*>(shared_mem);
    float* ms_sum = ms_max + S * ROWS;
    // The following barrier ends all max/sum reads before compact reuse.
    float* red = COMPACT ? reinterpret_cast<float*>(shared_mem) : ms_sum + S * ROWS + 16;
    if (g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            ms_max[warp * ROWS + m * 16 + r] = row_max[m];
            ms_sum[warp * ROWS + m * 16 + r] = row_sum[m];
        }
    }
    __syncthreads();
    float fin_max[MW], fin_sum[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        float M = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < S; ++w) M = MAX_T(M, ms_max[w * ROWS + m * 16 + r]);
        float L = 0.0f;
        #pragma unroll
        for (int w = 0; w < S; ++w)
            L += ms_sum[w * ROWS + m * 16 + r] * __builtin_amdgcn_exp2f((ms_max[w * ROWS + m * 16 + r] - M) * scale_log2);
        fin_max[m] = M;
        fin_sum[m] = L;
        const float f = __builtin_amdgcn_exp2f((row_max[m] - M) * scale_log2) / (L + 1e-6f);
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] *= f;
    }

    const bool final_out = nparts == 1;
    constexpr int RC = red_cols(S, D);
    constexpr int RS = COMPACT ? RC : RC + 4;
    const int tid = threadIdx.x;
    using sx4 = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    using sx8 = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    auto head_of = [&](int gr) { return (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk; };
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        #pragma unroll
        for (int blk = 0; blk < D / RC; ++blk) {
            __syncthreads();
            // acc tile (2*b32 + f) holds dims 32*b32 + 8j + 2g + f.
            #pragma unroll
            for (int b32 = 0; b32 < RC / 32; ++b32)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        red[(warp * 16 + r) * RS + (COMPACT ? ((32 * b32 + 8 * j + 2 * g + f) ^ (r * 4)) : (32 * b32 + 8 * j + 2 * g + f))] =
                            acc[m][(blk * (RC / 32) + b32) * 2 + f][j];
            __syncthreads();
            if constexpr (COMPACT && D == 256 && MW == 2 && S == 4 && !FUSE && is_e4m3 && std::is_same<scalar_t, uint16_t>::value && std::is_same<q_type, uint16_t>::value) {
                // Keep FP32 LDS payloads, original per-element wave order,
                // and the same BF16 rounding point; pair two output vectors.
                for (int item = tid; item < 16 * RC / 8; item += S * WARP_SIZE) {
                    const int rr = item / (RC / 8);
                    const int d8 = (item % (RC / 8)) * 8;
                    float4_t sum_lo = {0.f, 0.f, 0.f, 0.f};
                    float4_t sum_hi = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int w = 0; w < S; ++w) {
                        // Separate 16-byte reads preserve XOR row swizzling
                        // and alignment; one float8 read would not do so.
                        sum_lo += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + (d8 ^ (rr * 4)));
                        sum_hi += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + ((d8 + 4) ^ (rr * 4)));
                    }
                    const int gr = row0 + m * 16 + rr;
                    if (gr < R) {
                        const int head_all = head_of(gr);
                        sx8 o;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            o[e] = from_float<scalar_t>(sum_lo[e]);
                            o[e + 4] = from_float<scalar_t>(sum_hi[e]);
                        }
                        scalar_t* dst = final_out
                            ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                            : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                        *reinterpret_cast<sx8*>(dst + blk * RC + d8) = o;
                    }
                }
            } else {
            for (int item = tid; item < 16 * RC / 4; item += S * WARP_SIZE) {
                const int rr = item / (RC / 4);
                const int d4 = (item % (RC / 4)) * 4;
                float4_t sum4 = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w)
                    sum4 += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + (COMPACT ? (d4 ^ (rr * 4)) : d4));
                const int gr = row0 + m * 16 + rr;
                if (gr < R) {
                    const int head_all = head_of(gr);
                    sx4 o = {from_float<scalar_t>(sum4[0]), from_float<scalar_t>(sum4[1]),
                             from_float<scalar_t>(sum4[2]), from_float<scalar_t>(sum4[3])};
                    scalar_t* dst = final_out
                        ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                        : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                    *reinterpret_cast<sx4*>(dst + blk * RC + d4) = o;
                }
            }
            }
        }
    }
    if (final_out) return;
    if (warp == 0 && g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            const int gr = row0 + m * 16 + r;
            if (gr < R) {
                const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(gr)) * grid_parts + part;
                reinterpret_cast<float*>(out_tmp)[off] = fin_sum[m];
                reinterpret_cast<float*>(out_tmp + max_tmp_offset)[off] = fin_max[m] * scale_log2;
            }
        }
    }

    if constexpr (MW > 1 || !FUSE) return;
    // ---- Fused split-K combine: the last CTA of this (batch, head, row group) merges.
    __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(shared_mem);
    const int slot = (b * num_kv_heads + kv_head) * row_groups + rg;
    if (tid == 0) flag[0] = atomicAdd(counters + slot, 1);
    __syncthreads();
    if (flag[0] != nparts - 1) return;
    __threadfence();
    const int rows_here = MIN_T(ROWS, R - row0);
    float* wts = reinterpret_cast<float*>(shared_mem) + 16;     // [rows][nparts]
    // Partials written by other CTAs: read through L2 (glc), never a stale L1 line.
    float* sums = reinterpret_cast<float*>(out_tmp);
    float* maxs = reinterpret_cast<float*>(out_tmp + max_tmp_offset);
    auto ld_l2 = [](float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
    for (int row = warp; row < rows_here; row += S) {
        const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(row0 + row)) * grid_parts;
        float M = -INFINITY;
        for (int i = lane; i < nparts; i += WARP_SIZE) M = MAX_T(M, ld_l2(maxs + off + i));
        for (int mask = 32; mask > 0; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
        float L = 0.0f;
        for (int i = lane; i < nparts; i += WARP_SIZE) {
            const float sv = ld_l2(sums + off + i);
            const float w = sv > 0.0f ? sv * __builtin_amdgcn_exp2f(ld_l2(maxs + off + i) - M) : 0.0f;
            wts[row * nparts + i] = w;
            L += w;
        }
        for (int mask = 32; mask > 0; mask >>= 1) L += __shfl_xor(L, mask);
        const float inv = __fdividef(1.0f, L + 1e-6f);
        for (int i = lane; i < nparts; i += WARP_SIZE) wts[row * nparts + i] *= inv;
    }
    __syncthreads();
    // Keep the buffer descriptor uniform across the entire CTA; only the byte
    // offset varies by lane. A per-row descriptor serializes vector loads.
    const auto partial_rsrc = make_rsrc(reinterpret_cast<const uint8_t*>(out_tmp + out_tmp_offset),
        (b + 1) * num_heads * grid_parts * D * sizeof(scalar_t));
    // Spread the merge across all waves even for a four-row query group.
    // Four-byte (D=128) or eight-byte (D=256) vectors give one row per wave.
    constexpr int OV = D == 128 ? 2 : 4;
    using sxv = __attribute__((__vector_size__(OV * sizeof(scalar_t)))) scalar_t;
    for (int item = tid; item < rows_here * (D / OV); item += S * WARP_SIZE) {
        const int row = item / (D / OV);
        const int col = (item % (D / OV)) * OV;
        const int head_all = head_of(row0 + row);
        const int partial_base = ((b * num_heads + head_all) * grid_parts * D + col) * sizeof(scalar_t);
        const float* w = wts + row * nparts;
        float accv[OV] = {};
        constexpr int ILP = 8;
        for (int i0 = 0; i0 < nparts; i0 += ILP) {
            sxv x[ILP];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int offset = partial_base + MIN_T(i0 + u, nparts - 1) * D * sizeof(scalar_t);
                if constexpr (OV == 2)
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load32(partial_rsrc, offset, 0, 1));
                else
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load64(partial_rsrc, offset, 0, 1));
            }
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const float wi = i0 + u < nparts ? w[i0 + u] : 0.0f;
                #pragma unroll
                for (int e = 0; e < OV; ++e) accv[e] += combine_to_float(x[u][e]) * wi;
            }
        }
        sxv o;
        #pragma unroll
        for (int e = 0; e < OV; ++e) o[e] = from_float<scalar_t>(accv[e]);
        *reinterpret_cast<sxv*>(out + (static_cast<int64_t>(b) * num_heads + head_all) * D + col) = o;
    }
    if (tid == 0) atomicExch(counters + slot, 0);
#if defined(__gfx92a__)
    }
#endif
#endif
}



template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true>
__launch_bounds__(S * WARP_SIZE)
__global__ void pa_bhsd_fp8(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
{
#if defined(__gfx938__) || defined(__gfx92a__)
#if defined(__gfx92a__)
    // gfx92a 不启动本命名空间的 kernel (Q 全部由 bhsd_fp8_gfx92a 处理), 函数体只在 gfx938 上编译
    if constexpr (sizeof(q_type) == 0) {
#endif
    extern __shared__ char shared_mem[];
    constexpr int KC = D / 64;
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = blockIdx.x % row_groups;
    const int part = blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int chunks = DIVIDE_ROUND_UP(valid, CHUNK);
    const int nparts = valid_partitions(chunks, grid_parts, one_wave_parts, min_chunks);
    if (part >= nparts) return;

    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int r = lane & 15;
    const int g = lane >> 4;
    const int hpk = num_q_heads / num_kv_heads;
    const int R = hpk * mtp;
    const int row0 = rg * ROWS;
    const int num_heads = num_q_heads * mtp;

    const int c_begin = static_cast<int>(static_cast<int64_t>(part) * chunks / nparts);
    const int c_end = static_cast<int>(static_cast<int64_t>(part + 1) * chunks / nparts);

    const int* table = block_tables + static_cast<int64_t>(b) * max_num_blocks_per_seq;
    const uint8_t* k_head = k_cache + kv_head * (64 * D);
    const uint8_t* v_head = v_cache + kv_head * (64 * D);
    auto chunk_k = [&](int c, int phys) {
        return k_head + static_cast<int64_t>(phys) * k_block_stride + (c & 1) * (CHUNK * D);
    };
    auto chunk_v = [&](int c, int phys) {
        return v_head + static_cast<int64_t>(phys) * v_block_stride + (c & 1) * (CHUNK * D);
    };

    // K fragment rows: tile i column n holds token 16*i + 4*(n&3) + (n>>2), so the
    // P fragment of a 32-token chunk packs straight from the two score tiles.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * D + 16 * g;
    // V rows: load (T, h) covers rows 16h..16h+15 of 64-dim column tile T.
    const int v_row = lane >> 2;
    const int v_voff = v_row * D + 16 * (lane & 3);
    const int v_lds = v_row * 64 + ((16 * (lane & 3)) ^ ((v_row & 2) << 4));

    uint8x16_t kf[2][KC];
    uint8x16_t vf[KC][2];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_k(c, phys), CHUNK * D);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 16 * i * D + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_v(c, phys), CHUNK * D);
        #pragma unroll
        for (int t = 0; t < KC; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) vf[t][h] = load16(rs, v_voff, 16 * h * D + 64 * t);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[c >> 1]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * STAGE * V_TILE_BYTES;
    auto load_q16 = [&](int gr, int col) {
        uint8x16_t v = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            const q_type* qp = q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col;
            if constexpr (std::is_same<q_type, uint8_t>::value) {
                v = *reinterpret_cast<const uint8x16_t*>(qp);
            } else {
                using qx8 = __attribute__((__vector_size__(8 * sizeof(q_type)))) q_type;
                const qx8 a0 = *reinterpret_cast<const qx8*>(qp);
                const qx8 a1 = *reinterpret_cast<const qx8*>(qp + 8);
                intx4 w;
                w[0] = to_f8_from_f32<is_e4m3>(to_float(a0[0]), to_float(a0[1]), to_float(a0[2]), to_float(a0[3]));
                w[1] = to_f8_from_f32<is_e4m3>(to_float(a0[4]), to_float(a0[5]), to_float(a0[6]), to_float(a0[7]));
                w[2] = to_f8_from_f32<is_e4m3>(to_float(a1[0]), to_float(a1[1]), to_float(a1[2]), to_float(a1[3]));
                w[3] = to_f8_from_f32<is_e4m3>(to_float(a1[4]), to_float(a1[5]), to_float(a1[6]), to_float(a1[7]));
                v = __builtin_bit_cast(uint8x16_t, w);
            }
        }
        return v;
    };
    uint8x16_t qf[QL ? 1 : MW][QL ? 1 : KC];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 16); i += S * WARP_SIZE) {
            const int row = i / (D / 16), col = (i % (D / 16)) * 16;
            *reinterpret_cast<uint8x16_t*>(q_lds + row * QS + col) = load_q16(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) qf[m][kc] = load_q16(row0 + m * 16 + r, 64 * kc + 16 * g);
    }
    auto q_frag = [&](int m, int kc) {
        if constexpr (QL) return *reinterpret_cast<const uint8x16_t*>(q_lds + (m * 16 + r) * QS + 64 * kc + 16 * g);
        else return qf[m][kc];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
    float4_t acc[MW][D / 16];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_mem) + warp * STAGE * V_TILE_BYTES;

    for (; c < c_end; c += S) {
        const int next = c + S;
        const int next_phys = next < c_end ? table[next >> 1] : 0;
        const int base_tok = c * CHUNK;
        // ---- S = Q K^T over the 32-token chunk.
        float4_t s[MW][2];
        #pragma unroll
        for (int m = 0; m < MW; ++m) { s[m][0] = {0.f, 0.f, 0.f, 0.f}; s[m][1] = {0.f, 0.f, 0.f, 0.f}; }
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CHUNK - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = MAX_T(MAX_T(MAX_T(s[m][0][0], s[m][0][1]), MAX_T(s[m][0][2], s[m][0][3])),
                             MAX_T(MAX_T(s[m][1][0], s[m][1][1]), MAX_T(s[m][1][2], s[m][1][3])));
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            const float alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[m][i][j] = __builtin_amdgcn_exp2f(s[m][i][j] * scale_log2 - base);
                    sum += s[m][i][j];
                }
            sum += __shfl_xor_tmp(sum, 32);
            sum += __shfl_swap16(sum);
            row_sum[m] = row_sum[m] * alpha + sum;
            row_max[m] = new_max;
            #pragma unroll
            for (int i = 0; i < D / 16; ++i) acc[m][i] *= alpha;
            p[m][0] = to_f8_from_f32<is_e4m3>(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3]);
            p[m][1] = to_f8_from_f32<is_e4m3>(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3]);
        }

        // ---- V: zero tokens past the sequence end, stage, transpose-read, MMAC.
        if (base_tok + CHUNK > valid) {
            #pragma unroll
            for (int t = 0; t < KC; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    if (base_tok + 16 * h + v_row >= valid) vf[t][h] = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int t0 = 0; t0 < KC; t0 += STAGE) {
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                uint8_t* slot = stage + (t - t0) * V_TILE_BYTES;
                *reinterpret_cast<uint8x16_t*>(slot + v_lds) = vf[t][0];
                *reinterpret_cast<uint8x16_t*>(slot + 1024 + v_lds) = vf[t][1];
            }
            asm volatile("" ::: "memory");
            if (t0 + STAGE >= KC && next < c_end) issue_v(next, next_phys);
            intx4 frag[STAGE][2];
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                int8_t* slot = reinterpret_cast<int8_t*>(stage + (t - t0) * V_TILE_BYTES);
                frag[t - t0][0] = __builtin_hcu_ds_read_matrix_format_u8(slot, 0, 2, 2, 1);
                frag[t - t0][1] = __builtin_hcu_ds_read_matrix_format_u8(slot + 32, 0, 2, 2, 1);
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const intx2* fv = reinterpret_cast<const intx2*>(&frag[t - t0][half]);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[0], acc[m][t * 4 + half * 2 + 0]);
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[1], acc[m][t * 4 + half * 2 + 1]);
                    }
                }
        }
    }

    // ---- Merge the S waves of this CTA (same rows, disjoint chunks).
    __syncthreads();
    float* ms_max = reinterpret_cast<float*>(shared_mem);
    float* ms_sum = ms_max + S * ROWS;
    float* red = ms_sum + S * ROWS + 16;
    if (g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            ms_max[warp * ROWS + m * 16 + r] = row_max[m];
            ms_sum[warp * ROWS + m * 16 + r] = row_sum[m];
        }
    }
    __syncthreads();
    float fin_max[MW], fin_sum[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        float M = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < S; ++w) M = MAX_T(M, ms_max[w * ROWS + m * 16 + r]);
        float L = 0.0f;
        #pragma unroll
        for (int w = 0; w < S; ++w)
            L += ms_sum[w * ROWS + m * 16 + r] * __builtin_amdgcn_exp2f((ms_max[w * ROWS + m * 16 + r] - M) * scale_log2);
        fin_max[m] = M;
        fin_sum[m] = L;
        const float f = __builtin_amdgcn_exp2f((row_max[m] - M) * scale_log2) / (L + 1e-6f);
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] *= f;
    }

    const bool final_out = nparts == 1;
    constexpr int RC = red_cols(S, D);
    constexpr int RS = RC + 4;
    const int tid = threadIdx.x;
    using sx4 = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    using sx8 = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    auto head_of = [&](int gr) { return (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk; };
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        #pragma unroll
        for (int blk = 0; blk < D / RC; ++blk) {
            __syncthreads();
            // acc tile (2*b32 + f) holds dims 32*b32 + 8j + 2g + f.
            #pragma unroll
            for (int b32 = 0; b32 < RC / 32; ++b32)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        red[(warp * 16 + r) * RS + 32 * b32 + 8 * j + 2 * g + f] =
                            acc[m][(blk * (RC / 32) + b32) * 2 + f][j];
            __syncthreads();
            for (int item = tid; item < 16 * RC / 4; item += S * WARP_SIZE) {
                const int rr = item / (RC / 4);
                const int d4 = (item % (RC / 4)) * 4;
                float4_t sum4 = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w)
                    sum4 += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + d4);
                const int gr = row0 + m * 16 + rr;
                if (gr < R) {
                    const int head_all = head_of(gr);
                    sx4 o = {from_float<scalar_t>(sum4[0]), from_float<scalar_t>(sum4[1]),
                             from_float<scalar_t>(sum4[2]), from_float<scalar_t>(sum4[3])};
                    scalar_t* dst = final_out
                        ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                        : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                    *reinterpret_cast<sx4*>(dst + blk * RC + d4) = o;
                }
            }
        }
    }
    if (final_out) return;
    if (warp == 0 && g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            const int gr = row0 + m * 16 + r;
            if (gr < R) {
                const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(gr)) * grid_parts + part;
                reinterpret_cast<float*>(out_tmp)[off] = fin_sum[m];
                reinterpret_cast<float*>(out_tmp + max_tmp_offset)[off] = fin_max[m] * scale_log2;
            }
        }
    }

    if constexpr (MW > 1 || !FUSE) return;
    // ---- Fused split-K combine: the last CTA of this (batch, head, row group) merges.
    __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(shared_mem);
    const int slot = (b * num_kv_heads + kv_head) * row_groups + rg;
    if (tid == 0) flag[0] = atomicAdd(counters + slot, 1);
    __syncthreads();
    if (flag[0] != nparts - 1) return;
    __threadfence();
    const int rows_here = MIN_T(ROWS, R - row0);
    float* wts = reinterpret_cast<float*>(shared_mem) + 16;     // [rows][nparts]
    // Partials written by other CTAs: read through L2 (glc), never a stale L1 line.
    float* sums = reinterpret_cast<float*>(out_tmp);
    float* maxs = reinterpret_cast<float*>(out_tmp + max_tmp_offset);
    auto ld_l2 = [](float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
    for (int row = warp; row < rows_here; row += S) {
        const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(row0 + row)) * grid_parts;
        float M = -INFINITY;
        for (int i = lane; i < nparts; i += WARP_SIZE) M = MAX_T(M, ld_l2(maxs + off + i));
        for (int mask = 32; mask > 0; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
        float L = 0.0f;
        for (int i = lane; i < nparts; i += WARP_SIZE) {
            const float sv = ld_l2(sums + off + i);
            const float w = sv > 0.0f ? sv * __builtin_amdgcn_exp2f(ld_l2(maxs + off + i) - M) : 0.0f;
            wts[row * nparts + i] = w;
            L += w;
        }
        for (int mask = 32; mask > 0; mask >>= 1) L += __shfl_xor(L, mask);
        const float inv = __fdividef(1.0f, L + 1e-6f);
        for (int i = lane; i < nparts; i += WARP_SIZE) wts[row * nparts + i] *= inv;
    }
    __syncthreads();
    // Keep the buffer descriptor uniform across the entire CTA; only the byte
    // offset varies by lane. A per-row descriptor serializes vector loads.
    const auto partial_rsrc = make_rsrc(reinterpret_cast<const uint8_t*>(out_tmp + out_tmp_offset),
        (b + 1) * num_heads * grid_parts * D * sizeof(scalar_t));
    // Spread the merge across all waves even for a four-row query group.
    // Four-byte (D=128) or eight-byte (D=256) vectors give one row per wave.
    constexpr int OV = D == 128 ? 2 : 4;
    using sxv = __attribute__((__vector_size__(OV * sizeof(scalar_t)))) scalar_t;
    for (int item = tid; item < rows_here * (D / OV); item += S * WARP_SIZE) {
        const int row = item / (D / OV);
        const int col = (item % (D / OV)) * OV;
        const int head_all = head_of(row0 + row);
        const int partial_base = ((b * num_heads + head_all) * grid_parts * D + col) * sizeof(scalar_t);
        const float* w = wts + row * nparts;
        float accv[OV] = {};
        constexpr int ILP = 8;
        for (int i0 = 0; i0 < nparts; i0 += ILP) {
            sxv x[ILP];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int offset = partial_base + MIN_T(i0 + u, nparts - 1) * D * sizeof(scalar_t);
                if constexpr (OV == 2)
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load32(partial_rsrc, offset, 0, 1));
                else
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load64(partial_rsrc, offset, 0, 1));
            }
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const float wi = i0 + u < nparts ? w[i0 + u] : 0.0f;
                #pragma unroll
                for (int e = 0; e < OV; ++e) accv[e] += combine_to_float(x[u][e]) * wi;
            }
        }
        sxv o;
        #pragma unroll
        for (int e = 0; e < OV; ++e) o[e] = from_float<scalar_t>(accv[e]);
        *reinterpret_cast<sxv*>(out + (static_cast<int64_t>(b) * num_heads + head_all) * D + col) = o;
    }
    if (tid == 0) atomicExch(counters + slot, 0);
#if defined(__gfx92a__)
    }
#endif
#endif
}



template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true>
__launch_bounds__(S * WARP_SIZE)
__global__ void pa_bhsd_fp8_bf16red(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
{
#if defined(__gfx938__) || defined(__gfx92a__)
#if defined(__gfx92a__)
    // gfx92a 不启动本命名空间的 kernel (Q 全部由 bhsd_fp8_gfx92a 处理), 函数体只在 gfx938 上编译
    if constexpr (sizeof(q_type) == 0) {
#endif
    extern __shared__ char shared_mem[];
    constexpr int KC = D / 64;
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = blockIdx.x % row_groups;
    const int part = blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int chunks = DIVIDE_ROUND_UP(valid, CHUNK);
    const int nparts = valid_partitions(chunks, grid_parts, one_wave_parts, min_chunks);
    if (part >= nparts) return;

    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int r = lane & 15;
    const int g = lane >> 4;
    const int hpk = num_q_heads / num_kv_heads;
    const int R = hpk * mtp;
    const int row0 = rg * ROWS;
    const int num_heads = num_q_heads * mtp;

    const int c_begin = static_cast<int>(static_cast<int64_t>(part) * chunks / nparts);
    const int c_end = static_cast<int>(static_cast<int64_t>(part + 1) * chunks / nparts);

    const int* table = block_tables + static_cast<int64_t>(b) * max_num_blocks_per_seq;
    const uint8_t* k_head = k_cache + kv_head * (64 * D);
    const uint8_t* v_head = v_cache + kv_head * (64 * D);
    auto chunk_k = [&](int c, int phys) {
        return k_head + static_cast<int64_t>(phys) * k_block_stride + (c & 1) * (CHUNK * D);
    };
    auto chunk_v = [&](int c, int phys) {
        return v_head + static_cast<int64_t>(phys) * v_block_stride + (c & 1) * (CHUNK * D);
    };

    // K fragment rows: tile i column n holds token 16*i + 4*(n&3) + (n>>2), so the
    // P fragment of a 32-token chunk packs straight from the two score tiles.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * D + 16 * g;
    // V rows: load (T, h) covers rows 16h..16h+15 of 64-dim column tile T.
    const int v_row = lane >> 2;
    const int v_voff = v_row * D + 16 * (lane & 3);
    const int v_lds = v_row * 64 + ((16 * (lane & 3)) ^ ((v_row & 2) << 4));

    uint8x16_t kf[2][KC];
    uint8x16_t vf[KC][2];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_k(c, phys), CHUNK * D);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 16 * i * D + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_v(c, phys), CHUNK * D);
        #pragma unroll
        for (int t = 0; t < KC; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) vf[t][h] = load16(rs, v_voff, 16 * h * D + 64 * t);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[c >> 1]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * STAGE * V_TILE_BYTES;
    auto load_q16 = [&](int gr, int col) {
        uint8x16_t v = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            const q_type* qp = q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col;
            if constexpr (std::is_same<q_type, uint8_t>::value) {
                v = *reinterpret_cast<const uint8x16_t*>(qp);
            } else {
                using qx8 = __attribute__((__vector_size__(8 * sizeof(q_type)))) q_type;
                const qx8 a0 = *reinterpret_cast<const qx8*>(qp);
                const qx8 a1 = *reinterpret_cast<const qx8*>(qp + 8);
                intx4 w;
                w[0] = to_f8_from_f32<is_e4m3>(to_float(a0[0]), to_float(a0[1]), to_float(a0[2]), to_float(a0[3]));
                w[1] = to_f8_from_f32<is_e4m3>(to_float(a0[4]), to_float(a0[5]), to_float(a0[6]), to_float(a0[7]));
                w[2] = to_f8_from_f32<is_e4m3>(to_float(a1[0]), to_float(a1[1]), to_float(a1[2]), to_float(a1[3]));
                w[3] = to_f8_from_f32<is_e4m3>(to_float(a1[4]), to_float(a1[5]), to_float(a1[6]), to_float(a1[7]));
                v = __builtin_bit_cast(uint8x16_t, w);
            }
        }
        return v;
    };
    uint8x16_t qf[QL ? 1 : MW][QL ? 1 : KC];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 16); i += S * WARP_SIZE) {
            const int row = i / (D / 16), col = (i % (D / 16)) * 16;
            *reinterpret_cast<uint8x16_t*>(q_lds + row * QS + col) = load_q16(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) qf[m][kc] = load_q16(row0 + m * 16 + r, 64 * kc + 16 * g);
    }
    auto q_frag = [&](int m, int kc) {
        if constexpr (QL) return *reinterpret_cast<const uint8x16_t*>(q_lds + (m * 16 + r) * QS + 64 * kc + 16 * g);
        else return qf[m][kc];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
    float4_t acc[MW][D / 16];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_mem) + warp * STAGE * V_TILE_BYTES;

    // Peel the known-zero initial recurrence at compile time. The steady
    // loop retains its original rescaling without a per-chunk first test.
    auto process_chunk = [&]<bool INITIAL>() {
        const int next = c + S;
        const int next_phys = next < c_end ? table[next >> 1] : 0;
        const int base_tok = c * CHUNK;
        // ---- S = Q K^T over the 32-token chunk.
        float4_t s[MW][2];
        #pragma unroll
        for (int m = 0; m < MW; ++m) { s[m][0] = {0.f, 0.f, 0.f, 0.f}; s[m][1] = {0.f, 0.f, 0.f, 0.f}; }
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CHUNK - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = MAX_T(MAX_T(MAX_T(s[m][0][0], s[m][0][1]), MAX_T(s[m][0][2], s[m][0][3])),
                             MAX_T(MAX_T(s[m][1][0], s[m][1][1]), MAX_T(s[m][1][2], s[m][1][3])));
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            float alpha = 1.0f;
            if constexpr (!INITIAL)
                alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[m][i][j] = __builtin_amdgcn_exp2f(s[m][i][j] * scale_log2 - base);
                    sum += s[m][i][j];
                }
            sum += __shfl_xor_tmp(sum, 32);
            sum += __shfl_swap16(sum);
            if constexpr (INITIAL) {
                row_sum[m] = sum;
            } else {
                row_sum[m] = row_sum[m] * alpha + sum;
                #pragma unroll
                for (int i = 0; i < D / 16; ++i) acc[m][i] *= alpha;
            }
            row_max[m] = new_max;
            p[m][0] = to_f8_from_f32<is_e4m3>(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3]);
            p[m][1] = to_f8_from_f32<is_e4m3>(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3]);
        }

        // ---- V: zero tokens past the sequence end, stage, transpose-read, MMAC.
        if (base_tok + CHUNK > valid) {
            #pragma unroll
            for (int t = 0; t < KC; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    if (base_tok + 16 * h + v_row >= valid) vf[t][h] = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int t0 = 0; t0 < KC; t0 += STAGE) {
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                uint8_t* slot = stage + (t - t0) * V_TILE_BYTES;
                *reinterpret_cast<uint8x16_t*>(slot + v_lds) = vf[t][0];
                *reinterpret_cast<uint8x16_t*>(slot + 1024 + v_lds) = vf[t][1];
            }
            asm volatile("" ::: "memory");
            if (t0 + STAGE >= KC && next < c_end) issue_v(next, next_phys);
            intx4 frag[STAGE][2];
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                int8_t* slot = reinterpret_cast<int8_t*>(stage + (t - t0) * V_TILE_BYTES);
                frag[t - t0][0] = __builtin_hcu_ds_read_matrix_format_u8(slot, 0, 2, 2, 1);
                frag[t - t0][1] = __builtin_hcu_ds_read_matrix_format_u8(slot + 32, 0, 2, 2, 1);
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const intx2* fv = reinterpret_cast<const intx2*>(&frag[t - t0][half]);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[0], acc[m][t * 4 + half * 2 + 0]);
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[1], acc[m][t * 4 + half * 2 + 1]);
                    }
                }
        }
    };
    if (c < c_end) {
        process_chunk.template operator()<true>();
        c += S;
    }
    for (; c < c_end; c += S)
        process_chunk.template operator()<false>();

    // ---- Merge the S waves of this CTA (same rows, disjoint chunks).
    __syncthreads();
    float* ms_max = reinterpret_cast<float*>(shared_mem);
    float* ms_sum = ms_max + S * ROWS;
    // Accumulators and max/sum stay FP32. Only normalized inter-wave
    // communication rounds to BF16; this deliberately changes rounding order.
    // The precision contract is validated without relaxing atol/rtol.
    scalar_t* red = reinterpret_cast<scalar_t*>(ms_sum + S * ROWS + 16);
    if (g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            ms_max[warp * ROWS + m * 16 + r] = row_max[m];
            ms_sum[warp * ROWS + m * 16 + r] = row_sum[m];
        }
    }
    __syncthreads();
    float fin_max[MW], fin_sum[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        float M = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < S; ++w) M = MAX_T(M, ms_max[w * ROWS + m * 16 + r]);
        float L = 0.0f;
        #pragma unroll
        for (int w = 0; w < S; ++w)
            L += ms_sum[w * ROWS + m * 16 + r] * __builtin_amdgcn_exp2f((ms_max[w * ROWS + m * 16 + r] - M) * scale_log2);
        fin_max[m] = M;
        fin_sum[m] = L;
        const float f = __builtin_amdgcn_exp2f((row_max[m] - M) * scale_log2) / (L + 1e-6f);
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] *= f;
    }

    const bool final_out = nparts == 1;
    constexpr int RC = red_cols(S, D);
    constexpr int RS = RC + 4;
    const int tid = threadIdx.x;
    using sx4 = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    using sx8 = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    auto head_of = [&](int gr) { return (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk; };
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        #pragma unroll
        for (int blk = 0; blk < D / RC; ++blk) {
            __syncthreads();
            // acc tile (2*b32 + f) holds dims 32*b32 + 8j + 2g + f.
            #pragma unroll
            for (int b32 = 0; b32 < RC / 32; ++b32)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        red[(warp * 16 + r) * RS + 32 * b32 + 8 * j + 2 * g + f] =
                            from_float<scalar_t>(acc[m][(blk * (RC / 32) + b32) * 2 + f][j]);
            __syncthreads();
            for (int item = tid; item < 16 * RC / 8; item += S * WARP_SIZE) {
                const int rr = item / (RC / 8);
                const int d8 = (item % (RC / 8)) * 8;
                float sum8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w) {
                    // RS is a multiple of four BF16 values, so retain two
                    // aligned 64-bit LDS reads while doubling output width.
                    const sx4 lo = *reinterpret_cast<const sx4*>(red + (w * 16 + rr) * RS + d8);
                    const sx4 hi = *reinterpret_cast<const sx4*>(red + (w * 16 + rr) * RS + d8 + 4);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        sum8[e] += to_float(lo[e]);
                        sum8[e + 4] += to_float(hi[e]);
                    }
                }
                const int gr = row0 + m * 16 + rr;
                if (gr < R) {
                    const int head_all = head_of(gr);
                    sx8 o;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) o[e] = from_float<scalar_t>(sum8[e]);
                    scalar_t* dst = final_out
                        ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                        : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                    *reinterpret_cast<sx8*>(dst + blk * RC + d8) = o;
                }
            }
        }
    }
    if (final_out) return;
    if (warp == 0 && g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            const int gr = row0 + m * 16 + r;
            if (gr < R) {
                const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(gr)) * grid_parts + part;
                reinterpret_cast<float*>(out_tmp)[off] = fin_sum[m];
                reinterpret_cast<float*>(out_tmp + max_tmp_offset)[off] = fin_max[m] * scale_log2;
            }
        }
    }

    if constexpr (MW > 1 || !FUSE) return;
    // ---- Fused split-K combine: the last CTA of this (batch, head, row group) merges.
    __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(shared_mem);
    const int slot = (b * num_kv_heads + kv_head) * row_groups + rg;
    if (tid == 0) flag[0] = atomicAdd(counters + slot, 1);
    __syncthreads();
    if (flag[0] != nparts - 1) return;
    __threadfence();
    const int rows_here = MIN_T(ROWS, R - row0);
    float* wts = reinterpret_cast<float*>(shared_mem) + 16;     // [rows][nparts]
    // Partials written by other CTAs: read through L2 (glc), never a stale L1 line.
    float* sums = reinterpret_cast<float*>(out_tmp);
    float* maxs = reinterpret_cast<float*>(out_tmp + max_tmp_offset);
    auto ld_l2 = [](float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
    for (int row = warp; row < rows_here; row += S) {
        const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(row0 + row)) * grid_parts;
        float M = -INFINITY;
        for (int i = lane; i < nparts; i += WARP_SIZE) M = MAX_T(M, ld_l2(maxs + off + i));
        for (int mask = 32; mask > 0; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
        float L = 0.0f;
        for (int i = lane; i < nparts; i += WARP_SIZE) {
            const float sv = ld_l2(sums + off + i);
            const float w = sv > 0.0f ? sv * __builtin_amdgcn_exp2f(ld_l2(maxs + off + i) - M) : 0.0f;
            wts[row * nparts + i] = w;
            L += w;
        }
        for (int mask = 32; mask > 0; mask >>= 1) L += __shfl_xor(L, mask);
        const float inv = __fdividef(1.0f, L + 1e-6f);
        for (int i = lane; i < nparts; i += WARP_SIZE) wts[row * nparts + i] *= inv;
    }
    __syncthreads();
    // Keep the buffer descriptor uniform across the entire CTA; only the byte
    // offset varies by lane. A per-row descriptor serializes vector loads.
    const auto partial_rsrc = make_rsrc(reinterpret_cast<const uint8_t*>(out_tmp + out_tmp_offset),
        (b + 1) * num_heads * grid_parts * D * sizeof(scalar_t));
    // Spread the merge across all waves even for a four-row query group.
    // Four-byte (D=128) or eight-byte (D=256) vectors give one row per wave.
    constexpr int OV = D == 128 ? 2 : 4;
    using sxv = __attribute__((__vector_size__(OV * sizeof(scalar_t)))) scalar_t;
    for (int item = tid; item < rows_here * (D / OV); item += S * WARP_SIZE) {
        const int row = item / (D / OV);
        const int col = (item % (D / OV)) * OV;
        const int head_all = head_of(row0 + row);
        const int partial_base = ((b * num_heads + head_all) * grid_parts * D + col) * sizeof(scalar_t);
        const float* w = wts + row * nparts;
        float accv[OV] = {};
        constexpr int ILP = 8;
        for (int i0 = 0; i0 < nparts; i0 += ILP) {
            sxv x[ILP];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int offset = partial_base + MIN_T(i0 + u, nparts - 1) * D * sizeof(scalar_t);
                if constexpr (OV == 2)
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load32(partial_rsrc, offset, 0, 1));
                else
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load64(partial_rsrc, offset, 0, 1));
            }
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const float wi = i0 + u < nparts ? w[i0 + u] : 0.0f;
                #pragma unroll
                for (int e = 0; e < OV; ++e) accv[e] += combine_to_float(x[u][e]) * wi;
            }
        }
        sxv o;
        #pragma unroll
        for (int e = 0; e < OV; ++e) o[e] = from_float<scalar_t>(accv[e]);
        *reinterpret_cast<sxv*>(out + (static_cast<int64_t>(b) * num_heads + head_all) * D + col) = o;
    }
    if (tid == 0) atomicExch(counters + slot, 0);
#if defined(__gfx92a__)
    }
#endif
#endif
}



template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true>
__launch_bounds__(S * WARP_SIZE)
__global__ void pa_bhsd_fp8_compact(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
{
    pa_bhsd_fp8_body<scalar_t, q_type, is_e4m3, D, MW, S, STAGE, FUSE, true>(out, out_tmp, q, k_cache, v_cache, block_tables, seq_lens,
        num_q_heads, num_kv_heads, mtp, row_groups, max_num_blocks_per_seq,
        q_stride, k_block_stride, v_block_stride, grid_parts, one_wave_parts, min_chunks, counters);
}

// Split-K merge for wide (32-row) CTAs, one CTA per output head.
template <typename scalar_t, int D>
__global__ __launch_bounds__(256) void pa_bhsd_fp8_combine(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    __shared__ float weight[1024];
    __shared__ float partial[4 * D];
    __shared__ float scalars[8];
    const int tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), grid_parts, one_wave_parts, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    float lmax = -INFINITY;
    for (int i = tid; i < np; i += 256) lmax = MAX_T(lmax, maxs[i]);
    for (int mask = 32; mask > 0; mask >>= 1) lmax = MAX_T(lmax, __shfl_xor(lmax, mask));
    if (lane == 0) scalars[warp] = lmax;
    __syncthreads();
    const float gmax = MAX_T(MAX_T(scalars[0], scalars[1]), MAX_T(scalars[2], scalars[3]));
    float lsum = 0.0f;
    for (int i = tid; i < np; i += 256) {
        const float s = sums[i];
        const float w = s > 0.0f ? s * __builtin_amdgcn_exp2f(maxs[i] - gmax) : 0.0f;
        weight[i] = w;
        lsum += w;
    }
    for (int mask = 32; mask > 0; mask >>= 1) lsum += __shfl_xor(lsum, mask);
    __syncthreads();
    if (lane == 0) scalars[4 + warp] = lsum;
    __syncthreads();
    const float inv = __fdividef(1.0f, (scalars[4] + scalars[5]) + (scalars[6] + scalars[7]) + 1e-6f);
    // A three-element vector has padded size/alignment and cannot be stored
    // at a six-byte stride. Use three scalar tiles for D=192 instead.
    constexpr int V = D % 128 == 0 ? D / 64 : 1;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    #pragma unroll
    for (int tile = 0; tile < D / (64 * V); ++tile) {
    const int column = tile * 64 * V + lane * V;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + column;
    float acc[V];
    #pragma unroll
    for (int i = 0; i < V; ++i) acc[i] = 0.0f;
    int j = warp;
    for (; j + 12 < np; j += 16) {
        sv x0 = *reinterpret_cast<const sv*>(src + j * D), x1 = *reinterpret_cast<const sv*>(src + (j + 4) * D);
        sv x2 = *reinterpret_cast<const sv*>(src + (j + 8) * D), x3 = *reinterpret_cast<const sv*>(src + (j + 12) * D);
        const float w0 = weight[j], w1 = weight[j + 4], w2 = weight[j + 8], w3 = weight[j + 12];
        #pragma unroll
        for (int i = 0; i < V; ++i)
            acc[i] += (combine_to_float(x0[i]) * w0 + combine_to_float(x1[i]) * w1) + (combine_to_float(x2[i]) * w2 + combine_to_float(x3[i]) * w3);
    }
    for (; j < np; j += 4) {
        sv x = *reinterpret_cast<const sv*>(src + j * D);
        #pragma unroll
        for (int i = 0; i < V; ++i) acc[i] += combine_to_float(x[i]) * weight[j];
    }
    #pragma unroll
    for (int i = 0; i < V; ++i) partial[warp * D + column + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        sv o;
        #pragma unroll
        for (int i = 0; i < V; ++i) {
            const int col = column + i;
            scalar_t t;
            from_float(t, ((partial[col] + partial[D + col]) + (partial[2 * D + col] + partial[3 * D + col])) * inv);
            o[i] = t;
        }
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + column) = o;
    }
    if constexpr (D / (64 * V) > 1) __syncthreads();
    }
}

template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true, bool WHOLE = false>
__device__ __forceinline__ void pa_bhsd_fp8_dynamic_body(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
{
#if defined(__gfx938__) || defined(__gfx92a__)
#if defined(__gfx92a__)
    // gfx92a 不启动本命名空间的 kernel (Q 全部由 bhsd_fp8_gfx92a 处理), 函数体只在 gfx938 上编译
    if constexpr (sizeof(q_type) == 0) {
#endif
    extern __shared__ char shared_mem[];
    constexpr int KC = D / 64;
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = WHOLE ? 0 : blockIdx.x % row_groups;
    const int part = WHOLE ? blockIdx.y * row_groups + blockIdx.x % row_groups : blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int chunks = DIVIDE_ROUND_UP(valid, CHUNK);
    const int valid_grid = WHOLE ? grid_parts : one_wave_parts;
    const int nparts = valid_partitions(chunks, valid_grid, valid_grid, min_chunks);
    if (part >= nparts) return;

    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int r = lane & 15;
    const int g = lane >> 4;
    const int hpk = num_q_heads / num_kv_heads;
    const int R = hpk * mtp;
    const int row0 = rg * ROWS;
    const int num_heads = num_q_heads * mtp;

    const int c_begin = static_cast<int>(static_cast<int64_t>(part) * chunks / nparts);
    const int c_end = static_cast<int>(static_cast<int64_t>(part + 1) * chunks / nparts);

    const int* table = block_tables + static_cast<int64_t>(b) * max_num_blocks_per_seq;
    const uint8_t* k_head = k_cache + kv_head * (64 * D);
    const uint8_t* v_head = v_cache + kv_head * (64 * D);
    auto chunk_k = [&](int c, int phys) {
        return k_head + static_cast<int64_t>(phys) * k_block_stride + (c & 1) * (CHUNK * D);
    };
    auto chunk_v = [&](int c, int phys) {
        return v_head + static_cast<int64_t>(phys) * v_block_stride + (c & 1) * (CHUNK * D);
    };

    // K fragment rows: tile i column n holds token 16*i + 4*(n&3) + (n>>2), so the
    // P fragment of a 32-token chunk packs straight from the two score tiles.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * D + 16 * g;
    // V rows: load (T, h) covers rows 16h..16h+15 of 64-dim column tile T.
    const int v_row = lane >> 2;
    const int v_voff = v_row * D + 16 * (lane & 3);
    const int v_lds = v_row * 64 + ((16 * (lane & 3)) ^ ((v_row & 2) << 4));

    uint8x16_t kf[2][KC];
    uint8x16_t vf[KC][2];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_k(c, phys), CHUNK * D);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 16 * i * D + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_v(c, phys), CHUNK * D);
        #pragma unroll
        for (int t = 0; t < KC; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) vf[t][h] = load16(rs, v_voff, 16 * h * D + 64 * t);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[c >> 1]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * STAGE * V_TILE_BYTES;
    auto load_q16 = [&](int gr, int col) {
        uint8x16_t v = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            const q_type* qp = q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col;
            if constexpr (std::is_same<q_type, uint8_t>::value) {
                v = *reinterpret_cast<const uint8x16_t*>(qp);
            } else {
                using qx8 = __attribute__((__vector_size__(8 * sizeof(q_type)))) q_type;
                const qx8 a0 = *reinterpret_cast<const qx8*>(qp);
                const qx8 a1 = *reinterpret_cast<const qx8*>(qp + 8);
                intx4 w;
                w[0] = to_f8_from_f32<is_e4m3>(to_float(a0[0]), to_float(a0[1]), to_float(a0[2]), to_float(a0[3]));
                w[1] = to_f8_from_f32<is_e4m3>(to_float(a0[4]), to_float(a0[5]), to_float(a0[6]), to_float(a0[7]));
                w[2] = to_f8_from_f32<is_e4m3>(to_float(a1[0]), to_float(a1[1]), to_float(a1[2]), to_float(a1[3]));
                w[3] = to_f8_from_f32<is_e4m3>(to_float(a1[4]), to_float(a1[5]), to_float(a1[6]), to_float(a1[7]));
                v = __builtin_bit_cast(uint8x16_t, w);
            }
        }
        return v;
    };
    uint8x16_t qf[QL ? 1 : MW][QL ? 1 : KC];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 16); i += S * WARP_SIZE) {
            const int row = i / (D / 16), col = (i % (D / 16)) * 16;
            *reinterpret_cast<uint8x16_t*>(q_lds + row * QS + col) = load_q16(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) qf[m][kc] = load_q16(row0 + m * 16 + r, 64 * kc + 16 * g);
    }
    auto q_frag = [&](int m, int kc) {
        if constexpr (QL) return *reinterpret_cast<const uint8x16_t*>(q_lds + (m * 16 + r) * QS + 64 * kc + 16 * g);
        else return qf[m][kc];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
    float4_t acc[MW][D / 16];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_mem) + warp * STAGE * V_TILE_BYTES;

    for (; c < c_end; c += S) {
        const int next = c + S;
        const int next_phys = next < c_end ? table[next >> 1] : 0;
        const int base_tok = c * CHUNK;
        // ---- S = Q K^T over the 32-token chunk.
        float4_t s[MW][2];
        #pragma unroll
        for (int m = 0; m < MW; ++m) { s[m][0] = {0.f, 0.f, 0.f, 0.f}; s[m][1] = {0.f, 0.f, 0.f, 0.f}; }
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CHUNK - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = MAX_T(MAX_T(MAX_T(s[m][0][0], s[m][0][1]), MAX_T(s[m][0][2], s[m][0][3])),
                             MAX_T(MAX_T(s[m][1][0], s[m][1][1]), MAX_T(s[m][1][2], s[m][1][3])));
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            const float alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[m][i][j] = __builtin_amdgcn_exp2f(s[m][i][j] * scale_log2 - base);
                    sum += s[m][i][j];
                }
            sum += __shfl_xor_tmp(sum, 32);
            sum += __shfl_swap16(sum);
            row_sum[m] = row_sum[m] * alpha + sum;
            row_max[m] = new_max;
            #pragma unroll
            for (int i = 0; i < D / 16; ++i) acc[m][i] *= alpha;
            p[m][0] = to_f8_from_f32<is_e4m3>(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3]);
            p[m][1] = to_f8_from_f32<is_e4m3>(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3]);
        }

        // ---- V: zero tokens past the sequence end, stage, transpose-read, MMAC.
        if (base_tok + CHUNK > valid) {
            #pragma unroll
            for (int t = 0; t < KC; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    if (base_tok + 16 * h + v_row >= valid) vf[t][h] = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int t0 = 0; t0 < KC; t0 += STAGE) {
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                uint8_t* slot = stage + (t - t0) * V_TILE_BYTES;
                *reinterpret_cast<uint8x16_t*>(slot + v_lds) = vf[t][0];
                *reinterpret_cast<uint8x16_t*>(slot + 1024 + v_lds) = vf[t][1];
            }
            asm volatile("" ::: "memory");
            if (t0 + STAGE >= KC && next < c_end) issue_v(next, next_phys);
            intx4 frag[STAGE][2];
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                int8_t* slot = reinterpret_cast<int8_t*>(stage + (t - t0) * V_TILE_BYTES);
                frag[t - t0][0] = __builtin_hcu_ds_read_matrix_format_u8(slot, 0, 2, 2, 1);
                frag[t - t0][1] = __builtin_hcu_ds_read_matrix_format_u8(slot + 32, 0, 2, 2, 1);
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const intx2* fv = reinterpret_cast<const intx2*>(&frag[t - t0][half]);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[0], acc[m][t * 4 + half * 2 + 0]);
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[1], acc[m][t * 4 + half * 2 + 1]);
                    }
                }
        }
    }

    // ---- Merge the S waves of this CTA (same rows, disjoint chunks).
    __syncthreads();
    float* ms_max = reinterpret_cast<float*>(shared_mem);
    float* ms_sum = ms_max + S * ROWS;
    float* red = ms_sum + S * ROWS + 16;
    if (g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            ms_max[warp * ROWS + m * 16 + r] = row_max[m];
            ms_sum[warp * ROWS + m * 16 + r] = row_sum[m];
        }
    }
    __syncthreads();
    float fin_max[MW], fin_sum[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        float M = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < S; ++w) M = MAX_T(M, ms_max[w * ROWS + m * 16 + r]);
        float L = 0.0f;
        #pragma unroll
        for (int w = 0; w < S; ++w)
            L += ms_sum[w * ROWS + m * 16 + r] * __builtin_amdgcn_exp2f((ms_max[w * ROWS + m * 16 + r] - M) * scale_log2);
        fin_max[m] = M;
        fin_sum[m] = L;
        const float f = __builtin_amdgcn_exp2f((row_max[m] - M) * scale_log2) / (L + 1e-6f);
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] *= f;
    }

    const bool final_out = nparts == 1;
    constexpr int RC = red_cols(S, D);
    constexpr int RS = RC + 4;
    const int tid = threadIdx.x;
    using sx4 = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    using sx8 = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    auto head_of = [&](int gr) { return (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk; };
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        #pragma unroll
        for (int blk = 0; blk < D / RC; ++blk) {
            __syncthreads();
            // acc tile (2*b32 + f) holds dims 32*b32 + 8j + 2g + f.
            #pragma unroll
            for (int b32 = 0; b32 < RC / 32; ++b32)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        red[(warp * 16 + r) * RS + 32 * b32 + 8 * j + 2 * g + f] =
                            acc[m][(blk * (RC / 32) + b32) * 2 + f][j];
            __syncthreads();
            if constexpr (D == 128 && S == 4 && !FUSE && is_e4m3 && std::is_same<scalar_t, uint16_t>::value && std::is_same<q_type, uint16_t>::value) {
                // Keep FP32 LDS payloads, original per-element wave order,
                // and the same BF16 rounding point; pair two output vectors.
                for (int item = tid; item < 16 * RC / 8; item += S * WARP_SIZE) {
                    const int rr = item / (RC / 8);
                    const int d8 = (item % (RC / 8)) * 8;
                    float4_t sum_lo = {0.f, 0.f, 0.f, 0.f};
                    float4_t sum_hi = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int w = 0; w < S; ++w) {
                        // Separate 16-byte reads preserve XOR row swizzling
                        // and alignment; one float8 read would not do so.
                        sum_lo += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + d8);
                        sum_hi += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + (d8 + 4));
                    }
                    const int gr = row0 + m * 16 + rr;
                    if (gr < R) {
                        const int head_all = head_of(gr);
                        sx8 o;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            o[e] = from_float<scalar_t>(sum_lo[e]);
                            o[e + 4] = from_float<scalar_t>(sum_hi[e]);
                        }
                        scalar_t* dst = final_out
                            ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                            : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                        *reinterpret_cast<sx8*>(dst + blk * RC + d8) = o;
                    }
                }
            } else {
            for (int item = tid; item < 16 * RC / 4; item += S * WARP_SIZE) {
                const int rr = item / (RC / 4);
                const int d4 = (item % (RC / 4)) * 4;
                float4_t sum4 = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w)
                    sum4 += *reinterpret_cast<const float4_t*>(red + (w * 16 + rr) * RS + d4);
                const int gr = row0 + m * 16 + rr;
                if (gr < R) {
                    const int head_all = head_of(gr);
                    sx4 o = {from_float<scalar_t>(sum4[0]), from_float<scalar_t>(sum4[1]),
                             from_float<scalar_t>(sum4[2]), from_float<scalar_t>(sum4[3])};
                    scalar_t* dst = final_out
                        ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                        : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                    *reinterpret_cast<sx4*>(dst + blk * RC + d4) = o;
                }
            }
            }
        }
    }
    if (final_out) return;
    if (warp == 0 && g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            const int gr = row0 + m * 16 + r;
            if (gr < R) {
                const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(gr)) * grid_parts + part;
                reinterpret_cast<float*>(out_tmp)[off] = fin_sum[m];
                reinterpret_cast<float*>(out_tmp + max_tmp_offset)[off] = fin_max[m] * scale_log2;
            }
        }
    }

    if constexpr (MW > 1 || !FUSE) return;
    // ---- Fused split-K combine: the last CTA of this (batch, head, row group) merges.
    __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(shared_mem);
    const int slot = (b * num_kv_heads + kv_head) * row_groups + rg;
    if (tid == 0) flag[0] = atomicAdd(counters + slot, 1);
    __syncthreads();
    if (flag[0] != nparts - 1) return;
    __threadfence();
    const int rows_here = MIN_T(ROWS, R - row0);
    float* wts = reinterpret_cast<float*>(shared_mem) + 16;     // [rows][nparts]
    // Partials written by other CTAs: read through L2 (glc), never a stale L1 line.
    float* sums = reinterpret_cast<float*>(out_tmp);
    float* maxs = reinterpret_cast<float*>(out_tmp + max_tmp_offset);
    auto ld_l2 = [](float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
    for (int row = warp; row < rows_here; row += S) {
        const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(row0 + row)) * grid_parts;
        float M = -INFINITY;
        for (int i = lane; i < nparts; i += WARP_SIZE) M = MAX_T(M, ld_l2(maxs + off + i));
        for (int mask = 32; mask > 0; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
        float L = 0.0f;
        for (int i = lane; i < nparts; i += WARP_SIZE) {
            const float sv = ld_l2(sums + off + i);
            const float w = sv > 0.0f ? sv * __builtin_amdgcn_exp2f(ld_l2(maxs + off + i) - M) : 0.0f;
            wts[row * nparts + i] = w;
            L += w;
        }
        for (int mask = 32; mask > 0; mask >>= 1) L += __shfl_xor(L, mask);
        const float inv = __fdividef(1.0f, L + 1e-6f);
        for (int i = lane; i < nparts; i += WARP_SIZE) wts[row * nparts + i] *= inv;
    }
    __syncthreads();
    // Keep the buffer descriptor uniform across the entire CTA; only the byte
    // offset varies by lane. A per-row descriptor serializes vector loads.
    const auto partial_rsrc = make_rsrc(reinterpret_cast<const uint8_t*>(out_tmp + out_tmp_offset),
        (b + 1) * num_heads * grid_parts * D * sizeof(scalar_t));
    // Spread the merge across all waves even for a four-row query group.
    // Four-byte (D=128) or eight-byte (D=256) vectors give one row per wave.
    constexpr int OV = D == 128 ? 2 : 4;
    using sxv = __attribute__((__vector_size__(OV * sizeof(scalar_t)))) scalar_t;
    for (int item = tid; item < rows_here * (D / OV); item += S * WARP_SIZE) {
        const int row = item / (D / OV);
        const int col = (item % (D / OV)) * OV;
        const int head_all = head_of(row0 + row);
        const int partial_base = ((b * num_heads + head_all) * grid_parts * D + col) * sizeof(scalar_t);
        const float* w = wts + row * nparts;
        float accv[OV] = {};
        constexpr int ILP = 8;
        for (int i0 = 0; i0 < nparts; i0 += ILP) {
            sxv x[ILP];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int offset = partial_base + MIN_T(i0 + u, nparts - 1) * D * sizeof(scalar_t);
                if constexpr (OV == 2)
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load32(partial_rsrc, offset, 0, 1));
                else
                    x[u] = __builtin_bit_cast(sxv, bhsd_fp8_buffer_load64(partial_rsrc, offset, 0, 1));
            }
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const float wi = i0 + u < nparts ? w[i0 + u] : 0.0f;
                #pragma unroll
                for (int e = 0; e < OV; ++e) accv[e] += combine_to_float(x[u][e]) * wi;
            }
        }
        sxv o;
        #pragma unroll
        for (int e = 0; e < OV; ++e) o[e] = from_float<scalar_t>(accv[e]);
        *reinterpret_cast<sxv*>(out + (static_cast<int64_t>(b) * num_heads + head_all) * D + col) = o;
    }
    if (tid == 0) atomicExch(counters + slot, 0);
#if defined(__gfx92a__)
    }
#endif
#endif
}
template <int MW>
__launch_bounds__(256)
__global__ void pa_bhsd_fp8_dynamic_whole(
    uint16_t* __restrict__ out, uint16_t* __restrict__ out_tmp,
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters) {
    if (__builtin_amdgcn_readfirstlane(seq_lens[blockIdx.z]) >= 32768) {
        pa_bhsd_fp8_dynamic_body<uint16_t,uint16_t,true,128,MW,4,2,false,true>(out,out_tmp,q,k_cache,v_cache,block_tables,seq_lens,num_q_heads,num_kv_heads,mtp,row_groups,max_num_blocks_per_seq,q_stride,k_block_stride,v_block_stride,grid_parts,one_wave_parts,min_chunks,counters);
    } else {
        pa_bhsd_fp8_dynamic_body<uint16_t,uint16_t,true,128,2,4,2,false,false>(out,out_tmp,q,k_cache,v_cache,block_tables,seq_lens,num_q_heads,num_kv_heads,mtp,row_groups,max_num_blocks_per_seq,q_stride,k_block_stride,v_block_stride,grid_parts,one_wave_parts,min_chunks,counters);
    }
}

template <typename scalar_t, int D>
__global__ __launch_bounds__(256) void pa_bhsd_fp8_dynamic_combine(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    __shared__ float weight[1024];
    __shared__ float partial[4 * D];
    __shared__ float scalars[8];
    const int tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), valid >= 32768 ? grid_parts : one_wave_parts, valid >= 32768 ? grid_parts : one_wave_parts, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    float lmax = -INFINITY;
    for (int i = tid; i < np; i += 256) lmax = MAX_T(lmax, maxs[i]);
    for (int mask = 32; mask > 0; mask >>= 1) lmax = MAX_T(lmax, __shfl_xor(lmax, mask));
    if (lane == 0) scalars[warp] = lmax;
    __syncthreads();
    const float gmax = MAX_T(MAX_T(scalars[0], scalars[1]), MAX_T(scalars[2], scalars[3]));
    float lsum = 0.0f;
    for (int i = tid; i < np; i += 256) {
        const float s = sums[i];
        const float w = s > 0.0f ? s * __builtin_amdgcn_exp2f(maxs[i] - gmax) : 0.0f;
        weight[i] = w;
        lsum += w;
    }
    for (int mask = 32; mask > 0; mask >>= 1) lsum += __shfl_xor(lsum, mask);
    __syncthreads();
    if (lane == 0) scalars[4 + warp] = lsum;
    __syncthreads();
    const float inv = __fdividef(1.0f, (scalars[4] + scalars[5]) + (scalars[6] + scalars[7]) + 1e-6f);
    // A three-element vector has padded size/alignment and cannot be stored
    // at a six-byte stride. Use three scalar tiles for D=192 instead.
    constexpr int V = D % 128 == 0 ? D / 64 : 1;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    #pragma unroll
    for (int tile = 0; tile < D / (64 * V); ++tile) {
    const int column = tile * 64 * V + lane * V;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + column;
    float acc[V];
    #pragma unroll
    for (int i = 0; i < V; ++i) acc[i] = 0.0f;
    int j = warp;
    for (; j + 12 < np; j += 16) {
        sv x0 = *reinterpret_cast<const sv*>(src + j * D), x1 = *reinterpret_cast<const sv*>(src + (j + 4) * D);
        sv x2 = *reinterpret_cast<const sv*>(src + (j + 8) * D), x3 = *reinterpret_cast<const sv*>(src + (j + 12) * D);
        const float w0 = weight[j], w1 = weight[j + 4], w2 = weight[j + 8], w3 = weight[j + 12];
        #pragma unroll
        for (int i = 0; i < V; ++i)
            acc[i] += (combine_to_float(x0[i]) * w0 + combine_to_float(x1[i]) * w1) + (combine_to_float(x2[i]) * w2 + combine_to_float(x3[i]) * w3);
    }
    for (; j < np; j += 4) {
        sv x = *reinterpret_cast<const sv*>(src + j * D);
        #pragma unroll
        for (int i = 0; i < V; ++i) acc[i] += combine_to_float(x[i]) * weight[j];
    }
    #pragma unroll
    for (int i = 0; i < V; ++i) partial[warp * D + column + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        sv o;
        #pragma unroll
        for (int i = 0; i < V; ++i) {
            const int col = column + i;
            scalar_t t;
            from_float(t, ((partial[col] + partial[D + col]) + (partial[2 * D + col] + partial[3 * D + col])) * inv);
            o[i] = t;
        }
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + column) = o;
    }
    if constexpr (D / (64 * V) > 1) __syncthreads();
    }
}


// Four waves own disjoint output columns; each wave divides its partitions
// across four 16-lane subgroups. Each wave independently computes the weights.
// No inter-wave communication or LDS allocation is needed.
// The launcher uses this path only when grid_parts <= 64 and D is 128 or 256.
template <typename scalar_t, int D, bool DYNAMIC = false>
__global__ __launch_bounds__(256) void pa_bhsd_fp8_combine_column_waves(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int cap = DYNAMIC ? (valid >= 32768 ? grid_parts : one_wave_parts) : grid_parts;
    const int one = DYNAMIC ? cap : one_wave_parts;
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), cap, one, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    const float mx = lane < np ? maxs[lane] : -INFINITY;
    const float s = lane < np ? sums[lane] : 0.0f;
    float M = mx;
    for (int mask = 32; mask; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
    const float w = s > 0.f ? s * __builtin_amdgcn_exp2f(mx - M) : 0.f;
    float L = w;
    for (int mask = 32; mask; mask >>= 1) L += __shfl_xor(L, mask);
    const float inv = __fdividef(1.f, L + 1e-6f);
    constexpr int V = D / 64;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    const int col = warp * (D / 4) + (lane & 15) * V, part_lane = lane >> 4;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + col;
    float acc[V] = {};
    // Match the baseline's four independent partial loads per iteration.
    for (int j = 0; j < np; j += 16) {
        sv x[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            x[u] = *reinterpret_cast<const sv*>(src + MIN_T(j + 4*u + part_lane, np - 1) * D);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int p = j + 4*u + part_lane;
            const float raw = __shfl(w, p & 63);
            const float wi = p < np ? raw : 0.f;
            #pragma unroll
            for (int e = 0; e < V; ++e) acc[e] += combine_to_float(x[u][e]) * wi;
        }
    }
    sv o;
    #pragma unroll
    for (int e = 0; e < V; ++e) {
        float sum = acc[e];
        sum += __shfl_xor(sum, 16);
        sum += __shfl_xor(sum, 32);
        scalar_t v; from_float(v, sum * inv); o[e] = v;
    }
    if (part_lane == 0)
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + col) = o;
}

template <typename scalar_t, int D, bool DYNAMIC = false>
__global__ __launch_bounds__(256) void pa_bhsd_fp8_combine_column_waves128(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int cap = DYNAMIC ? (valid >= 32768 ? grid_parts : one_wave_parts) : grid_parts;
    const int one = DYNAMIC ? cap : one_wave_parts;
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), cap, one, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    const float mx = lane < np ? maxs[lane] : -INFINITY;
    const float s = lane < np ? sums[lane] : 0.0f;
    const float mx1 = lane + 64 < np ? maxs[lane + 64] : -INFINITY;
    const float s1 = lane + 64 < np ? sums[lane + 64] : 0.0f;
    float M = MAX_T(mx, mx1);
    for (int mask = 32; mask; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
    const float w = s > 0.f ? s * __builtin_amdgcn_exp2f(mx - M) : 0.f;
    const float w1 = s1 > 0.f ? s1 * __builtin_amdgcn_exp2f(mx1 - M) : 0.f;
    float L = w + w1;
    for (int mask = 32; mask; mask >>= 1) L += __shfl_xor(L, mask);
    const float inv = __fdividef(1.f, L + 1e-6f);
    constexpr int V = D / 64;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    const int col = warp * (D / 4) + (lane & 15) * V, part_lane = lane >> 4;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + col;
    float acc[V] = {};
    // Match the baseline's four independent partial loads per iteration.
    for (int j = 0; j < np; j += 16) {
        sv x[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            x[u] = *reinterpret_cast<const sv*>(src + MIN_T(j + 4*u + part_lane, np - 1) * D);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int p = j + 4*u + part_lane;
            const float low = __shfl(w, p & 63);
            const float high = __shfl(w1, p & 63);
            const float raw = p < 64 ? low : high;
            const float wi = p < np ? raw : 0.f;
            #pragma unroll
            for (int e = 0; e < V; ++e) acc[e] += combine_to_float(x[u][e]) * wi;
        }
    }
    sv o;
    #pragma unroll
    for (int e = 0; e < V; ++e) {
        float sum = acc[e];
        sum += __shfl_xor(sum, 16);
        sum += __shfl_xor(sum, 32);
        scalar_t v; from_float(v, sum * inv); o[e] = v;
    }
    if (part_lane == 0)
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + col) = o;
}

} // namespace bhsd_fp8_gfx938


static void paged_attention_kernel_bhsd_fp8_native_launcher(
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
    (void)scale;
    (void)kv_cache_dtype;
    (void)q_scale;
    (void)k_scale;
    (void)v_scale;

    int batch = query.size(0);
    int mtp = query.size(1);
    int num_q_heads = query.size(2);
    int head_dim_qk = query.size(3);

    int num_blocks = key_cache.size(0);
    int num_kv_heads = key_cache.size(1);
    int block_size = key_cache.size(2);

    int head_dim_v = value_cache.size(3);
    int max_num_blocks_per_seq = block_tables.size(1);
    int num_heads = num_q_heads * mtp;


    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const int device_index = query.get_device();
    // Device properties are cached per host thread and device. Tensor
    // shapes select the kernel specialization; no sequence-length host read is needed.
    static thread_local std::unordered_map<int, int> compute_units_by_device;
    auto cu_entry = compute_units_by_device.find(device_index);
    if (cu_entry == compute_units_by_device.end()) {
        hipDeviceProp_t properties;
        TORCH_CHECK(hipGetDeviceProperties(&properties, device_index) == hipSuccess,
                    "BHSD FP8 device-property query failed");
        cu_entry = compute_units_by_device.emplace(device_index, properties.multiProcessorCount).first;
    }
    const int compute_units = cu_entry->second;
    int balanced_seq_len_tile = 1;

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

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(head_dim_v == head_dim_qk, "BHSD FP8 requires equal QK and V head sizes");
    TORCH_CHECK(!alibi_slopes.has_value() && !s_aux.has_value(), "BHSD FP8 native path does not support alibi or sinks");
    // Rows sharing one KV head (query heads x MTP tokens). Groups above 16 rows
    // use two 16-row MMAC tiles per CTA; larger groups split across CTAs.
    const int group_rows = num_heads / num_kv_heads;
    const bool wide = group_rows > 16;
    const int rows_per_cta = wide ? 32 : 16;
    const int row_groups = DIVIDE_ROUND_UP(group_rows, rows_per_cta);
    const int base_ctas = batch * num_kv_heads * row_groups;
    constexpr int S = 4;                  // waves per CTA
    constexpr int STAGE = 2;              // V tiles staged per LDS round
    // A single last CTA can merge a small group efficiently. Larger groups
    // retain a separate parallel merge so one CTA does not become the bottleneck.
    const bool fuse_combine = group_rows <= 4;
    // Smaller D makes the merge a larger fraction of the work. Give each
    // partition more chunks to reduce its partial-output and merge traffic.
    const int min_chunks = fuse_combine ? (head_dim_qk == 128 ? 8 : 4) : 2;
    int* counters = nullptr;
    if (fuse_combine) {
        // The ticket counters belong to this host thread, device and stream.
        // The last CTA resets them before the next launch on the same stream.
        // Keep each shape's allocation alive: a captured graph may still refer
        // to it after a larger batch or a different query group is launched.
        using CounterShapes = std::unordered_map<int, torch::Tensor>;
        static thread_local std::unordered_map<int, std::unordered_map<cudaStream_t, CounterShapes>> counter_buffers;
        auto& buffer = counter_buffers[device_index][stream][base_ctas];
        if (!buffer.defined()) {
            buffer = torch::zeros({base_ctas}, query.options().dtype(torch::kInt32));
        }
        counters = buffer.data_ptr<int>();
    }

    const bool dynamic_whole128 = head_dim_qk == 128 && group_rows == 48 &&
        query.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16 && key_cache.dtype() == torch::kFloat8_e4m3fn;
    auto launch_dynamic_whole = [&]<int MW>() {
        constexpr int smem = bhsd_fp8_gfx938::lds_bytes(S, STAGE, MW, 128);
        const int one_wave_parts = MAX_T(1, compute_units / base_ctas);
        const int temp_parts = 2 * one_wave_parts;
        const dim3 grid(num_kv_heads * row_groups, one_wave_parts, batch);
        if (print_param) printf("dynamic_whole [1], MW [%d], grid_size [%d] [%d] [%d], temp_parts [%d], one_wave_parts [%d], T [32768], smem [%d]\n",MW,grid.x,grid.y,grid.z,temp_parts,one_wave_parts,smem);
        bhsd_fp8_gfx938::pa_bhsd_fp8_dynamic_whole<MW><<<grid,dim3(256),smem,stream>>>(
            reinterpret_cast<uint16_t*>(out_ptr),reinterpret_cast<uint16_t*>(out_tmp),reinterpret_cast<const uint16_t*>(query_ptr),
            reinterpret_cast<const uint8_t*>(ker_cache_ptr),reinterpret_cast<const uint8_t*>(value_cache_ptr),
            block_tables_ptr,seq_lens_ptr,num_q_heads,num_kv_heads,mtp,row_groups,max_num_blocks_per_seq,
            q_stride,k_block_stride,v_block_stride,temp_parts,one_wave_parts,min_chunks,counters);
        if (temp_parts <= 64) {
            bhsd_fp8_gfx938::pa_bhsd_fp8_combine_column_waves<uint16_t,128,true><<<dim3(num_heads,batch),dim3(256),0,stream>>>(
                reinterpret_cast<uint16_t*>(out_ptr),reinterpret_cast<const uint16_t*>(out_tmp),seq_lens_ptr,
                temp_parts,one_wave_parts,num_heads,min_chunks);
        } else {
        bhsd_fp8_gfx938::pa_bhsd_fp8_dynamic_combine<uint16_t,128><<<dim3(num_heads,batch),dim3(256),0,stream>>>(
            reinterpret_cast<uint16_t*>(out_ptr),reinterpret_cast<const uint16_t*>(out_tmp),seq_lens_ptr,
            temp_parts,one_wave_parts,num_heads,min_chunks);
        }
    };
    if (dynamic_whole128) {
        launch_dynamic_whole.template operator()<3>();
        return;
    }

    HEADSIZE_SWITCH(head_dim_qk, [&] {
        Output_Type_SWITCH(out.dtype(), [&] {
            Input_Type_SWITCH(scalar_t, query.dtype(), key_cache.dtype(), [&] {
                BOOL_SWITCH(wide, WIDE, [&] {
                  BOOL_SWITCH(fuse_combine, FUSE, [&] {
                    constexpr int MW = WIDE ? 2 : 1;
                    constexpr int D = HEAD_SIZE;
                    TORCH_CHECK(2 == sizeof(scalar_t), "PA only support scalar_t b16 type for now!");
                    constexpr int smem = bhsd_fp8_gfx938::lds_bytes(S, STAGE, MW, D);
                    // One wave of CTAs gives enough parallelism without paying
                    // for additional partial outputs and combine traffic.
                    const int one_wave_parts = MAX_T(1, compute_units / base_ctas);
                    constexpr bool can_compact = MW == 2 && !FUSE && (D == 128 || D == 256) &&
                        is_e4m3 && std::is_same<scalar_t, uint16_t>::value && std::is_same<q_type, uint16_t>::value;
                    const bool compact = can_compact && D == 256 && group_rows == 48 &&
                        (D == 128 || D == 256) && query.dtype() == torch::kBFloat16 &&
                        out.dtype() == torch::kBFloat16 && key_cache.dtype() == torch::kFloat8_e4m3fn;
                    constexpr bool structural_type = !FUSE && (D == 128 || D == 256) && is_e4m3 &&
                        std::is_same<scalar_t, uint16_t>::value && std::is_same<q_type, uint16_t>::value;
                    const bool structural = structural_type && group_rows != 48;
                    balanced_seq_len_tile = one_wave_parts * ((compact || (structural && MW == 2)) ? 2 : 1);
                    constexpr int hier_main = S * STAGE * 2048 + (D * MW > 256 ? 16 * MW * (D + 16) : 0);
                    constexpr int hier_epilogue = 2 * S * 16 * MW * 4 + 64 + S * 16 * (128 + 4) * sizeof(scalar_t);
                    constexpr int hier_smem = hier_main > hier_epilogue ? hier_main : hier_epilogue;
                    const int selected_smem = structural ? hier_smem :
                        (compact ? bhsd_fp8_gfx938::lds_bytes(S, STAGE, MW, D, true) : smem);
                    const int launch_smem = FUSE ? MAX_T(selected_smem, (16 + 16 * MW * balanced_seq_len_tile) * 4) : selected_smem;
                    const dim3 grid(num_kv_heads * row_groups, balanced_seq_len_tile, batch);
                    if (print_param) {
                        printf("is_e4m3 [%d], HEAD_SIZE [%d], BLOCK_SIZE [%d], REUSE_KV_TIMES [%d], MW [%d], row_groups [%d], batch [%d], mtp [%d], "
                               "num_q_heads [%d], head_dim_qk [%d], num_blocks [%d], num_kv_heads [%d], head_dim_v [%d], "
                               "max_num_blocks_per_seq [%d], max_seq_len [%d], "
                               " balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d], one_wave_parts [%d], S [%d], min_chunks [%d], compact [%d], smem [%d]\n",
                               is_e4m3, D, block_size, group_rows, MW, row_groups, batch, mtp, num_q_heads, head_dim_qk,
                               num_blocks, num_kv_heads, head_dim_v, max_num_blocks_per_seq, max_seq_len,
                               balanced_seq_len_tile, grid.x, grid.y, grid.z, one_wave_parts, S, min_chunks, compact, launch_smem);
                    }
                    // Static targets preserve the original kernel symbols. Only
                    // the validated BF16/E4M3 wide instances need a compact entry.
                    auto launch_main = [&]<bool COMPACT>() {
                        if constexpr (COMPACT && can_compact) {
                            auto kernel = bhsd_fp8_gfx938::pa_bhsd_fp8_compact<scalar_t, q_type, is_e4m3, D, MW, S, STAGE, FUSE>;
                        kernel<<<grid, dim3(S * WARP_SIZE), launch_smem, stream>>>(
                        reinterpret_cast<scalar_t*>(out_ptr), reinterpret_cast<scalar_t*>(out_tmp),
                        reinterpret_cast<const q_type*>(query_ptr),
                        reinterpret_cast<const uint8_t*>(ker_cache_ptr), reinterpret_cast<const uint8_t*>(value_cache_ptr),
                        block_tables_ptr, seq_lens_ptr, num_q_heads, num_kv_heads, mtp, row_groups,
                        max_num_blocks_per_seq, q_stride, k_block_stride, v_block_stride,
                        balanced_seq_len_tile, one_wave_parts, min_chunks, counters);
                        } else {
                            if constexpr (structural_type) {
                                if (structural) {
                                    auto kernel = bhsd_fp8_gfx938::pa_bhsd_fp8_bf16red<scalar_t, q_type, is_e4m3, D, MW, S, STAGE, FUSE>;
                        kernel<<<grid, dim3(S * WARP_SIZE), launch_smem, stream>>>(
                        reinterpret_cast<scalar_t*>(out_ptr), reinterpret_cast<scalar_t*>(out_tmp),
                        reinterpret_cast<const q_type*>(query_ptr),
                        reinterpret_cast<const uint8_t*>(ker_cache_ptr), reinterpret_cast<const uint8_t*>(value_cache_ptr),
                        block_tables_ptr, seq_lens_ptr, num_q_heads, num_kv_heads, mtp, row_groups,
                        max_num_blocks_per_seq, q_stride, k_block_stride, v_block_stride,
                        balanced_seq_len_tile, one_wave_parts, min_chunks, counters);
                                    return;
                                }
                            }
                            auto kernel = bhsd_fp8_gfx938::pa_bhsd_fp8<scalar_t, q_type, is_e4m3, D, MW, S, STAGE, FUSE>;
                        kernel<<<grid, dim3(S * WARP_SIZE), launch_smem, stream>>>(
                        reinterpret_cast<scalar_t*>(out_ptr), reinterpret_cast<scalar_t*>(out_tmp),
                        reinterpret_cast<const q_type*>(query_ptr),
                        reinterpret_cast<const uint8_t*>(ker_cache_ptr), reinterpret_cast<const uint8_t*>(value_cache_ptr),
                        block_tables_ptr, seq_lens_ptr, num_q_heads, num_kv_heads, mtp, row_groups,
                        max_num_blocks_per_seq, q_stride, k_block_stride, v_block_stride,
                        balanced_seq_len_tile, one_wave_parts, min_chunks, counters);
                        }
                    };
                    if constexpr (can_compact) {
                        if (compact) launch_main.template operator()<true>();
                        else launch_main.template operator()<false>();
                    } else {
                        launch_main.template operator()<false>();
                    }
                    if ((!FUSE || MW > 1) && balanced_seq_len_tile > 1) {
                        if constexpr ((D == 128 || D == 256) && is_e4m3 &&
                            std::is_same<scalar_t, uint16_t>::value && std::is_same<q_type, uint16_t>::value) {
                            if (balanced_seq_len_tile <= 64) {
                                bhsd_fp8_gfx938::pa_bhsd_fp8_combine_column_waves<scalar_t,D><<<dim3(num_heads,batch),dim3(256),0,stream>>>(
                                    reinterpret_cast<scalar_t*>(out_ptr),reinterpret_cast<const scalar_t*>(out_tmp),seq_lens_ptr,
                                    balanced_seq_len_tile,one_wave_parts,num_heads,min_chunks);
                            } else if (balanced_seq_len_tile <= 128) {
                                bhsd_fp8_gfx938::pa_bhsd_fp8_combine_column_waves128<scalar_t,D><<<dim3(num_heads,batch),dim3(256),0,stream>>>(
                                    reinterpret_cast<scalar_t*>(out_ptr),reinterpret_cast<const scalar_t*>(out_tmp),seq_lens_ptr,
                                    balanced_seq_len_tile,one_wave_parts,num_heads,min_chunks);
                            } else {
                        bhsd_fp8_gfx938::pa_bhsd_fp8_combine<scalar_t, D><<<dim3(num_heads, batch), dim3(256), 0, stream>>>(
                            reinterpret_cast<scalar_t*>(out_ptr), reinterpret_cast<const scalar_t*>(out_tmp),
                            seq_lens_ptr, balanced_seq_len_tile, one_wave_parts, num_heads, min_chunks);
                            }
                        } else {
                        bhsd_fp8_gfx938::pa_bhsd_fp8_combine<scalar_t, D><<<dim3(num_heads, batch), dim3(256), 0, stream>>>(
                            reinterpret_cast<scalar_t*>(out_ptr), reinterpret_cast<const scalar_t*>(out_tmp),
                            seq_lens_ptr, balanced_seq_len_tile, one_wave_parts, num_heads, min_chunks);
                        }
                    }
                  });
                });
            });
        });
    });

    return;
}

// ---------------------------------------------------------------------------
// Public BHSD E4M3FN entry. Choose the implementation using the query device;
// __gfx* macros only select device compilation, not host runtime dispatch.
// ---------------------------------------------------------------------------
void paged_attention_kernel_bhsd_fp8_opt_launcher(
                                        torch::Tensor& out,                                 // [num_seqs, seqlen_q, num_q_heads, head_dim_v]
                                        torch::Tensor& query_in,                            // [num_seqs, seqlen_q, num_q_heads, head_dim_qk]
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
    // Q 在 token/head/dim 维不紧密时 (例如 QKV 融合张量的切片视图) 先复制为紧密布局; 已紧密时不复制
    // (batch 维可以有跨度)。下面的实现都按紧密的 token/head/dim 排列读取 Q。
    torch::Tensor query = query_in;
    if (query.dim() == 4 && !(query.stride(3) == 1 && query.stride(2) == query.size(3) &&
                              query.stride(1) == query.size(2) * query.size(3))) {
        query = query.contiguous();
    }
    TORCH_CHECK(key_cache.scalar_type() == torch::kFloat8_e4m3fn &&
                value_cache.scalar_type() == torch::kFloat8_e4m3fn,
                "BHSD FP8 requires E4M3FN K and V");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat16,
                "BHSD FP8 output must be BF16 or FP16");
    const bool query_b16 = query.scalar_type() == out.scalar_type();
    TORCH_CHECK(query_b16 || query.scalar_type() == torch::kFloat8_e4m3fn,
                "BHSD FP8 Q must match the output's 16-bit dtype or be E4M3FN");
    TORCH_CHECK(query.dim() == 4 && key_cache.dim() == 4 && value_cache.dim() == 4,
                "BHSD FP8 expects 4D Q, K and V tensors");
    TORCH_CHECK(key_cache.size(1) > 0 && query.size(2) % key_cache.size(1) == 0 &&
                query.size(3) == key_cache.size(3) && query.size(3) == value_cache.size(3),
                "BHSD FP8 requires divisible Q/KV heads and equal head dimensions");
    // gfx938: size 为 1 的维度下标恒为 0, 其 stride 不参与寻址 (PyTorch 对这类维度不规定 stride, 例如 kv_heads = 1 时
    // permute + contiguous 不重排), 只检查其余维度; gfx92a 保持原来的严格检查。
    auto dense = [](const torch::Tensor& t, int dim, int64_t stride, bool relaxed) {
        return (relaxed && t.size(dim) == 1) || t.stride(dim) == stride;
    };
    auto dense_layout = [&](bool relaxed) {
        return query.stride(3) == 1 && dense(query, 2, query.size(3), relaxed) &&
               dense(query, 1, query.size(2) * query.size(3), relaxed) && out.is_contiguous() &&
               key_cache.sizes() == value_cache.sizes() &&
               key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
               key_cache.stride(2) == key_cache.size(3) && value_cache.stride(2) == value_cache.size(3) &&
               dense(key_cache, 1, key_cache.size(2) * key_cache.size(3), relaxed) &&
               dense(value_cache, 1, value_cache.size(2) * value_cache.size(3), relaxed);
    };
    TORCH_CHECK(!alibi_slopes.has_value() && !s_aux.has_value() &&
                !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value(),
                "BHSD FP8 refactoring kernels require no alibi, sinks or scale tensors");
    TORCH_CHECK(std::abs(scale - 1.0 / std::sqrt(static_cast<double>(query.size(3)))) <= 1e-6,
                "BHSD FP8 refactoring kernels require scale = 1/sqrt(head_dim)");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    static thread_local std::unordered_map<int, std::string> architectures;
    auto arch = architectures.find(query.get_device());
    if (arch == architectures.end()) {
        hipDeviceProp_t properties;
        TORCH_CHECK(hipGetDeviceProperties(&properties, query.get_device()) == hipSuccess,
                    "BHSD FP8 device-property query failed");
        std::string name(properties.gcnArchName);
        name = name.substr(0, name.find(':'));
        arch = architectures.emplace(query.get_device(), name).first;
    }
    TORCH_CHECK(arch->second == "gfx938" || arch->second == "gfx92a",
                "BHSD E4M3FN refactoring is only available on gfx938/gfx92a");
    TORCH_CHECK(dense_layout(arch->second == "gfx938"),
                "BHSD FP8 requires dense Q token/head dimensions, contiguous output, "
                "and equal K/V shapes with dense dimensions inside each page");

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx92a__)
    if (arch->second == "gfx92a") {
        // E4M3FN Q 的每个值在 BF16/FP16 中都可精确表示: 转换后与 16 位 Q 走同一实现
        torch::Tensor query16 = query_b16 ? query : query.to(out.scalar_type());
        if (print_param) printf("BHSD_FP8 implementation=gfx92a kv=E4M3FN q=%s\n", query_b16 ? "b16" : "e4m3->b16");
        bhsd_fp8_gfx92a::launch(out, query16, key_cache, value_cache, scale, block_tables, seq_lens,
            alibi_slopes, kv_cache_dtype, q_scale, k_scale, v_scale, max_seq_len,
            s_aux, out_tmp, print_param);
        return;
    }
#endif
    // Teacher's gfx938 implementation (gfx938 only; gfx92a returned above).
    if (print_param) printf("BHSD_FP8 implementation=native_fp8 arch=%s kv=E4M3FN\n", arch->second.c_str());
    paged_attention_kernel_bhsd_fp8_native_launcher(out, query, key_cache, value_cache, scale, block_tables, seq_lens,
            alibi_slopes, kv_cache_dtype, q_scale, k_scale, v_scale, max_seq_len,
            s_aux, out_tmp, print_param);
}


