// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "fwd/utils.h"
#include "fwd/pv_gemm_utils.h"

template <typename Element>
__forceinline__ __device__ void mixed_decode_buffer_load_4_raw_lds(Element *const shared_addr,
                                                                   const vec4_uint global_addr, const int lds_offset,
                                                                   const int gv_offset_s, const int gv_offset_v) {
    static_assert(std::is_same_v<Element, BFloat16>);
    const int lds_addr = reinterpret_cast<size_t>(shared_addr + lds_offset);
    asm volatile("s_mov_b32 m0, %1\n\t"
                 "buffer_load_dword %0, %2, %3, offen offset:0, lds\n" ::"v"(gv_offset_v),
                 "s"(lds_addr), "s"(global_addr), "s"(gv_offset_s)
                 : "memory");
}

template <typename Element>
__forceinline__ __device__ void mixed_decode_load_v_32x64_stage(const vec4_uint v_addr, Element *const v_lds,
                                                                const int stage_id, const int head_dim_base,
                                                                const int warp_id, const int vcache_seqlen_stride,
                                                                const int max_seq_k_offset) {
    static_assert(std::is_same_v<Element, BFloat16>);
    constexpr int kRows = 32;
    constexpr int kColumns = 64;
    constexpr int kRowsPerRequest = 16;
    constexpr int kStageElements = kRows * kColumns / sizeof(Element);
    constexpr int kRequestElements = kRowsPerRequest * kColumns / sizeof(Element);
    const int lane_id = threadIdx.x & 63;
    const int row_in_request = lane_id >> 2;
    const int column = (lane_id & 3) * 16;
    const int wave_stage = warp_id * 2 + stage_id;

#pragma unroll
    for (int load = 0; load < 2; ++load) {
        const int row = min(warp_id * 32 + load * kRowsPerRequest + row_in_request, max_seq_k_offset - 1);
        fp8_buffer_load_16_raw_lds(v_lds, v_addr, wave_stage * kStageElements + load * kRequestElements, head_dim_base,
                                   row * vcache_seqlen_stride + column);
    }
}

template <typename Element>
__forceinline__ __device__ void mixed_decode_read_v_32x64_half(Element *const v_lds, const int stage_id,
                                                               const int half_id, const int warp_id,
                                                               union_vec2_f16x2<Element> v_reg[4]) {
    static_assert(std::is_same_v<Element, BFloat16>);
    constexpr int kRows = 32;
    constexpr int kColumns = 64;
    constexpr int kStageBytes = kRows * kColumns;
    const int lane_id = threadIdx.x & 63;
    const int row_in_group = lane_id >> 4;
    const int column = (lane_id & 15) * 2 + half_id * 32;
    const uint8_t *const tile = reinterpret_cast<const uint8_t *>(v_lds) + (warp_id * 2 + stage_id) * kStageBytes;

    uint32_t input0[4];
    uint32_t input1[4];
#pragma unroll
    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
        const int row0 = vec_idx * 8 + row_in_group;
        const int row1 = row0 + 4;
        inline_ds_read_b16_no_wait_bytes(reinterpret_cast<size_t>(tile + row0 * kColumns + column), input0[vec_idx]);
        inline_ds_read_b16_no_wait_bytes(reinterpret_cast<size_t>(tile + row1 * kColumns + column), input1[vec_idx]);
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
        const uint64_t packed_bf16 =
            static_cast<uint64_t>(e5m2x2_to_bf16x2(static_cast<uint16_t>(input0[vec_idx]))) |
            (static_cast<uint64_t>(e5m2x2_to_bf16x2(static_cast<uint16_t>(input1[vec_idx]))) << 32);
        v_reg[vec_idx].f32 = __builtin_bit_cast(vec2_fp32, packed_bf16);
    }
}

template <int K_LOOP_COUNT, int kBlockM, int kBlockN, int kBlockK, int M_WARP_COUNT, int PV_N_WARP_COUNT,
          int PV_K_WARP_COUNT, int STAGES, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void
mixed_kvcache_pv_gemm_32x64(const vec4_uint v_addr, Element *const v_lds,
                            union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * PV_K_WARP_COUNT][4],
                            vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * (kBlockN / 32)][4],
                            const int warp_id, const int vcache_seqlen_stride, const int max_seq_k_offset) {
    static_assert(kBlockM == 32 && kBlockN == 32);
    static_assert(kBlockK == 64 || kBlockK == 128);
    static_assert(M_WARP_COUNT == 1 && PV_N_WARP_COUNT == 1 && PV_K_WARP_COUNT == 1);
    static_assert(STAGES == 2);
    static_assert(M_MMAC_COUNT == 1 || M_MMAC_COUNT == 2);
    static_assert(K_LOOP_COUNT % 2 == 0 && (WARP_NUM == 2 || WARP_NUM == 4));
    constexpr int kRawStageCount = K_LOOP_COUNT / 2;
    int stage_id = 0;

#pragma unroll
    for (int raw_stage = kRawStageCount - 1; raw_stage > 0; --raw_stage) {
        const int next_stage_id = stage_id ^ 1;
        mixed_decode_load_v_32x64_stage(v_addr, v_lds, next_stage_id, (raw_stage - 1) * 64, warp_id,
                                        vcache_seqlen_stride, max_seq_k_offset);
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait_nosync<2>();
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (int half_id = 0; half_id < 2; ++half_id) {
            union_vec2_f16x2<Element> v_reg[4];
            mixed_decode_read_v_32x64_half(v_lds, stage_id, half_id, warp_id, v_reg);
#pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                vec4_Element<Element> v_vgpr[2];
#pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    v_vgpr[min_tile_n] = vec4_Element<Element>{
                        v_reg[min_tile_k * 2].f16x2[0][min_tile_n], v_reg[min_tile_k * 2].f16x2[1][min_tile_n],
                        v_reg[min_tile_k * 2 + 1].f16x2[0][min_tile_n], v_reg[min_tile_k * 2 + 1].f16x2[1][min_tile_n]};
                }
                asm volatile("s_setprio 1");
                const int pv_tile_id = raw_stage * 2 + half_id;
#pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    const vec4_Element<Element> p_vgpr = make_vec4_f16(
                        p_reg[0][min_tile_k * 2 + min_tile_m].f16[0], p_reg[0][min_tile_k * 2 + min_tile_m].f16[1],
                        p_reg[0][min_tile_k * 2 + min_tile_m].f16[2], p_reg[0][min_tile_k * 2 + min_tile_m].f16[3]);
#pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                            p_vgpr, v_vgpr[min_tile_n], pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                asm volatile("s_setprio 0");
            }
        }
        stage_id = next_stage_id;
    }

    __builtin_amdgcn_sched_barrier(0);
    buffer_load_lds_dwordx1_wait_nosync<0>();
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int half_id = 0; half_id < 2; ++half_id) {
        union_vec2_f16x2<Element> v_reg[4];
        mixed_decode_read_v_32x64_half(v_lds, stage_id, half_id, warp_id, v_reg);
#pragma unroll
        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
            vec4_Element<Element> v_vgpr[2];
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                v_vgpr[min_tile_n] = vec4_Element<Element>{
                    v_reg[min_tile_k * 2].f16x2[0][min_tile_n], v_reg[min_tile_k * 2].f16x2[1][min_tile_n],
                    v_reg[min_tile_k * 2 + 1].f16x2[0][min_tile_n], v_reg[min_tile_k * 2 + 1].f16x2[1][min_tile_n]};
            }
            asm volatile("s_setprio 1");
#pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                const vec4_Element<Element> p_vgpr = make_vec4_f16(
                    p_reg[0][min_tile_k * 2 + min_tile_m].f16[0], p_reg[0][min_tile_k * 2 + min_tile_m].f16[1],
                    p_reg[0][min_tile_k * 2 + min_tile_m].f16[2], p_reg[0][min_tile_k * 2 + min_tile_m].f16[3]);
#pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    pv_reg[half_id][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                        p_vgpr, v_vgpr[min_tile_n], pv_reg[half_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            asm volatile("s_setprio 0");
        }
    }

    __syncthreads();
}


template<int kHeadDim, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_K, int stage_id, int WARP_NUM, typename Element, int STAGES>
__forceinline__ __device__ void kvcache_prefetch_v_to_lds_tile16x32(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_kv_offset=-1) {

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    constexpr int NEXT_DWORD_OFFSET = 64; // 8x32 的数据, 一个 wave 每个线程读 4 个 half, 即 2 个 dword, 使用 ds_read2_b32 指令, 第二个 dword 偏移 64 个 dword

#if defined(USE_BUFFER_LOAD_DWORDX4)
    constexpr int READ_ONCE_LINES    = 16;                                 // 一个 warp 每次读几行数据, loadx4, 每个线程读取 8 个 Half, 每行 32 个 Half 需要 32 / 8 = 4 个线程, 所以一个 wave 64 线程会读取 16 行
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;               // 一个 warp 每次 load 多少数据, 16x32
    constexpr int V_LDS_LOAD_NUM     = kBlockN * WARP_K / READ_ONCE_COUNT; // 一个 warp 一共要发几次读取请求
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;                     // 一个 warp 一共要发几次读取请求
    constexpr int READ_ELEMENT_COUNT = 8;                                  // 每个线程一次读取几个 Half
    int v_lane_headdim_n_idx         = lane_id & 3;                        // 当前 lane 负责这个 warp 的第几个 dwordx2
    int base                         = (laneid_shfl_2 & 0xc);              // 第几个 4 线程组的最小id
    int tail                         = (laneid_shfl_2 & 0x3);              // 4 线程组中的第几个线程
    int v_lane_seq_k_idx             = laneid_shfl_2;                      // global -> lds, seqlen 方向的坐标
    int v_ds_read_offset             = laneid_shfl_4 * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 按照线程 [0, 16, 32, 48] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dwordx4_lds<Element, 2>;
#else
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = laneid_shfl_4; // 0-15 read row 0; 16-31 read row 1; 32-47 read row 2; 48-63 read row 3
    int v_ds_read_offset             = laneid_shfl_4 * 32 + (lane_id & 15) * 2;
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element, 2>;
#endif

    int n_loop = (kHeadDim / kBlockN) - 1;
    if constexpr (STAGES > 1) {
        int v_block_buffer_load_global_offset = n_loop * kBlockN;
        #pragma unroll
        for (int load = 0; load < V_LOAD_REQUESTS; ++load) {
            int v_warp_buffer_load_lds_offset = load * READ_ONCE_COUNT;
            int v_gvoffset_s = v_block_buffer_load_global_offset / 2;
            int v_gvoffset_v = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load * READ_ONCE_LINES + warp_id * WARP_K, max_seq_kv_offset - 1) * kvcache_seqlen_stride) / 2;
            int v_lds_offset = v_warp_buffer_load_lds_offset / 2;
            BUFFER_LOAD_FUNC(v_lds + warp_id * STAGES * WARP_K * kBlockN + stage_id * WARP_K * kBlockN, v_addr, v_lds_offset, v_gvoffset_s, v_gvoffset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}
