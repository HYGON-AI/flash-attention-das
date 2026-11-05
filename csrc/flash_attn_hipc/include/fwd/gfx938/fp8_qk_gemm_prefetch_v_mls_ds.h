// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "fp8_pv_gemm_utils_mls_ds.h"
#include "fp8_ds_read.h"
// #define USE_DS_READ_B128_FOR_INTERLEAVE4

template<int KLoop>
__forceinline__ __device__ void fp8_qk_load_k_d192_fragment(
    union_vec16_fp8 k_regs[2][3], int8_t* k_lds
) {
    constexpr int k_lds_load_offset = KLoop * 32 * 256;
    #pragma unroll
    for (int h_idx = 0; h_idx < 3; ++h_idx) {
        const int h_offset = h_idx == 2 ? 3 * 2048 : h_idx * 2048;
        k_regs[0][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(
            k_lds + k_lds_load_offset + h_offset, 0, 3, 1, 0);
        k_regs[1][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(
            k_lds + k_lds_load_offset + h_offset + 1024, 0, 3, 1, 0);
    }
}

template<int KLoop, int kBlockN, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_qk_mmac_d192_fragment(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    union_vec16_fp8 q_regs[WARP_M / 16][3],
    union_vec16_fp8 k_regs[WARP_N / 16][3]
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
            #pragma unroll
            for (int h_idx = 0; h_idx < 3; ++h_idx) {
                s_reg[KLoop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[0], k_regs[n_idx][h_idx].i8x8[0], s_reg[KLoop][m_idx][n_idx].f32);
                s_reg[KLoop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[1], k_regs[n_idx][h_idx].i8x8[1], s_reg[KLoop][m_idx][n_idx].f32);
            }
        }
    }
}

template<int kBlockN, int WARP_M, int WARP_N, typename Element,
         typename ElementAccum, bool PrefetchEarlyV = true>
__forceinline__ __device__ void fp8_qk_gemm_d192_v128_pipeline(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    union_vec16_fp8 q_regs[WARP_M / 16][3],
    int8_t* k_lds,
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int64_t v_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kBlockN == 128 && WARP_M == 32 && WARP_N == 32);
    // Ping-pong two D192 K fragments. This halves the K-register footprint
    // while retaining one complete fragment of LDS latency hiding.
    union_vec16_fp8 k_regs_0[2][3];
    union_vec16_fp8 k_regs_1[2][3];
    fp8_qk_load_k_d192_fragment<0>(k_regs_0, k_lds);
    fp8_qk_load_k_d192_fragment<1>(k_regs_1, k_lds);
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
            }
        }
    }

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(4)\n");
    __builtin_amdgcn_sched_barrier(0);
    fp8_qk_mmac_d192_fragment<0, kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, q_regs, k_regs_0);
    fp8_qk_load_k_d192_fragment<2>(k_regs_0, k_lds);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(4)\n");
    __builtin_amdgcn_sched_barrier(0);
    fp8_qk_mmac_d192_fragment<1, kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, q_regs, k_regs_1);
    fp8_qk_load_k_d192_fragment<3>(k_regs_1, k_lds);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(4)\n");
    __builtin_amdgcn_sched_barrier(0);
    fp8_qk_mmac_d192_fragment<2, kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, q_regs, k_regs_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (PrefetchEarlyV) {
        asm volatile("s_barrier" ::: "memory");
        fp8_prefetch_v_to_lds<128, WARP_N, Element>(
            v_ptr, v_lds, warp_id, v_row_stride, max_seq_kv_offset);
    }
    fp8_qk_mmac_d192_fragment<3, kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, q_regs, k_regs_1);
}

template<int kBlockN, int kHeadDim, int WARP_M, int WARP_N, typename Element, typename ElementAccum,
         bool UseD128DsRead = false, bool UseEarlyV128Pipeline = false,
         bool UseD192FragmentPipeline = false>
__forceinline__ __device__ void fp8_qk_gemm(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64],
    int8_t* k_lds,
    Element* v_ptr = nullptr,
    int8_t* v_lds = nullptr,
    int warp_id = 0,
    int64_t v_row_stride = 0,
    int max_seq_kv_offset = 0
) {
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    int tx = threadIdx.x;
    int lane_id = tx & 63;
    int row = (lane_id & 15) >> 1;
    int col = lane_id >> 4;
    int col_swizzle = (row + col) & 3;

    // 等待 K 的数据都写到 lds 了, 4 wave 同步
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n"); // hint: pv mmac 太短, 不够隐藏这一段时延, 可以考虑把 vmcnt 拆细一点, 先等一部分数据回来计算也可以
    __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();

    if constexpr ((UseEarlyV128Pipeline || UseD192FragmentPipeline) && kHeadDim == 192) {
        fp8_qk_gemm_d192_v128_pipeline<kBlockN, WARP_M, WARP_N, Element,
                                       ElementAccum, UseEarlyV128Pipeline>(
            s_reg, q_regs, k_lds, v_ptr, v_lds, warp_id,
            v_row_stride, max_seq_kv_offset);
        return;
    }

    // 直接从 lds 读数据, 看看 lds 的数据排布
    union_vec16_fp8 k_regs[kBlockN / WARP_N][WARP_N / 16][kHeadDim / 64];
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
         #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
            int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
            inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0,    k_regs[k_loop][0][0].i32x4);
            inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024, k_regs[k_loop][1][0].i32x4);
            inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[k_loop][0][1].i32x4);
            inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[k_loop][1][1].i32x4);
        #else
            int k_lds_load_offset = k_loop * WARP_N * kLdsHeadDimStride;
            if constexpr (UseD128DsRead) {
                static_assert(kHeadDim == 128 && WARP_N == 32,
                              "D128 QK pipeline expects D128/WARP_N32");
                inline_ds_read_matrix_trans_64x16_b8x4_no_wait(
                    k_lds + k_lds_load_offset,
                    k_regs[k_loop][0][0].i32x4, k_regs[k_loop][1][0].i32x4,
                    k_regs[k_loop][0][1].i32x4, k_regs[k_loop][1][1].i32x4);
            } else {
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0,    k_regs[k_loop][0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024, k_regs[k_loop][1][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[k_loop][0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[k_loop][1][1].i32x4, true/*transpose*/)
                #pragma unroll
                for (int h_idx = 0; h_idx < kHeadDim / 64; ++h_idx) {
                    const int h_offset = (kHeadDim == 192 && h_idx == 2) ? 3 * 2048 : h_idx * 2048;
                    k_regs[k_loop][0][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + h_offset + 0, 0, 3, 1, 0);
                    k_regs[k_loop][1][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + h_offset + 1024, 0, 3, 1, 0);
                }
            }
        #endif
    }
    // init s_reg
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
            }
        }
    }

    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(%0)\n":: "B"((kBlockN / WARP_N - k_loop - 1) * 4));
        // asm volatile("s_waitcnt lgkmcnt(0)\n");
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (UseEarlyV128Pipeline) {
            static_assert((kHeadDim == 128 || kHeadDim == 192) && kBlockN == 128
                          && WARP_M == 32 && WARP_N == 32,
                          "early V128 pipeline expects a D128/D192 QK 128x128 tile");
            if (k_loop == kBlockN / WARP_N - 1) {
                // Every wave has completed all K LDS reads before V starts
                // overwriting the shared K/V buffer. The final QK MMAC then
                // overlaps the V global-to-LDS MLS transaction.
                asm volatile("s_barrier" ::: "memory");
                fp8_prefetch_v_to_lds<128, WARP_N, Element>(
                    v_ptr, v_lds, warp_id, v_row_stride, max_seq_kv_offset);
            }
        }
        // ======================================================== QK mmac ======================================================================
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                #pragma unroll
                for (int h_idx = 0; h_idx < kHeadDim / 64; ++h_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[0], k_regs[k_loop][n_idx][h_idx].i8x8[0], s_reg[k_loop][m_idx][n_idx].f32);
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[1], k_regs[k_loop][n_idx][h_idx].i8x8[1], s_reg[k_loop][m_idx][n_idx].f32);
                }
            }
        }
    }

}
