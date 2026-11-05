// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "fp8_qk_gemm_utils_mls_ds.h"
#include "static_switch.h"

// PrefetchK=false 版本：不 prefetch K，不需要额外参数
template<bool PrefetchK, int kHeadDimQK, int kHeadDimV,
         int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k(
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 32][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32],
    int8_t* v_lds,
    Element*& k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    // 等待从 lds 的数据返回
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);

    if constexpr (PrefetchK) {
        k_ptr += kBlockN * k_row_stride;
        if constexpr (kHeadDimQK == 192 && kBlockN == 64) {
            fp8_prefetch_k_to_lds_d192<false, Element>(
                k_ptr, k_lds, warp_id, k_row_stride, max_seq_kv_offset);
        } else {
            fp8_prefetch_k_to_lds<kHeadDimQK, WARP_N, Element>(
                k_ptr, k_lds, warp_id, k_row_stride, max_seq_kv_offset);
        }
    }

    // mmac stream
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDimV / 32; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(p_reg[m_idx].i8x8[k_loop], v_regs[k_loop][pv_loop].i8x8[mmac_id], acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// D128 steady-state pipeline. The next K tile reuses the V LDS tile,
// so all waves finish their V reads first. Its two MLS halves are then issued
// after the first and second groups of four PV MMACs to cover their latency.
template<bool FullNextK, int kHeadDimQK, int kHeadDimV, int kBlockN,
         int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k_d128(
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 32][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32],
    int8_t* v_lds,
    Element*& k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kHeadDimQK == 128 && kHeadDimV == 128
                  && kBlockN == 128 && WARP_M == 32 && WARP_N == 32);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);

    k_ptr += kBlockN * k_row_stride;
    flash::wait_all_warp_arrived();

    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDimV / 32; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 =
                        mmac_4interleave_b8<int8_t, ElementAccum>(
                            p_reg[m_idx].i8x8[k_loop],
                            v_regs[k_loop][pv_loop].i8x8[mmac_id],
                            acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
            if (k_loop == 0 && pv_loop == 0) {
                __builtin_amdgcn_sched_barrier(0);
                fp8_prefetch_k_to_lds_d128_part<FullNextK, 0, Element>(
                    k_ptr, k_lds, warp_id, k_row_stride,
                    FullNextK ? kBlockN : max_seq_kv_offset);
                __builtin_amdgcn_sched_barrier(0);
            } else if (k_loop == 0 && pv_loop == 1) {
                __builtin_amdgcn_sched_barrier(0);
                fp8_prefetch_k_to_lds_d128_part<FullNextK, 1, Element>(
                    k_ptr, k_lds, warp_id, k_row_stride,
                    FullNextK ? kBlockN : max_seq_kv_offset);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// QK192/V128 steady-state pipeline. V is already resident in VGPRs when this
// function starts, so the shared K/V buffer can be reused immediately. Spread
// the two D192 row halves across the first PV groups to hide their 128+64 MLS
// latency behind the remaining V128 MMAC stream.
template<bool FullNextK, int kBlockN, int WARP_M, int WARP_N,
         typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k_qk192_v128(
    vec4_Accum<ElementAccum> acc_o[4][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][4],
    int8_t* v_lds,
    Element*& k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kBlockN == 128 && WARP_M == 32 && WARP_N == 32);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);

    k_ptr += kBlockN * k_row_stride;
    flash::wait_all_warp_arrived();

    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < 4; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 =
                        mmac_4interleave_b8<int8_t, ElementAccum>(
                            p_reg[m_idx].i8x8[k_loop],
                            v_regs[k_loop][pv_loop].i8x8[mmac_id],
                            acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
            if (k_loop == 0 && pv_loop == 0) {
                __builtin_amdgcn_sched_barrier(0);
                fp8_prefetch_k_to_lds_d192_part<FullNextK, 0, Element>(
                    k_ptr, k_lds, warp_id, k_row_stride,
                    FullNextK ? kBlockN : max_seq_kv_offset);
                __builtin_amdgcn_sched_barrier(0);
            } else if (k_loop == 0 && pv_loop == 2) {
                __builtin_amdgcn_sched_barrier(0);
                fp8_prefetch_k_to_lds_d192_part<FullNextK, 1, Element>(
                    k_ptr, k_lds, warp_id, k_row_stride,
                    FullNextK ? kBlockN : max_seq_kv_offset);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// PrefetchK=true 版本：在 PV MMAC 期间 prefetch 下一块 K（paged KV）
template<int kHeadDimQK, int kHeadDimV, int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k_paged(
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 32][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32],
    int8_t* v_lds,
    Element* k_ptr_next,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset_next
) {
    // 等待从 lds 的数据返回
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);

    // Prefetch 下一块 K 到 LDS（与 MMAC 重叠）
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kHeadDimQK == 192 && kBlockN == 64) {
        fp8_prefetch_k_to_lds_d192<false, Element>(
            k_ptr_next, k_lds, warp_id, k_row_stride, max_seq_kv_offset_next);
    } else {
        fp8_prefetch_k_to_lds<kHeadDimQK, WARP_N, Element>(
            k_ptr_next, k_lds, warp_id, k_row_stride, max_seq_kv_offset_next);
    }
    __builtin_amdgcn_sched_barrier(0);

    // mmac stream
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDimV / 32; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(p_reg[m_idx].i8x8[k_loop], v_regs[k_loop][pv_loop].i8x8[mmac_id], acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<bool FullNextK, int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k_paged_d128(
    vec4_Accum<ElementAccum> acc_o[4][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][4],
    int8_t* v_lds,
    Element* k_ptr_next,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset_next
) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_sched_barrier(0);
    fp8_prefetch_k_to_lds_d128<FullNextK, Element>(
        k_ptr_next, k_lds, warp_id, k_row_stride, max_seq_kv_offset_next);
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < 4; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(
                        p_reg[m_idx].i8x8[k_loop], v_regs[k_loop][pv_loop].i8x8[mmac_id],
                        acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<bool PrefetchK, bool FullNextK, typename Element>
__forceinline__ __device__ void fp8_wait_v_and_prefetch_k_d256(
    Element*& k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n");
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (PrefetchK) {
        k_ptr += 64 * k_row_stride;
        fp8_prefetch_k_to_lds_d256<FullNextK, Element>(
            k_ptr, k_lds, warp_id, k_row_stride, max_seq_kv_offset);
    }
}

template<int KLoop, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_d256_kloop(
    vec4_Accum<ElementAccum> acc_o[8][2][2],
    union_vec32_fp8 p_reg[2],
    int8_t* v_lds
) {
    union_vec16_fp8 v_regs[8];
    int8_t* lds_load_ptr = v_lds + KLoop * 32 * 256 * sizeof(Element);
    #pragma unroll
    for (int pv_pair = 0; pv_pair < 8; pv_pair += 2) {
        const int pv_offset = (pv_pair / 2) * 128 * 16 * sizeof(Element);
        v_regs[pv_pair].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(
            lds_load_ptr + pv_offset, 0, 2, 2, 0);
        v_regs[pv_pair + 1].i32x4 = __builtin_hcu_ds_read_matrix_format_u8(
            lds_load_ptr + pv_offset + 32, 0, 2, 2, 0);
    }

    // The first four ordered LDS reads feed pv_loop 0..3. Start that half as
    // soon as it is ready and cover the remaining four reads with its MMACs.
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(4)\n");
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (int pv_loop = 0; pv_loop < 4; ++pv_loop) {
        #pragma unroll
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            #pragma unroll
            for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                acc_o[pv_loop][m_idx][mmac_id].f32 =
                    mmac_4interleave_b8<int8_t, ElementAccum>(
                        p_reg[m_idx].i8x8[KLoop],
                        v_regs[pv_loop].i8x8[mmac_id],
                        acc_o[pv_loop][m_idx][mmac_id].f32);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for (int pv_loop = 4; pv_loop < 8; ++pv_loop) {
        #pragma unroll
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            #pragma unroll
            for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                acc_o[pv_loop][m_idx][mmac_id].f32 =
                    mmac_4interleave_b8<int8_t, ElementAccum>(
                        p_reg[m_idx].i8x8[KLoop],
                        v_regs[pv_loop].i8x8[mmac_id],
                        acc_o[pv_loop][m_idx][mmac_id].f32);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_pv_gemm_d256(
    vec4_Accum<ElementAccum> acc_o[8][2][2],
    union_vec32_fp8 p_reg[2],
    int8_t* v_lds
) {
    fp8_pv_gemm_d256_kloop<0, Element, ElementAccum>(
        acc_o, p_reg, v_lds);
    fp8_pv_gemm_d256_kloop<1, Element, ElementAccum>(
        acc_o, p_reg, v_lds);
}
