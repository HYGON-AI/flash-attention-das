#pragma once
#include "fp8_qk_gemm_utils_mls_ds_gfx92a.h"
#include "static_switch.h"

namespace gfx92a_fp8_pa {

template<bool PrefetchK, bool Is_even_MN, int kHeadDim, int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum, int WARP_NUM = 4>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32],
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
        __builtin_amdgcn_sched_barrier(0);
        k_ptr += kBlockN * k_row_stride;
        __builtin_amdgcn_sched_barrier(0);
        fp8_prefetch_k_to_lds<Is_even_MN, kHeadDim, WARP_N, Element, WARP_NUM>(
            k_ptr, k_lds, warp_id, k_row_stride, max_seq_kv_offset);
    }

    // mmac stream
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 = mmac_4interleave_fp8_gfx92a<fp8_e4m3, Element, ElementAccum>(p_reg[m_idx].i8x8[k_loop], v_regs[k_loop][pv_loop].i8x8[mmac_id], acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    flash::lower_priority();
}



// PrefetchK=true 版本：在 PV MMAC 期间 prefetch 下一块 K（paged KV）
template<bool Is_even_MN, int kHeadDim, int kBlockN, int WARP_M, int WARP_N, typename Element, typename ElementAccum, int WARP_NUM = 4>
__forceinline__ __device__ void fp8_pv_gemm_and_prefetch_k_paged(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    union_vec32_fp8 p_reg[WARP_M / 16],
    union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32],
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
    fp8_prefetch_k_to_lds<Is_even_MN, kHeadDim, WARP_N, Element, WARP_NUM>(k_ptr_next, k_lds, warp_id, k_row_stride, max_seq_kv_offset_next);
    __builtin_amdgcn_sched_barrier(0);

    // mmac stream
    #pragma unroll
    for (int k_loop = 0; k_loop < kBlockN / WARP_N; k_loop += 1) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                    acc_o[pv_loop][m_idx][mmac_id].f32 = mmac_4interleave_fp8_gfx92a<fp8_e4m3, Element, ElementAccum>(p_reg[m_idx].i8x8[k_loop], v_regs[k_loop][pv_loop].i8x8[mmac_id], acc_o[pv_loop][m_idx][mmac_id].f32);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    flash::lower_priority();
}

} // namespace gfx92a_fp8_pa
