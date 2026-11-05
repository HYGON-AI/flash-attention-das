// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "numeric_types.h"
#include "intrinsic.h"

namespace gfx92a_fp8_pa {

template<int kHeadDim, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_reduce_acc_o_across_waves(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum* reduce_lds,
    int warp_id,
    int lane_id
) {
    constexpr int WARP_NUM = 4;
    constexpr int WAVE_STRIDE = (kHeadDim / 32) * 64 * 4;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
            #pragma unroll
            for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                ElementAccum* dst =
                    reduce_lds + warp_id * WAVE_STRIDE +
                    pv_loop * 64 * 4 + lane_id * 4;
                *(vec4_fp32*)dst = acc_o[pv_loop][m_idx][mmac_id].f32;
            }
            __syncthreads();
            if (warp_id == 0) {
                #pragma unroll
                for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                    ElementAccum* src =
                        reduce_lds + pv_loop * 64 * 4 + lane_id * 4;
                    union_vec4_fp32 sum = *(union_vec4_fp32*)src;
                    #pragma unroll
                    for (int warp = 1; warp < WARP_NUM; ++warp) {
                        union_vec4_fp32 value =
                            *(union_vec4_fp32*)(src + warp * WAVE_STRIDE);
                        sum.u64[0] = __builtin_hcu_pk_add_f32(
                            sum.u64[0], value.u64[0]);
                        sum.u64[1] = __builtin_hcu_pk_add_f32(
                            sum.u64[1], value.u64[1]);
                    }
                    acc_o[pv_loop][m_idx][mmac_id].f32 = sum.f32;
                }
            }
            __syncthreads();
        }
    }
}

template<bool AssumeValidRows, int kHeadDim, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_rescale_acc_o(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum lse[WARP_M / 16],
    ElementAccum softmax_scale,
    ElementAccum v_descale
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        ElementAccum sum = scores_sum[m_idx];
        lse[m_idx] = (sum == 0.f || sum != sum) ? INFINITY : __llvm_fma_f32(scores_max[m_idx], softmax_scale, __logf(sum));
        ElementAccum total_rescale;
        if constexpr (AssumeValidRows) {
            total_rescale = v_descale / sum;
        } else {
            total_rescale = (sum == 0.f || sum != sum) ? 0.f : v_descale / sum;
        }
        __float2 total_scale_pair;
        total_scale_pair[0] = total_rescale;
        total_scale_pair[1] = total_rescale;
        // __float2 inv_sum_pair;
        // inv_sum_pair[0] = 1.0f / sum;
        // inv_sum_pair[1] = inv_sum_pair[0];
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 32; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                acc_o[k_loop][m_idx][n_idx].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[0], total_scale_pair);
                acc_o[k_loop][m_idx][n_idx].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[1], total_scale_pair);
            }
        }
    }
}

template<bool Is_even_MN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_lse(
    // ElementAccum* scores_max_ptr,
    // ElementAccum* scores_sum_ptr,
    ElementAccum* softmax_lse_ptr,
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum lse[WARP_M / 16],
    int row_offset_lse, /*(bidb * h + bidh) * actual_seqlen_q*/
    int actual_seqlen_q,
    int wave_row_offset, /*m_block * kBlockM + warp_id * WARP_M*/
    int lane_id
) {
#if defined(__gfx92a__)
    const int row_id = lane_id & 15;
    const int col_id = lane_id >> 4;
    if (col_id == 0) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            int thread_row_offset = wave_row_offset + m_idx * 16 + row_id;
            if (thread_row_offset < actual_seqlen_q) {
                softmax_lse_ptr[row_offset_lse + thread_row_offset] = lse[m_idx];
            }
        }
    }
#else
    if (lane_id < 16) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            int lse_row_id = row_offset_lse + wave_row_offset + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);;
            // scores_max_ptr[lse_row_id] = scores_max[m_idx];
            // scores_sum_ptr[lse_row_id] = scores_sum[m_idx];
            if (lse_row_id-row_offset_lse < actual_seqlen_q){
                softmax_lse_ptr[lse_row_id] = lse[m_idx];
            }
        }
    }
#endif
}

template<bool Is_even_MN, int kBlockM, int kHeadDim, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_output(
    Element* acc_o_ptr,
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    int m_block,
    int warp_id,
    int lane_id,
    int o_row_stride,
    int actual_seqlen_q
) {
#if defined(__gfx92a__)
    const int row_id = lane_id & 15;
    const int col_id = lane_id >> 4;
    if constexpr (Is_even_MN) {
        auto o_lds = (__attribute__((address_space(3))) ElementAccum *)(0)
            + warp_id * (kHeadDim / 32) * 2 * 16 * 16;
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            __syncthreads();

            #pragma unroll
            for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        o_lds[(pv_loop * 2 + mmac_id) * 16 * 16
                            + row_id * 16 + col_id + (((row_id & 3) + vec_idx) & 3) * 4]
                            = acc_o[pv_loop][m_idx][mmac_id].f32[vec_idx];
                    }
                }
            }

            #pragma unroll
            for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                #pragma unroll
                for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                    union_vec4_fp32 o_data;
                    o_data.f32 = *(vec4_fp32 *)(o_lds + (pv_loop * 2 + mmac_id) * 16 * 16
                        + row_id * 16 + (((row_id & 3) + col_id) & 3) * 4);

                    union_vec2_f16x2<Element> v_data;
                    v_data.f16x2[0] = DownCastPair<ElementAccum, Element>(o_data.f32x2[0]);
                    v_data.f16x2[1] = DownCastPair<ElementAccum, Element>(o_data.f32x2[1]);

                    int row_idx = warp_id * WARP_M + m_idx * 16 + row_id;
                    int col_idx = pv_loop * 32 + mmac_id * 16 + col_id * 4;
                    int offset = row_idx * o_row_stride + col_idx;
                    *(union_vec2_f16x2<Element>*)(acc_o_ptr + offset) = v_data;
                }
            }
        }
    } else {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            int row_idx = warp_id * WARP_M + m_idx * 16 + row_id;
            if (m_block * kBlockM + row_idx < actual_seqlen_q) {
                #pragma unroll
                for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                    #pragma unroll
                    for (int mmac_id = 0; mmac_id < WARP_N / 16; ++mmac_id) {
                        if constexpr (std::is_same<Element, bhalf_t>::value) {
                            #pragma unroll
                            for (int vec_pair = 0; vec_pair < 2; ++vec_pair) {
                                auto converted = DownCastPairNoPack<ElementAccum, Element>(
                                    acc_o[pv_loop][m_idx][mmac_id].f32[vec_pair * 2],
                                    acc_o[pv_loop][m_idx][mmac_id].f32[vec_pair * 2 + 1]);
                                #pragma unroll
                                for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
                                    int vec_idx = vec_pair * 2 + pair_idx;
                                    int col_idx = pv_loop * 32 + mmac_id * 16 + col_id + vec_idx * 4;
                                    int offset = row_idx * o_row_stride + col_idx;
                                    acc_o_ptr[offset] = converted[pair_idx];
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                int col_idx = pv_loop * 32 + mmac_id * 16 + col_id + vec_idx * 4;
                                int offset = row_idx * o_row_stride + col_idx;
                                acc_o_ptr[offset] = DownCast<ElementAccum, Element>(acc_o[pv_loop][m_idx][mmac_id].f32[vec_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
#else
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
            #pragma unroll
            for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                int row_idx = warp_id * WARP_M + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);
                int col_idx = pv_loop * 32 + mmac_id * 16 + (lane_id >> 4) * 4;
                int offset = row_idx * o_row_stride + col_idx;
                union_vec2_f16x2<Element> v_data;
                #pragma unroll
                for (int vec_index = 0; vec_index < 2; ++vec_index) {
                    v_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(acc_o[pv_loop][m_idx][mmac_id].f32x2[vec_index]);
                    // v_data = __builtin_hcu_cvt_pk_f16_f32(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2], acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2+1], false/*clamp*/, 0/*omod*/);
                    // v_data[0] = DownCast<ElementAccum, Float16>(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2]);
                    // v_data[1] = DownCast<ElementAccum, Float16>(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2+1]);
                }
                if (m_block * kBlockM + row_idx < actual_seqlen_q) {
                    *(union_vec2_f16x2<Element>*)(acc_o_ptr + offset) = v_data;
                }
                // *(vec4_fp32*)(acc_o_ptr + offset) = acc_o[pv_loop][m_idx][mmac_id].f32;
            }
        }
    }
#endif
}

} // namespace gfx92a_fp8_pa
