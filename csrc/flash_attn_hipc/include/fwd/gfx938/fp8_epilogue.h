#include "numeric_types.h"
#include "intrinsic.h"

__forceinline__ __device__ float fp8_attention_sink_load(const void *s_aux_ptr, int s_aux_type, int head_idx) {
    if (s_aux_type == 1) {
        return reinterpret_cast<const float *>(s_aux_ptr)[head_idx];
    } else if (s_aux_type == 2) {
        return UpCast<half_t, float>(reinterpret_cast<const half_t *>(s_aux_ptr)[head_idx]);
    } else {
        return UpCast<BFloat16, float>(reinterpret_cast<const BFloat16 *>(s_aux_ptr)[head_idx]);
    }
}

template<int kHeadDim, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_attention_sink_apply(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum softmax_scale,
    float sink_value
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        const ElementAccum old_scaled_max = scores_max[m_idx] * softmax_scale;
        const ElementAccum new_scaled_max = max(old_scaled_max, ElementAccum(sink_value));
        const ElementAccum old_rescale = __expf(old_scaled_max - new_scaled_max);
        scores_sum[m_idx] = scores_sum[m_idx] * old_rescale + __expf(ElementAccum(sink_value) - new_scaled_max);
        scores_max[m_idx] = new_scaled_max / softmax_scale;

        __float2 old_rescale_pair;
        old_rescale_pair[0] = old_rescale;
        old_rescale_pair[1] = old_rescale;
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 32; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                acc_o[k_loop][m_idx][n_idx].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[0], old_rescale_pair);
                acc_o[k_loop][m_idx][n_idx].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[1], old_rescale_pair);
            }
        }
    }
}

template<bool AssumeValidRows, int kHeadDim, int WARP_M, int WARP_N, bool StoreLSE, typename ElementAccum>
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
        if constexpr (StoreLSE) {
            lse[m_idx] = (sum == 0.f || sum != sum) ? INFINITY : __llvm_fma_f32(scores_max[m_idx], softmax_scale, __logf(sum));
        }
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




template<int WARP_M, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_lse(
    ElementAccum* softmax_lse_ptr,
    ElementAccum lse[WARP_M / 16],
    int row_offset_lse, /*(bidb * h + bidh) * actual_seqlen_q*/
    int actual_seqlen_q,
    int wave_row_offset, /*m_block * kBlockM + warp_id * WARP_M*/
    int lane_id
) {
    if (lane_id < 16) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            int lse_row_id = row_offset_lse + wave_row_offset + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);
            if (lse_row_id-row_offset_lse < actual_seqlen_q){
                softmax_lse_ptr[lse_row_id] = lse[m_idx];
            }
        }
    }
}



template<typename Element>
__forceinline__ __device__ vec4_uint fp8_make_output_rsrc(
    Element* ptr,
    int row_stride,
    uint32_t row_count,
    bool cache_swizzle
) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    const uint32_t stride_bytes = row_stride * sizeof(Element);
    const uint32_t descriptor_stride = cache_swizzle ? stride_bytes : 0u;
    return {
        uint32_t(base),
        uint32_t((base >> 32) & 0xffff) | (descriptor_stride << 16)
            | (cache_swizzle ? 0x40000000u : 0u),
        row_count * stride_bytes,
        0x00020000u
    };
}

template<bool FullQ, int kBlockM, int kHeadDim, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_output(
    Element* acc_o_ptr,
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    int m_block,
    int warp_id,
    int lane_id,
    int o_row_stride,
    int actual_seqlen_q
) {
    if constexpr (!FullQ) {
        const int wave_row_end = m_block * kBlockM
            + (warp_id + 1) * WARP_M;
        if (wave_row_end > actual_seqlen_q) {
            // A partial output wave retains the original masked stores.  All
            // preceding full waves in a non-even problem still use dwordx4.
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
                    #pragma unroll
                    for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                        int row_idx = warp_id * WARP_M
                            + ((lane_id & 15) >> 2) * 8
                            + m_idx * 4 + (lane_id & 3);
                        int col_idx = pv_loop * 32 + mmac_id * 16
                            + (lane_id >> 4) * 4;
                        int offset = row_idx * o_row_stride + col_idx;
                        union_vec2_f16x2<Element> v_data;
                        #pragma unroll
                        for (int vec_index = 0; vec_index < 2; ++vec_index) {
                            v_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(
                                acc_o[pv_loop][m_idx][mmac_id].f32x2[vec_index]);
                        }
                        if (m_block * kBlockM + row_idx < actual_seqlen_q) {
                            *(union_vec2_f16x2<Element>*)(acc_o_ptr + offset) = v_data;
                        }
                    }
                }
            }
            return;
        }
    }

    Element* wave_output = acc_o_ptr
        + warp_id * WARP_M * o_row_stride;
    const uint32_t stride_bytes = o_row_stride * sizeof(Element);
    // The descriptor stride field is 14 bits.  Very-wide BSHD rows retain
    // dwordx4 stores but leave cache-bank swizzle disabled.
    const bool cache_swizzle = stride_bytes < (1u << 14);
    const vec4_uint output_resource = fp8_make_output_rsrc(
        wave_output, o_row_stride, uint32_t(WARP_M), cache_swizzle);

    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
            #pragma unroll
            for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                union_vec2_f16x2<Element> local_data;
                #pragma unroll
                for (int vec_index = 0; vec_index < 2; ++vec_index) {
                    local_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(
                        acc_o[pv_loop][m_idx][mmac_id].f32x2[vec_index]);
                }

                vec4_int32 packet;
                packet[0] = local_data.i32[0];
                packet[1] = local_data.i32[1];
                packet[2] = __shfl_swap16(local_data.i32[0]);
                packet[3] = __shfl_swap16(local_data.i32[1]);

                // Lane groups 0/2 combine their data with groups 1/3,
                // respectively, so each active lane owns eight contiguous
                // BF16/FP16 output elements for one dwordx4 store.
                if ((lane_id & 16) == 0) {
                    const int wave_row_idx = ((lane_id & 15) >> 2) * 8
                        + m_idx * 4 + (lane_id & 3);
                    const int col_idx = pv_loop * 32 + mmac_id * 16
                        + (lane_id >> 5) * 8;
                    const int offset = wave_row_idx * o_row_stride + col_idx;
                    inline_buffer_store_dwordx4<vec4_int32, 1>(
                        packet, offset, output_resource, 0);
                }
            }
        }
    }

    if (cache_swizzle) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        asm volatile("buffer_wbinvl1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
}
