// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "flash_fwd_b16_fa.h"


namespace flash {

#include "fwd/int8_pv_gemm_prefetch_k.h"
#include "fwd/int8_qk_gemm_prefetch_v.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                            GFX938 kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool FP8_DEBUG, bool Is_even_MN, int kBlockM, int kBlockN, int WARP_M, int WARP_N, typename Element>
__forceinline__ __device__ void fp8_debug_p_reg(
    Element* p_reg_ptr,
    union_vec32_fp8 p_reg[WARP_M / 16],
    int bidb,
    int bidh,
    int h,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int max_seq_q_offset,
    int max_seq_kv_offset,
    int m_block,
    int n_block_loop,
    int warp_id,
    int lane_id
) {
    if constexpr (FP8_DEBUG) {
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (FP8_DEBUG) {
            Element* p_reg_buffer = p_reg_ptr + (bidb * h + bidh) * actual_seqlen_q * actual_seqlen_k;
            #pragma unroll
            for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                        int row_pos = m_block * kBlockM + warp_id * WARP_M + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);
                        int col_pos = (lane_id >> 4) * 8 + n_idx * 4 + k_loop * WARP_N + n_block_loop * kBlockN;
                        *(int32_t*)(p_reg_buffer + row_pos * actual_seqlen_k + col_pos) = p_reg[m_idx].i32[k_loop * 2 + n_idx];
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
    }
}

#if defined(__gfx938__) || defined(__gfx946__)
#include "fwd/gfx938/fp8_qk_gemm_prefetch_v_mls_ds.h"
#include "fwd/gfx938/fp8_pv_gemm_prefetch_k_mls_ds.h"
#include "fwd/gfx938/fp8_softmax_gfx938.h"
#include "fwd/gfx938/fp8_epilogue.h"
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_fp8_attn_mha_1rowblock_gfx938(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int WARP_K    = 32;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr bool Is_d256  = kHeadDim == 256 && kHeadDimV == 256
                              && kBlockN == 64 && WARP_M == 32 && WARP_N == 32;
    constexpr bool UseLaneDeferredRowSum = kHeadDimV == 192 || Is_d256;
    using ScoresSumState = std::conditional_t<UseLaneDeferredRowSum,
        fp8_deferred_lane_sum<ElementAccum>, vec2_Accum<ElementAccum>>;

    // 获取当前 TG 处理的任务大小
    const flash::BlockInfo</*Varlen=*/Is_Varlen> binfo(params, bidb);

    // 判断任务边界
    int max_seq_q_offset = binfo.actual_seqlen_q - m_block * kBlockM;
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0/* || bidh >= h*/) return;

    // 获取 wave id
    // int __warp_id = threadIdx.x >> 6;
    // int warp_id = __builtin_amdgcn_readfirstlane(__warp_id);

    // 定义 lds, 128x128 个 fp8, 16384 bytes
    // __shared__ int8_t lds[16384 + 4096 + 16384 + 4096];
    extern __shared__ int8_t lds[];
    int8_t* q_lds = lds + 0;
    int8_t* k_lds = lds + 0;
    int8_t* v_lds = Is_d256 ? lds + kBlockN * kHeadDimV * sizeof(Element_k) : lds;

    // ========================================== 计算 offset ===========================================
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int64_t row_offset_lse_base;
    if constexpr (Is_Varlen) {
        if constexpr (Layout == 1) { /* bshd: q/o are [total_q, h, d] */
            row_offset_q = (int64_t(binfo.sum_s_q) + m_block * kBlockM) * int64_t(params.q_row_stride) + params.q_head_stride * bidh;
            row_offset_k = int64_t(binfo.sum_s_k) * int64_t(params.k_row_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
            row_offset_v = int64_t(binfo.sum_s_k) * int64_t(params.v_row_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
            row_offset_o = int64_t(binfo.sum_s_q) * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * int64_t(params.o_row_stride);
            row_offset_lse_base = bidh * int64_t(params.total_q) + binfo.sum_s_q;
        } else { /* bhsd */
            row_offset_q = int64_t(binfo.sum_s_q) * int64_t(params.q_row_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(params.q_row_stride);
            row_offset_k = int64_t(binfo.sum_s_k) * int64_t(params.k_row_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
            row_offset_v = int64_t(binfo.sum_s_k) * int64_t(params.v_row_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
            row_offset_o = int64_t(binfo.sum_s_q) * int64_t(params.o_row_stride) + bidh * params.o_head_stride + m_block * kBlockM * int64_t(params.o_row_stride);
            row_offset_lse_base = bidh * int64_t(params.total_q) + binfo.sum_s_q;
        }
    } else {
        row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(params.q_row_stride);
        row_offset_k = bidb * int64_t(params.k_batch_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = bidb * int64_t(params.v_batch_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + m_block * kBlockM * int64_t(params.o_row_stride);
        row_offset_lse_base = (bidb * params.h + bidh) * int64_t(binfo.actual_seqlen_q);
    }

    Element_k* q_ptr   = reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q;
    Element_k* k_ptr   = reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k;
    Element_k* v_ptr   = reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v;

#if defined(FP8_NO_SCALES)
    // Keep the long-lived softmax scales in VGPRs. Leaving the uniform kernarg
    // values in SGPRs increases SGPR pressure and degrades D256 scheduling.
    ElementAccum softmax_scale;
    ElementAccum softmax_scale_log2;
    asm volatile("v_mov_b32 %0, %2\n\t"
                 "v_mov_b32 %1, %3"
                 : "=v"(softmax_scale), "=v"(softmax_scale_log2)
                 : "s"(params.scale_softmax), "s"(params.scale_softmax_log2));
#else
    ElementAccum* q_descale_ptr   = reinterpret_cast<ElementAccum*>(params.q_descale_ptr);
    ElementAccum* k_descale_ptr   = reinterpret_cast<ElementAccum*>(params.k_descale_ptr);
    ElementAccum* v_descale_ptr   = reinterpret_cast<ElementAccum*>(params.v_descale_ptr);

    ElementAccum q_descale   = q_descale_ptr[0];
    ElementAccum k_descale   = k_descale_ptr[0];
    ElementAccum qk_descale = q_descale * k_descale;
    ElementAccum softmax_scale = params.scale_softmax * qk_descale;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2 * qk_descale;
    ElementAccum v_descale = v_descale_ptr[0];
#endif
    // acc_o_ptr = reinterpret_cast<ElementAccum*>(acc_o_ptr) + row_offset_o;
    ElementAccum* softmax_lse_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr);
    Element_k* p_reg_ptr = reinterpret_cast<Element_k *>(params.p_ptr);
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;

    // ======================================================== 读取 Q ======================================================================
    fp8_prefetch_q_to_lds<kHeadDim, WARP_M, Element_k>(q_ptr, q_lds, warp_id, params.q_row_stride, max_seq_q_offset);

    // 计算解决 bank 冲突必须的一些变量
    int tx = threadIdx.x;
    int lane_id = tx & 63;

    // 准备存储最大值, 求和, acc_o 寄存器 等
    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    ScoresSumState scores_sum_partial[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 32][WARP_M / 16][WARP_N / 16];
    fp8_attention_initialize<kHeadDimV, WARP_M, WARP_N, ElementAccum>(scores_max, scores_sum, acc_o);
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        if constexpr (UseLaneDeferredRowSum) {
            scores_sum_partial[m_idx].value = 0.0f;
        } else {
            scores_sum_partial[m_idx].data = 0.0;
        }
    }

    // 从 lds 读取 q 的数据, 不需要同步
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64];
    load_q_from_lds_to_vgpr<kHeadDim, WARP_M, Element_k>(q_regs, q_lds, warp_id, lane_id);

    // ======================================================== Prefetch K ======================================================================
    if constexpr (Is_d256) {
        fp8_prefetch_k_to_lds_d256<Is_even_MN, Element_k>(
            k_ptr, k_lds, warp_id, params.k_row_stride, binfo.actual_seqlen_k);
    } else if constexpr (kHeadDim == 192 && kBlockN == 64) {
        fp8_prefetch_k_to_lds_d192<Is_even_MN, Element_k>(
            k_ptr, k_lds, warp_id, params.k_row_stride, binfo.actual_seqlen_k);
    } else {
        fp8_prefetch_k_to_lds<kHeadDim, WARP_N, Element_k>(
            k_ptr, k_lds, warp_id, params.k_row_stride, binfo.actual_seqlen_k);
    }

    // ======================================================== Mainloop ======================================================================
    // 计算当前 block 计算任务的边界，带 causal mask 的场景可以少计算一些
    int n_block_min = 0;
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal) {
        n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + 0/*params.window_size_right*/, kBlockN));
    }
    constexpr int n_masking_steps = !Is_causal
        ? 1
        : ceil_div(kBlockM, kBlockN) + (Needs_extra_mask ? 1 : 0);
    constexpr bool Assume_valid_rows = !Is_local && (!Is_causal || !Is_Varlen);

#if defined(__gfx938__) && defined(FP8_NO_SCALES)
    constexpr bool UseD128FullTilePipeline = !Is_local && !Has_alibi && !Is_dropout
        && kHeadDim == 128 && kHeadDimV == 128
        && kBlockN == 128 && WARP_M == 32 && WARP_N == 32;
    constexpr bool UseQK192V128FullTilePipeline = !Is_local && !Has_alibi && !Is_dropout
        && kHeadDim == 192 && kHeadDimV == 128
        && kBlockN == 128 && WARP_M == 32 && WARP_N == 32;
    // Even-MN instances already omit KV filters; keep their original loop.
    constexpr bool UseD256FullTilePipeline = Is_d256 && !Is_even_MN && !Is_local && !Has_alibi && !Is_dropout;
#else
    constexpr bool UseD128FullTilePipeline = false;
    constexpr bool UseQK192V128FullTilePipeline = false;
    constexpr bool UseD256FullTilePipeline = false;
#endif

    if constexpr (UseD128FullTilePipeline || UseQK192V128FullTilePipeline || UseD256FullTilePipeline) {
        // The final KV tile stays in the masked tail below. All current V
        // tiles here are complete even when this CTA has a partial Q rowblock.
        auto run_full_tile = [&](int n_block_loop, auto full_next_k) __attribute__((always_inline)) {
            constexpr bool FullNextK = decltype(full_next_k)::value;
            const int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;
            vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
            if constexpr (UseD128FullTilePipeline || UseQK192V128FullTilePipeline) {
                fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum,
                            UseD128FullTilePipeline, true>(
                    s_reg, q_regs, k_lds, v_ptr, v_lds, warp_id, params.v_row_stride, kBlockN);

                union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32];
                union_vec32_fp8 p_reg[WARP_M / 16];
                fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDimV, kBlockN, WARP_M,
                                          WARP_N, WARP_K, Element_k, ElementAccum, true>(
                    s_reg, scores_max, scores_sum_partial, acc_o, softmax_scale_log2,
                    v_regs, v_lds);
                fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
                if constexpr (UseD128FullTilePipeline) {
                    fp8_pv_gemm_and_prefetch_k_d128<FullNextK, kHeadDim, kHeadDimV, kBlockN,
                                                  WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id,
                        params.k_row_stride, max_seq_kv_offset - kBlockN);
                } else {
                    fp8_pv_gemm_and_prefetch_k_qk192_v128<FullNextK, kBlockN, WARP_M, WARP_N,
                                                         Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id,
                        params.k_row_stride, max_seq_kv_offset - kBlockN);
                }
            } else {
                fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(
                    s_reg, q_regs, k_lds);
                fp8_prefetch_v_to_lds_b64<true, 256, Element_k>(
                    v_ptr, v_lds, warp_id, params.v_row_stride, kBlockN);
                ElementAccum scores_max_cur[WARP_M / 16];
                ElementAccum scores_sum_cur[WARP_M / 16];
                union_vec32_fp8 p_reg[WARP_M / 16];
                fp8_softmax_update_d256<Assume_valid_rows, kBlockN, WARP_M, WARP_N,
                                        ElementAccum>(
                    s_reg, scores_max, scores_max_cur, scores_sum_cur,
                    softmax_scale_log2);
                fp8_rescale_acc_d256<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, WARP_K, ElementAccum>(
                    acc_o, scores_max, scores_sum_partial, scores_max_cur, scores_sum_cur, softmax_scale_log2);
                fp8_wait_v_and_prefetch_k_d256<true, FullNextK, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
                fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
                fp8_pv_gemm_d256<Element_k, ElementAccum>(acc_o, p_reg, v_lds);
            }
            v_ptr += kBlockN * params.v_row_stride;
        };
        // Only the transition to a partial next K tile needs filtered MLS.
        const int full_prefetch_end = max(0, min(n_block_max - n_masking_steps,
                                               binfo.actual_seqlen_k / kBlockN - 1));
        int n = n_block_min;
        for (; n < full_prefetch_end; ++n) {
            run_full_tile(n, std::true_type{});
        }
        for (; n < n_block_max - n_masking_steps; ++n) {
            run_full_tile(n, std::false_type{});
        }
    } else {
        for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

            // 计算 kv 的边界
            int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

            // ======================================================== QK gemm ======================================================================
            vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
            fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(
                s_reg, q_regs, k_lds);

            // ========================================== load V ================================================
            if constexpr (Is_d256) {
                fp8_prefetch_v_to_lds_b64<Is_even_MN, 256, Element_k>(
                    v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
            } else if constexpr (kHeadDimV == 192 && kBlockN == 64) {
                fp8_prefetch_v_to_lds_d192<Is_even_MN, Element_k>(
                    v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
            } else {
                fp8_prefetch_v_to_lds<kHeadDimV, WARP_N, Element_k>(
                    v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
            }

            // ======================================================== s_reg ======================================================================
            // fp8_debug_s_reg<FP8_DEBUG, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, ElementAccum>(
            //     s_reg_ptr, s_reg, bidb, bidh, h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

            // ======================================================== Softmax ======================================================================
            if constexpr (Is_d256) {
                ElementAccum scores_max_cur[WARP_M / 16];
                ElementAccum scores_sum_cur[WARP_M / 16];
                union_vec32_fp8 p_reg[WARP_M / 16];
                fp8_softmax_update_d256<Assume_valid_rows, kBlockN, WARP_M, WARP_N,
                                        ElementAccum>(
                    s_reg, scores_max, scores_max_cur, scores_sum_cur,
                    softmax_scale_log2);
                fp8_rescale_acc_d256<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, WARP_K, ElementAccum>(
                    acc_o, scores_max, scores_sum_partial, scores_max_cur, scores_sum_cur, softmax_scale_log2);

                fp8_wait_v_and_prefetch_k_d256<true, Is_even_MN, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);

                fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
                fp8_pv_gemm_d256<Element_k, ElementAccum>(acc_o, p_reg, v_lds);
            } else {
                union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32];
                union_vec32_fp8 p_reg[WARP_M / 16];
                fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDimV, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum, false>(
                    s_reg, scores_max, scores_sum_partial, acc_o, softmax_scale_log2,
                    v_regs, v_lds);
                fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);

                // ======================================================== p_reg ======================================================================
                // fp8_debug_p_reg<1, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, Element_k>(
                //     p_reg_ptr, p_reg, bidb, bidh, params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

                fp8_pv_gemm_and_prefetch_k<true/*PrefetchK*/, kHeadDim, kHeadDimV, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                    acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id,
                    params.k_row_stride, max_seq_kv_offset - kBlockN);
            }

            // 计算 k, v 的偏移
            v_ptr += kBlockN * params.v_row_stride;
        }
    }

    // ========================================== Rest ===============================================
    // 剩下的需要做 causal mask
    int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        if constexpr (Is_causal && (Is_d256 || Needs_extra_mask)) {
            if (n_block_loop >= n_block_max) {
                continue;
            }
        }
        // 计算 kv 的边界
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // ======================================================== QK gemm ======================================================================
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, q_regs, k_lds);

        // ========================================== load V ================================================
        if constexpr (Is_d256) {
            fp8_prefetch_v_to_lds_b64<Is_even_MN, 256, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else if constexpr (kHeadDimV == 192 && kBlockN == 64) {
            fp8_prefetch_v_to_lds_d192<Is_even_MN, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else {
            fp8_prefetch_v_to_lds<kHeadDimV, WARP_N, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        }

        // ======================================================== causal mask ==================================================================
        if constexpr (Is_causal) {
            fp8_apply_causal_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN, lane_id);
        }

        // ======================================================== s_reg ======================================================================
        // fp8_debug_s_reg<FP8_DEBUG, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, ElementAccum>(
        //     s_reg_ptr, s_reg, bidb, bidh, h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

        // ======================================================== mask ==================================================================
        // 对齐 fp16 fwd：非 causal 的 rest loop 要屏蔽最后一个 partial KV tile 的越界列。
        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) {
                fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
            }
        }

        // ======================================================== Softmax ======================================================================
        if constexpr (Is_d256) {
            ElementAccum scores_max_cur[WARP_M / 16];
            ElementAccum scores_sum_cur[WARP_M / 16];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_update_d256<Assume_valid_rows, kBlockN, WARP_M, WARP_N,
                                    ElementAccum>(
                s_reg, scores_max, scores_max_cur, scores_sum_cur,
                softmax_scale_log2);
            fp8_rescale_acc_d256<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, WARP_K, ElementAccum>(
                acc_o, scores_max, scores_sum_partial, scores_max_cur, scores_sum_cur, softmax_scale_log2);

            if constexpr (n_masking_steps > 1) {
                if (n_block_loop + 1 < n_block_max) {
                    fp8_wait_v_and_prefetch_k_d256<true, Is_even_MN, Element_k>(
                        k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
                } else {
                    fp8_wait_v_and_prefetch_k_d256<false, Is_even_MN, Element_k>(
                        k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
                }
            } else {
                fp8_wait_v_and_prefetch_k_d256<false, Is_even_MN, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
            }

            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
            fp8_pv_gemm_d256<Element_k, ElementAccum>(acc_o, p_reg, v_lds);
        } else {
            union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDimV, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum>(
                s_reg, scores_max, scores_sum_partial, acc_o, softmax_scale_log2,
                v_regs, v_lds);
            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);

            // ======================================================== p_reg ======================================================================
            // fp8_debug_p_reg<0, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, Element_k>(
            //     p_reg_ptr, p_reg, bidb, bidh, params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

            if constexpr (n_masking_steps > 1) {
                if (n_block_loop + 1 < n_block_max) {
                    fp8_pv_gemm_and_prefetch_k<true, kHeadDim, kHeadDimV, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
                } else {
                    fp8_pv_gemm_and_prefetch_k<false, kHeadDim, kHeadDimV, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
                }
            } else {
                fp8_pv_gemm_and_prefetch_k<false, kHeadDim, kHeadDimV, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                    acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);
            }
        }

        // 计算 k, v 的偏移
        if constexpr (!Is_d256) {
            constexpr bool PrefetchK = n_masking_steps > 1;
            if (not PrefetchK) {
                k_ptr += kBlockN * params.k_row_stride;
            }
        }
        v_ptr += kBlockN * params.v_row_stride;
    }

    // ========================================== rescale by scores_sum ==========================================
    // 根据 scores_sum 对 acc_o 做缩放
#if defined(FP8_NO_SCALES)
    constexpr ElementAccum v_descale = ElementAccum(1.0f);
#endif
    fp8_finalize_rowsum<WARP_M, ElementAccum>(scores_sum_partial, scores_sum);
    ElementAccum lse[WARP_M / 16];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fp8_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fp8_attention_sink_apply<kHeadDimV, WARP_M, WARP_N, ElementAccum>(
            acc_o, scores_max, scores_sum, softmax_scale, sink_value);
    }
    if constexpr (Return_softmax) {
        fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, true/*StoreLSE*/, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);
    } else {
        fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, false/*StoreLSE*/, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);
    }

    // ========================================== lse storation ==========================================
    if constexpr (Return_softmax) {
        fp8_epilogue_store_lse<WARP_M, ElementAccum>(
            softmax_lse_ptr, lse, row_offset_lse_base, binfo.actual_seqlen_q, m_block * kBlockM + warp_id * WARP_M, lane_id);
    }

    // ========================================== Storation =============================================
    fp8_epilogue_store_output<Is_even_MN, kBlockM, kHeadDimV, WARP_M, WARP_N, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, params.o_row_stride, binfo.actual_seqlen_q);

}

#endif

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, bool D192MFirst, typename Params>
inline __device__ void compute_fp8_attn_gfx938(const Params &params) {

#if defined(__gfx938__) || defined(__gfx946__)
    constexpr bool Do_lpt = Is_causal
                            && (Is_GQA
                                || (Is_Varlen
                                    && Kernel_traits::kHeadDim == 256
                                    && Kernel_traits::kHeadDimV == 256));
    constexpr bool Reverse_single_rowblock = Is_causal
                                             && !Is_GQA
                                             && Kernel_traits::kHeadDim == 192
                                             && Kernel_traits::kHeadDimV == 192;
    static_assert(!D192MFirst || Reverse_single_rowblock);
    constexpr bool Use_head_first_grid = Do_lpt || (Reverse_single_rowblock && !D192MFirst);

    const int bidh = Use_head_first_grid ? blockIdx.x : blockIdx.y;
    const int bidb = Use_head_first_grid ? blockIdx.y : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = Use_head_first_grid ? gridDim.z - 1 - blockIdx.z
                                      : (Reverse_single_rowblock ? gridDim.x - 1 - blockIdx.x : blockIdx.x);
    flash::compute_fp8_attn_mha_1rowblock_gfx938<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);

    if constexpr (Is_causal and !Is_GQA && Kernel_traits::kHeadDim <= 128 /*MHA causal mask*/) {
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::compute_fp8_attn_mha_1rowblock_gfx938<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, gridDim.x * 2 - 1 - m_block, warp_id);
    }
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// FP8 Prefix Prefill (paged KV cache + varlen) for GFX938
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__gfx938__) || defined(__gfx946__)
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, bool Assume_no_empty_causal_rows, typename Params>
inline __device__ void compute_fp8_attn_prefix_prefill_1rowblock_gfx938(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int WARP_K    = 32;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr bool Is_d256  = kHeadDim == 256 && kHeadDimV == 256 && kBlockM == 128
                              && kBlockN == 64 && WARP_M == 32 && WARP_N == 32;
    constexpr bool UseLaneDeferredRowSum = Is_d256
        || (kHeadDim == 128 && kHeadDimV == 128) || (kHeadDim == 192 && kHeadDimV == 192);
    using ScoresSumState = std::conditional_t<UseLaneDeferredRowSum,
        fp8_deferred_lane_sum<ElementAccum>, vec2_Accum<ElementAccum>>;
    constexpr bool Is_d128_b64 = kHeadDim == 128 && kHeadDimV == 128 && kBlockM == 128
                                  && kBlockN == 64 && WARP_M == 32 && WARP_N == 32;
    constexpr bool PageMatchesTile = kBlockN == 128;
    constexpr bool UseD192FragmentPipeline = kHeadDim == 192 && kHeadDimV == 192
        && kBlockM == 128 && kBlockN == 128 && WARP_M == 32 && WARP_N == 32;

    // Q/K and V dimensions are independent for MLA-style attention.  Keep the
    // D192 K layout whenever Q/K are 192-wide, while selecting the V loader
    // from kHeadDimV so the 192/128 specialization only fetches 128 V columns.
    constexpr bool Is_qk_d192 = kHeadDim == 192;
    constexpr bool Is_v_d192  = kHeadDimV == 192;
    constexpr bool Guard_empty_rows =
        (Is_qk_d192 || Is_causal || Is_local)
        && !Assume_no_empty_causal_rows;

    // Varlen BlockInfo
    const flash::BlockInfo<true/*Varlen*/, false/*Is_kvcache*/> binfo(params, bidb);

    int max_seq_q_offset = binfo.actual_seqlen_q - m_block * kBlockM;
    if constexpr (Guard_empty_rows) {
        if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    } else {
        if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;
    }

    // 定义 lds
    extern __shared__ int8_t lds[];
    int8_t* q_lds = lds + 0;
    int8_t* k_lds = lds + 0;
    int8_t* v_lds = Is_d256 ? lds + kBlockN * kHeadDimV * sizeof(Element_k) : lds;

    // ========================================== 计算 offset (varlen + paged) ===========================================
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;

    int n_block_min = 0;
    if constexpr (Is_local) {
        n_block_min = max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    }
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        const int window_size_right = Is_local ? params.window_size_right : 0;
        n_block_max = min(n_block_max, ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {
        if constexpr (Guard_empty_rows) {
            // No visible keys: initialize every valid query row without reading a page.
            Element* out = reinterpret_cast<Element*>(params.o_ptr);
            const int64_t out_base = Layout == 1
                ? int64_t(binfo.sum_s_q) * params.o_head_stride * params.h + int64_t(bidh) * params.o_head_stride
                : int64_t(binfo.sum_s_q) * params.o_row_stride + int64_t(bidh) * params.o_head_stride;
            for (int i = threadIdx.x; i < kBlockM * kHeadDimV; i += blockDim.x) {
                const int row = m_block * kBlockM + i / kHeadDimV;
                if (row < binfo.actual_seqlen_q) {
                    out[out_base + int64_t(row) * params.o_row_stride + i % kHeadDimV] = Element(0.0f);
                }
            }
            if constexpr (Return_softmax) {
                ElementAccum* lse_out = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr);
                const int64_t lse_base = int64_t(bidh) * params.total_q + binfo.sum_s_q;
                const ElementAccum empty_lse = params.s_aux_ptr != nullptr
                    ? fp8_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh) : INFINITY;
                for (int i = threadIdx.x; i < kBlockM; i += blockDim.x) {
                    const int row = m_block * kBlockM + i;
                    if (row < binfo.actual_seqlen_q) lse_out[lse_base + row] = empty_lse;
                }
            }
        }
        return;
    }

    // A B128 kernel is only selected for a 128-token page, so its tile and
    // page indices are identical and there is no in-page row offset.
    const int first_block_table_idx = PageMatchesTile
        ? n_block_min : n_block_min * kBlockN / params.page_block_size;
    const int first_block_table_offset = PageMatchesTile
        ? 0 : n_block_min * kBlockN - first_block_table_idx * params.page_block_size;
    const int first_page = block_table[first_block_table_idx];
    int current_page = first_page;

    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;

    if constexpr (Layout == 1) { /*bshd layout*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(first_page) * int64_t(params.k_batch_stride) + first_block_table_offset * int64_t(params.k_row_stride) + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(first_page) * int64_t(params.v_batch_stride) + first_block_table_offset * int64_t(params.v_row_stride) + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * params.o_row_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    } else { /*bhsd layout*/
        row_offset_q = binfo.sum_s_q * int64_t(params.q_row_stride) + bidh * params.q_head_stride + m_block * kBlockM * params.q_row_stride;
        row_offset_k = int64_t(first_page) * int64_t(params.k_batch_stride) + first_block_table_offset * int64_t(params.k_row_stride) + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(first_page) * int64_t(params.v_batch_stride) + first_block_table_offset * int64_t(params.v_row_stride) + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_row_stride) + bidh * params.o_head_stride + m_block * kBlockM * params.o_row_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    }

    // 使用原始指针 (FP8 prefetch函数内部会调用 prepare_for_matrix_load)
    Element_k* q_ptr = reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q;
    Element_k* k_ptr = reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k;
    Element_k* v_ptr = reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v;

#if defined(FP8_NO_SCALES)
    const ElementAccum softmax_scale = params.scale_softmax;
    const ElementAccum softmax_scale_log2 = params.scale_softmax_log2;
    constexpr ElementAccum v_descale = ElementAccum(1.0f);
#else
    // FP8 descale tensors are broadcast by taking the first scalar value.
    ElementAccum* q_descale_ptr   = reinterpret_cast<ElementAccum*>(params.q_descale_ptr);
    ElementAccum* k_descale_ptr   = reinterpret_cast<ElementAccum*>(params.k_descale_ptr);
    ElementAccum* v_descale_ptr   = reinterpret_cast<ElementAccum*>(params.v_descale_ptr);

    ElementAccum q_descale   = q_descale_ptr[0];
    ElementAccum k_descale   = k_descale_ptr[0];
    ElementAccum qk_descale = q_descale * k_descale;
    ElementAccum softmax_scale = params.scale_softmax * qk_descale;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2 * qk_descale;
    ElementAccum v_descale   = v_descale_ptr[0];
#endif

    ElementAccum* softmax_lse_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr);
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;

    // ======================================================== 读取 Q ======================================================================
    fp8_prefetch_q_to_lds<kHeadDim, WARP_M, Element_k>(q_ptr, q_lds, warp_id, params.q_row_stride, max_seq_q_offset);

    int lane_id = threadIdx.x & 63;

    // 准备寄存器
    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    ScoresSumState scores_sum_partial[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 32][WARP_M / 16][WARP_N / 16];
    fp8_attention_initialize<kHeadDimV, WARP_M, WARP_N, ElementAccum>(scores_max, scores_sum, acc_o);
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        if constexpr (UseLaneDeferredRowSum) {
            scores_sum_partial[m_idx].value = 0.0f;
        } else {
            scores_sum_partial[m_idx].data = 0.0;
        }
    }

    // 从 lds 读取 q 的数据
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64];
    load_q_from_lds_to_vgpr<kHeadDim, WARP_M, Element_k>(q_regs, q_lds, warp_id, lane_id);

    // ======================================================== Mainloop ======================================================================
    int n_masking_steps = 1;
    if constexpr (Is_causal) {
        const int causal_start_col = m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q;
        const int first_mask_block = max(n_block_min, causal_start_col / kBlockN);
        n_masking_steps = n_block_max - first_mask_block;
    } else if constexpr (Is_local) {
        n_masking_steps = min(n_block_max - n_block_min, ceil_div(kBlockM, kBlockN));
    }
    n_masking_steps = min(max(n_masking_steps, 1), n_block_max - n_block_min);
    constexpr bool Assume_valid_rows = !Is_local
        && (!Is_causal || Assume_no_empty_causal_rows);

    // ======================================================== Prefetch 第一块 K ======================================================================
    if (n_block_max > n_masking_steps) {
        const int first_max_seq_kv_offset = binfo.actual_seqlen_k - n_block_min * kBlockN;
        if constexpr (Is_d256) {
            fp8_prefetch_k_to_lds_d256<false/*FullKV*/, Element_k>(
                k_ptr, k_lds, warp_id, params.k_row_stride, first_max_seq_kv_offset);
        } else if constexpr (Is_qk_d192 && kBlockN == 64) {
            fp8_prefetch_k_to_lds_d192<false, Element_k>(
                k_ptr, k_lds, warp_id, params.k_row_stride, first_max_seq_kv_offset);
        } else if constexpr (Is_d128_b64) {
            fp8_prefetch_k_to_lds_d128<false/*FullKV*/, Element_k>(
                k_ptr, k_lds, warp_id, params.k_row_stride, first_max_seq_kv_offset);
        } else {
            fp8_prefetch_k_to_lds<kHeadDim, WARP_N, Element_k>(
                k_ptr, k_lds, warp_id, params.k_row_stride, first_max_seq_kv_offset);
        }
    }

    // ======================================================== 主循环：不需要 causal mask + Prefetch K ============================================================
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // QK gemm（K 数据已在上一轮 prefetch 到 LDS）
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum,
                    false, false, UseD192FragmentPipeline>(s_reg, q_regs, k_lds);

        // Prefetch V
        if constexpr (Is_d256) {
            fp8_prefetch_v_to_lds_b64<false/*FullKV*/, 256, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else if constexpr (Is_v_d192 && kBlockN == 64) {
            fp8_prefetch_v_to_lds_d192<false, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else if constexpr (Is_d128_b64) {
            fp8_prefetch_v_to_lds_b64<false/*FullKV*/, 128, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else {
            fp8_prefetch_v_to_lds<kHeadDimV, WARP_N, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        }

        if constexpr (Is_qk_d192 && Is_local) {
            fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
        }
        if constexpr (Is_local) {
            fp8_apply_local_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(
                s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k,
                m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN,
                params.window_size_left, params.window_size_right, lane_id);
        }

        if constexpr (Is_d256) {
            ElementAccum scores_max_cur[WARP_M / 16];
            ElementAccum scores_sum_cur[WARP_M / 16];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_update_d256<Assume_valid_rows, kBlockN, WARP_M, WARP_N,
                                    ElementAccum>(
                s_reg, scores_max, scores_max_cur, scores_sum_cur,
                softmax_scale_log2);
            fp8_rescale_acc_d256<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, WARP_K, ElementAccum>(
                acc_o, scores_max, scores_sum_partial, scores_max_cur, scores_sum_cur, softmax_scale_log2);

            const int next_n_block_loop = n_block_loop + 1;
            // For a 128-token page, advance by 64 rows before consulting the next page-table entry.
            const bool advance_within_page = page_block_size == 128 && (n_block_loop & 1) == 0;
            const int next_block_table_idx = page_block_size == 128
                ? next_n_block_loop >> 1 : next_n_block_loop;
            const int next_page = advance_within_page
                ? current_page : block_table[next_block_table_idx];
            const int64_t table_delta = int64_t(next_page - current_page);
            const int64_t offset_delta = page_block_size == 128
                ? (advance_within_page ? int64_t(kBlockN) : -int64_t(kBlockN))
                : int64_t(0);
            Element_k* k_ptr_next = k_ptr
                + table_delta * int64_t(params.k_batch_stride)
                + offset_delta * int64_t(params.k_row_stride);
            const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;

            fp8_wait_v_and_prefetch_k_d256<false/*PrefetchK*/, false/*FullNextK*/, Element_k>(
                k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset_next);
            fp8_prefetch_k_to_lds_d256<false/*FullKV*/, Element_k>(
                k_ptr_next, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset_next);

            k_ptr = k_ptr_next;
            v_ptr += table_delta * int64_t(params.v_batch_stride)
                + offset_delta * int64_t(params.v_row_stride);
            current_page = next_page;

            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
            fp8_pv_gemm_d256<Element_k, ElementAccum>(acc_o, p_reg, v_lds);
        } else {
            // Softmax + 读取 V 到寄存器
            union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDimV, kBlockN, WARP_M,
                                      WARP_N, WARP_K, Element_k, ElementAccum>(
                s_reg, scores_max, scores_sum_partial, acc_o, softmax_scale_log2,
                v_regs, v_lds);
            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);

            const int next_n_block_loop = n_block_loop + 1;
            const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;
            if constexpr (Is_d128_b64) {
                const int64_t table_delta = int64_t(
                    block_table[next_n_block_loop] - block_table[n_block_loop]);
                Element_k* k_ptr_next =
                    k_ptr + table_delta * int64_t(params.k_batch_stride);
                fp8_pv_gemm_and_prefetch_k_paged_d128<
                    false/*Is_even_MN*/, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds,
                        k_ptr_next, k_lds, warp_id,
                        params.k_row_stride, max_seq_kv_offset_next);

                k_ptr = k_ptr_next;
                v_ptr += table_delta * int64_t(params.v_batch_stride);
            } else if constexpr (PageMatchesTile) {
                const int64_t table_delta = int64_t(
                    block_table[next_n_block_loop] - block_table[n_block_loop]);
                Element_k* k_ptr_next =
                    k_ptr + table_delta * int64_t(params.k_batch_stride);
                fp8_pv_gemm_and_prefetch_k_paged<kHeadDim, kHeadDimV, kBlockN,
                    WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds,
                        warp_id, params.k_row_stride, max_seq_kv_offset_next);
                k_ptr = k_ptr_next;
                v_ptr += table_delta * int64_t(params.v_batch_stride);
            } else {
                const int block_table_idx_cur = n_block_loop * kBlockN / page_block_size;
                const int block_table_offset_cur = n_block_loop * kBlockN - block_table_idx_cur * page_block_size;
                const int block_table_idx_next = next_n_block_loop * kBlockN / page_block_size;
                const int block_table_offset_next = next_n_block_loop * kBlockN - block_table_idx_next * page_block_size;
                const int64_t table_delta = int64_t(block_table[block_table_idx_next] - block_table[block_table_idx_cur]);
                const int64_t offset_delta = int64_t(block_table_offset_next - block_table_offset_cur);
                Element_k* k_ptr_next = k_ptr
                    + table_delta * int64_t(params.k_batch_stride)
                    + offset_delta * int64_t(params.k_row_stride);
                fp8_pv_gemm_and_prefetch_k_paged<kHeadDim, kHeadDimV, kBlockN,
                    WARP_M, WARP_N, Element_k, ElementAccum>(
                        acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds,
                        warp_id, params.k_row_stride, max_seq_kv_offset_next);

                k_ptr = k_ptr_next;
                v_ptr += table_delta * int64_t(params.v_batch_stride)
                    + offset_delta * int64_t(params.v_row_stride);
            }
        }
    }

    // ======================================================== Masking 循环：需要 causal mask，不 Prefetch K ============================================================
    int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // 如果主循环没有 prefetch（n_block_max <= n_masking_steps），需要在这里 prefetch K
        if (masking_step == 0 && n_block_max <= n_masking_steps) {
            if constexpr (Is_d256) {
                fp8_prefetch_k_to_lds_d256<false/*FullKV*/, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            } else if constexpr (Is_qk_d192 && kBlockN == 64) {
                fp8_prefetch_k_to_lds_d192<false, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            } else if constexpr (Is_d128_b64) {
                fp8_prefetch_k_to_lds_d128<false/*FullKV*/, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            } else {
                fp8_prefetch_k_to_lds<kHeadDim, WARP_N, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            }
        }

        // QK gemm
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum,
                    false, false, UseD192FragmentPipeline>(s_reg, q_regs, k_lds);

        // Prefetch V
        if constexpr (Is_d256) {
            fp8_prefetch_v_to_lds_b64<false/*FullKV*/, 256, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else if constexpr (Is_v_d192 && kBlockN == 64) {
            fp8_prefetch_v_to_lds_d192<false, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else if constexpr (Is_d128_b64) {
            fp8_prefetch_v_to_lds_b64<false/*FullKV*/, 128, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        } else {
            fp8_prefetch_v_to_lds<kHeadDimV, WARP_N, Element_k>(
                v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);
        }

        // Mask
        // 对齐 fp16 fwd：非 causal 的 rest loop 要屏蔽最后一个 partial KV tile 的越界列。
        if constexpr (!Is_causal && !Is_local) {
            fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
        }

        // Causal mask
        if constexpr (Is_qk_d192 && Is_local) {
            fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
        }
        if constexpr (Is_local) {
            fp8_apply_local_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(
                s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k,
                m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN,
                params.window_size_left, params.window_size_right, lane_id);
        } else if constexpr (Is_causal) {
            fp8_apply_causal_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN, lane_id);
        }

        const int next_n_block_loop = n_block_loop + 1;
        if constexpr (Is_d256) {
            ElementAccum scores_max_cur[WARP_M / 16];
            ElementAccum scores_sum_cur[WARP_M / 16];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_update_d256<Assume_valid_rows, kBlockN, WARP_M, WARP_N,
                                    ElementAccum>(
                s_reg, scores_max, scores_max_cur, scores_sum_cur,
                softmax_scale_log2);
            fp8_rescale_acc_d256<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, WARP_K, ElementAccum>(
                acc_o, scores_max, scores_sum_partial, scores_max_cur, scores_sum_cur, softmax_scale_log2);

            if (next_n_block_loop < n_block_max) {
                const bool advance_within_page = page_block_size == 128 && (n_block_loop & 1) == 0;
                const int next_block_table_idx = page_block_size == 128
                    ? next_n_block_loop >> 1 : next_n_block_loop;
                const int next_page = advance_within_page
                    ? current_page : block_table[next_block_table_idx];
                const int64_t table_delta = int64_t(next_page - current_page);
                const int64_t offset_delta = page_block_size == 128
                    ? (advance_within_page ? int64_t(kBlockN) : -int64_t(kBlockN))
                    : int64_t(0);
                Element_k* k_ptr_next = k_ptr
                    + table_delta * int64_t(params.k_batch_stride)
                    + offset_delta * int64_t(params.k_row_stride);
                const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;
                fp8_wait_v_and_prefetch_k_d256<false/*PrefetchK*/, false/*FullNextK*/, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset_next);
                fp8_prefetch_k_to_lds_d256<false/*FullKV*/, Element_k>(
                    k_ptr_next, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset_next);
                k_ptr = k_ptr_next;
                v_ptr += table_delta * int64_t(params.v_batch_stride)
                    + offset_delta * int64_t(params.v_row_stride);
                current_page = next_page;
            } else {
                fp8_wait_v_and_prefetch_k_d256<false/*PrefetchK*/, false/*FullNextK*/, Element_k>(
                    k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            }

            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);
            fp8_pv_gemm_d256<Element_k, ElementAccum>(acc_o, p_reg, v_lds);
        } else {
            // Softmax + 读取 V 到寄存器
            union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDimV / 32];
            union_vec32_fp8 p_reg[WARP_M / 16];
            fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDimV, kBlockN, WARP_M,
                                      WARP_N, WARP_K, Element_k, ElementAccum>(
                s_reg, scores_max, scores_sum_partial, acc_o, softmax_scale_log2,
                v_regs, v_lds);
            fp8_cvt_f32_to_fp8<kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, p_reg);

            if (next_n_block_loop < n_block_max) {
                const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;
                if constexpr (Is_d128_b64) {
                    const int64_t table_delta = int64_t(
                        block_table[next_n_block_loop] - block_table[n_block_loop]);
                    Element_k* k_ptr_next =
                        k_ptr + table_delta * int64_t(params.k_batch_stride);
                    fp8_pv_gemm_and_prefetch_k_paged_d128<
                        false/*Is_even_MN*/, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                            acc_o, p_reg, v_regs, v_lds,
                            k_ptr_next, k_lds, warp_id,
                            params.k_row_stride, max_seq_kv_offset_next);
                    k_ptr = k_ptr_next;
                    v_ptr += table_delta * int64_t(params.v_batch_stride);
                } else if constexpr (PageMatchesTile) {
                    const int64_t table_delta = int64_t(
                        block_table[next_n_block_loop] - block_table[n_block_loop]);
                    Element_k* k_ptr_next =
                        k_ptr + table_delta * int64_t(params.k_batch_stride);
                    fp8_pv_gemm_and_prefetch_k_paged<kHeadDim, kHeadDimV, kBlockN,
                        WARP_M, WARP_N, Element_k, ElementAccum>(
                            acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds,
                            warp_id, params.k_row_stride, max_seq_kv_offset_next);
                    k_ptr = k_ptr_next;
                    v_ptr += table_delta * int64_t(params.v_batch_stride);
                } else {
                    const int block_table_idx_cur = n_block_loop * kBlockN / page_block_size;
                    const int block_table_offset_cur = n_block_loop * kBlockN - block_table_idx_cur * page_block_size;
                    const int block_table_idx_next = next_n_block_loop * kBlockN / page_block_size;
                    const int block_table_offset_next = next_n_block_loop * kBlockN - block_table_idx_next * page_block_size;
                    const int64_t table_delta = int64_t(block_table[block_table_idx_next] - block_table[block_table_idx_cur]);
                    const int64_t offset_delta = int64_t(block_table_offset_next - block_table_offset_cur);
                    Element_k* k_ptr_next = k_ptr
                        + table_delta * int64_t(params.k_batch_stride)
                        + offset_delta * int64_t(params.k_row_stride);
                    fp8_pv_gemm_and_prefetch_k_paged<kHeadDim, kHeadDimV, kBlockN,
                        WARP_M, WARP_N, Element_k, ElementAccum>(
                            acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds,
                            warp_id, params.k_row_stride, max_seq_kv_offset_next);
                    k_ptr = k_ptr_next;
                    v_ptr += table_delta * int64_t(params.v_batch_stride)
                        + offset_delta * int64_t(params.v_row_stride);
                }
            } else {
                fp8_pv_gemm_and_prefetch_k<false/*PrefetchK*/, kHeadDim, kHeadDimV, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(
                    acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
            }
        }
    }

    // ========================================== rescale by scores_sum ==========================================
    fp8_finalize_rowsum<WARP_M, ElementAccum>(scores_sum_partial, scores_sum);
    ElementAccum lse[WARP_M / 16];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fp8_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fp8_attention_sink_apply<kHeadDimV, WARP_M, WARP_N, ElementAccum>(
            acc_o, scores_max, scores_sum, softmax_scale, sink_value);
    }
    if constexpr (Return_softmax) {
        fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, true/*StoreLSE*/, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);
    } else {
        fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDimV, WARP_M, WARP_N, false/*StoreLSE*/, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);
    }

    // ========================================== lse storation (varlen) ==========================================
    if constexpr (Return_softmax) {
        fp8_epilogue_store_lse<WARP_M, ElementAccum>(
            softmax_lse_ptr, lse, row_offset_lse, binfo.actual_seqlen_q, m_block * kBlockM + warp_id * WARP_M, lane_id);
    }

    // ========================================== Storation =============================================
    fp8_epilogue_store_output<false/*FullQ*/, kBlockM, kHeadDimV, WARP_M, WARP_N, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, params.o_row_stride, binfo.actual_seqlen_q);
}

#endif

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, bool Assume_no_empty_causal_rows>
__global__ void __launch_bounds__(256, 1) flash_fp8_fwd_prefix_prefill_kernel_gfx938(
    Flash_fwd_params params) {
#if defined(__gfx938__) || defined(__gfx946__)
    // LPT 调度：改变 blockIdx 到 m_block/bidh/bidb 的映射
    // causal 模式：blockIdx.x = bidh, blockIdx.y = bidb, blockIdx.z 倒序 = m_block
    // 非 causal 模式：blockIdx.x = m_block, blockIdx.y = bidh, blockIdx.z = bidb
    constexpr bool Do_lpt = Is_causal;
    const int bidh = Do_lpt ? blockIdx.x : blockIdx.y;
    const int bidb = Do_lpt ? blockIdx.y : blockIdx.z;
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z : blockIdx.x;
    flash::compute_fp8_attn_prefix_prefill_1rowblock_gfx938<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_K, Return_softmax, Has_alibi, Layout, Assume_no_empty_causal_rows, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                            GFX92A kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__gfx92a__)
#include "fwd/gfx92a/fp8_qk_gemm_prefetch_v_mls_ds_gfx92a.h"
#include "fwd/gfx92a/fp8_pv_gemm_prefetch_k_mls_ds_gfx92a.h"
#include "fwd/gfx92a/fp8_softmax_gfx92a.h"
#include "fwd/gfx92a/fp8_epilogue_gfx92a.h"

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_fp8_attn_mha_1rowblock_gfx92a(const Params &params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int WARP_K    = 32;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;

    // 获取当前 TG 处理的任务大小
    const flash::BlockInfo</*Varlen=*/Is_Varlen> binfo(params, bidb);

    // 判断任务边界
    int max_seq_q_offset = binfo.actual_seqlen_q - m_block * kBlockM;
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0/* || bidh >= h*/) return;

    // 获取 wave id
    // int __warp_id = threadIdx.x >> 6;
    // int warp_id = __builtin_amdgcn_readfirstlane(__warp_id);

    // 定义 lds, 128x128 个 fp8, 16384 bytes
    // __shared__ int8_t lds[16384 + 4096 + 16384 + 4096];
    extern __shared__ int8_t lds[];
    int8_t* q_lds = lds + 0;
    int8_t* k_lds = lds + 0;
    int8_t* v_lds = lds + 0;

    // ========================================== 计算 offset ===========================================
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int64_t row_offset_lse_base;
    if constexpr (Is_Varlen) {
        if constexpr (Layout == 1) { /* bshd: q/o are [total_q, h, d] */
            row_offset_q = (int64_t(binfo.sum_s_q) + m_block * kBlockM) * int64_t(params.q_row_stride) + params.q_head_stride * bidh;
            row_offset_k = int64_t(binfo.sum_s_k) * int64_t(params.k_row_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
            row_offset_v = int64_t(binfo.sum_s_k) * int64_t(params.v_row_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
            row_offset_o = int64_t(binfo.sum_s_q) * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * int64_t(params.o_row_stride);
            row_offset_lse_base = bidh * int64_t(params.total_q) + binfo.sum_s_q;
        } else { /* bhsd */
            row_offset_q = int64_t(binfo.sum_s_q) * int64_t(params.q_row_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(params.q_row_stride);
            row_offset_k = int64_t(binfo.sum_s_k) * int64_t(params.k_row_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
            row_offset_v = int64_t(binfo.sum_s_k) * int64_t(params.v_row_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
            row_offset_o = int64_t(binfo.sum_s_q) * int64_t(params.o_row_stride) + bidh * params.o_head_stride + m_block * kBlockM * int64_t(params.o_row_stride);
            row_offset_lse_base = bidh * int64_t(params.total_q) + binfo.sum_s_q;
        }
    } else {
        row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(params.q_row_stride);
        row_offset_k = bidb * int64_t(params.k_batch_stride) + int(bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = bidb * int64_t(params.v_batch_stride) + int(bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + m_block * kBlockM * int64_t(params.o_row_stride);
        row_offset_lse_base = (bidb * params.h + bidh) * int64_t(binfo.actual_seqlen_q);
    }

    Element_k* q_ptr   = reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q;
    Element_k* k_ptr   = reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k;
    Element_k* v_ptr   = reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v;

#if defined(FP8_NO_SCALES)
    ElementAccum softmax_scale = params.scale_softmax;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2;
    constexpr ElementAccum v_descale = ElementAccum(1.0f);
#else
    int row_offset_q_descale = bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride;
    int row_offset_k_descale = bidb * params.k_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.k_descale_head_stride;
    int row_offset_v_descale = bidb * params.v_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.v_descale_head_stride;

    ElementAccum* q_descale_ptr   = reinterpret_cast<ElementAccum*>(params.q_descale_ptr);
    ElementAccum* k_descale_ptr   = reinterpret_cast<ElementAccum*>(params.k_descale_ptr);
    ElementAccum* v_descale_ptr   = reinterpret_cast<ElementAccum*>(params.v_descale_ptr);

    ElementAccum q_descale   = q_descale_ptr[row_offset_q_descale];
    ElementAccum k_descale   = k_descale_ptr[row_offset_k_descale];
    ElementAccum qk_descale = q_descale * k_descale;
    ElementAccum softmax_scale = params.scale_softmax * qk_descale;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2 * qk_descale;
    ElementAccum v_descale   = v_descale_ptr[row_offset_v_descale];
#endif

    // acc_o_ptr = reinterpret_cast<ElementAccum*>(acc_o_ptr) + row_offset_o;
    ElementAccum* softmax_lse_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr);
    Element_k* p_reg_ptr = reinterpret_cast<Element_k *>(params.p_ptr);
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;

    // ======================================================== 读取 Q ======================================================================
    fp8_prefetch_q_to_lds<Is_even_MN, kHeadDim, WARP_M, Element_k>(q_ptr, q_lds, warp_id, params.q_row_stride, max_seq_q_offset);

    // 计算解决 bank 冲突必须的一些变量
    int tx = threadIdx.x;
    int lane_id = tx & 63;

    // 准备存储最大值, 求和, acc_o 寄存器 等
    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16];
    fp8_attention_initialize<kHeadDim, WARP_M, WARP_N, ElementAccum>(scores_max, scores_sum, acc_o);

    // 从 lds 读取 q 的数据, 不需要同步
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64];
    load_q_from_lds_to_vgpr<kHeadDim, WARP_M, Element_k>(q_regs, q_lds, warp_id, lane_id);

    // ======================================================== Prefetch K ======================================================================
    fp8_prefetch_k_to_lds<Is_even_MN, kHeadDim, WARP_N, Element_k>(k_ptr, k_lds, warp_id, params.k_row_stride, binfo.actual_seqlen_k);

    // ======================================================== Mainloop ======================================================================
    // 计算当前 block 计算任务的边界，带 causal mask 的场景可以少计算一些
    int n_block_min = 0;
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal) {
        n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + 0/*params.window_size_right*/, kBlockN));
    }
    // causal 且 seqlen_k - seqlen_q 不是 kBlockN 的整数倍时, 对角线跨 ceil_div(kBlockM, kBlockN) + 1 个 N tile, 需要多 mask 一步 (同 gfx938)
    constexpr int n_masking_steps = (!Is_causal/* && !Is_local*/) ? 1
        : ceil_div(kBlockM, kBlockN) + (Needs_extra_mask ? 1 : 0);
    constexpr bool Assume_valid_rows = !Is_local && (!Is_causal || !Is_Varlen);

    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        // 计算 kv 的边界
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // ======================================================== QK gemm ======================================================================
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, q_regs, k_lds);

        // ========================================== load V ================================================
        fp8_prefetch_v_to_lds<Is_even_MN, kBlockN, kHeadDim, WARP_N, Element_k>(v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);

        // ======================================================== s_reg ======================================================================
        // fp8_debug_s_reg<FP8_DEBUG, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, ElementAccum>(
        //     s_reg_ptr, s_reg, bidb, bidh, h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

        // ======================================================== Softmax ======================================================================
        union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDim, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum>(s_reg, p_reg, scores_max, scores_sum, acc_o, softmax_scale_log2, v_regs, v_lds);

        // ======================================================== p_reg ======================================================================
        // fp8_debug_p_reg<1, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, Element_k>(
        //     p_reg_ptr, p_reg, bidb, bidh, params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

        // ========================================== PV mmac ================================================
        fp8_pv_gemm_and_prefetch_k<true/*PrefetchK*/, Is_even_MN, kHeadDim, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);

        // 计算 k, v 的偏移
        v_ptr += kBlockN * params.v_row_stride;
    }

    // ========================================== Rest ===============================================
    // 剩下的需要做 causal mask
    int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        if constexpr (Is_causal && Needs_extra_mask) {
            // n_block_max < n_masking_steps 时多出来的那一步不要越过 n_block_max
            if (n_block_loop >= n_block_max) {
                continue;
            }
        }
        // 计算 kv 的边界
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // ======================================================== QK gemm ======================================================================
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, q_regs, k_lds);

        // ========================================== load V ================================================
        fp8_prefetch_v_to_lds<Is_even_MN, kBlockN, kHeadDim, WARP_N, Element_k>(v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);

        // ======================================================== causal mask ==================================================================
        if constexpr (Is_causal) {
            fp8_apply_causal_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN, lane_id);
        }

        // ======================================================== s_reg ======================================================================
        // fp8_debug_s_reg<FP8_DEBUG, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, ElementAccum>(
        //     s_reg_ptr, s_reg, bidb, bidh, h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

        // ======================================================== mask ==================================================================
        // 对齐 fp16 fwd：非 causal 的 rest loop 要屏蔽最后一个 partial KV tile 的越界列。
        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) {
                fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
            }
        }

        // ======================================================== Softmax ======================================================================
        union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDim, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum>(s_reg, p_reg, scores_max, scores_sum, acc_o, softmax_scale_log2, v_regs, v_lds);

        // ======================================================== p_reg ======================================================================
        // fp8_debug_p_reg<0, Is_even_MN, kBlockM, kBlockN, WARP_M, WARP_N, Element_k>(
        //     p_reg_ptr, p_reg, bidb, bidh, params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k, max_seq_q_offset, max_seq_kv_offset, m_block, n_block_loop, warp_id, lane_id);

        // ========================================== PV mmac ================================================
        constexpr bool PrefetchK = n_masking_steps > 1;
        fp8_pv_gemm_and_prefetch_k<PrefetchK, Is_even_MN, kHeadDim, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset - kBlockN);

        // 计算 k, v 的偏移
        if (not PrefetchK) {
            k_ptr += kBlockN * params.k_row_stride;
        }
        v_ptr += kBlockN * params.v_row_stride;
    }

    // ========================================== rescale by scores_sum ==========================================
    // 根据 scores_sum 对 acc_o 做缩放
    ElementAccum lse[WARP_M / 16];
    fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDim, WARP_M, WARP_N, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);

    // ========================================== lse storation ==========================================
    if constexpr (Return_softmax) {
        fp8_epilogue_store_lse<Is_even_MN, WARP_M, ElementAccum>(
            softmax_lse_ptr, scores_max, scores_sum, lse, row_offset_lse_base, binfo.actual_seqlen_q, m_block * kBlockM + warp_id * WARP_M, lane_id);
    }

    // ========================================== Storation =============================================
    fp8_epilogue_store_output<Is_even_MN, kBlockM, kHeadDim, WARP_M, WARP_N, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, params.o_row_stride, binfo.actual_seqlen_q);

}


#endif

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, typename Params>
inline __device__ void compute_fp8_attn_gfx92a(const Params &params) {
#if defined(__gfx92a__)
    constexpr bool Do_lpt = Is_causal and Is_GQA;

    const int bidh = Do_lpt ? blockIdx.x : blockIdx.y;
    const int bidb = Do_lpt ? blockIdx.y : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z : blockIdx.x;
    flash::compute_fp8_attn_mha_1rowblock_gfx92a<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);

    if constexpr (Is_causal and !Is_GQA /*MHA causal mask*/) {
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::compute_fp8_attn_mha_1rowblock_gfx92a<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, gridDim.x * 2 - 1 - m_block, warp_id);
    }
#endif
}

#if defined(__gfx92a__)
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_fp8_attn_prefix_prefill_1rowblock_gfx92a(const Params &params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int WARP_K    = 32;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;

    // Varlen BlockInfo
    const flash::BlockInfo<true/*Varlen*/, false/*Is_kvcache*/> binfo(params, bidb);

    int max_seq_q_offset = binfo.actual_seqlen_q - m_block * kBlockM;
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // 定义 lds
    extern __shared__ int8_t lds[];
    int8_t* q_lds = lds + 0;
    int8_t* k_lds = lds + 0;
    int8_t* v_lds = lds + 0;

    // ========================================== 计算 offset (varlen + paged) ===========================================
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = 0;
    const int block_table_offset = 0;

    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;

    if constexpr (Layout == 1) { /*bshd layout*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * params.o_row_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    } else { /*bhsd layout*/
        row_offset_q = binfo.sum_s_q * int64_t(params.q_row_stride) + bidh * params.q_head_stride + m_block * kBlockM * params.q_row_stride;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_row_stride) + bidh * params.o_head_stride + m_block * kBlockM * params.o_row_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    }

    // FP8 descale: per-head descale
    // 使用原始指针 (FP8 prefetch函数内部会调用 prepare_for_matrix_load)
    Element_k* q_ptr = reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q;
    Element_k* k_ptr = reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k;
    Element_k* v_ptr = reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v;

#if defined(FP8_NO_SCALES)
    ElementAccum softmax_scale = params.scale_softmax;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2;
    constexpr ElementAccum v_descale = ElementAccum(1.0f);
#else
    int row_offset_q_descale = bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride;
    int row_offset_k_descale = bidb * params.k_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.k_descale_head_stride;
    int row_offset_v_descale = bidb * params.v_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.v_descale_head_stride;

    ElementAccum* q_descale_ptr   = reinterpret_cast<ElementAccum*>(params.q_descale_ptr);
    ElementAccum* k_descale_ptr   = reinterpret_cast<ElementAccum*>(params.k_descale_ptr);
    ElementAccum* v_descale_ptr   = reinterpret_cast<ElementAccum*>(params.v_descale_ptr);

    ElementAccum q_descale   = q_descale_ptr[row_offset_q_descale];
    ElementAccum k_descale   = k_descale_ptr[row_offset_k_descale];
    ElementAccum qk_descale = q_descale * k_descale;
    ElementAccum softmax_scale = params.scale_softmax * qk_descale;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2 * qk_descale;
    ElementAccum v_descale   = v_descale_ptr[row_offset_v_descale];
#endif

    ElementAccum* softmax_lse_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr);
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;

    // ======================================================== 读取 Q ======================================================================
    fp8_prefetch_q_to_lds<false/*Is_even_MN*/, kHeadDim, WARP_M, Element_k>(q_ptr, q_lds, warp_id, params.q_row_stride, max_seq_q_offset);

    int lane_id = threadIdx.x & 63;

    // 准备寄存器
    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16];
    fp8_attention_initialize<kHeadDim, WARP_M, WARP_N, ElementAccum>(scores_max, scores_sum, acc_o);

    // 从 lds 读取 q 的数据
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64];
    load_q_from_lds_to_vgpr<kHeadDim, WARP_M, Element_k>(q_regs, q_lds, warp_id, lane_id);

    // ======================================================== Mainloop ======================================================================
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal) {
        n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + 0, kBlockN));
    }
    constexpr int n_masking_steps = (!Is_causal) ? 1 : ceil_div(kBlockM, kBlockN);
    constexpr bool Assume_valid_rows = !Is_local;

    // ======================================================== Prefetch 第一块 K ======================================================================
    if (n_block_max > n_masking_steps) {
        fp8_prefetch_k_to_lds<false/*Is_even_MN*/, kHeadDim, WARP_N, Element_k>(k_ptr, k_lds, warp_id, params.k_row_stride, binfo.actual_seqlen_k);
    }

    // ======================================================== 主循环：不需要 causal mask + Prefetch K ============================================================
    for (int n_block_loop = 0; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // QK gemm（K 数据已在上一轮 prefetch 到 LDS）
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, q_regs, k_lds);

        // Prefetch V
        fp8_prefetch_v_to_lds<false/*Is_even_MN*/, kBlockN, kHeadDim, WARP_N, Element_k>(v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);

        // Softmax + 读取 V 到寄存器
        union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDim, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum>(s_reg, p_reg, scores_max, scores_sum, acc_o, softmax_scale_log2, v_regs, v_lds);

        // PV MMAC + Prefetch 下一块 K（paged KV）
        const int next_n_block_loop = n_block_loop + 1;
        const int64_t table_delta = int64_t(block_table[min(n_block_max - 1, next_n_block_loop)] - block_table[n_block_loop]);
        Element_k* k_ptr_next = k_ptr + table_delta * int64_t(params.k_batch_stride);
        const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;
        fp8_pv_gemm_and_prefetch_k_paged<false/*Is_even_MN*/, kHeadDim, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset_next);

        // 更新 K/V 指针
        k_ptr = k_ptr_next;
        v_ptr += table_delta * int64_t(params.v_batch_stride);
    }

    // ======================================================== Masking 循环：需要 causal mask，不 Prefetch K ============================================================
    int n_block_loop = max(n_block_max - n_masking_steps, 0);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // 如果主循环没有 prefetch（n_block_max <= n_masking_steps），需要在这里 prefetch K
        if (masking_step == 0 && n_block_max <= n_masking_steps) {
            fp8_prefetch_k_to_lds<false/*Is_even_MN*/, kHeadDim, WARP_N, Element_k>(k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
        }

        // QK gemm
        vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16];
        fp8_qk_gemm<kBlockN, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(s_reg, q_regs, k_lds);

        // Prefetch V
        fp8_prefetch_v_to_lds<false/*Is_even_MN*/, kBlockN, kHeadDim, WARP_N, Element_k>(v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);

        // Mask
        // 对齐 fp16 fwd：非 causal 的 rest loop 要屏蔽最后一个 partial KV tile 的越界列。
        if constexpr (!Is_causal && !Is_local) {
            fp8_apply_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, max_seq_kv_offset, 0, lane_id);
        }

        // Causal mask
        if constexpr (Is_causal) {
            fp8_apply_causal_mask<kBlockN, WARP_M, WARP_N, ElementAccum>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN, lane_id);
        }

        // Softmax + 读取 V 到寄存器
        union_vec16_fp8 v_regs[kBlockN / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        fp8_softmax_and_schedule_v<Assume_valid_rows, kHeadDim, kBlockN, WARP_M, WARP_N, WARP_K, Element_k, ElementAccum>(s_reg, p_reg, scores_max, scores_sum, acc_o, softmax_scale_log2, v_regs, v_lds);

        // PV MMAC（不 Prefetch K）
        fp8_pv_gemm_and_prefetch_k<false/*PrefetchK*/, false/*Is_even_MN*/, kHeadDim, kBlockN, WARP_M, WARP_N, Element_k, ElementAccum>(acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);

        // 更新 K/V 指针
        const int next_n_block_loop = min(n_block_max - 1, n_block_loop + 1);
        const int64_t table_delta = int64_t(block_table[next_n_block_loop] - block_table[n_block_loop]);
        k_ptr += table_delta * int64_t(params.k_batch_stride);
        v_ptr += table_delta * int64_t(params.v_batch_stride);
    }

    // ========================================== rescale by scores_sum ==========================================
    ElementAccum lse[WARP_M / 16];
    fp8_epilogue_rescale_acc_o<Assume_valid_rows, kHeadDim, WARP_M, WARP_N, ElementAccum>(acc_o, scores_max, scores_sum, lse, softmax_scale, v_descale);

    // ========================================== lse storation (varlen) ==========================================
    if constexpr (Return_softmax) {
        fp8_epilogue_store_lse<false/*Is_even_MN*/, WARP_M, ElementAccum>(
            softmax_lse_ptr, scores_max, scores_sum, lse, row_offset_lse, binfo.actual_seqlen_q, m_block * kBlockM + warp_id * WARP_M, lane_id);
    }

    // ========================================== Storation =============================================
    fp8_epilogue_store_output<false/*Is_even_MN*/, kBlockM, kHeadDim, WARP_M, WARP_N, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, params.o_row_stride, binfo.actual_seqlen_q);
}


#endif

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout>
__global__ void __launch_bounds__(256, 1) flash_fp8_fwd_prefix_prefill_kernel_gfx92a(Flash_fwd_params params) {
#if defined(__gfx92a__)
    constexpr bool Do_lpt = Is_causal;
    const int bidh = Do_lpt ? blockIdx.x : blockIdx.y;
    const int bidb = Do_lpt ? blockIdx.y : blockIdx.z;
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z : blockIdx.x;
    flash::compute_fp8_attn_prefix_prefill_1rowblock_gfx92a<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_K, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}


} // namespace flash
