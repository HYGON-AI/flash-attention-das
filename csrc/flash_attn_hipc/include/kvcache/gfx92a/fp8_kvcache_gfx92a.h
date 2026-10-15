#pragma once
#include "kvcache/gfx92a/fp8_qk_gemm_prefetch_v_mls_ds_gfx92a.h"
#include "kvcache/gfx92a/fp8_pv_gemm_prefetch_k_mls_ds_gfx92a.h"
#include "kvcache/gfx92a/fp8_softmax_gfx92a.h"
#include "kvcache/gfx92a/fp8_epilogue_gfx92a.h"

namespace gfx92a_fp8_pa {

#if defined(__gfx92a__)
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Split, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_fp8_attn_prefix_prefill_1rowblock_gfx92a(
        const Params &params,
        const int bidb,
        const int bidh,
        const int m_block,
        const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using Element_p        = fp8_e4m3;
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
    constexpr int WARP_NUM  = kBlockM == WARP_M ? 4 : kBlockM / WARP_M;
    constexpr int COMPUTE_BLOCK_N = kBlockM == WARP_M ? WARP_N : kBlockN;

    // Varlen BlockInfo
    flash::BlockInfo<true/*Varlen*/, false/*Is_kvcache*/> binfo(params, bidb);

    const int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int split_id = 0;
    int partition_size = original_actual_seqlen_k;
    int split_seqlen_start = 0;
    if constexpr (Split) {
        split_id = blockIdx.y;
        partition_size = splitkv_get_partitionsize_of_fix_numsplits(
            original_actual_seqlen_k, params.num_splits);
        split_seqlen_start = split_id * partition_size;
        binfo.actual_seqlen_k = min(
            max(original_actual_seqlen_k - split_seqlen_start, 0),
            partition_size);
    }

    int max_seq_q_offset = binfo.actual_seqlen_q - m_block * kBlockM;
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // Shared LDS workspace.
    extern __shared__ int8_t lds[];
    int8_t* q_lds = lds + 0;
    int8_t* k_lds = lds + 0;
    int8_t* v_lds = lds + 0;
    ElementAccum* reduce_lds =
        reinterpret_cast<ElementAccum*>(lds + 16 * 1024);

    // Compute offsets for varlen paged attention.
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    if constexpr (Split) {
        block_table += ceil_div(split_seqlen_start, page_block_size);
    }
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    int current_page_idx = block_table[block_table_idx];

    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;

    if constexpr (Layout == 1) { /*bshd layout*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride)
                     + params.q_head_stride * bidh;
        row_offset_k = int64_t(current_page_idx) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(current_page_idx) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride)
                     + params.o_head_stride * bidh;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    } else { /*bhsd layout*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride)
                     + bidh * params.q_head_stride;
        row_offset_k = int64_t(current_page_idx) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(current_page_idx) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_o = (binfo.sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride)
                     + bidh * params.o_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q;
    }

    // FP8 descale: per-head descale
    int row_offset_q_descale = bidb * params.q_descale_batch_stride
                             + bidh * params.q_descale_head_stride;
    int row_offset_k_descale = bidb * params.k_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.k_descale_head_stride;
    int row_offset_v_descale = bidb * params.v_descale_batch_stride + int(bidh / params.h_h_k_ratio) * params.v_descale_head_stride;

    // Prefetch helpers prepare the raw FP8 pointers for matrix loads.
    Element_k* q_ptr = reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q;
    Element_k* k_ptr = reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k;
    Element_k* v_ptr = reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_v;

    ElementAccum* q_descale_ptr   = reinterpret_cast<ElementAccum*>(params.q_descale_ptr);
    ElementAccum* k_descale_ptr   = reinterpret_cast<ElementAccum*>(params.k_descale_ptr);
    ElementAccum* v_descale_ptr   = reinterpret_cast<ElementAccum*>(params.v_descale_ptr);

    ElementAccum q_descale   = q_descale_ptr[row_offset_q_descale];
    ElementAccum k_descale   = k_descale_ptr[row_offset_k_descale];
    ElementAccum qk_descale = q_descale * k_descale;
    ElementAccum softmax_scale = params.scale_softmax * qk_descale;
    ElementAccum softmax_scale_log2 = params.scale_softmax_log2 * qk_descale;
    ElementAccum v_descale   = v_descale_ptr[row_offset_v_descale];

    ElementAccum* softmax_lse_ptr = reinterpret_cast<ElementAccum *>(
        Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr);
    if constexpr (Split) {
        softmax_lse_ptr += split_id * params.h * params.total_q;
    }
    Element* o_ptr = reinterpret_cast<Element *>(
        Split ? params.oaccum_ptr : params.o_ptr) + row_offset_o;
    if constexpr (Split) {
        o_ptr += split_id * params.total_q * params.o_row_stride;
    }

    // Load Q. In KV-parallel decode all four waves consume the same Q tile,
    // so only wave 0 needs to issue the global-to-LDS MLS requests.
    if constexpr (kBlockM == WARP_M) {
        if (warp_id == 0) {
            fp8_prefetch_q_to_lds<
                false/*Is_even_MN*/, kHeadDim, WARP_M, Element_k>(
                    q_ptr, q_lds, 0, params.q_row_stride,
                    max_seq_q_offset);
        }
        // vmcnt is wave-local. Wave 0 completes its MLS writes before the
        // workgroup barrier releases the other waves to consume Q from LDS.
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n");
        __builtin_amdgcn_sched_barrier(0);
    } else {
        fp8_prefetch_q_to_lds<
            false/*Is_even_MN*/, kHeadDim, WARP_M, Element_k>(
                q_ptr, q_lds, warp_id, params.q_row_stride,
                max_seq_q_offset);
    }

    int lane_id = threadIdx.x & 63;

    // Score statistics and output accumulators.
    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16];
    fp8_attention_initialize<kHeadDim, WARP_M, WARP_N, ElementAccum>(scores_max, scores_sum, acc_o);

    // Load Q from LDS to VGPRs.
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64];
    load_q_from_lds_to_vgpr<kHeadDim, WARP_M, Element_k>(
        q_regs, q_lds, kBlockM == WARP_M ? 0 : warp_id, lane_id);

    // ======================================================== Mainloop ======================================================================
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal) {
        const int causal_seqlen_end = min(
            original_actual_seqlen_k,
            (m_block + 1) * kBlockM +
                original_actual_seqlen_k - binfo.actual_seqlen_q);
        const int causal_partition_end = max(
            causal_seqlen_end - split_seqlen_start, 0);
        n_block_max = min(
            n_block_max, ceil_div(causal_partition_end, kBlockN));
    }
    bool needs_second_masking_step = false;
    if constexpr (Is_causal) {
        const int first_causal_col =
            m_block * kBlockM +
            original_actual_seqlen_k - binfo.actual_seqlen_q;
        const int last_block_col =
            split_seqlen_start + max(n_block_max - 1, 0) * kBlockN;
        needs_second_masking_step =
            n_block_max > 1 && first_causal_col < last_block_col;
    }
    constexpr int n_masking_steps = 1;
    constexpr bool Assume_valid_rows = !Is_local;

    // Prefetch the first K tile.
    if (n_block_max > n_masking_steps) {
        fp8_prefetch_k_to_lds<
            true/*Is_even_MN*/, kHeadDim, WARP_N, Element_k, WARP_NUM>(
                k_ptr, k_lds, warp_id, params.k_row_stride,
                binfo.actual_seqlen_k);
    }

    // Main pipeline: compute the unmasked tiles and prefetch the next K tile.
    for (int n_block_loop = 0; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;
        const int next_n_block_loop = n_block_loop + 1;
        const int next_page_idx =
            block_table[min(n_block_max - 1, next_n_block_loop)];
        const int64_t table_delta =
            int64_t(next_page_idx - current_page_idx);
        Element_k* k_ptr_next =
            k_ptr + table_delta * int64_t(params.k_batch_stride);

        // QK GEMM; K was prefetched into LDS by the previous iteration.
        vec4_Accum<ElementAccum> s_reg[COMPUTE_BLOCK_N / WARP_N][WARP_M / 16][WARP_N / 16];
        int8_t* k_lds_compute = kBlockM == WARP_M
            ? k_lds + (warp_id >> 1) * 8192 + (warp_id & 1) * 2048
            : k_lds;
        fp8_qk_gemm<COMPUTE_BLOCK_N, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(
            s_reg, q_regs, k_lds_compute);

        // Prefetch V
        fp8_prefetch_v_to_lds<
            true/*Is_even_MN*/, kBlockN, kHeadDim, WARP_N, Element_k,
            WARP_NUM>(v_ptr, v_lds, warp_id, params.v_row_stride,
                      max_seq_kv_offset);

        if constexpr (Is_causal) {
            if (needs_second_masking_step &&
                n_block_loop == n_block_max - 2) {
                fp8_apply_causal_mask<
                    COMPUTE_BLOCK_N, WARP_M, WARP_N, ElementAccum>(
                        s_reg, binfo.actual_seqlen_q,
                        original_actual_seqlen_k,
                        m_block * kBlockM +
                            (kBlockM == WARP_M ? 0 : warp_id * WARP_M),
                        split_seqlen_start + n_block_loop * kBlockN +
                            (kBlockM == WARP_M ? warp_id * WARP_N : 0),
                        lane_id);
            }
        }

        // Softmax and load V into registers.
        union_vec16_fp8 v_regs[COMPUTE_BLOCK_N / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        if constexpr (kBlockM == WARP_M) {
            fp8_softmax_and_schedule_v<
                Assume_valid_rows, kHeadDim, WARP_M, WARP_N, WARP_K,
                Element_p, ElementAccum>(
                    s_reg, p_reg, scores_max, scores_sum, acc_o,
                    softmax_scale_log2, v_regs, v_lds, reduce_lds, warp_id);
        } else {
            fp8_softmax_and_schedule_v<
                Assume_valid_rows, kHeadDim, COMPUTE_BLOCK_N, WARP_M,
                WARP_N, WARP_K, Element_p, ElementAccum>(
                    s_reg, p_reg, scores_max, scores_sum, acc_o,
                    softmax_scale_log2, v_regs, v_lds);
        }

        // PV MMAC and prefetch K from the next physical page.
        const int max_seq_kv_offset_next = binfo.actual_seqlen_k - next_n_block_loop * kBlockN;
        fp8_pv_gemm_and_prefetch_k_paged<
            false/*Is_even_MN*/, kHeadDim, COMPUTE_BLOCK_N, WARP_M,
            WARP_N, Element_k, ElementAccum, WARP_NUM>(
                acc_o, p_reg, v_regs, v_lds, k_ptr_next, k_lds,
                warp_id, params.k_row_stride, max_seq_kv_offset_next);

        // Advance the paged K/V pointers.
        current_page_idx = next_page_idx;
        k_ptr = k_ptr_next;
        v_ptr += table_delta * int64_t(params.v_batch_stride);
    }

    // Final masked tile without prefetching another K tile.
    int n_block_loop = max(n_block_max - n_masking_steps, 0);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        int max_seq_kv_offset = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // If the main loop was skipped, prefetch K here.
        if (masking_step == 0 && n_block_max <= n_masking_steps) {
            fp8_prefetch_k_to_lds<false/*Is_even_MN*/, kHeadDim, WARP_N, Element_k, WARP_NUM>(k_ptr, k_lds, warp_id, params.k_row_stride, max_seq_kv_offset);
        }

        // QK gemm
        vec4_Accum<ElementAccum> s_reg[COMPUTE_BLOCK_N / WARP_N][WARP_M / 16][WARP_N / 16];
        int8_t* k_lds_compute = kBlockM == WARP_M
            ? k_lds + (warp_id >> 1) * 8192 + (warp_id & 1) * 2048
            : k_lds;
        fp8_qk_gemm<COMPUTE_BLOCK_N, kHeadDim, WARP_M, WARP_N, Element_k, ElementAccum>(
            s_reg, q_regs, k_lds_compute);

        // Prefetch V
        fp8_prefetch_v_to_lds<false/*Is_even_MN*/, kBlockN, kHeadDim, WARP_N, Element_k, WARP_NUM>(v_ptr, v_lds, warp_id, params.v_row_stride, max_seq_kv_offset);

        // Mask
        // Mask columns outside the final partial KV tile.
        if constexpr (!Is_causal && !Is_local) {
            fp8_apply_mask<COMPUTE_BLOCK_N, WARP_M, WARP_N, ElementAccum>(
                s_reg, max_seq_kv_offset,
                kBlockM == WARP_M ? warp_id * WARP_N : 0, lane_id);
        }

        // Causal mask
        if constexpr (Is_causal) {
            fp8_apply_causal_mask<
                COMPUTE_BLOCK_N, WARP_M, WARP_N, ElementAccum>(
                s_reg, binfo.actual_seqlen_q, original_actual_seqlen_k,
                m_block * kBlockM +
                    (kBlockM == WARP_M ? 0 : warp_id * WARP_M),
                split_seqlen_start + n_block_loop * kBlockN +
                    (kBlockM == WARP_M ? warp_id * WARP_N : 0),
                    lane_id);
        }

        // Softmax and load V into registers.
        union_vec16_fp8 v_regs[COMPUTE_BLOCK_N / WARP_N][kHeadDim / 32];
        union_vec32_fp8 p_reg[WARP_M / 16];
        if constexpr (kBlockM == WARP_M) {
            fp8_softmax_and_schedule_v<
                Assume_valid_rows, kHeadDim, WARP_M, WARP_N, WARP_K,
                Element_p, ElementAccum>(
                    s_reg, p_reg, scores_max, scores_sum, acc_o,
                    softmax_scale_log2, v_regs, v_lds, reduce_lds, warp_id);
        } else {
            fp8_softmax_and_schedule_v<
                Assume_valid_rows, kHeadDim, COMPUTE_BLOCK_N, WARP_M,
                WARP_N, WARP_K, Element_p, ElementAccum>(
                    s_reg, p_reg, scores_max, scores_sum, acc_o,
                    softmax_scale_log2, v_regs, v_lds);
        }

        // PV MMAC without prefetching another K tile.
        fp8_pv_gemm_and_prefetch_k<
            false/*PrefetchK*/, false/*Is_even_MN*/, kHeadDim,
            COMPUTE_BLOCK_N, WARP_M, WARP_N, Element_k, ElementAccum,
            WARP_NUM>(
                acc_o, p_reg, v_regs, v_lds, k_ptr, k_lds, warp_id,
                params.k_row_stride, max_seq_kv_offset);

    }

    if constexpr (kBlockM == WARP_M) {
        fp8_reduce_acc_o_across_waves<
            kHeadDim, WARP_M, WARP_N, ElementAccum>(
                acc_o, reinterpret_cast<ElementAccum*>(lds),
                warp_id, lane_id);
    }

    // ========================================== rescale by scores_sum ==========================================
    ElementAccum lse[WARP_M / 16];
    if constexpr (kBlockM != WARP_M) {
        fp8_epilogue_rescale_acc_o<
            Assume_valid_rows, kHeadDim, WARP_M, WARP_N, ElementAccum>(
                acc_o, scores_max, scores_sum, lse,
                softmax_scale, v_descale);
    } else if (warp_id == 0) {
        fp8_epilogue_rescale_acc_o<
            Assume_valid_rows, kHeadDim, WARP_M, WARP_N, ElementAccum>(
                acc_o, scores_max, scores_sum, lse,
                softmax_scale, v_descale);
    }

    // ========================================== lse storation (varlen) ==========================================
    if constexpr (Return_softmax && kBlockM != WARP_M) {
        fp8_epilogue_store_lse<false/*Is_even_MN*/, WARP_M, ElementAccum>(
            softmax_lse_ptr, scores_max, scores_sum, lse, row_offset_lse, binfo.actual_seqlen_q, m_block * kBlockM + warp_id * WARP_M, lane_id);
    } else if constexpr (Return_softmax && kBlockM == WARP_M) {
        if (warp_id == 0) {
            fp8_epilogue_store_lse<false/*Is_even_MN*/, WARP_M, ElementAccum>(
                softmax_lse_ptr, scores_max, scores_sum, lse,
                row_offset_lse, binfo.actual_seqlen_q,
                m_block * kBlockM, lane_id);
        }
    }

    // ========================================== Storation =============================================
    if constexpr (kBlockM != WARP_M) {
        fp8_epilogue_store_output<
            false/*Is_even_MN*/, kBlockM, kHeadDim, WARP_M, WARP_N,
            Element, ElementAccum>(
                o_ptr, acc_o, m_block, warp_id, lane_id,
                params.o_row_stride, binfo.actual_seqlen_q);
    } else if (warp_id == 0) {
        fp8_epilogue_store_output<
            false/*Is_even_MN*/, kBlockM, kHeadDim, WARP_M, WARP_N,
            Element, ElementAccum>(
                o_ptr, acc_o, m_block, 0, lane_id,
                params.o_row_stride, binfo.actual_seqlen_q);
    }
}

#endif

} // namespace gfx92a_fp8_pa
