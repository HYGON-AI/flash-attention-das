/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sAccs = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutAccs{});
    #endif
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sV = make_tensor(size(sK) > size(sAccs) ? sK.data() + size(sK):
        sAccs.data() + size(sAccs),
        typename Kernel_traits::SmemLayoutKV{});
    #else
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    #endif
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    #ifndef GEMM1_AMATRIX_WITH_SMEM
    typename Kernel_traits::TiledMma_FOR_GEMM1 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    #endif
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    #else
    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    #endif

    Tensor tSgS  = thr_mma.partition_C(gP);

    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #else
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #endif

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    #ifdef GEMM1_AMATRIX_WITH_SMEM
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    #else
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    #endif

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        // flash::cp_async_wait<1>();
        // __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    // if (cute::thread0()) {
    //     printf("n_masking_steps = %d\n", n_masking_steps);
    // }
    // return;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s);
        // return;
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);

        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // return;
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;

            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        #ifdef GEMM1_AMATRIX_WITH_SMEM
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));
        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // wangaq debug
        // if (thread0()) {
        //     printf("rP layout:"); print(rP.layout());printf("\n");
        //     printf("tOrP layout:"); print(tOrP.layout());printf("\n");
        //     printf("tOrVt layout:"); print(tOrVt.layout());printf("\n");
        // }
        // return;
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
    #if 1
    // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // if (n_block == n_block_max - 1) {
        __syncthreads();
        // }
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s);
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);


        // flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        //     acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        //     smem_thr_copy_Q, smem_thr_copy_K
        // );
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;
            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        #ifdef GEMM1_AMATRIX_WITH_SMEM
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        // Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));

        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }
    // if (cute::thread0())
    // {
    //     // for (int i = 0; i < size(acc_o); i++) {
    //     //     printf("i = %d acc_o = %f\n", i, float(acc_o[i]));
    //     // }
    //         // print_tensor(tOsVt);
    // }
    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        cute::copy(tKrK, tKsK);

        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        __syncthreads();
        cute::copy(tVrV, tVsV);
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        __syncthreads();
        cute::copy(tKrK, tKsK);
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);

        __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        __syncthreads();
        cute::copy(tVrV, tVsV);
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        __syncthreads();

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        // static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
        // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        cute::copy(tKrK, tKsK);
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        // __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // __syncthreads();
        cute::copy(tKrK, tKsK);
        __syncthreads();

        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);

        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        // __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        // __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem || kHeadDim == 64) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        __syncthreads();

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        // static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
        // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {
// #if 1
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d_value),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // if (thread0())
    // {
    //     print("tVgV "); print(tVgV); print("\n");
    //     print("tVsV "); print(tVsV); print("\n");

    // }
#if 1
    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_QKV.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();
        // wangaq debug
        // __syncthreads();
        // if (cute::thread(0, 0)) {
        //     __half * tmp = reinterpret_cast<__half*>(sK.data().get());
        //     int col = 8;
        //     for (int i = 0; i < size(sK)/col; ++i) {
        //         printf("K %3d: ", i);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%10.4f ", __half2float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        // __syncthreads();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // 
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV);

        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        // __syncthreads();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem || true) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // __syncthreads();
        asm volatile("s_barrier\n\t");

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch_long_context(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    // Maximun stage number is 8 - 1 = 7 since we have 4 tiles of K/V each, and minus 1 for the current tile.
    // Here we use 6 to avoid introducing extra s_barrier instructions between LDS-load and DMA-transfer on the same stage.
    constexpr int kStages = 6;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    auto buffer_k = DmaBuffer<4, _64x32, Is_even_K>(gK.data().get(), sK.data().get(), (n_block_max - 1) * kBlockN * params.k_row_stride, params.k_row_stride, params.d);
    auto buffer_v = DmaBuffer<4, _16x128, Is_even_K>(gV.data().get(), sV.data().get(), (n_block_max - 1) * kBlockN * params.v_row_stride, params.v_row_stride, params.d_value);
    // Prefetch K0-K3/V0-V2, 7 stages in total.
    buffer_k.template load<Is_even_MN>(0, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_k.template load<Is_even_MN>(1, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_k.template load<Is_even_MN>(2, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_k.template load<Is_even_MN>(3, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_v.template load<Is_even_MN>(0, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_v.template load<Is_even_MN>(1, binfo.actual_seqlen_k - n_block * kBlockN);
    buffer_v.template load<Is_even_MN>(2, binfo.actual_seqlen_k - n_block * kBlockN);
    Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    // LDS load K0
    s_waitcnt<kStages>();
    cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tSrK_copy_view(_, _, 0));
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);

        // Prefetch V3.
        buffer_v.template load<Is_even_MN>(3, binfo.actual_seqlen_k - n_block * kBlockN);
        if (n_block > n_block_min) {
            buffer_k.offset(-kBlockN * params.k_row_stride);
            buffer_v.offset(-kBlockN * params.v_row_stride);
        }
        // LDS load K1, MMA K0.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tSrK_copy_view(_, _, 1));
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K0.
        buffer_k.template load</*Is_even_MN=*/true>(0);
        // LDS load K2, MMA K1.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tSrK_copy_view(_, _, 2));
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K1.
        buffer_k.template load</*Is_even_MN=*/true>(1);
        // LDS load K3, MMA K2.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 3), tSrK_copy_view(_, _, 3));
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K2.
        buffer_k.template load</*Is_even_MN=*/true>(2);
        // LDS load V0, MMA K3.
        s_waitcnt<kStages>();
        // The earliest used data is loaded the latest to reduce s_wait instructions
        __ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tGrQ(_, _, 3), tSrK(_, _, 3), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        Tensor rP =
            masking_step == 0
            ? (softmax.template fast_softmax_rescale_o<Element, /*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2))
            : (softmax.template fast_softmax_rescale_o<Element, /*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2));
        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        // Prefetch K3.
        buffer_k.template load</*Is_even_MN=*/true>(3);
        // LDS load V1, MMA V0.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V0.
        buffer_v.template load</*Is_even_MN=*/true>(0);
        // LDS load V2, MMA V1.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V1.
        buffer_v.template load</*Is_even_MN=*/true>(1);
        // LDS load V3, MMA V2.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 2), tOrVt(_, _, 2), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V2.
        buffer_v.template load</*Is_even_MN=*/true>(2);
        // LDS load K0, MMA V3.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tSrK_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 3), tOrVt(_, _, 3), acc_o);
        __builtin_amdgcn_sched_barrier(0);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block > n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);

        // Prefetch V3.
        buffer_v.template load</*Is_even_MN=*/true>(3);
        buffer_k.offset(-kBlockN * params.k_row_stride);
        buffer_v.offset(-kBlockN * params.v_row_stride);
        // LDS load K1, MMA K0.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tSrK_copy_view(_, _, 1));
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K0.
        buffer_k.template load</*Is_even_MN=*/true>(0);
        // LDS load K2, MMA K1.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tSrK_copy_view(_, _, 2));
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K1.
        buffer_k.template load</*Is_even_MN=*/true>(1);
        // LDS load K3, MMA K2.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 3), tSrK_copy_view(_, _, 3));
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch K2.
        buffer_k.template load</*Is_even_MN=*/true>(2);
        // LDS load V0, MMA K3.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tGrQ(_, _, 3), tSrK(_, _, 3), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        Tensor rP = softmax.template fast_softmax_rescale_o<Element, /*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        // Prefetch K3.
        buffer_k.template load</*Is_even_MN=*/true>(3);
        // LDS load V1, MMA V0.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V0.
        buffer_v.template load</*Is_even_MN=*/true>(0);
        // LDS load V2, MMA V1.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V1.
        buffer_v.template load</*Is_even_MN=*/true>(1);
        // LDS load V3, MMA V2.
        s_waitcnt<kStages>();
        __ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 2), tOrVt(_, _, 2), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // Prefetch V2.
        buffer_v.template load</*Is_even_MN=*/true>(2);
        // LDS load K0, MMA V3.
        s_waitcnt<kStages>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tSrK_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 3), tOrVt(_, _, 3), acc_o);
        __builtin_amdgcn_sched_barrier(0);
    }
    if (n_block >= n_block_min) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);

        // Prefetch V3.
        buffer_v.template load</*Is_even_MN=*/true>(3);
        // LDS load K1, MMA K0.
        s_waitcnt<6>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tSrK_copy_view(_, _, 1));
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // LDS load K2, MMA K1.
        s_waitcnt<5>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tSrK_copy_view(_, _, 2));
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // LDS load K3, MMA K2.
        s_waitcnt<4>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 3), tSrK_copy_view(_, _, 3));
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);
        // LDS load V0, MMA K3.
        s_waitcnt<3>();
        __ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 0>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tGrQ(_, _, 3), tSrK(_, _, 3), acc_s_ori);
        __builtin_amdgcn_sched_barrier(0);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        Tensor rP = softmax.template fast_softmax_rescale_o<Element, /*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        // LDS load V1, MMA V0.
        s_waitcnt<2>();
        __ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 1>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // LDS load V2, MMA V1.
        s_waitcnt<1>();
        __ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 2>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // LDS load V3, MMA V2.
        s_waitcnt<0>();
        __ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<3, 3>(tOsVt, tOrVt_copy_view);
        __ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 2), tOrVt(_, _, 2), acc_o);
        __builtin_amdgcn_sched_barrier(0);
        // MMA V3.
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 3), tOrVt(_, _, 3), acc_o);
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template fast_normalize_softmax_lse_with_sinks<Element, Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template fast_normalize_softmax_lse<Element, Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n",
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    }
#endif
}

#define S_WAITCNT asm volatile("s_waitcnt vmcnt(3) \n s_barrier")
#define S_BARRIER asm volatile("s_barrier")
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    // __syncthreads();
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

#if defined(__gfx936__)
    constexpr bool kUseHcuPrefetch =
        std::is_same<Element, cutlass::bfloat16_t>::value && kHeadDim == 128
        && Is_even_K && !Is_dropout && !Is_causal && !Is_local
        && !Has_alibi && !Is_softcap && !Return_softmax && !Kernel_traits::kLongContext;
#else
    constexpr bool kUseHcuPrefetch = false;
#endif
    constexpr bool kSkipIdentityRescale = kUseHcuPrefetch && kBlockM == 128;
    constexpr bool kPipelineKey = kUseHcuPrefetch && kHeadDimV == 128
        && (kBlockM == 64 || kBlockM == 128) && kBlockN == 64
        && kStages == 3 && kNWarps == 4;
    flash::Softmax<size<1>(acc_o), kSkipIdentityRescale> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, i, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K>(gK, sK, kStages + i, params.k_row_stride, params.d);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, i, params.v_row_stride, params.d_value);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        s_waitcnt<2>();
        if constexpr (kPipelineKey) {
            if (n_block > n_block_min) {
                // The barrier releases the consumed V slice before K overwrites it.
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            }
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        if constexpr (kPipelineKey) {
            if (n_block > n_block_min) {
                s_waitcnt<2>();  // Account for the outstanding K loads as well as V.
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            } else {
                s_waitcnt<1>();
            }
        } else {
            s_waitcnt<1>();
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        if constexpr (kPipelineKey) {
            if (n_block > n_block_min) {
                s_waitcnt<2>();  // Account for the outstanding K loads as well as V.
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            } else {
                s_waitcnt<0>();
            }
        } else {
            s_waitcnt<0>();
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr (!kPipelineKey) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }
        }
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if constexpr (Is_even_K) {
                        col = (laneId / 16) * 2 + ni * 32;
                        using result_type = cutlass::Array<Element, 2>;
                        for (int ei = 0; ei < 4; ++ei)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                            res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                            *(result_type*)(&gO(row, col)) = res;
                            col += 8;
                        }
                    } else {
                        col = (laneId / 16) * 2 + ni * 32;
                        if (col < params.d_value) {
                            using result_type = cutlass::Array<Element, 2>;
                            for (int ei = 0; ei < 4; ++ei)
                            {
                                result_type res;
                                res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                                res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                                *(result_type*)(&gO(row, col)) = res;
                                col += 8;
                            }
                        }
                    }
                    // TODO dim!=128 with risk
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    // col += 4;
                }
            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch_fp8(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<ElementOUT>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)128,128

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));//64,128

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + 8192, typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);
   
    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol_fp8<_64x64, 128/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x32_B8, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_fp8<_32x128>(tOsVt8x64.layout()));

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[bidb * params.k_descale_batch_stride + bidh * params.k_descale_head_stride];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[bidb * params.v_descale_batch_stride + bidh * params.v_descale_head_stride];

    float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    float scale_softmax = params.scale_softmax*q_descale*k_descale;

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

   
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);



#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_barrier");


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
      
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

    
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

      
        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
       
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_barrier");
     
    
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }

       
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif

    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

         asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

       
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, scale_softmax_log2);
      
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
      
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
  
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }
       
       
    }

    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse_fp8<Is_dropout>(acc_o, scale_softmax,v_descale, params.rp_dropout);
    
 

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    using result_type = cutlass::Array<bfloat16_t, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16)*4 + ni * 32;
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(0, mi, ni), acc_o(1, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(2, mi, ni), acc_o(3, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);

                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];

                    col += 16;
                }
              
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(4, mi, ni), acc_o(5, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(6, mi, ni), acc_o(7, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);
                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];
                }
                
               

            }
        }
    } 
  
#endif
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages - 1;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDimV/32>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #define BIDX 0
    #define BIDY 0
    #define BIDZ 0

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        S_BARRIER;
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));


        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);


        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx / 64);
            const int row_idx_offset_in_block = (tidx % 16) + (wave_id * 16);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        
        
        s_waitcnt<6>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        
        s_waitcnt<4>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        S_BARRIER;
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 2, params.v_row_stride, params.d_value);
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 3, params.v_row_stride, params.d_value);
        
        s_waitcnt<6>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        s_waitcnt<4>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim32_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    // constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    // Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x64_64, kHeadDimV/32>(tOsVt8x64.layout()));


    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    // static_assert(k0_loops == 1 && k1_loops == 8);
    if constexpr (Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        
        lds_direct_copy_even_k<0, Is_even_MN, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, Is_even_MN, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    } else {
        auto gK1 = local_tile(gK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
        auto sK1 = local_tile(sK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
        lds_direct_copy<false, Is_even_MN>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<false, Is_even_MN>(gK1, sK1, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN - 64);
        lds_direct_copy<false, Is_even_MN, _64x32_load_v>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<false, Is_even_MN, _64x32_load_v>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        cute::copy(smem_tiled_copy_K, tSsK(_, 0, 0), tSrK(_, 0, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 0, 0), acc_s_ori(_, m, 0));
        }
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        cute::copy(smem_tiled_copy_K, tSsK(_, 1, 0), tSrK(_, 1, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 1, 0), acc_s_ori(_, m, 1));
        }
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<4>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<5>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<6>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<7>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<0, true, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<0, true, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                auto gK1 = local_tile(gK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
                auto sK1 = local_tile(sK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
                lds_direct_copy<false, true>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<false, true>(gK1, sK1, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN - 64);
                lds_direct_copy<false, true, _64x32_load_v>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<false, true, _64x32_load_v>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            }

        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        cute::copy(smem_tiled_copy_K, tSsK(_, 0, 0), tSrK(_, 0, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 0, 0), acc_s_ori(_, m, 0));
        }
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        cute::copy(smem_tiled_copy_K, tSsK(_, 1, 0), tSrK(_, 1, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 1, 0), acc_s_ori(_, m, 1));
        }
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<4>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<5>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<6>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<7>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<0, true, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<0, true, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                auto gK1 = local_tile(gK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
                auto sK1 = local_tile(sK, Shape<_64, Int<kHeadDim>>{}, make_coord(1, 0));
                lds_direct_copy<false, true>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<false, true>(gK1, sK1, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN - 64);
                lds_direct_copy<false, true, _64x32_load_v>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<false, true, _64x32_load_v>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            }


        }
    }

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    // if (get<1>(taccOcO(0)) == 0) {
    //     #pragma unroll
    //     for (int mi = 0; mi < size(lse); ++mi) {
    //         const int row = get<0>(taccOcO(0, mi, 0));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
    //     }
    // }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 2 + ni * 32;
                if constexpr (Is_even_K)
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        if (col < params.d_value)
                        {
                            gO(row, col) = res[0];
                        }
                        if (col + 1 < params.d_value)
                        {
                            gO(row, col + 1) = res[1];
                        }
                        col += 8;
                    }
                }

            }

            if (laneId / 16 == 0) {
                gLSE(row) = lse(mi);
            }
                
        }
    } 
   
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    // constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x64_64, kHeadDimV/32>(tOsVt8x64.layout()));


    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(k0_loops == 2 && k1_loops == 4);
    if constexpr(Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        
        lds_direct_copy_even_k<0, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < k0_loops; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        #pragma unroll
        for (int i = 0; i < k1_loops; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else
            {
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                

                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else
            {
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                

                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);
            }

        }
    }

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 2 + ni * 32;
                if constexpr (Is_even_K)
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        if (col < params.d_value)
                        {
                            gO(row, col) = res[0];
                        }
                        if (col + 1 < params.d_value)
                        {
                            gO(row, col + 1) = res[1];
                        }
                        col += 8;
                    }
                }

            }
        }
    } 
   
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim256_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr bool kEarlyNextK = Kernel_traits::kPrefetchNextK && Is_even_K;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // if (tidx == 0) printf("bidb:%d bidh:%d m_block:%d\n", bidb, bidh, m_block);
    // printf();

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_split = local_tile(acc_o, Shape<Int<8>, Int<kBlockM/64>, Int<kHeadDimV / 32 / 2>>{}, make_coord(0, 0, _)); 
    auto acc_o_temp0 = acc_o_split(_, _, _, 0);
    auto acc_o_temp1 = acc_o_split(_, _, _, 1);
    clear(acc_o);
    constexpr static int K_BUFF_SIZE = 4;
    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    s_waitcnt<0>();
    __syncthreads();
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    } else {
        lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

    #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int block_row_idx =  m_block * kBlockM + (tidx % 16) + (tidx / 64 * 16);
            const int block_col_idx = n_block * kBlockN;
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                } else {
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                } else {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr(!Is_even_K) {
                #pragma unroll 
                for (int i = 0; i < 3; i ++) {
                    lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
                }
            } else {
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        s_barrier();
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        s_barrier();

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        // In the steady loop, reuse each LDS slot only after its last V read
        // and the existing CTA barrier. The masked prologue is unchanged.
        const bool prefetch_next_k = kEarlyNextK && n_block > n_block_min;
        if constexpr (kEarlyNextK) {
            if (prefetch_next_k) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                s_waitcnt<3>();
            } else {
                s_waitcnt<2>();
            }
        } else {
            s_waitcnt<2>();
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if constexpr (kEarlyNextK) {
            if (prefetch_next_k) {
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                s_waitcnt<3>();
            } else {
                s_waitcnt<1>();
            }
        } else {
            s_waitcnt<1>();
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if constexpr (kEarlyNextK) {
            if (prefetch_next_k) {
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                s_waitcnt<3>();
            } else {
                s_waitcnt<0>();
            }
        } else {
            s_waitcnt<0>();
        }
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();


        if (n_block > n_block_min) {
            if constexpr (!kEarlyNextK) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            }
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr (!kEarlyNextK) {
                if constexpr(!Is_even_K) {
                    #pragma unroll
                    for (int i = 0; i < 3; i ++) {
                        lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
                    }
                } else {
                    lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                    lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                    lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                }
            }
        }
    }
    #endif

    __builtin_amdgcn_s_barrier();
    // Epilogue
    
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    s_waitcnt<0>();

    Tensor rO = flash::convert_type<Element>(acc_o);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{}, make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // __builtin_amdgcn_sched_barrier(0);
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    const int qo_len = binfo.actual_seqlen_q - m_block * kBlockM;

    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            // asm volatile("v_cmpx_lt_i32 exec, %0, %1":: "v"(row), "v"(qo_len) :);
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                if constexpr (Is_even_K) {
                    col = (laneId / 16) * 2 + ni * 32;
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                } else {
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                        col = (laneId / 16) + ni * 32 + ei * 4;
                        // wangaq debug
                        // printf("bidx:%d bidy:%d bidz:%d tid:%d mi:%d ni:%d ei:%d row:%d col:%d acc_o:%10.4f\n", 
                        // blockIdx.x, blockIdx.y, blockIdx.z, tidx, mi, ni, ei, row, col, acc_o(ei, mi, ni));
                        if (col < params.d_value) {
                            gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        } 
                        // else 
                        //     gO(row, col) = Element(0.0);
                    }
                }
            }
            // asm volatile("s_mov_b64 exec, 0xFFFFFFFFFFFFFFFF");
        }
    } 
    // __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim512_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // if (tidx == 0) printf("bidb:%d bidh:%d m_block:%d\n", bidb, bidh, m_block);
    // printf();

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_split = local_tile(acc_o, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 4>>{}, make_coord(0, 0, _)); 
    auto acc_o_temp0 = acc_o_split(_, _, _, 0);
    auto acc_o_temp1 = acc_o_split(_, _, _, 1);
    auto acc_o_temp2 = acc_o_split(_, _, _, 2);
    auto acc_o_temp3 = acc_o_split(_, _, _, 3);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    s_waitcnt<0>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 8, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 9, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 10, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 11, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 12, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 13, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 14, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 15, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 15, 3);


        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

    #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int block_row_idx =  m_block * kBlockM + (tidx % 16) + (tidx / 64 * 16);
            const int block_col_idx = n_block * kBlockN;
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                } else {
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                } else {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();


        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
         lds_direct_copy<Is_even_K>(gK, sK, 3, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 4, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 5, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 6, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 7, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 8, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 9, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 10, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 11, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 12, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 13, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 14, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 15, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 15, 3);


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        s_barrier();
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        s_barrier();

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }
        lds_direct_copy<Is_even_K, true, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
    #endif

    __builtin_amdgcn_s_barrier();
    // Epilogue
    #if 1

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    s_waitcnt<0>();

    Tensor rO = flash::convert_type<Element>(acc_o);
    

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{}, make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // __builtin_amdgcn_sched_barrier(0);
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    const int qo_len = binfo.actual_seqlen_q - m_block * kBlockM;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            // asm volatile("v_cmpx_lt_i32 exec, %0, %1":: "v"(row), "v"(qo_len) :);
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    col = (laneId / 16) + ni * 32 + ei * 4;
                    // wangaq debug
                    // printf("bidx:%d bidy:%d bidz:%d tid:%d mi:%d ni:%d ei:%d row:%d col:%d acc_o:%10.4f\n", 
                    // blockIdx.x, blockIdx.y, blockIdx.z, tidx, mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                }
            }
            // asm volatile("s_mov_b64 exec, 0xFFFFFFFFFFFFFFFF");
        }
    } 
    #endif
    // __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64_prefetch_padding_mask(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    // constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x64_64, kHeadDimV/32>(tOsVt8x64.layout()));


    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(k0_loops == 2 && k1_loops == 4);
    #pragma unroll
    for (int i = 0; i < k0_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    #pragma unroll
    for (int i = 0; i < k1_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP,  m_block * kBlockM, n_block * (kBlockN), kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, n_block * (kBlockN), kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }               

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP,  m_block * kBlockM, block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }
    }

    // Epilogue
    #if 1  // use all lds
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    // static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    #endif
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int K_BUFF_SIZE = 4;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 128/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol<_16x128>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = kHeadDim / 32;
    constexpr int k1_loops = kBlockN / 16;
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) { // 0 1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 4 5
            // k1 3 0 1
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gV, sV, (i+2)%4, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i+3, (i+3)%4);
            S_BARRIER;
        }
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        
        // tail kStages == 3
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d);
            }
        }  

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0 0 1 2
            // k1 0 1 2 
            lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK,  kStages + i, params.k_row_stride, params.d);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 4 5
            // k1 3 0 1
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128, K_BUFF_SIZE>(gV, sV, (i+2)%4, params.v_row_stride, params.d_value);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i+3, (i+3)%4);
            S_BARRIER;
        }
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        
        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128, K_BUFF_SIZE>(gV, sV, 1, params.v_row_stride, params.d_value);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        
        // tail kStages == 3
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d);
            }
        }  
    }

    // Epilogue
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }



    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    } else 
                        gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla_prefetch_fp8(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int K_BUFF_SIZE = 4;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<ElementOUT>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0)); 

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});//64,192
    Tensor sV = make_tensor(sK.data() + 8192, typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    //Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);      // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tSgS  = thr_mma.partition_C(gP);


    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K 128,128

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});//192,64
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol_fp8<_64x64, 192/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x32_B8, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    // Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_fp8<_32x128>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[bidb * params.k_descale_batch_stride + bidh * params.k_descale_head_stride];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[bidb * params.v_descale_batch_stride + bidh * params.v_descale_head_stride];

    float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    float scale_softmax = params.scale_softmax*q_descale*k_descale;

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    // constexpr int k0_loops = kHeadDim / 64;
    // constexpr int k1_loops = kBlockN / 32;
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
  
    
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tCrK_copy_view(_, _, 2));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);//0,1,2,3
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);//tid%warp_row_stride
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
        ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2)
        : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2);

     
        // Convert acc_s from fp32 to fp8
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);
  
        {   
            //dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) { 

            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 2, params.k_row_stride, params.d);
            
            // lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);


        }  


        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
  
      
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tCrK_copy_view(_, _, 2));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));


        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);//0,1,2,3
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);//tid%warp_row_stride
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, scale_softmax_log2);
      
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);
   
        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
               Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }
        
        lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);


        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
      
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
  
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 2, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);

          
        }  
       
   
    }

    //Epilogue
    Tensor lse = softmax.template normalize_softmax_lse_fp8<Is_dropout>(acc_o, scale_softmax,v_descale, params.rp_dropout);
    


    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    using result_type = cutlass::Array<bfloat16_t, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16)*4 + ni * 32;
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(0, mi, ni),  acc_o(1, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(2, mi, ni),  acc_o(3, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);

                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];

                    col += 16;
                }
              
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(4, mi, ni), acc_o(5, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(6, mi, ni), acc_o(7, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);
                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];
                }
                
               

            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch_skip_softmax(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // ★ Attention Sinks: Shared memory for s_aux (max 64 heads) ★
    __shared__ ElementAccum smem_s_aux[64];
    __shared__ uint32_t skip_softmax_vote[1];

    // The thread index.
    const int tidx = threadIdx.x;

    
    // wangaq debug
    // if (tidx == 0) {
    //     uint32_t * gSkipInfo_ptr = reinterpret_cast<uint32_t*>(params.skip_blocks_info_ptr) + (bidb * params.h + bidh) * 2;
    //     printf("bidx:%d gSkipInfo_ptr:%d %d \n", blockIdx.x, *gSkipInfo_ptr, *(gSkipInfo_ptr + 1));
    // }

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    // ★ Attention Sinks: Cooperatively load s_aux into shared memory (CuTe style) ★
    if (params.s_aux_ptr != nullptr) {
        const int num_heads = params.h;
        Tensor gS_aux = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element const*>(params.s_aux_ptr)),
            make_shape(num_heads)
        );
        Tensor sS_aux = make_tensor(
            make_smem_ptr(smem_s_aux),
            Layout<Shape<Int<64>>>{}
        );
        if (tidx < num_heads) {
            sS_aux(tidx) = gS_aux(tidx);
        }
        __syncthreads();
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;
    softmax.skip_softmax_threshold = params.skip_softmax_threshold_scale_factor / binfo.actual_seqlen_k;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        if (tidx == 0) *skip_softmax_vote = 1;
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        bool skip = false;
        if (masking_step == 0) {
            skip = softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        } else {
            skip = softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        }

        if (skip) {
            s_waitcnt<0>();
            if (n_block > n_block_min) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }

            if (n_masking_steps > 1 && n_block <= n_block_min) {
                --n_block;
                break;
            }
            continue;
        }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        if (tidx == 0) *skip_softmax_vote = 1;
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K>(gK, sK, 3, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        bool skip = softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        

        if (skip) {
            s_waitcnt<0>();
            if (n_block > n_block_min) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }
            continue;
        }

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        #if 1
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }
        #endif
    }
    #endif

    // Epilogue
    #if 1
    // ★ Attention Sinks: conditional normalize ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Read from shared memory (already loaded at kernel start) using CuTe Tensor
        Tensor sS_aux = make_tensor(
            make_smem_ptr(smem_s_aux),
            Layout<Shape<Int<64>>>{}
        );
        float s_aux_val = static_cast<float>(sS_aux(bidh));
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 

    if (tidx == 0) {
        uint32_t * gSkipInfo_ptr = reinterpret_cast<uint32_t*>(params.skip_blocks_info_ptr) + (bidb * params.h + bidh) * 2;
        // wangaq debug
        // printf("bidx:%d gSkipInfo_ptr:%d %d total_blocks:%d skipped_blocks:%d\n", blockIdx.x,
        //     *gSkipInfo_ptr, *(gSkipInfo_ptr + 1), softmax.total_blocks, softmax.skipped_blocks);
        atomicAdd(gSkipInfo_ptr, softmax.total_blocks);
        atomicAdd(gSkipInfo_ptr + 1, softmax.skipped_blocks);
    }
    
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64(const Params &params) {
    bool need_switch = Is_causal || (params.seqlen_q>8192&&params.h==40);
    const int m_block = need_switch?blockIdx.z:blockIdx.x;
    const int bidb = need_switch?blockIdx.y:blockIdx.z;
    const int bidh = need_switch?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128) { 
        flash::compute_attn_1rowblock_16x64_mla<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }
    else if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }
    else if constexpr (Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_16x64_dim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch(const Params &params) {
    // const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    // const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    // const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    __syncthreads();
    if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128){
        flash::compute_attn_1rowblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr (Kernel_traits::kHeadDim == 128){
        if constexpr(Kernel_traits::ENABLE_SKIP_SOFTMAX) {
            flash::compute_attn_1rowblock_16x64_prefetch_skip_softmax<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } else {
            if constexpr (Kernel_traits::kLongContext) {
                if(binfo.actual_seqlen_q<64){
                    flash::compute_attn_1rowblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
                }
                else{
                    flash::compute_attn_1rowblock_16x64_prefetch_long_context<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
                }
                
            } else {
                flash::compute_attn_1rowblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
            }
        }
    } else if constexpr (Kernel_traits::kHeadDim == 96){
        flash::compute_attn_1rowblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr(Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if  constexpr(Kernel_traits::kHeadDim == 32) { 
        flash::compute_attn_1rowblock_16x64_dim32_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr(Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        __syncthreads();
    }  else if constexpr(Kernel_traits::kHeadDim == 512) {
        flash::compute_attn_1rowblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        __syncthreads();
    }

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch_fp8(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    __syncthreads();
    #if (defined(__gfx938__)||defined(__gfx92a__))
        if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128){
            flash::compute_attn_1rowblock_16x64_mla_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } else if constexpr (Kernel_traits::kHeadDim == 128){
            flash::compute_attn_1rowblock_16x64_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } 
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch_padding_mask(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb, true);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    flash::compute_attn_1rowblock_16x64_dim64_prefetch_padding_mask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
}

} // namespace flash
