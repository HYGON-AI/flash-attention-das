// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <cstdlib>
#include <type_traits>
#include "flash_fwd_launch_template_pa.h"
#include "flash_singleton.h"
#include "fwd/causal_pair_grid.h"


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout>
__global__ void __launch_bounds__(256, 1) flash_fwd_kernel(Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    flash::compute_attn<Kernel_traits, /*Is_training*/true, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Is_GQA, Layout>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool USE_BSHD_LAYOUT>
__global__ void __launch_bounds__(256, 1) flash_fwd_padding_mask_kernel(Flash_fwd_params params) {
    flash::compute_attn_padding_mask<Kernel_traits, /*Is_training*/true, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, Is_GQA, USE_BSHD_LAYOUT>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool USE_BSHD_LAYOUT>
__global__ void __launch_bounds__(256, 1) flash_fwd_attn_mask_kernel(Flash_fwd_params params) {
    flash::compute_attn_attn_mask<Kernel_traits, /*Is_training*/true, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, Is_GQA, USE_BSHD_LAYOUT>(params);
}

template<typename Kernel_traits>
constexpr int flash_fwd_gfx938_min_blocks_per_cu() {
    return ( Kernel_traits::kHeadDim == 128 && Kernel_traits::kHeadDimV == 128 ? 4 : 1);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, int Layout>
__global__ void __launch_bounds__(256, flash_fwd_gfx938_min_blocks_per_cu<Kernel_traits>()) flash_fwd_kernel_gfx938(Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    flash::compute_attn_gfx938<Kernel_traits, true, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, bool D192MFirst>
__global__ void __launch_bounds__(256, 1) flash_fp8_fwd_kernel_gfx938(
    Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    flash::compute_fp8_attn_gfx938<Kernel_traits, true, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Is_GQA, Layout, D192MFirst>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout>
__global__ void __launch_bounds__(256, 1) flash_fp8_fwd_kernel_gfx92a(Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));
    flash::compute_fp8_attn_gfx92a<Kernel_traits, true, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Is_GQA, Layout>(params);
}

template<typename Kernel_traits,bool Is_Lpt, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Return_softmax, bool Has_alibi, int Layout>
__global__ void __launch_bounds__(256, 1) flash_fwd_kernel_gfx92a(Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));
    flash::compute_attn_gfx92a<Kernel_traits, true, Is_Lpt, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Return_softmax, Has_alibi, Layout>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance    = DeviceProperties<Kernel_traits, FAFUNC::FORWARD>::GetInstance();
    params.cu_count   = instance.cu_count;
    size_t smem_size  = instance.lds_size;
    const bool is_gqa = params.h != params.h_k;
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal);
    int num_m_block;
    num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h * Kernel_traits::SplitD, params.b);
    if (Is_causal) {
        if (is_gqa/*Do_lpt*/) {
            grid = dim3(params.h * Kernel_traits::SplitD, params.b, num_m_block);
        } else /*MHA apply balance*/{
            grid.x = (params.seqlen_q + 2 * Kernel_traits::kBlockM - 1) / (2 * Kernel_traits::kBlockM);
        }
    }

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);

#ifdef BUILD_ASM
    constexpr bool Has_Alibi = false;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_gqa, Is_GQA, [&] {
            constexpr int IsEvenKConst = true; // not used yet
            BOOL_SWITCH(is_swa, Is_local, [&] {
                if constexpr (Is_dropout) {
                    BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                        LAYOUT_SWITCH(params.layout, [&]{
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 256, ReturnSoftmaxConst, Has_Alibi, Is_GQA, Layout>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        });
                    });
                } else {
                    constexpr bool ReturnSoftmaxConst = false;
                    LAYOUT_SWITCH(params.layout, [&]{
                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 256, ReturnSoftmaxConst, Has_Alibi, Is_GQA, Layout>;
                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    });
                }
            });
        });
    });
#else
    BOOL_SWITCH(has_alibi, Has_Alibi, [&]{
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            BOOL_SWITCH(is_gqa, Is_GQA, [&] {
                constexpr int IsEvenKConst = true; // not used yet
                BOOL_SWITCH(is_swa, Is_local, [&] {
                    LAYOUT_SWITCH(params.layout, [&]{
                        auto launch = [&](auto return_softmax_const) {
                            constexpr bool ReturnSoftmaxConst = decltype(return_softmax_const)::value;
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 256, ReturnSoftmaxConst, Has_Alibi, Is_GQA, Layout>;
                            if (smem_size >= 64 * 1024) {
                                HIP_CHECK(hipFuncSetAttribute(
                                    (const void*)(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            }
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        };
                        if constexpr (Is_dropout) {
                            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                                launch(std::integral_constant<bool, ReturnSoftmaxConst>{});
                            });
                        } else {
                            launch(std::false_type{});
                        }
                    });
                });
            });
        });
    });
#endif // end of BUILD_ASM
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_gfx938(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance    = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_Enabled*/>::GetInstance();
    params.cu_count   = instance.cu_count;
    size_t smem_size  = instance.lds_size;
    const bool is_gqa = params.h != params.h_k;
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal);

    int num_m_block   = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h * Kernel_traits::SplitD, params.b);
    if constexpr (Is_causal/*LPT*/) {
        grid = dim3(params.h * Kernel_traits::SplitD, params.b, num_m_block);
    }

    const bool is_varlen  = params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr;
    const bool is_even_MN = !is_varlen && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);
    const bool needs_extra_mask = !is_even_MN && (!is_varlen || params.cu_seqlens_q != params.cu_seqlens_k);

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_varlen, Is_Varlen, [&] { // duplicate instantiation
            constexpr int IsEvenKConst = true; // not used yet
            BOOL_SWITCH(is_swa, Is_local, [&] {
                BOOL_SWITCH(has_alibi, Has_Alibi, [&]{
                    LAYOUT_SWITCH(params.layout, [&]{
                        auto launch = [&](auto return_softmax_const) {
                            constexpr bool ReturnSoftmaxConst = decltype(return_softmax_const)::value;
                            BOOL_SWITCH(needs_extra_mask, NeedsExtraMaskConst, [&] {
                                auto kernel = &flash_fwd_kernel_gfx938<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 256, true/*Is_even_K*/, Is_Varlen, NeedsExtraMaskConst, ReturnSoftmaxConst, Has_Alibi, Layout>;
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            });
                        };
                        if constexpr (Is_dropout) {
                            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                                launch(std::integral_constant<bool, ReturnSoftmaxConst>{});
                            });
                        } else {
                            launch(std::false_type{});
                        }
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fp8_fwd_gfx938(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance    = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_Enabled*/>::GetInstance();
    params.cu_count   = instance.cu_count;
    // D192 MLS uses two overlapping 128-column requests in a 256-column
    // physical LDS layout; allocate that footprint rather than 192 columns.
    constexpr int kMlsLdsHeadDim = Kernel_traits::kHeadDim == 192
        ? 256 : Kernel_traits::kHeadDim;
    const size_t smem_size = Kernel_traits::kNWarps * 32 * kMlsLdsHeadDim
        * sizeof(typename Kernel_traits::Element_k);
    const bool is_gqa = params.h != params.h_k;
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal);
    const bool is_varlen = params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr;

    int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h * Kernel_traits::SplitD, params.b);
    const bool use_d192_m_first = Is_causal
        && !is_gqa
        && Kernel_traits::kHeadDim == 192
        && Kernel_traits::kHeadDimV == 192
        && params.h >= 4
        && params.k_row_stride > Kernel_traits::kHeadDim
        && params.seqlen_q >= 65536;
    if (Is_causal) {
        if (is_gqa
            || ((Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 192)
                && !use_d192_m_first)
            || (is_varlen
                && Kernel_traits::kHeadDim == 256
                && Kernel_traits::kHeadDimV == 256)) {
            grid = dim3(params.h * Kernel_traits::SplitD, params.b, num_m_block);
        } else if constexpr (Kernel_traits::kHeadDim <= 128) {
            // D128 keeps paired rowblocks for triangular load balance.  D256
            // uses one rowblock per CTA to keep the large accumulator live once.
            grid.x = (params.seqlen_q + 2 * Kernel_traits::kBlockM - 1) / (2 * Kernel_traits::kBlockM);
        }
    }

    const bool is_even_MN = params.seqlen_k % Kernel_traits::kBlockN == 0
        && params.seqlen_q % Kernel_traits::kBlockM == 0
        && (!is_varlen || params.b == 1);
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);
    const bool needs_extra_mask = is_varlen
        ? params.cu_seqlens_q != params.cu_seqlens_k
        : !is_even_MN;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_varlen, Is_Varlen, [&] {
            BOOL_SWITCH(is_gqa, Is_GQA, [&] {
                constexpr int IsEvenKConst = true;
                BOOL_SWITCH(is_swa, Is_local, [&] {
                    BOOL_SWITCH(has_alibi, Has_Alibi, [&]{
                        // fwd_base always returns softmax_lse.  FP8 fixed-length
                        // kernels must therefore compute and store LSE even when
                        // dropout softmax probabilities are not requested.
                        constexpr bool StoreLSEConst = true;
                        LAYOUT_SWITCH(params.layout, [&]{
                            BOOL_SWITCH(needs_extra_mask, NeedsExtraMaskConst, [&] {
                                auto launch_kernel = [&](auto d192_m_first) {
                                    constexpr bool D192MFirst = decltype(d192_m_first)::value;
                                    auto kernel = &flash_fp8_fwd_kernel_gfx938<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 256, true/*Is_even_K*/, Is_Varlen, NeedsExtraMaskConst, StoreLSEConst, Has_Alibi, Is_GQA, Layout, D192MFirst>;
                                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                };
                                if constexpr (Is_causal
                                              && !Is_GQA
                                              && Kernel_traits::kHeadDim == 192
                                              && Kernel_traits::kHeadDimV == 192) {
                                    if (use_d192_m_first) {
                                        launch_kernel(std::true_type{});
                                    } else {
                                        launch_kernel(std::false_type{});
                                    }
                                } else {
                                    launch_kernel(std::false_type{});
                                }
                            });
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fp8_fwd_gfx92a(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance    = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_Enabled*/>::GetInstance();
    constexpr size_t smem_size = 32 * 1024;
    params.cu_count   = instance.cu_count;
    const bool is_gqa = params.h != params.h_k;
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal);

    int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h * Kernel_traits::SplitD, params.b);
    if (Is_causal) {
        if (is_gqa) {
            grid = dim3(params.h * Kernel_traits::SplitD, params.b, num_m_block);
        } else {
            grid.x = (params.seqlen_q + 2 * Kernel_traits::kBlockM - 1) / (2 * Kernel_traits::kBlockM);
        }
    }

    const bool is_varlen  = params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr;
    const bool is_even_MN = params.seqlen_k % Kernel_traits::kBlockN == 0
        && params.seqlen_q % Kernel_traits::kBlockM == 0
        && (!is_varlen || params.b == 1);
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);
    // 同 gfx938: causal 且 seqlen_k - seqlen_q 可能不是 kBlockN 整数倍时多 mask 一个 N tile.
    // varlen 且 cu_seqlens_q == cu_seqlens_k (自注意力, 每条 seqlen_q == seqlen_k) 时不需要.
    const bool needs_extra_mask = Is_causal && !is_even_MN && (!is_varlen || params.cu_seqlens_q != params.cu_seqlens_k);

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_varlen, Is_Varlen, [&] {
            BOOL_SWITCH(is_gqa, Is_GQA, [&] {
                constexpr int IsEvenKConst = true;
                BOOL_SWITCH(is_swa, Is_local, [&] {
                    BOOL_SWITCH(has_alibi, Has_Alibi, [&]{
                        constexpr bool ReturnSoftmaxConst = false;
                        LAYOUT_SWITCH(params.layout, [&]{
                            BOOL_SWITCH(needs_extra_mask, NeedsExtraMaskConst, [&] {
                                auto kernel = &flash_fp8_fwd_kernel_gfx92a<Kernel_traits, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 256, true/*Is_even_K*/, Is_Varlen, Is_causal && NeedsExtraMaskConst, ReturnSoftmaxConst && Is_dropout, Has_Alibi, Is_GQA, Layout>;
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            });
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_gfx92a(Flash_fwd_params &params, hipStream_t stream) {
    auto& instance    = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_Enabled*/>::GetInstance();
    params.cu_count   = instance.cu_count;
    size_t smem_size  = instance.lds_size;
    const char* fa_debug = std::getenv("FA_DEBUG");
    const bool do_fa_debug = fa_debug != nullptr;
    if (do_fa_debug) {
        printf("[gfx92a launch] gcn_arch=%d cu_count=%d smem_size=%zu q_smem=%zu k_smem=%zu v_smem=%zu seqlen_q=%d seqlen_k=%d h=%d b=%d causal=%d layout=%d\n",
               instance.gcn_arch, params.cu_count, smem_size,
               Kernel_traits::q_smem_size, Kernel_traits::k_smem_size, Kernel_traits::v_smem_size,
               params.seqlen_q, params.seqlen_k, params.h, params.b,
               static_cast<int>(Is_causal), params.layout);
    }
    const bool is_swa = ((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal);

    int num_m_block   = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int n_heads_grid = params.h * Kernel_traits::SplitD;
    dim3 grid(num_m_block, n_heads_grid, params.b);

    CausalPairPlan plan{false, 0, 0, 0, 0};
    if constexpr (Is_causal) {
        // 1) 只认 128x128 这套 tile
        constexpr bool kPairHeaddim =
            (Kernel_traits::kHeadDim == 128 && Kernel_traits::kHeadDimV == 128 &&
             Kernel_traits::kBlockM  == 128 && Kernel_traits::kBlockN  == 128);

        if constexpr (kPairHeaddim) {
            const int m_eff = (params.seqlen_q + Kernel_traits::kBlockM / 2) / Kernel_traits::kBlockM;  // 四舍五入

            // 2) 只放开: 等长 + BSHD(layout 1) + 连续 KV(非 paged)
            const bool is_varlen_gate = (params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr);
            const bool load_uniform = !is_varlen_gate
                    || ((long long)params.total_q == (long long)params.b * params.seqlen_q &&
                        (long long)params.total_k == (long long)params.b * params.seqlen_k);
            const bool pair_path_ok = load_uniform &&
                    (params.layout == 1) && (params.block_table == nullptr);

            if (pair_path_ok) {
                plan = causal_pair_grid_enabled(
                    kCausalPairCfg_gfx92a_128x128, params.h, n_heads_grid, params.h_k, params.b,
                    num_m_block, m_eff, params.cu_count, kGfx92aFwdCtasPerCu);
            }

            if (plan.use_pair) {
                grid = dim3((unsigned)(plan.n_pair + plan.dissolve_pairs), 1, 1);   // 1D 配对网格
            } else {
                grid = dim3(n_heads_grid, params.b, num_m_block);                   // 3D LPT 网格
            }
        } else {
            grid = dim3(n_heads_grid, params.b, num_m_block);                       // 其它 tile: LPT
        }
    }

    const bool is_varlen  = params.cu_seqlens_q != nullptr && params.cu_seqlens_k != nullptr;
    const bool is_even_MN = !is_varlen && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        constexpr int IsEvenKConst = true;
        BOOL_SWITCH(is_varlen, Is_Varlen, [&] {
            BOOL_SWITCH(is_swa, Is_local, [&] {
                BOOL_SWITCH(has_alibi, Has_Alibi, [&]{
                    constexpr bool ReturnSoftmaxConst = false;
                    BOOL_SWITCH(!plan.use_pair, Is_Lpt, [&]{
                        LAYOUT_SWITCH(params.layout, [&]{
                            auto kernel = &flash_fwd_kernel_gfx92a<Kernel_traits, Is_Lpt, Is_dropout, Is_causal, Is_local&&!Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 256, true/*Is_even_K*/, Is_Varlen, ReturnSoftmaxConst && Is_dropout, Has_Alibi, Layout>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        });
                    });
                });
            });
        });
    });
}



template<typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 32;
    int seq_len = params.seqlen_q;
    if (params.p_dropout < 1.f) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 32, 32, 32, 32, 1, false, false, T>, true, Is_causal>(params, stream);
        });
    } else {
        constexpr bool Is_dropout = false;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if(seq_len < 64)
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim,  32, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
            else if(seq_len < 128)
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
            else if(seq_len < 256)
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
            else
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
#endif
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 64;
    int seq_len = params.seqlen_q;
    if (params.p_dropout < 1.f) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 128, 32, 32, 32, 2, false, false, T>, true, Is_causal>(params, stream);
        });
    } else {
        constexpr bool Is_dropout = false;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
#endif
}

template<typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 96;
    bool is_sm8x = 1;
    if (params.p_dropout < 1.f) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, true, Is_causal>(params, stream);
        });
    } else {
        constexpr bool Is_dropout = false;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    }
#endif
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int Headdim = 128;
    int seq_len = params.seqlen_q;
    if (params.p_dropout < 1.f) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, true, Is_causal>(params, stream);
        });
    } else {
        constexpr bool Is_dropout = false;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // if arch >= 938, new MLS is allowed
            int gcn_arch = getArch();
            if (gcn_arch == 930 and std::getenv("FA_FWD_NO_MLS") == nullptr) {
                run_flash_fwd_gfx92a<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else if (gcn_arch >= 938 and std::getenv("FA_FWD_NO_MLS") == nullptr) {
                if (params.qkvheaddim_compute == 96) {
                    if (params.qkvheaddim_tail_tile16 == 1)
                        run_flash_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T, T, T, 64, 96, 96, 1>, Is_dropout, Is_causal>(params, stream);
                    else
                        run_flash_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T, T, T, 64, 96, 96, 2>, Is_dropout, Is_causal>(params, stream);
                }
                else
                    run_flash_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                // since double prefetch is applied in QK,PV gemm, and thus, BLOCK_N must >= 64
                if (seq_len < 32)
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 32, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
                else if (seq_len < 64)
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
                else if(seq_len < 128)
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
                else
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    }
}

template<typename T>
void run_fp8_mha_fwd_hdim128(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int Headdim = 128;
    constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        int gcn_arch = getArch();
        if (gcn_arch == 930) {
            run_flash_fp8_fwd_gfx92a<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
        } else if (gcn_arch >= 938) {
            run_flash_fp8_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
        } else {
            printf("\x1b[31mfp8 is not supported in this arch!\033[0m\n");
        }
    });
}

#ifndef HEADDIM_128_ONLY
template<typename T>
void run_fp8_mha_fwd_hdim192(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int Headdim = 192;
    constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (getArch() >= 938) {
            if constexpr (Is_causal) {
                // The smaller N tile keeps the D192 FP32 row sum out of excessive spills.
                run_flash_fp8_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fp8_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            printf("fp8 D192 forward is not supported in this arch!\n");
        }
    });
}

template<typename T>
void run_fp8_mha_fwd_qkhdim192_vhdim128(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int HeaddimQK = 192;
    constexpr int HeaddimV  = 128;
    constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (getArch() >= 938) {
            run_flash_fp8_fwd_gfx938<Flash_fwd_kernel_traits<HeaddimQK, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
        } else {
            printf("fp8 QK192/V128 forward is not supported in this arch!\n");
        }
    });
}

template<typename T>
void run_fp8_mha_fwd_hdim256(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int Headdim = 256;
    constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        int gcn_arch = getArch();
        if (gcn_arch >= 938) {
            run_flash_fp8_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_dropout, Is_causal>(params, stream);
        } else {
            printf("\x1b[31mfp8 is not supported in this arch!\033[0m\n");
        }
    });
}
#endif


template<typename T>
void run_mha_fwd_hdim160(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 160;
    bool is_sm8x = 1;
    if (params.p_dropout < 1.f) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, true, Is_causal>(params, stream);
        });
    } else {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, Is_causal ? 64: 128, 64, 32, 32, 32, 2, false, false, T>, false, Is_causal>(params, stream);
        });
    }
#endif
}

template<typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 192;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            int gcn_arch = getArch();
            if (gcn_arch == 930 and not Is_dropout and std::getenv("FA_FWD_NO_MLS") == nullptr) {
                run_flash_fwd_gfx92a<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else if (gcn_arch >= 938 and std::getenv("FA_FWD_NO_MLS") == nullptr) {
                run_flash_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                if (params.seqlen_q < 128)
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
                else
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
#endif
}

template<typename T>
void run_mha_fwd_qkhdim192_vhdim128(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int HeaddimQK = 192;
    constexpr int HeaddimV  = 128;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        int gcn_arch = getArch();
        if (gcn_arch == 930 and std::getenv("FA_FWD_NO_MLS") == nullptr) {
            run_flash_fwd_gfx92a<Flash_fwd_kernel_traits<HeaddimQK, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, false/*dropout*/, Is_causal>(params, stream);
        } else if (gcn_arch >= 938 and std::getenv("FA_FWD_NO_MLS") == nullptr) {
            run_flash_fwd_gfx938<Flash_fwd_kernel_traits<HeaddimQK, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, false/*dropout*/, Is_causal>(params, stream);
        } else {
            if (params.seqlen_q < 128)
                run_flash_fwd<Flash_fwd_kernel_traits<HeaddimQK, HeaddimV, 64, 32, 32, 32, 32, 2, false, false, T>, false/*dropout*/, Is_causal>(params, stream);
            else
                run_flash_fwd<Flash_fwd_kernel_traits<HeaddimQK, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, false/*dropout*/, Is_causal>(params, stream);
        }
    });
#endif
}

template<typename T>
void run_mha_fwd_hdim224(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 224;
    int device;
    HIP_CHECK(hipGetDevice(&device));
    int max_smem_per_block;
    hipError_t status_ = hipDeviceGetAttribute(
        &max_smem_per_block, hipDeviceAttributeSharedMemPerBlockOptin, device);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
#endif
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, hipStream_t stream) {
#ifndef HEADDIM_128_ONLY
    constexpr int Headdim = 256;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream); // exp intrinsic lead to unstable results
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 32, 32, 32, 32, 32, 1, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
#endif
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_padding_mask_launcher(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance   = DeviceProperties<Kernel_traits, FAFUNC::FORWARD>::GetInstance();
    size_t smem_size = instance.lds_size;
    int num_m_block  = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);

    constexpr bool IsEvenMNConst = false;
    BOOL_SWITCH(params.h != params.h_k, Is_GQA, [&] {
        BOOL_SWITCH(params.layout, USE_BSHD_LAYOUT, [&] {
            auto kernel = &flash_fwd_padding_mask_kernel<Kernel_traits, false/*dropout*/, Is_causal, false/*Is_local*/, IsEvenMNConst, true, false/*return softmax*/, false/*Has_Alibi*/, Is_GQA, USE_BSHD_LAYOUT>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        });
    });
}


template<typename T, int Headdim>
void run_flash_fwd_padding_mask(Flash_fwd_params &params, hipStream_t stream) {
    int seq_len = params.seqlen_q;
    // causal / 非 causal 两套在此展开（写法与 run_mha_fwd_hdimXX 一致）。
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (seq_len < 32)
            run_flash_fwd_padding_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 32, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        else if (seq_len < 64)
            run_flash_fwd_padding_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        else if(seq_len < 128)
            run_flash_fwd_padding_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        else
            run_flash_fwd_padding_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
    });
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_attn_mask_launcher(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance   = DeviceProperties<Kernel_traits, FAFUNC::FORWARD>::GetInstance();
    size_t smem_size = instance.lds_size;
    int num_m_block  = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);

    constexpr bool IsEvenMNConst = false; // not sure
    BOOL_SWITCH(params.h != params.h_k, Is_GQA, [&] {
        BOOL_SWITCH(params.layout, USE_BSHD_LAYOUT, [&] {
            auto kernel = &flash_fwd_attn_mask_kernel<Kernel_traits, false/*dropout*/, Is_causal, false/*Is_local*/, IsEvenMNConst, true, false/*return softmax*/, false/*Has_Alibi*/, Is_GQA, USE_BSHD_LAYOUT>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        });
    });
}


template<typename T, int Headdim>
void run_flash_fwd_attn_mask(Flash_fwd_params &params, hipStream_t stream) {
    constexpr bool Is_causal = false;
    int seq_len = params.seqlen_q;
    if (seq_len < 32)
        run_flash_fwd_attn_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 32, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
    else if (seq_len < 64)
        run_flash_fwd_attn_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
    else if(seq_len < 128)
        run_flash_fwd_attn_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
    else
        run_flash_fwd_attn_mask_launcher<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_prefix_prefill_launcher(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance   = DeviceProperties<Kernel_traits, FAFUNC::FORWARD>::GetInstance();
    size_t smem_size = instance.lds_size;
    int num_m_block  = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    constexpr bool IsEvenMNConst = false;
    LAYOUT_SWITCH(params.layout, [&] {
        // 同 run_mha_fwd* 的 is_swa 口径: wl == 0 也是真窗口 (只留 delta == 0), 旧判据 wl > 0 会算成全量。
        BOOL_SWITCH((params.window_size_left >= 0 or params.window_size_right >= 0) and !Is_causal, Is_local, [&] {
            dim3 grid = Layout == 0
                ? dim3(num_m_block * params.h_h_k_ratio, params.h_k, params.b)
                : dim3(params.h, params.b, num_m_block);
            auto kernel = &flash_fwd_prefix_prefill_kernel<Kernel_traits, false/*dropout*/, Is_causal, Is_local, IsEvenMNConst, false/*return softmax*/, false/*Has_Alibi*/, false/*Is_GQA*/, Layout, Flash_fwd_params>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_mixed_fwd_prefix_prefill_launcher(Flash_fwd_params &params, hipStream_t stream) {
    auto& instance = DeviceProperties<Kernel_traits, FAFUNC::FORWARD>::GetInstance();
    const size_t smem_size = instance.lds_size;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    constexpr bool IsEvenMNConst = false;
    LAYOUT_SWITCH(params.layout, [&] {
        // 同 run_flash_fwd_prefix_prefill_launcher。
        BOOL_SWITCH((params.window_size_left >= 0 or params.window_size_right >= 0) and !Is_causal, Is_local, [&] {
            const dim3 grid = Layout == 0 ? dim3(num_m_block * params.h_h_k_ratio, params.h_k, params.b)
                                         : dim3(params.h, params.b, num_m_block);
            auto kernel = &flash::flash_mixed_fwd_prefix_prefill_kernel<
                Kernel_traits, false /*dropout*/, Is_causal, Is_local, IsEvenMNConst, false /*return softmax*/,
                false /*Has_Alibi*/, false /*Is_GQA*/, Layout, Flash_fwd_params>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        });
    });
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_prefix_prefill_gfx938_launcher(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance   = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_enabled*/>::GetInstance();
    size_t smem_size = instance.lds_size;
    int num_m_block  = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(params.h, params.b, num_m_block);

    constexpr bool IsEvenMNConst = false;
    const bool is_local = !Is_causal && params.window_size_left > 0 && params.window_size_right >= 0;
    BOOL_SWITCH(is_local, Is_local, [&] {
        auto kernel = &flash_fwd_prefix_prefill_gfx938_kernel<Kernel_traits, false, false/*dropout*/, Is_causal, Is_local, IsEvenMNConst, true, false/*return softmax*/, false/*Has_Alibi*/, 1/*layout*/, Flash_fwd_params>;
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_prefix_prefill_gfx92a_launcher(Flash_fwd_params &params, hipStream_t stream) {

    auto& instance   = DeviceProperties<Kernel_traits, FAFUNC::FORWARD, true/*MLS_enabled*/>::GetInstance();
    size_t smem_size = instance.lds_size;
    int num_m_block  = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(params.h, params.b, num_m_block);

    constexpr bool IsEvenMNConst = false;
    // 同普通 forward 的 is_swa 口径 (wl == 0 也算真窗口; 旧判据 wl > 0 会把 (0,*) 算成全量)。
    // gfx938 分支 (:745) 仍是 wl > 0: 它的 rest loop 没收窄, 放开 (0,*) 会越界读 block_table。
    const bool is_local = !Is_causal && (params.window_size_left >= 0 || params.window_size_right >= 0);
    BOOL_SWITCH(is_local, Is_local, [&] {
        auto kernel = &flash_fwd_prefix_prefill_gfx92a_kernel<Kernel_traits, false, false/*dropout*/, Is_causal, Is_local, IsEvenMNConst, true, false/*return softmax*/, false/*HasAlibi*/, 1/*layout*/, Flash_fwd_params>;
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
}


template<typename T, int Headdim, int HeaddimV>
void run_flash_fwd_prefix_prefill(Flash_fwd_params &params, hipStream_t stream) {
    // is_causal = false, used in cascade attention
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        const bool use_mls_prefix = std::getenv("FA_FWD_NO_MLS") == nullptr && ((Headdim == 128 and HeaddimV == 128) or (Headdim == 192 and HeaddimV == 128));
        const bool use_mls_prefix_92a = std::getenv("FA_FWD_NO_MLS") == nullptr && ((Headdim == 128 and HeaddimV == 128)
                                                                               or (Headdim == 192 and HeaddimV == 128)
                                                                               or (Headdim == 192 and HeaddimV == 192));
        const int gcn_arch = getArch();
        if (gcn_arch == 930 and use_mls_prefix_92a) {
            if constexpr (Headdim == 192)
                run_flash_fwd_prefix_prefill_gfx92a_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else
                run_flash_fwd_prefix_prefill_gfx92a_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        } else if (gcn_arch >= 938 and use_mls_prefix) {
            if constexpr (Headdim == 192)
                run_flash_fwd_prefix_prefill_gfx938_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else if (params.page_block_size == 64)
                run_flash_fwd_prefix_prefill_gfx938_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else
                run_flash_fwd_prefix_prefill_gfx938_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        } else {
            int seq_len = params.seqlen_q;
            if (seq_len < 32)
                run_flash_fwd_prefix_prefill_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else if (seq_len < 64 or Headdim >= 256)
                run_flash_fwd_prefix_prefill_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 64, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else if (seq_len < 128 or params.page_block_size == 64)
                run_flash_fwd_prefix_prefill_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
            else
                run_flash_fwd_prefix_prefill_launcher<Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T>, Is_causal>(params, stream);
        }
    });
}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fp8_fwd_prefix_prefill_launcher_gfx938(Flash_fwd_params &params, hipStream_t stream) {
    constexpr bool NeedsWideFp8MlsLds = Kernel_traits::kHeadDim > 128 || Kernel_traits::kHeadDimV > 128;
    size_t smem_size = NeedsWideFp8MlsLds ? 32 * 1024 : 16 * 1024;
    int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    dim3 grid = Is_causal ? dim3(params.h, params.b, num_m_block)
                          : dim3(num_m_block, params.h, params.b);

    constexpr bool Has_Alibi = false;
    const bool is_local = !Is_causal && params.window_size_left > 0 && params.window_size_right >= 0;
    BOOL_SWITCH(params.softmax_lse_ptr != nullptr, ReturnSoftmaxConst, [&] {
        BOOL_SWITCH(is_local, IsLocalConst, [&] {
            LAYOUT_SWITCH(params.layout, [&]{
                auto launch = [&](auto assume_no_empty_causal_rows) {
                    constexpr bool AssumeNoEmptyCausalRows = decltype(assume_no_empty_causal_rows)::value;
                    auto kernel = &flash_fp8_fwd_prefix_prefill_kernel_gfx938<Kernel_traits, true/*Is_training*/, false/*Is_dropout*/, Is_causal, IsLocalConst, true/*Is_even_K*/, ReturnSoftmaxConst, Has_Alibi, Layout, AssumeNoEmptyCausalRows>;
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                };
                if constexpr (Is_causal && Kernel_traits::kHeadDim == 192) {
                    // For a single sequence with K at least as long as Q, every
                    // causal query row has a visible key.  Keep the fast softmax
                    // specialization while retaining the guarded variant for
                    // unequal or batched varlen requests.
                    if (params.b == 1 && params.seqlen_q <= params.seqlen_k) {
                        launch(std::true_type{});
                    } else {
                        launch(std::false_type{});
                    }
                } else {
                    launch(std::false_type{});
                }
            });
        });
    });
}

template<typename Kernel_traits>
void run_flash_fp8_fwd_prefix_prefill_launcher_gfx92a(Flash_fwd_params &params, hipStream_t stream) {

    constexpr size_t smem_size = 32 * 1024;
    int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    dim3 grid(params.h, params.b, num_m_block);

    constexpr bool Is_local = false;
    constexpr bool ReturnSoftmaxConst = false;
    constexpr bool Has_Alibi = false;
    LAYOUT_SWITCH(params.layout, [&]{
        auto kernel = &flash_fp8_fwd_prefix_prefill_kernel_gfx92a<Kernel_traits, true/*Is_training*/, false/*Is_dropout*/, true/*Is_causal*/, Is_local, true/*Is_even_K*/, ReturnSoftmaxConst, Has_Alibi, Layout>;
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
}

template<typename T, int Headdim, int HeaddimV>
void run_fp8_flash_fwd_prefix_prefill(Flash_fwd_params &params, hipStream_t stream) {
    int gcn_arch = getArch();
    if constexpr (Headdim == 128 && HeaddimV == 128) {
        if (gcn_arch == 930) {
            run_flash_fp8_fwd_prefix_prefill_launcher_gfx92a<
                Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>>(params, stream);
            return;
        }
    }
    if (gcn_arch == 936 && params.is_e5m2) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            const int seq_len = params.seqlen_q;
            if (seq_len < 32) {
                run_flash_mixed_fwd_prefix_prefill_launcher<
                    Flash_fp8_bf16_fwd_kernel_traits<Headdim, HeaddimV, 32, 64, 32, 32, 32, 2, T>, Is_causal>(
                    params, stream);
            } else if (seq_len < 64 or Headdim >= 256) {
                run_flash_mixed_fwd_prefix_prefill_launcher<
                    Flash_fp8_bf16_fwd_kernel_traits<Headdim, HeaddimV, 64, 64, 32, 32, 32, 2, T>, Is_causal>(
                    params, stream);
            } else if (seq_len < 128 or params.page_block_size == 64) {
                run_flash_mixed_fwd_prefix_prefill_launcher<
                    Flash_fp8_bf16_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, T>, Is_causal>(
                    params, stream);
            } else {
                run_flash_mixed_fwd_prefix_prefill_launcher<
                    Flash_fp8_bf16_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, T>, Is_causal>(
                    params, stream);
            }
        });
        return;
    }
    if (gcn_arch >= 938) {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr (Headdim == 192) {
                // D192 Q/K uses its dedicated K layout independently of V width.
                // B64 loaders require four waves even for short query sequences.
                if (params.page_block_size == 64 || (Is_causal && HeaddimV == 192)) {
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                } else {
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                }
            } else if (params.page_block_size == 64) {
                if (params.seqlen_q <= 128) {
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 64, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                } else {
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                }
            } else {
                if constexpr (Headdim == 256 && HeaddimV == 256) {
                    // Keep the D256 kernel on B64 and consume each 128-token page in two tiles.
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 64, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                } else {
                    run_flash_fp8_fwd_prefix_prefill_launcher_gfx938<
                        Flash_fwd_kernel_traits<Headdim, HeaddimV, 128, 128, 32, 32, 32, 2, false, false, T, Float16, fp8_e4m3>, Is_causal>(params, stream);
                }
            }
        });
    } else {
        printf("\x1b[31mfp8 prefix_prefill is not supported in this arch!\033[0m\n");
    }
}
