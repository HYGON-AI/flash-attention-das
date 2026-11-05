// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "flash_fwd_launch_template.h"
#include "flash_fwd_sparse_kernel.h"
#include "flash_sparse.h"

// namespace FLASH_NAMESPACE {

// A separate translation unit confines compiler scheduling to this exact instance.
void run_flash_sparse_fwd_hcu_bf16_exact(Flash_fwd_params_sparse &params, cudaStream_t stream);

#define DEFINE_FLASH_FORWARD_SPARSE_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params_sparse params)

DEFINE_FLASH_FORWARD_SPARSE_KERNEL(flash_fwd_sparse_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, bool Use_hcu_optimizations = false) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_sparse_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Use_hcu_optimizations>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

#if 1
DEFINE_FLASH_FORWARD_SPARSE_KERNEL(flash_fwd_sparse_sla_kernel, bool Is_even_MN, bool Is_even_K, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_sparse_attn_sla<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
#endif

#if 1
DEFINE_FLASH_FORWARD_SPARSE_KERNEL(flash_fwd_sparse_sla_fp8_kernel, bool Is_even_MN, bool Is_even_K, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_sparse_attn_sla_fp8<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
#endif

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    constexpr bool CanOptimize =
                        (std::is_same<typename Kernel_traits::Element, hytlass::bfloat16_t>::value
                         || std::is_same<typename Kernel_traits::Element, hytlass::half_t>::value)
                        && Kernel_traits::kHeadDim == 128 && IsEvenKConst
                        && !Is_dropout && !Is_causal && !Has_alibi && !Is_softcap
                        && !ReturnSoftmaxConst;
                    if constexpr (CanOptimize) {
                        // Low-occupancy calls retain the original kernel, avoiding setup overhead.
                        const int64_t num_tasks = int64_t(num_m_block) * params.h * params.b;
                        if (get_device_name() == "gfx936" && !params.enable_dynamic_skip
                            && num_tasks >= 4 * sm_count) {
                            using HcuExactTraits = Flash_fwd_kernel_16x64_prefetch_traits<
                                128, 64, 64, 4, hytlass::bfloat16_t, 3>;
                            if constexpr (std::is_same<Kernel_traits, HcuExactTraits>::value) {
                                run_flash_sparse_fwd_hcu_bf16_exact(params, stream);
                                return;
                            } else {
                                kernel = &flash_fwd_sparse_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap, true>;
                            }
                        }
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

#if 1
template<typename Kernel_traits>
void run_flash_sparse_sla_fwd(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            constexpr bool IsEvenMNConst = false;
            auto kernel = &flash_fwd_sparse_sla_kernel<Kernel_traits, IsEvenMNConst, IsEvenKConst, ReturnSoftmaxConst>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}
#endif

#if 1
template<typename Kernel_traits>
void run_flash_sparse_sla_fwd_fp8(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            constexpr bool IsEvenMNConst = false;
            auto kernel = &flash_fwd_sparse_sla_fp8_kernel<Kernel_traits, IsEvenMNConst, IsEvenKConst, ReturnSoftmaxConst>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}
#endif

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim32(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim64(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_sparse_sla_hdim64(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
        if (params.seqlen_q <= 2048)
            run_flash_sparse_sla_fwd<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 64, 64, 4, T>>(params, stream);
        else
            run_flash_sparse_sla_fwd<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 128, 64, 4, T>>(params, stream);
    }
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim96(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim128(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_sparse_fwd<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_sparse_sla_hdim128(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
        if (params.seqlen_q <= 2048)
            run_flash_sparse_sla_fwd<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 64, 64, 4, T, 3>>(params, stream);
        else
            run_flash_sparse_sla_fwd<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 128, 64, 4, T, 3>>(params, stream);
    }
}

template<typename T>
void run_mha_fwd_sparse_sla_hdim128_fp8(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    static constexpr bool Is_FP8 = hute::is_same_v<T, hytlass::float_e4m3_t> || hute::is_same_v<T, hytlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, hytlass::bfloat16_t>;
    if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
        // int num_blocks_64 = params.h * params.b * ((params.seqlen_q + 64 - 1) / 64);//3
        // int num_blocks_128 = params.h * params.b * ((params.seqlen_q + 128 - 1) / 128);//2
        // if ((num_blocks_64 <= sm_count || (num_blocks_128 / sm_count == 1 && num_blocks_128 % sm_count > 1 && (num_blocks_64 + sm_count - 1) / sm_count <= 3) || force_blockm64) && !force_blockm128) {
        if (params.seqlen_q <= 2048) {
            run_flash_sparse_sla_fwd_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 64, 64, 4, T,T_out, 3>>(params, stream);
        } else {
            run_flash_sparse_sla_fwd_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 128, 64, 4, T,T_out, 3>>(params, stream);
        }
    } else {
        printf("this device is not supoort fp8");
    }
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim160(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim192(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim224(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim256(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

// } // namespace FLASH_NAMESPACE
