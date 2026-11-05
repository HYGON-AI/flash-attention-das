/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/GradMode.h>
#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#if defined(HCU_ASM)
    #define ARCH_SUPPORTS_FLASH
#endif

#if (defined(__gfx936__)||defined(__gfx938__)||defined(__gfx92a__))
    #define ARCH_AFTER_936
#endif

#if (defined(__gfx938__)||defined(__gfx92a__))
    #define ARCH_AFTER_938
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

static const bool prefetch = get_env_("FLASH_ATTENTION_FWD_PREFETCH");
static const int small_seq = get_env_("FLASH_ATTENTION_SMALL_SEQ");
static const bool use_fast_softmax = get_env_("FLASH_ATTENTION_FAST_SOFTMAX");
static const bool is_training = at::GradMode::is_enabled();

static const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__launch_bounds__(Kernel_traits::kNThreads) \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        #if defined(ARCH_AFTER_936)
            if constexpr ((Kernel_traits::kHeadDim>=64&&Kernel_traits::kHeadDim<128)||Kernel_traits::kHeadDim%128==0){/*do nothing*/}
            else flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
        #else
            flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
        #endif
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(__gfx928__)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_AFTER_936)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch_fp8, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_AFTER_938)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch_padding_mask, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_AFTER_936)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch_padding_mask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if (Is_causal||(params.seqlen_q>8192&&params.h==40))grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if (Is_causal||(params.seqlen_q>8192&&params.h==40))grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel_16x64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            // constexpr static bool IsEvenKConst = true;
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                // constexpr static bool Is_local = false;
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    // constexpr static bool ReturnSoftmaxConst = true;
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        // constexpr static bool Has_alibi = false;
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // constexpr static bool Is_softcap = false;
                            auto kernel = &flash_fwd_kernel_16x64_prefetch<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    // printf("%d,%d\n",Is_causal, is_even_MN);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        auto kernel = &flash_fwd_kernel_16x64_prefetch_fp8<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                                // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}





template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch_padding_mask(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    // BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        constexpr static bool IsEvenMNConst = false;
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel_16x64_prefetch_padding_mask<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    // });
}


template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938" || get_device_name() == "gfx92a"&& !Is_dropout && !params.s_aux_ptr && params.d == 32 && params.seqlen_q > 64) {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 256, 128, 4, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int seq=small_seq;
    if(seq==1)seq=128;
    if(seq>0&&params.seqlen_q>seq)params.seqlen_q=seq;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count||seq>0) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 64, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, Is_dropout ? 128 : 256, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 256, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
template<typename T, bool Is_causal>
void run_mha_fwd_padding_mask_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a")  {
            run_flash_fwd_16x64_prefetch_padding_mask<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 128, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 128, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if(params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count){
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim96<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            }
            else{
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim96<Headdim, 128, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal, bool Is_skip_softmax = false>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // printf("run_mha_fwd_hdim128\n");
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 64, 64, 4, T, 3, Is_skip_softmax, /*kLongContext_=*/false>, Is_dropout, Is_causal>(params, stream);
            } else if((get_device_name() == "gfx936" || get_device_name() == "gfx92a") && params.seqlen_k >= 1024 && (use_fast_softmax||(Is_causal&&!is_training))) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 128, 64, 4, T, 3, Is_skip_softmax, /*kLongContext_=*/true>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 128, 64, 4, T, 3, Is_skip_softmax, /*kLongContext_=*/false>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 128, 64, 4, /*Is_Q_use_smem_=*/true, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    static constexpr bool Is_FP8 = hute::is_same_v<T, hytlass::float_e4m3_t> || hute::is_same_v<T, hytlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, hytlass::bfloat16_t>;
    // printf("run_mha_fwd_hdim128\n");
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count) {
                run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 64, 64, 4, T,T_out, 3>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 128, 64, 4, T,T_out, 3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            printf("this device is not supoort fp8");
        }
    });
}


template<typename T, bool Is_causal>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_mla_traits</*Headdim*/192, 128, 64, 4, T, 3, /*HeaddimV*/128>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits_MLA</*Headdim*/192, 128, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/true, T, 128>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {

    static constexpr bool Is_FP8 = hute::is_same_v<T, hytlass::float_e4m3_t> || hute::is_same_v<T, hytlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, hytlass::bfloat16_t>;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_mla_traits_fp8</*Headdim*/192, 128, 64, 4, T,T_out, 3, /*HeaddimV*/128>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            printf("this device is not supoort fp8");
        }
    });

}
template<typename T, bool Is_causal>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // constexpr static int Is_dropout = false;
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            const bool use_gfx936_d256_contiguous_varlen =
                (get_device_name() == "gfx936"||get_device_name() == "gfx938")
                && params.block_table == nullptr
                && params.cu_seqlens_q != nullptr
                && params.cu_seqlens_k != nullptr
                && params.d == Headdim
                && params.d_value == Headdim;
            if (use_gfx936_d256_contiguous_varlen && params.seqlen_q <= 64) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim256<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            } else if (use_gfx936_d256_contiguous_varlen) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_gfx936_d256_pipeline_traits<T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim256<Headdim, 128, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim512(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 512;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // constexpr static int Is_dropout = false;
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim512<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
