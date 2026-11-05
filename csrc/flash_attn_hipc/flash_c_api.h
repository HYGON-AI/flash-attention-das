// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "numeric_types.h"
#include "config.h"
#include "static_switch.h"
#include "flash_singleton.h"
#include "flash_memory.h"


// When defined, export declarations only; definitions come from the core.
// Undefined keeps the original header-as-implementation behavior.
#ifdef FLASH_C_API_EXTERN_ONLY

void run_mha_fwd(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel=false);
void run_mha_bwd(Flash_bwd_params &params, hipStream_t stream, const bool configure=false);
void run_mha_fwd_kvcache(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel=false);
// Dispatch hook owned by the core.
extern void (*run_mha_fwd_c)(Flash_fwd_params&, hipStream_t, bool);

#else  // !FLASH_C_API_EXTERN_ONLY

// ====================================================================================================================================
//                                                               FWD
// ====================================================================================================================================
void run_mha_fwd(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel=false) {
#if defined(BUILD_FA_FWD)
    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "5") == 0) return;
        else if (std::strcmp(fa_debug, "C") == 0) {PRINT_PARAMS}; // for c interface debug
    }
    if (params.seqused_k != nullptr) {
        // Prefix prefill attention
        if (params.is_e4m3 || params.is_e5m2) {
            // FP8 prefix prefill
            FP16_SWITCH(!params.is_bf16, [&] {
                if (params.d == 128 and params.d_value == 128) {
                    run_fp8_mha_fwd_prefix_prefill_<elem_type, 128, 128>(params, stream);
                } else if (params.d == 192 and params.d_value == 128) {
                    run_fp8_mha_fwd_prefix_prefill_<elem_type, 192, 128>(params, stream);
                } else if (params.d == 192 and params.d_value == 192) {
                    run_fp8_mha_fwd_prefix_prefill_<elem_type, 192, 192>(params, stream);
                } else if (params.d == 256 and params.d_value == 256) {
                    run_fp8_mha_fwd_prefix_prefill_<elem_type, 256, 256>(params, stream);
                } else {
                    assert(false && "FP8 prefix prefill only supports head_dim=128/128, 192/128, 192/192, or 256/256");
                }
            });
        } else{
            FP16_SWITCH(!params.is_bf16, [&] {
                if (params.d == 128 and params.d_value == 128) {
                    run_mha_fwd_prefix_prefill_<elem_type, 128, 128>(params, stream);
                } else if (params.d == 192 and params.d_value == 128) {
                    run_mha_fwd_prefix_prefill_<elem_type, 192, 128>(params, stream);
                } else if (params.d == 192 and params.d_value == 192) {
                    run_mha_fwd_prefix_prefill_<elem_type, 192, 192>(params, stream);
                } else if (params.d == 256 and params.d_value == 256) {
                    run_mha_fwd_prefix_prefill_<elem_type, 256, 256>(params, stream); // used in gemma2-9b
                } 
            });
        }
    }
    else if (params.attn_mask != nullptr) {
        // Broadcastable mask attention like torch.sdpa
        FP16_SWITCH(!params.is_bf16, [&] {
            if (params.d == 128) {
                run_mha_fwd_attn_mask_<elem_type, 128, 128>(params, stream);
            }
        });
    }
    else if (params.padding_mask != nullptr) {
        // Encoder attention, bert. etc.
        FP16_SWITCH(!params.is_bf16, [&] {
            if (params.d == 64) {
                run_mha_fwd_padding_mask_<elem_type, 64, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_padding_mask_<elem_type, 128, 128>(params, stream);
            }
        });
    }
    else {
        // Decoder-only attention
        FP16_SWITCH(!params.is_bf16, [&] {
            if (params.is_e4m3) {
                if (params.d == 128 and params.d_value == 128) {
                    run_fp8_mha_fwd_<elem_type, 128, 128>(params, stream);
#ifndef HEADDIM_128_ONLY
                } else if (params.d == 192 and params.d_value == 128) {
                    run_fp8_mha_fwd_<elem_type, 192, 128>(params, stream);
                } else if (params.d == 192 and params.d_value == 192) {
                    run_fp8_mha_fwd_<elem_type, 192, 192>(params, stream);
#endif
                } else if (params.d == 256 and params.d_value == 256) {
                    run_fp8_mha_fwd_<elem_type, 256, 256>(params, stream);
                } else {
                    assert(false && "FP8 forward only supports head_dim=128/128, 192/128, 192/192, or 256/256");
                }
            } else {
            #if defined(HEADDIM_128_ONLY)
                run_mha_fwd_<elem_type, 128, 128>(params, stream);
            #elif defined(HEADDIM_192_128_ONLY)
                run_mha_fwd_<elem_type, 192, 128>(params, stream);
            #else
                ALL_HEADDIM_SWITCH(params.d, params.d_value, [&] {
                    run_mha_fwd_<elem_type, kHeadDimQ, kHeadDimV>(params, stream);
                });
            #endif
            }
        });
    }
#else
    std::fprintf(stderr,
                 "[flash_hg] run_mha_fwd is not built in this core "
                 "(FLASH_ATTN_HG_CORE_MODE does not include 1)\n");
    std::abort();
#endif
}

void (*run_mha_fwd_c)(Flash_fwd_params&, hipStream_t, bool) = run_mha_fwd;



// ====================================================================================================================================
//                                                               BWD
// ====================================================================================================================================
void run_mha_bwd(Flash_bwd_params &params, hipStream_t stream, const bool configure=false) {
#if defined(BUILD_FA_BWD)
    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "5") == 0) return;
        else { printFlashBwdParams(params); };
    }
    ElementType_SWITCH(params.is_bf16, params.is_e4m3, [&] {
        #if defined(HEADDIM_128_ONLY)
            run_mha_bwd_<elem_type, 128, 128>(params, stream, configure);
        #elif defined(HEADDIM_192_128_ONLY)
            run_mha_bwd_<elem_type, 192, 128>(params, stream, configure);
        #else
            HEADDIM_SWITCH(params.d, params.d_value, [&] {
                run_mha_bwd_<elem_type, kHeadDimQ, kHeadDimV>(params, stream, configure);
            });
        #endif
    });
#else
    std::fprintf(stderr,
                 "[flash_hg] run_mha_bwd is not built in this core "
                 "(FLASH_ATTN_HG_CORE_MODE does not include 2)\n");
    std::abort();
#endif
}



// ====================================================================================================================================
//                                                              PA
// ====================================================================================================================================
void run_mha_fwd_kvcache(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel=false) {
#if defined(BUILD_FA_KVCACHE)
    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "5") == 0) return;
        else if (std::strcmp(fa_debug, "C") == 0) {PRINT_PARAMS}; // for c interface debug
    }
    // head dim 闸门: 不支持的组合在此报错, 避免被派发到不匹配的模板 (含 HEADDIM_*_ONLY 构建)。
    if (not PA_HEADDIM_SUPPORTED(params.d, params.d_value)) {
        printf("\x1b[31mPagedAttention (run_mha_fwd_kvcache) only supports headdim(q/k)/headdim(v) "
               "of 128/128, 192/128, 192/192, or 256/256 (subset depending on this build), "
               "but got %d/%d!\033[0m\n", params.d, params.d_value);
        assert(false && "unsupported head dimension for PagedAttention");
        return;
    }
    if (getArch() == 930 && (params.is_e4m3 || params.is_e5m2)) {
        FP16_SWITCH(!params.is_bf16, [&] {
            if (params.d != 128 or params.d_value != 128) {
                printf("gfx92a FP8 pa only supports headdim=128!\n");
                assert(params.d == 128 and params.d_value == 128);
            }
            run_fp8_fwd_splitkv_dispatch<elem_type, 128, 128>(params, stream);
        });
        return;
    }
    FP16_SWITCH(!params.is_bf16, [&] {
    #ifdef HEADDIM_128_ONLY
        run_mha_fwd_splitkv_dispatch<elem_type, 128, 128>(params, stream);
    #elif defined(HEADDIM_192_128_ONLY)
        if (params.d == 192 and params.d_value == 128)
            run_mha_fwd_splitkv_dispatch<elem_type, 192, 128>(params, stream);
    #else
        PA_HEADDIM_SWITCH(params.d, params.d_value, [&] {
            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDimQ, kHeadDimV>(params, stream);
        });
    #endif
    });
#else
    std::fprintf(stderr,
                 "[flash_hg] run_mha_fwd_kvcache is not built in this core "
                 "(FLASH_ATTN_HG_CORE_MODE does not include 3)\n");
    std::abort();
#endif
}

#endif  // FLASH_C_API_EXTERN_ONLY
