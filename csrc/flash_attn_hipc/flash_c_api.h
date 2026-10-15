#include <fstream>
#include <cstdlib>
#include <vector>
#include "numeric_types.h"
#include "config.h"
#include "static_switch.h"
#include "flash_singleton.h"
#include "flash_memory.h"


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
#endif
}
