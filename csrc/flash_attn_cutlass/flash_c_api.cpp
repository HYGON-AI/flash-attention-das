// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "flash_c_api.h"
#include "static_switch.h"

#include <c10/util/Exception.h>
#include <hytlass/bfloat16.h>
#include <hytlass/half.h>
extern "C"
void kv_cache_attention(void* q,int* seq_offsets,int max_seqlen_q,int max_seqlen_k,void* k_caches,void* v_caches,int* page_tables,int* cache_lens,
    void* out,int num_query,int num_tokens,int num_q_head,int num_kv_head,int num_dim,int num_pages,int page_size,const std::string& dtype,int layout);
void run_mha_fwd_attnmask(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask does not support splitkv.");
    TORCH_CHECK(params.d == 32 || params.d == 64 || params.d == 128,
                "run_mha_fwd_attnmask only supports headdim=32, 64, or 128 for now.");

    FP16_SWITCH(!params.is_bf16, [&] {
        if (params.d <= 32) {
            constexpr static int kHeadDim = 32;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        } else if (params.d <= 64) {
            constexpr static int kHeadDim = 64;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        } else {
            constexpr static int kHeadDim = 128;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        }
    });
}

void run_mha_fwd_attnmask_bias(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask_bias does not support splitkv.");
    TORCH_CHECK(params.d == 32, "run_mha_fwd_attnmask_bias only supports headdim=32.");
    TORCH_CHECK(params.attn_bias_ptr != nullptr, "run_mha_fwd_attnmask_bias requires attn_bias_ptr.");

    FP16_SWITCH(!params.is_bf16, [&] {
        constexpr static int kHeadDim = 32;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_mha_fwd_attnmask_bias_<elem_type, kHeadDim, Is_causal>(params, stream);
        });
    });
}

void run_mha_fwd_attnmask_bias_144(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask_bias_144 does not support splitkv.");
    TORCH_CHECK(params.d == 32, "run_mha_fwd_attnmask_bias_144 only supports headdim=32.");
    TORCH_CHECK(params.seqlen_q == 144 && params.seqlen_k == 144,
                "run_mha_fwd_attnmask_bias_144 only supports seqlen_q=144 and seqlen_k=144.");
    TORCH_CHECK(!params.is_causal, "run_mha_fwd_attnmask_bias_144 only supports non-causal attention.");
    TORCH_CHECK(params.attn_bias_ptr != nullptr, "run_mha_fwd_attnmask_bias_144 requires attn_bias_ptr.");

    FP16_SWITCH(!params.is_bf16, [&] {
        constexpr static int kHeadDim = 32;
        run_mha_fwd_attnmask_bias_<elem_type, kHeadDim, false>(params, stream);
    });
}

void run_mha_fwd_attnmask_additive(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask_additive does not support splitkv.");
    TORCH_CHECK(params.d == 32, "run_mha_fwd_attnmask_additive only supports headdim=32.");
    TORCH_CHECK(params.additive_mask_ptr != nullptr,
                "run_mha_fwd_attnmask_additive requires additive_mask_ptr.");
    TORCH_CHECK(params.attn_bias_ptr != nullptr, "run_mha_fwd_attnmask_additive requires attn_bias_ptr.");

    FP16_SWITCH(!params.is_bf16, [&] {
        constexpr static int kHeadDim = 32;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_mha_fwd_attnmask_additive_<elem_type, kHeadDim, Is_causal>(params, stream);
        });
    });
}

void run_mha_fwd_attnmask_additive_144(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask_additive_144 does not support splitkv.");
    TORCH_CHECK(params.d == 32, "run_mha_fwd_attnmask_additive_144 only supports headdim=32.");
    TORCH_CHECK(params.seqlen_q == 144 && params.seqlen_k == 144,
                "run_mha_fwd_attnmask_additive_144 only supports seqlen_q=144 and seqlen_k=144.");
    TORCH_CHECK(!params.is_causal, "run_mha_fwd_attnmask_additive_144 only supports non-causal attention.");
    TORCH_CHECK(params.additive_mask_ptr != nullptr,
                "run_mha_fwd_attnmask_additive_144 requires additive_mask_ptr.");
    TORCH_CHECK(params.attn_bias_ptr != nullptr, "run_mha_fwd_attnmask_additive_144 requires attn_bias_ptr.");

    FP16_SWITCH(!params.is_bf16, [&] {
        constexpr static int kHeadDim = 32;
        run_mha_fwd_attnmask_additive_<elem_type, kHeadDim, false>(params, stream);
    });
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {

    if(params.is_fp8==true)
    {
        using elem_type = hytlass::float_e4m3_t;
          
        if(params.d==128)
        {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
            if (params.num_splits <= 1 && !force_split_kernel) {
                run_mha_fwd_<elem_type, 128, Is_causal>(params, stream);
            } else {
                TORCH_CHECK(false, "fa fp8 not supoort splitkv .");
            }
            });      
        }
        else
        {
            TORCH_CHECK(false, "fa fp8 only support dim=128");
        }
    }
    else
    {
        FP16_SWITCH(!params.is_bf16, [&] {
            // using elem_type = cutlass::half_t;
            HEADDIM_SWITCH(params.d, [&] {
                // constexpr static int kHeadDim = 256;
                BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                    // constexpr static bool Is_causal = false; 
                    // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
                    if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                        run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                    } else {
                        // TORCH_CHECK(false, "fa not supoort splitkv dim=512");
                        std::string dtype = params.is_bf16 ? "fp16" : "fp16";
                        kv_cache_attention((void*)params.q_ptr,params.cu_seqlens_q,params.seqlen_q,params.seqlen_k,
                                            (void*)params.k_ptr,(void*)params.v_ptr,params.block_table,params.seqused_k,
                                            (void*)params.o_ptr,params.b,params.total_q,params.h,params.h_k,params.d,2000,params.page_block_size,dtype,2);
                    }
                });
            });
        });      
    }
}

void run_mha_fwd_padding_mask(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    FP16_SWITCH(!params.is_bf16, [&] {
        // using elem_type = cutlass::half_t;
        run_mha_fwd_padding_mask_<elem_type, 64, false>(params, stream);
    });
}


void run_mha_fwd_16x64(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd(params,stream);
}

void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd(params,stream,true);
}