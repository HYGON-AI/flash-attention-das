// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "flash.h"

// Parameter struct for attention mask forward path, isolated from the main
// Flash_fwd_params to avoid polluting the common kernel interface.
struct Flash_fwd_params_attnmask : public Flash_fwd_params {
    // Attention mask pointer and strides.
    // Expected layout: [b, h, seqlen_q, seqlen_k], with K dim contiguous
    // (K stride == 1). Only the Q stride is configurable here.
    void * __restrict__ mask_ptr;
    index_t mask_batch_stride;
    index_t mask_head_stride;
    index_t mask_seq_q_stride;

    // Value to write when mask is false.
    float masked_value;

    // Optional additive mask and attention bias for fixed-length inference.
    // Additive mask layout: [num_windows, seqlen_q, seqlen_k].
    // Bias layout: [num_heads, seqlen_q, seqlen_k].
    void * __restrict__ additive_mask_ptr;
    index_t additive_mask_window_stride;
    index_t additive_mask_seq_q_stride;
    void * __restrict__ attn_bias_ptr;
    index_t attn_bias_head_stride;
    index_t attn_bias_seq_q_stride;
    int num_windows;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward entry point for attention with explicit mask.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_attnmask_(Flash_fwd_params_attnmask &params, cudaStream_t stream);

// Forward entry point for attention with relative bias only.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_attnmask_bias_(Flash_fwd_params_attnmask &params, cudaStream_t stream);

// Forward entry point for additive mask + relative bias inference.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_attnmask_additive_(Flash_fwd_params_attnmask &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

// Parameter struct for attention mask backward path.
struct Flash_bwd_params_attnmask : public Flash_bwd_params {
    // Attention mask pointer and strides.
    // Expected layout: [b, h, seqlen_q, seqlen_k], with K dim contiguous
    // (K stride == 1). Only the Q stride is configurable here.
    void * __restrict__ mask_ptr;
    index_t mask_batch_stride;
    index_t mask_head_stride;
    index_t mask_seq_q_stride;

    // Value used when mask is false (typically -INFINITY in forward).
    // In backward, positions where mask is false have P=0, so dS should also be 0.
    float masked_value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Backward entry point for attention with explicit mask.
template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_attnmask_(Flash_bwd_params_attnmask &params, cudaStream_t stream);

at::Tensor fused_zero_copy(
        const at::Tensor& attn_mask,
        int mask_seqlen_q_padded,
        int mask_seqlen_k_padded);
