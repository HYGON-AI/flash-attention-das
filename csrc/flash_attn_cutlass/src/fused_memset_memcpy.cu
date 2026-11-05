// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include "flash_attnmask.h"

constexpr int VEC = 16;
constexpr int THREADS_PER_WARP = 64;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_WARP;

template <typename MaskType>
__global__ void fused_zero_copy_kernel(
    const MaskType* __restrict__ src,
    uint4*          __restrict__ dst_vec,
    const int64_t src_stride_b,
    const int64_t src_stride_h,
    const int64_t src_stride_q,
    const int64_t src_stride_k,
    const int seqlen_q,
    const int seqlen_k,
    const int padded_q,
    const int padded_k_vec,
    const int H,
    const int64_t q_stride,
    const int64_t h_stride,
    const int64_t b_stride) {

    const int tid = threadIdx.x;

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q = blockIdx.x;

    const int kv_base = tid;
    const int64_t base_addr_q = b * b_stride + h * h_stride + q * q_stride;

    for (int kv = kv_base; kv < padded_k_vec; kv += BLOCK_SIZE) {
        const int k_base = kv * VEC;
        const int64_t vid = base_addr_q + kv;
        uint4 packed;
        uint8_t* vals = reinterpret_cast<uint8_t*>(&packed);

        if (q >= seqlen_q) {
            packed = make_uint4(0, 0, 0, 0);
        }
        else {
            const int64_t src_addr = b * src_stride_b
                                   + h * src_stride_h
                                   + q * src_stride_q
                                   + k_base * src_stride_k;
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                const bool k_valid = (k_base + i) < seqlen_k;
                vals[i] = k_valid ? static_cast<uint8_t>(src[src_addr + i * src_stride_k] != MaskType(0)) : 0;
            }
        }
        dst_vec[vid] = packed;
    }
}

at::Tensor fused_zero_copy(
    const at::Tensor& attn_mask,
    int mask_seqlen_q_padded,
    int mask_seqlen_k_padded)
{
    TORCH_CHECK(attn_mask.dim() == 4, "mask must be 4D [B,H,Q,K]");
    TORCH_CHECK(attn_mask.is_cuda(), "must be on HCU");

    const int B = attn_mask.size(0);
    const int H = attn_mask.size(1);
    const int seqlen_q = attn_mask.size(2);
    const int seqlen_k = attn_mask.size(3);

    if (mask_seqlen_q_padded == seqlen_q &&
        mask_seqlen_k_padded == seqlen_k &&
        attn_mask.dtype() == at::kBool) {
        return attn_mask;
    }

    TORCH_CHECK(mask_seqlen_k_padded % VEC == 0, "padded K must be multiple of 16");

    auto dst = at::empty(
        {B, H, mask_seqlen_q_padded, mask_seqlen_k_padded},
        attn_mask.options().dtype(at::kBool)).contiguous();

    const int padded_k_vec = mask_seqlen_k_padded / VEC;

    const int64_t q_stride = padded_k_vec;
    const int64_t h_stride = (int64_t)mask_seqlen_q_padded * padded_k_vec;
    const int64_t b_stride = (int64_t)H * h_stride;

    dim3 grid(mask_seqlen_q_padded, H, B);
    dim3 block(BLOCK_SIZE);

    auto strides = attn_mask.strides();
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        attn_mask.scalar_type(),
        "fused_pad_mask_hcu",
        [&] {
            fused_zero_copy_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                    attn_mask.data_ptr<scalar_t>(),
                    reinterpret_cast<uint4*>(dst.data_ptr<bool>()),
                    strides[0], strides[1], strides[2], strides[3],
                    seqlen_q, seqlen_k,
                    mask_seqlen_q_padded, padded_k_vec,
                    H,
                    q_stride, h_stride, b_stride);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}
