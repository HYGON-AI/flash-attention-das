// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic_mls_ds.h"
#include "fp8_mls_bounds.h"
template<int kHeadDim, int WARP_N, typename Element>
__forceinline__ __device__ void fp8_prefetch_v_to_lds(
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int v_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint v_root = prepare_for_matrix_load<kHeadDim, Element>(v_ptr);
    const uint64_t v_root_addr = *(uint64_t*)&v_root;
    vec4_uint v_srsrc;
    v_srsrc[0] = v_root[0];
    v_srsrc[1] = v_root[1];
    v_srsrc[2] = v_row_stride; // stride
    v_srsrc[3] = 0x00000;

    // 4 个 wave 直接全量预取
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    int v_lds_write_bytes = warp_id * WARP_N * kLdsHeadDimStride * sizeof(Element);
    // 每次读取 32x128 的数据

    // Keep the aligned-tile descriptor select: its instruction form is
    // performance-sensitive even when saturated row filters are already zero.

    // tile1: 行 [warp_id*32, warp_id*32+16)
    // 整个 tile 被 filter 时保留一行合法 V，避免 0 * NaN 污染 PV。
    int nm_filter_warp0_tile1 = inline_min_max<0, 16>(16 - max_seq_kv_offset);
    int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
    const int tile1_source_row = (nm_filter == 16) ? 0 : (warp_id * 2) * 16;
    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
        v_root_addr + int64_t(tile1_source_row) * int64_t(v_row_stride) * sizeof(Element));
    nm_filter = (nm_filter == 16) ? min(nm_filter_warp0_tile1, 15) : nm_filter;
    v_srsrc[3] = v_srsrc[3] + ((max_seq_kv_offset % 128 == 0) ? 0 : (nm_filter << 8));
    flash::wait_all_warp_arrived();
    __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_write_bytes, 0, true, false, false, false, false);
    if constexpr (kHeadDim > 128) {
        // D192 loads only columns 128..191 into the existing +6144 DS layout.
        constexpr int kSecondHeadOffset = 128;
        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
            v_root_addr + int64_t(tile1_source_row) * int64_t(v_row_stride) * sizeof(Element) + kSecondHeadOffset * sizeof(Element));
        if constexpr (kHeadDim == 192) {
            __builtin_hcu_matrix_load_64x16_b8(
                v_srsrc, v_lds + v_lds_write_bytes + 6144,
                0, true, false, false, false, false);
        } else {
            __builtin_hcu_matrix_load_128x16_b8(
                v_srsrc, v_lds + v_lds_write_bytes + 4096,
                0, true, false, false, false, false);
        }
    }

    // tile2: 行 [warp_id*32+16, warp_id*32+32)
    int nm_filter_warp0_tile2 = inline_min_max<0, 16>(32 - max_seq_kv_offset);
    nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
    const int tile2_source_row = (nm_filter == 16) ? 0 : (warp_id * 2 + 1) * 16;
    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
        v_root_addr + int64_t(tile2_source_row) * int64_t(v_row_stride) * sizeof(Element));
    nm_filter = (nm_filter == 16) ? min(nm_filter_warp0_tile2, 15) : nm_filter;
    v_srsrc[3] = ((max_seq_kv_offset % 128 == 0) ? 0 : (nm_filter << 8));
    __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_write_bytes + (128 * 16 >> 1), 0, true, false, false, false, false);
    if constexpr (kHeadDim > 128) {
        // The second 16-row slice keeps its 1024-byte half-offset.
        constexpr int kSecondHeadOffset = 128;
        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
            v_root_addr + int64_t(tile2_source_row) * int64_t(v_row_stride) * sizeof(Element) + kSecondHeadOffset * sizeof(Element));
        if constexpr (kHeadDim == 192) {
            __builtin_hcu_matrix_load_64x16_b8(
                v_srsrc, v_lds + v_lds_write_bytes + 6144 + (128 * 16 >> 1),
                0, true, false, false, false, false);
        } else {
            __builtin_hcu_matrix_load_128x16_b8(
                v_srsrc, v_lds + v_lds_write_bytes + 4096 + (128 * 16 >> 1),
                0, true, false, false, false, false);
        }
    }
}



// Shared B64 V layout: four waves each fetch a 16-row slice. D256 adds
// the second 128-column half without changing the row or filter mapping.
// FullKV describes this KV tile only; it does not imply that Q rows are valid.
template<bool FullKV, int kHeadDim, typename Element>
__forceinline__ __device__ void fp8_prefetch_v_to_lds_b64(
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int v_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kHeadDim == 128 || kHeadDim == 256);
    constexpr int kBlockN = 64;
    vec4_uint v_root = prepare_for_matrix_load<kHeadDim, Element>(v_ptr);
    const uint64_t v_root_addr = *(uint64_t*)&v_root;
    vec4_uint v_srsrc;
    v_srsrc[1] = v_root[1];
    v_srsrc[2] = v_row_stride;

    const int row_group = warp_id;
    const int row_block = row_group >> 1;
    const int row_half = row_group & 1;
    const bool v_need_filter = fp8_mls_needs_filter<FullKV, kBlockN>(max_seq_kv_offset);
    const int first_group_filter = inline_min_max<0, 16>(16 - max_seq_kv_offset);
    int nm_filter = inline_min_max<0, 16>((row_group + 1) * 16 - max_seq_kv_offset);
    const bool whole_group_filtered = v_need_filter && nm_filter == 16;
    const int source_row = whole_group_filtered ? 0 : row_group * 16;
    if (whole_group_filtered) {
        nm_filter = min(first_group_filter, 15);
    }
    v_srsrc[3] = v_need_filter ? (nm_filter << 8) : 0;

    const int v_lds_base = row_block * 32 * kHeadDim;
    const int v_lds_half_offset = row_half * 1024;

    flash::wait_all_warp_arrived();

    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
        v_root_addr + int64_t(source_row) * int64_t(v_row_stride) * sizeof(Element));
    __builtin_hcu_matrix_load_128x16_b8(
        v_srsrc, v_lds + v_lds_base + v_lds_half_offset,
        0, true, false, false, false, false);

    if constexpr (kHeadDim == 256) {
        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
            v_root_addr + int64_t(source_row) * int64_t(v_row_stride) * sizeof(Element)
            + 128 * sizeof(Element));
        __builtin_hcu_matrix_load_128x16_b8(
            v_srsrc, v_lds + v_lds_base + 4096 + v_lds_half_offset,
            0, true, false, false, false, false);
    }
}

// D192/B64 keeps the established +6144 tail-column DS mapping.
template<bool FullKV, typename Element>
__forceinline__ __device__ void fp8_prefetch_v_to_lds_d192(
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int v_row_stride,
    int max_seq_kv_offset
) {
    constexpr int kHeadDim = 192;
    constexpr int kBlockN = 64;
    vec4_uint v_root = prepare_for_matrix_load<kHeadDim, Element>(v_ptr);
    const uint64_t v_root_addr = *(uint64_t*)&v_root;
    vec4_uint v_srsrc;
    v_srsrc[1] = v_root[1];
    v_srsrc[2] = v_row_stride;

    const int row_group = warp_id;
    const int row_block = row_group >> 1;
    const int row_half = row_group & 1;
    const bool v_need_filter = fp8_mls_needs_filter<FullKV, kBlockN>(max_seq_kv_offset);
    const int first_group_filter = inline_min_max<0, 16>(16 - max_seq_kv_offset);
    int nm_filter = inline_min_max<0, 16>((row_group + 1) * 16 - max_seq_kv_offset);
    const bool whole_group_filtered = v_need_filter && nm_filter == 16;
    const int source_row = whole_group_filtered ? 0 : row_group * 16;
    if (whole_group_filtered) {
        nm_filter = min(first_group_filter, 15);
    }
    v_srsrc[3] = v_need_filter ? (nm_filter << 8) : 0;

    const int v_lds_base = row_block * 32 * 256;
    const int v_lds_half_offset = row_half * 1024;

    flash::wait_all_warp_arrived();

    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
        v_root_addr + int64_t(source_row) * int64_t(v_row_stride) * sizeof(Element));
    __builtin_hcu_matrix_load_128x16_b8(
        v_srsrc, v_lds + v_lds_base + v_lds_half_offset,
        0, true, false, false, false, false);

    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(
        v_root_addr + int64_t(source_row) * int64_t(v_row_stride) * sizeof(Element)
        + 64 * sizeof(Element));
    __builtin_hcu_matrix_load_128x16_b8(
        v_srsrc, v_lds + v_lds_base + 4096 + v_lds_half_offset,
        0, true, false, false, false, false);
}
