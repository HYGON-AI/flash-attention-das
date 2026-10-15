#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic_mls_ds.h"
template<bool Is_even_MN, int kBlockN, int kHeadDim, int WARP_N, typename Element>
__forceinline__ __device__ void fp8_prefetch_v_to_lds(
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int v_row_stride,
    int max_seq_kv_offset
) {
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint v_root = prepare_for_matrix_load<kHeadDim, Element>(v_ptr);
    vec4_uint v_srsrc;
    v_srsrc[0] = v_root[0];
    v_srsrc[1] = v_root[1];
    v_srsrc[2] = v_row_stride; // stride
    v_srsrc[3] = 0x00000;

    // 4 个 wave 直接全量预取
    int v_lds_write_bytes = warp_id * WARP_N * kHeadDim * sizeof(Element);
    // 每次读取 32x128 的数据

#if defined(__gfx92a__)
    constexpr int WARP_NUM = 4;
    const int v_lds_mls_offset = warp_id * 1024;

    if constexpr (Is_even_MN) {
        v_srsrc[3] = 0;
        v_srsrc[0] = v_root[0] + warp_id * 16 * v_row_stride * sizeof(Element);
        __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_mls_offset, 0, true, false, false, false, false);

        v_srsrc[0] = v_root[0] + (warp_id + WARP_NUM) * 16 * v_row_stride * sizeof(Element);
        __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_mls_offset+8192, 0, true, false, false, false, false);
    } else {
        const bool has_tail = max_seq_kv_offset % 128 != 0;

        int v_row = warp_id * 16;
        int nm_filter = inline_min_max<0, 16>(v_row + 16 - max_seq_kv_offset);
        int v_load_row = has_tail && nm_filter == 16 ? 0 : v_row;
        v_srsrc[0] = v_root[0] + v_load_row * v_row_stride * sizeof(Element);
        v_srsrc[3] = has_tail ? nm_filter << 8 : 0;
        __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_mls_offset, 0, true, false, false, false, false);

        v_row = (warp_id + WARP_NUM) * 16;
        nm_filter = inline_min_max<0, 16>(v_row + 16 - max_seq_kv_offset);
        v_load_row = has_tail && nm_filter == 16 ? 0 : v_row;
        v_srsrc[0] = v_root[0] + v_load_row * v_row_stride * sizeof(Element);
        v_srsrc[3] = has_tail ? nm_filter << 8 : 0;
        __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_mls_offset+8192, 0, true, false, false, false, false);
    }
#else
    // tile1: 行 [warp_id*32, warp_id*32+16)
    // nm_filter=16 时整个 tile 超出边界：改为加载 warp0 的合法地址，nm_filter 对齐 warp0
    // PV MMAC 中 p_reg 对应列为 0，0 * 合法值 = 0，结果正确且不含 NaN
    int nm_filter_warp0_tile1 = inline_min_max<0, 16>(16 - max_seq_kv_offset);
    int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
    v_srsrc[0] = (nm_filter == 16) ? v_root[0] : v_root[0] + (warp_id * 2) * 16 * v_row_stride * sizeof(Element);
    nm_filter = (nm_filter == 16) ? nm_filter_warp0_tile1 : nm_filter;
    v_srsrc[3] = v_srsrc[3] + ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
    __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_write_bytes, 0, true, false, false, false, false);

    // tile2: 行 [warp_id*32+16, warp_id*32+32)
    int nm_filter_warp0_tile2 = inline_min_max<0, 16>(32 - max_seq_kv_offset);
    nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
    v_srsrc[0] = (nm_filter == 16) ? v_root[0] : v_root[0] + (warp_id * 2 + 1) * 16 * v_row_stride * sizeof(Element);
    nm_filter = (nm_filter == 16) ? nm_filter_warp0_tile2 : nm_filter;
    v_srsrc[3] = ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
    __builtin_hcu_matrix_load_128x16_b8(v_srsrc, v_lds+v_lds_write_bytes + (128 * 16 >> 1), 0, true, false, false, false, false);
#endif
}
