#pragma once
#include "intrinsic_mls_ds.h"
#include "numeric_types.h"
#include <type_traits>
#define USE_MLS_128B_REQUEST

namespace gfx92a_fp8_pa {

template<class ElementA, class ElementB, class AccumType>
inline __device__ vec4_fp32 mmac_4interleave_fp8_gfx92a(
    const vec8_Element<int8_t> &v1,
    const vec8_Element<int8_t> &v2,
    const vec4_fp32 &v3
) {
    constexpr bool AIsE4M3 = std::is_same<ElementA, fp8_e4m3>::value;
    constexpr bool AIsE5M2 = std::is_same<ElementA, fp8_e5m2>::value;
    constexpr bool BIsE4M3 = std::is_same<ElementB, fp8_e4m3>::value;
    constexpr bool BIsE5M2 = std::is_same<ElementB, fp8_e5m2>::value;
    static_assert(AIsE4M3 || AIsE5M2, "MMAC A must be E4M3 or E5M2");
    static_assert(BIsE4M3 || BIsE5M2, "MMAC B must be E4M3 or E5M2");
    if constexpr (AIsE4M3 && BIsE4M3) {
        return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(v1, v2, v3);
    } else if constexpr (AIsE4M3 && BIsE5M2) {
        return __builtin_hcu_mmac_f32_16x16x32_fp8_bf8(v1, v2, v3);
    } else if constexpr (AIsE5M2 && BIsE4M3) {
        return __builtin_hcu_mmac_f32_16x16x32_bf8_fp8(v1, v2, v3);
    } else {
        return __builtin_hcu_mmac_f32_16x16x32_bf8_bf8(v1, v2, v3);
    }
}

template<bool Is_even_MN, int kHeadDim, int WARP_M, typename Element>
__forceinline__ __device__ void fp8_prefetch_q_to_lds(
    Element* q_ptr,
    int8_t* q_lds,
    int warp_id,
    int q_row_stride,
    int max_seq_q_offset
) {
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint q_root = prepare_for_matrix_load<128, Element>(q_ptr);
    vec4_uint q_srsrc;
    q_srsrc[0] = q_root[0];
    q_srsrc[1] = q_root[1];
    q_srsrc[2] = q_row_stride; // stride
    q_srsrc[3] = 0x40000; // [17: 18], interleave 4

    // 计算 lds 写入地址
    int q_lds_offset = warp_id * WARP_M * kHeadDim/*4K bytes, 16K bytes in total*/ * sizeof(Element);
    int q_lds_write_bytes = reinterpret_cast<size_t>(q_lds) + q_lds_offset;

    // 计算 global 读取地址
    q_srsrc[0] = q_root[0] + (warp_id * WARP_M) * q_row_stride * sizeof(Element);

    // 启动 mls 读取
    #ifdef USE_MLS_128B_REQUEST
    #if defined(__gfx92a__)
        const int q_lds_mls_offset = warp_id * 1024;
        const int q_row = warp_id * WARP_M;
        if constexpr (Is_even_MN) {
            q_srsrc[0] = q_root[0] + q_row * q_row_stride * sizeof(Element);
            q_srsrc[3] = 0;
            __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_mls_offset, 0, true, false, false, false, false);
            __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_mls_offset+8192, 16, true, false, false, false, false);
        } else {
            const bool has_tail = max_seq_q_offset % 128 != 0;

            int nm_filter = q_row + 16 - max_seq_q_offset;
            int q_filter = inline_min_max<0, 16>(nm_filter);
            int q_load_row = has_tail && q_filter == 16 ? 0 : q_row;
            q_srsrc[0] = q_root[0] + q_load_row * q_row_stride * sizeof(Element);
            q_srsrc[3] = has_tail ? q_filter << 8 : 0;
            __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_mls_offset, 0, true, false, false, false, false);

            // Move the second 16-row tile into the descriptor base. When it
            // has no valid rows, reuse the first tile and recompute the filter.
            const int q_second_row = q_row + 16;
            const int q_second_filter_max =
                q_second_row + 16 - max_seq_q_offset;
            q_load_row = has_tail && q_second_filter_max >= 16
                ? 0 : q_second_row;
            nm_filter = q_load_row + 16 - max_seq_q_offset;
            q_filter = inline_min_max<0, 16>(nm_filter);
            q_srsrc[0] = q_root[0] + q_load_row * q_row_stride * sizeof(Element);
            q_srsrc[3] = has_tail ? q_filter << 8 : 0;
            __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_mls_offset+8192, 0, true, false, false, false, false);
        }
    #else
        int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_q_offset);
        q_srsrc[3] = 0x40000 + ((max_seq_q_offset % 128 == 0) ? 0: (nm_filter << 8)); // set only once
        // inline_matrix_load_128x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_offset/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_offset, 0, true, false, false, false, false);
        nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_q_offset);
        q_srsrc[3] = 0x40000 + ((max_seq_q_offset % 128 == 0) ? 0: (nm_filter << 8));
        __builtin_hcu_matrix_load_128x16_b8(q_srsrc, q_lds+q_lds_offset+512, 16, true, false, false, false, false);
    #endif
    #else
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_write_bytes/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        q_srsrc[0] = q_srsrc[0] + 64 * sizeof(Element);
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_write_bytes + 2048/*lds bytes*/, 0/*matrix_offset, 0 or 16*/); // Q 部分可以考虑 128x16 或者非 4-interleave 形式
    #endif
    __builtin_amdgcn_sched_barrier(0);
}



// #define USE_DS_READ_B128_FOR_INTERLEAVE4

template<int kHeadDim, int WARP_M, typename Element>
__forceinline__ __device__ void load_q_from_lds_to_vgpr(
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64],
    int8_t* q_lds,
    int warp_id,
    int lane_id
) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);
    // lds 写到两个地方去了, 注意是 rearrange, 所以跳 1K; transpose 跳 2K
    // MLS0: [0: 512) 和 [1024, 1536)
    // MLS1: [512: 1024) 和 [1536, 2048)
    // 分 4 次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
    #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
        int row = (lane_id & 15) >> 1;
        int col = lane_id >> 4;
        int col_swizzle = (row + col) & 3;
        int lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + warp_id * WARP_M * kHeadDim;
        q_regs[0][0].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 0);
        q_regs[1][0].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 1024/*ds fmt 0, dmft1 */);
        q_regs[0][1].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 2048/*ds fmt 0, dmft1 */);
        q_regs[1][1].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 3072/*ds fmt 0, dmft1 */);
    #else
#if defined(__gfx92a__)
        int8_t* q_lds_load_ptr = q_lds + warp_id * 1024;
        q_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds_load_ptr, 0, 3, 1, 0);
        q_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds_load_ptr + 4096, 0, 3, 1, 0);
        q_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds_load_ptr + 8192, 0, 3, 1, 0);
        q_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds_load_ptr + 12288, 0, 3, 1, 0);
    #else
        q_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + 0 + warp_id * WARP_M * kHeadDim,    0, 3, 1, 0);
        q_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + 1024 + warp_id * WARP_M * kHeadDim, 0, 3, 1, 0);
        q_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + 2048 + warp_id * WARP_M * kHeadDim, 0, 3, 1, 0);
        q_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + 3072 + warp_id * WARP_M * kHeadDim, 0, 3, 1, 0);
    #endif
    #endif
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
}



template<bool Is_even_MN, int kHeadDim, int WARP_N, typename Element, int WARP_NUM = 4>
__forceinline__ __device__ void fp8_prefetch_k_to_lds(
    Element* k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint k_root = prepare_for_matrix_load<kHeadDim, Element>(k_ptr);
    vec4_uint k_srsrc;
    k_srsrc[0] = k_root[0];
    k_srsrc[1] = k_root[1];
    k_srsrc[2] = k_row_stride; // stride
    k_srsrc[3] = 0x40000; // [17: 18], interleave 4

    // 计算 lds 写入地址
    int k_lds_offset = warp_id * WARP_N * kHeadDim * sizeof(Element);
    int k_lds_write_bytes = reinterpret_cast<size_t>(k_lds) + k_lds_offset;

    // 计算 global 读取地址
    k_srsrc[0] = k_root[0] + warp_id * 32 * k_row_stride * sizeof(Element);

    // 启动 mls 读取
    #ifdef USE_MLS_128B_REQUEST
    #if defined(__gfx92a__)
        if constexpr (WARP_NUM == 4) {
            const int k_lds_mls_offset = warp_id * 1024;
            if constexpr (Is_even_MN) {
                k_srsrc[3] = 0;
                k_srsrc[0] = k_root[0] + warp_id * 16 * k_row_stride * sizeof(Element);
                __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_mls_offset, 0, true, false, false, false, false);
                k_srsrc[0] = k_root[0] + (warp_id + WARP_NUM) * 16 * k_row_stride * sizeof(Element);
                __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_mls_offset+8192, 0, true, false, false, false, false);
            } else {
                const bool has_tail = max_seq_kv_offset % 128 != 0;

                int k_row = warp_id * 16;
                int nm_filter = k_row + 16 - max_seq_kv_offset;
                int k_filter = inline_min_max<0, 16>(nm_filter);
                int k_load_row = has_tail && k_filter == 16 ? 0 : k_row;
                k_srsrc[0] = k_root[0] + k_load_row * k_row_stride * sizeof(Element);
                k_srsrc[3] = has_tail ? k_filter << 8 : 0;
                __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_mls_offset, 0, true, false, false, false, false);

                k_row = (warp_id + WARP_NUM) * 16;
                nm_filter = k_row + 16 - max_seq_kv_offset;
                k_filter = inline_min_max<0, 16>(nm_filter);
                k_load_row = has_tail && k_filter == 16 ? 0 : k_row;
                k_srsrc[0] = k_root[0] + k_load_row * k_row_stride * sizeof(Element);
                k_srsrc[3] = has_tail ? k_filter << 8 : 0;
                __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_mls_offset+8192, 0, true, false, false, false, false);
            }
        } else {
            const bool has_tail = max_seq_kv_offset % 128 != 0;
            #pragma unroll
            for (int tile = warp_id; tile < 8; tile += WARP_NUM) {
                const int k_lds_mls_offset =
                    (tile & 3) * 1024 + (tile >= 4 ? 8192 : 0);
                const int k_row = tile * 16;
                int nm_filter = k_row + 16 - max_seq_kv_offset;
                int k_filter = inline_min_max<0, 16>(nm_filter);
                int k_load_row = has_tail && k_filter == 16 ? 0 : k_row;
                k_srsrc[0] = k_root[0] + k_load_row * k_row_stride * sizeof(Element);
                k_srsrc[3] = has_tail ? k_filter << 8 : 0;
                __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_mls_offset, 0, true, false, false, false, false);
            }
        }
    #else
        int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
        k_srsrc[3] = k_srsrc[3] + ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
        __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_offset, 0, true, false, false, false, false);
        nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
        k_srsrc[3] = 0x40000 + ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
        __builtin_hcu_matrix_load_128x16_b8(k_srsrc, k_lds+k_lds_offset+512, 16, true, false, false, false, false);
    #endif
    #else
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, k_lds_write_bytes/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        k_srsrc[0] = k_srsrc[0] + 64 * sizeof(Element);
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, k_lds_write_bytes + 2048/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
    #endif
}

} // namespace gfx92a_fp8_pa
