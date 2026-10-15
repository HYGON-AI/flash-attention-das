#pragma once
#include "fwd/gfx938/pv_gemm_utils_mls_ds.h"
#include "fwd/gfx92a/qk_gemm_utils_mls_ds_gfx92a.h"



template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds_gfx92a_2TG(
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_k_offset=0) {

    // Simplify
    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported!");
    static_assert (WARP_M == 32 and "To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "To simplify, only WARP_N = 32 is supported!");
    constexpr int WARP_NUM = kBlockM / WARP_M;
    constexpr int k_lds_load_num  = WARP_N * kHeadDim / (32 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // 准备 K 寄存器
    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // 计算 K lds 起始偏移量
    int k_lds_base = reinterpret_cast<size_t>(k_lds);

    // here, v_mov_b64 can be applied
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        inline_vgpr4_init_zero_4x4x4(s_reg);
    } else {
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) { // for kBlockN = 64, only wave 0 get the right QK gemm results
            for (int j = 0; j < 4; ++j) {
                s_reg[i][j].u64[0] = 0;
                s_reg[i][j].u64[1] = 0;
            }
        }
    }
    flash::lower_priority();

    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    k_srsrc[3] = 0;

    #pragma unroll
    for(int n_loop = 0; n_loop < (kBlockN / WARP_N); ++n_loop) {

        // Wait global data
        flash::wait_buffer_data_arrived<true>(kBlockN / WARP_N - n_loop - 1);

        // DS
        int stage_id = 0;
        {
            constexpr int k_loop = 0;
            int lds_load_offset = k_lds_base + (n_loop * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        #pragma unroll
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            // DS
            int lds_load_offset = k_lds_base + (n_loop * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            flash::wait_lds_data_arrived<false>(3);
            flash::raise_priority();
            // MMAC
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::lower_priority();
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        flash::raise_priority();
        // last mmac
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::lower_priority();
    }

    if constexpr (STAGES == 2) {
    #if defined(__gfx938__) || defined(__gfx946__)
        prefetch_v_to_lds_mls_ds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 2, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    #else

    #endif
    }

} // qk_gemm





















template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds_gfx92a(
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_k_offset=0) {

    // Simplify
    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported!");
    static_assert (WARP_M == 32 and "To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "To simplify, only WARP_N = 32 is supported!");
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr bool HasKTail       = (kHeadDim > 128); // 128 列主块之外还有一段 K
    constexpr int  kHeadDim_      = HasKTail ? 128 : kHeadDim;
    constexpr int K_LDS_LOAD_NUM  = WARP_N * kHeadDim_ / (32 * 32);
    constexpr int K_LOAD_REQUESTS = K_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    static_assert(kHeadDim != 192 or kBlockN == 64, "gfx92a (192,x) 的 K 尾巴只支持 kBlockN == 64 (= WARP_N * 2)");
    
    // Prepare regs for k
    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // Zero-initialize s_reg
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        inline_vgpr4_init_zero_4x4x4(s_reg);
    } else {
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) { // for kBlockN = 64, only wave 0 get the right QK gemm results
            for (int j = 0; j < 4; ++j) {
                s_reg[i][j].u64[0] = 0;
                s_reg[i][j].u64[1] = 0;
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    k_srsrc[3] = 0;
    int n_stage_id = 0;

    n_stage_id ^= 1;
    // n_loop = 0 第一阶段，prefetch_k_to_lds 已 load 128 * 32 K矩阵
    #pragma unroll
    for(int n_loop = 1; n_loop < (kBlockN / WARP_N); ++n_loop) {
        // MLS
        if constexpr (not HasKTail) {
            int k_row = n_loop * WARP_N;
            int nm_filter_max = k_row + 32 - max_seq_k_offset;
            int k_load_row = (max_seq_k_offset % kBlockN != 0 && nm_filter_max >= 32) ? 0: k_row;
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (k_load_row * seqlen_k_stride + warp_id * 32) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = max(0, min(31, k_load_row + 32 - max_seq_k_offset)); // v_med3_i32 applied, thus sgpr -> vgpr
                int __nm_filter = __builtin_amdgcn_readfirstlane(nm_filter);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0 ? 0: __nm_filter << 8; // set only once
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 32) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        } else {
            int warp_id_m = warp_id / 2;
            int warp_id_n = warp_id % 2;
            int k_load = 1;
            int n_loop_ = n_loop - 1; // Add this to support headdim>128, we have to deal with previous n_loop
            int k_row = n_loop_ * WARP_N + warp_id_n * 16;
            int nm_filter_max = k_row + 16 - max_seq_k_offset;
            int k_load_row = (max_seq_k_offset % kBlockN != 0 && nm_filter_max >= 16) ? 0: k_row;
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (k_load_row * seqlen_k_stride + warp_id_m * 32 + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 16>(k_load_row + 16 - max_seq_k_offset);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        }

        // Wait MLS
        n_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);

        // DS
        {
            constexpr int k_loop = 0;
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {

            // Wait for special headdim
            if constexpr (HasKTail) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);   
                }
            }

            // DS
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);
            asm volatile("s_setprio 2");
            // MMAC
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            asm volatile("s_setprio 0");
            // MLS for special headdim
            if constexpr (HasKTail) { // Once finish 128x32, we need issue next MLS instruction, and get now we need use data via wait
                if ((k_loop & 3) == 0x0) {
                    // n_stage_id ^=1;
                    int k_load = k_loop / 4;
                    int n_loop_ = ((kHeadDim / kBlockK) - k_loop) < 4 ? (k_load = 0, n_loop): n_loop_;   // if finish kHeadDim*WarpN prefetch, we prefetch next n_loop data
                    int nm_filter_max = n_loop_ * WARP_N + 32 - max_seq_k_offset;
                    int real_mls_loop = (max_seq_k_offset % kBlockN == 0x0 or nm_filter_max < 32) ? n_loop_: 0;
                    *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (real_mls_loop * WARP_N * seqlen_k_stride + warp_id * 32 + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
                    if constexpr (true) {
                        int nm_filter = inline_min_max<0, 31>(real_mls_loop * WARP_N + 32 - max_seq_k_offset);
                        k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
                    }
                    int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 32) * ELEMENT_BYTES;
                    flash::wait_all_warp_arrived();
                    inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
                }
            }
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        asm volatile("s_setprio 2");
        // MMAC
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        asm volatile("s_setprio 0");
    }

    {
        constexpr int n_loop = kBlockN / WARP_N;
        // MLS for special headdim
        if constexpr (HasKTail) {
            int warp_id_m = warp_id / 2;
            int warp_id_n = warp_id % 2;
            constexpr int n_loop_ = n_loop - 1;
            int k_load = 1;
            n_stage_id ^= 1;                
            // 同循环内那次尾巴 MLS: 16 行请求整块越界时回退到本 tile 第 0 行 (地址一定有效)。
            int k_row = n_loop_ * WARP_N + warp_id_n * 16;
            int nm_filter_max = k_row + 16 - max_seq_k_offset;
            int k_load_row = (max_seq_k_offset % kBlockN != 0 && nm_filter_max >= 16) ? 0: k_row;
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (k_load_row * seqlen_k_stride + warp_id_m * 32 + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 16>(k_load_row + 16 - max_seq_k_offset);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        }
        
        // Wait MLS
        n_stage_id ^= 1;
        int stage_id = 0;
        if constexpr (HasKTail) {
            flash::wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);
        } else {
            flash::wait_buffer_data_arrived<true>(0);
        }

        // DS
        {
            int k_loop = 0;
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);            
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {

            // Wait for special headdim
            if constexpr (HasKTail) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);
                }
            }

            // DS
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);
            asm volatile("s_setprio 2");
            // MMAC
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            asm volatile("s_setprio 0");
        }
        stage_id ^= 1;
        // MMAC
        flash::wait_lds_data_arrived<false>(1);
        asm volatile("s_setprio 2");
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        asm volatile("s_setprio 0");
    }

    if constexpr (STAGES == 2) {
        prefetch_v_to_lds_mls_ds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 2, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    }

} // qk_gemm
