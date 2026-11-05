// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "fwd/gfx92a/qk_gemm_utils_mls_ds_gfx92a.h"
#include "static_switch.h"

template<bool PREFETCH_K, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void pv_gemm_prefetch_k_mls_ds_gfx92a(
        vec4_uint v_ptr,
        vec4_uint k_ptr,
        Element* v_lds,
        Element* k_lds,
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockK / 32)][4],
        vec4_Accum<ElementAccum> pv_reg[(kHeadDimV / kBlockN) * (WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_kv_offset=0) {

    constexpr int WARP_NUM = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr int WARP_K   = 32;
    constexpr int READ_ONCE_COUNT = 32 * 32;
    constexpr bool HasVTail       = (kHeadDimV - 128) / 64; // 128 列主块之外还有一段 V
    constexpr int  kHeadDimV_OPT  = HasVTail ? 128 : kHeadDimV;
    constexpr int V_LDS_LOAD_NUM  = (kHeadDimV_OPT * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);
    static_assert (kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert (kBlockM >= WARP_M, "Error: pv gemm kBlockM must be equal or greater than WARP_M");
    static_assert (kBlockN == WARP_N, "Error: pv gemm kBlockN must be equal to WARP_N");
    static_assert (WARP_K == 32 and "Error: To simplify, only WARP_K = 32 is supported!");
    static_assert (WARP_M == 32 and "Error: To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "Error: To simplify, only WARP_N = 32 is supported!");
    static_assert ((kHeadDimV != 192) || (kBlockK == 64),
                   "gfx92a PV 的 V 尾巴分支目前只支持 kHeadDimV == 192 且 kBlockK == 64 (= WARP_K * 2)");

    // Prepare V regs
    union_vec4_f16x2<Element> v_reg[STAGES * (32 * WARP_N) / (32 * 32) * 2];

    // Prepare V lds offset
    // int v_lds_base = 0; // reinterpret_cast<size_t>(v_lds); // ===> 性能下降 ?
    int v_lds_base = reinterpret_cast<size_t>(v_lds); // ===> 性能下降 ?

    // Prepare MLS buffer resource sregs
    vec4_uint v_srsrc;
    v_srsrc[0] = v_ptr[0];
    v_srsrc[1] = v_ptr[1];
    v_srsrc[2] = seqlen_v_stride; // stride
    v_srsrc[3] = 0;

    int lds_stage_id = 1;

    // Main loop across blockN(128) among seqlenkv
    for (int n_loop = 1; n_loop < (kBlockK / WARP_K); ++n_loop) {

        // Do k-dim interleave for next mmac
        #if defined(__gfx92a__)
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 1].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 1].f16x4);
        #endif

        // MLS dispatch
        if constexpr (not HasVTail) {
            if constexpr (Is_even_MN) {
                *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop * WARP_K * seqlen_v_stride + warp_id * 32) * ELEMENT_BYTES);
                v_srsrc[3] = 0x20000;
            } else {
                int nm_filter_max = n_loop * WARP_K + 32 - max_seq_kv_offset;
                int real_mls_loop = nm_filter_max >= 32 ? 0 : n_loop;
                // 封顶 31: filter 字段只有 5 bit, 编不出 32 (回退到 loop 0 后正常不会到 32, 这里只是兜底)
                int nm_filter = inline_min_max<0, 31>(real_mls_loop * WARP_K + 32 - max_seq_kv_offset);
                v_srsrc[3] = (nm_filter << 8) + 0x20000;
                *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id * 32) * ELEMENT_BYTES);
            }

            int lds_write_offset = (lds_stage_id * WARP_K * kHeadDimV + warp_id * 32 * 32) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_write_offset, 0);
        } else {
            // head dimV > 128: 这一轮搬的是上一个 n_loop 的尾巴 (4 个 warp 各一块 32x16 半块)
            int warp_id_m = warp_id % 2;    // w0 w2
            int warp_id_n = warp_id / 2;    // w1 w3
            int n_load = 1;
            int n_loop_ = n_loop - 1;
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop_ * WARP_K * seqlen_v_stride + warp_id_m * 16 * seqlen_v_stride + warp_id_n * 32 + n_load * WARP_NUM * 32) * ELEMENT_BYTES);
            int nm_filter = inline_min_max<0, 16>(n_loop_ * WARP_K + warp_id_m * 16 + 16 - max_seq_kv_offset); // 重新计算 nm_filter
            v_srsrc[3]    = (max_seq_kv_offset % kBlockK == 0 ? 0: nm_filter << 8) + 0x20000;
            int lds_write_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds<0, 1>(v_lds, v_srsrc, lds_write_offset, 0);
        }

        // Wait buffer
        lds_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);

        // DS dispatch
        int lds_load_offset = (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // Wait for special headdimV
            if constexpr (HasVTail) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);
                }
            }

            // DS dispatch
            int lds_load_offset = (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);
            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
            // MLS for special headdimV (V 尾巴)
            if constexpr (HasVTail) {
                if ((k_loop & 3) == 0x0) {
                    int n_load = k_loop / 4;
                    int n_loop_ = ((kHeadDimV / kBlockN) - k_loop) < 4 ? (n_load = 0, n_loop): n_loop_;   // if finish kHeadDimV*WarpK prefetch, we prefetch next n_loop data
                    if constexpr (Is_even_MN) {
                        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop_ * WARP_K * seqlen_v_stride + warp_id * 32 + n_load * 32 * WARP_NUM) * ELEMENT_BYTES);
                        v_srsrc[3] = 0x20000;
                    } else {
                        int nm_filter_max = n_loop_ * WARP_K + 32 - max_seq_kv_offset;
                        int real_mls_loop = nm_filter_max >= 32 ? 0: n_loop_; // 如果全越界了, 则只访问 n_loop = 0 的那波数据
                        int nm_filter     = inline_min_max<0, 31>(real_mls_loop * WARP_K + 32 - max_seq_kv_offset); // 5-bit 字段编不出 32
                        v_srsrc[3]        = (nm_filter << 8) + 0x20000;
                        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id * 32 + n_load * 32 * WARP_NUM) * ELEMENT_BYTES);
                    }
                    int lds_write_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 32) * ELEMENT_BYTES;
                    flash::wait_all_warp_arrived();
                    inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_write_offset, 0);
                }
            }
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        {
            constexpr int min_tile_k = 0;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
    }

    // Prefetch K (V 尾巴路径放到最后发, 见函数末尾)
    if constexpr (PREFETCH_K and not HasVTail) {
        prefetch_k_to_lds_mls_ds<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, max_seq_kv_offset);
    }

    {
        constexpr int n_loop = kBlockK / WARP_K;

        // Do k-dim interleave for next mmac
        #if defined(__gfx92a__)
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 1].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 1].f16x4);
            // flash::wait_lds_data_arrived<true>(0);
        #endif
        lds_stage_id ^= 1;
        
        if constexpr (HasVTail) {
            int warp_id_m = warp_id % 2;    // w0 w2: 尾巴的 16 行 (key) 半块
            int warp_id_n = warp_id / 2;    // w1 w3: 尾巴的 32 列 (V dim) 半块
            constexpr int n_loop_ = n_loop - 1; // n_loop = kBlockK / WARP_K: 最后一个 chunk
            constexpr int n_load = 1;
            int nm_filter_max = n_loop_ * WARP_K + warp_id_m * 16 + 16 - max_seq_kv_offset;
            int real_mls_loop = (max_seq_kv_offset % kBlockK == 0 or nm_filter_max < 16) ? n_loop_: 0;
            int nm_filter     = inline_min_max<0, 16>(real_mls_loop * WARP_K + warp_id_m * 16 + 16 - max_seq_kv_offset);
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id_m * 16 * seqlen_v_stride + warp_id_n * 32 + n_load * WARP_NUM * 32) * ELEMENT_BYTES);
            v_srsrc[3] = (max_seq_kv_offset % kBlockK == 0 ? 0: nm_filter << 8) + 0x20000;
            int lds_write_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds<0, 1>(v_lds, v_srsrc, lds_write_offset, 0);
            lds_stage_id ^= 1;
        }
        int stage_id = 0;

        // Wait buffer
        if constexpr (HasVTail) {
            flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS); // 尾巴路径的 prefetch K 挪到最后, 这里统一按 V_LOAD_REQUESTS 等
        } else if constexpr (PREFETCH_K) {
            flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);
        } else {
            flash::wait_buffer_data_arrived<true>(0);
        }

        // DS dispatch
        int lds_load_offset = (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // Wait for special headdimV
            if constexpr (HasVTail) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);
                }
            }

            // DS dispatch
            int lds_load_offset = (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);
            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
        }

        stage_id ^= 1;
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        {
            constexpr int min_tile_k = 0;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
    }

    // Prefetch K (V 尾巴路径: 等 V 尾巴搬完再碰 k_lds)
    if constexpr (PREFETCH_K and HasVTail) {
        prefetch_k_to_lds_mls_ds<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, max_seq_kv_offset - kBlockK);
    }
}
