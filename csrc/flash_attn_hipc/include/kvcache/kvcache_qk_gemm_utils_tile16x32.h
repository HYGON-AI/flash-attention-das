#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "kvcache_pv_gemm_utils_tile16x32.h"

template <int WARP_NUM, int M_MMAC_COUNT, typename Element>
__forceinline__ __device__ void mixed_decode_load_q_16x64_stage(const vec4_uint q_addr, Element *const q_lds,
                                                                const int stage_id, const int head_dim_base,
                                                                const int query_seqlen_stride,
                                                                const int max_seq_q_offset, const int warp_id) {
    static_assert(WARP_NUM == 2 || WARP_NUM == 4);
    static_assert(M_MMAC_COUNT == 1 || M_MMAC_COUNT == 2);
    constexpr int kRows = 16 * M_MMAC_COUNT;
    constexpr int kColumns = 64;
    constexpr int kLoadBytes = 4;
    constexpr int kRowsPerRequest = 4;
    constexpr int kStageElements = kRows * kColumns / sizeof(Element);
    constexpr int kRequestElements = kRowsPerRequest * kColumns / sizeof(Element);
    constexpr int kLoadRequests = 4 * M_MMAC_COUNT / WARP_NUM;
    const int lane_id = threadIdx.x & 63;
    const int row_in_request = lane_id >> 4;
    const int column = (lane_id & 15) * kLoadBytes;

#pragma unroll
    for (int load = 0; load < kLoadRequests; ++load) {
        const int request_id = warp_id + load * WARP_NUM;
        const int row = min(request_id * kRowsPerRequest + row_in_request, max_seq_q_offset - 1);
        mixed_decode_buffer_load_4_raw_lds(q_lds, q_addr, stage_id * kStageElements + request_id * kRequestElements,
                                           head_dim_base, row * query_seqlen_stride + column);
    }
}

template <int M_MMAC_COUNT, typename Element>
__forceinline__ __device__ void mixed_decode_read_q_16x64_half(Element *const q_lds, const int stage_id,
                                                               const int half_id,
                                                               union_vec4_f16x2<Element> q_reg[M_MMAC_COUNT]) {
    static_assert(std::is_same_v<Element, BFloat16>);
    static_assert(M_MMAC_COUNT == 1 || M_MMAC_COUNT == 2);
    constexpr int kRows = 16 * M_MMAC_COUNT;
    constexpr int kColumns = 64;
    constexpr int kStageBytes = kRows * kColumns;
    const int lane_id = threadIdx.x & 63;
    const int lane_quarter = lane_id >> 4;
    const int row = lane_id & 15;
    // Match the existing BF16 MMAC fragment order: columns 0, 16, 8, 24.
    const int column = (lane_quarter & 1) * 16 + (lane_quarter >> 1) * 8;
    const uint8_t *const tile = reinterpret_cast<const uint8_t *>(q_lds) + stage_id * kStageBytes;
    vec2_uint input[M_MMAC_COUNT];
#pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        input[min_tile_m] =
            *reinterpret_cast<const vec2_uint *>(tile + (row + min_tile_m * 16) * kColumns + half_id * 32 + column);
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        vec4_uint output;
#pragma unroll
        for (int word = 0; word < 2; ++word) {
            output[word * 2] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input[min_tile_m][word]));
            output[word * 2 + 1] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input[min_tile_m][word] >> 16));
        }
        q_reg[min_tile_m].f32 = __builtin_bit_cast(vec4_fp32, output);
    }
}

template <int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, int STAGES,
          int M_MMAC_COUNT>
__forceinline__ __device__ void mixed_kvcache_prefetch_q_to_vgpr_16x64(
    const vec4_uint q_addr, Element *const q_lds,
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2], const int warp_id,
    const int query_seqlen_stride, const int max_seq_q_offset) {
    static_assert(std::is_same_v<Element, BFloat16>);
    static_assert(kBlockM == 32 && kBlockK == 32 && WARP_M == 32);
    static_assert(STAGES == 2);
    static_assert(M_MMAC_COUNT == 1 || M_MMAC_COUNT == 2);
    static_assert(kHeadDim % 64 == 0);
    constexpr int kRawStageCount = kHeadDim / 64;
    constexpr int kLoadRequests = 4 * M_MMAC_COUNT / WARP_NUM;
    int stage_id = 0;

    mixed_decode_load_q_16x64_stage<WARP_NUM, M_MMAC_COUNT>(q_addr, q_lds, stage_id, 0, query_seqlen_stride,
                                                            max_seq_q_offset, warp_id);

#pragma unroll
    for (int raw_stage = 0; raw_stage < kRawStageCount - 1; ++raw_stage) {
        const int next_stage_id = stage_id ^ 1;
        mixed_decode_load_q_16x64_stage<WARP_NUM, M_MMAC_COUNT>(q_addr, q_lds, next_stage_id, (raw_stage + 1) * 64,
                                                                query_seqlen_stride, max_seq_q_offset, warp_id);
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait_nosync<kLoadRequests>();
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

#pragma unroll
        for (int half_id = 0; half_id < 2; ++half_id) {
            const int logical_k_loop = raw_stage * 2 + half_id;
            mixed_decode_read_q_16x64_half<M_MMAC_COUNT>(q_lds, stage_id, half_id, q_reg + logical_k_loop * 2);
        }
        // All waves read the shared Q stage.  Do not reuse it until every
        // wave has captured its BF16 fragments in registers.
        __syncthreads();
        stage_id = next_stage_id;
    }

    __builtin_amdgcn_sched_barrier(0);
    buffer_load_lds_dwordx1_wait_nosync<0>();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
#pragma unroll
    for (int half_id = 0; half_id < 2; ++half_id) {
        const int logical_k_loop = (kRawStageCount - 1) * 2 + half_id;
        mixed_decode_read_q_16x64_half<M_MMAC_COUNT>(q_lds, stage_id, half_id, q_reg + logical_k_loop * 2);
    }
    __syncthreads();
}

template <typename Element>
__forceinline__ __device__ void mixed_decode_load_k_32x64_stage(const vec4_uint k_addr, Element *const k_lds,
                                                                const int stage_id, const int head_dim_base,
                                                                const int warp_id, const int kcache_seqlen_stride,
                                                                const int max_seq_k_offset) {
    static_assert(std::is_same_v<Element, BFloat16>);
    constexpr int kRows = 32;
    constexpr int kColumns = 64;
    constexpr int kRowsPerRequest = 16;
    constexpr int kStageElements = kRows * kColumns / sizeof(Element);
    constexpr int kRequestElements = kRowsPerRequest * kColumns / sizeof(Element);
    const int lane_id = threadIdx.x & 63;
    const int row_in_request = lane_id >> 2;
    const int column = (lane_id & 3) * 16;
    const int wave_stage = warp_id * 2 + stage_id;

#pragma unroll
    for (int load = 0; load < 2; ++load) {
        const int row = min(warp_id * 32 + load * kRowsPerRequest + row_in_request, max_seq_k_offset - 1);
        fp8_buffer_load_16_raw_lds(k_lds, k_addr, wave_stage * kStageElements + load * kRequestElements, head_dim_base,
                                   row * kcache_seqlen_stride + column);
    }
}

template <typename Element>
__forceinline__ __device__ void mixed_decode_read_k_32x64_half(Element *const k_lds, const int stage_id,
                                                               const int half_id, const int warp_id,
                                                               union_vec4_f16x2<Element> k_reg[2]) {
    static_assert(std::is_same_v<Element, BFloat16>);
    constexpr int kRows = 32;
    constexpr int kColumns = 64;
    constexpr int kStageBytes = kRows * kColumns;
    const int lane_id = threadIdx.x & 63;
    const int lane_quarter = lane_id >> 4;
    const int row = lane_id & 15;
    const int column = (lane_quarter & 1) * 16 + (lane_quarter >> 1) * 8;
    const uint8_t *const tile = reinterpret_cast<const uint8_t *>(k_lds) + (warp_id * 2 + stage_id) * kStageBytes;
    const vec2_uint input0 = *reinterpret_cast<const vec2_uint *>(tile + row * kColumns + half_id * 32 + column);
    const vec2_uint input1 = *reinterpret_cast<const vec2_uint *>(tile + (row + 16) * kColumns + half_id * 32 + column);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);

    vec4_uint output0;
    vec4_uint output1;
#pragma unroll
    for (int word = 0; word < 2; ++word) {
        output0[word * 2] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input0[word]));
        output0[word * 2 + 1] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input0[word] >> 16));
        output1[word * 2] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input1[word]));
        output1[word * 2 + 1] = e5m2x2_to_bf16x2(static_cast<uint16_t>(input1[word] >> 16));
    }
    k_reg[0].f32 = __builtin_bit_cast(vec4_fp32, output0);
    k_reg[1].f32 = __builtin_bit_cast(vec4_fp32, output1);
}

template <int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void mixed_kvcache_prefetch_k_to_lds_32x64(const vec4_uint k_addr, Element *const k_lds,
                                                                      const int warp_id, const int kcache_seqlen_stride,
                                                                      const int max_seq_k_offset) {
    static_assert(kBlockK == 32 && WARP_N == 32 && STAGES == 2);
    static_assert(WARP_NUM == 2 || WARP_NUM == 4);
    mixed_decode_load_k_32x64_stage(k_addr, k_lds, 0, 0, warp_id, kcache_seqlen_stride, max_seq_k_offset);
    __builtin_amdgcn_sched_barrier(0);
}

template <int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM,
          int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mixed_kvcache_qk_gemm_prefetch_v_32x64(
    const vec4_uint k_addr, const vec4_uint v_addr, Element *const k_lds, Element *const v_lds,
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
    vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4], const int warp_id, const int kcache_seqlen_stride,
    const int vcache_seqlen_stride, const int max_seq_k_offset) {
    static_assert(kBlockM == 32 && kBlockN == 32 && kBlockK == 32);
    static_assert(WARP_M == 32 && WARP_N == 32 && STAGES == 2);
    static_assert(M_MMAC_COUNT == 1 || M_MMAC_COUNT == 2);
    static_assert(WARP_NUM == 2 || WARP_NUM == 4);
    static_assert(kHeadDim % 64 == 0 && kHeadDimV % 64 == 0);
    constexpr int kRawStageCount = kHeadDim / 64;
    union_vec4_f16x2<Element> k_reg[2];

    uint64_t zero = 0;
#pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                s_reg[m_idx][min_tile_n * 2 + min_tile_m].u64[0] = zero;
                s_reg[m_idx][min_tile_n * 2 + min_tile_m].u64[1] = zero;
            }
        }
    }

    int stage_id = 0;
#pragma unroll
    for (int raw_stage = 0; raw_stage < kRawStageCount - 1; ++raw_stage) {
        const int next_stage_id = stage_id ^ 1;
        mixed_decode_load_k_32x64_stage(k_addr, k_lds, next_stage_id, (raw_stage + 1) * 64, warp_id,
                                        kcache_seqlen_stride, max_seq_k_offset);
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait_nosync<2>();
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (int half_id = 0; half_id < 2; ++half_id) {
            mixed_decode_read_k_32x64_half(k_lds, stage_id, half_id, warp_id, k_reg);
            asm volatile("s_setprio 1");
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
#pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
#pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        const int q_tile_id = (raw_stage * 2 + half_id) * 2 + min_tile_m;
                        s_reg[0][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                            vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                  q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                  q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                  q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                            vec4_Element<Element>{k_reg[min_tile_n].f16x2[2 * min_tile_k][0],
                                                  k_reg[min_tile_n].f16x2[2 * min_tile_k][1],
                                                  k_reg[min_tile_n].f16x2[2 * min_tile_k + 1][0],
                                                  k_reg[min_tile_n].f16x2[2 * min_tile_k + 1][1]},
                            s_reg[0][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            asm volatile("s_setprio 0");
        }
        stage_id = next_stage_id;
    }

    __builtin_amdgcn_sched_barrier(0);
    buffer_load_lds_dwordx1_wait_nosync<0>();
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int half_id = 0; half_id < 2; ++half_id) {
        mixed_decode_read_k_32x64_half(k_lds, stage_id, half_id, warp_id, k_reg);
        asm volatile("s_setprio 1");
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
#pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
#pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    const int q_tile_id = ((kRawStageCount - 1) * 2 + half_id) * 2 + min_tile_m;
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                        vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                              q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                              q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                              q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                        vec4_Element<Element>{k_reg[min_tile_n].f16x2[2 * min_tile_k][0],
                                              k_reg[min_tile_n].f16x2[2 * min_tile_k][1],
                                              k_reg[min_tile_n].f16x2[2 * min_tile_k + 1][0],
                                              k_reg[min_tile_n].f16x2[2 * min_tile_k + 1][1]},
                        s_reg[0][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        asm volatile("s_setprio 0");
    }

    // K is no longer needed.  Reuse its compact LDS allocation for the
    // first V 32x64 raw tile while mask/softmax work follows in registers.
    __syncthreads();
    mixed_decode_load_v_32x64_stage(v_addr, v_lds, 0, kHeadDimV - 64, warp_id, vcache_seqlen_stride, max_seq_k_offset);
}


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, int STAGES, int M_MMAC_COUNT>
__forceinline__ __device__ void kvcache_prefetch_q_to_vgpr_tile16x32(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset=-1) {

    constexpr int Q_LOAD_REQUESTS = (kBlockM * kBlockK >> 1/*16x32 tile*/) * M_MMAC_COUNT / (4 * 32 * WARP_NUM);
    constexpr int SEQUENCE_READ   = M_MMAC_COUNT;
    constexpr int READ_ONCE_LINES = 4;
    auto BUFFER_LOAD_FUNC         = &builtin_buffer_load_dword_lds<Element, float, 1>; // buffer_load_dwordx4 can also be applied if necessary

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = lane_id >> 4;
    int q_lane_head_dim_idx = lane_id & 15;
    int laneid_shfl_4       = lane_id >> 4;
    int laneid_and_15       = lane_id & 15;
    int q_ds_read_offset    = laneid_and_15 * 16 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32) * 4;

    int stage_id = 0;
    if constexpr (STAGES > 1) {
        int k_loop = 0;
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 32);
        for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / READ_ONCE_LINES - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 32) + (q_warp_buffer_load_m_id & 7) * (READ_ONCE_LINES * 32);
            int offset_s = q_block_buffer_load_global_offset / 2;
            int offset_v = q_warp_buffer_load_m_id * READ_ONCE_LINES + q_lane_m_idx;
            int lds_offset = q_warp_buffer_load_lds_offset / 2;
            offset_v = (min(offset_v, max_seq_q_offset - 1) * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(q_lds, q_addr, lds_offset, offset_s, offset_v);
        }
    }
    if constexpr (STAGES > 1) stage_id ^= 1;
    constexpr int K_LOOP_START = (STAGES > 1) ? 1: 0;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 32);
        for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / READ_ONCE_LINES - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + ((q_warp_buffer_load_m_id >> 3) * (32 * 32) + (q_warp_buffer_load_m_id & 7) * (READ_ONCE_LINES * 32));
            int offset_s = q_block_buffer_load_global_offset / 2;
            int offset_v = q_warp_buffer_load_m_id * READ_ONCE_LINES + q_lane_m_idx;
            int lds_offset = q_warp_buffer_load_lds_offset / 2;
            offset_v = (min(offset_v, max_seq_q_offset - 1) * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(q_lds, q_addr, lds_offset, offset_s, offset_v);
        }

        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if constexpr (STAGES > 1) stage_id ^= 1;
        q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 16);

        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int i = 0; i < SEQUENCE_READ; ++i) {
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset;
                    int k_loop_idx = (STAGES > 1) ? k_loop - 1: k_loop;
                    q_reg[k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].f32 = *(vec4_fp32*)(q_lds_v2fp16 + lds_offset);
                }
            }
        }
        __syncthreads();
        // __builtin_amdgcn_sched_barrier(0);
    }

    if constexpr (STAGES > 1) {
        __builtin_amdgcn_s_waitcnt(0);
        stage_id ^= 1;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 16);
        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int i = 0; i < SEQUENCE_READ; ++i) {
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset;
                    q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].f32 = *(vec4_fp32*)(q_lds_v2fp16 + lds_offset);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
}



template<int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void kvcache_prefetch_k_to_lds_tile16x32(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    // 预先计算一些表达式
    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;

#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    int qk_lane_m_idx             = lane_id >> 2;
    int qk_lane_head_dim_idx      = (lane_id & 3) << 2;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dwordx4_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 16;
#else // zd
    int qk_lane_m_idx             = laneid_shfl_4;
    int qk_lane_head_dim_idx      = laneid_and_15;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dword_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 4;
#endif
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (READ_ONCE_LINES * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    int stage_id = 0;
    int k_loop   = 0;
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
        #pragma unroll
        for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
            int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32);
            int lds_offset = k_warp_buffer_load_lds_offset / 2;
            int offset_s   = k_block_buffer_load_global_offset / 2;
            int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = (k_loop + 1) * kBlockK;
        int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2 + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
        #pragma unroll
        for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
            int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32);
            int lds_offset = k_warp_buffer_load_lds_offset / 2;
            int offset_s   = k_block_buffer_load_global_offset / 2;
            int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}
