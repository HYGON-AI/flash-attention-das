// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// ===========================================================================
// gfx938 / gfx936 原生 16 位 PA 实现（BF16/FP16 Q、同类型 K/V），三种数据布局共用。
//
// 本文件不是普通头文件：没有 include guard，由各布局的 .cu 在自己的命名空间里包含，每包含一次生成一套实现：
//   paged_attention_refactoring.cu       l0_b16_gfx938 / l0_b16_gfx936     layout=0：K [blocks, heads, 64, D]，V^T [blocks, heads, D, 64]
//   paged_attention_refactoring_bshd.cu  bshd_b16_gfx938 / bshd_b16_gfx936 BSHD：K/V [blocks, 64, heads, D]
//   paged_attention_refactoring_bhsd.cu  bhsd_b16_gfx938                   BHSD：K/V [blocks, heads, 64, D]
// kernel 名与原来相同（如 bshd_b16_gfx938::pa_bshd_b16），每个 kernel 的指令与合并前逐条相同。
//
// 包含前需定义（包含结束时本文件会 #undef）：
//   PA_B16_LAYOUT               PA_LAYOUT_L0 / PA_LAYOUT_BHSD / PA_LAYOUT_BSHD
//   PA_B16_ARCH                 938 或 936
//   PA_B16_PREFIX               kernel 名前缀，如 pa_bshd_b16（合并 kernel 为 <前缀>_combine 等）
//   PA_B16_REDUCE_BF16          BF16 输出时跨 wave 归并是否用 BF16（各布局的编译开关，默认 1）
//   PA_B16_WIDE_GRID_MULTIPLIER 只有 gfx938 需要：每组超过 16 行时长序列分区的放宽倍数
//
// 布局之间的差别（K/V 寻址；layout=0 的 V 已转置，直接从全局读入寄存器作 PV 的 B 片段，不经 LDS）
// 用 #if PA_B16_LAYOUT == ... 区分；架构之间只有分区数的取法不同，用 #if PA_B16_ARCH == 936 区分。
// 各布局的适用性检查与回退在各自 .cu 的入口函数中，本文件的 launch() 只负责检查通过之后的公共流程。
// ===========================================================================

#ifndef PA_LAYOUT_L0
#define PA_LAYOUT_L0 0      // layout=0（custom）：V 转置
#define PA_LAYOUT_BHSD 1    // layout=1
#define PA_LAYOUT_BSHD 2    // layout=2
#endif

#if !defined(PA_B16_LAYOUT) || !defined(PA_B16_ARCH) || !defined(PA_B16_PREFIX) || !defined(PA_B16_REDUCE_BF16)
#error "paged_attention_refactoring_b16_gfx938_gfx936.h: 包含前需定义 PA_B16_LAYOUT / PA_B16_ARCH / PA_B16_PREFIX / PA_B16_REDUCE_BF16"
#endif

// 当前是否为本实现所属架构的设备编译（host 编译与其他架构的设备编译中 kernel 体为空）
#if (PA_B16_ARCH == 938 && defined(__gfx938__)) || (PA_B16_ARCH == 936 && defined(__gfx936__))
#define PA_B16_DEVICE 1
#else
#define PA_B16_DEVICE 0
#endif

// kernel 名：<前缀>、<前缀>_combine、<前缀>_combine_column_waves、<前缀>_combine_column_waves128
#define PA_B16_CAT_(a, b) a##b
#define PA_B16_CAT(a, b) PA_B16_CAT_(a, b)
#define PA_B16_KERNEL PA_B16_PREFIX
#define PA_B16_COMBINE PA_B16_CAT(PA_B16_PREFIX, _combine)
#define PA_B16_COMBINE_CW PA_B16_CAT(PA_B16_PREFIX, _combine_column_waves)
#define PA_B16_COMBINE_CW128 PA_B16_CAT(PA_B16_PREFIX, _combine_column_waves128)

// 调试打印与报错信息中的名字
#if PA_B16_LAYOUT == PA_LAYOUT_L0
#define PA_B16_LAYOUT_STR "L0"
#define PA_B16_LAYOUT_DESC "layout=0"
#elif PA_B16_LAYOUT == PA_LAYOUT_BHSD
#define PA_B16_LAYOUT_STR "BHSD"
#define PA_B16_LAYOUT_DESC "BHSD"
#else
#define PA_B16_LAYOUT_STR "BSHD"
#define PA_B16_LAYOUT_DESC "BSHD"
#endif
#if PA_B16_ARCH == 936
#define PA_B16_ARCH_STR "gfx936"
#else
#define PA_B16_ARCH_STR "gfx938"
#endif

// ===== 设备端：kernel 与辅助函数 =====

using rsrc_t = int __attribute__((ext_vector_type(4)));
using u16x4_t = uint16_t __attribute__((ext_vector_type(4)));
using u16x8_t = uint16_t __attribute__((ext_vector_type(8)));
#if PA_B16_DEVICE
using dwords4_t = uint32_t __attribute__((ext_vector_type(4)));
#if PA_B16_LAYOUT == PA_LAYOUT_L0
using dwords2_t = uint32_t __attribute__((ext_vector_type(2)));
__device__ dwords4_t buffer_load128(rsrc_t, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ dwords2_t buffer_load64(rsrc_t, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
#else
__device__ dwords4_t buffer_load128(rsrc_t, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
#endif
#endif

// Partition accounting unit (tokens), shared with the combine kernels.
constexpr int CHUNK = 32;
#if PA_B16_LAYOUT == PA_LAYOUT_L0
#else
// V staging: 64 dims (128 bytes) per token row; 136-byte rows put the four lane groups of a fragment read on
// different LDS banks.
constexpr int V_GROUP_DIMS = 64;
constexpr int V_ROW_BYTES = 136;
#endif

#if PA_B16_DEVICE
__device__ __forceinline__ rsrc_t make_rsrc(const uint8_t* base, int bytes)
{
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    return {static_cast<int>(a), static_cast<int>(a >> 32), bytes, 0x00020000};
}

__device__ __forceinline__ u16x8_t load16(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(u16x8_t, buffer_load128(r, voff, soff, 0));
}

#if PA_B16_LAYOUT == PA_LAYOUT_L0
__device__ __forceinline__ u16x4_t load8(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(u16x4_t, buffer_load64(r, voff, soff, 0));
}

#else
#endif
template <bool F16>
__device__ __forceinline__ void mmac16(const u16x4_t& a, const u16x4_t& b, float4_t& c)
{
    if constexpr (F16) c = __builtin_hcu_mmac_f32_16x16x16_f16(__builtin_bit_cast(half4_t, a), __builtin_bit_cast(half4_t, b), c);
    else c = __builtin_hcu_mmac_f32_16x16x16_bf16(__builtin_bit_cast(v4bh, a), __builtin_bit_cast(v4bh, b), c);
}
#endif

// Large Q tiles stay in LDS so the accumulator, K and V fragments fit in VGPRs.
#if PA_B16_LAYOUT == PA_LAYOUT_L0
__host__ __device__ constexpr bool q_in_lds(int d, int mw) { return d * mw >= 256; }
__host__ __device__ constexpr int q_lds_stride(int d) { return 2 * d + 16; }      // bytes
#else
__host__ __device__ constexpr bool q_in_lds(int d, int mw) { return d * mw > 256; }
__host__ __device__ constexpr int q_lds_stride(int d) { return 2 * d + 16; }      // bytes
// Tokens a wave consumes per step: K and V fragments of a step take D / 4 bytes per lane each.
__host__ __device__ constexpr int step_tokens(int d) { return d <= 128 ? 32 : 16; }
#endif
// Epilogue merge width: one 16-row x RED_COLS block of every wave per round.
__host__ __device__ constexpr int red_cols(int waves, int d) { return (waves == 4 && d % 128 == 0) ? 128 : 64; }

// Partitions actually used: at most one CTA per CU (one_wave_parts) unless the
// sequence is long enough for the wider grid to pay for its extra partials.
__device__ __forceinline__ int valid_partitions(int chunks, int grid_parts, int one_wave_parts, int min_chunks)
{
#if PA_B16_ARCH == 936
    if (grid_parts > one_wave_parts) {
        if (chunks >= 8 * grid_parts) return grid_parts;
        const int two = MIN_T(grid_parts, 2 * one_wave_parts);
        if (two > one_wave_parts && chunks >= 4 * two) return two;
    }
#else
    if (grid_parts > one_wave_parts && chunks >= 32 * grid_parts) return grid_parts;
#endif
    return MAX_T(1, MIN_T(one_wave_parts, DIVIDE_ROUND_UP(chunks, min_chunks)));
}

#if PA_B16_LAYOUT == PA_LAYOUT_L0
// CH: tokens a wave consumes per step (16, 32 or 64). Lane group g scores TPG = CH / 4 consecutive tokens, i.e.
// TPG consecutive elements of every V^T row: 8-byte (CH = 16), 16-byte (CH = 32) or 2 x 16-byte (CH = 64) V loads.
template <typename scalar_t, int D, int MW, int CH, int S, bool BF16RED>
#else
template <typename scalar_t, int D, int MW, int S, bool BF16RED>
#endif
__launch_bounds__(S * WARP_SIZE)
__global__ void PA_B16_KERNEL(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k_cache, const scalar_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks)
{
#if PA_B16_DEVICE
    extern __shared__ char shared_mem[];
#if PA_B16_LAYOUT == PA_LAYOUT_L0
    static_assert(CH == 16 || CH == 32 || CH == 64, "layout=0 16-bit: CH must be 16, 32 or 64");
    constexpr bool F16 = std::is_same<scalar_t, _Float16>::value;
    constexpr int KB = D / 32;              // 32-dim (64-byte) K / Q blocks
    constexpr int NT = D / 16;              // 16-dim output tiles
    constexpr int NI = CH / 16;             // 16-token score tiles per step
    constexpr int TPG = CH / 4;             // consecutive tokens per lane group
    constexpr int VW = (TPG + 7) / 8;       // V^T pieces per output tile: one 8-byte (CH = 16) or 16-byte pieces
    using vpiece_t = typename std::conditional<CH == 16, u16x4_t, u16x8_t>::type;
#else
    constexpr bool F16 = std::is_same<scalar_t, _Float16>::value;
    constexpr int CH = step_tokens(D);
    constexpr int KB = D / 32;              // 32-dim (64-byte) K / Q blocks
    constexpr int NT = D / 16;              // 16-dim output tiles
    constexpr int NI = CH / 16;             // 16-token score tiles per step
    constexpr int NG = D / V_GROUP_DIMS;    // 64-dim V staging groups
#endif
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = blockIdx.x % row_groups;
    const int part = blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    // Partition count in 32-token units (shared with the combine kernels); work is split in CH-token steps.
    const int nparts = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), grid_parts, one_wave_parts, min_chunks);
    if (part >= nparts) return;
    const int chunks = DIVIDE_ROUND_UP(valid, CH);

    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int r = lane & 15;
    const int g = lane >> 4;
    const int hpk = num_q_heads / num_kv_heads;
    const int R = hpk * mtp;
    const int row0 = rg * ROWS;
    const int num_heads = num_q_heads * mtp;

    const int c_begin = static_cast<int>(static_cast<int64_t>(part) * chunks / nparts);
    const int c_end = static_cast<int>(static_cast<int64_t>(part + 1) * chunks / nparts);

    const int* table = block_tables + static_cast<int64_t>(b) * max_num_blocks_per_seq;
#if PA_B16_LAYOUT == PA_LAYOUT_BHSD
    // Byte addressing (BHSD): a page holds the 64 token rows (D elements each) of every KV head contiguously.
    constexpr int token_stride = D * 2;
    const uint8_t* k_head = reinterpret_cast<const uint8_t*>(k_cache) + static_cast<int64_t>(kv_head) * (64 * D * 2);
    const uint8_t* v_head = reinterpret_cast<const uint8_t*>(v_cache) + static_cast<int64_t>(kv_head) * (64 * D * 2);
#elif PA_B16_LAYOUT == PA_LAYOUT_L0
    // Byte addressing. K page-head: [64 tokens][D]; V page-head: [D][64 tokens] (128-byte rows).
    const uint8_t* k_head = reinterpret_cast<const uint8_t*>(k_cache) + static_cast<int64_t>(kv_head) * (64 * D * 2);
    const uint8_t* v_head = reinterpret_cast<const uint8_t*>(v_cache) + static_cast<int64_t>(kv_head) * (64 * D * 2);
#else
    // Byte addressing: token rows interleave the KV heads.
    const int token_stride = num_kv_heads * D * 2;
    const uint8_t* k_head = reinterpret_cast<const uint8_t*>(k_cache) + kv_head * D * 2;
    const uint8_t* v_head = reinterpret_cast<const uint8_t*>(v_cache) + kv_head * D * 2;
#endif
    auto page_of = [&](int c) { return c / (64 / CH); };
#if PA_B16_LAYOUT == PA_LAYOUT_L0
    auto tok0_of = [&](int c) { return (c % (64 / CH)) * CH; };

    // K fragment rows: tile i column n holds step token TPG*(n&3) + 4*i + (n>>2). With the MMAC result layout
    // (lane (r, g) value j = column g + 4j) lane group g scores tokens TPG*g + 4i + j, so the P fragment of tile i
    // pairs with elements 4i .. 4i + 3 of the lane's TPG-token V^T piece. Lane group g holds dims 32kb + 8g .. +7
    // of every 32-dim block, matching the Q fragment, so the two MMACs of a block use halves 0-3 and 4-7.
    const int k_voff = (TPG * (r & 3) + (r >> 2)) * (2 * D) + 16 * g;
    // V^T piece of output tile t: dim 16t + r, step tokens TPG*g .. TPG*g + TPG - 1.
    const int v_voff = r * 128 + 2 * TPG * g;

    u16x8_t kf[NI][KB];
    vpiece_t vf[NT][VW];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(k_head + static_cast<int64_t>(phys) * k_block_stride * 2 + tok0_of(c) * (2 * D), CH * 2 * D);
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int kb = 0; kb < KB; ++kb) kf[i][kb] = load16(rs, k_voff, 4 * i * (2 * D) + 64 * kb);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(v_head + static_cast<int64_t>(phys) * v_block_stride * 2, 64 * D * 2);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            if constexpr (CH == 16) {
                vf[t][0] = load8(rs, v_voff, t * 16 * 128 + tok0_of(c) * 2);
            } else {
                #pragma unroll
                for (int w = 0; w < VW; ++w) vf[t][w] = load16(rs, v_voff + 16 * w, t * 16 * 128 + tok0_of(c) * 2);
            }
        }
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[page_of(c)]; issue_k(c, phys); issue_v(c, phys); }

    // Q: register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem);
#else
    auto step_base = [&](const uint8_t* head, int block_stride, int c, int phys) {
        return head + static_cast<int64_t>(phys) * block_stride * 2 + (c % (64 / CH)) * CH * token_stride;
    };

    // K fragment rows: tile i column n holds step token 16*i + 4*(n&3) + (n>>2). With the MMAC result layout
    // (lane (r, g) value j = column g + 4j) lane group g scores tokens 16i + 4g .. 16i + 4g + 3, which is the
    // k-group order of the P fragment. Lane group g holds dims 32kb + 8g .. 32kb + 8g + 7 of every 32-dim block,
    // matching the Q fragment, so the two 16x16x16 MMACs of a block use halves 0-3 and 4-7.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * token_stride + 16 * g;
    // V rows: load (i, cb) covers step rows 16i + (lane >> 2), bytes 64cb + 16 (lane & 3) of the head row.
    const int v_row = lane >> 2;
    const int v_voff = v_row * token_stride + 16 * (lane & 3);

    u16x8_t kf[NI][KB];
    u16x8_t vf[NI][KB];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(step_base(k_head, k_block_stride, c, phys), CH * token_stride);
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int kb = 0; kb < KB; ++kb) kf[i][kb] = load16(rs, k_voff, 16 * i * token_stride + 64 * kb);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(step_base(v_head, v_block_stride, c, phys), CH * token_stride);
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int cb = 0; cb < KB; ++cb) vf[i][cb] = load16(rs, v_voff, 16 * i * token_stride + 64 * cb);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[page_of(c)]; issue_k(c, phys); issue_v(c, phys); }

    // Q: register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* v_lds = reinterpret_cast<uint8_t*>(shared_mem) + warp * (CH * V_ROW_BYTES);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * CH * V_ROW_BYTES;
#endif
    auto load_q8 = [&](int gr, int col) {     // 8 elements of Q row gr starting at column col
        u16x8_t v = {0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            v = *reinterpret_cast<const u16x8_t*>(q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col);
        }
        return v;
    };
    u16x8_t qf[QL ? 1 : MW][QL ? 1 : KB];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 8); i += S * WARP_SIZE) {
            const int row = i / (D / 8), col = (i % (D / 8)) * 8;
            *reinterpret_cast<u16x8_t*>(q_lds + row * QS + col * 2) = load_q8(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kb = 0; kb < KB; ++kb) qf[m][kb] = load_q8(row0 + m * 16 + r, 32 * kb + 8 * g);
    }
    auto q_frag = [&](int m, int kb) {
        if constexpr (QL) return *reinterpret_cast<const u16x8_t*>(q_lds + (m * 16 + r) * QS + 64 * kb + 16 * g);
        else return qf[m][kb];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
    float4_t acc[MW][NT];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < NT; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Peel the known-zero initial recurrence at compile time.
    auto process_chunk = [&]<bool INITIAL>() {
        const int next = c + S;
        const int next_phys = next < c_end ? table[page_of(next)] : 0;
        const int base_tok = c * CH;
        // ---- S = Q K^T over the CH-token step.
        float4_t s[MW][NI];
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int i = 0; i < NI; ++i) s[m][i] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const u16x8_t qcur = q_frag(m, kb);
                const u16x4_t* qv = reinterpret_cast<const u16x4_t*>(&qcur);
                #pragma unroll
                for (int i = 0; i < NI; ++i) {
                    const u16x4_t* kv = reinterpret_cast<const u16x4_t*>(&kf[i][kb]);
                    mmac16<F16>(qv[0], kv[0], s[m][i]);
                    mmac16<F16>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only steps reaching the last mtp tokens.
        if (base_tok + CH - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < NI; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
#if PA_B16_LAYOUT == PA_LAYOUT_L0
                        if (base_tok + TPG * g + 4 * i + j > limit[m]) s[m][i][j] = -INFINITY;
#else
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
#endif
        }

        // ---- Online softmax.
        u16x4_t p[MW][NI];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                mx = MAX_T(mx, MAX_T(MAX_T(s[m][i][0], s[m][i][1]), MAX_T(s[m][i][2], s[m][i][3])));
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            float alpha = 1.0f;
            if constexpr (!INITIAL)
                alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[m][i][j] = __builtin_amdgcn_exp2f(s[m][i][j] * scale_log2 - base);
                    sum += s[m][i][j];
                }
            sum += __shfl_xor_tmp(sum, 32);
            sum += __shfl_swap16(sum);
            if constexpr (INITIAL) {
                row_sum[m] = sum;
            } else {
                row_sum[m] = row_sum[m] * alpha + sum;
                #pragma unroll
                for (int i = 0; i < NT; ++i) acc[m][i] *= alpha;
            }
            row_max[m] = new_max;
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) p[m][i][j] = __builtin_bit_cast(uint16_t, from_float<scalar_t>(s[m][i][j]));
        }

        // ---- V: zero rows past the sequence end (P is already zero there), stage 64 dims at a time,
        // read 4-token columns, MMAC.
        if (base_tok + CH > valid) {
#if PA_B16_LAYOUT == PA_LAYOUT_L0
            const int nv = valid - (base_tok + TPG * g);       // valid elements of this lane's V^T piece
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                #pragma unroll
                for (int w = 0; w < VW; ++w)
                    #pragma unroll
                    for (int e = 0; e < (CH == 16 ? 4 : 8); ++e)
                        if (8 * w + e >= nv) vf[t][w][e] = 0;
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            #pragma unroll
            for (int i = 0; i < NI; ++i) {
                u16x4_t fv;
                if constexpr (CH == 16) {
                    fv = vf[t][0];
                } else {
                    const u16x8_t piece = vf[t][i / 2];
                    fv = (i % 2 == 0) ? u16x4_t{piece[0], piece[1], piece[2], piece[3]}
                                      : u16x4_t{piece[4], piece[5], piece[6], piece[7]};
                }
                #pragma unroll
                for (int m = 0; m < MW; ++m) mmac16<F16>(p[m][i], fv, acc[m][t]);
            }
        if (next < c_end) issue_v(next, next_phys);
#else
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                #pragma unroll
                for (int cb = 0; cb < KB; ++cb)
                    if (base_tok + 16 * i + v_row >= valid) vf[i][cb] = u16x8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int vg = 0; vg < NG; ++vg) {
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint8_t* dst = v_lds + (16 * i + v_row) * V_ROW_BYTES + 64 * h + 16 * (lane & 3);
                    const u16x8_t x = vf[i][2 * vg + h];
                    *reinterpret_cast<uint64_t*>(dst) = __builtin_bit_cast(uint64_t, u16x4_t{x[0], x[1], x[2], x[3]});
                    *reinterpret_cast<uint64_t*>(dst + 8) = __builtin_bit_cast(uint64_t, u16x4_t{x[4], x[5], x[6], x[7]});
                }
            asm volatile("" ::: "memory");
            if (vg + 1 == NG && next < c_end) issue_v(next, next_phys);
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                #pragma unroll
                for (int tl = 0; tl < 4; ++tl) {
                    // B fragment: dim 16tl + r of this group, tokens 16i + 4g .. 16i + 4g + 3.
                    const uint8_t* src = v_lds + (16 * i + 4 * g) * V_ROW_BYTES + 2 * (16 * tl + r);
                    u16x4_t fv;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) fv[e] = *reinterpret_cast<const uint16_t*>(src + e * V_ROW_BYTES);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) mmac16<F16>(p[m][i], fv, acc[m][4 * vg + tl]);
                }
            asm volatile("" ::: "memory");
        }
#endif
    };
    if (c < c_end) {
        process_chunk.template operator()<true>();
        c += S;
    }
    for (; c < c_end; c += S)
        process_chunk.template operator()<false>();

    // ---- Merge the S waves of this CTA (same rows, disjoint steps).
    __syncthreads();
    float* ms_max = reinterpret_cast<float*>(shared_mem);
    float* ms_sum = ms_max + S * ROWS;
    // Accumulators and max/sum stay FP32. BF16 output can use BF16
    // inter-wave exchange; other dtypes retain FP32 exchange.
    using red_t = typename std::conditional<BF16RED, scalar_t, float>::type;
    red_t* red = reinterpret_cast<red_t*>(ms_sum + S * ROWS + 16);
    if (g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            ms_max[warp * ROWS + m * 16 + r] = row_max[m];
            ms_sum[warp * ROWS + m * 16 + r] = row_sum[m];
        }
    }
    __syncthreads();
    float fin_max[MW], fin_sum[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        float M = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < S; ++w) M = MAX_T(M, ms_max[w * ROWS + m * 16 + r]);
        float L = 0.0f;
        #pragma unroll
        for (int w = 0; w < S; ++w)
            L += ms_sum[w * ROWS + m * 16 + r] * __builtin_amdgcn_exp2f((ms_max[w * ROWS + m * 16 + r] - M) * scale_log2);
        fin_max[m] = M;
        fin_sum[m] = L;
        const float f = __builtin_amdgcn_exp2f((row_max[m] - M) * scale_log2) / (L + 1e-6f);
        #pragma unroll
        for (int i = 0; i < NT; ++i) acc[m][i] *= f;
    }

    const bool final_out = nparts == 1;
    constexpr int RC = red_cols(S, D);
    constexpr int RS = RC + 4;
    const int tid = threadIdx.x;
    using rx4 = __attribute__((__vector_size__(4 * sizeof(red_t)))) red_t;
    using sx8 = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    auto head_of = [&](int gr) { return (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk; };
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        #pragma unroll
        for (int blk = 0; blk < D / RC; ++blk) {
            __syncthreads();
            // acc tile t holds dims 16t + g + 4j.
            #pragma unroll
            for (int tl = 0; tl < RC / 16; ++tl)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    red[(warp * 16 + r) * RS + 16 * tl + 4 * j + g] =
                        from_float<red_t>(acc[m][blk * (RC / 16) + tl][j]);
            __syncthreads();
            for (int item = tid; item < 16 * RC / 8; item += S * WARP_SIZE) {
                const int rr = item / (RC / 8);
                const int d8 = (item % (RC / 8)) * 8;
                float sum8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w) {
                    const rx4 lo = *reinterpret_cast<const rx4*>(red + (w * 16 + rr) * RS + d8);
                    const rx4 hi = *reinterpret_cast<const rx4*>(red + (w * 16 + rr) * RS + d8 + 4);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        sum8[e] += to_float(lo[e]);
                        sum8[e + 4] += to_float(hi[e]);
                    }
                }
                const int gr = row0 + m * 16 + rr;
                if (gr < R) {
                    const int head_all = head_of(gr);
                    sx8 o;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) o[e] = from_float<scalar_t>(sum8[e]);
                    scalar_t* dst = final_out
                        ? out + (static_cast<int64_t>(b) * num_heads + head_all) * D
                        : out_tmp + out_tmp_offset + ((static_cast<int64_t>(b) * num_heads + head_all) * grid_parts + part) * D;
                    *reinterpret_cast<sx8*>(dst + blk * RC + d8) = o;
                }
            }
        }
    }
    if (final_out) return;
    if (warp == 0 && g == 0) {
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            const int gr = row0 + m * 16 + r;
            if (gr < R) {
                const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(gr)) * grid_parts + part;
                reinterpret_cast<float*>(out_tmp)[off] = fin_sum[m];
                reinterpret_cast<float*>(out_tmp + max_tmp_offset)[off] = fin_max[m] * scale_log2;
            }
        }
    }
#endif
}

// f16 -> f32 as a separate v_cvt_f32_f16. On gfx938 the compiler otherwise folds the conversion of the high half
// of a packed FP16 pair into v_fma_mix_f32 op_sel, which reads the low half: FP16 combines then repeat every
// even output column in the following odd one.
template <typename T>
__device__ __forceinline__ float combine_to_float(T x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<T, _Float16>::value) {
        float f;
        asm("v_cvt_f32_f16 %0, %1" : "=v"(f) : "v"(x));
        return f;
    }
#endif
    return to_float(x);
}

template <typename scalar_t, int D>
__global__ __launch_bounds__(256) void PA_B16_COMBINE(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    __shared__ float weight[1024];
    __shared__ float partial[4 * D];
    __shared__ float scalars[8];
    const int tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), grid_parts, one_wave_parts, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    float lmax = -INFINITY;
    for (int i = tid; i < np; i += 256) lmax = MAX_T(lmax, maxs[i]);
    for (int mask = 32; mask > 0; mask >>= 1) lmax = MAX_T(lmax, __shfl_xor(lmax, mask));
    if (lane == 0) scalars[warp] = lmax;
    __syncthreads();
    const float gmax = MAX_T(MAX_T(scalars[0], scalars[1]), MAX_T(scalars[2], scalars[3]));
    float lsum = 0.0f;
    for (int i = tid; i < np; i += 256) {
        const float s = sums[i];
        const float w = s > 0.0f ? s * __builtin_amdgcn_exp2f(maxs[i] - gmax) : 0.0f;
        weight[i] = w;
        lsum += w;
    }
    for (int mask = 32; mask > 0; mask >>= 1) lsum += __shfl_xor(lsum, mask);
    __syncthreads();
    if (lane == 0) scalars[4 + warp] = lsum;
    __syncthreads();
    const float inv = __fdividef(1.0f, (scalars[4] + scalars[5]) + (scalars[6] + scalars[7]) + 1e-6f);
    // A three-element vector has padded size/alignment and cannot be stored
    // at a six-byte stride. Use three scalar tiles for D=192 instead.
    constexpr int V = D % 128 == 0 ? D / 64 : 1;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    #pragma unroll
    for (int tile = 0; tile < D / (64 * V); ++tile) {
    const int column = tile * 64 * V + lane * V;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + column;
    float acc[V];
    #pragma unroll
    for (int i = 0; i < V; ++i) acc[i] = 0.0f;
    int j = warp;
    for (; j + 12 < np; j += 16) {
        sv x0 = *reinterpret_cast<const sv*>(src + j * D), x1 = *reinterpret_cast<const sv*>(src + (j + 4) * D);
        sv x2 = *reinterpret_cast<const sv*>(src + (j + 8) * D), x3 = *reinterpret_cast<const sv*>(src + (j + 12) * D);
        const float w0 = weight[j], w1 = weight[j + 4], w2 = weight[j + 8], w3 = weight[j + 12];
        #pragma unroll
        for (int i = 0; i < V; ++i)
            acc[i] += (combine_to_float(x0[i]) * w0 + combine_to_float(x1[i]) * w1) + (combine_to_float(x2[i]) * w2 + combine_to_float(x3[i]) * w3);
    }
    for (; j < np; j += 4) {
        sv x = *reinterpret_cast<const sv*>(src + j * D);
        #pragma unroll
        for (int i = 0; i < V; ++i) acc[i] += combine_to_float(x[i]) * weight[j];
    }
    #pragma unroll
    for (int i = 0; i < V; ++i) partial[warp * D + column + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        sv o;
        #pragma unroll
        for (int i = 0; i < V; ++i) {
            const int col = column + i;
            scalar_t t;
            from_float(t, ((partial[col] + partial[D + col]) + (partial[2 * D + col] + partial[3 * D + col])) * inv);
            o[i] = t;
        }
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + column) = o;
    }
    if constexpr (D / (64 * V) > 1) __syncthreads();
    }
}

// Four waves own disjoint output columns; each wave divides its partitions
// across four 16-lane subgroups. Each wave independently computes the weights.
// No inter-wave communication or LDS allocation is needed.
// The launcher uses this path only when grid_parts <= 64 and D is 128 or 256.
template <typename scalar_t, int D, bool DYNAMIC = false>
__global__ __launch_bounds__(256) void PA_B16_COMBINE_CW(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int cap = DYNAMIC ? (valid >= 32768 ? grid_parts : one_wave_parts) : grid_parts;
    const int one = DYNAMIC ? cap : one_wave_parts;
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), cap, one, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    const float mx = lane < np ? maxs[lane] : -INFINITY;
    const float s = lane < np ? sums[lane] : 0.0f;
    float M = mx;
    for (int mask = 32; mask; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
    const float w = s > 0.f ? s * __builtin_amdgcn_exp2f(mx - M) : 0.f;
    float L = w;
    for (int mask = 32; mask; mask >>= 1) L += __shfl_xor(L, mask);
    const float inv = __fdividef(1.f, L + 1e-6f);
    constexpr int V = D / 64;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    const int col = warp * (D / 4) + (lane & 15) * V, part_lane = lane >> 4;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + col;
    float acc[V] = {};
    // Match the baseline's four independent partial loads per iteration.
    for (int j = 0; j < np; j += 16) {
        sv x[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            x[u] = *reinterpret_cast<const sv*>(src + MIN_T(j + 4*u + part_lane, np - 1) * D);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int p = j + 4*u + part_lane;
            const float raw = __shfl(w, p & 63);
            const float wi = p < np ? raw : 0.f;
            #pragma unroll
            for (int e = 0; e < V; ++e) acc[e] += combine_to_float(x[u][e]) * wi;
        }
    }
    sv o;
    #pragma unroll
    for (int e = 0; e < V; ++e) {
        float sum = acc[e];
        sum += __shfl_xor(sum, 16);
        sum += __shfl_xor(sum, 32);
        scalar_t v; from_float(v, sum * inv); o[e] = v;
    }
    if (part_lane == 0)
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + col) = o;
}

template <typename scalar_t, int D, bool DYNAMIC = false>
__global__ __launch_bounds__(256) void PA_B16_COMBINE_CW128(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ out_tmp, const int* __restrict__ seq_lens,
    int grid_parts, int one_wave_parts, int num_heads, int min_chunks)
{
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int head = blockIdx.x, b = blockIdx.y;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int cap = DYNAMIC ? (valid >= 32768 ? grid_parts : one_wave_parts) : grid_parts;
    const int one = DYNAMIC ? cap : one_wave_parts;
    const int np = valid_partitions(DIVIDE_ROUND_UP(valid, CHUNK), cap, one, min_chunks);
    if (np == 1) return;
    const int64_t off = (static_cast<int64_t>(b) * num_heads + head) * grid_parts;
    const float* sums = reinterpret_cast<const float*>(out_tmp) + off;
    const float* maxs = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + off;
    const float mx = lane < np ? maxs[lane] : -INFINITY;
    const float s = lane < np ? sums[lane] : 0.0f;
    const float mx1 = lane + 64 < np ? maxs[lane + 64] : -INFINITY;
    const float s1 = lane + 64 < np ? sums[lane + 64] : 0.0f;
    float M = MAX_T(mx, mx1);
    for (int mask = 32; mask; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
    const float w = s > 0.f ? s * __builtin_amdgcn_exp2f(mx - M) : 0.f;
    const float w1 = s1 > 0.f ? s1 * __builtin_amdgcn_exp2f(mx1 - M) : 0.f;
    float L = w + w1;
    for (int mask = 32; mask; mask >>= 1) L += __shfl_xor(L, mask);
    const float inv = __fdividef(1.f, L + 1e-6f);
    constexpr int V = D / 64;
    using sv = __attribute__((__vector_size__(V * sizeof(scalar_t)))) scalar_t;
    const int col = warp * (D / 4) + (lane & 15) * V, part_lane = lane >> 4;
    const scalar_t* src = out_tmp + out_tmp_offset + off * D + col;
    float acc[V] = {};
    // Match the baseline's four independent partial loads per iteration.
    for (int j = 0; j < np; j += 16) {
        sv x[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            x[u] = *reinterpret_cast<const sv*>(src + MIN_T(j + 4*u + part_lane, np - 1) * D);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int p = j + 4*u + part_lane;
            const float low = __shfl(w, p & 63);
            const float high = __shfl(w1, p & 63);
            const float raw = p < 64 ? low : high;
            const float wi = p < np ? raw : 0.f;
            #pragma unroll
            for (int e = 0; e < V; ++e) acc[e] += combine_to_float(x[u][e]) * wi;
        }
    }
    sv o;
    #pragma unroll
    for (int e = 0; e < V; ++e) {
        float sum = acc[e];
        sum += __shfl_xor(sum, 16);
        sum += __shfl_xor(sum, 32);
        scalar_t v; from_float(v, sum * inv); o[e] = v;
    }
    if (part_lane == 0)
        *reinterpret_cast<sv*>(out + (static_cast<int64_t>(b) * num_heads + head) * D + col) = o;
}

// ---------------------------------------------------------------------------
// host：按布局区分的启动参数（放在宏调用之外, HEADSIZE_SWITCH / BOOL_SWITCH 的参数中不能有预处理指令）
// ---------------------------------------------------------------------------
// 每个 wave 处理 16 * MW 行 Q, 每步消耗 CH 个 token
template <int HEAD_SIZE, bool WIDE>
struct LaunchShape {
#if PA_B16_LAYOUT == PA_LAYOUT_L0
    // layout=0: D > 128 时入口已保证每组不超过 16 行; 每步 32 个 token
    static constexpr int MW = (WIDE && HEAD_SIZE <= 128) ? 2 : 1;
    static constexpr int CH = 32;
#else
    static constexpr int MW = WIDE ? 2 : 1;
    static constexpr int CH = step_tokens(HEAD_SIZE);
#endif
};

// 主循环的 LDS 用量 (字节, 不含跨 wave 归并区)
constexpr int main_lds_bytes(int d, int mw, int ch, int s)
{
#if PA_B16_LAYOUT == PA_LAYOUT_L0
    (void)ch; (void)s;
    return q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0;     // V^T 直接读入寄存器, LDS 只放 Q
#else
    return s * ch * V_ROW_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0);
#endif
}

// 主 kernel 的模板实参: layout=0 多一个 CH (写成 PA_B16_KERNEL<PA_B16_MAIN_TARGS(...)>, hipify 才能识别 kernel 启动)
#if PA_B16_LAYOUT == PA_LAYOUT_L0
#define PA_B16_MAIN_TARGS(scalar_t, HS, MW, CH, S, RED) scalar_t, HS, MW, CH, S, RED
#else
#define PA_B16_MAIN_TARGS(scalar_t, HS, MW, CH, S, RED) scalar_t, HS, MW, S, RED
#endif

// 调试打印的结尾: BHSD 的入口不传 max_seq_len (多出的实参被 printf 忽略)
#if PA_B16_LAYOUT == PA_LAYOUT_BHSD
#define PA_B16_PRINT_TAIL "\n"
#else
#define PA_B16_PRINT_TAIL ", max_seq_len [%d]\n"
#endif

// ---------------------------------------------------------------------------
// host：适用性检查通过之后的公共流程。按 CU 数定分区数、检查 workspace、按 D 与每组行数选模板参数，
// 启动主 kernel 与合并 kernel。
// query_in：BSHD / layout=0 的入口已复制为紧密布局；BHSD 传入原 Q，通过 workspace 检查后再复制。
// 返回 false 表示 partial 结果放不进 workspace（out_tmp 至少 210 MB），调用方回退原实现。
// ---------------------------------------------------------------------------
static bool launch(torch::Tensor& out, torch::Tensor& query_in, torch::Tensor& key_cache, torch::Tensor& value_cache,
                   torch::Tensor& block_tables, torch::Tensor& seq_lens, float* out_tmp, int max_seq_len, bool print_param)
{
    (void)max_seq_len;
    const auto o_type = out.scalar_type();
    const int D = static_cast<int>(query_in.size(3));
    // size 为 1 的维度下标恒为 0, 其 stride 不参与寻址, 只检查其余维度是否紧密
    auto dense = [](const torch::Tensor& t, int dim, int64_t stride) { return t.size(dim) == 1 || t.stride(dim) == stride; };
    const int batch = query_in.size(0), mtp = query_in.size(1);
#if PA_B16_LAYOUT == PA_LAYOUT_BSHD
    const int num_q_heads = query_in.size(2), num_kv_heads = key_cache.size(2);
#else
    const int num_q_heads = query_in.size(2), num_kv_heads = key_cache.size(1);
#endif
    const int num_heads = num_q_heads * mtp;
    const int max_num_blocks_per_seq = block_tables.size(1);
#if PA_B16_LAYOUT != PA_LAYOUT_BHSD
    const at::cuda::OptionalCUDAGuard device_guard(device_of(query_in));   // BHSD 由调用方设定设备
#endif
    const int device_index = query_in.get_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static thread_local std::unordered_map<int, int> compute_units_by_device;
    auto cu = compute_units_by_device.find(device_index);
    if (cu == compute_units_by_device.end()) {
        hipDeviceProp_t prop;
        TORCH_CHECK(hipGetDeviceProperties(&prop, device_index) == hipSuccess,
                    PA_B16_LAYOUT_DESC " 16-bit device-property query failed");
        cu = compute_units_by_device.emplace(device_index, prop.multiProcessorCount).first;
    }
    // 每个 KV head 的 query 行数 > 16 时按 32 行分组 (MW = 2), 否则 16 行一组
    const int group_rows = num_heads / num_kv_heads;
    const bool wide = group_rows > 16;
    const int row_groups = DIVIDE_ROUND_UP(group_rows, wide ? 32 : 16);
    const int base_ctas = batch * num_kv_heads * row_groups;
    constexpr int S = 4;
    const int one_wave_parts = MAX_T(1, MIN_T(1024, cu->second / base_ctas));
    const int min_chunks = 2;
    // 每个 CU 一个 CTA; 长序列再放宽分区 (设备端按有效长度裁剪, 无需读取 seq_lens)
#if PA_B16_ARCH == 936
    // gfx936: 每个 KV head 一个行组时最多 3 倍, 两个行组时最多 2 倍 (gfx936 实测)
    const int grid_parts = MIN_T(1024, one_wave_parts * (row_groups == 1 ? 3 : 2));
#else
    // gfx938: 宽行组在长序列时再放宽 PA_B16_WIDE_GRID_MULTIPLIER 倍
    const int grid_parts = MIN_T(1024, one_wave_parts * (wide ? PA_B16_WIDE_GRID_MULTIPLIER : 1));
#endif
    const int64_t partial_slots = static_cast<int64_t>(batch) * num_heads * grid_parts;
    if (partial_slots * static_cast<int64_t>(sizeof(float)) > static_cast<int64_t>(max_tmp_offset) * sizeof(uint16_t) ||
        static_cast<int64_t>(out_tmp_offset) * sizeof(uint16_t) + partial_slots * D * static_cast<int64_t>(sizeof(uint16_t)) > 210000000) {
        return false;
    }
    // Q 在 token/head/dim 维不紧密时 (例如 QKV 融合张量的切片视图) 先复制为紧密布局 (batch 维可以有跨度)
    torch::Tensor query = query_in;
    if (!(query.stride(3) == 1 && dense(query, 2, D) && dense(query, 1, query.size(2) * D))) query = query.contiguous();
    const int q_stride = query.stride(0), k_stride = key_cache.stride(0), v_stride = value_cache.stride(0);
#if PA_B16_LAYOUT == PA_LAYOUT_BHSD
    if (print_param) printf("BHSD_IMPLEMENTATION=" PA_B16_ARCH_STR " Q=%s KV=%s\n", o_type == torch::kBFloat16 ? "BF16" : "FP16",
                            o_type == torch::kBFloat16 ? "BF16" : "FP16");
#endif
    auto run = [&](auto out_tag) {
        using scalar_t = decltype(out_tag);
        HEADSIZE_SWITCH(D, [&] {
            BOOL_SWITCH(wide, WIDE, [&] {
                constexpr int MW = LaunchShape<HEAD_SIZE, WIDE>::MW;
                constexpr int CH = LaunchShape<HEAD_SIZE, WIDE>::CH;
                constexpr bool BF16RED = PA_B16_REDUCE_BF16 && std::is_same<scalar_t, uint16_t>::value;
                constexpr int RED_BYTES = BF16RED ? sizeof(scalar_t) : sizeof(float);
                constexpr int main_bytes = main_lds_bytes(HEAD_SIZE, MW, CH, S);
                constexpr int red_bytes = 2 * S * 16 * MW * sizeof(float) + 64 +
                    S * 16 * (red_cols(S, HEAD_SIZE) + 4) * RED_BYTES;
                constexpr int smem = main_bytes > red_bytes ? main_bytes : red_bytes;
                const dim3 grid(num_kv_heads * row_groups, grid_parts, batch);
                if (print_param) printf(PA_B16_LAYOUT_STR "_B16 implementation=" PA_B16_ARCH_STR " D [%d], fp16 [%d], group_rows [%d], MW [%d], CH [%d], row_groups [%d], "
                    "grid [%d] [%d] [%d], one_wave_parts [%d], min_chunks [%d], BF16RED [%d], smem [%d]" PA_B16_PRINT_TAIL,
                    HEAD_SIZE, (int)std::is_same<scalar_t, _Float16>::value, group_rows, MW, CH, row_groups, grid.x, grid.y, grid.z,
                    one_wave_parts, min_chunks, (int)BF16RED, smem, max_seq_len);
                PA_B16_KERNEL<PA_B16_MAIN_TARGS(scalar_t, HEAD_SIZE, MW, CH, S, BF16RED)>
                    <<<grid, dim3(S * WARP_SIZE), smem, stream>>>(
                        reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(out_tmp),
                        reinterpret_cast<const scalar_t*>(query.data_ptr()),
                        reinterpret_cast<const scalar_t*>(key_cache.data_ptr()),
                        reinterpret_cast<const scalar_t*>(value_cache.data_ptr()),
                        block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
                        num_q_heads, num_kv_heads, mtp, row_groups, max_num_blocks_per_seq,
                        q_stride, k_stride, v_stride, grid_parts, one_wave_parts, min_chunks);
                if (grid_parts == 1) return;
                const dim3 combine_grid(num_heads, batch);
                if constexpr (HEAD_SIZE == 128 || HEAD_SIZE == 256) {
                    if (grid_parts <= 64) {
                        PA_B16_COMBINE_CW<scalar_t, HEAD_SIZE>
                            <<<combine_grid, dim3(256), 0, stream>>>(
                                reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                                seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
                        return;
                    }
                    if (grid_parts <= 128) {
                        PA_B16_COMBINE_CW128<scalar_t, HEAD_SIZE>
                            <<<combine_grid, dim3(256), 0, stream>>>(
                                reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                                seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
                        return;
                    }
                }
                PA_B16_COMBINE<scalar_t, HEAD_SIZE>
                    <<<combine_grid, dim3(256), 0, stream>>>(
                        reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                        seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
            });
        });
    };
    if (o_type == torch::kBFloat16) run(uint16_t{}); else run(_Float16{});
    return true;
}

#undef PA_B16_LAYOUT
#undef PA_B16_ARCH
#undef PA_B16_PREFIX
#undef PA_B16_REDUCE_BF16
#undef PA_B16_WIDE_GRID_MULTIPLIER
#undef PA_B16_DEVICE
#undef PA_B16_KERNEL
#undef PA_B16_COMBINE
#undef PA_B16_COMBINE_CW
#undef PA_B16_COMBINE_CW128
#undef PA_B16_CAT
#undef PA_B16_CAT_
#undef PA_B16_LAYOUT_STR
#undef PA_B16_LAYOUT_DESC
#undef PA_B16_ARCH_STR
#undef PA_B16_MAIN_TARGS
#undef PA_B16_PRINT_TAIL
