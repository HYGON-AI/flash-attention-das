// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// ===========================================================================
// gfx938 原生 E4M3 PA 实现（E4M3FN K/V，BF16/FP16 输出，Q 与输出同类型或 E4M3FN），layout=0 与 BSHD 共用。
// token chunk 流水、在线 softmax、按 CU 数分区、跨 wave 归并、column-wave combine，QK 与 PV 用 MMAC。
//
// 本文件不是普通头文件：没有 include guard，由各布局的 .cu 在自己的命名空间里包含，每包含一次生成一套实现：
//   paged_attention_refactoring_fp8.cu       l0_fp8_gfx938     layout=0：K [blocks, heads, 64, D]，V^T [blocks, heads, D, 64]
//   paged_attention_refactoring_bshd_fp8.cu  bshd_fp8_gfx938   BSHD：K/V [blocks, 64, heads, D]
// 每个 kernel 的指令与合并前逐条相同。
//
// 包含前需定义（包含结束时本文件会 #undef）：
//   PA_FP8_938_LAYOUT                PA_LAYOUT_L0 / PA_LAYOUT_BSHD
//   PA_FP8_938_PREFIX                kernel 名前缀，如 pa_bshd_fp8（合并 kernel 为 <前缀>_combine 等）
//   PA_FP8_938_REDUCE_BF16           BF16 输出时跨 wave 归并是否用 BF16（各布局的编译开关，默认 1）
//   PA_FP8_938_WIDE_GRID_MULTIPLIER  每组超过 16 行时长序列分区的放宽倍数（各布局的编译开关，默认 2）
//
// 布局之间的差别（K/V 寻址；layout=0 的 V 已转置，直接从全局读入寄存器）用 #if PA_FP8_938_LAYOUT == ... 区分。
// 各布局的适用性检查与回退在各自 .cu 的入口函数中，本文件的 launch() 只负责检查通过之后的公共流程。
// ===========================================================================

#ifndef PA_LAYOUT_L0
#define PA_LAYOUT_L0 0      // layout=0（custom）：V 转置
#define PA_LAYOUT_BHSD 1    // layout=1
#define PA_LAYOUT_BSHD 2    // layout=2
#endif

#if !defined(PA_FP8_938_LAYOUT) || !defined(PA_FP8_938_PREFIX) || !defined(PA_FP8_938_REDUCE_BF16) || !defined(PA_FP8_938_WIDE_GRID_MULTIPLIER)
#error "paged_attention_refactoring_fp8_gfx938.h: 包含前需定义 PA_FP8_938_LAYOUT / PA_FP8_938_PREFIX / PA_FP8_938_REDUCE_BF16 / PA_FP8_938_WIDE_GRID_MULTIPLIER"
#endif

// 当前是否为 gfx938 设备编译（host 编译与其他架构的设备编译中 kernel 体为空）
#if defined(__gfx938__)
#define PA_FP8_938_DEVICE 1
#else
#define PA_FP8_938_DEVICE 0
#endif

// kernel 名：<前缀>、<前缀>_combine、<前缀>_combine_column_waves、<前缀>_combine_column_waves128
#define PA_FP8_938_CAT_(a, b) a##b
#define PA_FP8_938_CAT(a, b) PA_FP8_938_CAT_(a, b)
#define PA_FP8_938_KERNEL PA_FP8_938_PREFIX
#define PA_FP8_938_COMBINE PA_FP8_938_CAT(PA_FP8_938_PREFIX, _combine)
#define PA_FP8_938_COMBINE_CW PA_FP8_938_CAT(PA_FP8_938_PREFIX, _combine_column_waves)
#define PA_FP8_938_COMBINE_CW128 PA_FP8_938_CAT(PA_FP8_938_PREFIX, _combine_column_waves128)

// ===== 设备端：kernel 与辅助函数 =====
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0

using rsrc_t = int __attribute__((ext_vector_type(4)));
#if PA_FP8_938_DEVICE
using dwords4_t = uint32_t __attribute__((ext_vector_type(4)));
using dwords2_t = uint32_t __attribute__((ext_vector_type(2)));
__device__ dwords4_t buffer_load128(rsrc_t, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ dwords2_t buffer_load64(rsrc_t, int, int, int) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
#endif

constexpr int CHUNK = 32;

#if PA_FP8_938_DEVICE
__device__ __forceinline__ rsrc_t make_rsrc(const uint8_t* base, int bytes)
{
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    return {static_cast<int>(a), static_cast<int>(a >> 32), bytes, 0x00020000};
}

__device__ __forceinline__ uint8x16_t load16(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(uint8x16_t, buffer_load128(r, voff, soff, 0));
}

__device__ __forceinline__ uint64_t load8(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(uint64_t, buffer_load64(r, voff, soff, 0));
}

using u64x2_t = unsigned long long __attribute__((ext_vector_type(2)));
__device__ __forceinline__ u64x2_t load16x2(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(u64x2_t, buffer_load128(r, voff, soff, 0));
}

// Bytes [0, n) of an 8-byte V^T fragment are valid tokens.
__device__ __forceinline__ uint64_t valid_bytes(int n)
{
    return n >= 8 ? ~0ull : (n <= 0 ? 0ull : ((1ull << (8 * n)) - 1));
}
#endif

// Large Q tiles stay in LDS so the accumulator, K and V fragments fit in VGPRs.
#else

constexpr int CHUNK = 32;
constexpr int V_TILE_BYTES = 32 * 64;

__device__ __forceinline__ bshd_fp8_rsrc_t make_rsrc(const uint8_t* base, int bytes)
{
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    return {static_cast<int>(a), static_cast<int>(a >> 32), bytes, 0x00020000};
}

#if PA_FP8_938_DEVICE
__device__ __forceinline__ uint8x16_t load16(bshd_fp8_rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(uint8x16_t, bshd_fp8_buffer_load(r, voff, soff, 0));
}
#endif

// Large Q tiles stay in LDS so the accumulator, K and V buffers fit in VGPRs.
#endif
__host__ __device__ constexpr bool q_in_lds(int d, int mw) { return d * mw > 256; }
__host__ __device__ constexpr int q_lds_stride(int d) { return d + 16; }

// Epilogue merge width: one 16-row x RED_COLS block of every wave per round.
__host__ __device__ constexpr int red_cols(int waves, int d) { return (waves == 4 && d % 128 == 0) ? 128 : 64; }
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0

// Partitions actually used: at most one CTA per CU (one_wave_parts) unless the
// sequence is long enough for the wider grid to pay for its extra partials.
__device__ __forceinline__ int valid_partitions(int chunks, int grid_parts, int one_wave_parts, int min_chunks)
{
    if (grid_parts > one_wave_parts && chunks >= 32 * grid_parts) return grid_parts;
    return MAX_T(1, MIN_T(one_wave_parts, DIVIDE_ROUND_UP(chunks, min_chunks)));
}

// CH: tokens a wave consumes per step, 32 (half page) or 64 (whole page). Lane group g scores TPG = CH / 4
// consecutive tokens, which equal TPG consecutive bytes of every V^T row: 8-byte (CH = 32) or 16-byte (CH = 64)
// V loads. CH = 64 doubles the K/V fragments in flight, so it is used where the accumulators leave room.
template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int CH, int S, bool BF16RED>
__launch_bounds__(S * WARP_SIZE)
__global__ void PA_FP8_938_KERNEL(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks)
#else

__host__ __device__ constexpr int lds_bytes(int waves, int stage, int mw, int d, bool compact = false)
{
    // Main loop: V staging per wave (+ Q tile). Epilogue: max/sum exchange plus
    // the merge block; the fused combine reuses the same space for its weights.
    return (waves * stage * V_TILE_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0)) >
           (compact ? waves * 16 * red_cols(waves, d) * 4 : (2 * waves * 16 * mw * 4 + 64 + waves * 16 * (red_cols(waves, d) + 4) * 4))
        ? (waves * stage * V_TILE_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0))
        : (compact ? waves * 16 * red_cols(waves, d) * 4 : (2 * waves * 16 * mw * 4 + 64 + waves * 16 * (red_cols(waves, d) + 4) * 4));
}

// Partitions actually used: at most one CTA per CU (one_wave_parts) unless the
// sequence is long enough for the wider grid to pay for its extra partials.
__device__ __forceinline__ int valid_partitions(int chunks, int grid_parts, int one_wave_parts, int min_chunks)
{
    if (grid_parts > one_wave_parts && chunks >= 32 * grid_parts) return grid_parts;
    return MAX_T(1, MIN_T(one_wave_parts, DIVIDE_ROUND_UP(chunks, min_chunks)));
}

template <typename scalar_t, typename q_type, bool is_e4m3, int D, int MW, int S, int STAGE, bool FUSE = true, bool BF16RED = false>
__launch_bounds__(S * WARP_SIZE)
__global__ void PA_FP8_938_KERNEL(
    scalar_t* __restrict__ out, scalar_t* __restrict__ out_tmp,
    const q_type* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int num_q_heads, int num_kv_heads, int mtp, int row_groups, int max_num_blocks_per_seq,
    int q_stride, int k_block_stride, int v_block_stride, int grid_parts, int one_wave_parts, int min_chunks,
    int* __restrict__ counters)
#endif
{
#if PA_FP8_938_DEVICE
    extern __shared__ char shared_mem[];
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    static_assert(CH == 32 || CH == 64, "layout=0 FP8: CH must be 32 or 64");
    constexpr int KC = D / 64;
    constexpr int NT = D / 16;
    constexpr int NI = CH / 16;     // 16-token score tiles per step
    constexpr int NH = CH / 32;     // 32-token PV MMAC k-steps per step
    constexpr int TPG = CH / 4;     // consecutive tokens per lane group
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
#else
    constexpr int KC = D / 64;
    constexpr int ROWS = 16 * MW;
    constexpr float scale = (D == 64 ? 0.125f : (D == 128 ? 0.0883883476f : (D == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_log2 = scale * M_LOG2E;

    const int kv_head = blockIdx.x / row_groups;
    const int rg = blockIdx.x % row_groups;
    const int part = blockIdx.y;
    const int b = blockIdx.z;
    const int valid = __builtin_amdgcn_readfirstlane(seq_lens[b]);
    const int chunks = DIVIDE_ROUND_UP(valid, CHUNK);
    const int nparts = valid_partitions(chunks, grid_parts, one_wave_parts, min_chunks);
    if (part >= nparts) return;
#endif

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
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    const uint8_t* k_head = k_cache + static_cast<int64_t>(kv_head) * (64 * D);
    const uint8_t* v_head = v_cache + static_cast<int64_t>(kv_head) * (64 * D);

    // K fragment rows: tile i column n holds step token TPG*(n&3) + 4*i + (n>>2). With the MMAC result
    // layout (lane (r, g) value j = column g + 4j) lane group g scores tokens TPG*g + 4i + j, i.e. the TPG
    // consecutive tokens TPG*g..TPG*g+TPG-1, so P packs into the same byte order as a V^T row.
    const int k_voff = (TPG * (r & 3) + (r >> 2)) * D + 16 * g;
    // V^T fragment of column tile t: dim 16t + r, step tokens TPG*g..TPG*g+TPG-1.
    const int v_voff = r * 64 + TPG * g;
    // Byte offset of the step inside its 64-token page (CH = 32 uses half pages).
    auto page_of = [&](int c) { return CH == 64 ? c : (c >> 1); };
    auto tok0_of = [&](int c) { return CH == 64 ? 0 : (c & 1) * 32; };

    uint8x16_t kf[NI][KC];
    uint64_t vf[NT][NH];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(k_head + static_cast<int64_t>(phys) * k_block_stride + tok0_of(c) * D, CH * D);
        #pragma unroll
        for (int i = 0; i < NI; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 4 * i * D + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(v_head + static_cast<int64_t>(phys) * v_block_stride, 64 * D);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            if constexpr (CH == 64) {
                const u64x2_t x = load16x2(rs, v_voff, t * 16 * 64);
                vf[t][0] = x[0];
                vf[t][1] = x[1];
            } else {
                vf[t][0] = load8(rs, v_voff, t * 16 * 64 + tok0_of(c));
            }
        }
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[page_of(c)]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem);
#else
    const int token_stride = num_kv_heads * D;
    const uint8_t* k_head = k_cache + kv_head * D;
    const uint8_t* v_head = v_cache + kv_head * D;
    auto chunk_k = [&](int c, int phys) {
        return k_head + static_cast<int64_t>(phys) * k_block_stride + (c & 1) * (CHUNK * token_stride);
    };
    auto chunk_v = [&](int c, int phys) {
        return v_head + static_cast<int64_t>(phys) * v_block_stride + (c & 1) * (CHUNK * token_stride);
    };

    // K fragment rows: tile i column n holds token 16*i + 4*(n&3) + (n>>2), so the
    // P fragment of a 32-token chunk packs straight from the two score tiles.
    const int k_voff = (4 * (r & 3) + (r >> 2)) * token_stride + 16 * g;
    // V rows: load (T, h) covers rows 16h..16h+15 of 64-dim column tile T.
    const int v_row = lane >> 2;
    const int v_voff = v_row * token_stride + 16 * (lane & 3);
    const int v_lds = v_row * 64 + ((16 * (lane & 3)) ^ ((v_row & 2) << 4));

    uint8x16_t kf[2][KC];
    uint8x16_t vf[KC][2];
    auto issue_k = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_k(c, phys), CHUNK * token_stride);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) kf[i][kc] = load16(rs, k_voff, 16 * i * token_stride + 64 * kc);
    };
    auto issue_v = [&](int c, int phys) {
        const auto rs = make_rsrc(chunk_v(c, phys), CHUNK * token_stride);
        #pragma unroll
        for (int t = 0; t < KC; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) vf[t][h] = load16(rs, v_voff, 16 * h * token_stride + 64 * t);
    };

    int c = c_begin + warp;
    if (c < c_end) { const int phys = table[c >> 1]; issue_k(c, phys); issue_v(c, phys); }

    // Q (fp8): register fragments for small tiles, an LDS tile otherwise.
    constexpr bool QL = q_in_lds(D, MW);
    constexpr int QS = q_lds_stride(D);
    uint8_t* q_lds = reinterpret_cast<uint8_t*>(shared_mem) + S * STAGE * V_TILE_BYTES;
#endif
    auto load_q16 = [&](int gr, int col) {
        uint8x16_t v = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (gr < R) {
            const int qrow = (gr / hpk) * num_q_heads + kv_head * hpk + gr % hpk;
            const q_type* qp = q + static_cast<int64_t>(b) * q_stride + static_cast<int64_t>(qrow) * D + col;
            if constexpr (std::is_same<q_type, uint8_t>::value) {
                v = *reinterpret_cast<const uint8x16_t*>(qp);
            } else {
                using qx8 = __attribute__((__vector_size__(8 * sizeof(q_type)))) q_type;
                const qx8 a0 = *reinterpret_cast<const qx8*>(qp);
                const qx8 a1 = *reinterpret_cast<const qx8*>(qp + 8);
                intx4 w;
                w[0] = to_f8_from_f32<is_e4m3>(to_float(a0[0]), to_float(a0[1]), to_float(a0[2]), to_float(a0[3]));
                w[1] = to_f8_from_f32<is_e4m3>(to_float(a0[4]), to_float(a0[5]), to_float(a0[6]), to_float(a0[7]));
                w[2] = to_f8_from_f32<is_e4m3>(to_float(a1[0]), to_float(a1[1]), to_float(a1[2]), to_float(a1[3]));
                w[3] = to_f8_from_f32<is_e4m3>(to_float(a1[4]), to_float(a1[5]), to_float(a1[6]), to_float(a1[7]));
                v = __builtin_bit_cast(uint8x16_t, w);
            }
        }
        return v;
    };
    uint8x16_t qf[QL ? 1 : MW][QL ? 1 : KC];
    if constexpr (QL) {
        for (int i = threadIdx.x; i < ROWS * (D / 16); i += S * WARP_SIZE) {
            const int row = i / (D / 16), col = (i % (D / 16)) * 16;
            *reinterpret_cast<uint8x16_t*>(q_lds + row * QS + col) = load_q16(row0 + row, col);
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) qf[m][kc] = load_q16(row0 + m * 16 + r, 64 * kc + 16 * g);
    }
    auto q_frag = [&](int m, int kc) {
        if constexpr (QL) return *reinterpret_cast<const uint8x16_t*>(q_lds + (m * 16 + r) * QS + 64 * kc + 16 * g);
        else return qf[m][kc];
    };
    // Causal limit per row: token index must be <= limit.
    int limit[MW];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        const int gr = MIN_T(row0 + m * 16 + r, R - 1);
        limit[m] = valid - mtp + gr / hpk;
    }

    float row_max[MW], row_sum[MW];
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    float4_t acc[MW][NT];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < NT; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
#else
    float4_t acc[MW][D / 16];
    #pragma unroll
    for (int m = 0; m < MW; ++m) {
        row_max[m] = -FLT_MAX;
        row_sum[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / 16; ++i) acc[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    uint8_t* stage = reinterpret_cast<uint8_t*>(shared_mem) + warp * STAGE * V_TILE_BYTES;
#endif

    // Peel the known-zero initial recurrence at compile time. The steady
    // loop retains its original rescaling without a per-chunk first test.
    auto process_chunk = [&]<bool INITIAL>() {
        const int next = c + S;
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
        const int next_phys = next < c_end ? table[page_of(next)] : 0;
        const int base_tok = c * CH;
        // ---- S = Q K^T over the CH-token step.
        float4_t s[MW][NI];
        #pragma unroll
        for (int m = 0; m < MW; ++m)
            #pragma unroll
            for (int i = 0; i < NI; ++i) s[m][i] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < NI; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CH - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < NI; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + TPG * g + 4 * i + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW][NH];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                mx = MAX_T(mx, MAX_T(MAX_T(s[m][i][0], s[m][i][1]), MAX_T(s[m][i][2], s[m][i][3])));
#else
        const int next_phys = next < c_end ? table[next >> 1] : 0;
        const int base_tok = c * CHUNK;
        // ---- S = Q K^T over the 32-token chunk.
        float4_t s[MW][2];
        #pragma unroll
        for (int m = 0; m < MW; ++m) { s[m][0] = {0.f, 0.f, 0.f, 0.f}; s[m][1] = {0.f, 0.f, 0.f, 0.f}; }
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            #pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint8x16_t qcur = q_frag(m, kc);
                const intx2* qv = reinterpret_cast<const intx2*>(&qcur);
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const intx2* kv = reinterpret_cast<const intx2*>(&kf[i][kc]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[0], kv[0], s[m][i]);
                    builtin_amdgcn_mmac_fp8<is_e4m3>(qv[1], kv[1], s[m][i]);
                }
            }
        }
        if (next < c_end) issue_k(next, next_phys);

        // ---- Causal / tail mask: only chunks reaching the last mtp tokens.
        if (base_tok + CHUNK - 1 > valid - mtp) {
            #pragma unroll
            for (int m = 0; m < MW; ++m)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (base_tok + 16 * i + 4 * g + j > limit[m]) s[m][i][j] = -INFINITY;
        }

        // ---- Online softmax.
        intx2 p[MW];
        #pragma unroll
        for (int m = 0; m < MW; ++m) {
            float mx = MAX_T(MAX_T(MAX_T(s[m][0][0], s[m][0][1]), MAX_T(s[m][0][2], s[m][0][3])),
                             MAX_T(MAX_T(s[m][1][0], s[m][1][1]), MAX_T(s[m][1][2], s[m][1][3])));
#endif
            mx = MAX_T(mx, __shfl_xor_tmp(mx, 32));
            mx = MAX_T(mx, __shfl_swap16(mx));
            const float new_max = MAX_T(row_max[m], mx);
            float alpha = 1.0f;
            if constexpr (!INITIAL)
                alpha = __builtin_amdgcn_exp2f((row_max[m] - new_max) * scale_log2);
            const float base = new_max * scale_log2;
            float sum = 0.0f;
            #pragma unroll
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
            for (int i = 0; i < NI; ++i)
#else
            for (int i = 0; i < 2; ++i)
#endif
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
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
                for (int i = 0; i < NT; ++i) acc[m][i] *= alpha;
            }
            row_max[m] = new_max;
            // k-step h covers tokens TPG*g + 8h .. TPG*g + 8h + 7: score tiles 2h and 2h + 1.
            #pragma unroll
            for (int h = 0; h < NH; ++h) {
                p[m][h][0] = to_f8_from_f32<is_e4m3>(s[m][2 * h][0], s[m][2 * h][1], s[m][2 * h][2], s[m][2 * h][3]);
                p[m][h][1] = to_f8_from_f32<is_e4m3>(s[m][2 * h + 1][0], s[m][2 * h + 1][1], s[m][2 * h + 1][2], s[m][2 * h + 1][3]);
            }
        }

        // ---- V: zero tokens past the sequence end (P is already zero there), then MMAC.
        if (base_tok + CH > valid) {
            const int nv = valid - (base_tok + TPG * g);
            #pragma unroll
            for (int h = 0; h < NH; ++h) {
                const uint64_t keep = valid_bytes(nv - 8 * h);
                #pragma unroll
                for (int t = 0; t < NT; ++t) vf[t][h] &= keep;
            }
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            #pragma unroll
            for (int h = 0; h < NH; ++h) {
                const intx2 fv = __builtin_bit_cast(intx2, vf[t][h]);
                #pragma unroll
                for (int m = 0; m < MW; ++m) builtin_amdgcn_mmac_fp8<is_e4m3>(p[m][h], fv, acc[m][t]);
            }
        if (next < c_end) issue_v(next, next_phys);
#else
                for (int i = 0; i < D / 16; ++i) acc[m][i] *= alpha;
            }
            row_max[m] = new_max;
            p[m][0] = to_f8_from_f32<is_e4m3>(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3]);
            p[m][1] = to_f8_from_f32<is_e4m3>(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3]);
        }

        // ---- V: zero tokens past the sequence end, stage, transpose-read, MMAC.
        if (base_tok + CHUNK > valid) {
            #pragma unroll
            for (int t = 0; t < KC; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    if (base_tok + 16 * h + v_row >= valid) vf[t][h] = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        #pragma unroll
        for (int t0 = 0; t0 < KC; t0 += STAGE) {
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                uint8_t* slot = stage + (t - t0) * V_TILE_BYTES;
                *reinterpret_cast<uint8x16_t*>(slot + v_lds) = vf[t][0];
                *reinterpret_cast<uint8x16_t*>(slot + 1024 + v_lds) = vf[t][1];
            }
            asm volatile("" ::: "memory");
            if (t0 + STAGE >= KC && next < c_end) issue_v(next, next_phys);
            intx4 frag[STAGE][2];
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t) {
                int8_t* slot = reinterpret_cast<int8_t*>(stage + (t - t0) * V_TILE_BYTES);
                frag[t - t0][0] = __builtin_hcu_ds_read_matrix_format_u8(slot, 0, 2, 2, 1);
                frag[t - t0][1] = __builtin_hcu_ds_read_matrix_format_u8(slot + 32, 0, 2, 2, 1);
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int t = t0; t < t0 + STAGE && t < KC; ++t)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const intx2* fv = reinterpret_cast<const intx2*>(&frag[t - t0][half]);
                    #pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[0], acc[m][t * 4 + half * 2 + 0]);
                        builtin_amdgcn_mmac_fp8<is_e4m3>(p[m], fv[1], acc[m][t * 4 + half * 2 + 1]);
                    }
                }
        }
#endif
    };
    if (c < c_end) {
        process_chunk.template operator()<true>();
        c += S;
    }
    for (; c < c_end; c += S)
        process_chunk.template operator()<false>();

    // ---- Merge the S waves of this CTA (same rows, disjoint chunks).
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
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
        for (int i = 0; i < NT; ++i) acc[m][i] *= f;
#else
        for (int i = 0; i < D / 16; ++i) acc[m][i] *= f;
#endif
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
            // acc tile (2*b32 + f) holds dims 32*b32 + 8j + 2g + f.
            #pragma unroll
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
            for (int tl = 0; tl < RC / 16; ++tl)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    red[(warp * 16 + r) * RS + 16 * tl + 4 * j + g] =
                        from_float<red_t>(acc[m][blk * (RC / 16) + tl][j]);
#else
            for (int b32 = 0; b32 < RC / 32; ++b32)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        red[(warp * 16 + r) * RS + 32 * b32 + 8 * j + 2 * g + f] =
                            from_float<red_t>(acc[m][(blk * (RC / 32) + b32) * 2 + f][j]);
#endif
            __syncthreads();
            for (int item = tid; item < 16 * RC / 8; item += S * WARP_SIZE) {
                const int rr = item / (RC / 8);
                const int d8 = (item % (RC / 8)) * 8;
                float sum8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int w = 0; w < S; ++w) {
                    // RS is a multiple of four BF16 values, so retain two
                    // aligned 64-bit LDS reads while doubling output width.
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
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
#else

    if constexpr (MW > 1 || !FUSE) return;
    // ---- Fused split-K combine: the last CTA of this (batch, head, row group) merges.
    __threadfence();
    __syncthreads();
    int* flag = reinterpret_cast<int*>(shared_mem);
    const int slot = (b * num_kv_heads + kv_head) * row_groups + rg;
    if (tid == 0) flag[0] = atomicAdd(counters + slot, 1);
    __syncthreads();
    if (flag[0] != nparts - 1) return;
    __threadfence();
    const int rows_here = MIN_T(ROWS, R - row0);
    float* wts = reinterpret_cast<float*>(shared_mem) + 16;     // [rows][nparts]
    // Partials written by other CTAs: read through L2 (glc), never a stale L1 line.
    float* sums = reinterpret_cast<float*>(out_tmp);
    float* maxs = reinterpret_cast<float*>(out_tmp + max_tmp_offset);
    auto ld_l2 = [](float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
    for (int row = warp; row < rows_here; row += S) {
        const int64_t off = (static_cast<int64_t>(b) * num_heads + head_of(row0 + row)) * grid_parts;
        float M = -INFINITY;
        for (int i = lane; i < nparts; i += WARP_SIZE) M = MAX_T(M, ld_l2(maxs + off + i));
        for (int mask = 32; mask > 0; mask >>= 1) M = MAX_T(M, __shfl_xor(M, mask));
        float L = 0.0f;
        for (int i = lane; i < nparts; i += WARP_SIZE) {
            const float sv = ld_l2(sums + off + i);
            const float w = sv > 0.0f ? sv * __builtin_amdgcn_exp2f(ld_l2(maxs + off + i) - M) : 0.0f;
            wts[row * nparts + i] = w;
            L += w;
        }
        for (int mask = 32; mask > 0; mask >>= 1) L += __shfl_xor(L, mask);
        const float inv = __fdividef(1.0f, L + 1e-6f);
        for (int i = lane; i < nparts; i += WARP_SIZE) wts[row * nparts + i] *= inv;
    }
    __syncthreads();
    // Keep the buffer descriptor uniform across the entire CTA; only the byte
    // offset varies by lane. A per-row descriptor serializes vector loads.
    const auto partial_rsrc = make_rsrc(reinterpret_cast<const uint8_t*>(out_tmp + out_tmp_offset),
        (b + 1) * num_heads * grid_parts * D * sizeof(scalar_t));
    // Spread the merge across all waves even for a four-row query group.
    // Four-byte (D=128) or eight-byte (D=256) vectors give one row per wave.
    constexpr int OV = D == 128 ? 2 : 4;
    using sxv = __attribute__((__vector_size__(OV * sizeof(scalar_t)))) scalar_t;
    for (int item = tid; item < rows_here * (D / OV); item += S * WARP_SIZE) {
        const int row = item / (D / OV);
        const int col = (item % (D / OV)) * OV;
        const int head_all = head_of(row0 + row);
        const int partial_base = ((b * num_heads + head_all) * grid_parts * D + col) * sizeof(scalar_t);
        const float* w = wts + row * nparts;
        float accv[OV] = {};
        constexpr int ILP = 8;
        for (int i0 = 0; i0 < nparts; i0 += ILP) {
            sxv x[ILP];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int offset = partial_base + MIN_T(i0 + u, nparts - 1) * D * sizeof(scalar_t);
                if constexpr (OV == 2)
                    x[u] = __builtin_bit_cast(sxv, bshd_fp8_buffer_load32(partial_rsrc, offset, 0, 1));
                else
                    x[u] = __builtin_bit_cast(sxv, bshd_fp8_buffer_load64(partial_rsrc, offset, 0, 1));
            }
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const float wi = i0 + u < nparts ? w[i0 + u] : 0.0f;
                #pragma unroll
                for (int e = 0; e < OV; ++e) accv[e] += to_float(x[u][e]) * wi;
            }
        }
        sxv o;
        #pragma unroll
        for (int e = 0; e < OV; ++e) o[e] = from_float<scalar_t>(accv[e]);
        *reinterpret_cast<sxv*>(out + (static_cast<int64_t>(b) * num_heads + head_all) * D + col) = o;
    }
    if (tid == 0) atomicExch(counters + slot, 0);
#endif
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
__global__ __launch_bounds__(256) void PA_FP8_938_COMBINE(
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
__global__ __launch_bounds__(256) void PA_FP8_938_COMBINE_CW(
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
__global__ __launch_bounds__(256) void PA_FP8_938_COMBINE_CW128(
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
// 每个 KV head 的 query 行数 > 16 时按 32 行分组 (MW = 2), 否则 16 行一组
inline int launch_mw(bool wide, int d)
{
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    return (wide && d != 256) ? 2 : 1;     // layout=0: D = 256 固定 16 行一组 (MW = 2 的累加器与 K/V 片段超出 256 VGPR)
#else
    (void)d;
    return wide ? 2 : 1;
#endif
}

template <int HEAD_SIZE, bool MW2>
struct LaunchShape {
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    static constexpr int MW = (MW2 && HEAD_SIZE != 256) ? 2 : 1;
    // 每个 wave 每步处理整页 64 token (V^T 16 字节读取); D = 192 且 MW = 2 时用半页 (实测选择, 均不 spill)
    static constexpr int CH = (HEAD_SIZE == 192 && MW == 2) ? 32 : 64;
#else
    static constexpr int MW = MW2 ? 2 : 1;
    static constexpr int CH = 0;            // BSHD: 每步 token 数由 kernel 内部决定
#endif
};

#if PA_FP8_938_LAYOUT != PA_LAYOUT_L0
constexpr int LAUNCH_STAGES = 2;            // BSHD: V tile 的 LDS 级数
#endif

// 主循环的 LDS 用量 (字节, 不含跨 wave 归并区)
constexpr int main_lds_bytes(int d, int mw, int s)
{
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    (void)s;
    return q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0;     // V^T 直接读入寄存器, LDS 只放 Q
#else
    return s * LAUNCH_STAGES * V_TILE_BYTES + (q_in_lds(d, mw) ? 16 * mw * q_lds_stride(d) : 0);
#endif
}

// 主 kernel 的模板实参与末尾实参 (写成 PA_FP8_938_KERNEL<PA_FP8_938_MAIN_TARGS(...)>, hipify 才能识别 kernel 启动)
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
#define PA_FP8_938_MAIN_TARGS(scalar_t, q_type, e4m3, HS, MW, CH, S, RED) scalar_t, q_type, e4m3, HS, MW, CH, S, RED
#define PA_FP8_938_MAIN_TAIL_ARGS
#else
#define PA_FP8_938_MAIN_TARGS(scalar_t, q_type, e4m3, HS, MW, CH, S, RED) scalar_t, q_type, e4m3, HS, MW, S, LAUNCH_STAGES, false, RED
#define PA_FP8_938_MAIN_TAIL_ARGS , nullptr
#endif

inline void print_launch(int d, int q_fp8, int group_rows, int mw, int ch, int row_groups, const dim3& grid,
                         int one_wave_parts, int min_chunks, int bf16red, int smem, int max_seq_len)
{
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    printf("L0_FP8 implementation=gfx938 D [%d], q_fp8 [%d], group_rows [%d], MW [%d], CH [%d], row_groups [%d], "
        "grid [%d] [%d] [%d], one_wave_parts [%d], min_chunks [%d], BF16RED [%d], smem [%d], max_seq_len [%d]\n",
        d, q_fp8, group_rows, mw, ch, row_groups, grid.x, grid.y, grid.z, one_wave_parts, min_chunks, bf16red, smem, max_seq_len);
#else
    (void)ch;
    printf("BSHD_FP8 implementation=gfx938 D [%d], q_fp8 [%d], group_rows [%d], MW [%d], row_groups [%d], "
        "grid [%d] [%d] [%d], one_wave_parts [%d], min_chunks [%d], BF16RED [%d], smem [%d], max_seq_len [%d]\n",
        d, q_fp8, group_rows, mw, row_groups, grid.x, grid.y, grid.z, one_wave_parts, min_chunks, bf16red, smem, max_seq_len);
#endif
}

// ---------------------------------------------------------------------------
// host：适用性检查通过之后的公共流程。按 CU 数定分区数、检查 workspace、按 D 与每组行数选模板参数，
// 启动主 kernel 与合并 kernel。query 已由入口复制为紧密布局。
// 返回 false 表示 partial 结果放不进 workspace（out_tmp 至少 210 MB），调用方回退原实现。
// ---------------------------------------------------------------------------
static bool launch(torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache,
                   torch::Tensor& block_tables, torch::Tensor& seq_lens, float* out_tmp, int max_seq_len, bool print_param)
{
    const auto o_type = out.scalar_type();
    const auto q_dtype = query.scalar_type();
    const int D = static_cast<int>(query.size(3));
    const int batch = query.size(0), mtp = query.size(1);
#if PA_FP8_938_LAYOUT == PA_LAYOUT_L0
    const int num_q_heads = query.size(2), num_kv_heads = key_cache.size(1);
    const char* const device_error = "layout=0 FP8 device-property query failed";
#else
    const int num_q_heads = query.size(2), num_kv_heads = key_cache.size(2);
    const char* const device_error = "BSHD FP8 device-property query failed";
#endif
    const int num_heads = num_q_heads * mtp;
    const int max_num_blocks_per_seq = block_tables.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const int device_index = query.get_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static thread_local std::unordered_map<int, int> compute_units_by_device;
    auto cu = compute_units_by_device.find(device_index);
    if (cu == compute_units_by_device.end()) {
        hipDeviceProp_t prop;
        TORCH_CHECK(hipGetDeviceProperties(&prop, device_index) == hipSuccess, device_error);
        cu = compute_units_by_device.emplace(device_index, prop.multiProcessorCount).first;
    }
    const int group_rows = num_heads / num_kv_heads;
    const bool wide = group_rows > 16;
    const int mw = launch_mw(wide, D);
    const int row_groups = DIVIDE_ROUND_UP(group_rows, 16 * mw);
    const int base_ctas = batch * num_kv_heads * row_groups;
    constexpr int S = 4;
    // 每个 CU 一个 CTA; 宽行组在长序列时再放宽分区 (设备端按有效长度裁剪, 无需读取 seq_lens)
    const int one_wave_parts = MAX_T(1, MIN_T(1024, cu->second / base_ctas));
    const int min_chunks = 2;
    const int grid_parts = MIN_T(1024, one_wave_parts * (wide ? PA_FP8_938_WIDE_GRID_MULTIPLIER : 1));
    const int64_t partial_slots = static_cast<int64_t>(batch) * num_heads * grid_parts;
    if (partial_slots * static_cast<int64_t>(sizeof(float)) > static_cast<int64_t>(max_tmp_offset) * sizeof(uint16_t) ||
        static_cast<int64_t>(out_tmp_offset) * sizeof(uint16_t) + partial_slots * D * static_cast<int64_t>(sizeof(uint16_t)) > 210000000) {
        return false;
    }
    const int q_stride = query.stride(0), k_stride = key_cache.stride(0), v_stride = value_cache.stride(0);
    constexpr bool is_e4m3 = true;
    auto run = [&](auto out_tag, auto q_tag) {
        using scalar_t = decltype(out_tag);
        using q_type = decltype(q_tag);
        HEADSIZE_SWITCH(D, [&] {
            BOOL_SWITCH(mw == 2, MW2, [&] {
                constexpr int MW = LaunchShape<HEAD_SIZE, MW2>::MW;
                constexpr int CH = LaunchShape<HEAD_SIZE, MW2>::CH;
                constexpr bool BF16RED = PA_FP8_938_REDUCE_BF16 && std::is_same<scalar_t, uint16_t>::value;
                constexpr int RED_BYTES = BF16RED ? sizeof(scalar_t) : sizeof(float);
                constexpr int main_bytes = main_lds_bytes(HEAD_SIZE, MW, S);
                constexpr int red_bytes = 2 * S * 16 * MW * sizeof(float) + 64 +
                    S * 16 * (red_cols(S, HEAD_SIZE) + 4) * RED_BYTES;
                constexpr int smem = main_bytes > red_bytes ? main_bytes : red_bytes;
                const dim3 grid(num_kv_heads * row_groups, grid_parts, batch);
                if (print_param) print_launch(HEAD_SIZE, (int)std::is_same<q_type, uint8_t>::value, group_rows, MW, CH, row_groups,
                                              grid, one_wave_parts, min_chunks, (int)BF16RED, smem, max_seq_len);
                PA_FP8_938_KERNEL<PA_FP8_938_MAIN_TARGS(scalar_t, q_type, is_e4m3, HEAD_SIZE, MW, CH, S, BF16RED)>
                    <<<grid, dim3(S * WARP_SIZE), smem, stream>>>(
                        reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(out_tmp),
                        reinterpret_cast<const q_type*>(query.data_ptr()),
                        reinterpret_cast<const uint8_t*>(key_cache.data_ptr()),
                        reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
                        block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
                        num_q_heads, num_kv_heads, mtp, row_groups, max_num_blocks_per_seq,
                        q_stride, k_stride, v_stride, grid_parts, one_wave_parts, min_chunks PA_FP8_938_MAIN_TAIL_ARGS);
                if (grid_parts == 1) return;
                const dim3 combine_grid(num_heads, batch);
                if constexpr (HEAD_SIZE == 128 || HEAD_SIZE == 256) {
                    if (grid_parts <= 64) {
                        PA_FP8_938_COMBINE_CW<scalar_t, HEAD_SIZE>
                            <<<combine_grid, dim3(256), 0, stream>>>(
                                reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                                seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
                        return;
                    }
                    if (grid_parts <= 128) {
                        PA_FP8_938_COMBINE_CW128<scalar_t, HEAD_SIZE>
                            <<<combine_grid, dim3(256), 0, stream>>>(
                                reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                                seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
                        return;
                    }
                }
                PA_FP8_938_COMBINE<scalar_t, HEAD_SIZE>
                    <<<combine_grid, dim3(256), 0, stream>>>(
                        reinterpret_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<const scalar_t*>(out_tmp),
                        seq_lens.data_ptr<int>(), grid_parts, one_wave_parts, num_heads, min_chunks);
            });
        });
    };
    if (o_type == torch::kBFloat16) {
        if (q_dtype == torch::kFloat8_e4m3fn) run(uint16_t{}, uint8_t{}); else run(uint16_t{}, uint16_t{});
    } else {
        if (q_dtype == torch::kFloat8_e4m3fn) run(_Float16{}, uint8_t{}); else run(_Float16{}, _Float16{});
    }
    return true;
}

#undef PA_FP8_938_LAYOUT
#undef PA_FP8_938_PREFIX
#undef PA_FP8_938_REDUCE_BF16
#undef PA_FP8_938_WIDE_GRID_MULTIPLIER
#undef PA_FP8_938_DEVICE
#undef PA_FP8_938_CAT_
#undef PA_FP8_938_CAT
#undef PA_FP8_938_KERNEL
#undef PA_FP8_938_COMBINE
#undef PA_FP8_938_COMBINE_CW
#undef PA_FP8_938_COMBINE_CW128
#undef PA_FP8_938_MAIN_TARGS
#undef PA_FP8_938_MAIN_TAIL_ARGS
