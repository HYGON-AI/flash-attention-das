// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// ===========================================================================
// gfx936 E5M2 KV 直接加载实现（BF16/FP16 Q + E5M2 KV，D = 64/128/192/256），layout=0 与 BSHD 共用。
// 由 gfx936 BHSD 的 direct E5M2 实现改写而来：加载、解码、online softmax、页内切分、epilogue、
// partition 预算与同一次启动内合并两种布局相同；差别只在 K/V 寻址与 V 的读取方式（layout=0 的 V 已转置，
// 每个通道的 token 连续）。16 位 KV 不走这里（见 paged_attention_refactoring_b16_gfx938_gfx936.h）。
//
// 本文件不是普通头文件：没有 include guard，由各布局的 .cu 在自己的命名空间里包含，每包含一次生成一套实现：
//   paged_attention_refactoring_bshd.cu      namespace bshd_gfx936
//   paged_attention_refactoring.cu           namespace l0_gfx936
// kernel 名与原来相同，每个 kernel 的指令与合并前逐条相同。
//
// 包含前需定义（包含结束时本文件会 #undef）：
//   PA_E5M2_936_LAYOUT      PA_LAYOUT_L0 / PA_LAYOUT_BSHD
//   PA_E5M2_936_KERNEL      本布局的 kernel 名
// 布局之间的差别用 #if PA_E5M2_936_LAYOUT == ... 区分。
// ===========================================================================

#ifndef PA_LAYOUT_L0
#define PA_LAYOUT_L0 0      // layout=0（custom）：V 转置
#define PA_LAYOUT_BHSD 1    // layout=1
#define PA_LAYOUT_BSHD 2    // layout=2
#endif

#if !defined(PA_E5M2_936_LAYOUT) || !defined(PA_E5M2_936_KERNEL)
#error "paged_attention_refactoring_e5m2_gfx936.h: 包含前需定义 PA_E5M2_936_LAYOUT / PA_E5M2_936_KERNEL"
#endif


// FP8 KV 直接加载路径 (HEAD_SIZE/HEAD_SIZE_V 为 64/128/192/256)
//  K: QK 的 B 操作数中 lane (n = lane & 15, g = lane >> 4) 需要 token 4n+i 的连续通道,
//     每个 lane 直接加载该 token 在每个 64 通道分组中的第 g 个 16 字节, 不需要 ds_permute。
//     一个 64 通道分组 (4 token x 16 字节) 称为一个 K chunk。
//  V: lane n 持有 CH = HEAD_SIZE_V / 16 个连续通道, 每 16 个 token 的 tile 中持有 4 个连续 token,
//     正好是 PV B 操作数的 token 顺序, 不需要经过 LDS 转置。acc_o 第 q 个 tile 的第 n 列对应
//     通道 CH * n + q, epilogue 写 LDS 时还原通道顺序。
//  E5M2 是截断的 FP16, 字节放到 FP16 高 8 位即为精确值, 因此 PV 使用 FP16 mmac (延迟重缩放时 P 不超过 256)。
//  K chunk / V tile 分别用 K_AHEAD / V_AHEAD 个寄存器槽位做环形预取: 一个槽位消费后立即发射
//  后续数据, 超出当前 page 时预取同一 warp 的下一个 page, 隐藏 page 之间的访存延迟。
// ---------------------------------------------------------------------------
template <int HEAD_SIZE, int HEAD_SIZE_V, int Mloop>
struct DirectFp8Config {
    static constexpr bool enabled = (HEAD_SIZE == 64 || HEAD_SIZE == 128 || HEAD_SIZE == 192 || HEAD_SIZE == 256) &&
                                    (HEAD_SIZE_V == 64 || HEAD_SIZE_V == 128 || HEAD_SIZE_V == 192 || HEAD_SIZE_V == 256);
    static constexpr int K_CHUNKS = HEAD_SIZE / 64;
    static constexpr int V_CH = HEAD_SIZE_V / 16;
    static constexpr int V_NW = V_CH / 4;
    // 寄存器预算: acc_o 占 Mloop * HEAD_SIZE_V / 4 个 VGPR, 规模越大预取槽位越少
    static constexpr int ACC_VGPRS = Mloop * HEAD_SIZE_V / 4;
    // HEAD_SIZE == 192 (3 个 chunk) 时槽位数须整除 3: 中等规模取 3
    static constexpr int K_AHEAD = ACC_VGPRS >= 128 ? 1 : (ACC_VGPRS >= 64 ? (K_CHUNKS < 2 ? K_CHUNKS : (K_CHUNKS % 2 == 0 ? 2 : K_CHUNKS)) : K_CHUNKS);
    static constexpr int V_AHEAD = ACC_VGPRS >= 128 ? 1 : (ACC_VGPRS >= 64 ? 2 : 4);
    static_assert(!enabled || (K_CHUNKS % K_AHEAD == 0 && 4 % V_AHEAD == 0), "ring slots must repeat per page");
    // 每个 CU 的线程块数: 每页 MMA 量 (Mloop * HEAD_SIZE) 较大时两个线程块共驻, 使 MMA 与另一线程块的访存/解码重叠
    static constexpr int CTAS_PER_CU = (Mloop * HEAD_SIZE / 128) < 1 ? 1 : ((Mloop * HEAD_SIZE / 128) > 2 ? 2 : (Mloop * HEAD_SIZE / 128));
};

// 直接加载路径的工作划分 (主 kernel 与合并 kernel 共用, 只依赖 page 数与 partition 上限):
//  page 数不足以让每个 warp 分到独立 page 时 (pages < 4 * P) 使用页内切分模式: 线程块的 4 个 warp
//  共同处理同一 page, 每个线程块至少 1 个 page, 并行线程块数最多为常规模式的 4 倍;
//  否则每个 warp 处理独立 page, 先确定每个 warp 的 page 数, 再只保留达到该 page 数所需的 partition,
//  关键路径不变时减少线程块、partial 输出与合并开销。
//  balanced_seq_len_tile = CTAS_PER_CU * CU 数 / (batch * kv_heads * q_tiles), 因此
//  slice_cap = balanced_seq_len_tile / CTAS_PER_CU 个 partition 时线程块数不超过 CU 数。
//  页内切分只在线程块数不超过 CU 数且每个线程块不超过 2 个 page 时使用: 更多 page 时
//  常规模式的每 warp 独立 page 流水更高效。
template <int CTAS_PER_CU, bool ONE_PAGE_SLICE = false>
__device__ __forceinline__ bool use_direct_slice_mode(int valid_num_block_pages, int balanced_seq_len_tile)
{
    const int slice_cap = MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU);
    return valid_num_block_pages <= 2 * slice_cap;
}

//  LONG_SINGLE_STREAM: 每个 CU 两路 KV 读取流且每页计算量不大时, 若 partition 数减半 (每个 CU 一路流) 后每个 warp
//  仍有 4 个以上 page, 只使用一半 partition: page 间预取已能隐藏访存延迟, 第二路流只增加 DRAM 上的并发流数。
//  主 kernel 与合并 kernel 使用相同的模板参数, 得到相同的 partition 数。
template <int CTAS_PER_CU, bool LONG_SINGLE_STREAM = false, bool RESIDENT_CAP = false, bool ONE_PAGE_SLICE = false>
__device__ __forceinline__ int get_direct_num_partitions(int valid_num_block_pages, int balanced_seq_len_tile)
{
    constexpr int MAIN_KERNEL_NUM_WARPS = 4;
    if (use_direct_slice_mode<CTAS_PER_CU, ONE_PAGE_SLICE>(valid_num_block_pages, balanced_seq_len_tile)) {
        return MAX_T(1, MIN_T(MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU), valid_num_block_pages));
    }
    // Derive the resident limit from the ORIGINAL budget, before LONG_SINGLE_STREAM.
    const int resident_seq_len_tile = MAX_T(1, balanced_seq_len_tile / CTAS_PER_CU);
    if constexpr (LONG_SINGLE_STREAM) {
        const int single_stream_tile = MAX_T(1, balanced_seq_len_tile / 2);
        if (DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * single_stream_tile) > 4) {
            balanced_seq_len_tile = single_stream_tile;
        }
    }
    const int max_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS));
    if (max_seq_len_tile <= balanced_seq_len_tile) {
        return max_seq_len_tile;
    }
    const int pages_per_warp = DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * balanced_seq_len_tile);
    return DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * pages_per_warp);
}

// ---- 变长 batch 的 partition 预算 (gfx936) ----
// total_tile 为整个 batch 共享的每组 partition 预算 (CTAS_PER_CU * CU 数 / (kv_heads * q_tiles)), 每个序列按 page 数占比
// 分得 floor(pages * total_tile / total_pages) 个 (至少 1, 至多 alloc_tile, 即 partial 输出的步长)。batch 为 1 或各序列等长时
// 等于按 batch 平均分配的 own_tile = CTAS_PER_CU * CU 数 / (batch * kv_heads * q_tiles); 长短不一时长序列分得更多 partition,
// 不再只占 1/batch 的 CU。LONG_SINGLE_STREAM 的减半由 batch 的平均序列统一决定 (等长时与逐序列判断相同), 避免只有部分
// 序列减半造成负载不均。主 kernel 与合并 kernel 用同一组函数, 得到相同的 partition 数。
struct DirectBatchBudget {
    long long total_pages;
    bool halve;
};

template <bool LONG_SINGLE_STREAM>
__device__ __forceinline__ DirectBatchBudget direct_batch_budget(const int *__restrict__ seq_lens, int batch, int total_tile,
                                                                 int seqlen_self)
{
    constexpr int MAIN_KERNEL_NUM_WARPS = 4;
    long long total;
    if (batch > 1) {
        // 每个 wave 独立求整个 batch 的 page 总数 (不需要 LDS 与线程块同步)
        long long pages = 0;
        for (int b = threadIdx.x % WARP_SIZE; b < batch; b += WARP_SIZE) pages += DIVIDE_ROUND_UP(seq_lens[b], 64);
        #pragma unroll
        for (int m = WARP_SIZE / 2; m >= 1; m /= 2) pages += __shfl_xor(pages, m);
        total = (static_cast<long long>(__builtin_amdgcn_readfirstlane(static_cast<int>(pages >> 32))) << 32) |
                static_cast<unsigned int>(__builtin_amdgcn_readfirstlane(static_cast<int>(pages)));
    } else {
        total = DIVIDE_ROUND_UP(seqlen_self, 64);   // batch 为 1 时即本序列
    }
    const int avg_pages = static_cast<int>(MIN_T(static_cast<long long>(0x7fffffff), (total + batch - 1) / batch));
    const int avg_tile = MAX_T(1, total_tile / batch);
    DirectBatchBudget budget;
    budget.total_pages = total;
    budget.halve = LONG_SINGLE_STREAM && DIVIDE_ROUND_UP(avg_pages, MAIN_KERNEL_NUM_WARPS * MAX_T(1, avg_tile / 2)) > 4;
    return budget;
}

// pages 页的序列分得的 partition 预算 (各 lane 可分别计算不同序列)
__device__ __forceinline__ int direct_tile_of(int pages, long long total_pages, int batch, int total_tile, int alloc_tile)
{
    if (batch <= 1 || total_pages <= 0) return MIN_T(total_tile, alloc_tile);
    const long long share = static_cast<long long>(pages) * total_tile / total_pages;
    return static_cast<int>(MAX_T(1ll, MIN_T(static_cast<long long>(alloc_tile), share)));
}

// 与 get_direct_num_partitions 相同的划分规则, LONG_SINGLE_STREAM 是否减半由 direct_batch_budget 给出;
// 减半时向上取整: 预算为奇数 (例如 batch * kv_heads 较大时每组 5 个) 时向下取整会让 1/5 以上的 CU 空闲
template <int CTAS_PER_CU>
__device__ __forceinline__ int direct_partitions_for_tile(int valid_num_block_pages, int tile, bool halve)
{
    constexpr int MAIN_KERNEL_NUM_WARPS = 4;
    if (use_direct_slice_mode<CTAS_PER_CU>(valid_num_block_pages, tile)) {
        return MAX_T(1, MIN_T(MAX_T(1, tile / CTAS_PER_CU), valid_num_block_pages));
    }
    if (halve) tile = MAX_T(1, (tile + 1) / 2);
    const int max_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS));
    if (max_seq_len_tile <= tile) {
        return max_seq_len_tile;
    }
    const int pages_per_warp = DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * tile);
    return DIVIDE_ROUND_UP(valid_num_block_pages, MAIN_KERNEL_NUM_WARPS * pages_per_warp);
}

// 溢出区第 row 行: 各序列超出 own_tile 的 partition 按序列号依次排列, 求该行对应的序列号与 partition 号;
// 超出所有序列的溢出 partition 总数时返回 false。每个 wave 独立计算 (wave 内前缀和), 结果在 wave 内一致。
// 溢出 partition 总数的上限 (host 端的溢出区行数): 分得超过 own_tile 的序列, 其预算之和不超过 total_tile, 因此总数
// 不超过 total_tile - own_tile; 每个序列的溢出数又不超过 min(alloc_tile, 每序列最大 page 数) - own_tile。
template <int CTAS_PER_CU>
__device__ __forceinline__ bool direct_overflow_slot(const int *__restrict__ seq_lens, int batch, int row, int own_tile,
                                                     const DirectBatchBudget &budget, int total_tile, int alloc_tile,
                                                     int &batch_idx, int &partition_idx)
{
    const int lane = threadIdx.x % WARP_SIZE;
    int base = 0;
    for (int b0 = 0; b0 < batch; b0 += WARP_SIZE) {
        const int b = b0 + lane;
        int extra = 0;
        if (b < batch) {
            const int pages = DIVIDE_ROUND_UP(seq_lens[b], 64);
            const int tile = direct_tile_of(pages, budget.total_pages, batch, total_tile, alloc_tile);
            extra = MAX_T(0, direct_partitions_for_tile<CTAS_PER_CU>(pages, tile, budget.halve) - own_tile);
        }
        int incl = extra;
        #pragma unroll
        for (int d = 1; d < WARP_SIZE; d *= 2) {
            const int t = __shfl_up(incl, d);
            if (lane >= d) incl += t;
        }
        const int chunk_total = __builtin_amdgcn_readfirstlane(__shfl(incl, WARP_SIZE - 1));
        if (row < base + chunk_total) {
            // 第一个前缀和超过 row 的 lane 即为所在序列
            const unsigned long long hit = __ballot(row < base + incl);
            const int first = __ffsll(static_cast<long long>(hit)) - 1;
            const int excl = __builtin_amdgcn_readfirstlane(__shfl(incl - extra, first));
            batch_idx = b0 + first;
            partition_idx = own_tile + (row - base - excl);
            return true;
        }
        base += chunk_total;
    }
    return false;
}

template <int NW> struct dword_vec;
template <> struct dword_vec<1> { using type = uint32_t; };
template <> struct dword_vec<2> { using type = uint32_t __attribute__((ext_vector_type(2))); };
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
template <> struct dword_vec<4> { using type = uint32_t __attribute__((ext_vector_type(4))); };

using half4_v = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;

// K/V 使用 buffer load: page 基址放在 SGPR 资源描述符中, 每个 lane 只保留一个与 page 无关的 32 位偏移
using buffer_rsrc_t = int __attribute__((ext_vector_type(4)));
__device__ typename dword_vec<4>::type raw_buffer_load_b128(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ typename dword_vec<2>::type raw_buffer_load_b64(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ uint32_t raw_buffer_load_b32(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

template <int NW>
__device__ __forceinline__ typename dword_vec<NW>::type raw_buffer_load_dwords(buffer_rsrc_t rsrc, int voffset, int soffset)
{
    if constexpr (NW == 4) return raw_buffer_load_b128(rsrc, voffset, soffset, 0);
#else
template <> struct dword_vec<3> { using type = uint32_t __attribute__((ext_vector_type(3))); };
template <> struct dword_vec<4> { using type = uint32_t __attribute__((ext_vector_type(4))); };

using half4_v = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;

// K/V 使用 buffer load: page 基址放在 SGPR 资源描述符中, 每个 lane 只保留一个与 page 无关的 32 位偏移
using buffer_rsrc_t = int __attribute__((ext_vector_type(4)));
__device__ typename dword_vec<4>::type raw_buffer_load_b128(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ typename dword_vec<2>::type raw_buffer_load_b64(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ uint32_t raw_buffer_load_b32(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
// HEAD_SIZE_V == 192: 每个 lane 12 个通道
__device__ typename dword_vec<3>::type raw_buffer_load_b96(buffer_rsrc_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v3i32");

template <int NW>
__device__ __forceinline__ typename dword_vec<NW>::type raw_buffer_load_dwords(buffer_rsrc_t rsrc, int voffset, int soffset)
{
    if constexpr (NW == 4) return raw_buffer_load_b128(rsrc, voffset, soffset, 0);
    else if constexpr (NW == 3) return raw_buffer_load_b96(rsrc, voffset, soffset, 0);
#endif
    else if constexpr (NW == 2) return raw_buffer_load_b64(rsrc, voffset, soffset, 0);
    else return raw_buffer_load_b32(rsrc, voffset, soffset, 0);
}

__device__ __forceinline__ buffer_rsrc_t make_page_rsrc(const uint8_t *page)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(page);
    buffer_rsrc_t rsrc;
    rsrc[0] = static_cast<int>(addr & 0xffffffffu);
    rsrc[1] = static_cast<int>(addr >> 32);   // stride = 0
    rsrc[2] = static_cast<int>(0x80000000u);  // num_records
    rsrc[3] = 0x00020000;                      // 32 位数据格式
    return rsrc;
}

// lane 偏移: token 4 * (lane & 15) 的第 lane >> 4 个 16 字节
__device__ __forceinline__ int k_fp8_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    const int n = lane_id & 15;
    return (16 * (n & 3) + 4 * (n >> 2)) * token_stride + (lane_id >> 4) * 16;
#else
    return (lane_id & 15) * 4 * token_stride + (lane_id >> 4) * 16;
#endif
}

// K 槽位 slot 的第 j 次加载对应的 token 组 i (token 4 * (lane & 15) + i) 与 64 字节 chunk:
//  配对槽位覆盖 2 个 token 组 x 相邻 2 个 chunk, 同一条 128 字节 cache line 的两半由相邻指令读取
//  (两半间隔数条指令读取会降低 DRAM 效率)。HEAD_SIZE == 128 总是配对; HEAD_SIZE == 256 在 Mloop == 1 时配对
//  (每个 S 累加器的 mmac 顺序不变), Mloop >= 2 保留未配对的 Q 半片段路径以控制寄存器;
//  其他情况槽位为 1 个 chunk x 4 个 token 组。每页的槽位数均为 K_CHUNKS。
template <int HEAD_SIZE, int Mloop = 1>
struct KSlotMap {
    static constexpr bool PAIRED = HEAD_SIZE == 128
        ;
    static constexpr __device__ __forceinline__ int token(int slot, int j) { return PAIRED ? 2 * (slot & 1) + (j >> 1) : j; }
    static constexpr __device__ __forceinline__ int chunk(int slot, int j) { return PAIRED ? 2 * (slot >> 1) + (j & 1) : slot; }
};

template <int HEAD_SIZE, int Mloop>
__device__ __forceinline__ void load_k_fp8_chunk(buffer_rsrc_t k_page, int lane_offset, int token_stride, int slot,
                                                 uint32_t raw[4][4])
{
    using Map = KSlotMap<HEAD_SIZE, Mloop>;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const typename dword_vec<4>::type value =
            raw_buffer_load_b128(k_page, lane_offset + Map::chunk(slot, j) * 64, Map::token(slot, j) * token_stride, 0);
        #pragma unroll
        for (int w = 0; w < 4; ++w) raw[j][w] = value[w];
    }
}

// lane 偏移: token 4 * (lane >> 4) 的通道 CH * (lane & 15) 起
template <int HEAD_SIZE_V>
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
__device__ __forceinline__ int v_fp8_lane_offset(int row_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * row_stride + (lane_id >> 4) * 16;
#else
__device__ __forceinline__ int v_fp8_lane_offset(int token_stride)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id >> 4) * 4 * token_stride + (lane_id & 15) * (HEAD_SIZE_V / 16);
#endif
}

template <int HEAD_SIZE_V>
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
__device__ __forceinline__ void load_v_fp8_group(buffer_rsrc_t v_page, int lane_offset, int row_stride, int group, uint32_t raw[4])
{
    const typename dword_vec<4>::type value = raw_buffer_load_b128(v_page, lane_offset, group * 16 * row_stride, 0);
    #pragma unroll
    for (int w = 0; w < 4; ++w) raw[w] = value[w];
}

// V^T 一个 dword 中 4 个连续 token 的有效字节掩码 (valid 为有效 token 数, 截断到 [0, 4])
__device__ __forceinline__ uint32_t vt_token_mask(int valid)
{
    return valid >= 4 ? 0xffffffffu : (valid <= 0 ? 0u : ((1u << (8 * valid)) - 1u));
#else
__device__ __forceinline__ void load_v_fp8_tile(buffer_rsrc_t v_page, int lane_offset, int token_stride, int tile,
                                                uint32_t raw[4][HEAD_SIZE_V / 64])
{
    constexpr int NW = HEAD_SIZE_V / 64;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const typename dword_vec<NW>::type value = raw_buffer_load_dwords<NW>(v_page, lane_offset, (tile * 16 + k) * token_stride);
        if constexpr (NW == 1) {
            raw[k][0] = value;
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) raw[k][w] = value[w];
        }
    }
#endif
}

// 4 个 E5M2 解码为 4 个 b16: FP16 只需移字节; BF16 由精确的 FP16 经 FP32 截取高 16 位 (同样精确)
template <typename scalarx4_t, bool is_query_fp16>
__device__ __forceinline__ scalarx4_t decode_e5m2x4_b16(uint32_t raw)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    using half2_v = _Float16 __attribute__((ext_vector_type(2)));
    const uint32_t h01 = __builtin_amdgcn_perm(raw, raw, 0x010C000Cu);
    const uint32_t h23 = __builtin_amdgcn_perm(raw, raw, 0x030C020Cu);
    if constexpr (is_query_fp16) {
        return __builtin_bit_cast(scalarx4_t, (u32x2){h01, h23});
    } else {
        const half2_v a = __builtin_bit_cast(half2_v, h01);
        const half2_v b = __builtin_bit_cast(half2_v, h23);
        const uint32_t p01 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, (float)a[1]),
                                                   __builtin_bit_cast(uint32_t, (float)a[0]), 0x07060302u);
        const uint32_t p23 = __builtin_amdgcn_perm(__builtin_bit_cast(uint32_t, (float)b[1]),
                                                   __builtin_bit_cast(uint32_t, (float)b[0]), 0x07060302u);
        return __builtin_bit_cast(scalarx4_t, (u32x2){p01, p23});
    }
}

// 直接加载路径的 online softmax:
//  每个 lane 只累计自己持有的 S 元素的行和 (行内 4 个 lane 组的部分和在 page 循环结束后再归约);
//  行 max 的增量不超过 2^8 (log2 域) 时沿用旧的参考 max, 不缩放 acc_o 与行和, P <= 2^8 仍在 FP16 范围内,
//  累加为 FP32, 在实数运算上等价于每页缩放, 浮点舍入可能不同。参考 max 只增不减, 全部被掩码的行保持 -FLT_MAX, 其 P 为 0。
constexpr float DIRECT_SOFTMAX_RESCALE_LOG2 = 8.0f;

// Preserve the subtraction before multiplication. Under fast-math the compiler
// may turn (a-b)*scale into a fused multiply-add; equal -FLT_MAX sentinels
// then leave a large rounding residual and exp2 can overflow.
__device__ __forceinline__ float direct_exp_difference(float a, float b, float scale)
{
    float difference;
    asm("v_sub_f32 %0, %1, %2" : "=v"(difference) : "v"(a), "v"(b));
    return __builtin_amdgcn_exp2f(difference * scale);
}

// FAST (行切分主循环): 行最大值用 v_max_f32, exp2 参数用 fma
template <int Mloop, int HEAD_SIZE_V, int NT, bool FAST = false>
__device__ __forceinline__ void direct_softmax(float4_t s_reg[Mloop][NT], float scores_max[Mloop], float scores_sum[Mloop],
                                               float4_t acc_o[Mloop][HEAD_SIZE_V / 16], float scale_log2, bool first_page)
{
    auto vmax = [](float a, float b) {
        if constexpr (FAST) return __builtin_fmaxf(a, b);
        else return MAX_T(a, b);
    };
    float row_max[Mloop];
    bool rescale = false;
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        float4_t partial = s_reg[m][0];
        #pragma unroll
        for (int i = 1; i < NT; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) partial[j] = vmax(partial[j], (s_reg[m][i])[j]);
        }
        float value = vmax(vmax(partial[0], partial[1]), vmax(partial[2], partial[3]));
        value = vmax(value, __shfl_xor_tmp(value, 32));
        value = vmax(value, __shfl_swap16(value));
        row_max[m] = vmax(value, scores_max[m]);
        rescale |= (row_max[m] - scores_max[m]) * scale_log2 > DIRECT_SOFTMAX_RESCALE_LOG2;
    }
    if (__any(rescale)) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const float factor = direct_exp_difference(scores_max[m], row_max[m], scale_log2);
            scores_sum[m] *= factor;
            if (!first_page) {
                #pragma unroll
                for (int i = 0; i < HEAD_SIZE_V / 16; ++i) acc_o[m][i] *= factor;
            }
            scores_max[m] = row_max[m];
        }
    }
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const float max_scaled = scores_max[m] * scale_log2;
        float4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < NT; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (FAST) (s_reg[m][i])[j] = __builtin_amdgcn_exp2f(__builtin_fmaf((s_reg[m][i])[j], scale_log2, -max_scaled));
                else (s_reg[m][i])[j] = __builtin_amdgcn_exp2f((s_reg[m][i])[j] * scale_log2 - max_scaled);
            }
            sum += s_reg[m][i];
        }
        scores_sum[m] += (sum[0] + sum[1]) + (sum[2] + sum[3]);
    }
}

// page 循环结束后归约行和 (同一行由 lane n, n + 16, n + 32, n + 48 持有)
template <int Mloop>
__device__ __forceinline__ void direct_softmax_finish(float scores_sum[Mloop])
{
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        scores_sum[m] += __shfl_xor_tmp(scores_sum[m], 32);
        scores_sum[m] += __shfl_swap16(scores_sum[m]);
    }
}

// 读取一个 64 维 chunk 的 Q 分片
template <typename scalarx8_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD>
__device__ __forceinline__ void load_q_fp8_chunk(const scalar_t *smem_q, int chunk, int seqlen_q_per_kv_head,
                                                 scalarx8_t q_fragment[Mloop][2])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int q_row = m * 16 + q_s2r_row;
        const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + chunk * 64 + q_s2r_col * 16;
        if (q_row < seqlen_q_per_kv_head) {
            q_fragment[m][0] = *reinterpret_cast<const scalarx8_t *>(q_src);
            q_fragment[m][1] = *reinterpret_cast<const scalarx8_t *>(q_src + 8);
        } else {
            q_fragment[m][0] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            q_fragment[m][1] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
}

// 槽位 slot 的 QK: 按 chunk 分组, 组内各 token 组的 mmac 写不同累加器。
// 配对槽位的 Q 分片由调用者按 chunk 对读取 (相邻两个槽位共用), 每页 Q 的 LDS 读取量与未配对时相同
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD, bool is_query_fp16>
__device__ __forceinline__ void qk_fp8_chunk(const uint32_t raw[4][4], const scalar_t *smem_q, int slot,
                                             int seqlen_q_per_kv_head, float4_t s_reg[Mloop][4],
                                             const scalarx8_t (*q_pair)[Mloop][2] = nullptr)
{
    using Map = KSlotMap<HEAD_SIZE, Mloop>;
    constexpr int GROUPS = Map::PAIRED ? 2 : 1;      // 槽位内的 chunk 数
    constexpr int PER_GROUP = 4 / GROUPS;            // 每个 chunk 的 token 组数
    #pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
        scalarx8_t q_fragment[Mloop][2];
        if constexpr (Map::PAIRED) {
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                q_fragment[m][0] = q_pair[g][m][0];
                q_fragment[m][1] = q_pair[g][m][1];
            }
        } else {
            load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, Map::chunk(slot, g), seqlen_q_per_kv_head, q_fragment);
        }
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            #pragma unroll
            for (int t = 0; t < PER_GROUP; ++t) {
                const int j = Map::PAIRED ? 2 * t + g : t;
                const scalarx4_t k_fragment = decode_e5m2x4_b16<scalarx4_t, is_query_fp16>(raw[j][sub]);
                #pragma unroll
                for (int m = 0; m < Mloop; ++m) {
                    const scalarx8_t q8 = q_fragment[m][sub / 2];
                    const scalarx4_t q_v4 = (sub % 2 == 0) ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)
                                                           : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);
                    builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, s_reg[m][Map::token(slot, j)]);
                }
            }
        }
    }
}

#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
// 一个 dword (4 个 token 的 E5M2) 解码为 PV 的 FP16 B 操作数: 字节放到 FP16 高 8 位即为精确值
__device__ __forceinline__ half4_v decode_vt_e5m2x4_f16(uint32_t raw)
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    // 结果字节为 {0, token0, 0, token1} 与 {0, token2, 0, token3}
    const u32x2 packed = {__builtin_amdgcn_perm(raw, raw, 0x010C000Cu), __builtin_amdgcn_perm(raw, raw, 0x030C020Cu)};
    return __builtin_bit_cast(half4_v, packed);
}

// 第 q 个 16 通道组的 PV: 4 个 token tile 依次累加到 acc_o[m][q]
template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void pv_fp8_group_f16(const uint32_t raw[4], const half4_v p[4][Mloop],
                                                 float4_t acc_o[Mloop][HEAD_SIZE_V / 16], int q)
{
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const half4_v v_fragment = decode_vt_e5m2x4_f16(raw[t]);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            builtin_amdgcn_mmac<half4_v, true>(p[t][m], v_fragment, acc_o[m][q]);
#else
template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void pv_fp8_tile_f16(const uint32_t raw[4][HEAD_SIZE_V / 64],
                                                const half4_v p[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16])
{
    using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
    constexpr int CH = HEAD_SIZE_V / 16;
    #pragma unroll
    for (int q = 0; q < CH; ++q) {
        const int w = q / 4;
        constexpr uint32_t zero = 0x0C;
        const uint32_t b = q % 4;
        // 结果字节为 {0, token0, 0, token1}, 即两个 FP16
        const uint32_t sel = zero | (b << 8) | (zero << 16) | ((4 + b) << 24);
        const u32x2 packed = {__builtin_amdgcn_perm(raw[1][w], raw[0][w], sel),
                              __builtin_amdgcn_perm(raw[3][w], raw[2][w], sel)};
        const half4_v v_fragment = __builtin_bit_cast(half4_v, packed);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            builtin_amdgcn_mmac<half4_v, true>(p[m], v_fragment, acc_o[m][q]);
#endif
        }
    }
}

// 每个 lane 写出 16 个 float, 将 acc_o 的交织通道还原到 smem_reduce 的 64 列分组
template <int Mloop, int HEAD_SIZE_V>
__device__ __forceinline__ void store_direct_v_acc_group(float *row_ptr, const float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                         int tile_row, int tile_col, int acc_col)
{
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    #pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) row_ptr[16 * qq + acc_col + 4 * j] = (acc_o[tile_row][tile_col * 4 + qq])[j];
#else
    constexpr int CH = HEAD_SIZE_V / 16;
    constexpr int NJ = 16 / CH;
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        const int j = tile_col * NJ + jj;
        #pragma unroll
        for (int q4 = 0; q4 < CH / 4; ++q4) {
            const float4_t value = {(acc_o[tile_row][q4 * 4 + 0])[j], (acc_o[tile_row][q4 * 4 + 1])[j],
                                    (acc_o[tile_row][q4 * 4 + 2])[j], (acc_o[tile_row][q4 * 4 + 3])[j]};
            *reinterpret_cast<float4_t *>(row_ptr + CH * acc_col + 4 * CH * jj + 4 * q4) = value;
        }
#endif
    }
}

// 线程块内只通过 LDS 交换数据时使用: __syncthreads() 的 workgroup release 会等待所有向量访存
// (包括仍在预取的 K/V 与已发射的 global store), 此处只需等待本 wave 的 LDS 操作完成
__device__ __forceinline__ void lds_barrier()
{
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Q 预取拆分为发射与写 LDS 两步, 两步之间发射首个 page 的 K/V 加载
template <typename scalarx8_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS>
struct QStage {
    static constexpr int ITERS = DIVIDE_ROUND_UP(MAX_ROWS * HEAD_SIZE, NUM_THREADS * 8);
    scalarx8_t data[ITERS];
};

// rows_begin/rows_count 为当前线程块处理的 Q 行 (行号包含 seqlen_q 与 q head 维度)
template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS>
__device__ __forceinline__ void issue_q_stage(QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                              const scalar_t *q, int num_heads, int num_kv_heads, int mtp,
                                              int kv_head_idx, int batch_idx, int q_stride,
                                              int rows_begin, int rows_count)
{
    const int num_q_heads = num_heads / mtp;
    const int num_q_heads_per_kv_head = num_q_heads / num_kv_heads;
    const scalar_t *q_ptr = q + batch_idx * q_stride + num_q_heads_per_kv_head * kv_head_idx * HEAD_SIZE;
    const int q_seqlen_stride = num_q_heads * HEAD_SIZE;
    const int q_seqlen_data_num = num_q_heads_per_kv_head * HEAD_SIZE;
    // 加载不放在分支内, 编译器才能只等待 Q 而不等待随后发射的 K/V; 多余线程重复读取最后 8 个元素
    const int q_elems = rows_count * HEAD_SIZE;
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = MIN_T((it * NUM_THREADS + static_cast<int>(threadIdx.x)) * 8, q_elems - 8);
        const int i = rows_begin * HEAD_SIZE + local_i;
        const int q_offset = (i / q_seqlen_data_num) * q_seqlen_stride + i % q_seqlen_data_num;
        stage.data[it] = *reinterpret_cast<const scalarx8_t *>(q_ptr + q_offset);
    }
}

template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS, int Q_LDS_PAD>
__device__ __forceinline__ void commit_q_stage(const QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                               scalar_t *smem_q, int rows_count)
{
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            *reinterpret_cast<scalarx8_t *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) =
                stage.data[it];
        }
    }
    lds_barrier();
}

// BF16 Q 以 FP16 写入 LDS, QK 使用 FP16 mmac, K 的 E5M2 解码只需移字节 (BF16 需经 FP32 转换, 每元素约多 1.5 条 VALU)。
// 按线程块内 max|Q| 选取 2 的幂缩放 2^-s 使最大值落在 [2^14, 2^15): BF16 的 8 位有效位在 FP16 的 11 位内,
// 转换前检查指数跨度; 过宽的动态范围保留 BF16 QK, 避免小分量下溢。返回 2^s, 调用者据此还原 softmax 缩放。
// reduce 为 Q 区之外的 LDS (每个 warp 一个 int); OVERLAP 时 reduce 可与 Q 区重叠 (读完归约结果后多一次屏障,
// 用于 LDS 紧凑的行切分 kernel)。
template <typename scalarx8_t, typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int MAX_ROWS, int Q_LDS_PAD, bool OVERLAP = false>
__device__ __forceinline__ float commit_q_stage_fp16(const QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, MAX_ROWS> &stage,
                                                     _Float16 *smem_q, int rows_count, uint32_t *reduce)
{
    // BF16 values are stored as uint16_t here. A scalar bit_cast of a vector
    // element is redundant and miscompiled by this DTK Clang frontend.
    static_assert(std::is_same<scalar_t, uint16_t>::value,
                  "BF16 Q range analysis requires uint16_t storage");
    using half8_v = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    uint32_t max_bits = 0, min_bits = 0x7fffu;
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t bits = static_cast<uint32_t>(stage.data[it][e]) & 0x7fffu;
                max_bits = MAX_T(max_bits, bits);
                if (bits != 0) min_bits = MIN_T(min_bits, bits);
            }
        }
    }
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        max_bits = MAX_T(max_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(max_bits), mask)));
        min_bits = MIN_T(min_bits, static_cast<uint32_t>(__shfl_xor(static_cast<int>(min_bits), mask)));
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        reduce[threadIdx.x / WARP_SIZE] = max_bits;
        reduce[NUM_WARPS + threadIdx.x / WARP_SIZE] = min_bits;
    }
    lds_barrier();
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        max_bits = MAX_T(max_bits, reduce[w]);
        min_bits = MIN_T(min_bits, reduce[NUM_WARPS + w]);
    }
    // 归约区与 Q 区重叠时: 所有 warp 读完归约结果后才写 Q
    if constexpr (OVERLAP) lds_barrier();
    if (max_bits != 0 && ((max_bits >> 7) == 255 || (max_bits >> 7) - (min_bits >> 7) > 31)) {
        #pragma unroll
        for (int it = 0; it < stage.ITERS; ++it) {
            const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
            if (local_i < rows_count * HEAD_SIZE) {
                *reinterpret_cast<scalarx8_t *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) = stage.data[it];
            }
        }
        lds_barrier();
        return -1.0f; // BF16 QK inside the same direct optimized kernel.
    }
    // BF16 指数 (含偏置 127); 全零 Q 不缩放
    const int exponent = static_cast<int>(max_bits >> 7);
    const int shift = max_bits == 0 ? 0 : MIN_T(MAX_T(exponent - 127 - 14, -126), 113);
    const float q_mul = __builtin_bit_cast(float, static_cast<uint32_t>(127 - shift) << 23);
    #pragma unroll
    for (int it = 0; it < stage.ITERS; ++it) {
        const int local_i = (it * NUM_THREADS + threadIdx.x) * 8;
        if (local_i < rows_count * HEAD_SIZE) {
            half8_v h;
            #pragma unroll
            for (int e = 0; e < 8; ++e) h[e] = static_cast<_Float16>(to_float(stage.data[it][e]) * q_mul);
            *reinterpret_cast<half8_v *>(smem_q + (local_i / HEAD_SIZE) * (HEAD_SIZE + Q_LDS_PAD) + local_i % HEAD_SIZE) = h;
        }
    }
    lds_barrier();
    return __builtin_bit_cast(float, static_cast<uint32_t>(127 + shift) << 23);
}

// ---- 页内切分模式: warp w 负责 page 内 token [16w, 16w + 16) ----
//  QK 的 B 操作数第 n 列对应 token 16w + n, 因此 S 的第 j 个元素对应 token 16w + g + 4j (g = lane >> 4);
//  PV 中 lane 组 g 的 4 个 K 元素即这 4 个 token, V 按相同顺序加载后仍用同一转置。
__device__ __forceinline__ int k_fp8_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    const int n = lane_id & 15;
    return (16 * warp_id + 4 * (n & 3) + (n >> 2)) * token_stride + (lane_id >> 4) * 16;
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_fp8_slice_lane_offset(int row_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (lane_id & 15) * row_stride + 16 * warp_id + 4 * (lane_id >> 4);
#else
    return (16 * warp_id + (lane_id & 15)) * token_stride + (lane_id >> 4) * 16;
}

template <int HEAD_SIZE_V>
__device__ __forceinline__ int v_fp8_slice_lane_offset(int token_stride, int warp_id)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    return (16 * warp_id + (lane_id >> 4)) * token_stride + (lane_id & 15) * (HEAD_SIZE_V / 16);
#endif
}

template <int HEAD_SIZE>
__device__ __forceinline__ void load_k_fp8_slice(buffer_rsrc_t k_page, int lane_offset, uint32_t raw[HEAD_SIZE / 64][4])
{
    #pragma unroll
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {
        const typename dword_vec<4>::type value = raw_buffer_load_b128(k_page, lane_offset + h * 64, 0, 0);
        #pragma unroll
        for (int w = 0; w < 4; ++w) raw[h][w] = value[w];
    }
}

template <int HEAD_SIZE_V>
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
__device__ __forceinline__ void load_v_fp8_slice(buffer_rsrc_t v_page, int lane_offset, int row_stride,
                                                 uint32_t raw[HEAD_SIZE_V / 16])
{
    #pragma unroll
    for (int q = 0; q < HEAD_SIZE_V / 16; ++q) raw[q] = raw_buffer_load_b32(v_page, lane_offset, q * 16 * row_stride, 0);
#else
__device__ __forceinline__ void load_v_fp8_slice(buffer_rsrc_t v_page, int lane_offset, int token_stride,
                                                 uint32_t raw[4][HEAD_SIZE_V / 64])
{
    constexpr int NW = HEAD_SIZE_V / 64;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const typename dword_vec<NW>::type value = raw_buffer_load_dwords<NW>(v_page, lane_offset, 4 * j * token_stride);
        if constexpr (NW == 1) {
            raw[j][0] = value;
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) raw[j][w] = value[w];
        }
    }
#endif
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int Q_LDS_PAD, bool is_query_fp16>
__device__ __forceinline__ void qk_fp8_slice(const uint32_t raw[HEAD_SIZE / 64][4], const scalar_t *smem_q,
                                             int seqlen_q_per_kv_head, float4_t s_reg[Mloop][1])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_s2r_row = lane_id & 15;
    const int q_s2r_col = lane_id >> 4;
    // Mloop == 1 时同一 tile 的 mmac 交替写两个累加器缩短依赖链; Mloop >= 2 时各 m 已是独立链
    constexpr int NACC = Mloop == 1 ? 2 : 1;
    float4_t partial[Mloop][2];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        partial[m][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        partial[m][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    #pragma unroll
    for (int h = 0; h < HEAD_SIZE / 64; ++h) {
        scalarx8_t q_fragment[Mloop][2];
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int q_row = m * 16 + q_s2r_row;
            const scalar_t *q_src = smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + h * 64 + q_s2r_col * 16;
            if (q_row < seqlen_q_per_kv_head) {
                q_fragment[m][0] = *reinterpret_cast<const scalarx8_t *>(q_src);
                q_fragment[m][1] = *reinterpret_cast<const scalarx8_t *>(q_src + 8);
            } else {
                q_fragment[m][0] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
                q_fragment[m][1] = scalarx8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const scalarx4_t k_fragment = decode_e5m2x4_b16<scalarx4_t, is_query_fp16>(raw[h][sub]);
            #pragma unroll
            for (int m = 0; m < Mloop; ++m) {
                const scalarx8_t q8 = q_fragment[m][sub / 2];
                const scalarx4_t q_v4 = (sub % 2 == 0) ? __builtin_shufflevector(q8, q8, 0, 1, 2, 3)
                                                       : __builtin_shufflevector(q8, q8, 4, 5, 6, 7);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4, k_fragment, partial[m][sub % NACC]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) s_reg[m][0] = NACC == 2 ? partial[m][0] + partial[m][1] : partial[m][0];
}

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_slice(
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
                                                 uint32_t k_raw[HEAD_SIZE / 64][4], uint32_t v_raw[HEAD_SIZE_V / 16],
                                                 int block_idx, bool first_page, int warp_id, int v_token_stride,
                                                 const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                 int valid_seqlen_kv, float scale_softmax_log2,
                                                 float scores_max[Mloop], float scores_sum[Mloop],
                                                 float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
#else
                                                 uint32_t k_raw[HEAD_SIZE / 64][4], uint32_t v_raw[4][HEAD_SIZE_V / 64],
                                                 int block_idx, bool first_page, int warp_id, int v_token_stride,
                                                 const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                 int valid_seqlen_kv, float scale_softmax_log2,
                                                 float scores_max[Mloop], float scores_sum[Mloop],
                                                 float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    constexpr int NW = HEAD_SIZE_V / 64;
#endif
    const int lane_id = threadIdx.x % WARP_SIZE;
    float4_t s_reg[Mloop][1];
    qk_fp8_slice<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
        k_raw, smem_q, seqlen_q_per_kv_head, s_reg);

    const int token_base = block_idx * BLOCK_SIZE + 16 * warp_id;
    const bool partial_page = token_base + 15 >= valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
                if (token_base + 4 * (lane_id >> 4) + j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 4, 1);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        p_reg[m] = {(_Float16)(s_reg[m][0])[0], (_Float16)(s_reg[m][0])[1],
                    (_Float16)(s_reg[m][0])[2], (_Float16)(s_reg[m][0])[3]};
    }
    if (partial_page) {
        // 超出有效长度的 token 的 V 可能为任意字节, 置零避免 0 * NaN
        const uint32_t keep = vt_token_mask(valid_seqlen_kv - (token_base + 4 * (lane_id >> 4)));
        #pragma unroll
        for (int q = 0; q < HEAD_SIZE_V / 16; ++q) v_raw[q] &= keep;
    }
    #pragma unroll
    for (int q = 0; q < HEAD_SIZE_V / 16; ++q) {
        const half4_v v_fragment = decode_vt_e5m2x4_f16(v_raw[q]);
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) builtin_amdgcn_mmac<half4_v, true>(p_reg[m], v_fragment, acc_o[m][q]);
    }
}

// 处理一个 page; PREFETCH 为 true 时在消费槽位后发射 next 页的数据, 所有加载都不在分支内,
// 编译器插入的 s_waitcnt 才能精确等待对应的槽位
template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                int k_lane_offset, int v_lane_offset,
                                                uint32_t k_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][4],
                                                uint32_t v_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::V_AHEAD * HEAD_SIZE_V / 64][4],
#else
                if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) (s_reg[m][0])[j] = -INFINITY;
            }
        }
    }
    score_mask(s_reg, token_base, 0, 1, 4);

    direct_softmax<Mloop, HEAD_SIZE_V, 1>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        p_reg[m] = {(_Float16)(s_reg[m][0])[0], (_Float16)(s_reg[m][0])[1],
                    (_Float16)(s_reg[m][0])[2], (_Float16)(s_reg[m][0])[3]};
    }
    if (partial_page) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (token_base + (lane_id >> 4) + 4 * j >= valid_seqlen_kv) {
                #pragma unroll
                for (int w = 0; w < NW; ++w) v_raw[j][w] = 0;
            }
        }
    }
    pv_fp8_tile_f16<Mloop, HEAD_SIZE_V>(v_raw, p_reg, acc_o);
}

// 处理一个 page; PREFETCH 为 true 时在消费槽位后发射 next 页的数据, 所有加载都不在分支内,
// 编译器插入的 s_waitcnt 才能精确等待对应的槽位
template <bool PREFETCH, typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page(buffer_rsrc_t k_page, buffer_rsrc_t v_page,
                                                buffer_rsrc_t k_next, buffer_rsrc_t v_next,
                                                int k_lane_offset, int v_lane_offset,
                                                uint32_t k_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::K_AHEAD][4][4],
                                                uint32_t v_slot[DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::V_AHEAD][4][HEAD_SIZE_V / 64],
#endif
                                                int block_idx, bool first_page, int k_token_stride, int v_token_stride,
                                                const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                int valid_seqlen_kv, float scale_softmax_log2,
                                                float scores_max[Mloop], float scores_sum[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], ScoreMask &score_mask)
{
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
    constexpr int K_CHUNKS = Cfg::K_CHUNKS;
    constexpr int V_TILES = BLOCK_SIZE / 16;
    const int lane_id = threadIdx.x % WARP_SIZE;

    float4_t s_reg[Mloop][V_TILES];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int i = 0; i < V_TILES; ++i) s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    scalarx8_t q_pair[2][Mloop][2];
    #pragma unroll
    for (int c = 0; c < K_CHUNKS; ++c) {
        if constexpr (KSlotMap<HEAD_SIZE, Mloop>::PAIRED) {
            if (c % 2 == 0) {
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2), seqlen_q_per_kv_head, q_pair[0]);
                load_q_fp8_chunk<scalarx8_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD>(smem_q, 2 * (c / 2) + 1, seqlen_q_per_kv_head, q_pair[1]);
            }
        }
        qk_fp8_chunk<scalarx8_t, scalarx4_t, scalar_t, Mloop, HEAD_SIZE, Q_LDS_PAD, is_query_fp16>(
            k_slot[c % KA], smem_q, c, seqlen_q_per_kv_head, s_reg, q_pair);
        if (c + KA < K_CHUNKS) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c + KA, k_slot[c % KA]);
        } else if constexpr (PREFETCH) {
            load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_next, k_lane_offset, k_token_stride, c + KA - K_CHUNKS, k_slot[c % KA]);
        }
    }

    const bool partial_page = (block_idx + 1) * BLOCK_SIZE > valid_seqlen_kv;
    if (partial_page) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            #pragma unroll
            for (int i = 0; i < V_TILES; ++i) {
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
                const int token_idx = block_idx * BLOCK_SIZE + i + (lane_id >> 4) * 16;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (token_idx + j * 4 >= valid_seqlen_kv) (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 16, 4);
#else
                const int token_idx = block_idx * BLOCK_SIZE + i + (lane_id >> 4) * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (token_idx + j * 16 >= valid_seqlen_kv) (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    score_mask(s_reg, block_idx * BLOCK_SIZE, 1, 4, 16);
#endif

    direct_softmax<Mloop, HEAD_SIZE_V, V_TILES>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2, first_page);

    half4_v p_reg[V_TILES][Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        #pragma unroll
        for (int t = 0; t < V_TILES; ++t) {
            p_reg[t][m] = {(_Float16)(s_reg[m][0])[t], (_Float16)(s_reg[m][1])[t],
                           (_Float16)(s_reg[m][2])[t], (_Float16)(s_reg[m][3])[t]};
        }
    }

#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    // V^T 按 16 通道组消费, 槽位数为 V_AHEAD 个 tile 对应的字节数 (VGA 个组); 消费后立即发射后续组
    constexpr int VG = HEAD_SIZE_V / 16;
    constexpr int VGA = VA * HEAD_SIZE_V / 64;
    static_assert(V_TILES == 4 && VG % VGA == 0, "V^T ring slots must repeat per page");
    #pragma unroll
    for (int q = 0; q < VG; ++q) {
        uint32_t (&raw)[4] = v_slot[q % VGA];
        if (partial_page) {
            // 超出有效长度的 token 的 V 可能为任意字节, 置零避免 0 * NaN
            const int token_idx = block_idx * BLOCK_SIZE + (lane_id >> 4) * 16;
            #pragma unroll
            for (int t = 0; t < 4; ++t) raw[t] &= vt_token_mask(valid_seqlen_kv - (token_idx + 4 * t));
        }
        pv_fp8_group_f16<Mloop, HEAD_SIZE_V>(raw, p_reg, acc_o, q);
        if (q + VGA < VG) {
            load_v_fp8_group<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, q + VGA, raw);
        } else if constexpr (PREFETCH) {
            load_v_fp8_group<HEAD_SIZE_V>(v_next, v_lane_offset, v_token_stride, q + VGA - VG, raw);
#else
    #pragma unroll
    for (int t = 0; t < V_TILES; ++t) {
        uint32_t (&raw)[4][Cfg::V_NW] = v_slot[t % VA];
        if (partial_page) {
            // 超出有效长度的 token 的 V 可能为任意字节, 置零避免 0 * NaN
            const int token_idx = block_idx * BLOCK_SIZE + t * 16 + (lane_id >> 4) * 4;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (token_idx + k >= valid_seqlen_kv) {
                    #pragma unroll
                    for (int w = 0; w < Cfg::V_NW; ++w) raw[k][w] = 0;
                }
            }
        }
        pv_fp8_tile_f16<Mloop, HEAD_SIZE_V>(raw, p_reg[t], acc_o);
        if (t + VA < V_TILES) {
            load_v_fp8_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t + VA, raw);
        } else if constexpr (PREFETCH) {
            load_v_fp8_tile<HEAD_SIZE_V>(v_next, v_lane_offset, v_token_stride, t + VA - V_TILES, raw);
#endif
        }
    }
}

// 常规模式: 每个 warp 依次处理 partition_idx + warp_id * P, 再每次前进 4P 的 page;
// 页内切分模式: 4 个 warp 依次处理 partition_idx, partition_idx + P, ... 的同一 page 的不同 token。
// q_commit 在首个 page 的加载发射后写 Q 到 LDS 并同步, 返回 LDS 中 Q 相对原值的缩放倒数 (softmax 缩放乘以该值);
// score_mask 处理 alibi/causal 掩码。
template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, int Mloop, int HEAD_SIZE, int HEAD_SIZE_V,
          int BLOCK_SIZE, int Q_LDS_PAD, bool is_query_fp16, typename QCommit, typename ScoreMask>
__device__ __forceinline__ void direct_fp8_page_loop(const uint8_t *k_base, const uint8_t *v_base,
                                                     int k_block_stride, int v_block_stride,
                                                     const int *block_table, int partition_idx, int num_partitions,
                                                     int warp_id, bool slice_mode, int end_block, int num_kv_heads,
                                                     const scalar_t *smem_q, int seqlen_q_per_kv_head,
                                                     int valid_seqlen_kv, float scale_softmax_log2,
                                                     float scores_max[Mloop], float scores_sum[Mloop],
                                                     float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                     QCommit q_commit, ScoreMask score_mask,
                                                     const int k_token_stride, const int v_token_stride)
{
    static_assert(BLOCK_SIZE == 64, "direct FP8 path assumes 64-token pages");
    using Cfg = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>;
    constexpr int KA = Cfg::K_AHEAD;
    constexpr int VA = Cfg::V_AHEAD;
    const int first_block = slice_mode ? partition_idx : partition_idx + warp_id * num_partitions;
    const int page_stride = slice_mode ? num_partitions : 4 * num_partitions;
    const int num_pages = first_block < end_block ? DIVIDE_ROUND_UP(end_block - first_block, page_stride) : 0;

    // 没有 page 的 warp 读取物理块 0 (总是有效), 使首个 page 的加载不在分支内
    const int first_entry = block_table[MAX_T(MIN_T(first_block, end_block - 1), 0)];
    const int64_t first_physical = num_pages > 0 ? static_cast<int64_t>(first_entry) : 0;
    buffer_rsrc_t k_page = make_page_rsrc(k_base + first_physical * k_block_stride);
    buffer_rsrc_t v_page = make_page_rsrc(v_base + first_physical * v_block_stride);

    // 页内切分模式每个线程块最多处理 2 个 page, 不做跨 page 预取, 只保留一份循环主体以控制代码体积
    if (__builtin_expect(slice_mode, 0)) {
        const int k_lane_offset = k_fp8_slice_lane_offset(k_token_stride, warp_id);
        const int v_lane_offset = v_fp8_slice_lane_offset<HEAD_SIZE_V>(v_token_stride, warp_id);
        uint32_t k_raw[Cfg::K_CHUNKS][4];
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
        uint32_t v_raw[Cfg::V_CH];
#else
        uint32_t v_raw[4][Cfg::V_NW];
#endif
        load_k_fp8_slice<HEAD_SIZE>(k_page, k_lane_offset, k_raw);
        load_v_fp8_slice<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, v_raw);
        const float committed_q_scale = q_commit();
        const bool q_bf16 = committed_q_scale < 0.0f;
        scale_softmax_log2 *= fabsf(committed_q_scale);
        auto run_pages = [&](auto query_scalar) {
            using page_scalar_t = decltype(query_scalar);
            using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
            using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
            int block_idx = first_block;
        for (int p = 0; p < num_pages; ++p) {
            if (p > 0) {
                const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
                load_k_fp8_slice<HEAD_SIZE>(make_page_rsrc(k_base + physical_block_number * k_block_stride), k_lane_offset, k_raw);
                load_v_fp8_slice<HEAD_SIZE_V>(make_page_rsrc(v_base + physical_block_number * v_block_stride), v_lane_offset,
                                              v_token_stride, v_raw);
            }
            direct_fp8_slice<page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
                k_raw, v_raw, block_idx, p == 0, warp_id, v_token_stride, reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv,
                scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
            block_idx += page_stride;
        }
        };
        if (q_bf16) run_pages(uint16_t{});
        else run_pages(scalar_t{});
        direct_softmax_finish<Mloop>(scores_sum);
        return;
    }

    const int k_lane_offset = k_fp8_lane_offset(k_token_stride);
    const int v_lane_offset = v_fp8_lane_offset<HEAD_SIZE_V>(v_token_stride);
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    constexpr int VGA = VA * HEAD_SIZE_V / 64;
    uint32_t k_slot[KA][4][4];
    uint32_t v_slot[VGA][4];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int g = 0; g < VGA; ++g) load_v_fp8_group<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, g, v_slot[g]);
#else
    uint32_t k_slot[KA][4][4];
    uint32_t v_slot[VA][4][Cfg::V_NW];
    #pragma unroll
    for (int c = 0; c < KA; ++c) load_k_fp8_chunk<HEAD_SIZE, Mloop>(k_page, k_lane_offset, k_token_stride, c, k_slot[c]);
    #pragma unroll
    for (int t = 0; t < VA; ++t) load_v_fp8_tile<HEAD_SIZE_V>(v_page, v_lane_offset, v_token_stride, t, v_slot[t]);
#endif
    const float committed_q_scale = q_commit();
    const bool q_bf16 = committed_q_scale < 0.0f;
    scale_softmax_log2 *= fabsf(committed_q_scale);

    auto run_pages = [&](auto query_scalar) {
        using page_scalar_t = decltype(query_scalar);
        using page_x8_t = __attribute__((__vector_size__(8 * sizeof(page_scalar_t)))) page_scalar_t;
        using page_x4_t = __attribute__((__vector_size__(4 * sizeof(page_scalar_t)))) page_scalar_t;
        int block_idx = first_block;
    for (int p = 0; p + 1 < num_pages; ++p) {
        const int64_t next_physical = static_cast<int64_t>(block_table[block_idx + page_stride]);
        const buffer_rsrc_t k_next = make_page_rsrc(k_base + next_physical * k_block_stride);
        const buffer_rsrc_t v_next = make_page_rsrc(v_base + next_physical * v_block_stride);
        direct_fp8_page<true, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
            k_page, v_page, k_next, v_next, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, p == 0, k_token_stride, v_token_stride,
            reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
        k_page = k_next;
        v_page = v_next;
        block_idx += page_stride;
    }
    if (num_pages > 0) {
        direct_fp8_page<false, page_x8_t, page_x4_t, page_scalar_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, std::is_same<page_scalar_t, _Float16>::value>(
            k_page, v_page, k_page, v_page, k_lane_offset, v_lane_offset, k_slot, v_slot, block_idx, num_pages == 1, k_token_stride, v_token_stride,
            reinterpret_cast<const page_scalar_t *>(smem_q), seqlen_q_per_kv_head, valid_seqlen_kv, scale_softmax_log2, scores_max, scores_sum, acc_o, score_mask);
    }
    };
    if (q_bf16) run_pages(uint16_t{});
    else run_pages(scalar_t{});
    direct_softmax_finish<Mloop>(scores_sum);
}

constexpr int DIRECT_COMBINE_MAX_GROUPS = 4096;
constexpr int DIRECT_COMBINE_COUNTER_STRIDE = 16;   // 每组计数器独占 64 字节
__device__ __forceinline__ void direct_combine_arrive(unsigned int *counter);
template <typename scalar_t, int HEAD_SIZE_V, bool REPLICATE_META = false>
__device__ __forceinline__ void direct_combine_head_in_kernel(unsigned int *counter, int num_partitions, int num_combiners,
                                                              scalar_t *out, scalar_t *out_tmp, int head, int alloc_partitions,
                                                              char *shared_mem);



// ---- 直接加载路径的 epilogue ----
//  各 warp 将行 max/sum 写入独立于 Q 与归约区的 LDS, 一次同步后各自求全局 max/sum 并缩放 acc_o
//  (同时确认所有 warp 已结束读取 Q); 随后按 16x64 分块在 LDS 中累加 4 个 warp 的结果。
template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS>
struct DirectEpilogueShape {
    static constexpr int TILE_FLOATS = 16 * (64 + 4);                  // 每个 warp 的 16x64 分块, 行填充 4
    static constexpr int RED_BYTES = NUM_WARPS * TILE_FLOATS * static_cast<int>(sizeof(float));
    static constexpr int meta_offset(int q_bytes)
    {
        return (((q_bytes > RED_BYTES ? q_bytes : RED_BYTES) + 15) / 16) * 16;
    }
    static constexpr int smem_bytes(int q_bytes, int max_rows)
    {
        return meta_offset(q_bytes) + max_rows * NUM_WARPS * 2 * static_cast<int>(sizeof(float));
    }
};

template <int Mloop, int HEAD_SIZE_V, int NUM_WARPS, int Q_BYTES, typename StoreO, typename StoreMeta>
__device__ __forceinline__ void direct_epilogue(float scores_max[Mloop], float scores_sum[Mloop],
                                                float4_t acc_o[Mloop][HEAD_SIZE_V / 16], char *shared_mem,
                                                int rows, int warp_id, float scale_softmax_log2,
                                                StoreO &store_o, StoreMeta &store_meta)
{
    using Shape = DirectEpilogueShape<Mloop, HEAD_SIZE_V, NUM_WARPS>;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int acc_row = lane_id & 15;
    const int acc_col = lane_id >> 4;
    float *meta = reinterpret_cast<float *>(shared_mem + Shape::meta_offset(Q_BYTES));
    float *red = reinterpret_cast<float *>(shared_mem);
    using float2_v = float __attribute__((ext_vector_type(2)));

    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int row = m * 16 + acc_row;
        if (row < rows && acc_col == 0) {
            *reinterpret_cast<float2_v *>(meta + (row * NUM_WARPS + warp_id) * 2) = {scores_max[m], scores_sum[m]};
        }
    }
    // 同时保证所有 warp 已结束对 Q 的读取, 归约区可以复用 Q 的 LDS
    lds_barrier();

    float row_max[Mloop];
    float row_sum[Mloop];
    #pragma unroll
    for (int m = 0; m < Mloop; ++m) {
        const int row = m * 16 + acc_row;
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
        if (row < rows) {
            float2_v values[NUM_WARPS];
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) values[w] = *reinterpret_cast<const float2_v *>(meta + (row * NUM_WARPS + w) * 2);
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) row_max[m] = MAX_T(row_max[m], values[w][0]);
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                row_sum[m] += values[w][1] * direct_exp_difference(values[w][0], row_max[m], scale_softmax_log2);
            }
            const float scale = direct_exp_difference(scores_max[m], row_max[m], scale_softmax_log2) / (row_sum[m] + 1e-6f);
            #pragma unroll
            for (int i = 0; i < HEAD_SIZE_V / 16; ++i) acc_o[m][i] *= scale;
        }
    }

#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
    #pragma unroll
    for (int tile_idx = 0; tile_idx < num_16x64_tiles; ++tile_idx) {
        const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
        const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
        if (tile_idx > 0) lds_barrier();
        store_direct_v_acc_group<Mloop, HEAD_SIZE_V>(red + warp_id * Shape::TILE_FLOATS + acc_row * (64 + 4),
                                                     acc_o, tile_idx_row, tile_idx_col, acc_col);
        lds_barrier();
        const int out_r2g_row = threadIdx.x >> 4;
        const int out_r2g_col = threadIdx.x & 15;
        const int row = tile_idx_row * 16 + out_r2g_row;
        if (row < rows) {
            float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                sums += *reinterpret_cast<const float4_t *>(red + w * Shape::TILE_FLOATS + out_r2g_row * (64 + 4) + out_r2g_col * 4);
            }
            store_o(row, tile_idx_col * 64 + out_r2g_col * 4, sums);
#else
    // 每轮归约的列数: acc_o 第 q 个 tile 中 lane 组 g 的第 j 个元素为通道 CH * (4j + g) + q, 每个 j 覆盖 4 * CH 个连续通道;
    // CH 整除 16 时每轮 64 列 (16 / CH 个 j), HEAD_SIZE_V == 192 (CH = 12) 时每轮一个 j, 即 48 列
    constexpr int V_CH = HEAD_SIZE_V / 16;
    constexpr int GROUP_COLS = 16 % V_CH == 0 ? 64 : 4 * V_CH;
    constexpr int GROUPS = HEAD_SIZE_V / GROUP_COLS;
    static_assert(GROUPS * GROUP_COLS == HEAD_SIZE_V && GROUP_COLS <= 64, "epilogue column groups");
    constexpr int num_16x64_tiles = Mloop * GROUPS;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < num_16x64_tiles; ++tile_idx) {
        const int tile_idx_row = tile_idx / GROUPS;
        const int tile_idx_col = tile_idx % GROUPS;
        if (tile_idx > 0) lds_barrier();
        store_direct_v_acc_group<Mloop, HEAD_SIZE_V>(red + warp_id * Shape::TILE_FLOATS + acc_row * (64 + 4),
                                                     acc_o, tile_idx_row, tile_idx_col, acc_col);
        lds_barrier();
        const int out_r2g_row = threadIdx.x >> 4;
        const int out_r2g_col = threadIdx.x & 15;
        const int row = tile_idx_row * 16 + out_r2g_row;
        if (row < rows && out_r2g_col * 4 < GROUP_COLS) {
            float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                sums += *reinterpret_cast<const float4_t *>(red + w * Shape::TILE_FLOATS + out_r2g_row * (64 + 4) + out_r2g_col * 4);
            }
            store_o(row, tile_idx_col * GROUP_COLS + out_r2g_col * 4, sums);
#endif
        }
    }

    if (warp_id == 0 && acc_col == 0) {
        #pragma unroll
        for (int m = 0; m < Mloop; ++m) {
            const int row = m * 16 + acc_row;
            if (row < rows) store_meta(row, row_max[m], row_sum[m]);
        }
    }
}

// ---------------------------------------------------------------------------
// 直接加载路径的 partition 合并:
//  每个线程以 16 字节读取一个 partition 输出行的一段, 所需的全部 partial O 与 max/sum
//  一次性发射 (按加载数选择直线化实例, 不做冗余读取), 访存延迟只暴露一次。

template <typename scalar_t, int HEAD_SIZE>
struct CombineDirectShape {
    static constexpr int VEC = 8;                         // 每次加载 8 个 b16
    static constexpr int TPR = HEAD_SIZE / VEC;           // 每个 partition 行的线程数
    static constexpr int ROWS = 256 / TPR;                // 每轮覆盖的 partition 数
    static constexpr int MAX_LOADS = HEAD_SIZE > 128 ? 32 : 16;
    static constexpr int META_SLOTS = 2;                  // 每线程直线化处理的 max/sum 个数
    static constexpr int RED_FLOATS = ROWS * HEAD_SIZE;
};

template <typename scalar_t, int HEAD_SIZE, int NL, bool GROUPED = false>
__device__ __forceinline__ void combine_direct_body(scalar_t *__restrict__ out_head,
                                                    const scalar_t *__restrict__ tmp_out,
                                                    const float *__restrict__ exp_sums,
                                                    const float *__restrict__ max_logits,
                                                    int num_partitions, float *smem, bool write_out = true)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    constexpr int ROWS = Shape::ROWS;
    constexpr int TPR = Shape::TPR;
    constexpr int VEC = Shape::VEC;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const bool active = tid < ROWS * TPR;
    const int row = active ? tid / TPR : 0;
    const int col = active ? (tid % TPR) * VEC : 0;
    float *smem_red = smem;
    float *smem_scalar = smem_red + Shape::RED_FLOATS;
    float *smem_weight = smem_scalar + 8;

    // max/sum 与 partial O 同时发射; 越界项读取最后一个有效 partition, 随后以权重 0 屏蔽
    float part_max[Shape::META_SLOTS];
    float part_sum[Shape::META_SLOTS];
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        const int i = MIN_T(tid + j * 256, num_partitions - 1);
        part_max[j] = max_logits[i];
        part_sum[j] = exp_sums[i];
    }
    u32x4 values[NL];
    if constexpr (!GROUPED) {
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int p = MIN_T(row + k * ROWS, num_partitions - 1);
            values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
        }
    }

    float local_max = -INFINITY;
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        if (tid + j * 256 < num_partitions) local_max = MAX_T(local_max, part_max[j]);
    }
    for (int i = tid + Shape::META_SLOTS * 256; i < num_partitions; i += 256) local_max = MAX_T(local_max, max_logits[i]);
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
    if (lane == 0) smem_scalar[warp] = local_max;
    lds_barrier();
    const float global_max = MAX_T(MAX_T(smem_scalar[0], smem_scalar[1]), MAX_T(smem_scalar[2], smem_scalar[3]));

    float local_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < Shape::META_SLOTS; ++j) {
        const int i = tid + j * 256;
        if (i < num_partitions) {
            const float weight = part_sum[j] > 0.0f ? part_sum[j] * __builtin_amdgcn_exp2f(part_max[j] - global_max) : 0.0f;
            smem_weight[i] = weight;
            local_sum += weight;
        }
    }
    for (int i = tid + Shape::META_SLOTS * 256; i < num_partitions; i += 256) {
        const float s = exp_sums[i];
        const float weight = s > 0.0f ? s * __builtin_amdgcn_exp2f(max_logits[i] - global_max) : 0.0f;
        smem_weight[i] = weight;
        local_sum += weight;
    }
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) local_sum += __shfl_xor(local_sum, mask);
    if (lane == 0) smem_scalar[4 + warp] = local_sum;
    lds_barrier();
    const float inv_sum = __fdividef(1.0f, (smem_scalar[4] + smem_scalar[6]) + (smem_scalar[5] + smem_scalar[7]) + 1e-6f);

    float acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.0f;
    // GROUPED: partition 数超过直线化实例覆盖范围时, 按 NL 次加载为一组循环处理
    const int groups = GROUPED ? DIVIDE_ROUND_UP(num_partitions, NL * ROWS) : 1;
    for (int group = 0; group < groups; ++group) {
        if constexpr (GROUPED) {
            #pragma unroll
            for (int k = 0; k < NL; ++k) {
                const int p = MIN_T(row + (group * NL + k) * ROWS, num_partitions - 1);
                values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
            }
        }
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int p = row + (group * NL + k) * ROWS;
            const float weight = (active && p < num_partitions) ? smem_weight[p] : 0.0f;
            const scalar_t *v = reinterpret_cast<const scalar_t *>(&values[k]);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) acc[e] += to_float(v[e]) * weight;
        }
    }
    if (active) {
        #pragma unroll
        for (int e = 0; e < VEC; e += 4) {
            *reinterpret_cast<float4_t *>(smem_red + row * HEAD_SIZE + col + e) = {acc[e], acc[e + 1], acc[e + 2], acc[e + 3]};
        }
    }
    lds_barrier();
    for (int c = tid; write_out && c < HEAD_SIZE; c += 256) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) sum += smem_red[r * HEAD_SIZE + c];
        scalar_t converted;
        from_float(converted, sum * inv_sum);
        out_head[c] = converted;
    }
}

// 按 partition 数选择直线化实例合并一个 head 的 partial; write_out 为 false 时只执行不写出 (用于预热指令缓存)
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void combine_direct_head(scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
                                                    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
                                                    int num_partitions, float *smem, bool write_out = true)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    const int loads = DIVIDE_ROUND_UP(num_partitions, Shape::ROWS);
    if (loads <= 1) {
        combine_direct_body<scalar_t, HEAD_SIZE, 1>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 2) {
        combine_direct_body<scalar_t, HEAD_SIZE, 2>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 4) {
        combine_direct_body<scalar_t, HEAD_SIZE, 4>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 8) {
        combine_direct_body<scalar_t, HEAD_SIZE, 8>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (loads <= 16) {
        combine_direct_body<scalar_t, HEAD_SIZE, 16>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else if (Shape::MAX_LOADS == 32 && loads <= 32) {
        combine_direct_body<scalar_t, HEAD_SIZE, Shape::MAX_LOADS>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    } else {
        combine_direct_body<scalar_t, HEAD_SIZE, Shape::MAX_LOADS, true>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, write_out);
    }
}

// Each wave independently prepares the same <=64 partition weights. The
// original eight-row partial-output mapping and final LDS reduction stay intact.
template <typename scalar_t, int HEAD_SIZE, int NL>
__device__ __forceinline__ void combine_direct_replicated_body(
    scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
    int num_partitions, float *smem)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    constexpr int ROWS = Shape::ROWS, TPR = Shape::TPR, VEC = Shape::VEC;
    static_assert(HEAD_SIZE == 256 && NL * ROWS <= WARP_SIZE, "replicated metadata geometry");
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int row = tid / TPR;
    const int col = (tid % TPR) * VEC;

    const int i = MIN_T(lane, num_partitions - 1);
    const float part_max = max_logits[i];
    const float part_sum = exp_sums[i];
    u32x4 values[NL];
    #pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int p = MIN_T(row + k * ROWS, num_partitions - 1);
        values[k] = *reinterpret_cast<const u32x4 *>(tmp_out + p * HEAD_SIZE + col);
    }

    float local_max = lane < num_partitions ? MAX_T(-INFINITY, part_max) : -INFINITY;
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        local_max = MAX_T(local_max, __shfl_xor(local_max, mask));
    // Match the old metadata publication by lane 0, including the empty waves.
    const float wave_max = __builtin_bit_cast(float,
        __builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t, local_max)));
    const float global_max = MAX_T(MAX_T(wave_max, -INFINITY), MAX_T(-INFINITY, -INFINITY));
    float weight_lane = 0.0f;
    if (lane < num_partitions)
        weight_lane = part_sum > 0.0f ? part_sum * __builtin_amdgcn_exp2f(part_max - global_max) : 0.0f;
    float local_sum = weight_lane;
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        local_sum += __shfl_xor(local_sum, mask);
    float ordered_sum = __builtin_bit_cast(float,
        __builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t, local_sum)));
    // Preserve the original reduction/epsilon boundary under fast math.
    asm volatile("" : "+v"(ordered_sum));
    const float inv_sum = __fdividef(1.0f, (ordered_sum + 0.0f) + (0.0f + 0.0f) + 1e-6f);

    float acc[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.0f;
    #pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int p = row + k * ROWS;
        const float weight = __shfl(weight_lane, p);
        const scalar_t *v = reinterpret_cast<const scalar_t *>(&values[k]);
        #pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] += to_float(v[e]) * weight;
    }
    #pragma unroll
    for (int e = 0; e < VEC; e += 4)
        *reinterpret_cast<float4_t *>(smem + row * HEAD_SIZE + col + e) =
            {acc[e], acc[e + 1], acc[e + 2], acc[e + 3]};
    lds_barrier();
    for (int c = tid; c < HEAD_SIZE; c += 256) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) sum += smem[r * HEAD_SIZE + c];
        scalar_t converted;
        from_float(converted, sum * inv_sum);
        out_head[c] = converted;
    }
}

template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void combine_direct_replicated_head(
    scalar_t *__restrict__ out_head, const scalar_t *__restrict__ tmp_out,
    const float *__restrict__ exp_sums, const float *__restrict__ max_logits,
    int num_partitions, float *smem)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    const int loads = DIVIDE_ROUND_UP(num_partitions, Shape::ROWS);
    if (loads <= 1)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 1>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else if (loads <= 2)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 2>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else if (loads <= 4)
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 4>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
    else
        combine_direct_replicated_body<scalar_t, HEAD_SIZE, 8>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
}

// 直接加载路径在同一次启动内合并 partition:
//  一维 grid 在全部 partition 线程块 (自有与溢出区) 之后追加每组每行一个合并线程块
//  (与合并 kernel 一样每个 head 一个线程块)。partition 线程块写完 partial 后对组计数器加一;
//  合并线程块排在所有 partition 线程块之后; 此融合路径依赖目标设备的 grid 派发次序。
//  合并线程块等到所有 partial 写出后才读取, 避免预读未初始化的临时输出。
//  省去合并 kernel 的启动开销与 kernel 间隔; 计数器在每次调用结束时由最后一个合并线程块复位。
// ---------------------------------------------------------------------------

// 每个设备、执行流各自持有计数器; 独立流之间不共享合并状态。
// 首次调用在流上清零, 此后的调用由最后一个合并线程块复位。
static unsigned int *direct_combine_counters(int device, cudaStream_t stream)
{
    using Key = std::pair<int, uintptr_t>;
    static std::mutex lock;
    static std::map<Key, unsigned int *> counters;
    struct LastCounter { int device; uintptr_t stream; unsigned int *ptr; };
    static thread_local LastCounter last = {-1, 0, nullptr};
    const Key key(device, reinterpret_cast<uintptr_t>(stream));
    // Always check capture before the cache: graphs use the stateless optimized
    // merge even after a normal invocation on the same stream.
    hipStreamCaptureStatus capture_status;
    if (hipStreamIsCapturing(stream, &capture_status) != hipSuccess || capture_status != hipStreamCaptureStatusNone) return nullptr;
    // Repeated calls on one thread/stream need no global lock or map lookup.
    // The map owns the allocation for the process lifetime; stream changes
    // still go through the synchronized lookup below.
    if (last.ptr != nullptr && last.device == device && last.stream == key.second) return last.ptr;
    std::lock_guard<std::mutex> guard(lock);
    const auto found = counters.find(key);
    if (found != counters.end()) {
        last = {device, key.second, found->second};
        return found->second;
    }
    constexpr size_t bytes = sizeof(unsigned int) * DIRECT_COMBINE_MAX_GROUPS * DIRECT_COMBINE_COUNTER_STRIDE;
    unsigned int *ptr = nullptr;
    if (hipMalloc(&ptr, bytes) != hipSuccess) return nullptr;
    if (hipMemsetAsync(ptr, 0, bytes, stream) != hipSuccess) {
        (void)hipFree(ptr);
        return nullptr;
    }
    counters.emplace(key, ptr);
    last = {device, key.second, ptr};
    return ptr;
}

// partition 线程块: partial 输出与 max/sum 写出后到达组计数器
__device__ __forceinline__ void direct_combine_arrive(unsigned int *counter)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// 合并线程块: 等待组内 num_partitions 个 partition 线程块到达后合并一个 head; num_combiners 个合并线程块都
// 完成计数后由最后一个复位计数器
template <typename scalar_t, int HEAD_SIZE_V, bool REPLICATE_META>
__device__ __forceinline__ void direct_combine_head_in_kernel(unsigned int *counter, int num_partitions, int num_combiners,
                                                              scalar_t *out, scalar_t *out_tmp, int head, int alloc_partitions,
                                                              char *shared_mem)
{
    float *smem = reinterpret_cast<float *>(shared_mem);
    int *flag = reinterpret_cast<int *>(smem + CombineDirectShape<scalar_t, HEAD_SIZE_V>::RED_FLOATS + 8 + alloc_partitions);
    const int offset = head * alloc_partitions;
    const float *exp_sums = reinterpret_cast<const float *>(out_tmp) + offset;
    const float *max_logits = reinterpret_cast<const float *>(out_tmp + max_tmp_offset) + offset;
    const scalar_t *tmp_out = out_tmp + out_tmp_offset + static_cast<int64_t>(offset) * HEAD_SIZE_V;
    scalar_t *out_head = out + static_cast<int64_t>(head) * HEAD_SIZE_V;
    const unsigned int expected = static_cast<unsigned int>(num_partitions);
    if (threadIdx.x == 0) *flag = __hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) >= expected;
    __syncthreads();
    if (*flag == 0) {
        if (threadIdx.x == 0) {
            while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < expected) __builtin_amdgcn_s_sleep(4);
        }
        __syncthreads();
    }
    __threadfence();
    if constexpr (REPLICATE_META) {
        if (num_partitions <= WARP_SIZE)
            combine_direct_replicated_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem);
        else
            combine_direct_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, true);
    } else {
        combine_direct_head<scalar_t, HEAD_SIZE_V>(out_head, tmp_out, exp_sums, max_logits, num_partitions, smem, true);
    }
    if (threadIdx.x == 0) {
        if (atomicAdd(counter, 1u) == expected + static_cast<unsigned int>(num_combiners) - 1) atomicExch(counter, 0u);
    }
}

// 行切分路径的合并: partition 数按 rowsplit_d256::num_partitions 计算, 其余与常规合并 kernel 相同

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int CTAS_PER_CU, bool LONG_SINGLE_STREAM, bool RESIDENT_CAP, bool ONE_PAGE_SLICE>
__global__ __launch_bounds__(256, 1) void paged_attention_combine_direct(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ out_tmp,
    const int* __restrict__ seq_lens,
    int balanced_seq_len_tile,
    int total_tile,
    int num_heads)
{
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    extern __shared__ char shared_mem[];
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    const DirectBatchBudget budget = direct_batch_budget<LONG_SINGLE_STREAM>(seq_lens, gridDim.y, total_tile, valid_seqlen_kv);
    const int seq_tile = direct_tile_of(valid_num_block_pages, budget.total_pages, gridDim.y, total_tile, balanced_seq_len_tile);
    const int num_partitions = direct_partitions_for_tile<CTAS_PER_CU>(valid_num_block_pages, seq_tile, budget.halve);
    if (num_partitions == 1) {
        return;
    }
    const int offset = (batch_idx * num_heads + head_idx) * balanced_seq_len_tile;
    const float* exp_sums = reinterpret_cast<const float*>(out_tmp) + offset;
    const float* max_logits = reinterpret_cast<const float*>(out_tmp + max_tmp_offset) + offset;
    const scalar_t* tmp_out = out_tmp + out_tmp_offset + offset * HEAD_SIZE;
    scalar_t* out_head = out + (batch_idx * num_heads + head_idx) * HEAD_SIZE;
    combine_direct_head<scalar_t, HEAD_SIZE>(out_head, tmp_out, exp_sums, max_logits, num_partitions,
                                             reinterpret_cast<float*>(shared_mem));
}



// ---------------------------------------------------------------------------
// 主 kernel: 一个线程块处理 (kv head, Q tile, partition, batch)。NUM_Q_TILES = 1 时与 BHSD 单 tile kernel 的 direct
// 路径相同, NUM_Q_TILES = 2 时与 BHSD qtile kernel 的 direct 路径相同; 只有 K/V 寻址改为 BSHD:
//  page 内同一 KV head 相邻 token 相隔 token_stride (= num_kv_heads * head_dim) 字节, head 起点相隔 head_stride 字节。
// grid: 一维, num_kv_heads * NUM_Q_TILES * (batch * own_tile + 溢出区行数 + batch * 合并线程块数) (角色划分见 kernel 开头)
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_SIZE, int Q_TILE, int NUM_Q_TILES, bool LONG_SINGLE_STREAM>
__launch_bounds__(256) __global__ void PA_E5M2_936_KERNEL(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,
    const scalar_t* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const uint8_t* __restrict__ k_cache,        // [num_blocks, block_size, num_kv_heads, head_size]
    const uint8_t* __restrict__ v_cache,        // [num_blocks, block_size, num_kv_heads, head_size]
    const int num_heads,                        // seqlen_q * num_q_heads
    const int num_kv_heads,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int k_block_stride, const int v_block_stride,     // 字节
    const int k_token_stride, const int v_token_stride,     // 字节
    const int k_head_stride, const int v_head_stride,       // 字节
    int balanced_seq_len_tile,                  // 每个序列 partition 数的上限 (partial 输出的步长)
    int total_tile,                             // 整个 batch 共享的每组 partition 预算
    int own_tile,                               // 每个 batch 自有的 partition 线程块数
    int overflow_rows,                          // 溢出区行数 (每行 num_kv_heads * NUM_Q_TILES 个线程块)
    int batch,
    int combine_ctas,                           // 每个 batch 每组的合并线程块数 (0 表示使用合并 kernel)
    int mtp,
    unsigned int* __restrict__ combine_counters)
{
    constexpr int HEAD_SIZE_V = HEAD_SIZE;
    constexpr int BLOCK_SIZE = 64;
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int Q_LDS_PAD = 8;
    extern __shared__ char shared_mem[];
    scalar_t *smem_q = (scalar_t *)shared_mem;
    using scalarx4_t = __attribute__((__vector_size__(4 * sizeof(scalar_t)))) scalar_t;
    using scalarx8_t = __attribute__((__vector_size__(8 * sizeof(scalar_t)))) scalar_t;
    constexpr int Mloop = DIVIDE_ROUND_UP(Q_TILE, 16);
    constexpr int direct_ctas_per_cu = DirectFp8Config<HEAD_SIZE, HEAD_SIZE_V, Mloop>::CTAS_PER_CU;
    // 一维 grid, 线程块角色按线性编号 (派发顺序) 依次为 (G = num_kv_heads * NUM_Q_TILES):
    //  1. 各 batch 自有的 own_tile 个 partition: batch * G * own_tile 个;
    //  2. 溢出区: 长序列超出 own_tile 的 partition, G * overflow_rows 个 (只在 batch > 1 时存在);
    //  3. 同一次启动内的合并线程块: batch * G * combine_ctas 个。
    // 合并线程块在所有 partition 线程块之后派发, 自旋等待时不会挡住其他 batch 的 partition 线程块驻留;
    // 未用到的溢出区线程块排在有效 partition 之后, 不推迟其派发。
    const int G = num_kv_heads * NUM_Q_TILES;
    const int linear_cta = blockIdx.x;
    const int own_ctas = batch * G * own_tile;
    const int overflow_ctas = G * overflow_rows;
    const bool overflow_role = linear_cta >= own_ctas && linear_cta < own_ctas + overflow_ctas;
    int batch_idx = 0, partition_idx = 0, role_x;
    if (linear_cta < own_ctas) {
        batch_idx = linear_cta / (G * own_tile);
        partition_idx = linear_cta % (G * own_tile) / G;
        role_x = linear_cta % G;
    } else if (overflow_role) {
        role_x = (linear_cta - own_ctas) % G;
    } else {
        const int c = linear_cta - own_ctas - overflow_ctas;
        batch_idx = c / (G * combine_ctas);
        partition_idx = balanced_seq_len_tile + c % (G * combine_ctas) / G;
        role_x = c % G;
    }
    // 自有与合并线程块的序列号不依赖访存: 先读取其 seq_lens, 与整个 batch 的 page 总数归约重叠
    int valid_seqlen_kv = overflow_role ? 0 : __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    const DirectBatchBudget budget = direct_batch_budget<LONG_SINGLE_STREAM>(seq_lens, batch, total_tile, valid_seqlen_kv);
    if (overflow_role) {
        if (!direct_overflow_slot<direct_ctas_per_cu>(seq_lens, batch, (linear_cta - own_ctas) / G, own_tile, budget, total_tile,
                                                      balanced_seq_len_tile, batch_idx, partition_idx)) {
            return;
        }
        batch_idx = __builtin_amdgcn_readfirstlane(batch_idx);
        partition_idx = __builtin_amdgcn_readfirstlane(partition_idx);
        valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    }
    const int q_tile_idx = role_x % NUM_Q_TILES;
    const int kv_head_idx = role_x / NUM_Q_TILES;
    const int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    const int seq_tile = direct_tile_of(valid_num_block_pages, budget.total_pages, batch, total_tile, balanced_seq_len_tile);
    const int valid_num_partition = direct_partitions_for_tile<direct_ctas_per_cu>(valid_num_block_pages, seq_tile, budget.halve);
    const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
    const int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
    const int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
    const int tile_start = q_tile_idx * rows_per_qtile;
    const int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);
    if (combine_counters != nullptr && partition_idx >= balanced_seq_len_tile) {
        if (valid_num_partition > 1) {
            const int local_row = partition_idx - balanced_seq_len_tile;
            if (local_row >= tile_rows) return;
            const int row = tile_start + local_row;
            const int head = batch_idx * num_heads + q_heads_per_kv_head * kv_head_idx
                             + (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
            direct_combine_head_in_kernel<scalar_t, HEAD_SIZE_V>(
                combine_counters + ((batch_idx * num_kv_heads + kv_head_idx) * NUM_Q_TILES + q_tile_idx) * DIRECT_COMBINE_COUNTER_STRIDE,
                valid_num_partition, tile_rows, out, out_tmp, head, balanced_seq_len_tile, shared_mem);
        }
        return;
    }
    if (partition_idx >= valid_num_partition) {
        return;
    }
    constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);
    const uint8_t *k_ptr_base = k_cache + static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const uint8_t *v_ptr_base = v_cache + static_cast<int64_t>(kv_head_idx) * v_head_stride;
    const int *block_table = block_tables + static_cast<int64_t>(batch_idx) * max_num_blocks_per_seq;
    constexpr bool q_as_fp16 = !is_query_fp16;
    using qk_t = std::conditional_t<q_as_fp16, _Float16, scalar_t>;
    using qk_x8_t = __attribute__((__vector_size__(8 * sizeof(qk_t)))) qk_t;
    using qk_x4_t = __attribute__((__vector_size__(4 * sizeof(qk_t)))) qk_t;
    float q_scale = 1.0f;
    const bool slice_mode = use_direct_slice_mode<direct_ctas_per_cu>(valid_num_block_pages, seq_tile);

    QStage<scalarx8_t, HEAD_SIZE, NUM_THREADS, Q_TILE> q_stage;
    issue_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE>(
        q_stage, q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, tile_start, tile_rows);
    auto q_commit = [&]() {
        if constexpr (q_as_fp16) {
            const bool amortize_fp16 = Mloop * HEAD_SIZE >= 512 ||
                (!slice_mode && valid_num_block_pages > NUM_WARPS * valid_num_partition);
            if (!amortize_fp16) {
                commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(q_stage, smem_q, tile_rows);
                return -1.0f;
            }
            uint32_t *reduce = reinterpret_cast<uint32_t *>(
                shared_mem + DirectEpilogueShape<Mloop, HEAD_SIZE_V, NUM_WARPS>::meta_offset(Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)));
            q_scale = commit_q_stage_fp16<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(
                q_stage, reinterpret_cast<_Float16 *>(smem_q), tile_rows, reduce);
            const bool q_bf16 = q_scale < 0.0f;
            q_scale = fabsf(q_scale);
            return q_bf16 ? -q_scale : q_scale;
        } else {
            commit_q_stage<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_TILE, Q_LDS_PAD>(q_stage, smem_q, tile_rows);
        }
        return q_scale;
    };
    // MTP 因果掩码: Q 行 row 对应 query token row / q_heads_per_kv_head, 只能看到 token <= valid - mtp + q_token
    auto score_mask = [&](auto &s_reg, const int token_base, const int stride_i, const int stride_g, const int stride_j) {
        constexpr int NT = std::extent<std::remove_reference_t<decltype(s_reg)>, 1>::value;
        const int token_last = token_base + (NT - 1) * stride_i + 3 * stride_g + 3 * stride_j;
        if (token_last + mtp > valid_seqlen_kv) {
            for (int m = 0; m < Mloop; m++) {
                const int local_row = m * 16 + (lane_id & 15);
                const int col = lane_id >> 4;
                if (local_row < tile_rows) {
                    const int global_row = tile_start + local_row;
                    const int q_token_id = global_row / q_heads_per_kv_head;
                    const int casual_offset = mtp - q_token_id;
                    for (int i = 0; i < NT; i++) {
                        for (int j = 0; j < 4; j++) {
                            const int token_k_index = token_base + i * stride_i + col * stride_g + j * stride_j;
                            if (token_k_index < valid_seqlen_kv && token_k_index + casual_offset > valid_seqlen_kv) {
                                (s_reg[m][i])[j] = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    };
    direct_fp8_page_loop<qk_x8_t, qk_x4_t, qk_t, Mloop, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, Q_LDS_PAD, true>(
        k_ptr_base, v_ptr_base, k_block_stride, v_block_stride, block_table, partition_idx, valid_num_partition, warp_id,
        slice_mode, valid_num_block_pages, num_kv_heads, reinterpret_cast<const qk_t *>(smem_q), tile_rows, valid_seqlen_kv,
        scale_softmax_log2, scores_max, scores_sum, acc_o, q_commit, score_mask, k_token_stride, v_token_stride);

    const int head_q_index = q_heads_per_kv_head * kv_head_idx;
    auto out_row_of = [&](const int local_row) {
        const int row = tile_start + local_row;
        return (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
    };
    const bool split = valid_num_partition > 1;
    const int out_head_stride = split ? balanced_seq_len_tile * HEAD_SIZE_V : HEAD_SIZE_V;
    scalar_t *out_ptr_base = split ? out_tmp + out_tmp_offset + static_cast<int64_t>(batch_idx) * num_heads * out_head_stride
                                         + static_cast<int64_t>(head_q_index) * out_head_stride + partition_idx * HEAD_SIZE_V
                                   : out + static_cast<int64_t>(batch_idx) * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
    auto store_o = [&](const int row, const int col, const float4_t sums) {
        const scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]),
                                     from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
        *reinterpret_cast<scalarx4_t *>(out_ptr_base + static_cast<int64_t>(out_row_of(row)) * out_head_stride + col) = out_sums;
    };
    auto store_meta = [&](const int row, const float max_new, const float sum_new) {
        if (split) {
            const int offset = batch_idx * num_heads * balanced_seq_len_tile
                               + (head_q_index + out_row_of(row)) * balanced_seq_len_tile + partition_idx;
            reinterpret_cast<float *>(out_tmp)[offset] = sum_new;
            reinterpret_cast<float *>(out_tmp + max_tmp_offset)[offset] = max_new * (scale_softmax_log2 * q_scale);
        }
    };
    direct_epilogue<Mloop, HEAD_SIZE_V, NUM_WARPS, Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t)>(
        scores_max, scores_sum, acc_o, shared_mem, tile_rows, warp_id, scale_softmax_log2 * q_scale, store_o, store_meta);
    if (split && combine_counters != nullptr) {
        direct_combine_arrive(combine_counters + ((batch_idx * num_kv_heads + kv_head_idx) * NUM_Q_TILES + q_tile_idx)
                              * DIRECT_COMBINE_COUNTER_STRIDE);
    }
}

// ---------------------------------------------------------------------------
// host: Q tile、partition 预算与合并方式和 BHSD 936 direct 路径相同 (FP16 Q 使用同一预算)。
// 返回 false 表示输入不满足 direct 路径的前提, 由调用者回退原实现。
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_SIZE, int Q_TILE, int NUM_Q_TILES>
static void launch_direct(scalar_t *out, scalar_t *out_tmp, const scalar_t *query, const uint8_t *k_cache, const uint8_t *v_cache,
                          int batch, int num_heads, int num_kv_heads, const int *block_tables, const int *seq_lens,
                          int max_num_blocks_per_seq, int q_stride, int k_block_stride, int v_block_stride,
                          int k_token_stride, int v_token_stride, int k_head_stride, int v_head_stride,
                          int balanced_seq_len_tile, int total_tile, int own_tile, int overflow_rows, int mtp, unsigned int *combine_counters, int combine_ctas,
                          bool print_param, cudaStream_t stream)
{
    constexpr int NUM_WARPS = 4;
    constexpr int Q_LDS_PAD = 8;
    constexpr int Mloop = DIVIDE_ROUND_UP(Q_TILE, 16);
    constexpr int CTAS = DirectFp8Config<HEAD_SIZE, HEAD_SIZE, Mloop>::CTAS_PER_CU;
    constexpr bool LSS = NUM_Q_TILES == 1 && CTAS == 2 && Mloop * HEAD_SIZE <= 256;
    using Shape = CombineDirectShape<scalar_t, HEAD_SIZE>;
    const int smem_q_size = Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t);
    int smem_size = DirectEpilogueShape<Mloop, HEAD_SIZE, NUM_WARPS>::smem_bytes(smem_q_size, Q_TILE);
    // 同一次启动内的合并线程块: 归约区 + 标量 + 权重 + 标志
    if (combine_counters != nullptr) smem_size = MAX_T(smem_size, (Shape::RED_FLOATS + 9 + balanced_seq_len_tile) * static_cast<int>(sizeof(float)));
    // 一维 grid: 自有 partition, 溢出区, 合并线程块 (见主 kernel)
    const dim3 grid(num_kv_heads * NUM_Q_TILES * (batch * own_tile + overflow_rows + batch * combine_ctas));
    if (print_param) {
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
        printf("L0 implementation=gfx936 direct E5M2, fp16 [%d], HEAD_SIZE [%d], Q_TILE [%d], NUM_Q_TILES [%d], CTAS_PER_CU [%d], "
#else
        printf("BSHD implementation=gfx936 direct E5M2, fp16 [%d], HEAD_SIZE [%d], Q_TILE [%d], NUM_Q_TILES [%d], CTAS_PER_CU [%d], "
#endif
               "LONG_SINGLE_STREAM [%d], batch [%d], num_heads [%d], num_kv_heads [%d], balanced_seq_len_tile [%d], total_tile [%d], own_tile [%d], overflow_rows [%d], "
               "fused_combine [%d], grid [%d] [%d] [%d], smem [%d]\n",
               (int)std::is_same<scalar_t, _Float16>::value, HEAD_SIZE, Q_TILE, NUM_Q_TILES, CTAS, (int)LSS, batch, num_heads,
               num_kv_heads, balanced_seq_len_tile, total_tile, own_tile, overflow_rows, (int)(combine_counters != nullptr), grid.x, grid.y, grid.z, smem_size);
    }
    PA_E5M2_936_KERNEL<scalar_t, HEAD_SIZE, Q_TILE, NUM_Q_TILES, LSS><<<grid, dim3(256), smem_size, stream>>>(
        out, out_tmp, query, k_cache, v_cache, num_heads, num_kv_heads, block_tables, seq_lens, max_num_blocks_per_seq,
        q_stride, k_block_stride, v_block_stride, k_token_stride, v_token_stride, k_head_stride, v_head_stride,
        balanced_seq_len_tile, total_tile, own_tile, overflow_rows, batch, combine_ctas, mtp, combine_counters);
    if (balanced_seq_len_tile > 1 && combine_counters == nullptr) {
        const int combine_smem = (Shape::RED_FLOATS + 8 + balanced_seq_len_tile) * sizeof(float);
        paged_attention_combine_direct<scalar_t, HEAD_SIZE, 64, CTAS, LSS, false, false>
            <<<dim3(num_heads, batch), dim3(256), combine_smem, stream>>>(out, out_tmp, seq_lens, balanced_seq_len_tile, total_tile, num_heads);
    }
}

static bool launch(torch::Tensor &out, torch::Tensor &query_in, torch::Tensor &key_cache, torch::Tensor &value_cache,
                   torch::Tensor &block_tables, torch::Tensor &seq_lens, const c10::optional<torch::Tensor> &alibi_slopes,
                   const c10::optional<torch::Tensor> &q_scale, const c10::optional<torch::Tensor> &k_scale,
                   const c10::optional<torch::Tensor> &v_scale, const c10::optional<at::Tensor> &s_aux,
                   float *out_tmp, bool print_param)
{
    if (query_in.dim() != 4 || key_cache.dim() != 4 || value_cache.dim() != 4 || out.dim() < 3) return false;
    const auto o_type = out.scalar_type();
    const int D = static_cast<int>(query_in.size(3));
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    const int num_kv_heads = static_cast<int>(key_cache.size(1));
    const bool types = (o_type == torch::kBFloat16 || o_type == torch::kFloat16) && query_in.scalar_type() == o_type &&
        key_cache.scalar_type() == torch::kFloat8_e5m2 && value_cache.scalar_type() == torch::kFloat8_e5m2;
    // K [num_blocks, num_kv_heads, 64, D], V [num_blocks, num_kv_heads, D, 64]
    const bool shapes = (D == 64 || D == 128 || D == 192 || D == 256) && key_cache.size(3) == D && key_cache.size(2) == 64 &&
        value_cache.size(0) == key_cache.size(0) && value_cache.size(1) == num_kv_heads &&
        value_cache.size(2) == D && value_cache.size(3) == 64 && num_kv_heads > 0 &&
#else
    const int num_kv_heads = static_cast<int>(key_cache.size(2));
    const bool types = (o_type == torch::kBFloat16 || o_type == torch::kFloat16) && query_in.scalar_type() == o_type &&
        key_cache.scalar_type() == torch::kFloat8_e5m2 && value_cache.scalar_type() == torch::kFloat8_e5m2;
    const bool shapes = (D == 64 || D == 128 || D == 192 || D == 256) && key_cache.size(3) == D && value_cache.size(3) == D &&
        key_cache.size(1) == 64 && key_cache.sizes() == value_cache.sizes() && num_kv_heads > 0 &&
#endif
        query_in.size(2) % num_kv_heads == 0 && query_in.size(1) >= 1 &&
        query_in.size(1) * query_in.size(2) <= static_cast<int64_t>(num_kv_heads) * 64 &&
        out.numel() == query_in.numel() && out.is_contiguous();
    const bool features = !alibi_slopes.has_value() && !s_aux.has_value() &&
        !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value();
    // 16 字节 buffer load: K/V 的 token/head/block 跨度与基址按 16 字节对齐, 最低维紧密
    auto aligned16 = [](const torch::Tensor &t) {
        return t.stride(3) == 1 && t.stride(2) % 16 == 0 && t.stride(1) % 16 == 0 && t.stride(0) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
            t.stride(0) <= std::numeric_limits<int>::max() && t.stride(1) <= std::numeric_limits<int>::max() &&
            t.stride(2) <= std::numeric_limits<int>::max() / 256;
#else
            t.stride(0) <= std::numeric_limits<int>::max() && t.stride(1) <= std::numeric_limits<int>::max() / 64;
#endif
    };
    if (!types || !shapes || !features || !aligned16(key_cache) || !aligned16(value_cache) ||
        block_tables.scalar_type() != torch::kInt || seq_lens.scalar_type() != torch::kInt) {
        return false;
    }
    // Q 在 token/head/dim 维不紧密时 (例如 QKV 融合张量的切片视图) 先复制为紧密布局 (batch 维可以有跨度)
    torch::Tensor query = query_in;
    auto dense = [](const torch::Tensor &t, int dim, int64_t stride) { return t.size(dim) == 1 || t.stride(dim) == stride; };
    if (!(query.stride(3) == 1 && dense(query, 2, D) && dense(query, 1, query.size(2) * D) &&
          reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0 && query.stride(0) % 8 == 0)) {
        query = query.contiguous();
    }
    const int batch = static_cast<int>(query.size(0));
    const int mtp = static_cast<int>(query.size(1));
    const int num_q_heads = static_cast<int>(query.size(2));
    const int num_heads = num_q_heads * mtp;
    const int rows = num_heads / num_kv_heads;
    const int reuse_kv = get_reusekv(num_heads, num_kv_heads);
    const bool bf16 = o_type == torch::kBFloat16;
    // Q tile 数与 BHSD 936 相同: 每组 > 32 行两个 tile; BF16 Q 的 acc_o 占 128 VGPR 以上 (每组 Mloop * D >= 512) 且行数为偶数时两个 tile
    const int bucket_mloop = DIVIDE_ROUND_UP(reuse_kv, 16);
    const int num_q_tiles = (reuse_kv > 32 || (bf16 && bucket_mloop * D >= 512 && rows % 2 == 0)) ? 2 : 1;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const int device = query.get_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static thread_local std::unordered_map<int, int> cus_by_device;
    auto cu = cus_by_device.find(device);
    if (cu == cus_by_device.end()) {
        hipDeviceProp_t prop;
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
        TORCH_CHECK(hipGetDeviceProperties(&prop, device) == hipSuccess, "layout=0 gfx936: device-property query failed");
#else
        TORCH_CHECK(hipGetDeviceProperties(&prop, device) == hipSuccess, "BSHD gfx936: device-property query failed");
#endif
        cu = cus_by_device.emplace(device, prop.multiProcessorCount).first;
    }
    const int num_cus = MAX_T(1, cu->second);
    const int base_blocks = batch * num_kv_heads;
    const int mloop = DIVIDE_ROUND_UP(DIVIDE_ROUND_UP(reuse_kv, num_q_tiles), 16);
    const int ctas_per_cu = MIN_T(2, MAX_T(1, mloop * D / 128));
    // 整个 batch 共享的每组 partition 预算 (设备端按各序列 page 数占比分配, 见 direct_batch_budget) 与每个 batch 自有的预算
    const int total_tile = MAX_T(1, ctas_per_cu * num_cus / (num_kv_heads * num_q_tiles));
    const int own_tile = MAX_T(1, ctas_per_cu * num_cus / (base_blocks * num_q_tiles));
    // partial 输出与 max/sum 必须放进调用方的 workspace (paged_attention.cu 至少分配 210 MB)
    auto fits = [&](int tile) {
        const int64_t partial_rows = static_cast<int64_t>(batch) * num_heads * tile;
        return partial_rows * static_cast<int64_t>(sizeof(float)) <= static_cast<int64_t>(max_tmp_offset) * sizeof(uint16_t) &&
            static_cast<int64_t>(out_tmp_offset) * sizeof(uint16_t) + partial_rows * D * static_cast<int64_t>(sizeof(uint16_t)) <= 210000000;
    };
    // 每个序列 partition 数的上限 (partial 输出的步长): batch > 1 时为 total_tile, 放不进 workspace 时逐步收回到 own_tile
    int balanced_seq_len_tile = batch > 1 ? MAX_T(own_tile, total_tile) : own_tile;
    while (balanced_seq_len_tile > own_tile && !fits(balanced_seq_len_tile)) balanced_seq_len_tile = MAX_T(own_tile, balanced_seq_len_tile / 2);
    if (!fits(balanced_seq_len_tile)) {
        return false;
    }
    // 溢出区行数: 溢出 partition 总数的上限 (见 direct_overflow_slot); 每个序列最多 block_tables.size(1) 个 page
    const int max_seq_pages = static_cast<int>(block_tables.size(1));
    const int overflow_rows = batch > 1 ?
        static_cast<int>(MIN_T(static_cast<int64_t>(total_tile - own_tile),
                               static_cast<int64_t>(batch) * MAX_T(0, MIN_T(balanced_seq_len_tile, max_seq_pages) - own_tile))) : 0;
    unsigned int *combine_counters = nullptr;
    if (balanced_seq_len_tile > 1 && base_blocks * num_q_tiles <= DIRECT_COMBINE_MAX_GROUPS) {
        combine_counters = direct_combine_counters(device, stream);
    }
    const int combine_ctas = combine_counters != nullptr ? DIVIDE_ROUND_UP(rows, num_q_tiles) : 0;
    const int q_stride = static_cast<int>(query.stride(0));
    const int k_block_stride = static_cast<int>(key_cache.stride(0)), v_block_stride = static_cast<int>(value_cache.stride(0));
#if PA_E5M2_936_LAYOUT == PA_LAYOUT_L0
    // K: token 跨度 stride(2), head 跨度 stride(1); V^T: 行 (通道) 跨度 stride(2), head 跨度 stride(1)
    const int k_token_stride = static_cast<int>(key_cache.stride(2)), v_token_stride = static_cast<int>(value_cache.stride(2));
    const int k_head_stride = static_cast<int>(key_cache.stride(1)), v_head_stride = static_cast<int>(value_cache.stride(1));
#else
    const int k_token_stride = static_cast<int>(key_cache.stride(1)), v_token_stride = static_cast<int>(value_cache.stride(1));
    const int k_head_stride = static_cast<int>(key_cache.stride(2)), v_head_stride = static_cast<int>(value_cache.stride(2));
#endif
    const int max_num_blocks_per_seq = static_cast<int>(block_tables.size(1));
    auto run = [&](auto out_tag) {
        using scalar_t = decltype(out_tag);
        HEADSIZE_SWITCH(D, [&] {
            REUSEKV_SWITCH(reuse_kv, [&] {
                auto go = [&](auto tiles) {
                    constexpr int NUM_Q_TILES = decltype(tiles)::value;
                    constexpr int Q_TILE = DIVIDE_ROUND_UP(REUSE_KV_TIMES, NUM_Q_TILES);
                    if constexpr (HEAD_SIZE == 64 || HEAD_SIZE == 128 || HEAD_SIZE == 192 || HEAD_SIZE == 256) {
                        launch_direct<scalar_t, HEAD_SIZE, Q_TILE, NUM_Q_TILES>(
                            reinterpret_cast<scalar_t *>(out.data_ptr()), reinterpret_cast<scalar_t *>(out_tmp),
                            reinterpret_cast<const scalar_t *>(query.data_ptr()),
                            reinterpret_cast<const uint8_t *>(key_cache.data_ptr()), reinterpret_cast<const uint8_t *>(value_cache.data_ptr()),
                            batch, num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
                            max_num_blocks_per_seq, q_stride, k_block_stride, v_block_stride, k_token_stride, v_token_stride,
                            k_head_stride, v_head_stride, balanced_seq_len_tile, total_tile, own_tile, overflow_rows, mtp, combine_counters, combine_ctas, print_param, stream);
                    }
                };
                // 每组 <= 32 行的单 tile 与每组 > 32 行的两个 tile 分别实例化; 每组 24/32 行时两种都需要
                if constexpr (REUSE_KV_TIMES > 32) {
                    go(std::integral_constant<int, 2>{});
                } else if constexpr (REUSE_KV_TIMES > 16) {
                    if (num_q_tiles == 2) go(std::integral_constant<int, 2>{});
                    else go(std::integral_constant<int, 1>{});
                } else {
                    go(std::integral_constant<int, 1>{});
                }
            });
        });
    };
    if (bf16) run(uint16_t{}); else run(_Float16{});
    return true;
}


#undef PA_E5M2_936_LAYOUT
#undef PA_E5M2_936_KERNEL
