// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
// Paged GQA family: storage policies, shared kernels and host orchestration.
// Public entries live in the two precision CUs. Shared merge definitions live in the main CU.
// Reading order: operands/storage -> per-call preparation -> attention -> host dispatch.
#include "sparse_attention.h"
#include "hytlass/fast_math.h"
#include <type_traits>

namespace sparse_gqa_paged_detail {

using sparse_attention_detail::mmac::assemble_inlane_value_operand;

// Keep BF16 storage on its original conversion path. gfx938 FP8 uses the
// measured packed RNE conversion for probabilities and vectorized output.
template<bool Fp8>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 paged_probability(
    sparse_attention_detail::mmac::Float4 values) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    if constexpr (Fp8) {
        using F2 = __attribute__((ext_vector_type(2))) float;
        using B2 = __attribute__((ext_vector_type(2))) __bf16;
        // Match c10 BF16 RNE, including its sign-independent canonical NaN.
        // Native conversion otherwise preserves a NaN payload/sign.
        const F2 a = {__builtin_isnan(values[0]) ? __builtin_nanf("") : values[0],
                      __builtin_isnan(values[1]) ? __builtin_nanf("") : values[1]};
        const F2 b = {__builtin_isnan(values[2]) ? __builtin_nanf("") : values[2],
                      __builtin_isnan(values[3]) ? __builtin_nanf("") : values[3]};
        const B2 lo = __builtin_convertvector(a, B2);
        const B2 hi = __builtin_convertvector(b, B2);
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, &lo, 4);
        __builtin_memcpy(reinterpret_cast<char*>(&result) + 4, &hi, 4);
        return result;
    }
#endif
    return sparse_attention_detail::mmac::bf16_prob(values);
}

constexpr int kGroup = 12;

// Every nonnegative INT32 token lies in page zero when page_size > INT_MAX.
// Keep this exact fallback without narrowing a caller's INT64 page size.
struct LargePageDivmod {
    __device__ __forceinline__ int divmod(int &offset, int token) const {
        offset = token;
        return 0;
    }
};

// ---- Shared cross-TU merge declarations; definitions exist once in the main CU ----
void launch_merge(const float *partial, const float *lse, at::BFloat16 *out, int64_t rows,
                  int64_t heads, int splits, int64_t os0, int64_t os1, hipStream_t stream);

void launch_merge_tiny32(const float *partial, const float *lse, at::BFloat16 *out,
                          int64_t rows, int64_t heads, int64_t os0, int64_t os1,
                          hipStream_t stream);

// ---- Storage policies: BF16 access is also reused by mixed FP8 preparation ----
struct Bf16Cache {
    using Element = at::BFloat16;
    using KElement = Element;
    using VElement = Element;
    using KFragment = sparse_attention_detail::mmac::Vec8;
    using VFragment = sparse_attention_detail::mmac::Half4;
    static constexpr bool fp8 = false;
    static constexpr bool packed_nan = false;
    static constexpr bool shared_offsets = false;
    static constexpr bool e5 = false;
    static constexpr bool mixed_predecoded = false;
    __device__ static __forceinline__ KFragment load_k(const Element *p) {
        return sparse_attention_detail::mmac::load8(p);
    }
    __device__ static __forceinline__ VFragment load_v(const Element *p) {
        return *reinterpret_cast<const VFragment *>(p);
    }
    __device__ static __forceinline__ KFragment shuffle_k(KFragment k, int lane) {
        return sparse_attention_detail::mmac::shuffle8(k, lane);
    }
    __device__ static __forceinline__ VFragment transpose_v(VFragment v, int g, int r) {
        return sparse_attention_detail::mmac::transpose_v_dpp(v, g, r);
    }
    __device__ static __forceinline__ VFragment lane_v(const VFragment *values, int d) {
        const int chunk = d / 4;
        return assemble_inlane_value_operand(values[chunk], values[4 + chunk],
                                  values[8 + chunk], values[12 + chunk], d % 4);
    }
    __device__ static __forceinline__ float decode(Element v) { return float(v); }
};

// Private mixed storage: K keeps E5 bytes with NaNs canonicalized once to
// 0x7e; finite/Inf/signed-zero values are unchanged. V alone widens to BF16.
// This is still BF16 Q/P MMAC with FP32 accumulation, not FP8 MMAC.
// Both compact owners have identical *element*
// strides, so a shared numeric offset remains valid for both typed pointers.
// Semantic K/V scales remain at their original score/output sites.
template<class SourceCache>
struct MixedPagedCache : SourceCache {
    using Element = typename SourceCache::Element; // Public alias remains K.
    using KElement = typename SourceCache::KElement;
    using VElement = at::BFloat16;
    using VFragment = Bf16Cache::VFragment;
    static constexpr bool mixed_predecoded = true;
    static constexpr bool shared_offsets = true;
    // The private prepass canonicalizes every E5 NaN byte once. Avoid
    // repeating that four-byte classification on every attention tile.
    __device__ static __forceinline__ sparse_attention_detail::mmac::Vec8 shuffle_k(
        typename SourceCache::KFragment key, int lane) {
        return {SourceCache::decode4_canonical(__shfl(key[0], lane, 64)),
                SourceCache::decode4_canonical(__shfl(key[1], lane, 64))};
    }
    __device__ static __forceinline__ VFragment load_v(const VElement *p) {
        return Bf16Cache::load_v(p);
    }
    __device__ static __forceinline__ VFragment transpose_v(VFragment value, int g, int r) {
        return Bf16Cache::transpose_v(value, g, r);
    }
    __device__ static __forceinline__ VFragment lane_v(const VFragment *values, int d) {
        return Bf16Cache::lane_v(values, d);
    }
    using SourceCache::decode;
    __device__ static __forceinline__ float decode(VElement value) { return float(value); }
};

using sparse_attention_detail::CacheStrides;
constexpr int kThreads = 64;
constexpr int kOutputTile = 256;
constexpr int kSharedStride = 260;
constexpr float kLog2E = 1.4426950408889634f;

// ---- Private FP8 preparation: metadata and KV stay on the current stream ----
// Canonicalize once while preparing private E5 K bytes; all other encodings
// remain bit-identical. The private decoder can then omit repeated NaN tests.
__device__ __forceinline__ uint32_t paged_canonical_e5_word(uint32_t bits) {
    const uint32_t nan_high = ((bits & 0x7f7f7f7fu) + 0x03030303u) & 0x80808080u;
    const uint32_t nan_mask = (nan_high >> 7) * 0xffu;
    return (bits & ~nan_mask) | (0x7e7e7e7eu & nan_mask);
}

// Canonicalize K NaNs once and exactly widen V for each logical page, independently
// of the potentially huge, mixed-layer physical cache backing. Repeated table
// pages may be duplicated; invalid source pages are never dereferenced.
// One block page axis and one wave per token/head remove per-element INT64
// divisions. Invalid pages only write their -1 table entry: attention masks
// that page, so zero-filling its unused temporary storage is unnecessary.
template<class Cache>
__global__ void sparse_gqa_paged_attention_predecode_kernel(
    const typename Cache::KElement *k, const typename Cache::VElement *v,
    const int32_t *table, uint8_t *compact_k, at::BFloat16 *decoded_v,
    int32_t *decoded_table, int token_heads, int page_size, int kv_heads,
    int64_t source_pages, int64_t ts0, CacheStrides ks, CacheStrides vs,
    hytlass::FastDivmod table_divmod, hytlass::FastDivmod head_divmod,
    const int32_t *active_limits) {
    const int logical_page = blockIdx.y;
    int table_column = 0;
    const int request = table_divmod.divmod(table_column, logical_page);
    // All threads in the block read the same entry and take the same page
    // validity branch. Table initialization is performed by exactly one lane.
    // Prefix bound is rebuilt from this call's valid selected indices.
    // Still launch every table slot so inactive entries are rewritten to -1.
    const bool active = !active_limits || table_column <= active_limits[request];
    const int32_t source_page = active
        ? table[int64_t(request) * ts0 + table_column] : -1;
    const bool valid = source_page >= 0 && int64_t(source_page) < source_pages;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        decoded_table[logical_page] = valid ? int32_t(logical_page) : -1;
    if (!valid) return;
    const int token_head = int(blockIdx.x) * 4 + int(threadIdx.x) / 64;
    if (token_head >= token_heads) return;
    int head = 0;
    const int token = head_divmod.divmod(head, token_head);
    const int column = (int(threadIdx.x) & 63) * 4;
    const int64_t k_base = int64_t(source_page) * ks.block + int64_t(token) * ks.token +
                           int64_t(head) * ks.head;
    const int64_t v_base = int64_t(source_page) * vs.block + int64_t(token) * vs.token +
                           int64_t(head) * vs.head;
    uint32_t kb, vb;
    __builtin_memcpy(&kb, k + k_base + column, sizeof(kb));
    __builtin_memcpy(&vb, v + v_base + column, sizeof(vb));
    // Preserve every finite/Inf/zero byte; map only E5 NaNs to positive
    // canonical 0x7e, exactly matching the direct K decoder's old behavior.
    kb = paged_canonical_e5_word(kb);
    const auto value = Cache::decode4(vb);
    // Two owning tensors avoid dtype-reinterpreted or non-owning views.  They
    // share head-major logical element coordinates, not storage or byte stride.
    const int64_t destination = ((int64_t(logical_page) * kv_heads + head) * page_size +
                                 token) * 256 + column;
    __builtin_memcpy(compact_k + destination, &kb, sizeof(kb));
    __builtin_memcpy(decoded_v + destination, &value, sizeof(value));
}

// Persistent prefix preparation: one request axis plus shape-only LP slices.
// Only this call's selected valid logical prefix is copied; K/V owners still
// reserve the full original table-shaped workspace (no fixed byte budget).
// The decoded table is cleared to -1 before launch on every call/replay.
template<class Cache, bool Routed = false>
__global__ void sparse_gqa_paged_attention_predecode_prefix_kernel(
    const typename Cache::KElement *k, const typename Cache::VElement *v,
    const int32_t *table, uint8_t *compact_k, at::BFloat16 *decoded_v,
    int32_t *decoded_table, int token_heads, int page_size, int kv_heads,
    int64_t source_pages, int64_t table_width, int64_t ts0,
    CacheStrides ks, CacheStrides vs, hytlass::FastDivmod head_divmod,
    const int32_t *active_limits, const int32_t *mixed_route = nullptr) {
    if constexpr (Routed) {
        if (!*mixed_route) return;
    }
    const int64_t request = blockIdx.y;
    const int token_head = int(blockIdx.x) * 4 + int(threadIdx.x) / 64;
    if (token_head >= token_heads) return;
    int head = 0;
    const int token = head_divmod.divmod(head, token_head);
    const int column = (int(threadIdx.x) & 63) * 4;
    const int64_t destination_within =
        (int64_t(head) * page_size + token) * 256 + column;
    const int limit = active_limits[request];
    // Shape-only LP partitions keep small-page calls from becoming a handful
    // of long-lived CTAs. Each logical page belongs to exactly one z slice.
    // INT64 increment also protects a last column near INT32_MAX.
    for (int64_t logical_column = blockIdx.z; logical_column <= int64_t(limit);
         logical_column += int64_t(gridDim.z)) {
        const int32_t source_page = table[request * ts0 + logical_column];
        if (source_page < 0 || int64_t(source_page) >= source_pages)
            continue;
        const int64_t logical_page = request * table_width + logical_column;
        if (blockIdx.x == 0 && threadIdx.x == 0)
            decoded_table[logical_page] = int32_t(logical_page);
        const int64_t k_base =
            int64_t(source_page) * ks.block + int64_t(token) * ks.token +
            int64_t(head) * ks.head + column;
        const int64_t v_base =
            int64_t(source_page) * vs.block + int64_t(token) * vs.token +
            int64_t(head) * vs.head + column;
        uint32_t kb, vb;
        __builtin_memcpy(&kb, k + k_base, sizeof(kb));
        __builtin_memcpy(&vb, v + v_base, sizeof(vb));
        kb = paged_canonical_e5_word(kb);
        const auto value = Cache::decode4(vb);
        const int64_t destination =
            logical_page * int64_t(kv_heads) * page_size * 256 + destination_within;
        __builtin_memcpy(compact_k + destination, &kb, sizeof(kb));
        __builtin_memcpy(decoded_v + destination, &value, sizeof(value));
    }
}

// Choose from the *current selected prefix*, not logical capacity. No host
// metadata read or cross-call cache: this scalar is rebuilt each graph replay.
// The full owners are still reserved; only their active decoded-V footprint
// decides whether this call prepares/reads Mixed or the original FP8 layout.
template<int Unused = 0>
__global__ void sparse_gqa_paged_attention_choose_route_kernel(
    const int32_t *active_limits, int32_t *mixed_route, int64_t requests,
    int64_t page_size, int64_t kv_heads) {
    // Divide the fixed *policy* budget before multiplying any dimensions.
    // The full workspace remains unchanged; this is not an allocation limit.
    // D==256 and positive page_size/Hkv are proved by host fast validation.
    const uint32_t budget_pages = uint32_t(
        (uint64_t(24) * 1024u * 1024u) / uint64_t(page_size) /
        uint64_t(kv_heads) / 256u / 2u);
    const uint32_t rejected = budget_pages + 1;
    uint32_t pages = 0;
    for (int64_t request = threadIdx.x; request < requests; request += 64) {
        const int32_t limit = active_limits[request];
        if (limit >= 0)
            pages = min(rejected, pages + uint32_t(limit) + 1);
    }
#pragma unroll
    for (int delta = 32; delta > 0; delta /= 2) {
        pages = min(rejected, pages + __shfl_xor(pages, delta, 64));
    }
    if (threadIdx.x == 0) {
        // Saturation also handles enormous INT32 limits without overflowing
        // sums; rejected<=49153, and its sum with any positive INT32 limit
        // or another rejected counter stays within UINT32.
        *mixed_route = pages <= budget_pages ? 1 : 0;
    }
}

// Explicit private initialization keeps both metadata resets in the measured
// paged GPU sequence and avoids separate generic Tensor fill launches.
template<int Unused = 0>
__global__ void sparse_gqa_paged_attention_initialize_kernel(
    int32_t *decoded_table, int32_t *active_limits,
    int64_t table_entries, int64_t requests) {
    for (int64_t slot = int64_t(blockIdx.x) * 256 + threadIdx.x;
         slot < table_entries; slot += int64_t(gridDim.x) * 256) {
        decoded_table[slot] = -1;
        if (slot < requests)
            active_limits[slot] = -1;
    }
}

// Per-row work bound; unlike expanded metadata, this writes only M ints.
// Original index order and tile partition are retained for all live work.
template<int Unused = 0>
__global__ void sparse_gqa_paged_attention_last_tile_kernel(
    const int32_t *indices, const int32_t *table, const int32_t *requests,
    int32_t *row_tiles, int32_t *active_limits, int64_t topk, int64_t is0, int64_t ts0,
    int64_t num_requests, int64_t table_width, int64_t source_pages,
    hytlass::FastDivmod page_lookup) {
    const int64_t row = blockIdx.x;
    const int32_t request = requests[row];
    int local_last = -1, local_page = -1;
    if (request >= 0 && int64_t(request) < num_requests) {
        for (int64_t slot = threadIdx.x; slot < topk; slot += 64) {
            const int32_t token = indices[row * is0 + slot];
            if (token >= 0) {
                int offset = 0;
                const int logical_page = page_lookup.divmod(offset, token);
                if (int64_t(logical_page) < table_width) {
                    const int32_t physical_page =
                        table[int64_t(request) * ts0 + logical_page];
                    if (physical_page >= 0 && int64_t(physical_page) < source_pages) {
                        local_last = int(slot);
                        local_page = max(local_page, logical_page);
                    }
                }
            }
        }
    }
    // All 64 threads reconverge here, including threads with no input slot.
    // Exact wave max replaces shared arrays and eight block barriers.
#pragma unroll
    for (int delta = 32; delta > 0; delta /= 2) {
        local_last = max(local_last, __shfl_xor(local_last, delta, 64));
        local_page = max(local_page, __shfl_xor(local_page, delta, 64));
    }
    if (threadIdx.x == 0) {
        row_tiles[row] = int32_t((int64_t(local_last) + 16) / 16);
        if (request >= 0 && int64_t(request) < num_requests && local_page >= 0)
            atomicMax(active_limits + request, local_page);
    }
}

template<class Cache, bool Masked>
__device__ __forceinline__ void paged_load_lane_values(
    const typename Cache::VElement *v, int page, int offset, int g, int r,
    int64_t kv_head, CacheStrides strides, typename Cache::VFragment *values) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int token_page = __shfl(page, g + 4 * j, 64);
        const int token_offset = __shfl(offset, g + 4 * j, 64);
        if (!Masked || token_page >= 0) {
            const int64_t base = int64_t(token_page) * strides.block +
                                int64_t(token_offset) * strides.token + kv_head * strides.head;
            __builtin_memcpy(values + 4 * j, v + base + 16 * r,
                             4 * sizeof(typename Cache::VFragment));
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) values[4 * j + i] = typename Cache::VFragment{};
        }
    }
}

template<class Cache, bool Masked>
__device__ __forceinline__ void paged_load_lane_bases(
    const typename Cache::VElement *v, const int64_t *bases, int r,
    typename Cache::VFragment *values) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (!Masked || bases[j] >= 0) {
            __builtin_memcpy(values + 4 * j, v + bases[j] + 16 * r,
                             4 * sizeof(typename Cache::VFragment));
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) values[4 * j + i] = typename Cache::VFragment{};
        }
    }
}

// One wave needs no LDS merge; gfx936 BF16/small-FP8 use four-wave token partitioning.
// Reuse each K/V tile across Group query heads in the same 16-row MMAC tile.
// FP8 and split kernels take the tuned metadata path; Ahead adds lookahead to
// the gfx936 group-12 BF16 unsplit kernel on small grids.
// ---- Shared attention kernels ----
template <class Cache, int Waves, bool Split = false, bool Ahead = false,
          bool Grid3D = false, bool TileRows = false, class PageLookup = hytlass::FastDivmod,
          int RowsPerCTA = 1, int Routing = 0>
__global__ __launch_bounds__(Waves * 64 * RowsPerCTA) void sparse_gqa_paged_attention_mmac_kernel(
    const at::BFloat16 *q, const typename Cache::KElement *k, const typename Cache::VElement *v,
    const int32_t *indices, const int32_t *table, const int32_t *requests, at::BFloat16 *out,
    float *partial, float *lse, int64_t rows, int64_t heads, int64_t kv_heads, int64_t pages,
    int64_t table_width, int64_t num_requests, int64_t topk, int64_t qs0,
    int64_t qs1, CacheStrides ks, CacheStrides vs, int64_t is0, int64_t ts0, int64_t os0,
    int64_t os1, int splits, const float *k_scale_ptr, const float *v_scale_ptr,
    PageLookup page_lookup, const int32_t *row_tile_counts,
    const int32_t *mixed_route = nullptr) {
    static_assert(Routing == 0 || (Waves == 1 && !Split && Grid3D && !TileRows),
                  "dynamic route requires the unsplit one-wave 3D path");
    // Route 0 is the original kernel, without even a scalar metadata load.
    // Route 1 owns Mixed output, route 2 owns direct output, never both.
    if constexpr (Routing != 0) {
        const bool use_mixed = *mixed_route != 0;
        if (use_mixed != (Routing == 1)) return;
    }
    using namespace sparse_attention_detail::mmac;
    constexpr int Group = kGroup;
    // Keep the measured storage/architecture choices local to this kernel.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool InLaneV = true;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool InLaneV = true;
#else
    constexpr bool InLaneV = false;
#endif
    // gfx938 FP8 K: one 16-byte load covers two MMAC k-steps per lane; Q uses
    // the same dimension permutation so every product pairs matching dims.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool WideK = Cache::fp8 && sizeof(typename Cache::KElement) == 1;
#else
    constexpr bool WideK = false;
#endif
    using Words4 = __attribute__((vector_size(16))) uint32_t;
    const float k_scale = Cache::fp8 ? *k_scale_ptr : 1.0f;
    constexpr bool is_split = Split;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    // W1 target split uses LateV; W4 tiny grids retain EarlyV.
    constexpr bool load_v_first = (Cache::packed_nan && Waves == 1)
        ? false : is_split || !Cache::fp8;
#else
    const bool load_v_first = is_split;
#endif
    // Several independent query waves may share a CTA. Waves remains the
    // number collaborating on ONE query; it stays one for this experiment.
    const int lane = threadIdx.x % 64;
    const int wave = RowsPerCTA > 1 ? 0 : threadIdx.x / 64;
    const int r = lane % 16, g = lane / 16;
    static_assert(!TileRows || Grid3D, "row tiling requires the 3D grid");
    static_assert(RowsPerCTA == 1 || (Waves == 1 && !Split && Grid3D && !TileRows),
                  "independent query CTAs require unsplit one-wave 3D mapping");
    const int64_t row = RowsPerCTA > 1
        ? int64_t(blockIdx.x) * RowsPerCTA + threadIdx.x / 64
        : (TileRows ? int64_t(blockIdx.y) * 8 + (blockIdx.x & 7)
                    : (Grid3D ? blockIdx.x : blockIdx.x % rows));
    // No block barrier/merge exists in the RowsPerCTA>1 instantiation. Tail
    // waves may return without preventing any live wave from making progress.
    if constexpr (RowsPerCTA > 1) {
        if (row >= rows) return;
    }
    const int64_t kv_head = TileRows ? blockIdx.x >> 3
                                    : (Grid3D ? blockIdx.y : (blockIdx.x / rows) % kv_heads);
    const int split = Grid3D ? blockIdx.z : blockIdx.x / rows / kv_heads;
    const int64_t request = requests[row];
    const bool request_valid = request >= 0 && request < num_requests;
    const int64_t first_head = kv_head * Group;
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    Vec8 query[8];
    Float4 pv[16];
    bool query_loaded = false;
    float row_max = -1.0e20f, row_sum = 0.0f;
    int64_t tiles = (topk + 15) / 16;
    if constexpr (Cache::mixed_predecoded && Waves == 1 && !Split) {
        if (row_tile_counts)
            tiles = row_tile_counts[row];
    }
    const int shift = __ffs(splits * Waves) - 1;
    const int64_t first = ((split * Waves + wave) * tiles) >> shift;
    const int64_t last = ((split * Waves + wave + 1) * tiles) >> shift;
    // gfx938: FP8 and BF16 split kernels only; its BF16 unsplit (prefill)
    // kernel measured slower with these and keeps the per-tile form.
    // gfx936 group 12: every kernel. Lookahead costs its BF16 unsplit kernel
    // above eight CTAs per CU, so the host enables it there through Ahead.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool Tuned = Cache::fp8 || Split;
    constexpr bool Lookahead = Tuned;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool Tuned = true;
    constexpr bool Lookahead = Tuned && (Cache::fp8 || Split || Ahead);
#else
    constexpr bool Tuned = false;
    constexpr bool Lookahead = false;
#endif
    // Every consumer reads page/offset from lanes 0..15, which already hold
    // tokens 0..15, so the per-tile broadcast is skipped.
    constexpr bool DirectMeta = Tuned;
    // Lookahead keeps index(t+2) and table(t+1) loads in flight behind tile t's K/V.
    // Lanes 0..15 form each token's 64-bit K/V element offsets once (-1 when
    // masked); consumers shuffle those instead of redoing page/offset math.
    constexpr bool BaseMeta = Tuned && InLaneV;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    // The remaining four MMAC head rows never contribute to an output.
    // Wave reductions preserve r, so their local score/P values need no mask.
    constexpr bool UnmaskedHeads = Cache::packed_nan;
#else
    constexpr bool UnmaskedHeads = false;
#endif
    constexpr int KWindow = UnmaskedHeads ? 8 : 4;
    // The permuted WideK Q layout is only produced by the up-front load.
    static_assert(!WideK || Lookahead, "WideK requires the up-front Q load");
    auto fetch_token = [&](int64_t t) {
        return g == 0 && t < last && t * 16 + r < topk && request_valid
                   ? indices[row * is0 + t * 16 + r] : -1;
    };
    // Returns the raw table entry; the consuming tile range-checks it so the
    // load is not waited on (vmcnt is in order) in the tile that issues it.
    auto lookup_page = [&](int token, int &offset) {
        int page = -1;
        offset = 0;
        if (token >= 0) {
            const int logical_page = page_lookup.divmod(offset, token);
            if (logical_page < table_width)
                page = table[request * ts0 + logical_page];
        }
        return page;
    };
    auto load_query = [&] {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            query[i] =
                r < Group ? load8(q + row * qs0 + (first_head + r) * qs1 +
                                  (WideK ? (i / 2) * 64 + g * 16 + (i % 2) * 8 : i * 32 + g * 8))
                          : zero;
#pragma unroll
        for (int d = 0; d < 16; ++d)
            pv[d] = {0, 0, 0, 0};
        query_loaded = true;
    };
    int ahead_token = -1, ahead_page = -1, ahead_offset = 0;
    if constexpr (Lookahead) {
        ahead_page = lookup_page(fetch_token(first), ahead_offset);
        ahead_token = fetch_token(first + 1);
        // Q up front: loaded inside the loop it is waited on with the
        // lookahead loads (vmcnt is in order) on every tile.
        load_query();
    }
    for (int64_t tile = first; tile < last; ++tile) {
        int page = -1, offset = 0;
        if constexpr (Lookahead) {
            page = ahead_page >= 0 && ahead_page < pages ? ahead_page : -1;
            offset = ahead_offset;
            ahead_page = lookup_page(ahead_token, ahead_offset);
            ahead_token = fetch_token(tile + 2);
        } else if (g == 0 && tile * 16 + r < topk && request_valid) {
            const int token = indices[row * is0 + tile * 16 + r];
            page = lookup_page(token, offset);
            if (page < 0 || page >= pages)
                page = -1;
        }
        if constexpr (!DirectMeta) {
            page = __shfl(page, r, 64);
            offset = __shfl(offset, r, 64);
        }
        const uint64_t live_mask = __ballot(page >= 0);
        if (live_mask == 0)
            continue;
        if (!query_loaded) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                query[i] =
                    r < Group ? load8(q + row * qs0 + (first_head + r) * qs1 + i * 32 + g * 8) : zero;
#pragma unroll
            for (int d = 0; d < 16; ++d)
                pv[d] = {0, 0, 0, 0};
            query_loaded = true;
        }
        bool valid;
        int64_t kb, vb, v_lane[4];
        if constexpr (BaseMeta) {
            int64_t k_token = -1, v_token = -1;
            if (page >= 0) {
                k_token = int64_t(page) * ks.block + offset * ks.token + kv_head * ks.head;
                if constexpr (UnmaskedHeads && Cache::shared_offsets)
                    v_token = k_token;
                else
                    v_token = int64_t(page) * vs.block + offset * vs.token + kv_head * vs.head;
            }
            kb = __shfl(k_token, lane / 4, 64);
            valid = kb >= 0;
            vb = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                v_lane[j] = __shfl(v_token, g + 4 * j, 64);
        } else {
            const int natural_page = __shfl(page, lane / 4, 64);
            const int natural_offset = __shfl(offset, lane / 4, 64);
            valid = natural_page >= 0;
            kb = valid ? static_cast<int64_t>(natural_page) * ks.block +
                             natural_offset * ks.token + kv_head * ks.head
                       : 0;
            vb = valid ? static_cast<int64_t>(natural_page) * vs.block +
                             natural_offset * vs.token + kv_head * vs.head
                       : 0;
        }
        typename Cache::VFragment values[16];
        Float4 scores = {0, 0, 0, 0};
        if (live_mask == (DirectMeta ? uint64_t{0xffff} : ~uint64_t{0})) {
            if (load_v_first) {
                if constexpr (BaseMeta) {
                    paged_load_lane_bases<Cache, false>(v, v_lane, r, values);
                } else if constexpr (InLaneV) {
                    paged_load_lane_values<Cache, false>(v, page, offset, g, r, kv_head, vs, values);
                } else {
#pragma unroll
                    for (int d = 0; d < 16; ++d)
                        values[d] = Cache::load_v(v + vb + d * 16 + (lane % 4) * 4);
                }
            }
            if constexpr (WideK) {
                Words4 wide[2];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    __builtin_memcpy(wide + i, k + kb + i * 64 + (lane % 4) * 16, 16);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const Words4 natural = wide[i % 2];
                    if (i + 2 < 4)
                        __builtin_memcpy(wide + i % 2, k + kb + (i + 2) * 64 + (lane % 4) * 16, 16);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const Vec8 key = Cache::shuffle_k(
                            typename Cache::KFragment{natural[2 * h], natural[2 * h + 1]}, r * 4 + g);
                        dot(query[2 * i + h].x, key.x, scores);
                        dot(query[2 * i + h].y, key.y, scores);
                    }
                }
            } else {
            typename Cache::KFragment window[KWindow];
#pragma unroll
            for (int i = 0; i < KWindow; ++i)
                window[i] = Cache::load_k(k + kb + i * 32 + (lane % 4) * 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto natural = window[i % KWindow];
                if (i + KWindow < 8)
                    window[i % KWindow] = Cache::load_k(k + kb + (i + KWindow) * 32 + (lane % 4) * 8);
                const Vec8 key = Cache::shuffle_k(natural, r * 4 + g);
                dot(query[i].x, key.x, scores);
                dot(query[i].y, key.y, scores);
            }
            }
            if (!load_v_first) {
                if constexpr (BaseMeta) {
                    paged_load_lane_bases<Cache, false>(v, v_lane, r, values);
                } else if constexpr (InLaneV) {
                    paged_load_lane_values<Cache, false>(v, page, offset, g, r, kv_head, vs, values);
                } else {
#pragma unroll
                    for (int d = 0; d < 16; ++d)
                        values[d] = Cache::load_v(v + vb + d * 16 + (lane % 4) * 4);
                }
            }
        } else {
            if (load_v_first) {
                if constexpr (BaseMeta) {
                    paged_load_lane_bases<Cache, true>(v, v_lane, r, values);
                } else if constexpr (InLaneV) {
                    paged_load_lane_values<Cache, true>(v, page, offset, g, r, kv_head, vs, values);
                } else {
#pragma unroll
                    for (int d = 0; d < 16; ++d)
                        values[d] = valid ? Cache::load_v(v + vb + d * 16 + (lane % 4) * 4)
                                          : typename Cache::VFragment{};
                }
            }
            if constexpr (WideK) {
                Words4 wide[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wide[i] = Words4{};
                    if (valid)
                        __builtin_memcpy(wide + i, k + kb + i * 64 + (lane % 4) * 16, 16);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const Words4 natural = wide[i % 2];
                    if (i + 2 < 4) {
                        wide[i % 2] = Words4{};
                        if (valid)
                            __builtin_memcpy(wide + i % 2, k + kb + (i + 2) * 64 + (lane % 4) * 16, 16);
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const Vec8 key = Cache::shuffle_k(
                            typename Cache::KFragment{natural[2 * h], natural[2 * h + 1]}, r * 4 + g);
                        dot(query[2 * i + h].x, key.x, scores);
                        dot(query[2 * i + h].y, key.y, scores);
                    }
                }
            } else {
            typename Cache::KFragment window[KWindow];
#pragma unroll
            for (int i = 0; i < KWindow; ++i)
                window[i] = valid ? Cache::load_k(k + kb + i * 32 + (lane % 4) * 8)
                                  : typename Cache::KFragment{};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto natural = window[i % KWindow];
                if (i + KWindow < 8)
                    window[i % KWindow] = valid ? Cache::load_k(k + kb + (i + KWindow) * 32 + (lane % 4) * 8)
                                          : typename Cache::KFragment{};
                const Vec8 key = Cache::shuffle_k(natural, r * 4 + g);
                dot(query[i].x, key.x, scores);
                dot(query[i].y, key.y, scores);
            }
            }
            if (!load_v_first) {
                if constexpr (BaseMeta) {
                    paged_load_lane_bases<Cache, true>(v, v_lane, r, values);
                } else if constexpr (InLaneV) {
                    paged_load_lane_values<Cache, true>(v, page, offset, g, r, kv_head, vs, values);
                } else {
#pragma unroll
                    for (int d = 0; d < 16; ++d)
                        values[d] = valid ? Cache::load_v(v + vb + d * 16 + (lane % 4) * 4)
                                          : typename Cache::VFragment{};
                }
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const bool live = BaseMeta ? v_lane[e] >= 0 : __shfl(page, g + e * 4, 64) >= 0;
            scores[e] =
                (UnmaskedHeads || r < Group) && live
                    ? scores[e] * (Cache::fp8 ? (0.0625f * kLog2E) * k_scale : (0.0625f * kLog2E))
                    : -1.0e20f;
        }
        float maximum = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 64));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 32, 64));
        maximum = fmaxf(maximum, row_max);
        const float alpha = sparse_attention_detail::mmac::exp2(row_max - maximum);
        Float4 probabilities;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const bool live = BaseMeta ? v_lane[e] >= 0 : __shfl(page, g + e * 4, 64) >= 0;
            probabilities[e] = (UnmaskedHeads || r < Group) && live
                                   ? sparse_attention_detail::mmac::exp2(scores[e] - maximum)
                                   : 0.0f;
        }
        float sum = (probabilities[0] + probabilities[1]) + (probabilities[2] + probabilities[3]);
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        row_sum = row_sum * alpha + sum;
        row_max = maximum;
        const Half4 p = paged_probability<Cache::fp8>(
            InLaneV ? probabilities : transpose_p(probabilities, g));
        if constexpr (Tuned) {
            // One wave-uniform branch around all 32 multiplies; the per-d form is
            // if-converted into unconditional multiplies by alpha or 1.0.
            if (__ballot(alpha != 1.0f) != 0) {
#pragma unroll
                for (int d = 0; d < 16; ++d)
                    pv[d] *= alpha;
            }
        }
#pragma unroll
        for (int d = 0; d < 16; ++d) {
            if constexpr (!Tuned) {
                if (__ballot(alpha != 1.0f) != 0)
                    pv[d] *= alpha;
            }
            if constexpr (InLaneV)
                dot(p, Cache::lane_v(values, d), pv[d]);
            else
                dot(p, Cache::transpose_v(values[d], g, r), pv[d]);
        }
    }
    if (!query_loaded) {
#pragma unroll
        for (int d = 0; d < 16; ++d)
            pv[d] = {0, 0, 0, 0};
    }
    constexpr bool RawWaveMerge = UnmaskedHeads && Waves > 1;
    float final_lse;
    if constexpr (RawWaveMerge)
        final_lse = row_sum > 0 ? row_max : -INFINITY;
    else
        final_lse = row_sum > 0 ? row_max + log2f(row_sum) : -INFINITY;
    if constexpr (RawWaveMerge) {
        if (!(row_sum > 0)) {
#pragma unroll
            for (int d = 0; d < 16; ++d) pv[d] = {0, 0, 0, 0};
        }
    } else {
#pragma unroll
        for (int d = 0; d < 16; ++d) {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                pv[d][e] = row_sum > 0 ? pv[d][e] / row_sum : 0.0f;
        }
    }
    if constexpr (Waves > 1) {
        constexpr int stride = kSharedStride;
        constexpr bool KeepWaveZero = UnmaskedHeads;
        constexpr int SharedWaves = Waves - (KeepWaveZero ? 1 : 0);
        extern __shared__ float shared[];
        float *shared_lse = shared + SharedWaves * Group * stride;
        float *shared_sum = shared_lse + Waves * Group;
        if (r < Group) {
#pragma unroll
            for (int d = 0; d < 16; ++d) {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (!KeepWaveZero || wave != 0)
                        shared[((wave - (KeepWaveZero ? 1 : 0)) * Group + r) * stride +
                               (InLaneV ? 16 * (g + 4 * e) + d : d * 16 + g + e * 4)] = pv[d][e];
            }
            if (g == 0) {
                shared_lse[wave * Group + r] = final_lse;
                if constexpr (RawWaveMerge)
                    shared_sum[wave * Group + r] = row_sum > 0 ? row_sum : 0.0f;
            }
        }
        __syncthreads();
        if (wave != 0)
            return;
        if (r < Group) {
            float maximum = -INFINITY, denominator = 0;
            float weights[Waves];
#pragma unroll
            for (int w = 0; w < Waves; ++w)
                maximum = fmaxf(maximum, shared_lse[w * Group + r]);
#pragma unroll
            for (int w = 0; w < Waves; ++w) {
                weights[w] =
                    maximum > -INFINITY
                        ? sparse_attention_detail::mmac::exp2(shared_lse[w * Group + r] - maximum)
                        : 0;
                if constexpr (RawWaveMerge)
                    denominator += weights[w] * shared_sum[w * Group + r];
                else
                    denominator += weights[w];
            }
#pragma unroll
            for (int d = 0; d < 16; ++d) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float value = 0;
#pragma unroll
                    for (int w = 0; w < Waves; ++w) {
                        const float state = (KeepWaveZero && w == 0)
                            ? pv[d][e]
                            : shared[((w - (KeepWaveZero ? 1 : 0)) * Group + r) * stride +
                                     (InLaneV ? 16 * (g + 4 * e) + d : d * 16 + g + e * 4)];
                        value += state * weights[w];
                    }
                    pv[d][e] = denominator > 0 ? value / denominator : 0;
                }
            }
            final_lse = denominator > 0 ? maximum + log2f(denominator) : -INFINITY;
        }
    }
    const int64_t pr = (static_cast<int64_t>(split) * rows + row) * heads + first_head + r;
    const float vs_out = Cache::fp8 ? *v_scale_ptr : 1.0f;
    if constexpr (InLaneV) {
        if (r < Group) {
#pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const Float4 value = Float4{pv[4 * chunk][e], pv[4 * chunk + 1][e],
                                               pv[4 * chunk + 2][e], pv[4 * chunk + 3][e]} * vs_out;
                    const int column = 16 * (g + 4 * e) + 4 * chunk;
                    if (is_split) {
                        *reinterpret_cast<Float4 *>(partial + pr * 256 + column) = value;
                    } else {
                        const Half4 packed = paged_probability<Cache::fp8>(value);
                        __builtin_memcpy(out + row * os0 + (first_head + r) * os1 + column,
                                         &packed, sizeof(packed));
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int d = 0; d < 16; ++d) {
            // All lanes participate in the transpose, including masked heads.
            const Float4 value = transpose_p(pv[d], g) * vs_out;
            if (r < Group) {
                if (is_split) {
                    *reinterpret_cast<Float4 *>(partial + pr * 256 + d * 16 + g * 4) = value;
                } else {
                    const Half4 packed = bf16_prob(value);
                    // memcpy preserves noncontiguous/un-aligned out view support.
                    __builtin_memcpy(out + row * os0 + (first_head + r) * os1 + d * 16 + g * 4,
                                     &packed, sizeof(packed));
                }
            }
        }
    }
    if (r < Group && is_split && g == 0)
        lse[pr] = final_lse;
}

// Compatibility path for other head shapes/unaligned views; no split workspace.
template <class Cache>
__global__ void sparse_gqa_paged_attention_generic_kernel(
    const at::BFloat16 *q, const typename Cache::KElement *k, const typename Cache::VElement *v,
    const int32_t *indices, const int32_t *table, const int32_t *requests, at::BFloat16 *out,
    int64_t heads, int64_t kv_heads, int64_t dim, int64_t pages, int64_t page_size,
    int64_t table_width, int64_t num_requests, int64_t topk, int64_t qs0, int64_t qs1,
    CacheStrides ks, CacheStrides vs, int64_t is0, int64_t ts0, int64_t os0, int64_t os1,
    float scale_log2, const float *k_scale_ptr, const float *v_scale_ptr) {
    if constexpr (Cache::fp8)
        scale_log2 *= *k_scale_ptr;
    constexpr int block_n = 64;
    const int64_t dim_tiles = (dim + kOutputTile - 1) / kOutputTile;
    int64_t program = blockIdx.x;
    const int64_t dim_start = (program % dim_tiles) * kOutputTile;
    program /= dim_tiles;
    const int64_t head = program % heads;
    program /= heads;
    const int64_t row = program;
    const int lane = threadIdx.x;
    const int64_t kv_head = head / kGroup;
    const int64_t request = requests[row];
    const bool request_valid = request >= 0 && request < num_requests;
    const int64_t tiles = (topk + block_n - 1) / block_n;
    __shared__ float scores[64];
    __shared__ at::BFloat16 probabilities[64];
    __shared__ int64_t physical_pages[64];
    __shared__ int64_t offsets[64];
    __shared__ float maximum, normalizer, alpha;
    if (lane == 0) {
        maximum = -1.0e20f;
        normalizer = 0.0f;
    }
    float acc[4] = {0, 0, 0, 0};
    __syncthreads();
    for (int64_t tile = 0; tile < tiles; ++tile) {
        if (lane < block_n) {
            const int64_t column = tile * block_n + lane;
            const int64_t token = column < topk ? indices[row * is0 + column] : -1;
            const bool valid = request_valid && token >= 0 && token / page_size < table_width;
            int64_t page = valid ? table[request * ts0 + token / page_size] : -1;
            if (page < 0 || page >= pages)
                page = -1;
            physical_pages[lane] = page;
            offsets[lane] = token >= 0 ? token % page_size : 0;
            float dot = 0.0f;
            if (page >= 0) {
                const int64_t kb = page * ks.block + offsets[lane] * ks.token + kv_head * ks.head;
                for (int64_t d = 0; d < dim; ++d)
                    dot += static_cast<float>(q[row * qs0 + head * qs1 + d]) *
                           Cache::decode(k[kb + d]);
            }
            scores[lane] = page >= 0 ? dot * scale_log2 : -1.0e20f;
        }
        __syncthreads();
        if (lane == 0) {
            float next_max = maximum;
            for (int t = 0; t < block_n; ++t)
                next_max = fmaxf(next_max, scores[t]);
            alpha = exp2f(maximum - next_max);
            float sum = 0;
            for (int t = 0; t < block_n; ++t) {
                const float p = physical_pages[t] >= 0 ? exp2f(scores[t] - next_max) : 0.0f;
                sum += p;
                probabilities[t] = at::BFloat16(p);
            }
            normalizer = normalizer * alpha + sum;
            maximum = next_max;
        }
        __syncthreads();
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t d = dim_start + lane + part * kThreads;
            if (d < dim) {
                acc[part] *= alpha;
                for (int t = 0; t < block_n; ++t) {
                    const int64_t page = physical_pages[t];
                    if (page >= 0) {
                        const int64_t vb =
                            page * vs.block + offsets[t] * vs.token + kv_head * vs.head;
                        acc[part] +=
                            static_cast<float>(probabilities[t]) * Cache::decode(v[vb + d]);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int64_t d = dim_start + lane + part * kThreads;
        if (d < dim) {
            float value = normalizer > 0 ? acc[part] / fmaxf(normalizer, 1.0e-20f) : 0.0f;
            if constexpr (Cache::fp8)
                value *= *v_scale_ptr;
            out[row * os0 + head * os1 + d] = at::BFloat16(value);
        }
    }
}

// Host input contract only; allocation and current-stream work stay in run_paged_attention.
inline void validate_paged_attention_inputs(const at::Tensor &q, const at::Tensor &k_cache,
    const at::Tensor &v_cache, const at::Tensor &logical_indices,
    const at::Tensor &block_table, const at::Tensor &token_to_req) {
    using namespace sparse_attention_detail;
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) > 0,
                "q must be [rows, heads, head_dim] on the accelerator");
    check_device(k_cache, q, "k_cache");
    check_device(v_cache, q, "v_cache");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "paged sparse GQA requires BF16 Q");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "K/V must have matching [pages, page_size, kv_heads, head_dim] shapes");
    TORCH_CHECK(k_cache.size(0) > 0 && k_cache.size(1) > 0 && k_cache.size(2) > 0,
                "K/V cache must be nonempty");
    const int64_t dim = q.size(2);
    TORCH_CHECK(dim >= 16 && (dim & (dim - 1)) == 0 && k_cache.size(3) == dim,
                "head_dim must match K/V and be a power of two >= 16");
    TORCH_CHECK(q.size(1) % k_cache.size(2) == 0 && q.size(1) / k_cache.size(2) == kGroup,
                "paged sparse GQA requires group=12 (query_heads / kv_heads)");
    check_i32(logical_indices, q, "logical_indices");
    check_i32(block_table, q, "block_table");
    check_i32(token_to_req, q, "token_to_req");
    TORCH_CHECK(logical_indices.dim() == 2 && logical_indices.size(0) == q.size(0) &&
                    logical_indices.size(1) > 0,
                "logical_indices must be [rows, positive_topk]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) > 0 && block_table.size(1) > 0,
                "block_table must be nonempty and two-dimensional");
    TORCH_CHECK(token_to_req.dim() == 1 && token_to_req.numel() == q.size(0) &&
                    token_to_req.stride(0) == 1,
                "token_to_req must be contiguous [rows]");
    TORCH_CHECK(q.stride(2) == 1 && k_cache.stride(3) == 1 && v_cache.stride(3) == 1 &&
                    logical_indices.stride(1) == 1 && block_table.stride(1) == 1,
                "Q/K/V head dimensions and metadata columns must be contiguous");
}

// ---- Host orchestration: preserve validation order and workspace lifetime ----
template <class Cache>
at::Tensor run_paged_attention(const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &v_cache,
               const at::Tensor &logical_indices, const at::Tensor &block_table,
               const at::Tensor &token_to_req, c10::optional<at::Tensor> out_,
               const float *k_scale_ptr = nullptr, const float *v_scale_ptr = nullptr,
               const hipDeviceProp_t *known_props = nullptr,
               const int32_t *row_tile_counts = nullptr,
               const int32_t *mixed_route = nullptr, bool skip_prepare = false) {
    using namespace sparse_attention_detail;
    validate_paged_attention_inputs(q, k_cache, v_cache, logical_indices, block_table, token_to_req);
    const int64_t dim = q.size(2);
    at::hip::HIPGuardMasqueradingAsCUDA guard{static_cast<char>(q.get_device())};
    auto out = out_.has_value() ? *out_ : at::empty_like(q);
    check_device(out, q, "out");
    TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == q.scalar_type() &&
                    out.stride(2) == 1,
                "out must match Q shape/dtype/device with contiguous head dimension");
    if (q.size(0) == 0)
        return out;
    const int64_t programs = q.size(0) * k_cache.size(2);
    bool fast = dim == 256 && is_pointer_aligned_8(q) && is_pointer_aligned_8(k_cache) && is_pointer_aligned_8(v_cache) &&
                q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && k_cache.stride(0) % 8 == 0 &&
                k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                v_cache.stride(0) % 8 == 0 && v_cache.stride(1) % 8 == 0 &&
                v_cache.stride(2) % 8 == 0;
    bool one_wave = true;
    bool arch938 = false;
    bool arch936 = false;
    int device_cus = 1;
    int grid_limits[3] = {0, 0, 0};
    // MMAC helper is implemented on gfx936/gfx938 only. Other targets use scalar math.
    if (fast) {
        hipDeviceProp_t props{};
        if (known_props)
            props = *known_props;
        else
            C10_HIP_CHECK(hipGetDeviceProperties(&props, q.get_device()));
        const std::string arch(props.gcnArchName);
        fast = arch.compare(0, 6, "gfx936") == 0 || arch.compare(0, 6, "gfx938") == 0;
        // gfx936 BF16 and mid-size FP8 grids favor four waves in the paired sweep;
        // group-12 FP8 grids of at most four programs split wider with one wave.
        arch938 = arch.compare(0, 6, "gfx938") == 0;
        arch936 = arch.compare(0, 6, "gfx936") == 0;
        one_wave = (Cache::fp8 && (programs >= 128 || programs <= 4)) || arch938;
        device_cus = std::max(1, props.multiProcessorCount);
        for (int i = 0; i < 3; ++i) grid_limits[i] = props.maxGridSize[i];
    }
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if constexpr (Cache::mixed_predecoded)
        TORCH_CHECK(fast && arch936,
                    "private mixed K/V workspace requires the validated gfx936 MMAC path");
    if (!fast) {
        const int64_t grid = q.size(0) * q.size(1) * ((dim + kOutputTile - 1) / kOutputTile);
        sparse_gqa_paged_attention_generic_kernel<Cache><<<checked_grid_extent(grid), kThreads, 0, stream>>>(
            q.data_ptr<at::BFloat16>(),
            reinterpret_cast<const typename Cache::KElement *>(k_cache.data_ptr()),
            reinterpret_cast<const typename Cache::VElement *>(v_cache.data_ptr()),
            logical_indices.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
            token_to_req.data_ptr<int32_t>(), out.data_ptr<at::BFloat16>(), q.size(1),
            k_cache.size(2), dim, k_cache.size(0), k_cache.size(1), block_table.size(1),
            block_table.size(0), logical_indices.size(1), q.stride(0), q.stride(1),
            cache_strides(k_cache), cache_strides(v_cache), logical_indices.stride(0),
            block_table.stride(0), out.stride(0), out.stride(1),
            kLog2E / std::sqrt(static_cast<float>(dim)), k_scale_ptr, v_scale_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
// ---- FP8 preparation eligibility: representation safety precedes allocation ----
    if constexpr (Cache::fp8 && Cache::e5 && Cache::packed_nan &&
                  sizeof(typename Cache::KElement) == 1 && !Cache::mixed_predecoded) {
        // Large-M/high-reuse path. Every call (including every
        // graph replay) re-decodes on the current stream; no stale cross-call
        // cache or CPU metadata copy. Graph/event timing includes both GPU
        // prepass and main attention, not CPU allocator execution; allocations
        // remain within this public call, as in the original graph protocol.
        // No fixed workspace budget: allocate the full table-shaped K/V.
        // Retain representation and launch limits independently of capacity;
        // allocation failure propagates normally, including during capture.
        constexpr int64_t index_limit = std::numeric_limits<int32_t>::max();
        const int64_t max_table_entries = std::min<int64_t>(
            index_limit, grid_limits[1]);
        if (!skip_prepare && arch936 && q.size(0) >= 4096 &&
            block_table.size(0) <= max_table_entries / block_table.size(1) &&
            k_cache.size(1) <= index_limit / k_cache.size(2)) {
            const int64_t table_entries = block_table.size(0) * block_table.size(1);
            const int64_t table_bytes = table_entries * sizeof(int32_t);
            // Prove the aggregate byte count fits INT64 before multiplying.
            // This is an integer-safety bound, not a tunable memory budget.
            const int64_t cache_element_limit =
                (std::numeric_limits<int64_t>::max() - table_bytes) /
                (sizeof(uint8_t) + sizeof(at::BFloat16));
            int64_t cache_elements = table_entries;
            bool representable = cache_elements <= cache_element_limit;
            for (const int64_t extent : {k_cache.size(1), k_cache.size(2), dim}) {
                if (!representable || cache_elements > cache_element_limit / extent) {
                    representable = false;
                    break;
                }
                cache_elements *= extent;
            }
            // D==256 and positive dimensions follow from validation/fast.
            // page_size*Hkv and logical pages were explicitly bounded to
            // INT32 above. Unsupported metadata/grids retain the direct path.
            if (representable &&
                (k_cache.size(1) * k_cache.size(2) + 3) / 4 <= grid_limits[0]) {
                // Two safe owners, each head-major.  Public view shape stays
                // [page,token,head,D], with K uint8 and V BF16 only internally.
                auto compact_k_owner = at::empty({table_entries, k_cache.size(2),
                                                  k_cache.size(1), dim}, q.options().dtype(at::kByte));
                auto decoded_v_owner = at::empty({table_entries, k_cache.size(2),
                                                  k_cache.size(1), dim}, q.options());
                const int64_t token_stride = dim;
                const int64_t head_stride = k_cache.size(1) * token_stride;
                const int64_t page_stride = k_cache.size(2) * head_stride;
                const std::vector<int64_t> decoded_shape =
                    {table_entries, k_cache.size(1), k_cache.size(2), dim};
                const std::vector<int64_t> decoded_strides =
                    {page_stride, token_stride, head_stride, 1};
                auto compact_k = compact_k_owner.as_strided(decoded_shape, decoded_strides, 0);
                auto decoded_v = decoded_v_owner.as_strided(decoded_shape, decoded_strides, 0);
                auto decoded_table = at::empty({block_table.size(0), block_table.size(1)},
                                               block_table.options());
                at::Tensor row_tiles, active_limits, route_owner;
                const int32_t *route_ptr = nullptr;
                const int32_t *row_tiles_ptr = nullptr, *active_limits_ptr = nullptr;
                if (logical_indices.size(1) < index_limit &&
                    q.size(0) <= grid_limits[0]) {
                    row_tiles = at::empty({q.size(0)}, block_table.options());
                    active_limits = at::empty({block_table.size(0)}, block_table.options());
                    const int64_t init_blocks = std::min<int64_t>(
                        (table_entries + 255) / 256, grid_limits[0]);
                    sparse_gqa_paged_attention_initialize_kernel<>
                        <<<init_blocks, 256, 0, stream>>>(
                            decoded_table.data_ptr<int32_t>(), active_limits.data_ptr<int32_t>(),
                            table_entries, block_table.size(0));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    sparse_gqa_paged_attention_last_tile_kernel<>
                        <<<q.size(0), 64, 0, stream>>>(
                            logical_indices.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                            token_to_req.data_ptr<int32_t>(), row_tiles.data_ptr<int32_t>(),
                            active_limits.data_ptr<int32_t>(),
                            logical_indices.size(1), logical_indices.stride(0), block_table.stride(0),
                            block_table.size(0), block_table.size(1), k_cache.size(0),
                            hytlass::FastDivmod(int(k_cache.size(1))));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    row_tiles_ptr = row_tiles.data_ptr<int32_t>();
                    active_limits_ptr = active_limits.data_ptr<int32_t>();
                    // Routed attention is only instantiated for the exact
                    // unsplit W1 3D grid. Unsupported dimensions retain V19.
                    if (k_cache.size(2) <= grid_limits[1] && grid_limits[2] >= 1) {
                        route_owner = at::empty({1}, block_table.options());
                        sparse_gqa_paged_attention_choose_route_kernel<>
                            <<<1, 64, 0, stream>>>(
                                active_limits_ptr, route_owner.data_ptr<int32_t>(),
                                block_table.size(0), k_cache.size(1), k_cache.size(2));
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                        route_ptr = route_owner.data_ptr<int32_t>();
                    }
                }
                // Preparation must follow the metadata scan on this stream.
                // Owners still cover the complete original table-shaped cache.
                const int token_heads = int(k_cache.size(1) * k_cache.size(2));
                const dim3 predecode_grid((int64_t(token_heads) + 3) / 4, table_entries);
                if (active_limits_ptr) {
                    const int64_t token_head_blocks = (int64_t(token_heads) + 3) / 4;
                    const int64_t base_prefix_ctas = token_head_blocks * block_table.size(0);
                    // Two CTA rounds when the table/grid permit. No fixed page
                    // values or device metadata reads participate in selection.
                    const int64_t target_prefix_ctas = 2 * int64_t(device_cus);
                    const int64_t desired_parts =
                        1 + (target_prefix_ctas - 1) / base_prefix_ctas;
                    const int64_t prefix_parts = std::min<int64_t>(
                        block_table.size(1),
                        std::min<int64_t>(grid_limits[2], desired_parts));
                    const dim3 prefix_grid(token_head_blocks,
                                           block_table.size(0), prefix_parts);
                    auto emit_predecode = [&](auto routed_tag) {
                      sparse_gqa_paged_attention_predecode_prefix_kernel<Cache,
                          decltype(routed_tag)::value>
                        <<<prefix_grid, 256, 0, stream>>>(
                            reinterpret_cast<const typename Cache::KElement *>(k_cache.data_ptr()),
                            reinterpret_cast<const typename Cache::VElement *>(v_cache.data_ptr()),
                            block_table.data_ptr<int32_t>(), compact_k.data_ptr<uint8_t>(),
                            decoded_v.data_ptr<at::BFloat16>(), decoded_table.data_ptr<int32_t>(),
                            token_heads, int(k_cache.size(1)), int(k_cache.size(2)),
                            k_cache.size(0), block_table.size(1), block_table.stride(0),
                            cache_strides(k_cache), cache_strides(v_cache),
                            hytlass::FastDivmod(int(k_cache.size(2))), active_limits_ptr,
                            route_ptr);
                    };
                    if (route_ptr)
                        emit_predecode(std::true_type{});
                    else
                        emit_predecode(std::false_type{});
                } else {
                    sparse_gqa_paged_attention_predecode_kernel<Cache>
                        <<<predecode_grid, 256, 0, stream>>>(
                            reinterpret_cast<const typename Cache::KElement *>(k_cache.data_ptr()),
                            reinterpret_cast<const typename Cache::VElement *>(v_cache.data_ptr()),
                            block_table.data_ptr<int32_t>(), compact_k.data_ptr<uint8_t>(),
                            decoded_v.data_ptr<at::BFloat16>(), decoded_table.data_ptr<int32_t>(),
                            token_heads, int(k_cache.size(1)), int(k_cache.size(2)), k_cache.size(0),
                            block_table.stride(0), cache_strides(k_cache), cache_strides(v_cache),
                            hytlass::FastDivmod(int(block_table.size(1))),
                            hytlass::FastDivmod(int(k_cache.size(2))), active_limits_ptr);
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                // Row and prefix summaries are owning locals and rebuilt every
                // call/replay; no data-dependent CPU read or persistent cache.
                run_paged_attention<MixedPagedCache<Cache>>(
                    q, compact_k, decoded_v, logical_indices, decoded_table,
                    token_to_req, out, k_scale_ptr, v_scale_ptr, known_props,
                    row_tiles_ptr, route_ptr);
                if (!route_ptr) return out;
                // Same output and same stream; the two attention launches are
                // complementary. Explicit skip prevents reentering preparation.
                return run_paged_attention<Cache>(q, k_cache, v_cache, logical_indices, block_table,
                    token_to_req, out, k_scale_ptr, v_scale_ptr, known_props,
                    nullptr, route_ptr, true);
            }
        }
    }
// ---- Family performance policy: choose waves and private splits without changing the API ----
    const int64_t tiles = (logical_indices.size(1) + 15) / 16;
    const bool target936 = arch936 && Cache::packed_nan;
    const bool tiny_target = target936 && programs <= 2 && tiles >= 128 && tiles <= 256;
    int splits = 1;
    // gfx938 FP8: 129-tile tiny grids
    // favor four waves and smaller external merge fan-in. Keep V8
    // scheduling when the four-wave split CTAs exceed one per CU.
    if (Cache::fp8 && arch938 && tiles == 129) {
        const int64_t target = 1 + 127 / programs;
        int four_splits = 1;
        while (four_splits < target && four_splits * 2 <= 32)
            four_splits *= 2;
        if (programs <= device_cus / four_splits)
            one_wave = false;
    }
    if (one_wave) {
        // Keep >=2 tiles per split. With enough base CTAs, cap merge fan-in
        // while exposing more independent work; all choices use shape/device only.
        const bool populated = programs >= device_cus;
        const int64_t cap = std::min<int64_t>(populated ? 8 : 64, std::max<int64_t>(1, tiles / 2));
        if (arch938 && populated && programs <= 4 * int64_t(device_cus)) {
            // Fill eight CTA waves per CU, rounding split count upward.
            // A floor budget underfills neighboring grids (e.g. 65/96/129).
            const int64_t target = 1 + (8 * int64_t(device_cus) - 1) / programs;
            while (splits < target && splits * 2 <= cap)
                splits *= 2;
        } else {
            const int64_t target_ctas = (populated ? 32 : 8) * int64_t(device_cus);
            while (splits * 2 <= cap && programs <= target_ctas / (splits * 2))
                splits *= 2;
        }
    } else {
        // Retain the measured four-wave split policy on gfx936 BF16.
        const int64_t cap = std::min<int64_t>(32, std::max<int64_t>(1, tiles / 4));
        const int64_t target = 1 + 127 / programs;
        while (splits < target && splits * 2 <= cap)
            splits *= 2;
    }
    // E5 group-12 grids between two and four CTAs/CU: eight external
    // partitions increase partial/merge traffic without reducing main-kernel
    // latency. Four partitions won the paired full-interface sweep.
    if (arch936 && Cache::e5 &&
        programs > 2 * int64_t(device_cus) && programs <= 4 * int64_t(device_cus))
        splits = std::min(splits, 4);
    // Target G12 W4 keeps wave0 in registers. Raw merge state uses
    // 37824 bytes of LDS, still at most one CTA per 64-KiB CU.
    // Retain the measured one-CTA-round schedule below and the historical
    // schedule above one base CTA/CU (81..112 programs regressed).
    if (target936 &&
        programs > 2 && programs <= int64_t(device_cus)) {
        one_wave = false;
        splits = 1;
        const int64_t cap = std::min<int64_t>(32, std::max<int64_t>(1, tiles / 4));
        while (splits * 2 <= cap && programs <= int64_t(device_cus) / (splits * 2))
            splits *= 2;
    }
    if (tiny_target) {
        one_wave = false;
        splits = 32;
    }
    at::Tensor partial, lse;
    float *partial_ptr = nullptr;
    float *lse_ptr = nullptr;
    if (splits > 1) {
        partial = at::empty({splits, q.size(0), q.size(1), dim}, q.options().dtype(at::kFloat));
        lse = at::empty({splits, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
        partial_ptr = partial.data_ptr<float>();
        lse_ptr = lse.data_ptr<float>();
    }
    // Preserve the flat fallback when a 3D grid dimension is too large.
    const bool grid3d = target936 &&
                        q.size(0) <= grid_limits[0] &&
                        k_cache.size(2) <= grid_limits[1] && splits <= grid_limits[2];
    const bool row8_grid = target936 && splits > 1 && q.size(0) % 8 == 0 &&
                           k_cache.size(2) <= grid_limits[0] / 8 &&
                           q.size(0) / 8 <= grid_limits[1] && splits <= grid_limits[2];
    const auto flat_grid = checked_grid_extent(programs * splits);
    const dim3 grid = row8_grid ? dim3(k_cache.size(2) * 8, q.size(0) / 8, splits)
                               : (grid3d ? dim3(q.size(0), k_cache.size(2), splits)
                                         : dim3(flat_grid));
    auto launch = [&](auto wave_tag, auto split_tag, auto ahead_tag, auto page_lookup) {
        constexpr int Waves = decltype(wave_tag)::value;
        constexpr bool TargetShared = Cache::packed_nan;
        constexpr int target_shared_bytes = Waves == 1 ? 0 :
            ((Waves - (TargetShared ? 1 : 0)) * kGroup * kSharedStride +
             Waves * kGroup * (TargetShared ? 2 : 1)) * sizeof(float);
        const int shared_bytes = arch936 ? target_shared_bytes :
            (Waves == 1 ? 0 : Waves * kGroup * (kSharedStride + 1) * sizeof(float));
        auto launch_grid = [&](auto grid_tag, auto tile_tag) {
          auto emit = [&](auto rows_tag) {
            constexpr int RowsPerCTA = decltype(rows_tag)::value;
            const dim3 kernel_grid = RowsPerCTA > 1
                ? dim3((q.size(0) + RowsPerCTA - 1) / RowsPerCTA, k_cache.size(2), 1)
                : grid;
            auto emit_route = [&](auto routing_tag) {
              sparse_gqa_paged_attention_mmac_kernel<Cache, Waves, decltype(split_tag)::value,
                                                  decltype(ahead_tag)::value,
                                                  decltype(grid_tag)::value,
                                                  decltype(tile_tag)::value, decltype(page_lookup),
                                                  RowsPerCTA, decltype(routing_tag)::value>
                <<<kernel_grid, Waves * 64 * RowsPerCTA, shared_bytes, stream>>>(
                    q.data_ptr<at::BFloat16>(),
                    reinterpret_cast<const typename Cache::KElement *>(k_cache.data_ptr()),
                    reinterpret_cast<const typename Cache::VElement *>(v_cache.data_ptr()),
                    logical_indices.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                    token_to_req.data_ptr<int32_t>(), out.data_ptr<at::BFloat16>(), partial_ptr,
                    lse_ptr, q.size(0), q.size(1), k_cache.size(2), k_cache.size(0),
                    block_table.size(1), block_table.size(0), logical_indices.size(1), q.stride(0),
                    q.stride(1), cache_strides(k_cache), cache_strides(v_cache),
                    logical_indices.stride(0), block_table.stride(0), out.stride(0), out.stride(1),
                    splits, k_scale_ptr, v_scale_ptr, page_lookup, row_tile_counts,
                    mixed_route);
            };
            if constexpr (Cache::fp8 && Cache::e5 && Cache::packed_nan && Waves == 1 &&
                          !decltype(split_tag)::value && decltype(grid_tag)::value &&
                          !decltype(tile_tag)::value) {
                if (mixed_route) {
                    emit_route(std::integral_constant<int, Cache::mixed_predecoded ? 1 : 2>{});
                    return;
                }
            }
            emit_route(std::integral_constant<int, 0>{});
          };
          if constexpr (Cache::mixed_predecoded && Waves == 1 &&
                        !decltype(split_tag)::value && decltype(grid_tag)::value &&
                        !decltype(tile_tag)::value) {
              if (arch936) {
                  emit(std::integral_constant<int, 4>{});
                  return;
              }
          }
          emit(std::integral_constant<int, 1>{});
        };
        if constexpr (Cache::packed_nan) {
            if constexpr (decltype(split_tag)::value) {
                if (row8_grid) {
                    launch_grid(std::true_type{}, std::true_type{});
                    return;
                }
            }
            if (grid3d) {
                launch_grid(std::true_type{}, std::false_type{});
                return;
            }
        }
        launch_grid(std::false_type{}, std::false_type{});
    };
    // gfx936 BF16 unsplit lookahead: 512-row grids gain, 1024-row grids lose (80 CUs).
    const bool ahead = !arch938 && splits == 1 && programs <= 8 * int64_t(device_cus);
    auto launch_waves = [&](auto split_tag) {
        auto launch_lookup = [&](auto wave_tag, auto ahead_tag) {
            if (k_cache.size(1) <= int64_t(std::numeric_limits<int>::max()))
                launch(wave_tag, split_tag, ahead_tag,
                       hytlass::FastDivmod(static_cast<int>(k_cache.size(1))));
            else
                launch(wave_tag, split_tag, ahead_tag, LargePageDivmod{});
        };
        auto launch_ahead = [&](auto wave_tag) {
            if constexpr (!Cache::fp8 && !decltype(split_tag)::value) {
                if (ahead) {
                    launch_lookup(wave_tag, std::true_type{});
                    return;
                }
            }
            launch_lookup(wave_tag, std::false_type{});
        };
        if (one_wave)
            launch_ahead(std::integral_constant<int, 1>{});
        else
            launch_ahead(std::integral_constant<int, 4>{});
    };
    if (splits > 1)
        launch_waves(std::true_type{});
    else
        launch_waves(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1) {
        if (tiny_target) {
            launch_merge_tiny32(partial_ptr, lse_ptr, out.data_ptr<at::BFloat16>(),
                                q.size(0), q.size(1), out.stride(0), out.stride(1), stream);
        } else {
            launch_merge(partial_ptr, lse_ptr, out.data_ptr<at::BFloat16>(), q.size(0), q.size(1),
                         splits, out.stride(0), out.stride(1), stream);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

} // namespace sparse_gqa_paged_detail
