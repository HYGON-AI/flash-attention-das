#pragma once
// Shared paged attention math. Storage policies keep BF16 and FP8 in separate TUs.
#include "sparse_attention.h"
#include <type_traits>

namespace sparse_gqa_paged_detail {

__device__ __forceinline__ sparse_attention_detail::mmac::Half4 paged_lane_operand(
    sparse_attention_detail::mmac::Half4 a, sparse_attention_detail::mmac::Half4 b,
    sparse_attention_detail::mmac::Half4 c, sparse_attention_detail::mmac::Half4 d,
    int component) {
    using namespace sparse_attention_detail::mmac;
    Words2 words[4];
    __builtin_memcpy(words, &a, 8);
    __builtin_memcpy(words + 1, &b, 8);
    __builtin_memcpy(words + 2, &c, 8);
    __builtin_memcpy(words + 3, &d, 8);
    const int half = component / 2;
    Words2 packed;
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    const uint32_t selector = (component & 1) ? 0x07060302u : 0x05040100u;
    packed = {__builtin_amdgcn_perm(words[1][half], words[0][half], selector),
              __builtin_amdgcn_perm(words[3][half], words[2][half], selector)};
#else
    packed = (component & 1)
        ? Words2{(words[0][half] >> 16) | (words[1][half] & 0xffff0000u),
                 (words[2][half] >> 16) | (words[3][half] & 0xffff0000u)}
        : Words2{(words[0][half] & 0xffffu) | (words[1][half] << 16),
                 (words[2][half] & 0xffffu) | (words[3][half] << 16)};
#endif
    Half4 result;
    __builtin_memcpy(&result, &packed, 8);
    return result;
}

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

// Compiler BF16 packing plus an original-bit NaN fix preserves c10 rounding.
// gfx936 lowers this to integer operations, not native BF16 conversion.
template<bool Compiler936, bool Fp8>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 paged_pair_probability(
    sparse_attention_detail::mmac::Float4 values) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    if constexpr (Compiler936) {
        using F2 = __attribute__((ext_vector_type(2))) float;
        using B2 = __attribute__((ext_vector_type(2))) __bf16;
        const F2 a = {values[0], values[1]}, b = {values[2], values[3]};
        const B2 lo = __builtin_convertvector(a, B2);
        const B2 hi = __builtin_convertvector(b, B2);
        uint32_t source_bits[4], packed[2];
        __builtin_memcpy(source_bits, &values, 16);
        __builtin_memcpy(packed, &lo, 4);
        __builtin_memcpy(packed + 1, &hi, 4);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if ((source_bits[e] & 0x7fffffffu) > 0x7f800000u) {
                constexpr uint32_t nan = 0x7fc0u;
                const int shift = 16 * (e & 1);
                packed[e / 2] = (packed[e / 2] & ~(0xffffu << shift)) | (nan << shift);
            }
        }
        sparse_attention_detail::mmac::Half4 result;
        __builtin_memcpy(&result, packed, 8);
        return result;
    }
#endif
    return paged_probability<Fp8>(values);
}

template<class Cache> struct paged_page3200_policy : std::false_type {};

void launch_merge(const float *partial, const float *lse, at::BFloat16 *out, int64_t rows,
                  int64_t heads, int splits, int64_t os0, int64_t os1, hipStream_t stream);

void launch_merge_tiny32(const float *partial, const float *lse, at::BFloat16 *out,
                          int64_t rows, int64_t heads, int64_t os0, int64_t os1,
                          hipStream_t stream);

struct Bf16Cache {
    using Element = at::BFloat16;
    using KFragment = sparse_attention_detail::mmac::Vec8;
    using VFragment = sparse_attention_detail::mmac::Half4;
    static constexpr bool fp8 = false;
    static constexpr bool packed_nan = false;
    static constexpr bool shared_offsets = false;
    static constexpr bool e5 = false;
    static constexpr bool tuned936 = false;
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
        return paged_lane_operand(values[chunk], values[4 + chunk],
                                  values[8 + chunk], values[12 + chunk], d % 4);
    }
    __device__ static __forceinline__ float decode(Element v) { return float(v); }
};

using sparse_attention_detail::CacheStrides;
constexpr int kThreads = 64;
constexpr int kOutputTile = 256;
constexpr int kSharedStride = 260;
constexpr float kLog2E = 1.4426950408889634f;

// Opt-in policy: the host restricts this BF16 layout to small gfx936/page64
// group-6 grids. Large grids keep the original BF16 layout.
struct Bf16CacheSmall936 : Bf16Cache {
    static constexpr bool tuned936 = true;
};

template<class Cache, bool Masked>
__device__ __forceinline__ void paged_load_lane_values(
    const typename Cache::Element *v, int page, int offset, int g, int r,
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
    const typename Cache::Element *v, const int64_t *bases, int r,
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
template <class Cache, int Waves, bool Split = false, int PageSize = 0, int Group = 6,
          bool Ahead = false, bool Grid3D = false, bool TileRows = false>
__global__ __launch_bounds__(Waves * 64) void sparse_gqa_paged_attention_mmac_kernel(
    const at::BFloat16 *q, const typename Cache::Element *k, const typename Cache::Element *v,
    const int32_t *indices, const int32_t *table, const int32_t *requests, at::BFloat16 *out,
    float *partial, float *lse, int64_t rows, int64_t heads, int64_t kv_heads, int64_t pages,
    int64_t page_size, int64_t table_width, int64_t num_requests, int64_t topk, int64_t qs0,
    int64_t qs1, CacheStrides ks, CacheStrides vs, int64_t is0, int64_t ts0, int64_t os0,
    int64_t os1, int splits, const float *k_scale_ptr, const float *v_scale_ptr) {
    using namespace sparse_attention_detail::mmac;
    // Keep the measured storage/architecture choices local to this kernel.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool InLaneV = true;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool InLaneV = Cache::fp8 || Group == 12 || (Cache::tuned936 && PageSize == 64);
#else
    constexpr bool InLaneV = false;
#endif
    static_assert(Group == 6 || Group == 12, "validated MMAC head groups only");
    // gfx938 FP8 K: one 16-byte load covers two MMAC k-steps per lane; Q uses
    // the same dimension permutation so every product pairs matching dims.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool WideK = Cache::fp8;
#else
    constexpr bool WideK = false;
#endif
    using Words4 = __attribute__((vector_size(16))) uint32_t;
    const float k_scale = Cache::fp8 ? *k_scale_ptr : 1.0f;
    const bool is_split = (Cache::fp8 || Split) ? Split : splits > 1;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    // W1 target split uses LateV; W4 tiny grids retain EarlyV.
    const bool load_v_first = (Cache::packed_nan && Group == 12 && PageSize == 64 && Waves == 1)
        ? false : is_split || (Group == 12 && !Cache::fp8);
#else
    const bool load_v_first = is_split;
#endif
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    const int r = lane % 16, g = lane / 16;
    static_assert(!TileRows || Grid3D, "row tiling requires the 3D grid");
    const int64_t row = TileRows ? int64_t(blockIdx.y) * 8 + (blockIdx.x & 7)
                                : (Grid3D ? blockIdx.x : blockIdx.x % rows);
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
    const int64_t tiles = (topk + 15) / 16;
    const int shift = __ffs(splits * Waves) - 1;
    const int64_t first = ((split * Waves + wave) * tiles) >> shift;
    const int64_t last = ((split * Waves + wave + 1) * tiles) >> shift;
    const int64_t page_span = PageSize > 0 ? PageSize : page_size;
    // gfx938: FP8 and BF16 split kernels only; its BF16 unsplit (prefill)
    // kernel measured slower with these and keeps the per-tile form.
    // gfx936 group 12: every kernel. Lookahead costs its BF16 unsplit kernel
    // above eight CTAs per CU, so the host enables it there through Ahead.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    constexpr bool Tuned = Cache::fp8 || Split;
    constexpr bool Lookahead = Tuned;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    constexpr bool Tuned = Group == 12 || (Cache::tuned936 && PageSize == 64);
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
    constexpr bool UnmaskedHeads = Cache::packed_nan && Group == 12 &&
                                   (PageSize == 64 || PageSize == 3200);
#else
    constexpr bool UnmaskedHeads = false;
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
    // Combine two 16-token QK tiles before one softmax/PV-state update.
    // Decode keeps its original loop and independent split/merge choices.
    constexpr bool PairTiles = Cache::fp8 && Cache::e5 && Cache::packed_nan &&
                               Group == 12 && PageSize == 64 && Waves == 1 && !Split;
#else
    constexpr bool PairTiles = false;
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
        if (token >= 0 && token / page_span < table_width) {
            page = table[request * ts0 + token / page_span];
            offset = token % page_span;
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
    if constexpr (PairTiles) {
    // Preserve 16-token partition first/last exactly; pair only inside it.
    for (int64_t tile = first; tile < last; tile += 2) {
        Float4 pair_scores[2];
        int64_t pair_v_bases[2][4];
        uint64_t pair_live[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int64_t current = tile + half;
            int page = ahead_page >= 0 && ahead_page < pages ? ahead_page : -1;
            const int offset = ahead_offset;
            ahead_page = lookup_page(ahead_token, ahead_offset);
            ahead_token = fetch_token(current + 2);
            // fetch_token/lookup_page already mask a final absent half. Keep
            // the explicit bound for clarity and arbitrary partition tails.
            if (current >= last) page = -1;
            const uint64_t live = __ballot(page >= 0);
            pair_live[half] = live;
            int64_t k_token = -1, v_token = -1;
            if (page >= 0) {
                k_token = int64_t(page) * ks.block + offset * ks.token + kv_head * ks.head;
                if constexpr (Cache::shared_offsets)
                    v_token = k_token;
                else
                    v_token = int64_t(page) * vs.block + offset * vs.token + kv_head * vs.head;
            }
            const int64_t kb = __shfl(k_token, lane / 4, 64);
            const bool valid = kb >= 0;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                pair_v_bases[half][e] = __shfl(v_token, g + 4 * e, 64);
            Float4 scores = {0, 0, 0, 0};
            if (live == uint64_t{0xffff}) {
                typename Cache::KFragment window[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    window[i] = Cache::load_k(k + kb + i * 32 + (lane % 4) * 8);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const Vec8 key = Cache::shuffle_k(window[i], r * 4 + g);
                    dot(query[i].x, key.x, scores);
                    dot(query[i].y, key.y, scores);
                }
            } else if (live != 0) {
                typename Cache::KFragment window[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    window[i] = valid ? Cache::load_k(k + kb + i * 32 + (lane % 4) * 8)
                                      : typename Cache::KFragment{};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const Vec8 key = Cache::shuffle_k(window[i], r * 4 + g);
                    dot(query[i].x, key.x, scores);
                    dot(query[i].y, key.y, scores);
                }
            }
#pragma unroll
            for (int e = 0; e < 4; ++e)
                scores[e] = pair_v_bases[half][e] >= 0
                                ? scores[e] * ((0.0625f * kLog2E) * k_scale)
                                : -1.0e20f;
            pair_scores[half] = scores;
        }
        if ((pair_live[0] | pair_live[1]) == 0)
            continue;
        const Float4 a = pair_scores[0], b = pair_scores[1];
        float maximum = fmaxf(fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3])),
                              fmaxf(fmaxf(b[0], b[1]), fmaxf(b[2], b[3])));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 64));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 32, 64));
        maximum = fmaxf(maximum, row_max);
        const float alpha = sparse_attention_detail::mmac::exp2(row_max - maximum);
        Float4 probabilities[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                probabilities[half][e] = pair_v_bases[half][e] >= 0
                    ? sparse_attention_detail::mmac::exp2(pair_scores[half][e] - maximum)
                    : 0.0f;
        }
        const Float4 pa = probabilities[0], pb = probabilities[1];
        float sum = ((pa[0] + pa[1]) + (pa[2] + pa[3])) +
                    ((pb[0] + pb[1]) + (pb[2] + pb[3]));
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        row_sum = row_sum * alpha + sum;
        row_max = maximum;
        if (__ballot(alpha != 1.0f) != 0) {
#pragma unroll
            for (int d = 0; d < 16; ++d) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
                using F2 = __attribute__((ext_vector_type(2))) float;
                F2 lo, hi;
                __builtin_memcpy(&lo, &pv[d], 8);
                __builtin_memcpy(&hi, reinterpret_cast<const char*>(&pv[d]) + 8, 8);
                const F2 scale = {alpha, alpha};
                lo = __builtin_hcu_pk_mul_f32(lo, scale);
                hi = __builtin_hcu_pk_mul_f32(hi, scale);
                __builtin_memcpy(&pv[d], &lo, 8);
                __builtin_memcpy(reinterpret_cast<char*>(&pv[d]) + 8, &hi, 8);
#else
                pv[d] *= alpha;
#endif
            }
        }
        // Only one V-fragment array and one PV accumulator are required.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (pair_live[half] != 0) {
                typename Cache::VFragment values[16];
                if (pair_live[half] == uint64_t{0xffff})
                    paged_load_lane_bases<Cache, false>(v, pair_v_bases[half], r, values);
                else
                    paged_load_lane_bases<Cache, true>(v, pair_v_bases[half], r, values);
                const Half4 p = paged_pair_probability<true, Cache::fp8>(probabilities[half]);
#pragma unroll
                for (int d = 0; d < 16; ++d)
                    dot(p, Cache::lane_v(values, d), pv[d]);
            }
        }
    }
    } else {
    for (int64_t tile = first; tile < last; ++tile) {
        int page = -1, offset = 0;
        if constexpr (Lookahead) {
            page = ahead_page >= 0 && ahead_page < pages ? ahead_page : -1;
            offset = ahead_offset;
            ahead_page = lookup_page(ahead_token, ahead_offset);
            ahead_token = fetch_token(tile + 2);
        } else if (g == 0 && tile * 16 + r < topk && request_valid) {
            const int token = indices[row * is0 + tile * 16 + r];
            if (token >= 0 && token / page_span < table_width) {
                page = table[request * ts0 + token / page_span];
                if (page < 0 || page >= pages)
                    page = -1;
                offset = token % page_span;
            }
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
                        const Half4 packed = paged_pair_probability<PairTiles, Cache::fp8>(value);
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
    const at::BFloat16 *q, const typename Cache::Element *k, const typename Cache::Element *v,
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
    const int64_t kv_head = head / (heads / kv_heads);
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

template <class Cache>
at::Tensor run(const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &v_cache,
               const at::Tensor &logical_indices, const at::Tensor &block_table,
               const at::Tensor &token_to_req, c10::optional<at::Tensor> out_,
               const float *k_scale_ptr = nullptr, const float *v_scale_ptr = nullptr,
               const hipDeviceProp_t *known_props = nullptr) {
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
    TORCH_CHECK(q.size(1) % k_cache.size(2) == 0, "query heads must be divisible by KV heads");
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
    at::hip::HIPGuardMasqueradingAsCUDA guard{static_cast<char>(q.get_device())};
    auto out = out_.has_value() ? *out_ : at::empty_like(q);
    check_device(out, q, "out");
    TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == q.scalar_type() &&
                    out.stride(2) == 1,
                "out must match Q shape/dtype/device with contiguous head dimension");
    if (q.size(0) == 0)
        return out;
    const int64_t programs = q.size(0) * k_cache.size(2);
    const int64_t group = q.size(1) / k_cache.size(2);
    bool fast = dim == 256 && (group == 6 || group == 12) && aligned8(q) && aligned8(k_cache) && aligned8(v_cache) &&
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
        one_wave = (Cache::fp8 && (programs >= 128 || (group == 12 && programs <= 4))) || arch938;
        device_cus = std::max(1, props.multiProcessorCount);
        for (int i = 0; i < 3; ++i) grid_limits[i] = props.maxGridSize[i];
    }
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (!fast) {
        const int64_t grid = q.size(0) * q.size(1) * ((dim + kOutputTile - 1) / kOutputTile);
        sparse_gqa_paged_attention_generic_kernel<Cache><<<grid_size(grid), kThreads, 0, stream>>>(
            q.data_ptr<at::BFloat16>(),
            reinterpret_cast<const typename Cache::Element *>(k_cache.data_ptr()),
            reinterpret_cast<const typename Cache::Element *>(v_cache.data_ptr()),
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
    const int64_t tiles = (logical_indices.size(1) + 15) / 16;
    const bool target936 = arch936 && Cache::packed_nan && group == 12 && k_cache.size(1) == 64;
    const bool tiny_target = target936 && programs <= 2 && tiles >= 128 && tiles <= 256;
    int splits = 1;
    // Group-12 gfx938 FP8: validated page-16 / 129-tile tiny grids
    // favor four waves and smaller external merge fan-in. Keep V8
    // scheduling when the four-wave split CTAs exceed one per CU.
    if (group == 12 && Cache::fp8 && arch938 && k_cache.size(1) == 16 && tiles == 129) {
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
        if (group == 12 && arch938 && populated && programs <= 4 * int64_t(device_cus)) {
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
    if (arch936 && Cache::e5 && group == 12 && k_cache.size(1) == 64 &&
        programs > 2 * int64_t(device_cus) && programs <= 4 * int64_t(device_cus))
        splits = std::min(splits, 4);
    // Target G12 W4 keeps wave0 in registers. Raw merge state uses
    // 37824 bytes of LDS, still at most one CTA per 64-KiB CU.
    // Retain the measured one-CTA-round schedule below and the historical
    // schedule above one base CTA/CU (81..112 programs regressed).
    if (arch936 && Cache::packed_nan && group == 12 && k_cache.size(1) == 64 &&
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
    const bool grid3d = arch936 && Cache::packed_nan && group == 12 &&
                        (k_cache.size(1) == 64 || paged_page3200_policy<Cache>::value) &&
                        q.size(0) <= grid_limits[0] &&
                        k_cache.size(2) <= grid_limits[1] && splits <= grid_limits[2];
    const bool row8_grid = target936 && splits > 1 && q.size(0) % 8 == 0 &&
                           k_cache.size(2) <= grid_limits[0] / 8 &&
                           q.size(0) / 8 <= grid_limits[1] && splits <= grid_limits[2];
    const auto flat_grid = grid_size(programs * splits);
    const dim3 grid = row8_grid ? dim3(k_cache.size(2) * 8, q.size(0) / 8, splits)
                               : (grid3d ? dim3(q.size(0), k_cache.size(2), splits)
                                         : dim3(flat_grid));
    auto launch = [&](auto wave_tag, auto split_tag, auto page_tag, auto ahead_tag) {
        constexpr int Waves = decltype(wave_tag)::value;
        auto run_group = [&](auto group_tag) {
            constexpr int Group = decltype(group_tag)::value;
            constexpr bool TargetShared = Cache::packed_nan && Group == 12 &&
                                          (decltype(page_tag)::value == 64 || decltype(page_tag)::value == 3200);
            constexpr int target_shared_bytes = Waves == 1 ? 0 :
                ((Waves - (TargetShared ? 1 : 0)) * Group * kSharedStride +
                 Waves * Group * (TargetShared ? 2 : 1)) * sizeof(float);
            const int shared_bytes = arch936 ? target_shared_bytes :
                (Waves == 1 ? 0 : Waves * Group * (kSharedStride + 1) * sizeof(float));
            auto launch_grid = [&](auto grid_tag, auto tile_tag) {
            sparse_gqa_paged_attention_mmac_kernel<Cache, Waves, decltype(split_tag)::value,
                                                  decltype(page_tag)::value, Group,
                                                  decltype(ahead_tag)::value,
                                                  decltype(grid_tag)::value,
                                                  decltype(tile_tag)::value>
                <<<grid, Waves * 64, shared_bytes, stream>>>(
                    q.data_ptr<at::BFloat16>(),
                    reinterpret_cast<const typename Cache::Element *>(k_cache.data_ptr()),
                    reinterpret_cast<const typename Cache::Element *>(v_cache.data_ptr()),
                    logical_indices.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                    token_to_req.data_ptr<int32_t>(), out.data_ptr<at::BFloat16>(), partial_ptr,
                    lse_ptr, q.size(0), q.size(1), k_cache.size(2), k_cache.size(0), k_cache.size(1),
                    block_table.size(1), block_table.size(0), logical_indices.size(1), q.stride(0),
                    q.stride(1), cache_strides(k_cache), cache_strides(v_cache),
                    logical_indices.stride(0), block_table.stride(0), out.stride(0), out.stride(1),
                    splits, k_scale_ptr, v_scale_ptr);
            };
            if constexpr (Cache::packed_nan && Group == 12 &&
                          (decltype(page_tag)::value == 64 || decltype(page_tag)::value == 3200)) {
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
        if (group == 12) run_group(std::integral_constant<int, 12>{});
        else run_group(std::integral_constant<int, 6>{});
    };
    // Fixed page sizes avoid runtime divide/remainder in the lookup loop: page 16
    // for one-wave FP8; page 64 except gfx938 BF16, which measured no gain.
    const bool page64 = k_cache.size(1) == 64 && (Cache::fp8 || !arch938);
    // gfx936 BF16 unsplit lookahead: 512-row grids gain, 1024-row grids lose (80 CUs).
    const bool ahead = !arch938 && group == 12 && splits == 1 && programs <= 8 * int64_t(device_cus);
    auto launch_waves = [&](auto split_tag) {
        auto launch_page = [&](auto wave_tag, auto ahead_tag) {
            if constexpr (Cache::fp8 && decltype(wave_tag)::value == 1) {
                if (k_cache.size(1) == 16) {
                    launch(wave_tag, split_tag, std::integral_constant<int, 16>{}, ahead_tag);
                    return;
                }
            }
            // Only the explicit gfx936/E5/G12 large-prefill policy emits page3200.
            if constexpr (paged_page3200_policy<Cache>::value) {
                launch(wave_tag, split_tag, std::integral_constant<int, 3200>{}, ahead_tag);
                return;
            }
            if (page64)
                launch(wave_tag, split_tag, std::integral_constant<int, 64>{}, ahead_tag);
            else
                launch(wave_tag, split_tag, std::integral_constant<int, 0>{}, ahead_tag);
        };
        auto launch_ahead = [&](auto wave_tag) {
            if constexpr (!Cache::fp8 && !decltype(split_tag)::value) {
                if (ahead) {
                    launch_page(wave_tag, std::true_type{});
                    return;
                }
            }
            launch_page(wave_tag, std::false_type{});
        };
        if (one_wave)
            launch_ahead(std::integral_constant<int, 1>{});
        else
            launch_ahead(std::integral_constant<int, 4>{});
    };
    // BF16 split kernels are specialized on gfx938 and for gfx936 group 12.
    if (splits > 1 && (Cache::fp8 || arch938 || group == 12 ||
                       (arch936 && Cache::tuned936 && k_cache.size(1) == 64)))
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
