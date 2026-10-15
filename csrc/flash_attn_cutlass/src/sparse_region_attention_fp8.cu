// E4M3FN KV support for both sparse-region public attention interfaces.
#include "sparse_region_attention.h"
#include <cstring>

namespace sparse_region_detail {
// E4M3FN cache bytes widen exactly to BF16 MMAC operands. gfx938 has a
// packed converter; gfx936 uses the exact software mapping (including signed
// zero and the two NaN encodings) rather than uint82half's E4M3 fallback.
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_decode_fp8x4(uint32_t bits) {
    uint32_t words[4];
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(bits, false);
    const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(bits, true);
    __builtin_memcpy(words, &lo, 8);
    __builtin_memcpy(words + 2, &hi, 8);
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte = static_cast<uint8_t>(bits >> (8 * i));
        const uint32_t magnitude = byte & 0x7fu;
        const uint32_t encoded = magnitude << 20;
        float value;
        __builtin_memcpy(&value, &encoded, sizeof(value));
        value *= 0x1p120f;
        uint32_t raw;
        __builtin_memcpy(&raw, &value, sizeof(raw));
        raw |= (byte & 0x80u) << 24;
        if (magnitude == 0x7fu) raw = 0x7fc00000u;
        words[i] = raw;
    }
#endif
    const sparse_attention_detail::mmac::Words2 packed = {
        (words[0] >> 16) | (words[1] & 0xffff0000u),
        (words[2] >> 16) | (words[3] & 0xffff0000u)};
    sparse_attention_detail::mmac::Half4 result;
    __builtin_memcpy(&result, &packed, sizeof(result));
    return result;
}

__device__ __forceinline__ uint32_t region_load_fp8x4(const void *base, int64_t offset) {
    uint32_t bits;
    __builtin_memcpy(&bits, static_cast<const uint8_t*>(base) + offset, sizeof(bits));
    return bits;
}

__device__ __forceinline__ uint32_t region_transpose_v_fp8(uint32_t bits, int g, int r) {
    return sparse_attention_detail::mmac::transpose_v_fp8(bits, g, r);
}


template<>
__device__ __forceinline__ RegionV4<true> region_load_v4<true>(const void *base, int64_t offset) {
    return region_load_fp8x4(base, offset);
}
template<>
__device__ __forceinline__ RegionK8<true> region_load_k8<true>(const void *base, int64_t offset) {
    return {region_load_fp8x4(base, offset), region_load_fp8x4(base, offset + 4)};
}
template<>
__device__ __forceinline__ sparse_attention_detail::mmac::Vec8 region_shuffle_k8<true>(RegionK8<true> value, int source) {
    const uint32_t lo = __shfl(value[0], source, 64);
    const uint32_t hi = __shfl(value[1], source, 64);
    return {region_decode_fp8x4(lo), region_decode_fp8x4(hi)};
}
template<>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_transpose_v4<true>(
    RegionV4<true> value, int g, int r) {
    return region_decode_fp8x4(region_transpose_v_fp8(value, g, r));
}

// Same exact widening as region_decode_fp8x4. Every E4M3FN value, NaN
// included, widens to an FP32 word with zero low 16 bits, so the odd word
// needs no mask: one v_or_b32_sdwa per pair. (v_perm_b32 on bit_cast float2
// elements was miscompiled by DTK: both operands became the even word.)
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_decode_fp8x4_perm(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    uint32_t words[4];
    const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(bits, false);
    const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(bits, true);
    __builtin_memcpy(words, &lo, 8);
    __builtin_memcpy(words + 2, &hi, 8);
    const sparse_attention_detail::mmac::Words2 packed = {
        (words[0] >> 16) | words[1], (words[2] >> 16) | words[3]};
    sparse_attention_detail::mmac::Half4 result;
    __builtin_memcpy(&result, &packed, sizeof(result));
    return result;
#else
    return region_decode_fp8x4(bits);
#endif
}

template<int Opt>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_decode_fp8x4_opt(uint32_t bits) {
    if constexpr ((Opt & kFp8Perm) != 0) return region_decode_fp8x4_perm(bits);
    else return region_decode_fp8x4(bits);
}

// 4x4 byte transpose inside each quad of r; the lane-source step is separate.
__device__ __forceinline__ uint32_t region_quad_transpose_fp8(uint32_t word, int r) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
    uint32_t other = __builtin_amdgcn_mov_dpp(word, 0xb1, 15, 15, false);
    word = __builtin_amdgcn_perm(other, word, (r & 1) ? 0x03070105u : 0x06020400u);
    other = __builtin_amdgcn_mov_dpp(word, 0x4e, 15, 15, false);
    return __builtin_amdgcn_perm(other, word, (r & 2) ? 0x03020706u : 0x05040100u);
#else
    uint32_t other = __shfl_xor(word, 1, 64);
    word = (r & 1) ? ((word & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                   : ((word & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
    other = __shfl_xor(word, 2, 64);
    return (r & 2) ? ((word & 0xffff0000u) | (other >> 16))
                   : ((word & 0xffffu) | (other << 16));
#endif
}

template<int Opt>
__device__ sparse_attention_detail::mmac::Vec8 region_fp8_key(sparse_attention_detail::mmac::Words2 value, int source) {
    value[0] = __shfl(value[0], source, 64);
    value[1] = __shfl(value[1], source, 64);
    return {region_decode_fp8x4_opt<Opt>(value[0]), region_decode_fp8x4_opt<Opt>(value[1])};
}

template<int Opt>
__device__ sparse_attention_detail::mmac::Half4 region_fp8_value(uint32_t value, int g, int r) {
    // Before the quad transpose, lane (g,r) needs chunk r>>2 of token
    // 4g+(r&3); P stays in the MMAC C layout on this path.
    value = __shfl(value, (4 * g + (r & 3)) * 4 + (r >> 2), 64);
    return region_decode_fp8x4_opt<Opt>(region_quad_transpose_fp8(value, r));
}

// InLaneV value rows: 12 bytes at tokens[j] * vs0 + group * vs1 + 12r.
template<bool Masked, class Token>
__device__ __forceinline__ void region_fp8_load_lane_values(const RegionArgs &a, const Token (&tokens)[4],
    int64_t group, int r, uint32_t (&words)[12]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool live = !Masked || tokens[j] >= 0;
        const int64_t offset = live ? int64_t(tokens[j]) * a.vs0 + group * a.vs1 + r * 12 : 0;
        uint32_t row[3] = {0, 0, 0};
        if (live)
            __builtin_memcpy(row, __builtin_assume_aligned(static_cast<const uint8_t*>(a.v) + offset, 4), 12);
        words[3 * j] = row[0];
        words[3 * j + 1] = row[1];
        words[3 * j + 2] = row[2];
    }
}

// Offset32 form: 32-bit offsets from an SGPR group base.
template<bool Masked, class Token>
__device__ __forceinline__ void region_fp8_load_lane_values32(const uint8_t *base, const Token (&tokens)[4],
    uint32_t stride, int r, uint32_t (&words)[12]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool live = !Masked || tokens[j] >= 0;
        const uint32_t offset = __umul24(uint32_t(tokens[j]), stride) + uint32_t(r * 12);
        uint32_t row[3] = {0, 0, 0};
        if (live) __builtin_memcpy(row, __builtin_assume_aligned(base + uint64_t(offset), 4), 12);
        words[3 * j] = row[0];
        words[3 * j + 1] = row[1];
        words[3 * j + 2] = row[2];
    }
}

// Element j of each operand is tile token g+4j. Pair skips the byte perm by
// pairing the converted tokens directly.
template<int K, bool Pair>
__device__ __forceinline__ void region_fp8_lane_values(const uint32_t (&words)[12],
    sparse_attention_detail::mmac::Half4 (&out)[4]) {
    sparse_attention_detail::mmac::Words2 packed[4];
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    if constexpr (Pair) {
        // f[j][b]: FP32 word of token g+4j, dim 12r+4K+b (low 16 bits zero).
        uint32_t f[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(words[3 * j + K], false);
            const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(words[3 * j + K], true);
            __builtin_memcpy(f[j], &lo, 8);
            __builtin_memcpy(f[j] + 2, &hi, 8);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b)
            packed[b] = {(f[0][b] >> 16) | f[1][b], (f[2][b] >> 16) | f[3][b]};
    } else {
#endif
    uint16_t h[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sparse_attention_detail::mmac::Half4 half = region_decode_fp8x4(words[3 * j + K]);
        __builtin_memcpy(h[j], &half, sizeof(half));
    }
#pragma unroll
    for (int b = 0; b < 4; ++b)
        packed[b] = {h[0][b] | (uint32_t(h[1][b]) << 16), h[2][b] | (uint32_t(h[3][b]) << 16)};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    }
#endif
    __builtin_memcpy(out, packed, sizeof(packed));
}
} // namespace sparse_region_detail

namespace {
using namespace sparse_region_detail;
__global__ void region_predecode_fp8_kv_kernel(const uint32_t *k, const uint32_t *v,
    sparse_attention_detail::mmac::Half4 *decoded_k, sparse_attention_detail::mmac::Half4 *decoded_v,
    int64_t words) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) {
        decoded_k[i] = region_decode_fp8x4(k[i]);
        decoded_v[i] = region_decode_fp8x4(v[i]);
    }
}

} // namespace

// Offset32 needs u24 token/stride products and 32-bit byte offsets.
static bool region_fp8_offset32_fits(const sparse_region_detail::RegionArgs &a) {
    constexpr int64_t kU24 = int64_t{1} << 24, kU32 = int64_t{1} << 32;
    return a.tokens < kU24 && a.ks0 >= 0 && a.ks0 < kU24 && a.vs0 >= 0 && a.vs0 < kU24 &&
           a.tokens * a.ks0 + 256 <= kU32 && a.tokens * a.vs0 + 256 <= kU32;
}

template<bool Decode>
static void launch_fp8_opt(sparse_region_detail::RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    using namespace sparse_region_detail;
    constexpr int Opt = kFp8Opt;
    // Out-of-range strides fall back to the same options without Offset32.
    if (!region_fp8_offset32_fits(a)) {
        launch_region_kernel<Decode, true, Opt & ~kFp8Offset32>(a, grid, stream, parts, dim);
        return;
    }
    launch_region_kernel<Decode, true, Opt>(a, grid, stream, parts, dim);
}

static void launch_prefill_fp8(
    sparse_region_detail::RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    launch_fp8_opt<false>(a, grid, stream, parts, dim);
}

static void launch_decode_fp8(
    sparse_region_detail::RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    launch_fp8_opt<true>(a, grid, stream, parts, dim);
}

std::vector<at::Tensor> fa_sparse_attention_prefill_fp8(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &packed, const at::Tensor &counts, int64_t groups,
    int64_t region, c10::optional<double> scale, double k_scale, double v_scale) {
    using namespace sparse_region_detail;
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                v.scalar_type() == at::ScalarType::Float8_e4m3fn, "FP8 attention requires E4M3FN K/V");
    // gfx936 benefits from decoding a heavily reused FP8 cache once. Bound
    // the temporary BF16 K/V storage and retain the direct path on OOM.
    if (q.is_cuda() && k.is_cuda() && v.is_cuda() &&
        q.device() == k.device() && q.device() == v.device() &&
        q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && packed.dim() == 2 &&
        q.size(0) >= 4096 &&
        k.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        v.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        q.scalar_type() == at::kBFloat16 && q.size(2) == 192 &&
        k.sizes() == v.sizes() && k.is_contiguous() && v.is_contiguous() &&
        k.numel() > 0 && k.numel() % 4 == 0 &&
        k.numel() <= (64LL << 20) / 4 &&
        double(q.size(0)) * double(std::min<int64_t>(k.size(0), packed.size(1) * region)) >=
            double(k.size(0)) * 16.0 &&
        k_scale == 1.0 && v_scale == 1.0) {
        at::hip::HIPGuardMasqueradingAsCUDA guard{static_cast<char>(q.get_device())};
        hipDeviceProp_t properties{};
        C10_HIP_CHECK(hipGetDeviceProperties(&properties, q.get_device()));
        if (std::strncmp(properties.gcnArchName, "gfx936", 6) == 0) {
            at::Tensor decoded_k, decoded_v;
            try {
                decoded_k = at::empty(k.sizes(), q.options());
                decoded_v = at::empty(v.sizes(), q.options());
            } catch (const c10::OutOfMemoryError&) {
                decoded_k = at::Tensor();
                decoded_v = at::Tensor();
            }
            if (decoded_k.defined() && decoded_v.defined()) {
                const int64_t words = k.numel() / 4;
                const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
                region_predecode_fp8_kv_kernel<<<sparse_attention_detail::grid_size((words + 255) / 256), 256, 0, stream>>>(
                    reinterpret_cast<const uint32_t*>(k.data_ptr()),
                    reinterpret_cast<const uint32_t*>(v.data_ptr()),
                    reinterpret_cast<sparse_attention_detail::mmac::Half4*>(decoded_k.data_ptr()),
                    reinterpret_cast<sparse_attention_detail::mmac::Half4*>(decoded_v.data_ptr()), words);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return run_region_attention<false>(q, decoded_k, decoded_v, packed, counts,
                    c10::nullopt, groups, region, scale, 1.0, 1.0,
                    launch_prefill_bf16, region_default_parts);
            }
        }
    }
    return run_region_attention<true>(q, k, v, packed, counts, c10::nullopt, groups, region,
        scale, k_scale, v_scale, launch_prefill_fp8, region_default_parts);
}


std::vector<at::Tensor> fa_sparse_attention_decode_fp8(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &packed, const at::Tensor &counts, const at::Tensor &seqlens,
    int64_t groups, int64_t region, c10::optional<double> scale, int64_t /*num_splits*/,
    double k_scale, double v_scale) {
    using namespace sparse_region_detail;
    // Run inside the validated launcher so a temporary BF16 cache cannot hide
    // an invalid original dtype, shape, alignment, scale or metadata layout.
    const auto launch = [&](RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
        // gfx936's software E4 decoder can dominate a heavily reused cache.
        // Use only static shape information; never read device metadata on host.
        // Keep small batches, sparse/large pools, non-unit scales and strided KV
        // on the direct path. The combined temporary K/V storage is <=64 MiB.
        const bool reuse_cache = a.rows >= 128 && a.heads_per_group == 16 && a.region == 8 &&
            k_scale == 1.0 && v_scale == 1.0 && k.is_contiguous() && v.is_contiguous() &&
            k.numel() > 0 && k.numel() <= (64LL << 20) / 4 &&
            double(a.rows) * std::min(double(a.tokens), double(a.topk) * a.region) >=
                double(a.tokens) * 8.0;
        if (reuse_cache) {
            hipDeviceProp_t properties{};
            C10_HIP_CHECK(hipGetDeviceProperties(&properties, q.get_device()));
            if (std::strncmp(properties.gcnArchName, "gfx936", 6) == 0) {
                at::Tensor decoded_k, decoded_v;
                try {
                    decoded_k = at::empty(k.sizes(), q.options());
                    decoded_v = at::empty(v.sizes(), q.options());
                } catch (const c10::OutOfMemoryError&) {
                    decoded_k = at::Tensor();
                    decoded_v = at::Tensor();
                }
                if (decoded_k.defined() && decoded_v.defined()) {
                    const int64_t words = k.numel() / 4;
                    region_predecode_fp8_kv_kernel<<<sparse_attention_detail::grid_size((words + 255) / 256), 256, 0, stream>>>(
                        reinterpret_cast<const uint32_t*>(k.data_ptr()),
                        reinterpret_cast<const uint32_t*>(v.data_ptr()),
                        reinterpret_cast<sparse_attention_detail::mmac::Half4*>(decoded_k.data_ptr()),
                        reinterpret_cast<sparse_attention_detail::mmac::Half4*>(decoded_v.data_ptr()), words);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    // Storage changes, but unit scales and exact E4->BF16
                    // widening preserve the attention math and split plan.
                    a.k = decoded_k.data_ptr();
                    a.v = decoded_v.data_ptr();
                    launch_decode_bf16(a, grid, stream, parts, dim);
                    return;
                }
            }
        }
        launch_decode_fp8(a, grid, stream, parts, dim);
    };
    return run_region_attention<true>(q, k, v, packed, counts, seqlens, groups, region,
        scale, k_scale, v_scale, launch, region_decode_parts);
}
