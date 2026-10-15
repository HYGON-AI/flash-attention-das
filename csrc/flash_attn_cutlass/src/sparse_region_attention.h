#pragma once
// Shared sparse-region attention math, storage policies and host validation.
#include "sparse_attention.h"
#include <type_traits>

namespace sparse_region_detail {
// Shared by prefill and decode; defined in sparse_region_attention.cu.
void merge_internal_states(const at::Tensor &workspace, const at::Tensor &workspace_lse,
    at::Tensor &out, at::Tensor &lse, int64_t state_rows, int dim, int parts, hipStream_t stream,
    int device_cus, bool decode);
// Scores and internal LSE use log2; public LSE uses natural logarithms.
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr int kTargetWavesPerCu = 16;

__device__ __forceinline__ float region_exp2(float value) {
    return __builtin_amdgcn_exp2f(value);
}

// Half4 carries BF16 bits; MMAC accumulates in FP32.
__device__ __forceinline__ void region_dot(sparse_attention_detail::mmac::Half4 a,
    sparse_attention_detail::mmac::Half4 b, sparse_attention_detail::mmac::Float4 &acc) {
#if defined(__HIP_DEVICE_COMPILE__)
    using sparse_attention_detail::mmac::Bf4;
    acc = __builtin_hcu_mmac_f32_16x16x16_bf16(
        *reinterpret_cast<const Bf4*>(&a), *reinterpret_cast<const Bf4*>(&b), acc);
#endif
}

// Repack BF16 V fragments to the MMAC operand layout without conversion.
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_transpose_v(
    sparse_attention_detail::mmac::Half4 value, int g, int r) {
    return sparse_attention_detail::mmac::transpose_v_dpp(value, g, r);
}

// KV storage policies are specialized in sparse_region_attention_fp8.cu for FP8.
template<bool Fp8KV>
using RegionV4 = std::conditional_t<Fp8KV, uint32_t, sparse_attention_detail::mmac::Half4>;
template<bool Fp8KV>
using RegionK8 = std::conditional_t<Fp8KV, sparse_attention_detail::mmac::Words2, sparse_attention_detail::mmac::Vec8>;

template<bool Fp8KV>
__device__ __forceinline__ RegionV4<Fp8KV> region_load_v4(const void *base, int64_t offset);
template<bool Fp8KV>
__device__ __forceinline__ RegionK8<Fp8KV> region_load_k8(const void *base, int64_t offset);
template<bool Fp8KV>
__device__ __forceinline__ sparse_attention_detail::mmac::Vec8 region_shuffle_k8(RegionK8<Fp8KV> value, int source);
template<bool Fp8KV>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_transpose_v4(RegionV4<Fp8KV> value, int g, int r);

// FP8 K/V decode options, selected per architecture below and validated on
// gfx938. kFp8Opt is the only combination instantiated by the public
// launchers; the fallback drops just kFp8Offset32.
enum : int {
    kFp8Perm = 1,          // E4M3FN widened words need one v_or_b32_sdwa per pair
    kFp8CvtP = 2,          // gfx938 v_cvt_pk_bf16_f32 for P (RNE; NaN payload differs)
    kFp8NoTransposeP = 4,  // permute V token order to the C layout instead of moving P
    kFp8InLaneV = 32,      // lane loads 4 tokens x 12 dims; MMAC d column n is dim 12n+d
    kFp8PairV = 128,       // InLaneV pairs converted tokens directly; no byte perm
    kFp8PrefetchMeta = 256, // FastRegion8 loads the next tile's region words one tile ahead
    kFp8Offset32 = 512,    // InLaneV: SGPR group base plus u24 lane offsets (launcher-checked)
};
constexpr int kFp8Opt = kFp8Perm | kFp8CvtP | kFp8NoTransposeP | kFp8InLaneV | kFp8PairV |
                        kFp8PrefetchMeta | kFp8Offset32;
// gfx936 keeps software E4M3FN widening and uses the validated V layout and
// checked lane offsets; native conversion options remain gfx938-only.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
constexpr int kFp8ArchOpt = kFp8NoTransposeP | kFp8InLaneV | kFp8Offset32;
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx938__)
constexpr int kFp8ArchOpt = 0;
#else
constexpr int kFp8ArchOpt = ~0;
#endif
// BF16 K/V decode options. kFp8InLaneV, kFp8NoTransposeP, kFp8CvtP,
// kFp8PrefetchMeta, kFp8Offset32 are shared with FP8 and mean the same thing;
// kBf16EarlyV has no FP8 counterpart. Validated on gfx938.
enum : int { kBf16EarlyV = 2048 };
constexpr int kBf16Opt = kFp8CvtP | kFp8NoTransposeP | kFp8InLaneV | kFp8PrefetchMeta | kFp8Offset32 | kBf16EarlyV;
// gfx936 benefits from layout/address changes, but not the extra EarlyV
// register lifetime. Other unmeasured devices retain the V6 BF16 path.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
constexpr int kBf16ArchOpt = kFp8NoTransposeP | kFp8InLaneV | kFp8PrefetchMeta | kFp8Offset32;
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx938__)
constexpr int kBf16ArchOpt = 0;
#else
constexpr int kBf16ArchOpt = ~0;
#endif

template<int Opt>
__device__ sparse_attention_detail::mmac::Vec8 region_fp8_key(sparse_attention_detail::mmac::Words2 value, int source);
template<int Opt>
__device__ sparse_attention_detail::mmac::Half4 region_fp8_value(uint32_t value, int g, int r);

template<bool Fp8KV, int Opt>
__device__ __forceinline__ sparse_attention_detail::mmac::Vec8 region_key(RegionK8<Fp8KV> value, int source) {
    if constexpr (Fp8KV && Opt != 0) return region_fp8_key<Opt>(value, source);
    else return region_shuffle_k8<Fp8KV>(value, source);
}
template<bool Fp8KV, int Opt>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_value(RegionV4<Fp8KV> value, int g, int r) {
    if constexpr (Fp8KV && Opt != 0) return region_fp8_value<Opt>(value, g, r);
    else return region_transpose_v4<Fp8KV>(value, g, r);
}
template<int Opt>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_prob(
    sparse_attention_detail::mmac::Float4 prob, int g) {
    using namespace sparse_attention_detail::mmac;
    if constexpr (!(Opt & kFp8NoTransposeP)) prob = transpose_p(prob, g);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
    if constexpr ((Opt & kFp8CvtP) != 0) {
        using F2 = float __attribute__((ext_vector_type(2)));
        using B2 = __bf16 __attribute__((ext_vector_type(2)));
        const B2 lo = __builtin_convertvector(F2{prob[0], prob[1]}, B2);
        const B2 hi = __builtin_convertvector(F2{prob[2], prob[3]}, B2);
        const Words2 words = {__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hi)};
        Half4 result;
        __builtin_memcpy(&result, &words, sizeof(result));
        return result;
    }
#endif
    return bf16_prob(prob);
}

template<>
__device__ __forceinline__ RegionV4<false> region_load_v4<false>(const void *base, int64_t offset) {
    return *reinterpret_cast<const sparse_attention_detail::mmac::Half4*>(
        static_cast<const at::BFloat16*>(base) + offset);
}
template<>
__device__ __forceinline__ RegionK8<false> region_load_k8<false>(const void *base, int64_t offset) {
    return sparse_attention_detail::mmac::load8(static_cast<const at::BFloat16*>(base) + offset);
}
template<>
__device__ __forceinline__ sparse_attention_detail::mmac::Vec8 region_shuffle_k8<false>(RegionK8<false> value, int source) {
    return sparse_attention_detail::mmac::shuffle8(value, source);
}
template<>
__device__ __forceinline__ sparse_attention_detail::mmac::Half4 region_transpose_v4<false>(
    RegionV4<false> value, int g, int r) {
    return region_transpose_v(value, g, r);
}

// Strides are in elements; metadata rows are group-major: group * rows + row.
struct RegionArgs {
    const at::BFloat16 *q;
    const void *k, *v;
    const void *packed;
    const int32_t *counts, *seqlens;
    at::BFloat16 *out;
    float *lse;
    // FP32 normalized partial output and log2 LSE; allocated only for parts > 1.
    float *workspace, *workspace_lse;
    int64_t rows, heads, groups, tokens, topk;
    int64_t qs0, qs1, ks0, ks1, vs0, vs1, ps0;
    int region, parts, heads_per_group;
    float scale; // softmax_scale * k_scale * log2(e).
    float v_scale;
};

// Internal cross-translation-unit launchers; not Python/public APIs.
void launch_prefill_bf16(RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim);
void launch_decode_bf16(RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim);

// Prefill: int32 [valid:8 | physical:24]; decode: int64 [physical:32 | logical_start:32].
struct RegionMeta { int64_t physical; int valid; };
template<bool Decode, int StaticRegion = 0>
__device__ __forceinline__ RegionMeta load_region(RegionArgs a, int64_t row, int64_t slot, int64_t kv_len) {
    const int region = StaticRegion ? StaticRegion : a.region;
    if constexpr (Decode) {
        const uint64_t meta=static_cast<const uint64_t*>(a.packed)[row*a.ps0+slot];
        return {int64_t(meta>>32),int(min(int64_t(region),max(int64_t{0},kv_len-int64_t(meta&0xffffffffULL))))};
    } else {
        const int32_t meta=static_cast<const int32_t*>(a.packed)[row*a.ps0+slot];
        return {meta&0xffffff,meta>=0 ? meta>>24 : 0};
    }
}
// PrefetchMeta splits load_region: the raw word is loaded a tile early and
// decoded with the same layout. load_region stays whole because routing it
// through decode_region changes the codegen of every other kernel.
template<bool Decode>
using RegionWord = std::conditional_t<Decode, uint64_t, int32_t>;
template<bool Decode>
__device__ __forceinline__ RegionWord<Decode> load_region_word(const RegionArgs &a, int64_t row, int64_t slot) {
    return static_cast<const RegionWord<Decode>*>(a.packed)[row*a.ps0+slot];
}
template<bool Decode, int StaticRegion = 0>
__device__ __forceinline__ RegionMeta decode_region(const RegionArgs &a, RegionWord<Decode> meta, int64_t kv_len) {
    const int region = StaticRegion ? StaticRegion : a.region;
    if constexpr (Decode)
        return {int64_t(meta>>32),int(min(int64_t(region),max(int64_t{0},kv_len-int64_t(meta&0xffffffffULL))))};
    else
        return {meta&0xffffff,meta>=0 ? meta>>24 : 0};
}

inline int region_default_parts(int64_t programs, int64_t slots, int64_t region,
    int device_cus, int64_t /*rows*/, int64_t /*cache_tokens*/) {
    int parts = 1;
    const int64_t target = int64_t(device_cus) * kTargetWavesPerCu;
    const int64_t tiles=(slots * region + 15) / 16;
    int64_t useful = std::min<int64_t>(128, std::max<int64_t>(1, tiles / 4));
    // Prefer four tiles per part. For small base grids, allow finer
    // parts until at least eight waves/CU can be exposed; this depends
    // on device resources and work size, not benchmark case IDs.
    if (useful * programs < int64_t(device_cus) * 8)
        useful=std::min<int64_t>(128,std::max<int64_t>(1,tiles/2));
    while (parts * programs < target && parts * 2 <= useful) parts *= 2;
    return parts;
}

// Decode retains the existing occupancy heuristic for private parts.
inline int region_decode_parts(int64_t programs, int64_t slots, int64_t region, int cus,
                               int64_t rows, int64_t cache_tokens) {
    const int baseline = region_default_parts(programs, slots, region, cus, rows, cache_tokens);
    return programs >= cus ? std::min(baseline, 8) : baseline;
}

template<bool Fp8KV, class Launch, class Plan>
std::vector<at::Tensor> run_region_attention(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &packed, const at::Tensor &counts,
    const c10::optional<at::Tensor> &seqlens, int64_t groups, int64_t region,
    c10::optional<double> scale, double k_scale, double v_scale,
    Launch launch, Plan plan) {
    using namespace sparse_attention_detail;
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(1) > 0, "query must be [rows, heads, dim] on the accelerator");
    check_device(k, q, "key_cache"); check_device(v, q, "value_cache");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "region sparse attention requires BF16 Q");
    constexpr auto cache_dtype = Fp8KV ? at::ScalarType::Float8_e4m3fn : at::kBFloat16;
    TORCH_CHECK(k.scalar_type() == cache_dtype && v.scalar_type() == cache_dtype,
                Fp8KV ? "FP8 attention requires E4M3FN K/V" : "BF16 attention requires BF16 K/V");
    TORCH_CHECK(groups > 0 && q.size(1) % groups == 0 && q.size(1) / groups <= 16,
                "heads_per_group must be in [1,16]");
    TORCH_CHECK(k.dim() == 3 && k.sizes() == v.sizes() && k.size(1) == groups && k.size(2) == q.size(2),
                "K/V must have matching [tokens, num_kv_groups, head_dim] shapes");
    TORCH_CHECK(q.size(2) == 192,
                "fa_sparse_attention_prefill/decode require head_dim == 192; got ", q.size(2));
    TORCH_CHECK(region > 0 && region <= 127, "region_size must be in [1,127]");
    check_device(packed, q, "packed_regions"); check_i32(counts, q, "region_counts");
    TORCH_CHECK(packed.scalar_type() == (seqlens.has_value() ? at::kLong : at::kInt), "wrong packed_regions dtype");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == groups * q.size(0), "packed_regions has wrong row count");
    TORCH_CHECK(counts.dim() == 1 && counts.numel() == packed.size(0) && counts.is_contiguous(),
                "region_counts must be contiguous [groups * rows]");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && packed.stride(1) == 1,
                "head dimensions and packed columns must be contiguous");
    TORCH_CHECK(aligned8(q) && aligned8(k) && aligned8(v) && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0
                && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 && v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0,
                "Q/K/V rows must be aligned for 16-byte loads");
    if (seqlens.has_value()) {
        check_i32(*seqlens, q, "kv_seqlens");
        TORCH_CHECK(seqlens->dim() == 1 && seqlens->numel() == q.size(0) && seqlens->is_contiguous(),
                    "kv_seqlens must be contiguous [rows]");
    }

    const double softmax_scale = scale.value_or(1.0 / std::sqrt(double(q.size(2))));
    
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");
    TORCH_CHECK(std::isfinite(k_scale) && k_scale > 0 && std::isfinite(v_scale) && v_scale > 0,
                "k_scale and v_scale must be finite and positive");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                (k_scale == 1.0 && v_scale == 1.0),
                "BF16 K/V require unit k_scale and v_scale");

    at::hip::HIPGuardMasqueradingAsCUDA guard{static_cast<char>(q.get_device())};
    auto out = at::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    auto lse = at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));

    if (q.size(0) > 0) {
        int parts = 1;
        int device_cus = 1;
        const int64_t programs = q.size(0) * groups;
        const int64_t slots = packed.size(1);
        // Static shape/device properties only: no host reads of device metadata.
        if (slots * region >= 64) {
            hipDeviceProp_t properties;
            C10_HIP_CHECK(hipGetDeviceProperties(&properties, q.get_device()));
            device_cus=properties.multiProcessorCount;
            parts = plan(programs, slots, region, device_cus, q.size(0), k.size(0));
        }
        at::Tensor workspace, workspace_lse;
        if (parts > 1) {
            workspace = at::empty({parts, q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
            workspace_lse = at::empty({parts, q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
        }
        RegionArgs a{q.data_ptr<at::BFloat16>(), k.data_ptr(), v.data_ptr(),
            packed.data_ptr(), counts.data_ptr<int32_t>(), seqlens.has_value() ? seqlens->data_ptr<int32_t>() : nullptr,
            out.data_ptr<at::BFloat16>(), lse.data_ptr<float>(),
            parts > 1 ? workspace.data_ptr<float>() : nullptr,
            parts > 1 ? workspace_lse.data_ptr<float>() : nullptr,
            q.size(0), q.size(1), groups, k.size(0), packed.size(1),
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), packed.stride(0),
            int(region), parts, int(q.size(1) / groups),
            float(softmax_scale * k_scale * kLog2E), float(v_scale)};
        const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        TORCH_CHECK(groups <= 65535 && parts <= 65535, "group/part grid exceeds device limits");
        const dim3 grid(grid_size(q.size(0)), static_cast<unsigned int>(groups),
                        static_cast<unsigned int>(parts));
        launch(a, grid, stream, parts, int(q.size(2)));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (parts > 1) {
            sparse_region_detail::merge_internal_states(
                workspace, workspace_lse, out, lse, q.size(0) * q.size(1),
                q.size(2), parts, stream, device_cus, seqlens.has_value());
        }
    }
    return {out, lse};
}

// InLaneV: words[3j+k] hold dims 12r+4k..12r+4k+3 of tile token g+4j.
template<bool Masked, class Token>
__device__ void region_fp8_load_lane_values(const RegionArgs &a, const Token (&tokens)[4],
    int64_t group, int r, uint32_t (&words)[12]);
// Operands for MMACs 4K..4K+3 (dims 12r+4K+b); element j is token g+4j.
template<int K, bool Pair>
__device__ void region_fp8_lane_values(const uint32_t (&words)[12],
    sparse_attention_detail::mmac::Half4 (&out)[4]);
// InLaneV rows at base + tokens[j] * stride + 12r with 32-bit offsets.
template<bool Masked, class Token>
__device__ void region_fp8_load_lane_values32(const uint8_t *base, const Token (&tokens)[4],
    uint32_t stride, int r, uint32_t (&words)[12]);

// BF16 InLaneV mirrors the FP8 layout: values[3j+k] holds dims
// 12r+4k..12r+4k+3 of tile token g+4j, 24 contiguous bytes per row.
template<bool Masked, class Token>
__device__ __forceinline__ void region_bf16_load_lane_values(const RegionArgs &a, const Token (&tokens)[4],
    int64_t group, int r, sparse_attention_detail::mmac::Half4 (&values)[12]) {
    using sparse_attention_detail::mmac::Half4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool live = !Masked || tokens[j] >= 0;
        const int64_t offset = live ? int64_t(tokens[j]) * a.vs0 + group * a.vs1 + r * 12 : 0;
        Half4 row[3] = {};
        if (live)
            __builtin_memcpy(row, __builtin_assume_aligned(static_cast<const at::BFloat16*>(a.v) + offset, 8), 24);
        values[3 * j] = row[0];
        values[3 * j + 1] = row[1];
        values[3 * j + 2] = row[2];
    }
}
// Offset32 form: u24(token, stride_bytes) + 24r bytes from the SGPR group base.
template<bool Masked, class Token>
__device__ __forceinline__ void region_bf16_load_lane_values32(const uint8_t *base, const Token (&tokens)[4],
    uint32_t stride, int r, sparse_attention_detail::mmac::Half4 (&values)[12]) {
    using sparse_attention_detail::mmac::Half4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool live = !Masked || tokens[j] >= 0;
        const uint32_t offset = __umul24(uint32_t(tokens[j]), stride) + uint32_t(r * 24);
        Half4 row[3] = {};
        if (live) __builtin_memcpy(row, __builtin_assume_aligned(base + uint64_t(offset), 8), 24);
        values[3 * j] = row[0];
        values[3 * j + 1] = row[1];
        values[3 * j + 2] = row[2];
    }
}
// Operands for MMACs 4K..4K+3, i.e. dims 12r+4K+b; element j is token g+4j.
// The selectors read the low/high halves of both tokens, so no conversion is
// needed: the operands are the BF16 words already in registers.
template<int K>
__device__ __forceinline__ void region_bf16_lane_values(const sparse_attention_detail::mmac::Half4 (&values)[12],
    sparse_attention_detail::mmac::Half4 (&out)[4]) {
    using sparse_attention_detail::mmac::Words2;
    Words2 w[4], packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) __builtin_memcpy(&w[j], &values[3 * j + K], sizeof(Words2));
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int h = b >> 1;
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx936__))
        const uint32_t sel = (b & 1) ? 0x07060302u : 0x05040100u;
        packed[b] = {__builtin_amdgcn_perm(w[1][h], w[0][h], sel), __builtin_amdgcn_perm(w[3][h], w[2][h], sel)};
#else
        packed[b] = (b & 1) ? Words2{(w[0][h] >> 16) | (w[1][h] & 0xffff0000u), (w[2][h] >> 16) | (w[3][h] & 0xffff0000u)}
                            : Words2{(w[0][h] & 0xffffu) | (w[1][h] << 16), (w[2][h] & 0xffffu) | (w[3][h] << 16)};
#endif
    }
    __builtin_memcpy(out, packed, sizeof(packed));
}

// BF16 score and V loads for one tile: natural K plus shuffle, or Offset32
// byte offsets; V is always the InLaneV layout.
template<bool Masked, bool EarlyV, bool Offset32, class Token>
__device__ __forceinline__ void region_bf16_tile(const RegionArgs &a,
    const sparse_attention_detail::mmac::Vec8 (&query)[6], Token natural_token,
    const Token (&lane_tokens)[4], int64_t group, int lane, int r, int g, const void *k_base,
    const uint8_t *v_group, sparse_attention_detail::mmac::Float4 &score,
    sparse_attention_detail::mmac::Half4 (&values)[12]) {
    using namespace sparse_attention_detail::mmac;
    constexpr int D = 192;
    const bool live = !Masked || natural_token >= 0;
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    // Offset32: bytes from the SGPR group base; otherwise elements from a.k.
    const int64_t kb = Offset32
        ? int64_t(__umul24(uint32_t(natural_token), uint32_t(a.ks0 * 2)) + uint32_t((lane % 4) * 16))
        : live ? int64_t(natural_token) * a.ks0 + group * a.ks1 + (lane % 4) * 8 : 0;
    auto load_k = [&](int i) -> Vec8 {
        if (!live) return zero;
        if constexpr (Offset32)
            return load8(reinterpret_cast<const at::BFloat16*>(static_cast<const uint8_t*>(k_base) + (kb + i * 64)));
        else
            return load8(static_cast<const at::BFloat16*>(a.k) + (kb + i * 32));
    };
    auto load_v = [&]() {
        if constexpr (Offset32) region_bf16_load_lane_values32<Masked>(v_group, lane_tokens, uint32_t(a.vs0 * 2), r, values);
        else region_bf16_load_lane_values<Masked>(a, lane_tokens, group, r, values);
    };
    // EarlyV issues the independent V loads before the K window; the split
    // (Internal) kernels already do this through their own ordering.
    if constexpr (EarlyV) load_v();
    Vec8 window_k[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) window_k[i] = load_k(i);
#pragma unroll
    for (int i = 0; i < D / 32; ++i) {
        const Vec8 natural = window_k[i % 4];
        if (i + 4 < D / 32) window_k[i % 4] = load_k(i + 4);
        const Vec8 key = shuffle8(natural, r * 4 + g);
        region_dot(query[i].x, key.x, score);
        region_dot(query[i].y, key.y, score);
    }
    if constexpr (!EarlyV) load_v();
}

// Internal selects FP32 workspace output; FastRegion8 specializes R=8 and 16 heads per KV group.
template<bool Decode, bool Internal, bool FastRegion8, bool Fp8KV, int OptIn = 0>
__global__ __launch_bounds__(64) void sparse_region_attention_kernel(RegionArgs a) {
    constexpr int Opt = Fp8KV ? (OptIn & kFp8ArchOpt) : (OptIn & kBf16ArchOpt);
    // The BF16 tile only exists in the InLaneV layout; NoTransposeP keeps P in
    // the C layout that this V layout requires.
    constexpr bool Bf16Tile = !Fp8KV && (Opt & kFp8InLaneV) != 0;
    constexpr bool EarlyV = Internal || (Bf16Tile && (Opt & kBf16EarlyV) != 0);
    static_assert(Fp8KV || ((Opt & kFp8InLaneV) != 0) == ((Opt & kFp8NoTransposeP) != 0),
                  "BF16 NoTransposeP and InLaneV go together");
    constexpr bool InLaneV = (Opt & kFp8InLaneV) != 0;
    constexpr bool PairV = (Opt & kFp8PairV) != 0;
    constexpr bool PrefetchMeta = FastRegion8 && (Opt & kFp8PrefetchMeta) != 0;
    constexpr bool Offset32 = FastRegion8 && InLaneV && (Opt & kFp8Offset32) != 0;
    constexpr int D = 192;
    using namespace sparse_attention_detail::mmac;
    const int lane = threadIdx.x, r = lane % 16, g = lane / 16;
    const int64_t row = blockIdx.x;
    const int64_t group = blockIdx.y;
    const int part = Internal ? int(blockIdx.z) : 0;
    // Compile-time geometry only; query count, cache length and selected slots
    // remain dynamic. Other supported geometries retain the generic path.
    const int heads = FastRegion8 ? 16 : a.heads_per_group;
    const int region = FastRegion8 ? 8 : a.region;
    using Token = std::conditional_t<FastRegion8, int32_t, int64_t>;
    const int64_t first_head = group * heads;
    const int64_t meta_row = group * a.rows + row;
    const int64_t end = min(a.topk, max(int64_t{0}, int64_t(a.counts[meta_row])));
    const int64_t kv_len = Decode ? int64_t(a.seqlens[row]) : 0;
    const Vec8 zero = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    const RegionK8<Fp8KV> zero_key{};
    const RegionV4<Fp8KV> zero_value{};
    Vec8 query[D / 32];
    Float4 pv[D / 16];
#pragma unroll
    for (int i = 0; i < D / 32; ++i)
        query[i] = r < heads ? load8(a.q + row * a.qs0 + (first_head + r) * a.qs1 + i * 32 + g * 8) : zero;
#pragma unroll
    for (int d = 0; d < D / 16; ++d) pv[d] = {0, 0, 0, 0};
    float maximum = -INFINITY, denominator = 0;

    const int64_t tiles = (end * region + 15) / 16;
    // Internal part counts are powers of two; preserve the exact balanced
    // partition without two variable-width integer divisions per wave.
    const int part_shift = Internal ? __ffs(a.parts) - 1 : 0;
    const int64_t first_tile = Internal ? (part * tiles) >> part_shift : 0;
    const int64_t last_tile = Internal ? ((part + 1) * tiles) >> part_shift : tiles;
    // Offset32: byte offsets from the SGPR group bases. The launchers check
    // tokens and the byte strides < 2^24 and tokens * stride_bytes + row_bytes <= 2^32.
    const void *k_base = a.k;
    const uint8_t *v_group = nullptr;
    if constexpr (Offset32) {
        constexpr int element_bytes = Fp8KV ? 1 : 2;
        k_base = static_cast<const uint8_t*>(a.k) + group * a.ks1 * element_bytes;
        v_group = static_cast<const uint8_t*>(a.v) + group * a.vs1 * element_bytes;
    }
    // PrefetchMeta: lanes 0/1 hold the raw region words of the upcoming tile.
    RegionWord<Decode> next_word{};
    if constexpr (PrefetchMeta) {
        const int64_t slot = ((first_tile * 16) >> 3) + r;
        if (g == 0 && r < 2 && slot < end) next_word = load_region_word<Decode>(a, meta_row, slot);
    }
    for (int64_t tile = first_tile; tile < last_tile; ++tile) {
        const int64_t base = tile * 16;
        Token token = -1;
        if (region == 8) {
            // Two region words per tile; decode once, then broadcast to token lanes.
            RegionMeta meta{0,0};
            const int64_t slot=(base>>3)+r;
            if constexpr (PrefetchMeta) {
                const RegionWord<Decode> word = next_word;
                if (g == 0 && r < 2 && slot + 2 < end) next_word = load_region_word<Decode>(a, meta_row, slot + 2);
                if (g == 0 && r < 2 && slot < end) meta = decode_region<Decode, 8>(a, word, kv_len);
            } else if (g==0 && r<2 && slot<end) {
                meta=load_region<Decode, FastRegion8 ? 8 : 0>(a,meta_row,slot,kv_len);
            }
            const int64_t physical=FastRegion8 ? int64_t(__shfl(uint32_t(meta.physical),r>>3,64))
                                              : __shfl(meta.physical,r>>3,64);
            const int valid=__shfl(meta.valid,r>>3,64);
            const int64_t address=physical*8+(r&7);
            if ((r&7)<valid && address>=0 && address<a.tokens) token=address;
        } else {
          if (g == 0) {
            const int64_t slot = (base + r) / region;
            const int offset = (base + r) % region;
            if (slot < end) {
                const RegionMeta meta=load_region<Decode>(a,meta_row,slot,kv_len);
                if (offset < meta.valid) {
                    const int64_t address = meta.physical * region + offset;
                    if (address >= 0 && address < a.tokens) token = address;
                }
            }
          }
          token = __shfl(token, r, 64);
        }
        const uint64_t live_mask=__ballot(token >= 0);
        if (live_mask == 0) continue;
        const Token natural_token = __shfl(token, lane / 4, 64);
        const bool live = natural_token >= 0;
        RegionV4<Fp8KV> values[D / 16];
        Float4 score = {0, 0, 0, 0};
        // InLaneV value rows are tile tokens g+4j, which are also this lane's
        // score columns.
        Token lane_tokens[4];
        if constexpr (InLaneV) {
#pragma unroll
            for (int j = 0; j < 4; ++j) lane_tokens[j] = __shfl(token, g + 4 * j, 64);
        }
        // Offset32 folds the K column into its 32-bit lane offset.
        constexpr int k_column_scale = Offset32 ? 0 : 8;
        // Reuse the empty-tile ballot. Only a fully valid wave takes the
        // unconditional-load path; mixed/tail tiles retain the original masks.
        if constexpr (Bf16Tile) {
            if (live_mask == ~uint64_t{0})
                region_bf16_tile<false, EarlyV, Offset32>(a, query, natural_token, lane_tokens, group, lane, r, g,
                    k_base, v_group, score, values);
            else
                region_bf16_tile<true, EarlyV, Offset32>(a, query, natural_token, lane_tokens, group, lane, r, g,
                    k_base, v_group, score, values);
        } else if (live_mask == ~uint64_t{0}) {
            const int64_t kb = Offset32 ? int64_t(__umul24(uint32_t(natural_token), uint32_t(a.ks0)) + uint32_t((lane % 4) * 8))
                                        : natural_token * a.ks0 + group * a.ks1;
            const int64_t vb = natural_token * a.vs0 + group * a.vs1;
            if constexpr (Internal) {
                if constexpr (Offset32) {
                    region_fp8_load_lane_values32<false>(v_group, lane_tokens, uint32_t(a.vs0), r, values);
                } else if constexpr (InLaneV) {
                    region_fp8_load_lane_values<false>(a, lane_tokens, group, r, values);
                } else {
#pragma unroll
                for (int d = 0; d < D / 16; ++d)
                    values[d] = region_load_v4<Fp8KV>(a.v, vb + d * 16 + (lane % 4) * 4);
                }
            }
            RegionK8<Fp8KV> window_k[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Compile-time bound, folded away for the fixed head_dim=192.
                if (i < D / 32)
                    window_k[i] = region_load_k8<Fp8KV>(k_base, kb + i * 32 + (lane % 4) * k_column_scale);
                else
                    window_k[i] = zero_key;
            }
#pragma unroll
            for (int i = 0; i < D / 32; ++i) {
                const auto natural = window_k[i % 4];
                if (i + 4 < D / 32)
                    window_k[i % 4] = region_load_k8<Fp8KV>(k_base, kb + (i + 4) * 32 + (lane % 4) * k_column_scale);
                const Vec8 key = region_key<Fp8KV, Opt>(natural, r * 4 + g);
                region_dot(query[i].x, key.x, score);
                region_dot(query[i].y, key.y, score);
            }
            if constexpr (!Internal) {
                if constexpr (Offset32) {
                    region_fp8_load_lane_values32<false>(v_group, lane_tokens, uint32_t(a.vs0), r, values);
                } else if constexpr (InLaneV) {
                    region_fp8_load_lane_values<false>(a, lane_tokens, group, r, values);
                } else {
#pragma unroll
                for (int d = 0; d < D / 16; ++d)
                    values[d] = region_load_v4<Fp8KV>(a.v, vb + d * 16 + (lane % 4) * 4);
                }
            }
        } else {
        // Offset32 loads stay under live, so dead lanes need no zero offset.
        const int64_t kb = Offset32 ? int64_t(__umul24(uint32_t(natural_token), uint32_t(a.ks0)) + uint32_t((lane % 4) * 8))
                         : live ? natural_token * a.ks0 + group * a.ks1 : 0;
        const int64_t vb = live ? natural_token * a.vs0 + group * a.vs1 : 0;
        if constexpr (Internal) {
            if constexpr (Offset32) {
                region_fp8_load_lane_values32<true>(v_group, lane_tokens, uint32_t(a.vs0), r, values);
            } else if constexpr (InLaneV) {
                region_fp8_load_lane_values<true>(a, lane_tokens, group, r, values);
            } else {
#pragma unroll
        for (int d = 0; d < D / 16; ++d)
            values[d] = live ? region_load_v4<Fp8KV>(a.v, vb + d * 16 + (lane % 4) * 4) : zero_value;
            }
        }
        RegionK8<Fp8KV> window_k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            window_k[i] = live && i < D / 32 ? region_load_k8<Fp8KV>(k_base, kb + i * 32 + (lane % 4) * k_column_scale) : zero_key;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) {
            const auto natural = window_k[i % 4];
            if (i + 4 < D / 32)
                window_k[i % 4] = live ? region_load_k8<Fp8KV>(k_base, kb + (i + 4) * 32 + (lane % 4) * k_column_scale) : zero_key;
            const Vec8 key = region_key<Fp8KV, Opt>(natural, r * 4 + g);
            region_dot(query[i].x, key.x, score);
            region_dot(query[i].y, key.y, score);
        }
        // Long unsplit work benefits from releasing the K window before V
        // becomes live. Short internal parts retain the earlier V prefetch.
        if constexpr (!Internal) {
            if constexpr (Offset32) {
                region_fp8_load_lane_values32<true>(v_group, lane_tokens, uint32_t(a.vs0), r, values);
            } else if constexpr (InLaneV) {
                region_fp8_load_lane_values<true>(a, lane_tokens, group, r, values);
            } else {
#pragma unroll
            for (int d = 0; d < D / 16; ++d)
                values[d] = live ? region_load_v4<Fp8KV>(a.v, vb + d * 16 + (lane % 4) * 4) : zero_value;
            }
        }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            // All lanes must participate: masked head lanes still own token IDs.
            const Token score_token = InLaneV ? lane_tokens[e] : __shfl(token, g + e * 4, 64);
            score[e] = r < heads && score_token >= 0
                         ? score[e] * a.scale : -INFINITY;
        }
        float m = fmaxf(fmaxf(score[0], score[1]), fmaxf(score[2], score[3]));
        m = fmaxf(m, __shfl_xor(m, 16, 64));
        m = fmaxf(m, __shfl_xor(m, 32, 64));
        m = fmaxf(m, maximum);
        const float shift = m == -INFINITY ? 0.0f : m;
        const float alpha = m == -INFINITY ? 1.0f : region_exp2(maximum - shift);
        Float4 prob;
#pragma unroll
        for (int e = 0; e < 4; ++e) prob[e] = region_exp2(score[e] - shift);
        float sum = (prob[0] + prob[1]) + (prob[2] + prob[3]);
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        denominator = denominator * alpha + sum;
        maximum = m;
        // Wave-uniform shortcut; only skip an exactly identity rescale.
        if (__ballot(alpha != 1.0f) != 0) {
#pragma unroll
            for (int d = 0; d < D / 16; ++d) pv[d] *= alpha;
        }
        const Half4 p = region_prob<Opt>(prob, g);
        if constexpr (InLaneV) {
            Half4 v[3][4];
            if constexpr (Fp8KV) {
                region_fp8_lane_values<0, PairV>(values, v[0]);
                region_fp8_lane_values<1, PairV>(values, v[1]);
                region_fp8_lane_values<2, PairV>(values, v[2]);
            } else {
                region_bf16_lane_values<0>(values, v[0]);
                region_bf16_lane_values<1>(values, v[1]);
                region_bf16_lane_values<2>(values, v[2]);
            }
#pragma unroll
            for (int k = 0; k < 3; ++k)
#pragma unroll
                for (int b = 0; b < 4; ++b) region_dot(p, v[k][b], pv[4 * k + b]);
        } else {
#pragma unroll
        for (int d = 0; d < D / 16; ++d) {
            region_dot(p, region_value<Fp8KV, Opt>(values[d], g, r), pv[d]);
        }
        }
    }
    const int64_t output_row = row * a.heads + first_head + r;
    const int64_t state_row = (int64_t(part) * a.rows * a.heads) + output_row;
    const float inverse = denominator > 0 ? 1.0f / denominator : 0.0f;
    if constexpr (InLaneV) {
        // pv[4c+b][e] is dim 12(g+4e)+4c+b: 12 contiguous dims per e.
        const float factor = Fp8KV ? inverse * a.v_scale : inverse;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const Float4 value = Float4{pv[4 * c][e], pv[4 * c + 1][e], pv[4 * c + 2][e], pv[4 * c + 3][e]} * factor;
                const int column = 12 * (g + 4 * e) + 4 * c;
                if (r < heads) {
                    if constexpr (Internal)
                        *reinterpret_cast<Float4*>(a.workspace + state_row * D + column) = value;
                    else
                        *reinterpret_cast<Half4*>(a.out + output_row * D + column) = bf16_prob(value);
                }
            }
        }
    } else {
#pragma unroll
    for (int d = 0; d < D / 16; ++d) {
        // MMAC accumulators use col=g+4e. Repack to four adjacent columns
        // per lane, then issue one vector store instead of four scatter stores.
        const Float4 value = transpose_p(pv[d], g) *
            (Fp8KV ? inverse * a.v_scale : inverse);
        if (r < heads) {
            if constexpr (Internal)
                *reinterpret_cast<Float4*>(a.workspace + state_row * D + d * 16 + g * 4) = value;
            else
                *reinterpret_cast<Half4*>(a.out + output_row * D + d * 16 + g * 4) = bf16_prob(value);
        }
    }
    }
    if (r < heads && g == 0) {
        const float log2_lse = denominator > 0 ? maximum + log2f(denominator) : -INFINITY;
        if constexpr (Internal) a.workspace_lse[state_row] = log2_lse;
        else a.lse[output_row] = log2_lse * kLn2;
    }
}

template<bool Decode, bool Fp8KV, int Opt = 0>
void launch_region_kernel(RegionArgs a, dim3 grid, hipStream_t stream, int parts, int dim) {
    TORCH_CHECK(dim == 192, "sparse region attention requires head_dim == 192");
    if (a.region == 8 && a.heads_per_group == 16 && a.tokens <= INT32_MAX) {
        if (parts > 1) {
            sparse_region_attention_kernel<Decode, true, true, Fp8KV, Opt><<<grid, 64, 0, stream>>>(a);
        } else {
            sparse_region_attention_kernel<Decode, false, true, Fp8KV, Opt><<<grid, 64, 0, stream>>>(a);
        }
    } else {
        if (parts > 1) {
            sparse_region_attention_kernel<Decode, true, false, Fp8KV, Opt><<<grid, 64, 0, stream>>>(a);
        } else {
            sparse_region_attention_kernel<Decode, false, false, Fp8KV, Opt><<<grid, 64, 0, stream>>>(a);
        }
    }
}
} // namespace sparse_region_detail
