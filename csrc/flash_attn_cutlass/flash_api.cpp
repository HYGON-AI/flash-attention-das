/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hytlass/numeric_types.h>

#include "flash_c_api.h"
#include "hytlass/float8.h"
#include "static_switch.h"

#ifdef HAS_HG_DISPATCH
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
// Symbols defined in libflash_attention.so (HG), linked when HAS_HG_DISPATCH is set.
std::vector<at::Tensor>
hg_fwd_bshd(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        c10::optional<at::Tensor> s_aux_,
        const bool is_bf16_output);

std::vector<at::Tensor>
hg_fwd_bhsd(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        c10::optional<at::Tensor> s_aux_,
        const bool is_bf16_output);

std::vector<at::Tensor>
hg_varlen_fwd_bshd(at::Tensor &q,
        at::Tensor &k,
        at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output,
        c10::optional<at::Tensor> s_aux_);

std::vector<at::Tensor>
hg_prefix_prefill_varlen_fwd(
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        c10::optional<at::Tensor> &cu_seqlens_k,
        at::Tensor &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        at::Tensor &block_table,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        const int layout,
        c10::optional<at::Tensor> scales_q_,
        c10::optional<at::Tensor> scales_k_,
        c10::optional<at::Tensor> scales_v_,
        c10::optional<at::Tensor> s_aux_,
        const bool is_bf16_output);

std::vector<at::Tensor>
hg_prefix_decode_varlen_fwd(
        at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        c10::optional<at::Tensor> &cu_seqlens_k,
        at::Tensor &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        at::Tensor &block_table,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        const int layout,
        c10::optional<at::Tensor> scales_q_,
        c10::optional<at::Tensor> scales_k_,
        c10::optional<at::Tensor> scales_v_,
        c10::optional<at::Tensor> s_aux_,
        const bool is_bf16_output);

std::vector<at::Tensor>
hg_bwd_bshd(const at::Tensor &dout,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &softmax_lse,
        c10::optional<at::Tensor> &dq_,
        c10::optional<at::Tensor> &dk_,
        c10::optional<at::Tensor> &dv_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor>
hg_bwd_bhsd(const at::Tensor &dout,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &softmax_lse,
        c10::optional<at::Tensor> &dq_,
        c10::optional<at::Tensor> &dk_,
        c10::optional<at::Tensor> &dv_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor>
hg_varlen_bwd_bshd(const at::Tensor &dout,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &softmax_lse,
        c10::optional<at::Tensor> &dq_,
        c10::optional<at::Tensor> &dk_,
        c10::optional<at::Tensor> &dv_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state);
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static const bool print_param = get_env_("FLASH_ATTENTION_PRINT_PARAM");
static const bool print_hg_path = get_env_("FLASH_ATTENTION_PRINT_HG");
static const bool disable_varlen_tiny_dim64 = get_env_("FLASH_ATTENTION_DISABLE_VARLEN_TINY_DIM64");
static const bool enable_hg_bwd = !get_env_("FLASH_ATTENTION_DISABLE_HG_BWD");


#ifdef HAS_HG_DISPATCH
static inline bool is_gfx92a() {
    return get_device_name() == "gfx92a";
}

static inline bool is_gfx938() {
    return get_device_name() == "gfx938";
}

static inline bool is_gfx936() {
    return get_device_name() == "gfx936";
}

static inline bool is_hg_f16bf16(const at::ScalarType dtype) {
    return dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

enum class Hg92aMode {
    Off,
    Fp16,
    Fp16Bf16,
    Fp8,
};

static inline Hg92aMode get_hg_92a_mode() {
    const char *mode = std::getenv("FLASH_ATTENTION_HG_92A_MODE");
    if (mode == nullptr || std::strcmp(mode, "") == 0 || std::strcmp(mode, "fp16") == 0) {
        return Hg92aMode::Fp16;
    }
    if (std::strcmp(mode, "off") == 0 || std::strcmp(mode, "0") == 0 || std::strcmp(mode, "false") == 0) {
        return Hg92aMode::Off;
    }
    if (std::strcmp(mode, "fp16_bf16") == 0 || std::strcmp(mode, "f16_bf16") == 0 || std::strcmp(mode, "all") == 0) {
        return Hg92aMode::Fp16Bf16;
    }
    if (std::strcmp(mode, "fp8") == 0) {
        return Hg92aMode::Fp8;
    }
    return Hg92aMode::Fp16;
}

static inline bool is_hg_92a_dtype_enabled(const at::ScalarType dtype) {
    const Hg92aMode mode = get_hg_92a_mode();
    if (mode == Hg92aMode::Off) {
        return false;
    }
    if (dtype == torch::kFloat16) {
        return true;
    }
    if (dtype == torch::kFloat8_e4m3fn) {
        return mode == Hg92aMode::Fp8;
    }
    return mode == Hg92aMode::Fp16Bf16 && dtype == torch::kBFloat16;
}

static inline bool is_hg_938_dtype_enabled(const at::ScalarType dtype) {
    return is_hg_f16bf16(dtype);
}

static inline bool is_hg_fp8_d192(
        const at::ScalarType q_dtype,
        const int head_size_qk,
        const int head_size_v) {
    return q_dtype == torch::kFloat8_e4m3fn
        && head_size_qk == 192
        && head_size_v == 192;
}

static inline bool is_hg_legacy_head_dim(const int head_size_qk, const int head_size_v) {
    return (head_size_qk == 64 && head_size_v == 64)
        || (head_size_qk == 128 && head_size_v == 128)
        || (head_size_qk == 192 && head_size_v == 128);
}

static inline bool is_hg_regular_head_dim(const int head_size_qk, const int head_size_v) {
    return (head_size_qk == head_size_v
            && (head_size_qk == 32 || head_size_qk == 64 || head_size_qk == 96
                || head_size_qk == 128 || head_size_qk == 160 || head_size_qk == 192
                || head_size_qk == 224 || head_size_qk == 256 || head_size_qk == 512))
        || (head_size_qk == 192 && head_size_v == 128);
}

static inline bool is_hg_regular_varlen_head_dim(const int head_size_qk, const int head_size_v) {
    return is_hg_regular_head_dim(head_size_qk, head_size_v) && head_size_qk <= 256;
}

static inline bool is_hg_prefix_head_dim(const int head_size_qk, const int head_size_v) {
    return (head_size_qk == 128 && head_size_v == 128)
        || (head_size_qk == 192 && head_size_v == 128)
        || (head_size_qk == 192 && head_size_v == 192)
        || (head_size_qk == 256 && head_size_v == 256);
}

static inline bool is_hg_pa_head_dim(const int head_size_qk, const int head_size_v) {
    return is_hg_regular_varlen_head_dim(head_size_qk, head_size_v)
        || (head_size_qk == 576 && head_size_v == 512);
}

static inline std::vector<at::Tensor> make_hg_varlen_fwd_result(std::vector<at::Tensor> hg_result) {
    std::vector<at::Tensor> result(8);
    if (!hg_result.empty()) {
        result[0] = hg_result[0];
    }
    if (hg_result.size() > 1) {
        result[5] = hg_result[1];
    }
    return result;
}

static inline bool can_use_hg_92a_dense_fwd(
        const at::ScalarType q_dtype,
        const int head_size_qk,
        const int head_size_v,
        const float p_dropout,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &s_aux_,
        const float softcap,
        const float skip_softmax_threshold_scale_factor,
        const bool is_causal,
        const int seqlen_q,
        const int seqlen_k,
        const int window_size_left,
        const int window_size_right) {
    return is_gfx92a()
        && q_dtype == torch::kFloat8_e4m3fn
        && p_dropout == 0.f
        && !alibi_slopes_.has_value()
        && !s_aux_.has_value()
        && softcap == 0.f
        && window_size_left < 0
        && window_size_right <= 0
        && skip_softmax_threshold_scale_factor <= 0.f
        && (!is_causal || seqlen_q == seqlen_k)
        && is_hg_92a_dtype_enabled(q_dtype)
        && is_hg_regular_head_dim(head_size_qk, head_size_v)
        && seqlen_k > 0;
}

static inline bool can_use_hg_938_dense_fwd(
        const at::ScalarType q_dtype,
        const int head_size_qk,
        const int head_size_v,
        const float p_dropout,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const float softcap,
        const float skip_softmax_threshold_scale_factor,
        const bool is_causal,
        const int seqlen_q,
        const int seqlen_k,
        const int window_size_left,
        const int window_size_right,
        const bool is_mha) {
    // The hipc HG kernels do NOT implement softcap
    return (is_gfx938()||(is_gfx936()&&is_mha&&is_causal))
        && p_dropout == 0.f
        && !alibi_slopes_.has_value()
        && softcap == 0.f
        && skip_softmax_threshold_scale_factor <= 0.f
        && (seqlen_q == seqlen_k)
        && window_size_left < 0 && window_size_right <= 0
        && ((is_hg_938_dtype_enabled(q_dtype)
             && is_hg_legacy_head_dim(head_size_qk, head_size_v))
            || is_hg_fp8_d192(q_dtype, head_size_qk, head_size_v))
        && seqlen_k > 0;
}

static inline bool can_use_hg_92a_varlen_fwd(
        const at::ScalarType q_dtype,
        const bool paged_kv,
        const c10::optional<const at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const c10::optional<at::Tensor> &s_aux_,
        const float softcap,
        const float p_dropout,
        const int head_size_qk,
        const int head_size_v,
        const int max_seqlen_k,
        const int window_size_left,
        const int window_size_right) {
    return is_gfx92a()
        && q_dtype == torch::kFloat8_e4m3fn
        && p_dropout == 0.f
        && !paged_kv
        && !leftpad_k_.has_value()
        && !alibi_slopes_.has_value()
        && !s_aux_.has_value()
        && softcap == 0.f
        && head_size_qk==head_size_v
        && (head_size_v==128||head_size_v==256)
        // gfx92a only has the D128 FP8 HG kernel; the D256 FP8 launcher prints
        // "not supported" and leaves out unwritten, so fall back instead.
        && (q_dtype != torch::kFloat8_e4m3fn || head_size_v == 128)
        && max_seqlen_k > 0
        && window_size_left < 0
        && window_size_right <= 0;
}

static inline bool can_use_hg_938_varlen_fwd(
        const at::ScalarType q_dtype,
        const bool paged_kv,
        const c10::optional<const at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const c10::optional<at::Tensor> &s_aux_,
        const float softcap,
        const float p_dropout,
        const int head_size_qk,
        const int head_size_v,
        const int max_seqlen_k,
        const int window_size_left,
        const int window_size_right) {
    const bool s_aux_hg_supported = q_dtype == at::ScalarType::Half
        || q_dtype == at::ScalarType::BFloat16
        || (q_dtype == torch::kFloat8_e4m3fn);
    return is_gfx938()
        && p_dropout == 0.f
        && !paged_kv
        && !leftpad_k_.has_value()
        && !alibi_slopes_.has_value()
        && (!s_aux_.has_value() || s_aux_hg_supported)
        && softcap == 0.f
        && head_size_qk==head_size_v
        && (head_size_v==128||(head_size_v==256&&q_dtype == torch::kFloat8_e4m3fn))
        && max_seqlen_k > 0
        && window_size_left < 0
        && window_size_right <= 0;
}

static inline bool can_use_hg_b16_d192_varlen_fwd(
        const at::ScalarType q_dtype,
        const bool paged_kv,
        const c10::optional<at::Tensor> &seqused_k,
        const c10::optional<const at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const float p_dropout,
        const float softcap,
        const bool return_softmax,
        const int head_size_qk,
        const int head_size_v,
        const int max_seqlen_k) {
    return (is_gfx936() || is_gfx938())
        && is_hg_f16bf16(q_dtype)
        && !paged_kv
        && !seqused_k.has_value()
        && !leftpad_k_.has_value()
        && !alibi_slopes_.has_value()
        && !q_descale_.has_value()
        && !k_descale_.has_value()
        && !v_descale_.has_value()
        && p_dropout == 0.f
        && softcap == 0.f
        && !return_softmax
        && head_size_qk == 192
        && head_size_v == 192
        && max_seqlen_k > 0;
}

static inline bool can_use_hg_prefix_fwd(
        const at::ScalarType q_dtype,
        const bool paged_kv,
        const c10::optional<at::Tensor> &seqused_k,
        const c10::optional<const at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const c10::optional<at::Tensor> &s_aux_,
        const float softcap,
        const float p_dropout,
        const int head_size_qk,
        const int head_size_v,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const int window_size_left,
        const int window_size_right,
        const int page_block_size) {
    return paged_kv
        && seqused_k.has_value()
        && p_dropout == 0.f
        && softcap == 0.f
        && is_hg_prefix_head_dim(head_size_qk, head_size_v)
        && max_seqlen_q > 0
        && max_seqlen_k > 0
        && (page_block_size == 64 || page_block_size == 128);
}

static inline bool can_use_hg_dense_bwd(
        const at::ScalarType q_dtype,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const int head_size_qk,
        const int head_size_v,
        const bool is_causal,
        const int seqlen_q,
        const int seqlen_k,
        const int window_size_left,
        const int window_size_right,
        const float p_dropout) {
    return (get_device_name() == "gfx938"||get_device_name() == "gfx936")
        && p_dropout == 0.f
        && !alibi_slopes_.has_value()
        && window_size_left < 0 && window_size_right <= 0
        && q_dtype == torch::kBFloat16
        && head_size_qk == head_size_v
        && head_size_qk == 128
        && seqlen_q > 2048
        && seqlen_k == seqlen_q
        && enable_hg_bwd;
}

static inline bool can_use_hg_varlen_bwd(
        const at::ScalarType q_dtype,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const int head_size_qk,
        const int head_size_v,
        const int total_q,
        const int total_k,
        const int max_seqlen_k,
        const int window_size_left,
        const int window_size_right,
        const float p_dropout) {
    return (get_device_name() == "gfx938"||get_device_name() == "gfx936")
        && p_dropout == 0.f
        && !alibi_slopes_.has_value()
        && window_size_left < 0 && window_size_right <= 0
        && q_dtype == torch::kBFloat16
        && head_size_qk == head_size_v
        && head_size_qk == 128
        && total_q > 0
        && total_k >= total_q
        && max_seqlen_k > 2048
        && enable_hg_bwd;
}
#endif

static inline bool is_varlen_bshd_dense_hdim64(const at::Tensor &tensor, const int num_heads) {
    return tensor.stride(-1) == 1
        && tensor.stride(-2) == 64
        && tensor.stride(0) >= num_heads * 64;
}

static inline bool can_use_varlen_tiny_dim64_fwd(
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const c10::optional<at::Tensor> &out_,
        const at::ScalarType q_dtype,
        const int head_size_qk,
        const int head_size_v,
        const int num_heads,
        const int num_heads_k,
        const bool paged_kv,
        const bool is_causal,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const int window_size_left,
        const int window_size_right,
        const float p_dropout,
        const float softcap,
        const bool return_softmax,
        const c10::optional<at::Tensor> &seqused_k,
        const c10::optional<const at::Tensor> &leftpad_k_,
        const c10::optional<at::Tensor> &alibi_slopes_,
        const c10::optional<at::Tensor> &q_descale_,
        const c10::optional<at::Tensor> &k_descale_,
        const c10::optional<at::Tensor> &v_descale_,
        const c10::optional<at::Tensor> &s_aux_) {
    const std::string device_name = get_device_name();
    return (device_name == "gfx936" || device_name == "gfx938")
        && !disable_varlen_tiny_dim64
        && (q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16)
        && head_size_qk == 64
        && head_size_v == 64
        && is_causal
        && max_seqlen_q <= 4
        && max_seqlen_k <= 4
        && !paged_kv
        && !alibi_slopes_.has_value()
        && !s_aux_.has_value()
        && !leftpad_k_.has_value()
        && !seqused_k.has_value()
        && !q_descale_.has_value()
        && !k_descale_.has_value()
        && !v_descale_.has_value()
        && p_dropout == 0.f
        && softcap == 0.f
        && !return_softmax
        && window_size_left < 0
        && window_size_right <= 0
        && is_varlen_bshd_dense_hdim64(q, num_heads)
        && is_varlen_bshd_dense_hdim64(k, num_heads_k)
        && is_varlen_bshd_dense_hdim64(v, num_heads_k)
        && (!out_.has_value() || is_varlen_bshd_dense_hdim64(out_.value(), num_heads))
        && num_heads % num_heads_k == 0;
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool is_bhsd = false,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false,
                      int d_v=0,
                      int d_v_rounded=0
                    ) {

    // Reset the parameters
    params = {};

    params.is_bf16 = out.dtype() == torch::kBFloat16;
    params.is_fp8 = q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kFloat8_e5m2;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = !is_bhsd ? q.stride(-3) : q.stride(-2);
    params.k_row_stride = !is_bhsd ? k.stride(-3) : k.stride(-2);
    params.v_row_stride = !is_bhsd ? v.stride(-3) : v.stride(-2);
    params.q_head_stride = !is_bhsd ? q.stride(-2) : q.stride(-3);
    params.k_head_stride = !is_bhsd ? k.stride(-2) : k.stride(-3);
    params.v_head_stride = !is_bhsd ? v.stride(-2) : v.stride(-3);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = !is_bhsd ? out.stride(-3) : out.stride(-2);
    params.o_head_stride = !is_bhsd ? out.stride(-2) : out.stride(-3);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.d_value = d_v;
    params.d_value_rounded = d_v_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse,
                      bool is_bhsd = false,
                      int d_v=0,
                      int d_v_rounded=0) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     is_bhsd,
                     false,
                     unpadded_lse,
                     d_v,
                     d_v_rounded);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = !is_bhsd ? dout.stride(-3) : dout.stride(-2);
    params.do_head_stride = !is_bhsd ? dout.stride(-2) : dout.stride(-3);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = !is_bhsd ? dq.stride(-3) : dq.stride(-2);
    params.dk_row_stride = !is_bhsd ? dk.stride(-3) : dk.stride(-2);
    params.dv_row_stride = !is_bhsd ? dv.stride(-3) : dv.stride(-2);
    params.dq_head_stride = !is_bhsd ? dq.stride(-2) : dq.stride(-3);
    params.dk_head_stride = !is_bhsd ? dk.stride(-2) : dk.stride(-3);
    params.dv_head_stride = !is_bhsd ? dv.stride(-2) : dv.stride(-3);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_mha_fwd_mla(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
	params.num_splits=1;
	 if(params.is_fp8==true)
    {
        using elem_type = hytlass::float_e4m3_t;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_mha_fwd_mla_<elem_type, 192, 128, Is_causal>(params, stream);
        });
    }
    else
    {
        FP16_SWITCH(!params.is_bf16, [&] {
        // using elem_type = cutlass::half_t;
        // constexpr static int kHeadDim = 128;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
            run_mha_fwd_mla_<elem_type, 192, 128, Is_causal>(params, stream);
        });
		});
    }
}

void run_mha_blasst_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        // using elem_type = cutlass::bfloat16_t;
        // HEADDIM_SWITCH(params.d, [&] {
            constexpr static int kHeadDim = 128;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // constexpr static bool Is_causal = false; 
                run_mha_blasst_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        // });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}


void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

extern "C"
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        bool is_bhsd = false,
        const c10::optional<at::Tensor> &s_aux_ = c10::nullopt, 
        float skip_softmax_threshold_scale_factor = 0.0f) {  // Attention Sinks: precomputed LSE for sink tokens

    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto q_dtype = q.dtype();
    if(q.data_ptr()==k.data_ptr()){
        q=q.clone();
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = !is_bhsd ? sizes[1] : sizes[2];
    int num_heads = !is_bhsd ? sizes[2] : sizes[1];
    const int head_size_og = sizes[3];
    const int head_size_og_value = v.sizes()[3];
    const int seqlen_k = !is_bhsd ? k.size(1) : k.size(2);
    const int num_heads_k = !is_bhsd ? k.size(2) : k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(head_size_og_value <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(head_size_og >= head_size_og_value, "Head dimension of query/key must greater or equal to head dimension in query");

    bool is_mla = head_size_og != head_size_og_value;

#ifdef HAS_HG_DISPATCH
    bool hg_is_causal = is_causal;
    int hg_window_size_left = window_size_left;
    int hg_window_size_right = window_size_right;
    if (hg_window_size_left >= seqlen_k) { hg_window_size_left = -1; }
    if (hg_window_size_right >= seqlen_k) { hg_window_size_right = -1; }
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { hg_is_causal = false; }
    if (hg_is_causal) { hg_window_size_right = 0; }
    const bool use_hg_92a_dense = can_use_hg_92a_dense_fwd(
            q.scalar_type(), head_size_og, head_size_og_value, p_dropout,
            alibi_slopes_, s_aux_, softcap,
            skip_softmax_threshold_scale_factor,
            hg_is_causal, seqlen_q, seqlen_k,
            hg_window_size_left, hg_window_size_right);
    const bool use_hg_938_dense = can_use_hg_938_dense_fwd(
            q.scalar_type(), head_size_og, head_size_og_value, p_dropout,
            alibi_slopes_, softcap,
            skip_softmax_threshold_scale_factor,
            is_causal, seqlen_q, seqlen_k,
            window_size_left, window_size_right,num_heads == num_heads_k)
            && (q.scalar_type() != torch::kFloat8_e4m3fn
                || (!s_aux_.has_value() && !return_softmax));
    if (!is_bhsd && (use_hg_92a_dense || use_hg_938_dense)) {
        const bool hg_call_is_causal = use_hg_92a_dense ? hg_is_causal : is_causal;
        const int hg_call_window_size_left = use_hg_92a_dense ? hg_window_size_left : window_size_left;
        const int hg_call_window_size_right = use_hg_92a_dense ? hg_window_size_right : window_size_right;
        const bool hg_bf16_output = !is_hg_fp8_d192(
                q.scalar_type(), head_size_og, head_size_og_value)
            || !out_.has_value() || out_->scalar_type() != torch::kFloat16;
        if (print_param || print_hg_path) {
            const std::string hg_arch = get_device_name();
            printf("[flash_attn] HG PATH %s layout=%s q=(%d,%d,%d,%d) k=(%d,%d,%d,%d) v=(%d,%d,%d,%d)\n",
                hg_arch.c_str(), is_bhsd ? "bhsd" : "bshd",
                (int)q.size(0), (int)q.size(1), (int)q.size(2), (int)q.size(3),
                (int)k.size(0), (int)k.size(1), (int)k.size(2), (int)k.size(3),
                (int)v.size(0), (int)v.size(1), (int)v.size(2), (int)v.size(3));
        }
        // HG kernel does not support S_dmask output (ReturnSoftmaxConst=false),
        // always pass return_softmax=false to avoid returning uninitialized data.
        auto hg_result=hg_fwd_bshd(q, k, v, out_, alibi_slopes_,
                p_dropout, softmax_scale, hg_call_is_causal,
                hg_call_window_size_left, hg_call_window_size_right,
                softcap, false /*return_softmax*/, gen_,
                c10::nullopt, c10::nullopt, c10::nullopt, s_aux_, hg_bf16_output);
        hg_result.push_back(at::Tensor());
        return hg_result;
    }
#endif
    if(print_param){
        printf("mha_fwd fa input size bshd=(%d,%d,%d,%d),p_dropout=%.3f,softmax_scale=%.3f,is_causal=%d,window_size_left=%d,window_size_right=%d,softcap=%f,return_softmax=%d,is_bhsd=%d\n",
        (int)sizes[0],(int)sizes[1],(int)sizes[2],(int)sizes[3],p_dropout,softmax_scale,(int)is_causal,window_size_left,window_size_right,softcap,return_softmax,is_bhsd);
    }
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    if(!is_bhsd) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og_value);
    } else {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og_value);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_value % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    auto opts = q.options();

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if(!is_bhsd) {
            CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og_value);
        } else {
            CHECK_SHAPE(out, batch_size, sizes[2], sizes[1], head_size_og_value);
        }

        // if (head_size_og_value % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (!is_mla)
            out = torch::empty_like(q_padded);
        else
            out = torch::empty({sizes[0], sizes[1], sizes[2], head_size_og_value}, opts);
        //out = torch::empty({sizes[0], sizes[1], sizes[2], head_size_og_value}, opts);
    }
    if (head_size_og_value % 8 != 0) {
        out = torch::nn::functional::pad(out, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og_value % 8}));
    }
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_og_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32);
    // dim64的滑块大小为256，需要保证p的大小为256的整数倍
    const int seqlen_q_rounded = head_size_rounded == 64 ? round_multiple(seqlen_q, 256) : round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax&&p_dropout>0.0f) {
        // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap, 
                     is_bhsd,
                     false,
                     /*unpadded_lse*/false,
                     head_size_v,
                     head_size_v_rounded);
    params.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor;
    params.skip_blocks_info_ptr = nullptr;
    at::Tensor skip_blocks_info;
    if (skip_softmax_threshold_scale_factor > 0) {
        skip_blocks_info = torch::zeros({batch_size, num_heads, 2}, opts.dtype(at::kInt));
        params.skip_blocks_info_ptr = skip_blocks_info.data_ptr();
    }
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // Attention Sinks: set s_aux_ptr
    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        // ★ s_aux must match Q/K/V dtype (Element type) for mixed precision
        TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match Q/K/V). Got: ", s_aux.dtype());
        TORCH_CHECK(num_heads <= 64, "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (!is_mla) {
            if (skip_softmax_threshold_scale_factor > 0.f)
                run_mha_blasst_fwd(params, stream);
            else
                run_mha_fwd(params, stream);
        }
        else
            run_mha_fwd_mla(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og_value % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og_value)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state, skip_blocks_info};
}
extern "C"
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               c10::optional<const at::Tensor> &leftpad_k_, // batch_size
               c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
			   c10::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
               c10::optional<at::Tensor> k_descale_,  // (b, h_k)
               c10::optional<at::Tensor> v_descale_,  // (b, h_k)
               c10::optional<at::Generator> gen_,
               const c10::optional<at::Tensor> &s_aux_ = c10::nullopt
            ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || q_dtype == torch::kFloat8_e4m3fn|| q_dtype == torch::kFloat8_e5m2,
                "FlashAttention only support fp16 and bf16 or fp8 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }
    TORCH_CHECK(!paged_KV || softcap <= 0.f,
                "mha_varlen_fwd: page attention (block_table) does not support softcap");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    if(q.data_ptr()==k.data_ptr()){
        q=q.clone();
    }
    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int head_size_value = paged_KV ? v.size(3) : v.size(2);
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    bool is_mla = head_size_og != head_size_value;

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    if (get_device_name() == "gfx936" || get_device_name() == "gfx938")
    {
        TORCH_CHECK(!paged_KV || page_block_size % 64 == 0, "Paged KV cache block size must be divisible by 64 at gfx936 platform");
    }
    else
    {
        TORCH_CHECK(!paged_KV || page_block_size % 16 == 0, "Paged KV cache block size must be divisible by 16 at gfx928 platform");
    }

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const bool use_varlen_tiny_dim64 = can_use_varlen_tiny_dim64_fwd(
            q, k, v, out_, q.scalar_type(), head_size_og, head_size_value, num_heads, num_heads_k,
            paged_KV, is_causal, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right,
            p_dropout, softcap, return_softmax, seqused_k, leftpad_k_, alibi_slopes_,
            q_descale_, k_descale_, v_descale_, s_aux_);

#ifdef HAS_HG_DISPATCH
    bool hg_is_causal = is_causal;
    int hg_window_size_left = window_size_left;
    int hg_window_size_right = window_size_right;
    if (hg_window_size_left >= max_seqlen_k) { hg_window_size_left = -1; }
    if (hg_window_size_right >= max_seqlen_k) { hg_window_size_right = -1; }
    if (hg_is_causal) { hg_window_size_right = 0; }
    if (can_use_hg_prefix_fwd(
            q.scalar_type(), paged_KV, seqused_k, leftpad_k_, alibi_slopes_,
            q_descale_, k_descale_, v_descale_, s_aux_, softcap,
            p_dropout, head_size_og, head_size_value, max_seqlen_q, max_seqlen_k,
            hg_window_size_left, hg_window_size_right, page_block_size)) {
        // D192 FP8 paged attention uses the prefill kernel for short Q too.
        const bool use_hg_fp8_d192_prefix = is_gfx938()
            && is_hg_fp8_d192(q.scalar_type(), head_size_og, head_size_value);
        const bool prefix_bf16_output = !use_hg_fp8_d192_prefix || !out_.has_value()
            || out_->scalar_type() == at::ScalarType::BFloat16;
        if (print_param || print_hg_path) {
            printf("[flash_attn] HG PATH prefix %s q=(%d,%d,%d) k=(%d,%d,%d,%d) v=(%d,%d,%d,%d) batch_size=%d max_seqlen_q=%d max_seqlen_k=%d\n",
                (max_seqlen_q >= 16 || use_hg_fp8_d192_prefix) ? "prefill" : "decode",
                (int)q.size(0), (int)q.size(1), (int)q.size(2),
                (int)k.size(0), (int)k.size(1), (int)k.size(2), (int)k.size(3),
                (int)v.size(0), (int)v.size(1), (int)v.size(2), (int)v.size(3),
                batch_size, max_seqlen_q, max_seqlen_k);
        }
        at::Tensor seqused_k_tensor = seqused_k.value();
        c10::optional<at::Tensor> hg_cu_seqlens_k = c10::nullopt;
        auto hg_result = (max_seqlen_q >= 16 || use_hg_fp8_d192_prefix)
            ? hg_prefix_prefill_varlen_fwd(
                q, k, v, out_, cu_seqlens_q, hg_cu_seqlens_k, seqused_k_tensor,
                alibi_slopes_, block_table, max_seqlen_q, max_seqlen_k, 0.f,
                softmax_scale, zero_tensors, hg_is_causal, hg_window_size_left,
                hg_window_size_right, softcap, return_softmax, 1 /*bshd*/,
                q_descale_, k_descale_, v_descale_, s_aux_,
                prefix_bf16_output)
            : hg_prefix_decode_varlen_fwd(
                q, k, v, out_, cu_seqlens_q, hg_cu_seqlens_k, seqused_k_tensor,
                alibi_slopes_, block_table, max_seqlen_q, max_seqlen_k, 0.f,
                softmax_scale, zero_tensors, hg_is_causal, hg_window_size_left,
                hg_window_size_right, softcap, return_softmax, 1 /*bshd*/,
                q_descale_, k_descale_, v_descale_, s_aux_,
                true /*is_bf16_output*/);
        return make_hg_varlen_fwd_result(std::move(hg_result));
    }
    const bool use_hg_92a_varlen = can_use_hg_92a_varlen_fwd(
            q.scalar_type(), paged_KV, leftpad_k_, alibi_slopes_,
            q_descale_, k_descale_, v_descale_, s_aux_, softcap,
            p_dropout, head_size_og, head_size_value, max_seqlen_k,
            hg_window_size_left, hg_window_size_right);
    const bool use_hg_938_varlen = can_use_hg_938_varlen_fwd(
            q.scalar_type(), paged_KV, leftpad_k_, alibi_slopes_,
            q_descale_, k_descale_, v_descale_, s_aux_, softcap,
            p_dropout, head_size_og, head_size_value, max_seqlen_k,
            window_size_left, window_size_right);
    const bool use_hg_b16_d192_varlen = can_use_hg_b16_d192_varlen_fwd(
            q.scalar_type(), paged_KV, seqused_k, leftpad_k_, alibi_slopes_,
            q_descale_, k_descale_, v_descale_, p_dropout, softcap,
            return_softmax, head_size_og, head_size_value, max_seqlen_k);
    const bool use_hg_fp8_d192_varlen = is_gfx938()
            && is_hg_fp8_d192(q.scalar_type(), head_size_og, head_size_value)
            && !paged_KV && !seqused_k.has_value() && !leftpad_k_.has_value()
            && !alibi_slopes_.has_value() && !s_aux_.has_value()
            && !q_descale_.has_value() && !k_descale_.has_value()
            && !v_descale_.has_value()
            && p_dropout == 0.f && softcap == 0.f && !return_softmax
            && max_seqlen_k > 0 && window_size_left < 0 && window_size_right <= 0;
    const bool use_hg_fp8_d192_d128_varlen = is_gfx938()
            && q.scalar_type() == at::ScalarType::Float8_e4m3fn
            && head_size_og == 192 && head_size_value == 128
            && !paged_KV && !seqused_k.has_value() && !leftpad_k_.has_value()
            && !alibi_slopes_.has_value()
            && p_dropout == 0.f && softcap == 0.f && !return_softmax
            && max_seqlen_k > 0 && is_causal && window_size_left < 0;
    if (!use_varlen_tiny_dim64
            && (use_hg_92a_varlen || use_hg_938_varlen || use_hg_b16_d192_varlen
                || use_hg_fp8_d192_varlen || use_hg_fp8_d192_d128_varlen)) {
        const bool hg_call_is_causal = use_hg_92a_varlen ? hg_is_causal : is_causal;
        const int hg_call_window_size_left = use_hg_92a_varlen ? hg_window_size_left : window_size_left;
        const int hg_call_window_size_right = use_hg_92a_varlen ? hg_window_size_right : window_size_right;
        const bool hg_bf16_output = !(use_hg_fp8_d192_varlen || use_hg_fp8_d192_d128_varlen)
                || !out_.has_value() || out_->scalar_type() != torch::kFloat16;
        if (use_hg_fp8_d192_d128_varlen) {
            // HG varlen records batch/head strides; preserve existing 2D+ tensors.
            if (q_descale_.has_value() && q_descale_->dim() < 2) {
                q_descale_ = q_descale_->view({1, -1});
            }
            if (k_descale_.has_value() && k_descale_->dim() < 2) {
                k_descale_ = k_descale_->view({1, -1});
            }
            if (v_descale_.has_value() && v_descale_->dim() < 2) {
                v_descale_ = v_descale_->view({1, -1});
            }
        }
        if (print_param || print_hg_path) {
            const std::string hg_arch = get_device_name();
            printf("[flash_attn] HG PATH %s layout=bshd varlen q=(%d,%d,%d) k=(%d,%d,%d) v=(%d,%d,%d) batch_size=%d max_seqlen_q=%d max_seqlen_k=%d\n",
                hg_arch.c_str(), (int)q.size(0), (int)q.size(1), (int)q.size(2),
                (int)k.size(0), (int)k.size(1), (int)k.size(2),
                (int)v.size(0), (int)v.size(1), (int)v.size(2),
                batch_size, max_seqlen_q, max_seqlen_k);
        }
        // Packed varlen tensors use [total_tokens, num_heads, head_dim], which
        // matches HG's bshd layout semantics even though batch and seqlen are
        // flattened into a single leading dimension.
        // HG kernel does not support S_dmask output (ReturnSoftmaxConst=false).
        auto hg_result = hg_varlen_fwd_bshd(q, const_cast<at::Tensor &>(k), const_cast<at::Tensor &>(v), out_,
            cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes_,
            max_seqlen_q, max_seqlen_k,
            p_dropout, softmax_scale, zero_tensors, hg_call_is_causal,
            hg_call_window_size_left, hg_call_window_size_right,
            softcap, false /*return_softmax*/, gen_,
            q_descale_, k_descale_, v_descale_, hg_bf16_output, s_aux_);
        return hg_result;
    }
#endif

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(head_size_value <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size_value);
        if (print_param) {
          printf(
              "mha_varlen_fwd fa input size "
              "batch_size,total_q,num_heads_q,total_kv,num_heads_kv,dim_qk,dim_v=(%d,%d,"
              "%d,%d,%d,%d,%d),p_dropout=%.3f,softmax_scale=%.3f,is_causal=%d,window_size_left=%d,window_size_right=%d,softcap=%f,return_softmax=%d\n",
              batch_size, total_q, num_heads, total_k, num_heads_k, head_size_og, head_size_value, p_dropout, softmax_scale, 
              (int)is_causal,window_size_left,window_size_right,softcap,return_softmax);
        }
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_value);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
        if (print_param) {
          printf(
              "mha_varlen_fwd fa input size "
              "batch_size,total_q,num_heads_q,num_blocks,page_block_size,num_"
              "heads_kv,dim_qk,dim_v=(%d,%d,%d,%d,%d,%d,%d,%d),p_dropout=%.3f,softmax_scale=%."
              "3f,is_causal=%d,window_size_left=%d,window_size_right=%d,softcap=%f,return_softmax=%d\n",
              batch_size, total_q, num_heads, num_blocks, page_block_size,
              num_heads_k, head_size_og, head_size_value, p_dropout, softmax_scale,
              (int)is_causal,window_size_left,window_size_right,softcap,return_softmax);
        }
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        // v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        // v_padded = v;
    }
    if (head_size_value % 8 != 0) {
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_value % 8}));
    } else {
        v_padded = v;
    }
    auto opts = q.options();

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype || q_dtype == torch::kFloat8_e4m3fn, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size_value);
        
        // if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        // out = torch::empty_like(q_padded);
        if(q_dtype == torch::kFloat8_e4m3fn)
        {
          out = torch::empty({sizes[0], sizes[1], head_size_value}, opts.dtype(torch::kBFloat16));
        }
        else
        {
           out = torch::empty({sizes[0], sizes[1], head_size_value}, opts);
        }
    }
    if (head_size_value % 8 != 0) {
        out = torch::nn::functional::pad(out, torch::nn::functional::PadFuncOptions({0, 8 - head_size_value % 8}));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = head_size_rounded == 64 ? round_multiple(max_seqlen_q, 256) : round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax&&p_dropout>0.0f) {
        // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }
    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    if (use_varlen_tiny_dim64) {
        Flash_fwd_params tiny_params;
        set_params_fprop(tiny_params,
                         batch_size,
                         max_seqlen_q, max_seqlen_k,
                         seqlen_q_rounded, seqlen_k_rounded,
                         num_heads, num_heads_k,
                         head_size, head_size_rounded,
                         q_padded, k_padded, v_padded, out,
                         cu_seqlens_q_d,
                         cu_seqlens_k.data_ptr(),
                         nullptr,
                         nullptr,
                         softmax_lse.data_ptr(),
                         0.f,
                         softmax_scale,
                         window_size_left,
                         window_size_right,
                         0.f,
                         /*is_bhsd*/false,
                         false,
                         /*unpadded_lse*/true,
                         head_size_v,
                         head_size_v_rounded);
        tiny_params.total_q = total_q;
        if (print_param) {
            printf("VARLEN_TINY_DIM64 PATH\n");
        }
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const int64_t total_k = k_padded.sizes()[0];
        run_mha_varlen_tiny_fwd_dim64(tiny_params, stream, total_k);
        auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
        return {out, q_padded, k_padded, v_padded, out, softmax_lse, p, rng_state};
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*is_bhsd*/false,
                     false,
                     /*unpadded_lse*/true,
                     head_size_v,
                     head_size_v_rounded);
    params.total_q = total_q;

    if (q_dtype == torch::kFloat8_e4m3fn || q_dtype == torch::kFloat8_e5m2) {
        // This FP8 path uses unit descale. Accept and ignore all descale inputs.
        params.q_descale_ptr = nullptr;
        params.k_descale_ptr = nullptr;
        params.v_descale_ptr = nullptr;
    }

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k_padded.stride(0);
        params.v_batch_stride = v_padded.stride(0);
    }
    params.page_block_size = page_block_size;

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match Q/K/V). Got: ", s_aux.dtype());
        TORCH_CHECK(num_heads <= 64,
                    "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (!is_mla)
            run_mha_fwd(params, stream, paged_KV);
        else
            run_mha_fwd_mla(params, stream, paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_value % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_value)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        // using elem_type = cutlass::half_t;
        HEADDIM_SWITCH(params.d, [&] {
            // constexpr static int kHeadDim = 256;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // constexpr static bool Is_causal = false;
                if (kHeadDim == 192 && params.d_value <= 128 ) {
                    run_mha_bwd_mla_<elem_type, 192, 128, Is_causal>(params, stream);
                } else {
                    run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
extern "C"
std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state,
        bool is_bhsd = false) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if(q.data_ptr()==k.data_ptr()){
        q=q.clone();
    }
    if (is_causal) { window_size_right = 0; }
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    // bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    // bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // if (q_dtype == torch::kBFloat16) {
    //     TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    // }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = !is_bhsd ? sizes[1] : sizes[2];
    const int num_heads = !is_bhsd ? sizes[2] : sizes[1];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int head_size_value = v.sizes()[3];
    const int seqlen_k = !is_bhsd ? k.size(1) : k.size(2);
    const int num_heads_k = !is_bhsd ? k.size(2) : k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 512, "FlashAttention backward only supports head dimension at most 512");
    TORCH_CHECK(head_size_value <= 512, "FlashAttention backward only supports head dimension at most 512");
    // if (head_size > 192 && is_dropout) {
    //     TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 with dropout requires A100/A800 or H100/H800");
    // }
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(head_size_value == round_multiple(head_size_og, 8), "head_size_value must be head_size_og rounded to a multiple of 8");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    if(!is_bhsd) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_value);
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_value);
        CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);
    } else {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_value);
        CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_value);
        CHECK_SHAPE(dout, batch_size, num_heads, seqlen_q, head_size_og);
    }

#ifdef HAS_HG_DISPATCH
    if (can_use_hg_dense_bwd(
            q.scalar_type(), alibi_slopes_,
            head_size, head_size_value, is_causal, seqlen_q, seqlen_k,
            window_size_left, window_size_right, p_dropout)) {
        if (print_param || print_hg_path) {
            printf("[flash_attn] HG BWD PATH layout=%s q=(%d,%d,%d,%d) k=(%d,%d,%d,%d) v=(%d,%d,%d,%d) dout=(%d,%d,%d,%d)\n",
                is_bhsd ? "bhsd" : "bshd",
                (int)q.size(0), (int)q.size(1), (int)q.size(2), (int)q.size(3),
                (int)k.size(0), (int)k.size(1), (int)k.size(2), (int)k.size(3),
                (int)v.size(0), (int)v.size(1), (int)v.size(2), (int)v.size(3),
                (int)dout.size(0), (int)dout.size(1), (int)dout.size(2), (int)dout.size(3));
        }
        return is_bhsd
            ? hg_bwd_bhsd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_,
                alibi_slopes_, p_dropout, softmax_scale, is_causal,
                window_size_left, window_size_right, softcap,
                deterministic, gen_, rng_state)
            : hg_bwd_bshd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_,
                alibi_slopes_, p_dropout, softmax_scale, is_causal,
                window_size_left, window_size_right, softcap,
                deterministic, gen_, rng_state);
    }
#endif

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        if(!is_bhsd) {
            CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
        } else {
            CHECK_SHAPE(dq, batch_size, num_heads, seqlen_q, head_size);
        }
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        if(!is_bhsd) {
            CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(dk, batch_size, num_heads_k, seqlen_k, head_size);
        }
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        if(!is_bhsd) {
            CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size_value);
        } else {
            CHECK_SHAPE(dv, batch_size, num_heads_k, seqlen_k, head_size_value);
        }
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = false;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    // at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = !is_bhsd ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts) 
            : torch::empty({batch_size, num_heads, seqlen_k, head_size}, opts);
        dv_expanded = !is_bhsd ? torch::empty({batch_size, seqlen_k, num_heads, head_size_value}, opts) 
            : torch::empty({batch_size, num_heads, seqlen_k, head_size_value}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum.data_ptr() : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/false, is_bhsd,
                     head_size_value,
                     head_size_value_rounded
                    );
    params.dq_accum_split_stride = !(loop && deterministic) ? 0 : dq_accum.stride(0);

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        if (!is_bhsd) {
            at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
            at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_value}), {3});
        } else {
            at::sum_out(dk, at::reshape(dk_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size}), {2});
            at::sum_out(dv, at::reshape(dv_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size_value}), {2});
        }
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_value)});
    }

    return { dq, dk, dv, softmax_d };
}
extern "C"
std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
               c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               c10::optional<at::Generator> gen_,
               c10::optional<at::Tensor> &rng_state) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if(q.data_ptr()==k.data_ptr()){
        q=q.clone();
    }
    if (is_causal) { window_size_right = 0; }
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    // bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    // bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");
    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // if (q_dtype == torch::kBFloat16) {
    //     TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    // }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int head_size_value = v.size(2);
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_value % 8 == 0, "head_size_value should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");
    // if (head_size > 192 && is_dropout) {
    //     TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 with dropout requires A100/A800 or H100/H800");
    // }
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    TORCH_CHECK(head_size_value == round_multiple(head_size_og, 8), "head_size_value must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_value);
    CHECK_SHAPE(out, total_q, num_heads, head_size_value);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

#ifdef HAS_HG_DISPATCH
    const bool use_hg_nmz_hdim128_varlen_bwd = head_size == 128 && head_size_value == 128;
    if ( use_hg_nmz_hdim128_varlen_bwd
        && can_use_hg_varlen_bwd(
            q.scalar_type(), alibi_slopes_,
            head_size, head_size_value, total_q, total_k, max_seqlen_k,
            window_size_left, window_size_right, p_dropout)) {
        if (print_param || print_hg_path) {
            printf("[flash_attn] HG VARLEN BWD PATH q=(%d,%d,%d) k=(%d,%d,%d) v=(%d,%d,%d) dout=(%d,%d,%d) batch_size=%d max_seqlen_q=%d max_seqlen_k=%d\n",
                (int)q.size(0), (int)q.size(1), (int)q.size(2),
                (int)k.size(0), (int)k.size(1), (int)k.size(2),
                (int)v.size(0), (int)v.size(1), (int)v.size(2),
                (int)dout.size(0), (int)dout.size(1), (int)dout.size(2),
                batch_size, max_seqlen_q, max_seqlen_k);
        }
        return hg_varlen_bwd_bshd(
            dout, q, k, v, out, softmax_lse, dq_, dk_, dv_,
            cu_seqlens_q, cu_seqlens_k, alibi_slopes_,
            max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale,
            zero_tensors, is_causal, window_size_left, window_size_right,
            softcap, deterministic, gen_, rng_state);
    }
#endif

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size_value);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = false;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
        // because that would be too large if there is a very long sequence and the rest of the sequences are short.
        // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
        // Note that 128 is the max block size on the seqlen_q dimension.
        // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
        // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
        // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
        // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
        // Same holds for softmax_d, since LSE is stored in unpadded format.
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size_value}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     loop ? dq_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/true, false,
                     head_size_value,
                     head_size_value_rounded);
    params.dq_accum_split_stride = !loop || !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total_q;

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_value}), {2});
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_value)});
    }

    return { dq, dk, dv, softmax_d };
}

extern "C"
std::vector<at::Tensor>
mha_fwd_padding_mask(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &padding_mask, // batch_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        bool is_bhsd = false) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    // bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // if (q_dtype == torch::kBFloat16) {
    //     TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    // }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    if(print_param){
        printf("mha_fwd_padding_mask fa input size bshd=(%d,%d,%d,%d),p_dropout=%.3f,softmax_scale=%.3f,is_causal=%d,window_size_left=%d,window_size_right=%d,softcap=%f,return_softmax=%d,is_bhsd=%d\n",
        (int)sizes[0],(int)sizes[1],(int)sizes[2],(int)sizes[3],p_dropout,softmax_scale,(int)is_causal,window_size_left,window_size_right,softcap,return_softmax,is_bhsd);
    }

    const int batch_size = sizes[0];
    int seqlen_q = !is_bhsd ? sizes[1] : sizes[2];
    int num_heads = !is_bhsd ? sizes[2] : sizes[1];
    const int head_size_og = sizes[3];
    const int seqlen_k = !is_bhsd ? k.size(1) : k.size(2);
    const int num_heads_k = !is_bhsd ? k.size(2) : k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // padding-mask 能力边界：kernel 只按 headdim=64 实现，且把同一个 padding_mask[i]
    // 当作 Q 的行上界与 K 的有效长度；不支持的组合在此明确报错。
    TORCH_CHECK(head_size_og == 64,
                "fwd_padding_mask only supports headdim == 64, but got headdim = ", head_size_og,
                ". The padding-mask forward kernel is implemented for headdim 64 only.");
    TORCH_CHECK(v.size(3) == head_size_og,
                "fwd_padding_mask requires headdim(v) == headdim(q/k), but got headdim(v) = ", v.size(3),
                " and headdim(q/k) = ", head_size_og);
    TORCH_CHECK(window_size_left < 0 && window_size_right <= 0,
                "fwd_padding_mask does not support sliding window, window_size must be (-1, -1) "
                "(or (-1, 0) for causal), but got (",
                window_size_left, ", ", window_size_right,
                "). With a single per-sample padding length the window alignment cannot match the reference.");
    TORCH_CHECK(!(return_softmax && p_dropout <= 0.f),
                "fwd_padding_mask: return_softmax = true requires p_dropout > 0, otherwise the P buffer is empty.");
    TORCH_CHECK(padding_mask.scalar_type() == torch::kInt32,
                "fwd_padding_mask: padding_mask must be int32, but got ", padding_mask.scalar_type());
    TORCH_CHECK(padding_mask.device() == q.device(),
                "fwd_padding_mask: padding_mask must be on the same device as q");
    CHECK_SHAPE(padding_mask, batch_size);
    {
        const int64_t max_actual_seqlen = std::min<int64_t>(seqlen_q, seqlen_k);
        auto padding_mask_cpu = padding_mask.detach().cpu().contiguous();
        const int32_t *padding_mask_ptr = padding_mask_cpu.data_ptr<int32_t>();
        for (int64_t ib = 0; ib < batch_size; ++ib) {
            const int64_t actual_seqlen = static_cast<int64_t>(padding_mask_ptr[ib]);
            TORCH_CHECK(actual_seqlen >= 1 && actual_seqlen <= max_actual_seqlen,
                        "fwd_padding_mask: invalid padding_mask[", ib, "] = ", actual_seqlen,
                        ", must satisfy 1 <= padding_mask[i] <= min(seqlen_q, seqlen_k) = ", max_actual_seqlen,
                        " (the kernel uses the same length for the query rows and the key columns)");
        }
    }

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    if(!is_bhsd) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    auto opts = q.options();

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if(!is_bhsd) {
            CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og);
        } else {
            CHECK_SHAPE(out, batch_size, sizes[2], sizes[1], head_size_og);
        }
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (!is_bhsd) {
            out = at::zeros({batch_size, seqlen_q, num_heads, head_size_og}, opts);
        } else {
            out = at::zeros({batch_size, num_heads, seqlen_q, head_size_og}, opts);
        }
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax&&p_dropout>0.0f) {
        // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap, 
                     is_bhsd,
                     /*seqlenq_ngroups_swapped=*/false,
                     /*unpadded_lse=*/false,
                     /*d_v=*/head_size,
                     /*d_v_rounded=*/head_size_rounded);
    params.padding_mask = padding_mask.data_ptr<int32_t>();
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_padding_mask(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

extern "C"
void paged_attention(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, //auto,int8,fp8/fp8_e4m3
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_ = c10::nullopt, // ★ Attention Sinks ★
    int layout = 0,
    int window_size_left = -1,
    int window_size_right = -1,
    bool is_causal = false);

extern "C"
void paged_attention_partial(
    torch::Tensor& out_tmp,
    torch::Tensor& query,          // [num_seqs, mtp, num_heads, head_size] or, with q_rows, [rows, num_heads, head_size]
    torch::Tensor& key_cache,      // [num_blocks, num_kv_heads, 64, head_size]
    torch::Tensor& value_cache,    // [num_blocks, num_kv_heads, head_size, 64]
    torch::Tensor& block_tables,   // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,       // [num_seqs]
    int mtp,
    int partition_size,
    int max_num_partitions,
    const c10::optional<torch::Tensor>& q_rows);

extern "C"
std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                const c10::optional<at::Tensor> &s_aux_ = c10::nullopt  // Attention Sinks: precomputed LSE for sink tokens
                ) {
    TORCH_CHECK(softcap <= 0.f,
                "mha_fwd_kvcache: paged KV cache does not support softcap");
    auto q_dtype = q.dtype();
    const auto sizes = q.sizes();
    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype || q_dtype == torch::kFloat8_e4m3fn|| q_dtype == torch::kFloat8_e5m2, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    } else {
        if(q_dtype == torch::kFloat8_e4m3fn|| q_dtype == torch::kFloat8_e5m2){
          out = torch::empty(sizes, opts.dtype(torch::kBFloat16));
        }
        else{
           out = torch::empty(sizes, opts);
        }
    }
    paged_attention(out,q,const_cast<at::Tensor &>(kcache),const_cast<at::Tensor &>(vcache),softmax_scale,const_cast<at::Tensor &>(block_table_.value()),
                const_cast<at::Tensor &>(seqlens_k_.value()),alibi_slopes_,"",c10::nullopt,c10::nullopt,c10::nullopt,0,s_aux_,2,window_size_left,window_size_right,is_causal);
    auto softmax_lse = torch::empty({sizes[0], sizes[2], sizes[1]}, opts.dtype(at::kFloat));
    return {out, softmax_lse};
}

// num_query条请求，所有token数加和为num_tokens
// q -> 		(fp16||bf16) 		[num_tokens, num_q_head, num_dim]
// seq_offsets -> 	(int32||int64) 		[num_query+1]	 				每个query的起始位置
// seq_lens -> 		(int32||int64) 		[num_query] 					每个query的长度 
// max_seq_len -> 	(int32||int64) 		[1] 						num_query中最大长度
// k_caches -> 		(fp16||bf16) 		[num_pages, num_kv_head, page_size, num_dim] 	(num_pages->?)
// v_caches -> 		(fp16||bf16)		[num_pages, num_kv_head, page_size, num_dim] 	(num_pages->?)
// page_tables -> 	(int32||int64) 		[num_query, max_page_num]
// cache_lens -> 	(int32||int64) 		[num_query]
// out ->		(fp16||bf16)		[num_tokens, num_q_head, num_dim]
extern "C"
void kv_cache_attention(void* q,int* seq_offsets,int max_seqlen_q,int max_seqlen_k,void* k_caches,void* v_caches,int* page_tables,int* cache_lens,
    void* out,int num_query,int num_tokens,int num_q_head,int num_kv_head,int num_dim,int num_pages,int page_size,const std::string& dtype,int layout)
{
  int max_page_num = (max_seqlen_k-1)/page_size+1;
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  if(dtype!="fp16")options=options.dtype(torch::kBFloat16);
  auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  at::Tensor k_t = torch::from_blob(k_caches, {num_pages, num_kv_head, page_size, num_dim}, options);
  at::Tensor v_t = torch::from_blob(v_caches, {num_pages, num_kv_head, page_size, num_dim}, options);
  at::Tensor q_lens_t = torch::from_blob(seq_offsets, {num_query+1}, options_int);
  at::Tensor kv_lens_t = torch::from_blob(cache_lens, {num_query}, options_int);
  at::Tensor block_table_t = torch::from_blob(page_tables, {num_query, max_page_num}, options_int);
  double scale = 1.f / sqrt(float(num_dim));
  if(num_query==num_tokens){
    at::Tensor q_t = torch::from_blob(q, {num_tokens,1, num_q_head,num_dim}, options);
    at::Tensor out_t = torch::from_blob(out, {num_tokens,1, num_q_head,num_dim}, options);
    paged_attention(out_t,q_t,k_t,v_t,scale,block_table_t,kv_lens_t,c10::nullopt,"",c10::nullopt,c10::nullopt,c10::nullopt,max_seqlen_k,c10::nullopt,layout,-1,-1,false);
  }
  else{
    #ifdef HAS_HG_DISPATCH
    TORCH_CHECK(layout == 1 || layout == 2, "kv_cache_attention only support bhsd or bshd");
    at::Tensor q_t = torch::from_blob(q, {num_tokens, num_q_head,num_dim}, options);
    at::Tensor out_t = torch::from_blob(out, {num_tokens, num_q_head,num_dim}, options);
    c10::optional<at::Tensor> out_mut=out_t;
    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    hg_prefix_prefill_varlen_fwd(
                q_t, k_t, v_t, out_mut, q_lens_t, nulltensor, kv_lens_t,
                nulltensor, block_table_t, max_seqlen_q, max_seqlen_k, 0.f,
                scale, false, true, -1,
                -1, 0.f, false, layout==2 /*bshd*/,
                nulltensor, nulltensor, nulltensor, nulltensor,
                true /*is_bf16_output*/);
    #endif
  }
}
extern "C"
void cal_paged_attention(void* q,int* seq_offsets,int max_seqlen_q,int max_seqlen_k,void* k_caches,void* v_caches,int* page_tables,int* cache_lens,
    void* out,int num_query,int num_tokens,int num_q_head,int num_kv_head,int num_dim,int num_pages,int page_size,const std::string& dtype)
{
  kv_cache_attention(q,seq_offsets,max_seqlen_q, max_seqlen_k,k_caches,v_caches, page_tables, cache_lens,
    out, num_query, num_tokens, num_q_head, num_kv_head, num_dim, num_pages, page_size, dtype, 1);
}

extern "C"
std::vector<at::Tensor>
mha_fwd_sparse(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &block_count,
               const at::Tensor &block_offset,
               const at::Tensor &column_count,
               const at::Tensor &column_index,
               const std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
               const std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
               const double p_dropout,
               const double softmax_scale,
               bool is_causal,
               const double softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_,
               bool is_sla = false,
               const double pv_threshold = 50.0,      // Dynamic PV skip threshold
               const bool enable_dynamic_skip = true  // Enable dynamic PV skip optimization
               );
extern "C"
std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                      const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &block_count,
                      const at::Tensor &block_offset,
                      const at::Tensor &column_count,
                      const at::Tensor &column_index,
                      const std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                      const at::Tensor &cu_seqlens_q,  // b+1
                      const at::Tensor &cu_seqlens_k,  // b+1
                      const std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                      const std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                      int64_t max_seqlen_q,
                      const int64_t max_seqlen_k,
                      const double p_dropout,
                      const double softmax_scale,
                      const bool zero_tensors,
                      bool is_causal,
                      const double softcap,
                      const bool return_softmax,
                      std::optional<at::Generator> gen_,
                      const double pv_threshold = 50.0,      // Dynamic PV skip threshold
                      const bool enable_dynamic_skip = true  // Enable dynamic PV skip optimization
                      );

extern "C"
std::vector<at::Tensor>
mha_fwd_attnmask(at::Tensor &q,
                 const at::Tensor &k,
                 const at::Tensor &v,
                 const at::Tensor &attn_mask,
                 c10::optional<at::Tensor> &out_,
                 c10::optional<at::Tensor> &alibi_slopes_,
                 const float p_dropout,
                 const float softmax_scale,
                 bool is_causal,
                 int window_size_left,
                 int window_size_right,
                 const float softcap,
                 const float masked_value,
                 const bool return_softmax,
                 c10::optional<at::Generator> gen_,
                 bool is_bhsd,
                 const c10::optional<at::Tensor> &s_aux_,
				 const c10::optional<at::Tensor> &attn_bias_=c10::nullopt);

extern "C"
std::vector<at::Tensor>
mha_varlen_fwd_attnmask(at::Tensor &q,
                        const at::Tensor &k,
                        const at::Tensor &v,
                        const at::Tensor &attn_mask,
                        c10::optional<at::Tensor> &out_,
                        const at::Tensor &cu_seqlens_q,
                        const at::Tensor &cu_seqlens_k,
                        c10::optional<at::Tensor> &seqused_k,
                        c10::optional<at::Tensor> &alibi_slopes_,
                        int max_seqlen_q,
                        const int max_seqlen_k,
                        const float p_dropout,
                        const float softmax_scale,
                        const bool zero_tensors,
                        bool is_causal,
                        int window_size_left,
                        int window_size_right,
                        const float softcap,
                        const float masked_value,
                        const bool return_softmax,
                        c10::optional<at::Generator> gen_,
                        const c10::optional<at::Tensor> &s_aux_);

extern "C"
std::vector<at::Tensor>
mha_bwd_attnmask(const at::Tensor &dout,
                 const at::Tensor &q,
                 const at::Tensor &k,
                 const at::Tensor &v,
                 const at::Tensor &out,
                 const at::Tensor &softmax_lse,
                 const at::Tensor &attn_mask,
                 c10::optional<at::Tensor> &dq_,
                 c10::optional<at::Tensor> &dk_,
                 c10::optional<at::Tensor> &dv_,
                 c10::optional<at::Tensor> &alibi_slopes_,
                 const float p_dropout,
                 const float softmax_scale,
                 const bool is_causal,
                 int window_size_left,
                 int window_size_right,
                 const float softcap,
                 const bool deterministic,
                 c10::optional<at::Generator> gen_,
                 c10::optional<at::Tensor> &rng_state,
                 bool is_bhsd);

extern "C"
std::vector<at::Tensor>
mha_varlen_bwd_attnmask(const at::Tensor &dout,
                        const at::Tensor &q,
                        const at::Tensor &k,
                        const at::Tensor &v,
                        const at::Tensor &out,
                        const at::Tensor &softmax_lse,
                        const at::Tensor &attn_mask,
                        c10::optional<at::Tensor> &dq_,
                        c10::optional<at::Tensor> &dk_,
                        c10::optional<at::Tensor> &dv_,
                        const at::Tensor &cu_seqlens_q,
                        const at::Tensor &cu_seqlens_k,
                        c10::optional<at::Tensor> &alibi_slopes_,
                        const int max_seqlen_q,
                        const int max_seqlen_k,
                        const float p_dropout,
                        const float softmax_scale,
                        const bool zero_tensors,
                        const bool is_causal,
                        int window_size_left,
                        int window_size_right,
                        const float softcap,
                        const bool deterministic,
                        c10::optional<at::Generator> gen_,
                        c10::optional<at::Tensor> &rng_state);

extern "C"
int flash_attn_api_changed() {
    return 1;
}

// ============================================================================
// TORCH_LIBRARY 算子注册 (通过 c10::Dispatcher 暴露接口)
// ============================================================================

static inline c10::optional<at::Tensor> tensor_to_opt(const at::Tensor& t) {
    return t.defined() ? c10::optional<at::Tensor>(t) : c10::nullopt;
}

static inline c10::optional<const at::Tensor> tensor_to_const_opt(const at::Tensor& t) {
    return t.defined() ? c10::optional<const at::Tensor>(t) : c10::nullopt;
}

TORCH_LIBRARY(flash_attn2_c_op, m) {
    m.def(
        "varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor? out, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? seqused_k, "
        "Tensor? leftpad_k_, Tensor? block_table_, Tensor? alibi_slopes_, "
        "int max_seqlen_q, int max_seqlen_k, float p_dropout, "
        "float softmax_scale, bool zero_tensors, bool is_causal, "
        "int window_size_left, int window_size_right, float softcap, "
        "bool return_softmax) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def(
        "fwd_kvcache(Tensor q, Tensor kcache, Tensor vcache, "
        "Tensor? k_, Tensor? v_, Tensor? seqlens_k_, "
        "Tensor? rotary_cos_, Tensor? rotary_sin_, "
        "Tensor? cache_batch_idx_, Tensor? leftpad_k_, "
        "Tensor? block_table_, Tensor? alibi_slopes_, Tensor? out_, "
        "float softmax_scale, bool is_causal, "
        "int window_size_left, int window_size_right, "
        "float softcap, bool is_rotary_interleaved, int num_splits) "
        "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attn2_c_op, CUDA, m) {
    m.impl("varlen_fwd", [](
        const torch::Tensor& q,
        const torch::Tensor& k,
        const torch::Tensor& v,
        const c10::optional<torch::Tensor>& out,
        const torch::Tensor& cu_seqlens_q,
        const torch::Tensor& cu_seqlens_k,
        const c10::optional<torch::Tensor>& seqused_k,
        const c10::optional<torch::Tensor>& leftpad_k_,
        const c10::optional<torch::Tensor>& block_table_,
        const c10::optional<torch::Tensor>& alibi_slopes_,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        double p_dropout,
        double softmax_scale,
        bool zero_tensors,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        double softcap,
        bool return_softmax
    ) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> {
        at::Tensor q_mut = q;
        c10::optional<at::Tensor> out_mut = out;
        c10::optional<at::Tensor> seqused_k_mut = seqused_k;
        c10::optional<const at::Tensor> leftpad_k_mut = leftpad_k_.has_value() ? c10::optional<const at::Tensor>(*leftpad_k_) : c10::nullopt;
        c10::optional<at::Tensor> block_table_mut = block_table_;
        c10::optional<at::Tensor> alibi_slopes_mut = alibi_slopes_;

        auto results = mha_varlen_fwd(
            q_mut, k, v, out_mut,
            cu_seqlens_q, cu_seqlens_k,
            seqused_k_mut,
            leftpad_k_mut,
            block_table_mut,
            alibi_slopes_mut,
            static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
            static_cast<float>(p_dropout), static_cast<float>(softmax_scale),
            zero_tensors, is_causal,
            static_cast<int>(window_size_left), static_cast<int>(window_size_right),
            static_cast<float>(softcap), return_softmax,
            c10::nullopt, c10::nullopt, c10::nullopt,
            c10::nullopt,
            c10::nullopt
        );

        auto p = results[6].defined() ? results[6] : at::empty({0}, q.options());
        auto rng = results[7].defined() ? results[7] : at::empty({0}, q.options().dtype(at::kLong));
        return std::make_tuple(results[0], results[5], p, rng);
    });

    m.impl("fwd_kvcache", [](
        const torch::Tensor& q,
        const torch::Tensor& kcache,
        const torch::Tensor& vcache,
        const c10::optional<torch::Tensor>& k_,
        const c10::optional<torch::Tensor>& v_,
        const c10::optional<torch::Tensor>& seqlens_k_,
        const c10::optional<torch::Tensor>& rotary_cos_,
        const c10::optional<torch::Tensor>& rotary_sin_,
        const c10::optional<torch::Tensor>& cache_batch_idx_,
        const c10::optional<torch::Tensor>& leftpad_k_,
        const c10::optional<torch::Tensor>& block_table_,
        const c10::optional<torch::Tensor>& alibi_slopes_,
        const c10::optional<torch::Tensor>& out_,
        double softmax_scale,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        double softcap,
        bool is_rotary_interleaved,
        int64_t num_splits
    ) -> std::tuple<torch::Tensor, torch::Tensor> {
        at::Tensor q_mut = q;
        c10::optional<const at::Tensor> seqlens_k_opt = seqlens_k_.has_value() ? c10::optional<const at::Tensor>(*seqlens_k_) : c10::nullopt;
        c10::optional<const at::Tensor> rotary_cos_opt = rotary_cos_.has_value() ? c10::optional<const at::Tensor>(*rotary_cos_) : c10::nullopt;
        c10::optional<const at::Tensor> rotary_sin_opt = rotary_sin_.has_value() ? c10::optional<const at::Tensor>(*rotary_sin_) : c10::nullopt;
        c10::optional<const at::Tensor> cache_batch_idx_opt = cache_batch_idx_.has_value() ? c10::optional<const at::Tensor>(*cache_batch_idx_) : c10::nullopt;
        c10::optional<const at::Tensor> leftpad_k_opt = leftpad_k_.has_value() ? c10::optional<const at::Tensor>(*leftpad_k_) : c10::nullopt;
        c10::optional<at::Tensor> block_table_mut = block_table_;
        c10::optional<at::Tensor> alibi_slopes_mut = alibi_slopes_;
        c10::optional<at::Tensor> out_mut = out_;
        c10::optional<const at::Tensor> nulltensor = c10::nullopt;
        if(block_table_.has_value()){
            auto results = mha_fwd_kvcache(
                q_mut, kcache, vcache,
                nulltensor, nulltensor, seqlens_k_opt,
                rotary_cos_opt, rotary_sin_opt,
                cache_batch_idx_opt, leftpad_k_opt,
                block_table_mut, alibi_slopes_mut, out_mut,
                static_cast<float>(softmax_scale), is_causal,
                static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                static_cast<float>(softcap), is_rotary_interleaved,
                static_cast<int>(num_splits),
                c10::nullopt
            );
            return std::make_tuple(results[0], results[1]);
        }
        else{
           c10::optional<at::Generator> gen_mut=c10::nullopt;
           auto results = mha_fwd(q_mut,*k_,*v_,out_mut,alibi_slopes_mut,0.0f,
                  static_cast<float>(softmax_scale),
                  is_causal,
                  static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                  static_cast<float>(softcap),
                  false,
                  gen_mut
                  );
           return std::make_tuple(results[0], results[5]);
        }
    });
}
at::Tensor mean_pool_fast(const at::Tensor &input,int blk,const c10::optional<at::Tensor> &mean);
extern "C"
void pagedkv_to_contiguouskv(
    torch::Tensor& k,             // [total_tokens, num_heads, head_size]
    torch::Tensor& v,             // [total_tokens, num_heads, head_size]
    torch::Tensor& key_cache,     // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    torch::Tensor& cu_seqlens_k,   // [num_seqs+1]
    int max_seq_len,
    int layout);

// Sparse GQA chunk-prefill. Python and C++ use scale before the optional
// Python max-query-length argument (which is concrete at this C++ boundary).
at::Tensor sparse_gqa_attention(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &indices,
    const at::Tensor &cu_q,
    const at::Tensor &cu_k,
    const at::Tensor &kv_lens,
    double scale,
    int64_t max_seqlen_q);

at::Tensor sparse_gqa_attention_fp8(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &indices, const at::Tensor &cu_q,
    const at::Tensor &cu_k, const at::Tensor &kv_lens,
    double scale, double k_scale, double v_scale, int64_t max_seqlen_q);

std::vector<at::Tensor> mqa_paged_score(
    const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &page_table,
    const at::Tensor &token_to_req, const at::Tensor &query_positions,
    const at::Tensor &sequence_lengths, int64_t compress_ratio,
    c10::optional<int64_t> num_columns, c10::optional<double> score_scale);

at::Tensor sparse_gqa_paged_attention(
    const at::Tensor &q, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &logical_indices, const at::Tensor &block_table,
    const at::Tensor &token_to_req, c10::optional<at::Tensor> out);

at::Tensor sparse_gqa_paged_attention_fp8(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, c10::optional<at::Tensor>,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<std::string>);

std::vector<at::Tensor> fa_sparse_attention_prefill(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, int64_t, c10::optional<double>);
std::vector<at::Tensor> fa_sparse_attention_decode(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t, c10::optional<double>, int64_t);
std::vector<at::Tensor> fa_sparse_attention_prefill_fp8(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, int64_t, c10::optional<double>, double, double);
std::vector<at::Tensor> fa_sparse_attention_decode_fp8(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t, c10::optional<double>, int64_t,
    double, double);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
#ifdef HAS_HG_DISPATCH
    m.def("hg_prefix_prefill_varlen_fwd", &hg_prefix_prefill_varlen_fwd, "HG prefix prefill forward pass (variable length)");
    m.def("hg_prefix_decode_varlen_fwd", &hg_prefix_decode_varlen_fwd, "HG prefix decode forward pass (variable length)");
#endif
    m.def("fwd_padding_mask", &mha_fwd_padding_mask, "Forward pass, with padding_mask");
    m.def("fwd_attnmask", &mha_fwd_attnmask, "Forward pass, with explicit attention mask");
    m.def("varlen_fwd_attnmask", &mha_varlen_fwd_attnmask, "Forward pass (variable length), with explicit attention mask");
    m.def("bwd_attnmask", &mha_bwd_attnmask, "Backward pass, with explicit attention mask");
    m.def("varlen_bwd_attnmask", &mha_varlen_bwd_attnmask, "Backward pass (variable length), with explicit attention mask");
    m.def("paged_attention", &paged_attention, "Forward pass, with KV-cache");
    m.def("paged_attention_partial", &paged_attention_partial,
          "Paged attention writing per-partition partials (out, max, exp_sum) to out_tmp; caller merges",
          py::arg("out_tmp"), py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_tables"), py::arg("seq_lens"), py::arg("mtp"), py::arg("partition_size"),
          py::arg("max_num_partitions"), py::arg("q_rows") = py::none());
    m.def("pagedkv_to_contiguouskv", &pagedkv_to_contiguouskv, "Forward pass, pagedkv_to_contiguouskv");
    m.def("fwd_sparse", &mha_fwd_sparse, "Forward sparse pass");
    m.def("fwd_sparse_mean_pool_fast", &mean_pool_fast, "before mha_fwd_sparse");
    m.def("varlen_fwd_sparse", &mha_varlen_fwd_sparse, "Forward pass sparse (variable length)");
    m.def("sparse_gqa_attention", &sparse_gqa_attention,
          "Sparse GQA forward (packed chunk-prefill)");
    m.def("sparse_gqa_attention_fp8", &sparse_gqa_attention_fp8,
          "Sparse GQA forward with FP8 KV and FP16/BF16 Q/output");
    m.def("mqa_paged_score", &mqa_paged_score,
          "Paged MQA scores and visible compressed-block counts");
    m.def("sparse_gqa_paged_attention", &sparse_gqa_paged_attention,
          "Sparse GQA forward directly over paged BF16 KV");
    m.def("sparse_gqa_paged_attention_fp8", &sparse_gqa_paged_attention_fp8,
          "Paged FP8 KV sparse GQA with BF16 Q/output and device FP32 scales");
    m.def("fa_sparse_attention_prefill", &fa_sparse_attention_prefill,
          "BF16 sparse prefill with natural-log LSE");
    m.def("fa_sparse_attention_decode", &fa_sparse_attention_decode,
          "BF16 sparse decode returning final output and natural-log LSE");
    m.def("fa_sparse_attention_prefill_fp8", &fa_sparse_attention_prefill_fp8,
          "E4M3FN KV sparse prefill with BF16 Q/output and natural-log LSE");
    m.def("fa_sparse_attention_decode_fp8", &fa_sparse_attention_decode_fp8,
          "E4M3FN KV sparse decode returning final BF16 output and natural-log LSE");
}
