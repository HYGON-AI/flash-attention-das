/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <hute/tensor.hpp>

#include <hytlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace hute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    HUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        #if defined(__gfx928__) 
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            // float ori = summary(mi);
            summary(mi) = op(summary(mi), tensor(mi, ni));
            // wangaq debug
            // if (thread0()) {
            //     printf("thread_reduce_ mi:%d ni:%d %7.4f %7.4f %7.4f\n", mi, ni, ori, tensor(mi, ni), summary(mi));
            // }
        }
        #else
        if constexpr (std::is_same_v<Operator, SumOp<float>>) {
            using __float2  = __attribute__((ext_vector_type(2))) float;
            __float2 sum_v = {zero_init ? 0.0f : summary(mi), 0.0f};
            if constexpr (decltype(size<1>(tensor))::value == 16) {
                for (int ni = 0; ni < size<1>(tensor) / 4; ni ++) {
                    __float2 vx2_0 = {tensor(mi, ni), tensor(mi, ni + 4)};
                    __float2 vx2_1 = {tensor(mi, ni + 8), tensor(mi, ni + 12)};
                    __float2 tmp_res = __builtin_hcu_pk_add_f32(vx2_1, vx2_0);
                    sum_v = __builtin_hcu_pk_add_f32(sum_v, tmp_res);
                }
            } else {
                for (int ni = 0; ni < size<1>(tensor); ni += 2) {
                    __float2 vx2 = {tensor(mi, ni), tensor(mi, ni + 1)};
                    sum_v = __builtin_hcu_pk_add_f32(sum_v, vx2);
                }
            }

            summary(mi) = sum_v.x + sum_v.y;
        }
        else {
            summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
            #pragma unroll
            for (int ni = 1; ni < size<1>(tensor); ni++) {
                // float ori = summary(mi);
                summary(mi) = op(summary(mi), tensor(mi, ni));
                // wangaq debug
                // if (thread0()) {
                //     printf("thread_reduce_ mi:%d ni:%d %7.4f %7.4f %7.4f\n", mi, ni, ori, tensor(mi, ni), summary(mi));
                // }
            }
        }
        #endif
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    HUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<64>::run(src(i), op);
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f \n", threadIdx.x,
        //     src(i), dst(i));
        // }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_sum_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    HUTE_STATIC_ASSERT_V(size(dst) == size(src));
    int tidx = threadIdx.x % 64;
    float a, b = 1.0;
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        v4f d = {0};
        a = src(i);
        d = __builtin_amdgcn_mmac_f32_16x16x4f32(a, b, d);
        dst(i) = d.x;
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f "
        //     "D:%10.4f %10.4f %10.4f %10.4f sum:%7.4f\n", threadIdx.x,
        //     a, b,
        //     d[0], d[1], d[2], d[3], dst(i));
        // }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_with_mmac_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    HUTE_STATIC_ASSERT_V(size(dst) == size(src));
    int tidx = threadIdx.x % 64;
    float a, b = 0.0 + (tidx / 4 == 0 && tidx / 16 == 0) + (tidx / 4 == 5 && tidx / 16 == 1) + 
        (tidx / 4 == 10 && tidx / 16 == 2) + (tidx / 4 == 15 && tidx / 16 == 3);
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        v4f d = {0};
        a = src(i) == -INFINITY ? -10000.0 : src(i);
        d = __builtin_amdgcn_mmac_f32_16x16x4f32(a, b, d);
        dst(i) = isnan(d.x) ? -INFINITY : d.x;
        dst(i) = op(dst(i), isnan(d.y) ? -INFINITY : d.y);
        dst(i) = op(dst(i), isnan(d.z) ? -INFINITY : d.z);
        dst(i) = op(dst(i), isnan(d.w) ? -INFINITY : d.w);
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f "
        //     "D:%10.4f %10.4f %10.4f %10.4f max:%7.4f\n", threadIdx.x,
        //     a, b,
        //     d[0], d[1], d[2], d[3], dst(i));
        // }
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    #if 1
    quad_allreduce_(summary, summary, op);
    #else
    quad_allreduce_with_mmac_(summary, summary, op);
    #endif
    
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    HUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #if defined(__gfx928__) 
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            tensor(mi, ni) = __builtin_amdgcn_exp2f(tensor(mi, ni) * scale - max_scaled);
        }
        #else
        using __float2  = __attribute__((ext_vector_type(2))) float;
        __float2 scalex2 = {scale, scale};
        __float2 max_scaledx2 = {-max_scaled, -max_scaled};
        if constexpr (decltype(size<1>(tensor))::value == 16) {
           for (int ni = 0; ni < 4; ni ++)  {
                __float2 vx2_0 = {tensor(mi, ni), tensor(mi, ni + 4)};
                __float2 vx2_1 = {tensor(mi, ni + 8), tensor(mi, ni + 12)};
                __float2  res_0 = __builtin_hcu_pk_fma_f32(vx2_0, scalex2, max_scaledx2);
                __float2  res_1 = __builtin_hcu_pk_fma_f32(vx2_1, scalex2, max_scaledx2);
                tensor(mi, ni) = __builtin_amdgcn_exp2f(res_0.x);
                tensor(mi, ni + 4) = __builtin_amdgcn_exp2f(res_0.y);
                tensor(mi, ni + 8) = __builtin_amdgcn_exp2f(res_1.x);
                tensor(mi, ni + 12) = __builtin_amdgcn_exp2f(res_1.y);
            }
        } else {
            for (int ni = 0; ni < size<1>(tensor); ni += 2)  {
                __float2 vx2 = {tensor(mi, ni), tensor(mi, ni + 1)};
                __float2  res = __builtin_hcu_pk_fma_f32(vx2, scalex2, max_scaledx2);
                tensor(mi, ni) = __builtin_amdgcn_exp2f(res.x);
                tensor(mi, ni + 1) = __builtin_amdgcn_exp2f(res.y);
            }
        }

        #endif
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void fast_reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    // When performing a 4-thread reduce, use 3 shfl + 3 fmaxf instead of
    // 2 shfl + 2 fmaxf operations to shorten the critical path and hide the
    // latency of the ds_bpermute instruction.
    // The 3 shfl operations are done parallelly, and the 3 fmaxf operations are done during the 3 shfl operations.
    MaxOp<float> max_op;
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    HUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    Tensor max1 = make_fragment_like(max);
    Tensor max2 = make_fragment_like(max);
    Tensor max3 = make_fragment_like(max);
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        // Use use-def chain to construct hard schedule barrier
        asm volatile("" : "+v"(tensor(mi, 0)) : : "memory");
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max1(mi) = __shfl_xor(max(mi), 16, 64);
        max2(mi) = __shfl_xor(max(mi), 32, 64);
        max3(mi) = __shfl_xor(max(mi), 48, 64);
        asm volatile("" : "+v"(max(mi)) : : "memory");
        if (mi > 0) {
            asm volatile("" : "+v"(max(mi - 1)) : : "memory");
            max(mi - 1) = max_op(max(mi - 1), max1(mi - 1));
            max(mi - 1) = max_op(max(mi - 1), max2(mi - 1));
            max(mi - 1) = max_op(max(mi - 1), max3(mi - 1));
            asm volatile("" : "+v"(max(mi - 1)) : : "memory");
        }
    }
    int mi = size<0>(tensor) - 1;
    asm volatile("" : "+v"(max(mi)) : : "memory");
    max(mi) = max_op(max(mi), max1(mi));
    max(mi) = max_op(max(mi), max2(mi));
    max(mi) = max_op(max(mi), max3(mi));
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void fast_reduce_sum_f16_f32(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    // Use v_pk_add_f16 instruction to accelerate the sum of fp16 numbers.
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    HUTE_STATIC_ASSERT_V(size<0>(sum) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        uint32_t a = tensor(mi, 0);
        #pragma unroll
        for (int ni = 2; ni < size<1>(tensor); ni += 2) {
            asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(a) : "v"(a), "v"(tensor(mi, ni)));
        }
        uint16_t lo = a & 0xffff;
        uint16_t hi = (a & 0xffff0000) >> 16;
        hytlass::half_t acc = reinterpret_cast<const hytlass::half_t&>(lo) + reinterpret_cast<const hytlass::half_t&>(hi);
        if (zero_init) {
            sum(mi) = (float)acc;
        } else {
            sum(mi) += (float)acc;
        }
    }
}

static constexpr float kSqrt2 = 1.414213562f;
static constexpr float kFastExpBias = 0.113f;  // reduce fast-exp LSE positive bias.
static constexpr float kFastExpScale = 1.5f - kFastExpBias + (1.0f - 0.5f * kFastExpBias) * kSqrt2;
static constexpr float kFastExpScaleInv = 1.0f / kFastExpScale;
static constexpr float kFp16Bf16Scale = 5.192296859e33f/*2^112*/;

// Apply the exp to all the elements.
template <bool Check_inf,typename Engine0, typename Layout0, typename EngineH, typename LayoutH, typename EngineO, typename LayoutO, typename Engine1, typename Layout1>
__forceinline__ __device__ void fast_scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<EngineH, LayoutH> &tensor_h, Tensor<EngineO, LayoutO> &tensor_o, Tensor<Engine1, Layout1> const &max, const float scale) {
    using TypeO = typename EngineO::value_type;
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    HUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    // Use `bit_cast_fp16(int16(x * kB + kC))` to fit `exp2(x)`, with an error of +-3%.
    // Since `exp2(x)` is expected to be between 0 and 1, we can use
    // `cvt_norm_u16(a / 65536)` instead of `int16(a)`.
    // We use fp16 rather than fp32 or bf16 to optimize funther calculation.
    constexpr float kB = float(1 << 10) / 65536;
    //  // Add an offset of -0.113 to reduce the error of the critical terms (when x = 0.0f).
    constexpr float kC = 0b1111 * kB - kFastExpBias * kB;
    constexpr float sm_scale = 0.0883883476f*1.4426950408889634f;
    constexpr float neg_sm_scale = -sm_scale;
    // Simplify `(x * scale - max_scaled) * kB + kC` to `x * b + c`
    constexpr float b = sm_scale * kB;
    constexpr float halfkb = kB * 0.5f;
    if constexpr (std::is_same_v<TypeO, hytlass::bfloat16_t>
                  && decltype(size<1>(tensor))::value == 16) {
        using __float2 = __attribute__((ext_vector_type(2))) float;
        const __float2 bx2 = {b, b};
        const __float2 halfkbx2 = {halfkb, halfkb};
        Tensor row_offsets = make_tensor<float>(layout(max));
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            float max_scaled_neg = Check_inf? (max(mi) == -INFINITY ? 0.0f : max(mi) * neg_sm_scale):max(mi) * neg_sm_scale;
            row_offsets(mi) = kC + max_scaled_neg * kB;
        }
        #pragma unroll
        for (int ni = 0; ni < 16; ni += 2) {
            if (ni & 4) continue;
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                const float c = row_offsets(mi);
                __float2 a04 = {tensor(mi, ni), tensor(mi, ni + 4)};
                __float2 a15 = {tensor(mi, ni + 1), tensor(mi, ni + 5)};
                const __float2 cx2 = {c, c};
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(a04) : "v"(a04), "v"(bx2), "v"(cx2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(a15) : "v"(a15), "v"(bx2), "v"(cx2));
                tensor(mi, ni) = a04.x;
                tensor(mi, ni + 4) = a04.y;
                tensor(mi, ni + 1) = a15.x;
                tensor(mi, ni + 5) = a15.y;
                uint32_t i0, i4, j0, j4;
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i0) : "v"(a04.x), "v"(a15.x));
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i4) : "v"(a04.y), "v"(a15.y));
                a04 = __builtin_hcu_pk_add_f32(a04, halfkbx2);
                a15 = __builtin_hcu_pk_add_f32(a15, halfkbx2);
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(j0) : "v"(a04.x), "v"(a15.x));
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(j4) : "v"(a04.y), "v"(a15.y));
                asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(i0) : "v"(i0), "v"(0x3da83da8), "v"(j0));
                asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(i4) : "v"(i4), "v"(0x3da83da8), "v"(j4));
                tensor_h(mi, ni) = i0;
                tensor_h(mi, ni + 4) = i4;
                asm volatile("v_pk_lshrrev_b16 %0, %1, %2" : "=v"(i0) : "v"(0x00030003), "v"(i0));
                asm volatile("v_pk_lshrrev_b16 %0, %1, %2" : "=v"(i4) : "v"(0x00030003), "v"(i4));
                uint32_t lo0 = i0 & 0xffff, hi0 = i0 >> 16;
                uint32_t lo4 = i4 & 0xffff, hi4 = i4 >> 16;
                tensor_o(mi, ni) = reinterpret_cast<const hytlass::bfloat16_t&>(lo0);
                tensor_o(mi, ni + 1) = reinterpret_cast<const hytlass::bfloat16_t&>(hi0);
                tensor_o(mi, ni + 4) = reinterpret_cast<const hytlass::bfloat16_t&>(lo4);
                tensor_o(mi, ni + 5) = reinterpret_cast<const hytlass::bfloat16_t&>(hi4);
            }
        }
        return;
    }
    // (x * kB + kC) / 65536:
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float max_scaled_neg = Check_inf?  (max(mi) == -INFINITY ? 0.0f : max(mi) * neg_sm_scale):max(mi) * neg_sm_scale;
        float c = kC + max_scaled_neg * kB;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            if (ni & 4) continue;
            if (ni + 4 >= size<1>(tensor)) {
                tensor(mi, ni) = tensor(mi, ni) * b + c;
                continue;
            }
            float2 a = {tensor(mi, ni), tensor(mi, ni + 4)};
#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
            asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(a) : "v"(a), "v"(float2{b, b}), "v"(float2{c, c}));
#else
            a.x = a.x * b + c;
            a.y = a.y * b + c;
#endif
            tensor(mi, ni) = a.x;
            tensor(mi, ni + 4) = a.y;
        }
    }
    // bit_cast_fp16(cvt_norm_u16(a))
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            if (ni & 1) continue;
            if (ni + 1 >= size<1>(tensor)) {
                float a = tensor(mi, ni);
                uint32_t i, i1;
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i) : "v"(0), "v"(a));
                asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i1) : "v"(0), "v"(a + halfkb));
                asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(i) : "v"(i), "v"(0x3da83da8), "v"(i1));
                tensor_h(mi, ni) = i;
                if constexpr (std::is_same_v<TypeO, hytlass::bfloat16_t>) {
                    // asm volatile("v_pk_add_u16 %0, %1, %2" : "=v"(i) : "v"(0x00080008), "v"(i));
                    asm volatile("v_pk_lshrrev_b16 %0, %1, %2" : "=v"(i) : "v"(0x00030003), "v"(i));
                    i &= 0xffff;
                    tensor_o(mi, ni) = reinterpret_cast<const hytlass::bfloat16_t&>(i);
                } else {
                    i &= 0xffff;
                    tensor_o(mi, ni) = reinterpret_cast<const hytlass::half_t&>(i);
                }
                continue;
            }
            float2 a;
            a.x = tensor(mi, ni);
            a.y = tensor(mi, ni + 1);
            uint32_t i, i1;
            asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i) : "v"(a.x), "v"(a.y));
            asm volatile("v_cvt_pknorm_u16_f32 %0, %1, %2" : "=v"(i1) : "v"(a.x + halfkb), "v"(a.y + halfkb));
            // to increase nonlinearity and reduce the error to +-0.7%.
            // The additional factor of `1.387 + 0.9435 * sqrt(2)` is canceled out by softmax.
            asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(i) : "v"(i), "v"(0x3da83da8), "v"(i1));
            tensor_h(mi, ni) = i;
            if constexpr (std::is_same_v<TypeO, hytlass::bfloat16_t>) {
                // Cast fp16 to bf16, leaving the exp_bias to be handled in the `normalize_softmax_lse` stage.
                //  asm volatile("v_pk_add_u16 %0, %1, %2" : "=v"(i) : "v"(0x00080008), "v"(i));
                asm volatile("v_pk_lshrrev_b16 %0, %1, %2" : "=v"(i) : "v"(0x00030003), "v"(i));
                uint32_t ix = i & 0xffff, iy = i >> 16;
                tensor_o(mi, ni) = reinterpret_cast<const hytlass::bfloat16_t&>(ix);
                tensor_o(mi, ni + 1) = reinterpret_cast<const hytlass::bfloat16_t&>(iy);
            } else {
                uint32_t ix = i & 0xffff, iy = i >> 16;
                tensor_o(mi, ni) = reinterpret_cast<const hytlass::half_t&>(ix);
                tensor_o(mi, ni + 1) = reinterpret_cast<const hytlass::half_t&>(iy);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, bool SkipIdentityRescale = false>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float skip_softmax_threshold;
    uint32_t total_blocks;
    uint32_t skipped_blocks;

    __forceinline__ __device__ Softmax() : skip_softmax_threshold(0.f), total_blocks(0), skipped_blocks(0)  {};

    template<typename Element, bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ auto fast_softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor out_s = make_tensor<Element>(layout(acc_s));
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor scores_out = make_tensor(out_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor scores_f16 = make_tensor<uint32_t>(layout(scores));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template fast_reduce_max</*zero_init=*/true>(scores, row_max);
            flash::fast_scale_apply_exp2<Check_inf>(scores, scores_f16, scores_out, row_max, softmax_scale_log2);
            flash::fast_reduce_sum_f16_f32</*zero_init=*/true>(scores_f16, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            constexpr bool kFinishMaxLate = std::is_same_v<Element, hytlass::bfloat16_t>
                && decltype(size<1>(scores))::value == 16;
            Tensor scores_quad1 = make_fragment_like(row_max);
            Tensor scores_quad2 = make_fragment_like(row_max);
            Tensor scores_quad3 = make_fragment_like(row_max);
            MaxOp<float> scores_max_op;
            if constexpr (kFinishMaxLate) {
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    asm volatile("" : "+v"(scores(mi, 0)) : : "memory");
                    row_max(mi) = scores_max_op(row_max(mi), scores(mi, 0));
                    #pragma unroll
                    for (int ni = 1; ni < size<1>(scores); ++ni) {
                        row_max(mi) = scores_max_op(row_max(mi), scores(mi, ni));
                    }
                    scores_quad1(mi) = __shfl_xor(row_max(mi), 16, 64);
                    scores_quad2(mi) = __shfl_xor(row_max(mi), 32, 64);
                    scores_quad3(mi) = __shfl_xor(row_max(mi), 48, 64);
                    asm volatile("" : "+v"(row_max(mi)) : : "memory");
                }
            } else {
                flash::template fast_reduce_max</*zero_init=*/false>(scores, row_max);
            }
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                if constexpr (kFinishMaxLate) {
                    asm volatile("" : "+v"(row_max(mi)) : : "memory");
                    row_max(mi) = scores_max_op(row_max(mi), scores_quad1(mi));
                    row_max(mi) = scores_max_op(row_max(mi), scores_quad2(mi));
                    row_max(mi) = scores_max_op(row_max(mi), scores_quad3(mi));
                    if (mi + 1 < size(row_max)) {
                        asm volatile("" : "+v"(row_max(mi)) : : "memory");
                    }
                }
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                if (scores_max_prev(mi) == scores_max_cur) continue;
                // Use regular exp2f to avoid error accumulation.
                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                using __float2 = __attribute__((ext_vector_type(2))) float;
                const __float2 scores_scalex2 = {scores_scale, scores_scale};
                constexpr int kAccCols = decltype(size<1>(acc_o_rowcol))::value;
                #pragma unroll
                for (int ni = 0; ni + 1 < kAccCols; ni += 2) {
                    const __float2 vx2 = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                    const __float2 res = __builtin_hcu_pk_mul_f32(vx2, scores_scalex2);
                    acc_o_rowcol(mi, ni) = res.x;
                    acc_o_rowcol(mi, ni + 1) = res.y;
                }
                if constexpr (kAccCols % 2 != 0) {
                    acc_o_rowcol(mi, kAccCols - 1) *= scores_scale;
                }
            }
            flash::fast_scale_apply_exp2<Check_inf>(scores, scores_f16, scores_out, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::fast_reduce_sum_f16_f32</*zero_init=*/false>(scores_f16, row_sum);
        }
        return out_s;
    };

    template<typename Element, bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT fast_normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = 1.f / sum;
            // Cancel out the factor that was added in the `fast_scale_apply_exp2` stage.
            lse(mi) =  row_max(mi) * softmax_scale + __logf(sum * kFastExpScaleInv);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            if constexpr (std::is_same_v<Element, hytlass::bfloat16_t>) {
                // Cancel out the factor of 2^(-112) that was added in the `fast_scale_apply_exp2` stage.
                scale *= kFp16Bf16Scale;
            }
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };


    // ★ Attention Sinks: normalize with precomputed sink LogSumExp ★
    template<typename Element, bool Is_dropout=false, bool Split=false, typename Tensor0, typename TensorSAux>
    __forceinline__ __device__ TensorT fast_normalize_softmax_lse_with_sinks(
        Tensor0 &acc_o,
        TensorSAux const& tSrS_aux,
        float softmax_scale,
        float softmax_scale_log2,
        float rp_dropout=1.0
    ) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif

        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            // Handle -INFINITY case for empty sequences
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            const float max_scaled = row_max(mi) * softmax_scale_log2;

            // Compute sink tokens' contribution to softmax denominator
            // exp(s_aux - max/√d) = exp2(log2(e) * s_aux - max * log2(e) / √d)
            #ifndef M_LOG2E
            #define M_LOG2E 1.44269504088896340736
            #endif
            // Scale up by to keep the same scale as fast_scale_apply_exp2.
            float sink_contrib = __builtin_amdgcn_exp2f(float(M_LOG2E) * tSrS_aux(mi) - max_scaled) * kFastExpScale;

            float sum = row_sum(mi) + sink_contrib;
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

            // Cancel out the factor that was added in the `fast_scale_apply_exp2` stage.
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY)
                                                  : row_max(mi) * softmax_scale + __logf(sum * kFastExpScaleInv);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            if constexpr (std::is_same_v<Element, hytlass::bfloat16_t>) {
                // Cancel out the factor of 2^(-112) that was added in the `fast_scale_apply_exp2` stage.
                scale *= kFp16Bf16Scale;
            }
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scale;
            }
        }
        return lse;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                // Preserve exceptional-value behavior, including zero times an infinite scale.
                if (SkipIdentityRescale && scores_max_prev(mi) == scores_max_cur
                    && isfinite(scores_max_cur) && isfinite(softmax_scale_log2)) {
                    continue;
                }
                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #if defined(__gfx928__) 
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
                #else
                using __float2  = __attribute__((ext_vector_type(2))) float;
                __float2 scores_scalex2 = {scores_scale, scores_scale};
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) { 
                    __float2 vx2 = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                    __float2  res = __builtin_hcu_pk_mul_f32(vx2, scores_scalex2);
                    acc_o_rowcol(mi, ni) = res.x; 
                    acc_o_rowcol(mi, ni + 1) = res.y; 
                }
                #endif
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
        
    };
 
    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ bool softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2, uint32_t * skip_softmax_vote) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            total_blocks++;
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
            return false;
        } else {
            total_blocks++;
            bool skip = true;
            float scores_scale[kNRows];
            Tensor scores_max_prev = make_fragment_like(row_max);
            Tensor scores_max_local = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/true>(scores, scores_max_local);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

            MaxOp<float> max_op;
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                skip &= (__builtin_amdgcn_exp2f((scores_max_local(mi) - scores_max_prev(mi)) * softmax_scale_log2) < skip_softmax_threshold);
                scores_max_local(mi) = max_op(scores_max_local(mi), scores_max_prev(mi));
            }

            skip = __all_sync((uint64_t)0xffffffffffffffff, skip);
            if (threadIdx.x % 64 == 0) {
                // The leader of each warp votes.
                atomicAnd(skip_softmax_vote, uint32_t(skip));
            }
            // __syncthreads();
            s_barrier();
            // asm volatile("s_waitcnt lgkmcnt(0); s_barrier\n");
            // skip = *((uint32_t volatile*) skip_softmax_vote);
            uint32_t skip_vote;
            int skip_softmax_vote_addr = reinterpret_cast<size_t>(skip_softmax_vote);
            asm volatile("ds_read_b32 %0, %1 offset:0\n" : "=v"(skip_vote) : "v"(skip_softmax_vote_addr) :);
            asm volatile("s_waitcnt lgkmcnt(0); s_barrier\n");
            if (skip_vote)
            {
                skipped_blocks++;
                return true;
            }

            hute::copy(scores_max_local, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
            return false;
        }
        
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, Tensor2 &acc_o_tail, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            Tensor acc_o_tail_rowcol = make_tensor(acc_o_tail.data(), flash::convert_layout_acc_rowcol(acc_o_tail.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_tail_rowcol); ++ni) { acc_o_tail_rowcol(mi, ni) *= scores_scale; }
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
        
    };
    template<bool Is_first, bool Check_inf=false,
            typename Tensor0,
            typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4>
    __forceinline__ __device__ void softmax_rescale_o(
        Tensor0 &acc_s,
        Tensor1 &acc_o0, Tensor2 &acc_o1, Tensor3 &acc_o2, Tensor4 &acc_o3,
        float softmax_scale_log2) {

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        static_assert(decltype(size<0>(scores))::value == kNRows);

        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);

            // === 将四个 acc_o 都转为 rowcol 布局 ===
            Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
            Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
            Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));
            Tensor acc_o3_rowcol = make_tensor(acc_o3.data(), flash::convert_layout_acc_rowcol(acc_o3.layout()));

            static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));

                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;

                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                    acc_o0_rowcol(mi, ni) *= scores_scale;
                    acc_o1_rowcol(mi, ni) *= scores_scale;
                    acc_o2_rowcol(mi, ni) *= scores_scale;
                    acc_o3_rowcol(mi, ni) *= scores_scale;
                }
            }

            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };
    template<bool Is_first, bool Check_inf=false,
            typename Tensor0,
            typename Tensor1, typename Tensor2, typename Tensor3>
    __forceinline__ __device__ void softmax_rescale_o(
        Tensor0 &acc_s,
        Tensor1 &acc_o0, Tensor2 &acc_o1, Tensor3 &acc_o2,
        float softmax_scale_log2) {

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        static_assert(decltype(size<0>(scores))::value == kNRows);

        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);

            // === 将四个 acc_o 都转为 rowcol 布局 ===
            Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
            Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
            Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));

            static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));

                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;

                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                    acc_o0_rowcol(mi, ni) *= scores_scale;
                    acc_o1_rowcol(mi, ni) *= scores_scale;
                    acc_o2_rowcol(mi, ni) *= scores_scale;
                }
            }

            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    // Softmax rescale with max_diff return for dynamic PV skip optimization
    // Returns max_diff = max(current_block_local_max - previous_global_max) following SpargeAttn convention
    // Execute P@V when: max_diff + pv_threshold > 0 (current block contribution significant)
    // Skip P@V when: max_diff + pv_threshold <= 0 (current block contribution negligible)
    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ float softmax_rescale_o_with_diff(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);

        float local_max_diff = -INFINITY;

        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // Note: row_sum will be initialized in accumulate_softmax_sum() for first block
            // First block must always compute P@V, return +INFINITY to force execution
            local_max_diff = INFINITY;
        } else {
            // ========== OPTIMIZED: Align with SpargeAttn, minimize overhead ==========
            // Step 1: Save previous global max
            Tensor scores_max_prev = make_fragment_like(row_max);
            hute::copy(row_max, scores_max_prev);

            // Step 2: Compute current block's LOCAL max into row_max temporarily
            // This overwrites row_max with local max (will restore cumulative later)
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);

            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

            // Step 3: Compute max_diff and update to cumulative max in single pass
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                // row_max now contains LOCAL max from current block
                float scores_max_cur_local = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));

                float scores_max_prev_val = scores_max_prev(mi);

                // Compute max_diff = local_max - global_max (can be negative!)
                // This matches SpargeAttn convention (attn_utils.cuh:445)
                float row_diff = (scores_max_cur_local - scores_max_prev_val) * softmax_scale_log2;
                local_max_diff = max(local_max_diff, row_diff);

                // Update row_max to cumulative max for rescaling
                float scores_max_new_global = max(scores_max_prev_val, scores_max_cur_local);
                row_max(mi) = scores_max_new_global;

                // Rescale previous accumulations if global max increased
                if (SkipIdentityRescale && scores_max_prev_val == scores_max_new_global
                    && isfinite(scores_max_new_global) && isfinite(softmax_scale_log2)) {
                    continue;
                }
                float scores_scale = __builtin_amdgcn_exp2f((scores_max_prev_val - scores_max_new_global) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;

                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_scale;
                }
            }
            // Compute exp(scores - max) for P@V, but don't accumulate to row_sum yet
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // NOTE: row_sum accumulation is deferred to accumulate_softmax_sum()
        }

        return local_max_diff;
    };

    // Accumulate softmax probabilities to row_sum (denominator)
    template<bool Is_first, typename Tensor0>
    __forceinline__ __device__ void accumulate_softmax_sum(Tensor0 &acc_s) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);

        // Accumulate exp(scores) to row_sum
        // acc_s already contains exp(scores - max) from softmax_rescale_o_with_diff
        flash::reduce_sum</*zero_init=*/Is_first>(scores, row_sum);
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #if defined(__gfx928__) 
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
            #else
            using __float2  = __attribute__((ext_vector_type(2))) float;
            __float2 scores_scalex2 = {scale, scale};
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) { 
                __float2 vx2 = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                __float2  res = __builtin_hcu_pk_mul_f32(vx2, scores_scalex2);
                acc_o_rowcol(mi, ni) = res.x; 
                acc_o_rowcol(mi, ni + 1) = res.y; 
            }
            #endif
        }
        return lse;
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, Tensor1 &acc_o_tail, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        Tensor acc_o_tail_rowcol = make_tensor(acc_o_tail.data(), flash::convert_layout_acc_rowcol(acc_o_tail.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
            for (int ni = 0; ni < size<1>(acc_o_tail_rowcol); ++ni) { acc_o_tail_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
    template<bool Is_dropout=false, bool Split=false,
         typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ TensorT normalize_softmax_lse(
        Tensor0 &acc_o0, Tensor1 &acc_o1, Tensor2 &acc_o2,
        float softmax_scale, float rp_dropout=1.0) {

        SumOp<float> sum_op;
    #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
    #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
    #endif

        TensorT lse = make_fragment_like(row_sum);

        // === 将四个 acc_o 转换为 rowcol 布局 ===
        Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
        Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
        Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));

        static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o0_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

            lse(mi) = (sum == 0.f || sum != sum)
                ? (Split ? -INFINITY : INFINITY)
                : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;

            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                acc_o0_rowcol(mi, ni) *= scale;
                acc_o1_rowcol(mi, ni) *= scale;
                acc_o2_rowcol(mi, ni) *= scale;
            }
        }

        return lse;
    }
    template<bool Is_dropout=false, bool Split=false,
         typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3>
    __forceinline__ __device__ TensorT normalize_softmax_lse(
        Tensor0 &acc_o0, Tensor1 &acc_o1, Tensor2 &acc_o2, Tensor3 &acc_o3,
        float softmax_scale, float rp_dropout=1.0) {

        SumOp<float> sum_op;
    #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
    #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
    #endif

        TensorT lse = make_fragment_like(row_sum);

        // === 将四个 acc_o 转换为 rowcol 布局 ===
        Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
        Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
        Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));
        Tensor acc_o3_rowcol = make_tensor(acc_o3.data(), flash::convert_layout_acc_rowcol(acc_o3.layout()));

        static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o0_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

            lse(mi) = (sum == 0.f || sum != sum)
                ? (Split ? -INFINITY : INFINITY)
                : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;

            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                acc_o0_rowcol(mi, ni) *= scale;
                acc_o1_rowcol(mi, ni) *= scale;
                acc_o2_rowcol(mi, ni) *= scale;
                acc_o3_rowcol(mi, ni) *= scale;
            }
        }

        return lse;
    }


    // ★ Attention Sinks: normalize with precomputed sink LogSumExp ★
    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename TensorSAux>
    __forceinline__ __device__ TensorT normalize_softmax_lse_with_sinks(
        Tensor0 &acc_o,
        TensorSAux const& tSrS_aux,
        float softmax_scale,
        float softmax_scale_log2,
        float rp_dropout=1.0
    ) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif

        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            // Handle -INFINITY case for empty sequences
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            const float max_scaled = row_max(mi) * softmax_scale_log2;

            // Compute sink tokens' contribution to softmax denominator
            // exp(s_aux - max/√d) = exp2(log2(e) * s_aux - max * log2(e) / √d)
            #ifndef M_LOG2E
            #define M_LOG2E 1.44269504088896340736
            #endif
            float sink_contrib = __builtin_amdgcn_exp2f(float(M_LOG2E) * tSrS_aux(mi) - max_scaled);

            float sum = row_sum(mi) + sink_contrib;
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY)
                                                  : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scale;
            }
        }
        return lse;
    };

    // ★ Attention Sinks: normalize with precomputed sink LogSumExp (with tail for VLLM) ★
    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename Tensor1, typename TensorSAux>
    __forceinline__ __device__ TensorT normalize_softmax_lse_with_sinks_tail(
        Tensor0 &acc_o,
        Tensor1 &acc_o_tail,
        TensorSAux const& tSrS_aux,
        float softmax_scale,
        float softmax_scale_log2,
        float rp_dropout=1.0
    ) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif

        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        Tensor acc_o_tail_rowcol = make_tensor(acc_o_tail.data(), flash::convert_layout_acc_rowcol(acc_o_tail.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            // Handle -INFINITY case for empty sequences
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            const float max_scaled = row_max(mi) * softmax_scale_log2;

            // Compute sink tokens' contribution to softmax denominator
            // exp(s_aux - max/√d) = exp2(log2(e) * s_aux - max * log2(e) / √d)
            #ifndef M_LOG2E
            #define M_LOG2E 1.44269504088896340736
            #endif
            float sink_contrib = __builtin_amdgcn_exp2f(float(M_LOG2E) * tSrS_aux(mi) - max_scaled);

            float sum = row_sum(mi) + sink_contrib;
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY)
                                                  : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scale;
            }
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_tail_rowcol); ++ni) {
                acc_o_tail_rowcol(mi, ni) *= scale;
            }
        }
        return lse;
    };
	
	template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse_fp8(Tensor0 &acc_o, float softmax_scale, float v_descale,float rp_dropout=1.0) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
     
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : v_descale / sum;

            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ TensorT normalize_softmax_lse_fp8(Tensor0 &acc_o, Tensor1 &acc_o_tail, float softmax_scale, float v_descale,float rp_dropout=1.0) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        Tensor acc_o_tail_rowcol = make_tensor(acc_o_tail.data(), flash::convert_layout_acc_rowcol(acc_o_tail.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : v_descale / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
            for (int ni = 0; ni < size<1>(acc_o_tail_rowcol); ++ni) { acc_o_tail_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };

    template<bool Is_dropout=false, bool Split=false,
         typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ TensorT normalize_softmax_lse_fp8(
        Tensor0 &acc_o0, Tensor1 &acc_o1, Tensor2 &acc_o2,
        float softmax_scale, float v_scale=1.0, float rp_dropout=1.0) {

        SumOp<float> sum_op;
    #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
    #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
    #endif

        TensorT lse = make_fragment_like(row_sum);

        // === 将四个 acc_o 转换为 rowcol 布局 ===
        Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
        Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
        Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));

        static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o0_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : v_scale / sum;

            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;

            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                acc_o0_rowcol(mi, ni) *= scale;
                acc_o1_rowcol(mi, ni) *= scale;
                acc_o2_rowcol(mi, ni) *= scale;
            }
        }

        return lse;
    }

    template<bool Is_dropout=false, bool Split=false,
         typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3>
    __forceinline__ __device__ TensorT normalize_softmax_lse_fp8(
        Tensor0 &acc_o0, Tensor1 &acc_o1, Tensor2 &acc_o2, Tensor3 &acc_o3,
        float softmax_scale, float v_scale=1.0, float rp_dropout=1.0) {

        SumOp<float> sum_op;
    #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
    #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
    #endif

        TensorT lse = make_fragment_like(row_sum);

        // === 将四个 acc_o 转换为 rowcol 布局 ===
        Tensor acc_o0_rowcol = make_tensor(acc_o0.data(), flash::convert_layout_acc_rowcol(acc_o0.layout()));
        Tensor acc_o1_rowcol = make_tensor(acc_o1.data(), flash::convert_layout_acc_rowcol(acc_o1.layout()));
        Tensor acc_o2_rowcol = make_tensor(acc_o2.data(), flash::convert_layout_acc_rowcol(acc_o2.layout()));
        Tensor acc_o3_rowcol = make_tensor(acc_o3.data(), flash::convert_layout_acc_rowcol(acc_o3.layout()));

        static_assert(decltype(size<0>(acc_o0_rowcol))::value == kNRows);

        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o0_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : v_scale / sum;

            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);

            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;

            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o0_rowcol); ++ni) {
                acc_o0_rowcol(mi, ni) *= scale;
                acc_o1_rowcol(mi, ni) *= scale;
                acc_o2_rowcol(mi, ni) *= scale;
                acc_o3_rowcol(mi, ni) *= scale;
            }
        }

        return lse;
    }

};

}  // namespace flash
