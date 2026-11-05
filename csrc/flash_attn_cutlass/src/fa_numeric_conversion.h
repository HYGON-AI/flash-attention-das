/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <hytlass/numeric_conversion.h>
#include <type_traits>

namespace flash {

// Keep the conversion policy of FA's former CUTLASS dependency local to FA.
template <typename To, typename From,
          hytlass::FloatRoundStyle Round = hytlass::FloatRoundStyle::round_to_nearest>
struct NumericConverter : hytlass::NumericConverter<To, From, Round> {};

template <typename To, hytlass::FloatRoundStyle Round>
struct NumericConverter<To, float, Round> {
    using result_type = To;
    using source_type = float;
    static constexpr auto round_style = Round;

    __forceinline__ __device__ static result_type convert(source_type const &source) {
        if constexpr (std::is_same_v<To, hytlass::bfloat16_t> &&
                      (Round == hytlass::FloatRoundStyle::round_half_ulp_truncate ||
                       Round == hytlass::FloatRoundStyle::round_to_nearest ||
                       Round == hytlass::FloatRoundStyle::round_toward_zero)) {
            uint32_t bits;
            __builtin_memcpy(&bits, &source, sizeof(bits));
            if constexpr (Round == hytlass::FloatRoundStyle::round_half_ulp_truncate) {
                bits += 0x8000u;
            } else if constexpr (Round == hytlass::FloatRoundStyle::round_to_nearest) {
                bits += 0x7fffu + ((bits >> 16) & 1u);
            }
            return hytlass::bfloat16_t::bitcast(uint16_t(bits >> 16));
#if defined(HIP_FP8_CVT_ENABLE)
        } else if constexpr (std::is_same_v<To, hytlass::float_e4m3_t> ||
                             std::is_same_v<To, hytlass::float_e5m2_t>) {
            uint32_t bits;
            __builtin_memcpy(&bits, &source, sizeof(bits));
            float value = source;
            if ((bits & 0x7f800000u) != 0x7f800000u) {
                if constexpr (std::is_same_v<To, hytlass::float_e4m3_t>) {
                    value = __builtin_amdgcn_fmed3f(value, 448.0f, -448.0f);
                } else {
                    // Retain the former scalar converter's saturation bound.
                    value = __builtin_amdgcn_fmed3f(value, 57344.0f, -57334.0f);
                }
            }
            uint32_t packed;
            if constexpr (std::is_same_v<To, hytlass::float_e4m3_t>) {
                packed = __builtin_hcu_cvt_pk_fp8_f32(value, value, 0, false);
            } else {
                packed = __builtin_hcu_cvt_pk_bf8_f32(value, value, 0, false);
            }
            return To::bitcast(uint8_t(packed));
#endif
        } else {
            return hytlass::NumericConverter<To, float, Round>{}(source);
        }
    }

    __forceinline__ __device__ result_type operator()(source_type const &source) const {
        return convert(source);
    }
};

template <typename To, typename From, int N,
          hytlass::FloatRoundStyle Round = hytlass::FloatRoundStyle::round_to_nearest>
struct NumericArrayConverter {
    using result_type = hytlass::Array<To, N>;
    using source_type = hytlass::Array<From, N>;
    static constexpr auto round_style = Round;

    __forceinline__ __device__ static result_type convert(source_type const &source) {
        if constexpr (std::is_same_v<From, float> && std::is_same_v<To, hytlass::half_t> &&
                      Round == hytlass::FloatRoundStyle::round_to_nearest) {
            result_type result;
#if defined(__HIP_DEVICE_COMPILE__) && (defined(MIX_FP16_EPILOGUE) || \
    (defined(HCU_ASM) && (defined(__gfx906__) || defined(__gfx928__) || \
                         defined(__gfx936__) || defined(__gfx938__))))
            #pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                auto packed = __builtin_amdgcn_cvt_pkrtz(source[2 * i], source[2 * i + 1]);
                __builtin_memcpy(reinterpret_cast<char *>(result.data()) + 2 * i * sizeof(To),
                                 &packed, sizeof(packed));
            }
            if constexpr (N % 2 != 0) {
                result[N - 1] = NumericConverter<To, From, Round>{}(source[N - 1]);
            }
#else
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                result[i] = NumericConverter<To, From, Round>{}(source[i]);
            }
#endif
            return result;
        } else if constexpr (std::is_same_v<From, float> &&
                             std::is_same_v<To, hytlass::bfloat16_t>) {
            result_type result;
#if defined(__gfx938__) || defined(__gfx92a__)
            if constexpr (Round == hytlass::FloatRoundStyle::round_to_nearest ||
                          Round == hytlass::FloatRoundStyle::round_half_ulp_truncate) {
                #pragma unroll
                for (int i = 0; i < N / 2; ++i) {
                    auto packed = __builtin_hcu_cvt_pk_bf16_f32(0, source[2 * i], 0, source[2 * i + 1], 0);
                    __builtin_memcpy(reinterpret_cast<char *>(result.data()) + 2 * i * sizeof(To),
                                     &packed, sizeof(packed));
                }
                if constexpr (N % 2 != 0) {
                    result[N - 1] = NumericConverter<To, From, Round>{}(source[N - 1]);
                }
                return result;
            }
#endif
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                result[i] = NumericConverter<To, From, Round>{}(source[i]);
            }
            return result;
        } else if constexpr (std::is_same_v<From, float> &&
                             (std::is_same_v<To, hytlass::float_e4m3_t> ||
                              std::is_same_v<To, hytlass::float_e5m2_t>)) {
            result_type result;
            int tail = 0;
#if defined(HIP_FP8_CVT_ENABLE)
            #pragma unroll
            for (int i = 0; i < N / 4; ++i) {
                uint32_t packed;
                if constexpr (std::is_same_v<To, hytlass::float_e4m3_t>) {
                    packed = __builtin_hcu_cvt_pk_fp8_f32(source[4 * i], source[4 * i + 1], 0, false);
                    packed = __builtin_hcu_cvt_pk_fp8_f32(source[4 * i + 2], source[4 * i + 3], packed, true);
                } else {
                    packed = __builtin_hcu_cvt_pk_bf8_f32(source[4 * i], source[4 * i + 1], 0, false);
                    packed = __builtin_hcu_cvt_pk_bf8_f32(source[4 * i + 2], source[4 * i + 3], packed, true);
                }
                __builtin_memcpy(reinterpret_cast<char *>(result.data()) + 4 * i, &packed, sizeof(packed));
            }
            tail = (N / 4) * 4;
#endif
            #pragma unroll
            for (int i = tail; i < N; ++i) {
                result[i] = NumericConverter<To, From, Round>{}(source[i]);
            }
            return result;
        } else {
            return hytlass::NumericArrayConverter<To, From, N, Round>{}(source);
        }
    }

    __forceinline__ __device__ result_type operator()(source_type const &source) const {
        return convert(source);
    }
};

} // namespace flash
