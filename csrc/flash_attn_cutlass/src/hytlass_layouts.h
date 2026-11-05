/***************************************************************************************************
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hute/tensor.hpp>

// These layouts describe FA's register packing and bank-conflict avoidance.
// In particular, *_BLayout tags are for tiled copies, not MMAC execution.
namespace flash::hytlass_ext {

// FA explicitly packs B before the LIT instruction; retain its original logical view.
struct GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT
    : hute::GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT {};

struct GFX928_DS_READ_DS_M32x16_B16 : hute::GFX928_DS_READ_DS_M32x16_B16 {};
struct GFX928_DS_READ_DS_M32x16_B16_WITH_8x64 : hute::GFX928_DS_READ_DS_M32x16_B16 {};
struct GFX928_DS_READ_DS_M32x32_B8 : hute::GFX928_DS_READ_DS_M32x32_B8 {};
struct GFX928_16x32x16_F32F16F16F32_NT : hute::GFX928_16x32x16_F32F16F16F32_NT {};
struct GFX928_16x32x16_F32BF16BF16F32_NT : hute::GFX928_16x32x16_F32BF16BF16F32_NT {};
struct GFX928_16x64x32_F32F16F16F32_NN : hute::GFX928_16x64x32_F32F16F16F32_NT {};
struct GFX928_16x64x32_F32BF16BF16F32_NN : hute::GFX928_16x64x32_F32BF16BF16F32_NT {};
struct GFX928_16x64x32_F32F16F16F32_NT_BLayout : hute::GFX928_16x64x32_F32F16F16F32_NT {};
struct GFX928_16x64x32_F32BF16BF16F32_NT_BLayout : hute::GFX928_16x64x32_F32BF16BF16F32_NT {};
struct GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout : hute::GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT {};

} // namespace flash::hytlass_ext

namespace hute {

template <>
struct MMA_Traits<flash::hytlass_ext::GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT>
    : MMA_Traits<GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT> {
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_2, _8>>,
                         Stride<Stride<_1, _256>, Stride<_16, _32>>>;
};

template <>
struct Copy_Traits<flash::hytlass_ext::GFX928_DS_READ_DS_M32x16_B16>
{
  using ThrID = Layout<_64>;
  using SrcLayout = Layout<Shape<Shape<_4, _16>, _128>,
                           Stride<Stride<_128, _512>, _1>>;
  using DstLayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2, _4>>,
                           Stride<Stride<_16, _2048>, Stride<_1, _256, _512>>>;
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<flash::hytlass_ext::GFX928_DS_READ_DS_M32x16_B16_WITH_8x64>
{
  using ThrID = Layout<_64>;
  using SrcLayout = Layout<Shape<Shape<_8, _8>, _128>,
                           Stride<Stride<_128, _1024>, _1>>;
  using DstLayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2, _4>>,
                           Stride<Stride<_16, _2048>, Stride<_1, _256, _512>>>;
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<flash::hytlass_ext::GFX928_DS_READ_DS_M32x32_B8>
{
  using ThrID = Layout<_64>;
  using SrcLayout = Layout<Shape<Shape<_2, _32>, _128>,
                           Stride<Stride<_128, _256>, _1>>;
  // Match FA's FP8 V register packing before its row/column layout conversion.
  using DstLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2, _8>>,
                           Stride<Stride<_8, _2048>, Stride<_1, _128, _256>>>;
  using RefLayout = DstLayout;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _32, _16>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _4>,
                         Stride<Stride<_1, _64>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_2, _4>>,
                         Stride<Stride<_1, _128>, Stride<_16, _32>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_8>>,
                         Stride<Stride<_1, _16>, Stride<_64>>>;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x32x16_F32BF16BF16F32_NT>
    : MMA_Traits<flash::hytlass_ext::GFX928_16x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x64x32_F32F16F16F32_NN>
    : MMA_Traits<GFX928_16x64x32_F32F16F16F32_NT>
{
  using BLayout = Layout<Shape<Shape<_4, _16>, Shape<_8, _4>>,
                         Stride<Stride<_512, _1>, Stride<_64, _16>>>;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x64x32_F32BF16BF16F32_NN>
    : MMA_Traits<GFX928_16x64x32_F32BF16BF16F32_NT>
{
  using BLayout = Layout<Shape<Shape<_4, _16>, Shape<_8, _4>>,
                         Stride<Stride<_512, _1>, Stride<_64, _16>>>;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x64x32_F32F16F16F32_NT_BLayout>
    : MMA_Traits<GFX928_16x64x32_F32F16F16F32_NT>
{
  using Shape_MNK = Shape<_16, _32, _64>;
  using BLayout = Layout<Shape<Shape<_8, _8>, Shape<_8, _4>>,
                         Stride<Stride<_256, _1>, Stride<_32, _8>>>;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    : MMA_Traits<GFX928_16x64x32_F32BF16BF16F32_NT>
{
  using Shape_MNK = Shape<_16, _32, _64>;
  using BLayout = Layout<Shape<Shape<_8, _8>, Shape<_8, _4>>,
                         Stride<Stride<_256, _1>, Stride<_32, _8>>>;
};

template <>
struct MMA_Traits<flash::hytlass_ext::GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>
    : MMA_Traits<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>
{
  using BLayout = Layout<Shape<Shape<_4, _16>, Shape<_16, _4>>,
                         Stride<Stride<_1024, _1>, Stride<_64, _16>>>;
};

} // namespace hute
