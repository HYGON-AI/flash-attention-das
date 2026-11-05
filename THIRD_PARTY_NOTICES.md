# Third-Party Notices

SPDX-License-Identifier: BSD-3-Clause

This file records the third-party components and source attributions used by
`flash-attention-das`. Original copyright notices and license terms are retained
in the corresponding source files. Component license texts are distributed in
[LICENSES](LICENSES/).

## FlashAttention

- Source: https://github.com/Dao-AILab/flash-attention
- Baseline tag: `v2.6.1`
- Baseline commit: `7551202cb2dd245432bc878447e19015c0af3c22`
- License: BSD-3-Clause; the complete original text is retained in [LICENSE](LICENSE).
- Original copyright: Copyright (c) 2022, the respective contributors, as shown
  by the AUTHORS file. 
- Local scope: FlashAttention-derived kernels, Python interfaces, support code,
  tests and documentation. Local `csrc/flash_attn_cutlass/` corresponds to upstream
  `csrc/flash_attn/`. Local `csrc/rotary/rotary_kernel.cu` corresponds to upstream
  `csrc/rotary/rotary_cuda.cu` and retains Copyright (c) 2023, Tri Dao.
  `csrc/flash_attn_cutlass/src/flash_varlen_fwd_tiny_hdim64.cu` retains
  Copyright (c) 2024, Tri Dao. and is recorded under BSD-3-Clause.

## HYTLASS dependency and local register layouts

- Source: https://github.com/HYGON-AI/hytlass
- Local path: `csrc/cutlass/`.
- Branch: `release`.
- Pinned commit: `4fad709bb75d1854daca5c3343886b4e2cc2da74`.
- License: BSD-3-Clause. The complete pinned root `LICENSE.txt` is reproduced in
  [LICENSES/HYTLASS-BSD-3-Clause.txt](LICENSES/HYTLASS-BSD-3-Clause.txt), including
  the Hygon and NVIDIA license texts and the GoogleTest notice.
- Original copyright: Copyright (c) 2023 - 2025 Hygon Information Technology
  Co., Ltd.; Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES;
  Copyright 2008, Google Inc. (GoogleTest). Dependency file-level notices are retained.
- Local scope: the HYTLASS/HuTe dependency and
  `csrc/flash_attn_cutlass/src/hytlass_layouts.h`.
- Register-layout source references: `include/hute/arch/mma_gfx938.hpp`,
  `include/hute/atom/mma_traits_gfx938.hpp` and corresponding gfx928 headers.
- Local register-layout notices: Copyright (c) 2026 Hygon Information Technology
  Co., Ltd.; Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
  The complete BSD-3-Clause source notice is retained in `hytlass_layouts.h`.

## NVIDIA CUTLASS numeric conversion notices

- Source: https://github.com/NVIDIA/cutlass
- Related dependency: HYTLASS at the pinned commit above.
- License: BSD-3-Clause; the complete NVIDIA copyright and license text is
  retained in `csrc/flash_attn_cutlass/src/fa_numeric_conversion.h`.
- Original copyright: Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES.
- Local scope: `csrc/flash_attn_cutlass/src/fa_numeric_conversion.h`.
- Header references: `include/cutlass/numeric_conversion.h` and
  `include/hytlass/numeric_conversion.h`.

## AOTriton

- Source: [ROCm/aotriton](https://github.com/ROCm/aotriton).
- Local paths: `flash_attn/attn_torch_function.py`, `flash_attn/bwd_preprocess.py`,
  `flash_attn/bwd_split_kernel.py`, `flash_attn/dropout_rng.py`,
  `flash_attn/flash.py` and `flash_attn/fwd_kernel.py`.
- Reference branch: `main`; the fixed reference is reachable from this branch.
- Fixed source reference: `00ccbf3c63278a1b863236c2384a79c327b64ba4`.
- License: MIT; the complete fixed-reference root license is retained in
  [LICENSES/AOTriton-MIT.txt](LICENSES/AOTriton-MIT.txt).
- Original copyright:
  - Root license: Copyright (C) 2024 Advanced Micro Devices, Inc.
  - `tritonsrc/dropout_rng.py`: Copyright © 2024 Advanced Micro Devices, Inc.
  - The other five mapped source files: Copyright © 2023-2024
    Advanced Micro Devices, Inc.
- Local scope: Triton attention forward/backward kernels, dropout RNG helpers,
  and their autograd/autotuning wrapper. Local module imports and the wrapper
  are adapted for this package. Original source credits remain in the files.

| Local path | Path at the fixed AOTriton reference |
| --- | --- |
| `flash_attn/attn_torch_function.py` | `tritonsrc/attn_torch_function.py` |
| `flash_attn/bwd_preprocess.py` | `tritonsrc/bwd_preprocess.py` |
| `flash_attn/bwd_split_kernel.py` | `tritonsrc/bwd_split_kernel.py` |
| `flash_attn/dropout_rng.py` | `tritonsrc/dropout_rng.py` |
| `flash_attn/flash.py` | `tritonsrc/flash.py` |
| `flash_attn/fwd_kernel.py` | `tritonsrc/fwd_kernel.py` |

## ROCm Triton

- Source: [ROCm/triton](https://github.com/ROCm/triton).
- Local paths: `flash_attn/flash_attn_triton.py` and
  `flash_attn/flash_attn_triton_mqa_gqa.py`.
- Reference branch: `backup/main_perf_06_19_2024`, whose head is the fixed
  reference below. This is the preserved historical `main_perf` snapshot;
  the current `main_perf` head does not contain this reference.
- Fixed source reference: `ea80aae0a0c4b73748e3e8a0c08e89ab78a55f95`.
- License: MIT; the complete fixed-reference root license is retained in
  [LICENSES/Triton-MIT.txt](LICENSES/Triton-MIT.txt).
- Original copyright in the root license: Copyright 2018-2020 Philippe Tillet;
  Copyright 2020-2022 OpenAI.
- Retained source credits: OpenAI kernel team in `flash_attn_triton.py`;
  OpenAI kernel team and AMD ML Frameworks Triton team in
  `flash_attn_triton_mqa_gqa.py`. These credits are recorded separately from
  the root-license copyright declarations.
- Local scope: the two adapted Python/Triton fused-attention implementations,
  including their local autograd, test and benchmark code. The mapping below
  identifies the public reference implementations; the local files include
  adaptations and are not byte-for-byte copies of that snapshot.

| Local path | Path at the fixed ROCm Triton reference |
| --- | --- |
| `flash_attn/flash_attn_triton.py` | `python/tutorials/06-fused-attention.py` |
| `flash_attn/flash_attn_triton_mqa_gqa.py` | `python/perf-kernels/flash-attention.py` |

## SpargeAttn

- Source: [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn).
- Local path: `flash_attn/utils/sparse_utils.py`.
- Reference branch: `main`; the fixed reference is reachable from this branch.
- Fixed source reference: `bfd980b781784c04ad6a53e7ee657c0645d99171`.
- Upstream path: `spas_sage_attn/utils.py`.
- License: Apache-2.0; the complete fixed-reference root license is retained in
  [LICENSES/SpargeAttn-Apache-2.0.txt](LICENSES/SpargeAttn-Apache-2.0.txt).
- Original source-file copyright: Copyright (c) 2025 by SpargeAttn team.
- Original root-license copyright: Copyright 2025 Jintao Zhang,
  Chendong Xiang, Haofeng Huang.
- Local scope: sparsity-selection and quantization helper code used by the
  local sparse/SLA attention integration, with HCU-specific adaptations and
  extensions. The original Apache-2.0 source notice remains in the local file.
- Local modification notice: Modified by Hygon Information Technology Co., Ltd.,
  2026; Copyright (c) 2026 Hygon Information Technology Co., Ltd.
- Upstream NOTICE: the complete fixed-reference tree contains no NOTICE file.

## Source Inventory

| Local path | Project | Repository URL | Fixed reference / commit | Upstream path / source notice | Copyright and attribution | License |
| --- | --- | --- | --- | --- | --- | --- |
| `flash_attn/attn_torch_function.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/attn_torch_function.py` | Copyright © 2023-2024 Advanced Micro Devices, Inc. | MIT |
| `flash_attn/bwd_preprocess.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/bwd_preprocess.py` | Copyright © 2023-2024 Advanced Micro Devices, Inc.<br>Source credits include Tri Dao and the original FlashAttention paper, Rabe and Staats, and Adam P. Goucher for simplified vector math. | MIT |
| `flash_attn/bwd_split_kernel.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/bwd_split_kernel.py` | Copyright © 2023-2024 Advanced Micro Devices, Inc.<br>Source credits include Tri Dao and the original FlashAttention paper, Rabe and Staats, and Adam P. Goucher for simplified vector math. | MIT |
| `flash_attn/dropout_rng.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/dropout_rng.py` | Copyright © 2024 Advanced Micro Devices, Inc. | MIT |
| `flash_attn/flash.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/flash.py` | Copyright © 2023-2024 Advanced Micro Devices, Inc.<br>Source credits include Tri Dao and the original FlashAttention paper, Rabe and Staats, and Adam P. Goucher for simplified vector math. | MIT |
| `flash_attn/fwd_kernel.py` | ROCm/aotriton | https://github.com/ROCm/aotriton | `00ccbf3c63278a1b863236c2384a79c327b64ba4` | `tritonsrc/fwd_kernel.py` | Copyright © 2023-2024 Advanced Micro Devices, Inc.<br>Source credits include Tri Dao and the original FlashAttention paper, Rabe and Staats, and Adam P. Goucher for simplified vector math. | MIT |
| `flash_attn/flash_attn_triton.py` | ROCm/triton | https://github.com/ROCm/triton | `ea80aae0a0c4b73748e3e8a0c08e89ab78a55f95` | `python/tutorials/06-fused-attention.py` | Copyright 2018-2020 Philippe Tillet and Copyright 2020-2022 OpenAI (upstream MIT license).<br>Source credits include the OpenAI kernel team, Tri Dao and the original FlashAttention paper, and Rabe and Staats. | MIT |
| `flash_attn/flash_attn_triton_mqa_gqa.py` | ROCm/triton | https://github.com/ROCm/triton | `ea80aae0a0c4b73748e3e8a0c08e89ab78a55f95` | `python/perf-kernels/flash-attention.py` | Copyright 2018-2020 Philippe Tillet and Copyright 2020-2022 OpenAI (upstream MIT license).<br>Source credits include the OpenAI kernel team, the AMD ML Frameworks Triton team, and Tri Dao for the FlashAttention v2 algorithm. | MIT |
| `flash_attn/utils/sparse_utils.py` | SpargeAttn | https://github.com/thu-ml/SpargeAttn | `bfd980b781784c04ad6a53e7ee657c0645d99171` | `spas_sage_attn/utils.py` | Copyright (c) 2025 by SpargeAttn team.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd.<br>Modified by Hygon Information Technology Co., Ltd., 2026. | Apache-2.0 |
| `csrc/flash_attn_cutlass/` | FlashAttention | https://github.com/Dao-AILab/flash-attention | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/` | Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file. Original file-level notices are retained. | BSD-3-Clause; separately listed components retain their own licenses |
| `csrc/rotary/rotary_kernel.cu` | FlashAttention | https://github.com/Dao-AILab/flash-attention | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/rotary/rotary_cuda.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/cutlass/` | HYTLASS / HuTe | https://github.com/HYGON-AI/hytlass | `4fad709bb75d1854daca5c3343886b4e2cc2da74` | Dependency tree at `4fad709bb75d1854daca5c3343886b4e2cc2da74` | Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.<br>Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.<br>Copyright 2008, Google Inc. (GoogleTest). Other original file-level notices are retained. | BSD-3-Clause at the component level. Eight gfx928 GEMM unit-test headers declare BSD-2-Clause in SPDX but include a non-endorsement clause in the license text. |
| `csrc/flash_attn_cutlass/src/hytlass_layouts.h` | HYTLASS / HuTe | https://github.com/HYGON-AI/hytlass | `4fad709bb75d1854daca5c3343886b4e2cc2da74` | `include/hute/arch/mma_gfx938.hpp`, `include/hute/atom/mma_traits_gfx938.hpp` and corresponding gfx928 headers | Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.<br>Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/fa_numeric_conversion.h` | NVIDIA CUTLASS / HYTLASS | https://github.com/NVIDIA/cutlass ; https://github.com/HYGON-AI/hytlass | HYTLASS reference: `4fad709bb75d1854daca5c3343886b4e2cc2da74` | `include/cutlass/numeric_conversion.h`, `include/hytlass/numeric_conversion.h` | Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_varlen_fwd_tiny_hdim64.cu` | FlashAttention | https://github.com/Dao-AILab/flash-attention | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | Retained FlashAttention source notice | Copyright (c) 2024, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_hipc/flash_api.cpp` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/flash_api.cpp`; locally adapted HIP interfaces, parameter setup and dispatch | Current local notice: Copyright (c) 2024, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_hipc/include/block_info.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/src/block_info.h`; local KV-cache and tensor-layout adaptations | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_hipc/include/flash.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/src/flash.h`; locally adapted HIP parameter structures and additional parameters | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_hipc/include/kernel_traits.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/src/kernel_traits.h`; locally adapted element types, tile parameters and kernel traits | Current local notice: Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_hipc/src/flash_bwd_kernel.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/src/flash_bwd_kernel.h`; retained backward-kernel code with local HIP adaptations | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_hipc/src/flash_bwd_launch_template.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Comparison baseline: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22`; original import revision pending verification | `csrc/flash_attn/src/flash_bwd_launch_template.h`; locally adapted HIP backward-kernel launch and architecture dispatch | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/flash_api_attnmask.cpp` | FlashAttention-derived mask interface | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Public comparison base: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/flash_api.cpp`; locally derived interface with explicit-mask parameter setup and forward/backward dispatch | Copyright (c) 2024, Tri Dao.; Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/flash_api_sparse.cpp` | FlashAttention-derived sparse interface | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Public comparison base: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/flash_api.cpp`; locally derived interface with sparse parameter setup and forward dispatch | Copyright (c) 2024, Tri Dao.; Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_attnmask_kernel.h` | FlashAttention-derived mask backward kernel | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Public comparison base: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_kernel.h`; locally derived backward implementation applying explicit masks when recomputing attention scores | Copyright (c) 2024, Tri Dao.; Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_attnmask_launch_template.h` | FlashAttention-derived mask backward launch template | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Public comparison base: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_launch_template.h`; locally derived masked backward launch and dispatch for head dimensions 64 and 128 | Copyright (c) 2024, Tri Dao.; Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_fwd_attnmask_launch_template.h` | FlashAttention-derived mask forward launch template | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Public comparison base: `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_fwd_launch_template.h`; locally derived forward launch and dispatch for mask, bias and additive variants | Copyright (c) 2023, Tri Dao.; Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_kernel.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_kernel.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_launch_template.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_launch_template.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_preprocess_kernel.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_fwd_kernel.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_fwd_kernel.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_fwd_launch_template.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_fwd_launch_template.h` | Copyright (c) 2023, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/kernel_traits.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/kernel_traits.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/mask.h` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/mask.h` | Copyright (c) 2024, Tri Dao.<br>Copyright (c) 2026 Hygon Information Technology Co., Ltd. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim32_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim32_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim64_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim64_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim96_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim96_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim128_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim128_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim160_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim160_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim192_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim192_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim256_bf16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
| `csrc/flash_attn_cutlass/src/flash_bwd_hdim256_fp16_sm80.cu` | FlashAttention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `v2.6.1` / `7551202cb2dd245432bc878447e19015c0af3c22` | `csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu` | Copyright (c) 2023, Tri Dao. | BSD-3-Clause |
