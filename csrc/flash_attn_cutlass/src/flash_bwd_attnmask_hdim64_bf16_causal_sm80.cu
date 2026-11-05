// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_attnmask_launch_template.h"

template void run_mha_bwd_attnmask_<hytlass::bfloat16_t, 64, true>(
    Flash_bwd_params_attnmask &params, cudaStream_t stream);
