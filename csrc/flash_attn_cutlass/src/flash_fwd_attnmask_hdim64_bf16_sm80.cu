// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_attnmask_launch_template.h"

template void run_mha_fwd_attnmask_<hytlass::bfloat16_t, 64, false>(
    Flash_fwd_params_attnmask &params, cudaStream_t stream);
