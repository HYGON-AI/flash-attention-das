// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_fp8_mha_fwd_prefix_prefill_<BFloat16, 192, 192>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_fp8_flash_fwd_prefix_prefill<BFloat16, 192, 192>(params, stream);
#endif
}
