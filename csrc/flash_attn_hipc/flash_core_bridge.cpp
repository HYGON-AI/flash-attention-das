// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Adapter-side public wrappers. They forward through unique flash_hg_* names;
// the core binds its internal run_mha_* calls with -Wl,-Bsymbolic.

#ifndef FLASH_C_API_EXTERN_ONLY
#define FLASH_C_API_EXTERN_ONLY
#endif
#include "flash_c_api.h"

// Core entry points implemented by flash_api.cpp.
#include "flash_hg_bridge.h"

void run_mha_fwd(Flash_fwd_params& params, hipStream_t stream,
                 bool force_split_kernel) {
    flash_hg_run_mha_fwd(params, stream, force_split_kernel);
}

void run_mha_bwd(Flash_bwd_params& params, hipStream_t stream,
                 const bool configure) {
    flash_hg_run_mha_bwd(params, stream, configure);
}

void run_mha_fwd_kvcache(Flash_fwd_params& params, hipStream_t stream,
                         bool force_split_kernel) {
    flash_hg_run_mha_fwd_kvcache(params, stream, force_split_kernel);
}
