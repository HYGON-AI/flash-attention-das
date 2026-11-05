// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Private bridge between the split adapter and the torch-free core.
// Keep these declarations in sync with their definitions in flash_api.cpp.

#pragma once

#include "flash.h"

void flash_hg_run_mha_fwd(Flash_fwd_params& params, hipStream_t stream, bool force_split_kernel);
void flash_hg_run_mha_bwd(Flash_bwd_params& params, hipStream_t stream, const bool configure);
void flash_hg_run_mha_fwd_kvcache(Flash_fwd_params& params, hipStream_t stream, bool force_split_kernel);
