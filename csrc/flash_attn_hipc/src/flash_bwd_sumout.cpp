// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include "flash_sumout_api.h"

template __global__ void flash_sum_out<BFloat16>(BFloat16*, BFloat16*, int, int);
template __global__ void flash_sum_out<Float16>(Float16*, Float16*, int, int);