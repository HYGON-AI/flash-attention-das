// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define PA_FIX_PARTITION 65536
#define MLA_MAX_SPLITS 1024

template<int MIN_PARTITION_SIZE=128>
__forceinline__ __device__ int splitkv_get_partitionsize_of_fix_numsplits(int actual_seqlen_k, int num_splits) {
    const int true_partition = max(
        1, actual_seqlen_k / num_splits + int(actual_seqlen_k % num_splits != 0));
    const int aligned_partition =
        ((true_partition + MIN_PARTITION_SIZE - 1) / MIN_PARTITION_SIZE) * MIN_PARTITION_SIZE;
    return max(aligned_partition, MIN_PARTITION_SIZE);
}
