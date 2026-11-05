#pragma once

#include "paged_attention_window.h"

// graph27's device-only round-robin scheduler, with the refactoring grid axes.
// All waves derive the same task without a global queue or replay-time reset.
template <bool Persistent>
struct PagedAttentionTileScheduler {
    const int* lengths;
    int batch, kv_heads, mtp, window_left, page_size, partition_size;
    int group = 0;
    long long task = 0, group_start = 0, inclusive = 0, count = 0, group_count = 0;
    bool loaded = false, done = false;

    __device__ __forceinline__ PagedAttentionTileScheduler(
        const int* lengths_, int batch_, int kv_heads_, int mtp_,
        int window_left_, int page_size_, int partition_size_)
        : lengths(lengths_), batch(batch_), kv_heads(kv_heads_), mtp(mtp_),
          window_left(window_left_), page_size(page_size_),
          partition_size(partition_size_), task(blockIdx.x) {}

    __device__ __forceinline__ bool next(int& seq, int& head, int& part) {
        if constexpr (!Persistent) {
            seq = blockIdx.z;
            head = blockIdx.x;
            part = blockIdx.y;
            return !done;
        } else {
            const int lane = threadIdx.x % 64;
            while (group < batch) {
                if (!loaded) {
                    const int index = group + lane;
                    const int nparts = index < batch ? get_paged_attention_active_range(
                        lengths[index], mtp, window_left, page_size, partition_size, 0)
                        .num_active_partitions : 0;
                    count = static_cast<long long>(nparts) * kv_heads;
                    inclusive = count;
                    #pragma unroll
                    for (int delta = 1; delta < 64; delta *= 2) {
                        const long long other = __shfl_up(inclusive, delta, 64);
                        if (lane >= delta) inclusive += other;
                    }
                    group_count = __shfl(inclusive, 63, 64);
                    loaded = true;
                }
                const long long relative = task - group_start;
                if (relative < group_count) {
                    const auto matches = __ballot(relative >= inclusive - count && relative < inclusive);
                    const int owner = __ffsll(matches) - 1;
                    const long long first = __shfl(inclusive - count, owner, 64);
                    const long long local = relative - first;
                    seq = __builtin_amdgcn_readfirstlane(group + owner);
                    head = __builtin_amdgcn_readfirstlane(static_cast<int>(local % kv_heads));
                    part = __builtin_amdgcn_readfirstlane(static_cast<int>(local / kv_heads));
                    return true;
                }
                group_start += group_count;
                group += 64;
                loaded = false;
            }
            return false;
        }
    }

    __device__ __forceinline__ void advance() {
        if constexpr (Persistent) task += gridDim.x;
        else done = true;
    }
};
