#pragma once

#include <string>
#include <algorithm>
#include <cstdint>
#include <limits>

extern const int PA_GRAPH_COMPACT;
extern const int PA_GRAPH_WORKERS;

inline int pa28_graph_workers(int configured, const std::string& arch, int compute_units) {
    if (configured > 0) return configured;
    if (configured == 0) return compute_units;
    if (arch == "gfx938") return 1728;
    if (arch == "gfx936") return 240;
    return 120;  // gfx92a and the unchanged gfx928 default.
}

// Runtime metadata for the window specialization of the refactoring kernel.
struct Pa28WindowConfig {
    int left = -1;
    int right = 0;
    int partition_left = -1;
    int partition_size = 512;
    int page_size = 64;
    int k_token_stride = 0;
    int v_token_stride = 0;
    int v_channel_stride = 1;
    int k_head_stride = 0;
    int v_head_stride = 0;
};

// One host plan owns the partition stride used by the producer, combine and
// allocation. Output storage is allocated to its actual size, not the native
// path's historical 210 MB limit. Fixed statistics regions still bound parts.
struct Pa28WindowPlan {
    int partition_size;
    int num_partitions;
    int64_t workspace_bytes;
};

inline Pa28WindowPlan pa28_make_window_plan(
    int capacity, int mtp, int left, int partition_size,
    int batch, int qheads, int value_dim) {
    TORCH_CHECK(capacity > 0 && mtp > 0 && batch > 0 && qheads > 0 && value_dim > 0,
                "PA28: invalid workspace dimensions");
    TORCH_CHECK(partition_size >= 64 && partition_size % 64 == 0,
                "PA28: partition size must be a positive multiple of 64");
    const int64_t rows = static_cast<int64_t>(batch) * mtp * qheads;
    int parts = get_paged_attention_max_num_partitions(
        capacity, mtp, left, 64, partition_size, left >= 0);
    if (parts == 1) return {partition_size, 1, 0};
    const int64_t max_parts = static_cast<int64_t>(max_tmp_offset) * 2 / sizeof(float) / rows;
    TORCH_CHECK(max_parts >= 1, "PA28: statistics workspace capacity exceeded");
    if (parts > max_parts) {
        const int64_t active_tokens = left < 0 ? capacity :
            std::min<int64_t>(capacity, static_cast<int64_t>(left) + mtp + 63);
        const int64_t pages = (active_tokens + 63) / 64;
        const int64_t larger_partition = ((pages + max_parts - 1) / max_parts) * 64;
        TORCH_CHECK(larger_partition <= std::numeric_limits<int>::max(),
                    "PA28: partition size overflow");
        partition_size = std::max(partition_size, static_cast<int>(larger_partition));
        parts = get_paged_attention_max_num_partitions(
            capacity, mtp, left, 64, partition_size, left >= 0);
    }
    if (parts == 1) return {partition_size, 1, 0};
    TORCH_CHECK(static_cast<int64_t>(parts) <= max_parts,
                "PA28: statistics workspace capacity exceeded");
    return {partition_size, parts,
            (static_cast<int64_t>(out_tmp_offset) + rows * parts * value_dim) * 2};
}

void paged_attention_window_opt(
    torch::Tensor& out, torch::Tensor& query,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<at::Tensor>& sink,
    float* workspace, int layout, int left, int right, const Pa28WindowPlan& plan);
