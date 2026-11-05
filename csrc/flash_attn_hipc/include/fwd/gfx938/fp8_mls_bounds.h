// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Callers issue MLS only for a tile containing at least one physical row.
// FullRows is local to this Q/K/V tile, not a promise about another tensor.
template<bool FullRows, int TileRows>
__forceinline__ __device__ bool fp8_mls_needs_filter(int remaining_rows) {
    return !FullRows && remaining_rows < TileRows;
}

// Keep the bounded interval for generic 128-row MLS: its instruction shape
// matters to scheduling. These loaders determine completeness at runtime.
template<int TileRows>
__forceinline__ __device__ bool fp8_mls_has_partial_rows(int remaining_rows) {
    return remaining_rows > 0 && remaining_rows < TileRows;
}
