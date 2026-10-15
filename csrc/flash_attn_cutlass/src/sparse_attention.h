#pragma once

// Common host validation and device primitives for sparse attention operators.
#include "sparse_attention_mmac.h"

#include <torch/python.h>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sparse_attention_detail {

struct CacheStrides {
    int64_t block;
    int64_t token;
    int64_t head;
    int64_t dim;
};

inline CacheStrides cache_strides(const at::Tensor &tensor) {
    TORCH_CHECK(tensor.dim() == 4, "cache must be four-dimensional");
    return {tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)};
}

inline void check_device(const at::Tensor &tensor, const at::Tensor &reference,
                         const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be on the accelerator");
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must be on q's device");
}

inline void check_i32(const at::Tensor &tensor, const at::Tensor &reference,
                      const char *name) {
    check_device(tensor, reference, name);
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
}

inline bool aligned8(const at::Tensor &tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 8 == 0;
}

inline unsigned int grid_size(int64_t programs) {
    TORCH_CHECK(programs > 0 && programs <= std::numeric_limits<unsigned int>::max(),
                "kernel grid is out of range");
    return static_cast<unsigned int>(programs);
}

}  // namespace sparse_attention_detail
