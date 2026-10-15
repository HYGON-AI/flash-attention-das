// Isolate scheduling of the exact BF16 sparse instance from fallback and SLA kernels.
#include "flash_fwd_sparse_launch_template.h"

using HcuSparseExactTraits = Flash_fwd_kernel_16x64_prefetch_traits<
    128, 64, 64, 4, cutlass::bfloat16_t, 3>;

__global__ void hcu_flash_fwd_sparse_exact_kernel(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params_sparse params) {
#if defined(ARCH_SUPPORTS_FLASH)
    flash::compute_sparse_attn<HcuSparseExactTraits, false, false, false, false,
                               false, true, false, false, true>(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

void run_flash_sparse_fwd_hcu_bf16_exact(Flash_fwd_params_sparse &params,
                                       cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + HcuSparseExactTraits::kBlockM - 1)
        / HcuSparseExactTraits::kBlockM;
    const dim3 grid(num_m_block, params.h, params.b);
    hcu_flash_fwd_sparse_exact_kernel<<<grid, HcuSparseExactTraits::kNThreads,
        HcuSparseExactTraits::kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
