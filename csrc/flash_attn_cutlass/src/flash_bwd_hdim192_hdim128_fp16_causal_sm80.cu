#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_mla_<hytlass::half_t, 192, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim192_hdim128<hytlass::half_t, true>(params, stream);
}