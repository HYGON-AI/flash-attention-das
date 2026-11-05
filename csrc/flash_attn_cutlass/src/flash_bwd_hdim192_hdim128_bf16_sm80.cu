#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_mla_<hytlass::bfloat16_t, 192, 128, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim192_hdim128<hytlass::bfloat16_t, false>(params, stream);
}