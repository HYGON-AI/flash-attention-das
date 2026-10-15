
#include "flash_attnmask.h"

void run_mha_fwd_attnmask(Flash_fwd_params_attnmask &params, cudaStream_t stream);
void run_mha_fwd_attnmask_bias(Flash_fwd_params_attnmask &params, cudaStream_t stream);
void run_mha_fwd_attnmask_bias_144(Flash_fwd_params_attnmask &params, cudaStream_t stream);
void run_mha_fwd_attnmask_additive(Flash_fwd_params_attnmask &params, cudaStream_t stream);
void run_mha_fwd_attnmask_additive_144(Flash_fwd_params_attnmask &params, cudaStream_t stream);
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false);
void run_mha_fwd_16x64(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_splitkv(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_padding_mask(Flash_fwd_params &params, cudaStream_t stream, bool =false);