#include "paged_attention_refactoring_common.h"

template <typename scalarx8_t, typename scalarx4_t, typename scalar_t, typename cache_t, int Mloop, int HEAD_SIZE, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, bool is_query_fp16, int STAGES, bool is_kv_fp8, bool is_e4m3, int Q_LDS_PAD = 0>
__forceinline__ __device__ void qk_gemm_func_with_prefetch_k(const cache_t *k_ptr,
                                                  const scalar_t *smem_q,
                                                  int seqlen_q_per_kv_head,
                                                  float4_t s_reg[Mloop][BLOCK_SIZE / 16],
                                                  int block_idx,
                                                  int valid_seqlen_kv
                                                )
{
    // 当 NUM_KV_TOKENS_PER_LOAD = 64, 则 s_reg 每个线程持有 一行 连续 的 4 个元素, 与 BLOCK_SIZE 无关
    //  例如: BLOCK_SIZE = 64, 则 (s_reg[0][0])[0]、(s_reg[0][1])[0]、(s_reg[0][2])[0]、(s_reg[0][3])[0] 为一行4个连续元素
    //       BLOCK_SIZE = 128, 则 (s_reg[0][0])[0]、(s_reg[0][1])[0]、(s_reg[0][2])[0]、(s_reg[0][3])[0] 为一行4个连续元素
    //                              (s_reg[0][4])[0]、(s_reg[0][5])[0]、(s_reg[0][6])[0]、(s_reg[0][7])[0] 为同一行另4个连续元素, 与上方元素的偏移是 64

    // 当前仅 实现了 BLOCK_SIZE = 64

    // 每个 warp 处理一个block_page
    // k_cache 每个 warp 一次加载 32x64
    // STAGES = 1, 不做流水
    // STAGES = 2, 双buffer, 做 k global->register 与 mmac 的流水
    // 假设 NUM_KV_TOKENS_PER_LOAD = 64, NUM_KV_TOKENS_PER_LOAD / 16 = 4
    //  加载 k global->register 的线程布局是:
    //  t0 t0 t0 t0 t1 t1 t1 t1....
    //  ...
    int stage_id = 0;
    using k_stage_t = std::conditional_t<is_kv_fp8, uint8x8_t, scalarx8_t>;
    k_stage_t k_reg[STAGES * (NUM_KV_TOKENS_PER_LOAD / 16)];
    // 此处假设 STAGES = 2
    // k global -> reigster 预取
    int k_tile = 0;
    int lane_id = threadIdx.x % WARP_SIZE;
    // 0-3, 4-7 线程连续 global->register, 下方通过 permute 转成 mmac 需要的布局
    int k_g2r_row = lane_id & 3;        // 0, 1, 2, 3
    int k_g2r_col = lane_id >> 2;       // 0-15
    int q_s2r_row = lane_id & 15;
    int q_s2r_col = lane_id >> 4;
    {
        const cache_t *k_tile_ptr = k_ptr + k_tile * 32;
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            const cache_t *src = k_tile_ptr + k_g2r_row * 8 + (k_g2r_col * (NUM_KV_TOKENS_PER_LOAD / 16) + i) * HEAD_SIZE;
            if constexpr (is_kv_fp8){
                // TODO 流水分析
                k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i] = *((uint8x8_t *)(src));
                // inline_global_load_dwordx2_no_wait(k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i], (const uint64_t *)src);
            }else{
                k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i] = *((scalarx8_t *)(src));
                // inline_global_load_dwordx4_no_wait(k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i], (const uint64_t *)src);
            }
        }
    }

    // 切换到第二个buffer
    stage_id ^= 1;

    #pragma unroll
    for(k_tile = 1;k_tile < (HEAD_SIZE / 32);k_tile++){

        const cache_t *k_tile_ptr = k_ptr + k_tile * 32;
        // 取下一份数据
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            const cache_t *src = k_tile_ptr + k_g2r_row * 8 + (k_g2r_col * (NUM_KV_TOKENS_PER_LOAD / 16) + i) * HEAD_SIZE;
            if constexpr (is_kv_fp8){
                // TODO 流水分析
                // inline_global_load_dwordx2_no_wait(k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i], (const uint64_t *)src);
                k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i] = *((uint8x8_t *)(src));
            }else{
                // inline_global_load_dwordx4_no_wait(k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i], (const uint64_t *)src);
                k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i] = *((scalarx8_t *)(src));
            }
        }
        scalarx8_t q_fragment[Mloop];
        for(int m = 0;m < Mloop;m++){
            int q_row = m * 16 + q_s2r_row;
            if(q_row < seqlen_q_per_kv_head){
                q_fragment[m] = *reinterpret_cast<const scalarx8_t *>(
                    smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + (k_tile - 1) * 32 + q_s2r_col * 8);
            }else{
                q_fragment[m] = {0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
        // 此时有 2 * (NUM_KV_TOKENS_PER_LOAD / 16) buffer_load 指令 on air
        // 等待 前 (NUM_KV_TOKENS_PER_LOAD / 16) 个指令完成, 并进行 mmac
        inline_wait_vmcnt(NUM_KV_TOKENS_PER_LOAD / 16);

        // 切换到 计算索引
        stage_id ^= 1;
        // k E5M2->b16
        scalarx8_t k_fragment[NUM_KV_TOKENS_PER_LOAD / 16];
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            if constexpr(is_kv_fp8){
                k_fragment[i] = decode_fp8x8_to_b16x8<scalarx8_t, scalar_t, is_e4m3>(
                    k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i]);
            } else {
                k_fragment[i] = k_reg[stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i];
            }
        }

        {
            // permute
            for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
                int *temp_ptr = reinterpret_cast<int *>(&(k_fragment[i]));
                for(int j = 0;j < 4;j++){                    
                    int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                    int res = __builtin_amdgcn_ds_permute(index, temp_ptr[j]);
                    temp_ptr[j] = res;
                }            
            }
        }

        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
                const scalarx4_t *q_v4 = reinterpret_cast<const scalarx4_t *>(&(q_fragment[m]));
                const scalarx4_t *k_v4 = reinterpret_cast<const scalarx4_t *>(&(k_fragment[i]));
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[0], k_v4[0], s_reg[m][i]);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[1], k_v4[1], s_reg[m][i]);
            }
        }
    }
    // 最后待计算的 buffer id
    int last_stage_id = stage_id ^ 1;

    scalarx8_t q_fragment[Mloop];
    for(int m = 0;m < Mloop;m++){
        int q_row = m * 16 + q_s2r_row;
        if(q_row < seqlen_q_per_kv_head){
            q_fragment[m] = *reinterpret_cast<const scalarx8_t *>(
                smem_q + q_row * (HEAD_SIZE + Q_LDS_PAD) + ((HEAD_SIZE / 32) - 1) * 32 + q_s2r_col * 8);
        }else{
                q_fragment[m] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    inline_wait_vmcnt(0);

    // k E5M2->b16
    scalarx8_t k_fragment[NUM_KV_TOKENS_PER_LOAD / 16];
    for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
        if constexpr(is_kv_fp8){
            k_fragment[i] = decode_fp8x8_to_b16x8<scalarx8_t, scalar_t, is_e4m3>(
                k_reg[last_stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i]);
        } else {
            k_fragment[i] = k_reg[last_stage_id * (NUM_KV_TOKENS_PER_LOAD / 16) + i];
        }
    }

    {
        // permute
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            int *temp_ptr = reinterpret_cast<int *>(&(k_fragment[i]));
            for(int j = 0;j < 4;j++){                    
                int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                int res = __builtin_amdgcn_ds_permute(index, temp_ptr[j]);
                temp_ptr[j] = res;
            }            
        }
    }

    for(int m = 0;m < Mloop;m++){
        for(int i = 0;i < (NUM_KV_TOKENS_PER_LOAD / 16);i++){
            const scalarx4_t *q_v4 = reinterpret_cast<const scalarx4_t *>(&(q_fragment[m]));
            const scalarx4_t *k_v4 = reinterpret_cast<const scalarx4_t *>(&(k_fragment[i]));
            builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[0], k_v4[0], s_reg[m][i]);
            builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(q_v4[1], k_v4[1], s_reg[m][i]);

            // 边界处理
            int token_idx = block_idx * BLOCK_SIZE + i + q_s2r_col * 4;
            for(int j = 0;j < 4;j++){
                if((token_idx + j * 16) >= valid_seqlen_kv){
                    (s_reg[m][i])[j] = -INFINITY;
                }
            }
        }
    }
    // TODO BLOCK_SIZE = 128 支持
    //  s_reg[m][NUM_KV_TOKENS_PER_LOAD / 16 + i]

  return;
}

template <typename scalarx4_t, typename scalar_t, typename cache_t, int Mloop, int HEAD_SIZE_V, int BLOCK_SIZE, int NUM_KV_TOKENS_PER_LOAD, bool is_query_fp16, int STAGES,
          bool is_kv_fp8=false, bool is_e4m3=false>
__device__ __forceinline__  void pv_gemm_func_with_prefetch_v(const cache_t *v_ptr,
                                                    scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16],
                                                    float4_t acc_o[Mloop][HEAD_SIZE_V / 16],
                                                    int block_idx,
                                                    int valid_seqlen_kv
                                                  )
{   
    // 每个 warp 处理一个 block_page
    // v_cache 每个 warp 一次加载 64x32
    // p 矩阵每个线程持有 4 个连续的元素, 对应的 v 每个线程 持有同一列的4个连续数据
    int stage_id = 0;
    // 假设 NUM_KV_TOKENS_PER_LOAD = 64, 则[4][2], [4] 代表加载 64 行, [2] 代表加载 32 列
    using v_stage_t = std::conditional_t<is_kv_fp8, uint8x4_t, scalarx4_t>;
    v_stage_t v_reg[NUM_KV_TOKENS_PER_LOAD / 16][STAGES * 2];
    // 此处假设 STAGES = 2
    // v global -> lds 预取
    int v_tile = 0;
    int lane_id = threadIdx.x % WARP_SIZE;
    // 0-3, 4-7 线程连续 global->register, 下方通过bpermute 转成 mmac 需要的布局
    int v_g2r_row = lane_id & 3;       // 0, 1, 2, 3
    int v_g2r_col = lane_id >> 2;       // 0-15
    {
        const cache_t *v_tile_ptr = v_ptr + v_tile * 32 * BLOCK_SIZE;
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const cache_t *cur_v_ptr =v_tile_ptr + i * 16 + v_g2r_row * 4 + (j * 16 + v_g2r_col) * BLOCK_SIZE;
                if constexpr(is_kv_fp8){
                    // inline_global_load_dword_no_wait(v_reg[i][stage_id * 2 + j], (const uint64_t *)cur_v_ptr);
                    v_reg[i][stage_id * 2 + j] = *reinterpret_cast<const uint8x4_t *>(cur_v_ptr);
                } else {
                    // inline_global_load_dwordx2_no_wait(v_reg[i][stage_id * 2 + j], (const uint64_t *)cur_v_ptr);
                    v_reg[i][stage_id * 2 + j] = *reinterpret_cast<const scalarx4_t *>(cur_v_ptr);
                }
            }
        }
    }

    // 切换到第二个buffer
    stage_id ^= 1;

    // Keep accumulator and ping-pong buffer indices constant after inlining.
    #pragma unroll
    for(v_tile = 1; v_tile < (HEAD_SIZE_V / 32);v_tile++){
        const cache_t *v_tile_ptr = v_ptr + v_tile * 32 * BLOCK_SIZE;
        // 取下一份数据
        for(int i = 0;i < NUM_KV_TOKENS_PER_LOAD / 16;i++){
            for(int j = 0;j < 2;j++){
                const cache_t *cur_v_ptr =v_tile_ptr + i * 16 + v_g2r_row * 4 + (j * 16 + v_g2r_col) * BLOCK_SIZE;
                if constexpr(is_kv_fp8){
                    // inline_global_load_dword_no_wait(v_reg[i][stage_id * 2 + j], (const uint64_t *)cur_v_ptr);
                    v_reg[i][stage_id * 2 + j] = *reinterpret_cast<const uint8x4_t *>(cur_v_ptr);
                } else {
                    // inline_global_load_dwordx2_no_wait(v_reg[i][stage_id * 2 + j], (const uint64_t *)cur_v_ptr);
                    v_reg[i][stage_id * 2 + j] = *reinterpret_cast<const scalarx4_t *>(cur_v_ptr);
                }
            }
        }
        // 此时有 2 * (NUM_KV_TOKENS_PER_LOAD / 16) * 2 buffer_loadx2 指令 on air
        // 等待 前 (NUM_KV_TOKENS_PER_LOAD / 16) * 2 个指令完成, 并进行 mmac

        // TODO 屏蔽后 sgpr溢出 待分析
        inline_wait_vmcnt((NUM_KV_TOKENS_PER_LOAD / 16) * 2);
        
        // 切换到 计算索引
        stage_id ^= 1;

        // 假设 NUM_KV_TOKENS_PER_LOAD = 64, 此时加载了 global v 的 64 行
        // 对应到 P 矩阵的 64 列, 当前 仅考虑 BLOCK_SIZE = 64 的场景
        // 若 BLOCK_SIZE = 128, 则 需要写另外的逻辑 加载 global v 的后64 行, 并与 P 的后 64 列对应进行mmac
        // Decode and permute each V fragment once, then reuse it across query tiles.
        for(int j = 0;j < (NUM_KV_TOKENS_PER_LOAD / 16);j++){
            scalarx4_t v_fragment0;
            scalarx4_t v_fragment1;
            if constexpr(is_kv_fp8){
                v_fragment0 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][stage_id * 2 + 0]);
                v_fragment1 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][stage_id * 2 + 1]);
            } else {
                v_fragment0 = v_reg[j][stage_id * 2 + 0];
                v_fragment1 = v_reg[j][stage_id * 2 + 1];
            }

            // v 边界处理
            int token_idx = block_idx * BLOCK_SIZE + j * 16 + v_g2r_row * 4;
            for(int i = 0;i < 4;i++){
                if((token_idx + i) >= valid_seqlen_kv){
                    v_fragment0[i] = 0;
                    v_fragment1[i] = 0;
                }
            }

            {
                // permute
                int *temp_ptr0 = reinterpret_cast<int *>(&(v_fragment0));
                int *temp_ptr1 = reinterpret_cast<int *>(&(v_fragment1));
                for(int j = 0;j < 2;j++){           
                    int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                    int res0 = __builtin_amdgcn_ds_permute(index, temp_ptr0[j]);
                    int res1 = __builtin_amdgcn_ds_permute(index, temp_ptr1[j]);
                    temp_ptr0[j] = res0;
                    temp_ptr1[j] = res1;
                }            
            }

            #pragma unroll
            for(int m = 0;m < Mloop;m++){
                // p 在需要的时候再构造, 减少 temp_p_reg 的生存时间
                scalarx4_t temp_p_reg = {(p_reg[m][0])[j], (p_reg[m][1])[j], (p_reg[m][2])[j], (p_reg[m][3])[j]};
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment0, acc_o[m][(v_tile - 1) * 2 + 0]);
                builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment1, acc_o[m][(v_tile - 1) * 2 + 1]);
            }
        }
    }
    // 最后待计算的 buffer id
    int last_stage_id = stage_id ^ 1;

    // TODO 屏蔽后 sgpr溢出 待分析
    inline_wait_vmcnt(0);
    
    // The final prefetched tile uses the same V-fragment reuse.
    for(int j = 0;j < (NUM_KV_TOKENS_PER_LOAD / 16);j++){
        scalarx4_t v_fragment0;
        scalarx4_t v_fragment1;
        if constexpr(is_kv_fp8){
            v_fragment0 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][last_stage_id * 2 + 0]);
            v_fragment1 = decode_fp8x4_to_b16x4<scalarx4_t, scalar_t, is_e4m3>(v_reg[j][last_stage_id * 2 + 1]);
        } else {
            v_fragment0 = v_reg[j][last_stage_id * 2 + 0];
            v_fragment1 = v_reg[j][last_stage_id * 2 + 1];
        }

        // v 边界处理
        int token_idx = block_idx * BLOCK_SIZE + j * 16 + v_g2r_row * 4;
        for(int i = 0;i < 4;i++){
            if((token_idx + i) >= valid_seqlen_kv){
                v_fragment0[i] = 0;
                v_fragment1[i] = 0;
            }
        }

        {
            // permute
            int *temp_ptr0 = reinterpret_cast<int *>(&(v_fragment0));
            int *temp_ptr1 = reinterpret_cast<int *>(&(v_fragment1));
            for(int j = 0;j < 2;j++){                    
                int index = (((lane_id & 3) << 4) + (lane_id >> 2)) << 2;
                int res0 = __builtin_amdgcn_ds_permute(index, temp_ptr0[j]);
                int res1 = __builtin_amdgcn_ds_permute(index, temp_ptr1[j]);
                temp_ptr0[j] = res0;
                temp_ptr1[j] = res1;
            }            
        }

        #pragma unroll
        for(int m = 0;m < Mloop;m++){
            scalarx4_t temp_p_reg = {(p_reg[m][0])[j], (p_reg[m][1])[j], (p_reg[m][2])[j], (p_reg[m][3])[j]};
            builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment0, acc_o[m][((HEAD_SIZE_V / 32) - 1) * 2 + 0]);
            builtin_amdgcn_mmac<scalarx4_t, is_query_fp16>(temp_p_reg, v_fragment1, acc_o[m][((HEAD_SIZE_V / 32) - 1) * 2 + 1]);
        }
    }

    // TODO BLOCK_SIZE = 128 支持
    // half4_t temp_p_reg = {(p_reg[m][4])[j], (p_reg[m][5])[j], (p_reg[m][6])[j], (p_reg[m][7])[j]};
    return;
}

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size_v]
    const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, head_size_v, block_size]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,       
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const scalar_t* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp)
{
    //优化点:
    //  1、S/P 不过lds, 消除bc, 增加 occupancy
    //  2、支持 PARTITION_SIZE 是 64 整数倍的任意数, 避免 splitk_buffer 溢出
    // TODO  功能支持
    // 1、支持 BLOCK_SIZE = 64/128, 当前版本仅支持 64
    // 2、支持HEAD_SIZE_V != HEAD_SIZE_QK

    extern __shared__ char shared_mem[];
    scalar_t *smem_q = (scalar_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);

    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;
    using scalarx8_t= __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;

    // grid_size: (num_kv_heads, max_partition, batch_size)
    const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;    

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    int valid_num_partition = get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_query_fp16 true, scalar_t = fp16; is_query_fp16 false, scalar_t = bf16; 
    constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_kv_fp8 true, cache_t = e5m2; is_kv_fp8 false, cache_t = scalar_t
    constexpr bool is_kv_fp8 = std::is_same<cache_t, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    // q global -> LDS; Q fragments are loaded on demand in QK GEMM.
    constexpr int Mloop = DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16);

    float alibi_slope[Mloop];
    // alibi slope
    if constexpr (ALIBI_ENABLE){
        int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
        int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
        for(int m = 0;m < Mloop;m++){
            int row = m * 16 + (lane_id & 15);
            if(row < seqlen_q_per_kv_head){
                int local_q_head = row % num_q_heads_per_kv_head;
                int slope_idx = q_head_idx + local_q_head;
                alibi_slope[m] = alibi_slopes[slope_idx] / scale_softmax;
            }
        }
    }

    prefetch_q_to_lds<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD>(
        q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride);

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    const int start_block_idx = static_cast<int>(static_cast<int64_t>(partition_idx) * valid_num_block_pages / valid_num_partition);
    const int end_block_idx  = static_cast<int>(static_cast<int64_t>(partition_idx + 1) * valid_num_block_pages / valid_num_partition);
    // 有效block_page_idx: [start_block_idx, end_block_idx)
    const int num_blocks = end_block_idx - start_block_idx;

    // k_cache 形状为 (num_block, num_heads, block_size, head_dim)
    // k_head_stride = BLOCK_SIZE * HEAD_DIM
    // kv_head_idx * k_head_stride 含义为一个 block_page 内, 当前 kv_head 处理的global kv_cache的偏移
    // k_ptr_base 没有考虑 num_blocks 维度的偏移
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
    const cache_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const cache_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;

    for(int block_idx = (start_block_idx + warp_id);block_idx < end_block_idx;block_idx += NUM_WARPS){
        // 1个warp处理一个block_page
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
        // 加上 num_blocks 维度的偏移
        const cache_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

        // v_cache 形状为 [num_blocks, num_heads, head_size, block_size]
        // v_cache 在 block_size 维度连续, pv gemm时, v 为列主序
        // 加上 num_blocks 维度的偏移
        const cache_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;
        
        float4_t s_reg[Mloop][BLOCK_SIZE / 16];
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // 每个warp 处理 一个 kv block_page
        qk_gemm_func_with_prefetch_k<scalarx8_t, scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, Q_LDS_PAD>
                                    (k_ptr, smem_q, num_heads / num_kv_heads, s_reg, block_idx, valid_seqlen_kv);

        // alibi
        if constexpr (ALIBI_ENABLE){
            int seqlen_q_per_kv_head = num_heads / num_kv_heads;
            int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
            int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;
            for(int m = 0;m < Mloop;m++){
                int row = m * 16 + (lane_id & 15);
                int col = lane_id >> 4;
                if(row < seqlen_q_per_kv_head){
                    int q_token_id = row / num_q_heads_per_kv_head;
                    int casual_offset = mtp - q_token_id;
                    int query_position = valid_seqlen_kv - casual_offset;
                    for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                        for(int j = 0;j < 4;j++){
                            int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                            if(token_k_index < valid_seqlen_kv){
                                // 边界内
                                int relative_position = token_k_index - query_position;
                                float alibi = alibi_slope[m] * relative_position;
                                // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                (s_reg[m][i])[j] += alibi;
                            }
                        }
                    }                        
                }
            }
        }        

        // mtp causal
        if constexpr (MTP_ENABLE){
            if (block_idx * BLOCK_SIZE + BLOCK_SIZE - 1 + mtp > valid_seqlen_kv) {
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
                for(int m = 0;m < Mloop;m++){
                    int row = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(row < seqlen_q_per_kv_head){
                        int q_token_id = row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                            for(int j = 0;j < 4;j++){
                                int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    if((token_k_index + casual_offset > valid_seqlen_kv)){
                                        (s_reg[m][i])[j] = -INFINITY;
                                    }                           
                                }
                            }
                        }
                    }                  
                }
            }
        }  

        // s_reg: S = 2^(S * softmax_scale_log2 - max_current * softmax_scale_log2) = e^(S * softmax_scale - max_current * softmax_scale)
        // scores_max = new_max, raw max, 没有 乘  scale_softmax_log2
        // scores_sum: 对应公式 d(i) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale) +  row_sum(e^((S - max_current) * softmax_scale))
        // acc_o: O(i-1) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2);

        scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16];
        // fp32 -> fp16
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                scalar_t tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[0]);
                (p_reg[m][i])[0] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[1]);
                (p_reg[m][i])[1] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[2]);
                (p_reg[m][i])[2] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[3]);
                (p_reg[m][i])[3] = tmp_vale;
            }
        }

        // pv gemm
        // acc_o: O(i) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale) + gemm_pv
        pv_gemm_func_with_prefetch_v<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
    }

    // Q remains live in LDS throughout the page loop. Prevent an early warp
    // from reusing the same LDS storage for max/sum while another warp reads Q.
    __syncthreads();

    // sink
    if constexpr(SINK_ENABLE)
    {
        // 对 apply_attention_sink 的理解:
        //      sink 为 人为 在 S 矩阵的每一行增加一个元素, 假设为 sink_value
        //      则 new_max = MAX_T(sink_value, old_max), 根据new_max 更新 scores_max, scores_sum, acc_o
        //      包含一下公式:
        //              max = new_max
        //              d(i) = d(i-1) * e^(m(i - 1) - m(i)) + row_sum(e^(xi(j) - mi))
        //              o(i - 1) = o(i - 1) * e^(m(i - 1) - m(i)), sink 对应的 v 为0, 不考虑 pv 项
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink<scalar_t, Mloop, HEAD_SIZE_V>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax,
                scale_softmax_log2,
                warp_id,
                partition_idx,
                mtp,
                num_heads,
                num_kv_heads,
                kv_head_idx);
        }
    }


    float scores_max_new[Mloop];
    float scores_sum_new[Mloop];

    epilogue_rescale_acco<Mloop, HEAD_SIZE_V, NUM_WARPS>(scores_max, scores_sum, acc_o, scores_max_new, scores_sum_new,
                                                        smem_max, smem_sum, scale_softmax_log2, warp_id, num_heads, num_kv_heads);

    // 当前 acc_o 为经过缩放结果, 需要将4个warp的 acc_o 加起来得到 PARTION_SIZE 的输出
    //  通过lds reduce实现
    {
        const int acc_row = lane_id & 15;
        const int acc_col = lane_id >> 4;
        // 每个kv head处理的token q数量
        const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        // 每个kv head对应的q head数量
        int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
        // 当前block处理的q head起始索引, 即当前kv_head_id对应的q_head_id_start
        int head_q_index = q_heads_per_kv_head * kv_head_idx;
        // 若 1 == valid_num_partition, 对应的 out 的形状是 (batch, seqlen_q * num_q_heads, head_dim_v), scalar_t
        // 若 valid_num_partition > 1, 对应的 out_buffer 的形状是(batch, seqlen_q * num_q_heads, max_num_partitions, head_dim_v) scalar_t
        scalar_t *out_ptr_base = nullptr;
        int out_head_stride = 0;
        // num_heads = seqlen_q * num_q_heads
        // out_ptr_base 考虑了 batch 维度, head 维度起始head索引的偏移, 没有考虑 seqlen 维度的偏移
        if (1 == valid_num_partition){
            out_head_stride = HEAD_SIZE_V;
            out_ptr_base = out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
        }else{
            out_head_stride = balanced_seq_len_tile  * HEAD_SIZE_V;
            // out_tmp_offset 元素, 为预留的 max sum buffer
            out_ptr_base = out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                            + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V;
        }

        // lds reduce 每个迭代处理 NUM_WARPS * 16x16 *4 的数据, 保证每个线程 b128 的写出, 且每个warp需要写
        constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
        // lds 形状为(NUM_WARPS, WARP_SIZE * 4)
        float *smem_reduce = (float *)shared_mem;
        for(int tile_idx = 0;tile_idx < num_16x64_tiles;tile_idx++){
            const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
            const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
            for(int j = 0;j < 4;j++){
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 0) = (acc_o[tile_idx_row][tile_idx_col * 4 + 0])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 1) = (acc_o[tile_idx_row][tile_idx_col * 4 + 1])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 2) = (acc_o[tile_idx_row][tile_idx_col * 4 + 2])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 3) = (acc_o[tile_idx_row][tile_idx_col * 4 + 3])[j];
            }
            __syncthreads();

            
            int out_r2g_row = threadIdx.x >> 4;
            int out_r2g_col = threadIdx.x & 15;
            const int row = tile_idx_row * 16 + out_r2g_row;

            // 边界限制
            if(row < seqlen_q_per_kv_head){
                // row / q_heads_per_kv_head 表示当前行对应 out 的 token id
                //   例如: num_q_heads = 16, seqlen_q = 2, num_kv_heads = 2
                //         则 q_heads_per_kv_head = num_q_heads / num_kv_heads = 8
                //            seqlen_q_per_kv_head = num_q_heads * seqlen_q / num_kv_heads = 16
                //            kv_head_idx = 0时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head0-7, 8-15 对应 token1 的 head0-7
                //            kv_head_idx = 1时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head8-16, 8-15 对应 token1 的 head8-15
                
                //      num_kv_heads * q_heads_per_kv_head 为 num_q_heads
                //      (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) 含义是 当前 row 对应的 q_token_id 的 q_head 偏移
                //      row % q_heads_per_kv_head + head_q_index, 对应当前行的实际的 q_head 的索引
                const int out_row = (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                const int our_col = tile_idx_col * 64 + out_r2g_col * 4;
                float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f};
                for(int i = 0;i < NUM_WARPS;i++){
                    float4_t temp_value = *((float4_t *)(smem_reduce + i * 16 * (64 + REDUCE_LDS_PAD) + out_r2g_row * (64 + REDUCE_LDS_PAD) + out_r2g_col * 4));
                    sums[0] += temp_value[0];
                    sums[1] += temp_value[1];
                    sums[2] += temp_value[2];
                    sums[3] += temp_value[3];
                }

                scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]), from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
                *((scalarx4_t *)(out_ptr_base + out_row * out_head_stride + our_col)) = out_sums;
            }
             __syncthreads();
        }
        
        // 保存 max sum
        if(valid_num_partition > 1){
            // max, sum 形状为 (batch, seqlen_q, num_q_heads, max_num_partitions)
            if(warp_id == 0 && acc_col == 0){
                for(int m = 0; m < Mloop;m++){
                    int row = m * 16 + acc_row;
                    if(row < seqlen_q_per_kv_head){
                        int out_row = (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + row % q_heads_per_kv_head;
                        int offset = batch_idx * num_heads * balanced_seq_len_tile 
                                      + (head_q_index + out_row) * balanced_seq_len_tile 
                                      + partition_idx;
                        float *exp_sums = reinterpret_cast<float*>(out_tmp);
                        // max_tmp_offset 元素单位
                        float *max = reinterpret_cast<float*>(out_tmp + max_tmp_offset);

                        exp_sums[offset] = scores_sum_new[m];
                        // 此处存到 global 的 max 经过 scale_softmax_log2
                        max[offset] = scores_max_new[m]  * scale_softmax_log2;
                    }
                }
            } 
        }
    }

    return;
}

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int STAGES, int Q_LDS_PAD, int REDUCE_LDS_PAD,
          bool SINK_ENABLE, bool ALIBI_ENABLE, bool MTP_ENABLE, int NUM_Q_TILES>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_qtile_opt(
    scalar_t* __restrict__ out,                 // [num_seqs, seqlen_q * num_q_heads, head_size]
    scalar_t* __restrict__ out_tmp,             // [num_seqs, seqlen_q * num_q_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,             // [num_seqs, seqlen_q * num_q_heads, head_size]
    const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size_v]
    const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, head_size_v, block_size]
    const int num_heads,                        // [seqlen_q * num_q_heads]
    const int num_kv_heads,                     // [num_kv_heads]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int max_num_blocks_per_seq,       
    const float* __restrict__ alibi_slopes,     // [num_q_heads]
    const int q_stride,const int k_block_stride,
    const int v_block_stride,
    int balanced_seq_len_tile,                  //
    const scalar_t* __restrict__ s_aux_ptr,     // [seqlen_q * num_q_heads]
    int mtp)
{
    //优化点:
    //  1、S/P 不过lds, 消除bc, 增加 occupancy
    //  2、支持 PARTITION_SIZE 是 64 整数倍的任意数, 避免 splitk_buffer 溢出
    // TODO  功能支持
    // 1、支持 BLOCK_SIZE = 64/128, 当前版本仅支持 64
    // 2、支持HEAD_SIZE_V != HEAD_SIZE_QK

    extern __shared__ char shared_mem[];
    scalar_t *smem_q = (scalar_t *)shared_mem;
    float *smem_max = (float *)shared_mem;
    float *smem_sum = (float *)shared_mem + REUSE_KV_TIMES * (NUM_THREADS / WARP_SIZE);

    using scalarx4_t= __attribute__( (__vector_size__(4 * sizeof(scalar_t)) )) scalar_t;
    using scalarx8_t= __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;

    // 当前线程块对应的 Q 方向的线程块索引
    // blockIdx.x   q_tile_idx      kv_head_idx
    //      0           0           0           对应 q_token = 0, Q head[0, 31], K head 0
    //      1           1           0           对应 q_token = 1, Q head[0, 31], K head 0
    //      2           0           1           对应 q_token = 0, Q head[32, 63], K head 1
    //      3           1           1           对应 q_token = 1, Q head[32, 63], K head 1
    const int q_tile_idx = blockIdx.x % NUM_Q_TILES;
    // 当前线程块处理的 kv_head 的索引
    const int kv_head_idx = blockIdx.x / NUM_Q_TILES;
    // 将 num_heads 均分, 每个 Q 方向的线程块处理 相同数量的数据
    const int local_num_heads = num_heads / NUM_Q_TILES;

    // grid_size: (num_kv_heads, max_partition, batch_size)
    // const int kv_head_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;    

    // 一个 warp 一次加载的 kv_token 的数量
    constexpr int NUM_KV_TOKENS_PER_LOAD = 64;

    // 根据当前batch有效的 seqlen_kv, partition 方向多余的线程块提前退出
    int valid_seqlen_kv = __builtin_amdgcn_readfirstlane(seq_lens[batch_idx]);
    int valid_num_block_pages = DIVIDE_ROUND_UP(valid_seqlen_kv, BLOCK_SIZE);
    int valid_num_partition = get_valid_num_partitions(valid_num_block_pages, balanced_seq_len_tile);

    if(partition_idx >= valid_num_partition){
        return;
    }

    // is_query_fp16 true, scalar_t = fp16; is_query_fp16 false, scalar_t = bf16; 
    constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
    // is_kv_fp8 true, cache_t = e5m2; is_kv_fp8 false, cache_t = scalar_t
    constexpr bool is_kv_fp8 = std::is_same<cache_t, uint8_t>::value;
    // qk_gemm 结果的scale
    constexpr float scale_softmax = (HEAD_SIZE == 64 ? 0.125f :
                            (HEAD_SIZE == 128 ? 0.0883883476f :
                            (HEAD_SIZE == 192 ? 0.0721687836f : 0.0625f)));
    constexpr float scale_softmax_log2 = scale_softmax * M_LOG2E;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    // q global -> LDS; Q fragments are loaded on demand in QK GEMM.
    constexpr int Mloop = DIVIDE_ROUND_UP(REUSE_KV_TIMES, 16);

    float alibi_slope[Mloop];
    // alibi slope
    if constexpr (ALIBI_ENABLE){
        int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
        int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;

        int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
        int tile_start = q_tile_idx * rows_per_qtile;
        int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

        for(int m = 0;m < Mloop;m++){
            int local_row  = m * 16 + (lane_id & 15);
            if(local_row < tile_rows){
                int global_row = tile_start + local_row;
                int local_q_head = global_row % num_q_heads_per_kv_head;
                int slope_idx = q_head_idx + local_q_head;
                alibi_slope[m] = alibi_slopes[slope_idx] / scale_softmax;
            }else{
                alibi_slope[m] = 0.0f;
            }
        }
    }

    // Q [num_seqs, seqlen_q, num_q_heads, head_size]
    prefetch_q_to_lds_qtile<scalarx8_t, scalar_t, HEAD_SIZE, NUM_THREADS, Q_LDS_PAD, NUM_Q_TILES>(
        q, smem_q, num_heads, num_kv_heads, mtp, kv_head_idx, batch_idx, q_stride, q_tile_idx);

    float scores_max[Mloop];
    float scores_sum[Mloop];
    float4_t acc_o[Mloop][HEAD_SIZE_V / 16];
    attention_initialize<Mloop, HEAD_SIZE_V>(scores_max, scores_sum, acc_o);

    const int start_block_idx = static_cast<int>(static_cast<int64_t>(partition_idx) * valid_num_block_pages / valid_num_partition);
    const int end_block_idx  = static_cast<int>(static_cast<int64_t>(partition_idx + 1) * valid_num_block_pages / valid_num_partition);
    // 有效block_page_idx: [start_block_idx, end_block_idx)
    const int num_blocks = end_block_idx - start_block_idx;

    // k_cache 形状为 (num_block, num_heads, block_size, head_dim)
    // k_head_stride = BLOCK_SIZE * HEAD_DIM
    // kv_head_idx * k_head_stride 含义为一个 block_page 内, 当前 kv_head 处理的global kv_cache的偏移
    // k_ptr_base 没有考虑 num_blocks 维度的偏移
    constexpr int k_head_stride = BLOCK_SIZE * HEAD_SIZE;
    constexpr int v_head_stride = BLOCK_SIZE * HEAD_SIZE_V;
    const cache_t* k_ptr_base = k_cache + kv_head_idx * k_head_stride;
    const cache_t* v_ptr_base = v_cache + kv_head_idx * v_head_stride;

    const int* block_table = block_tables + batch_idx * max_num_blocks_per_seq;

    for(int block_idx = (start_block_idx + warp_id);block_idx < end_block_idx;block_idx += NUM_WARPS){
        // 1个warp处理一个block_page
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
        // 加上 num_blocks 维度的偏移
        const cache_t *k_ptr = k_ptr_base + physical_block_number * k_block_stride;

        // v_cache 形状为 [num_blocks, num_heads, head_size, block_size]
        // v_cache 在 block_size 维度连续, pv gemm时, v 为列主序
        // 加上 num_blocks 维度的偏移
        const cache_t *v_ptr = v_ptr_base + physical_block_number * v_block_stride;
        
        float4_t s_reg[Mloop][BLOCK_SIZE / 16];
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                s_reg[m][i] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // 每个warp 处理 一个 kv block_page
        qk_gemm_func_with_prefetch_k<scalarx8_t, scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3, Q_LDS_PAD>
                                    (k_ptr, smem_q, local_num_heads / num_kv_heads, s_reg, block_idx, valid_seqlen_kv);

        // alibi
        if constexpr (ALIBI_ENABLE){
            int seqlen_q_per_kv_head = num_heads / num_kv_heads;
            int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;
            int q_head_idx = num_q_heads_per_kv_head * kv_head_idx;

            int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
            int tile_start = q_tile_idx * rows_per_qtile;
            const int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);
            
            for(int m = 0;m < Mloop;m++){
                int local_row  = m * 16 + (lane_id & 15);
                int col = lane_id >> 4;
                if(local_row < tile_rows){
                    int global_row = tile_start + local_row;
                    int q_token_id = global_row / num_q_heads_per_kv_head;
                    int casual_offset = mtp - q_token_id;
                    int query_position = valid_seqlen_kv - casual_offset;
                    for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                        for(int j = 0;j < 4;j++){
                            int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                            if(token_k_index < valid_seqlen_kv){
                                // 边界内
                                int relative_position = token_k_index - query_position;
                                float alibi = alibi_slope[m] * relative_position;
                                // alibi 的 标准形式是 score / 根号d + alibi, 因后面会除以根号d, alibi_slope 已经乘根号d, 此处不需要处理
                                (s_reg[m][i])[j] += alibi;
                            }
                        }
                    }                        
                }
            }
        }         

        // mtp causal
        if constexpr (MTP_ENABLE){
            if (block_idx * BLOCK_SIZE + BLOCK_SIZE - 1 + mtp > valid_seqlen_kv) {
                int seqlen_q_per_kv_head = num_heads / num_kv_heads;
                int num_q_heads_per_kv_head = (num_heads / mtp) / num_kv_heads;

                int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
                int tile_start = q_tile_idx * rows_per_qtile;
                int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

                for(int m = 0;m < Mloop;m++){
                    int local_row  = m * 16 + (lane_id & 15);
                    int col = lane_id >> 4;
                    if(local_row < tile_rows){
                        int global_row = tile_start + local_row;
                        int q_token_id = global_row / num_q_heads_per_kv_head;
                        int casual_offset = mtp - q_token_id;
                        for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                            for(int j = 0;j < 4;j++){
                                int token_k_index = block_idx * BLOCK_SIZE + j * 16 + col * 4 + i;
                                if(token_k_index < valid_seqlen_kv){
                                    // 边界内
                                    if((token_k_index + casual_offset > valid_seqlen_kv)){
                                        (s_reg[m][i])[j] = -INFINITY;
                                    }                           
                                }
                            }
                        }
                    }                  
                }
            }
        } 

        // s_reg: S = 2^(S * softmax_scale_log2 - max_current * softmax_scale_log2) = e^(S * softmax_scale - max_current * softmax_scale)
        // scores_max = new_max, raw max, 没有 乘  scale_softmax_log2
        // scores_sum: 对应公式 d(i) = d(i-1) * e^((m(i-1) - m(i)) * softmax_scale) +  row_sum(e^((S - max_current) * softmax_scale))
        // acc_o: O(i-1) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale)
        softmax_rescale_o<Mloop, HEAD_SIZE_V, BLOCK_SIZE>(s_reg, scores_max, scores_sum, acc_o, scale_softmax_log2);

        scalarx4_t p_reg[Mloop][BLOCK_SIZE / 16];
        // fp32 -> fp16
        for(int m = 0;m < Mloop;m++){
            for(int i = 0;i < (BLOCK_SIZE / 16);i++){
                scalar_t tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[0]);
                (p_reg[m][i])[0] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[1]);
                (p_reg[m][i])[1] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[2]);
                (p_reg[m][i])[2] = tmp_vale;
                from_float(tmp_vale, (s_reg[m][i])[3]);
                (p_reg[m][i])[3] = tmp_vale;
            }
        }

        // pv gemm
        // acc_o: O(i) = O(i-1) * e^((m(i-1) - m(i)) * softmax_scale) + gemm_pv
        pv_gemm_func_with_prefetch_v<scalarx4_t, scalar_t, cache_t, Mloop, HEAD_SIZE_V, BLOCK_SIZE, NUM_KV_TOKENS_PER_LOAD, is_query_fp16, STAGES, is_kv_fp8, is_e4m3>(v_ptr, p_reg, acc_o, block_idx, valid_seqlen_kv);
    }

    // Q remains live in LDS throughout the page loop. Prevent an early warp
    // from reusing the same LDS storage for max/sum while another warp reads Q.
    __syncthreads();

    // sink
    if constexpr(SINK_ENABLE)
    {
        // 对 apply_attention_sink 的理解:
        //      sink 为 人为 在 S 矩阵的每一行增加一个元素, 假设为 sink_value
        //      则 new_max = MAX_T(sink_value, old_max), 根据new_max 更新 scores_max, scores_sum, acc_o
        //      包含一下公式:
        //              max = new_max
        //              d(i) = d(i-1) * e^(m(i - 1) - m(i)) + row_sum(e^(xi(j) - mi))
        //              o(i - 1) = o(i - 1) * e^(m(i - 1) - m(i)), sink 对应的 v 为0, 不考虑 pv 项
        if(0 == partition_idx && 0 == warp_id){
            apply_attention_sink_q_tile<scalar_t, Mloop, HEAD_SIZE_V, NUM_Q_TILES>(
                scores_max,
                scores_sum,
                acc_o,
                s_aux_ptr,
                scale_softmax,
                scale_softmax_log2,
                warp_id,
                partition_idx,
                mtp,
                num_heads,
                num_kv_heads,
                kv_head_idx,
                q_tile_idx
            );
        }
    }


    float scores_max_new[Mloop];
    float scores_sum_new[Mloop];

    epilogue_rescale_acco<Mloop, HEAD_SIZE_V, NUM_WARPS>(scores_max, scores_sum, acc_o, scores_max_new, scores_sum_new,
                                                        smem_max, smem_sum, scale_softmax_log2, warp_id, local_num_heads, num_kv_heads);

    // 当前 acc_o 为经过缩放结果, 需要将4个warp的 acc_o 加起来得到 PARTION_SIZE 的输出
    //  通过lds reduce实现
    {
        const int acc_row = lane_id & 15;
        const int acc_col = lane_id >> 4;
        // 每个kv head处理的token q数量
        const int seqlen_q_per_kv_head = num_heads / num_kv_heads;
        // 每个kv head对应的q head数量
        int q_heads_per_kv_head = seqlen_q_per_kv_head / mtp;
        // 当前block处理的q head起始索引, 即当前kv_head_id对应的q_head_id_start
        int head_q_index = q_heads_per_kv_head * kv_head_idx;
        
        int rows_per_qtile = DIVIDE_ROUND_UP(seqlen_q_per_kv_head, NUM_Q_TILES);
        int tile_start = q_tile_idx * rows_per_qtile;
        int tile_rows = MIN_T(rows_per_qtile, seqlen_q_per_kv_head - tile_start);

        // 若 1 == valid_num_partition, 对应的 out 的形状是 (batch, seqlen_q * num_q_heads, head_dim_v), scalar_t
        // 若 valid_num_partition > 1, 对应的 out_buffer 的形状是(batch, seqlen_q * num_q_heads, max_num_partitions, head_dim_v) scalar_t
        scalar_t *out_ptr_base = nullptr;
        int out_head_stride = 0;
        // num_heads = seqlen_q * num_q_heads
        // out_ptr_base 考虑了 batch 维度, head 维度起始head索引的偏移, 没有考虑 seqlen 维度的偏移
        if (1 == valid_num_partition){
            out_head_stride = HEAD_SIZE_V;
            out_ptr_base = out + batch_idx * num_heads * HEAD_SIZE_V + head_q_index * HEAD_SIZE_V;
        }else{
            out_head_stride = balanced_seq_len_tile  * HEAD_SIZE_V;
            // out_tmp_offset 元素, 为预留的 max sum buffer
            out_ptr_base = out_tmp + out_tmp_offset + batch_idx * num_heads * out_head_stride
                            + head_q_index * out_head_stride + partition_idx * HEAD_SIZE_V;
        }

        // lds reduce 每个迭代处理 NUM_WARPS * 16x16 *4 的数据, 保证每个线程 b128 的写出, 且每个warp需要写
        constexpr int num_16x64_tiles = Mloop * (HEAD_SIZE_V / 64);
        // lds 形状为(NUM_WARPS, WARP_SIZE * 4)
        float *smem_reduce = (float *)shared_mem;
        for(int tile_idx = 0;tile_idx < num_16x64_tiles;tile_idx++){
            const int tile_idx_row = tile_idx / (HEAD_SIZE_V / 64);
            const int tile_idx_col = tile_idx % (HEAD_SIZE_V / 64);
            for(int j = 0;j < 4;j++){
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 0) = (acc_o[tile_idx_row][tile_idx_col * 4 + 0])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 1) = (acc_o[tile_idx_row][tile_idx_col * 4 + 1])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 2) = (acc_o[tile_idx_row][tile_idx_col * 4 + 2])[j];
                *(smem_reduce + warp_id * 16 * (64 + REDUCE_LDS_PAD) + acc_row * (64 + REDUCE_LDS_PAD) + acc_col + 4 * j + 16 * 3) = (acc_o[tile_idx_row][tile_idx_col * 4 + 3])[j];
            }
            __syncthreads();

            
            int out_r2g_row = threadIdx.x >> 4;
            int out_r2g_col = threadIdx.x & 15;
            int local_row = tile_idx_row * 16 + out_r2g_row;

            // 边界限制
            if(local_row < tile_rows){
                // row / q_heads_per_kv_head 表示当前行对应 out 的 token id
                //   例如: num_q_heads = 16, seqlen_q = 2, num_kv_heads = 2
                //         则 q_heads_per_kv_head = num_q_heads / num_kv_heads = 8
                //            seqlen_q_per_kv_head = num_q_heads * seqlen_q / num_kv_heads = 16
                //            kv_head_idx = 0时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head0-7, 8-15 对应 token1 的 head0-7
                //            kv_head_idx = 1时, seqlen_q_per_kv_head 的 0-7 对应 out token0 的 head8-16, 8-15 对应 token1 的 head8-15
                
                //      num_kv_heads * q_heads_per_kv_head 为 num_q_heads
                //      (row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) 含义是 当前 row 对应的 q_token_id 的 q_head 偏移
                //      row % q_heads_per_kv_head + head_q_index, 对应当前行的实际的 q_head 的索引
                int global_row = tile_start + local_row;

                const int out_row = (global_row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + global_row % q_heads_per_kv_head;
                const int our_col = tile_idx_col * 64 + out_r2g_col * 4;
                float4_t sums = {0.0f, 0.0f, 0.0f, 0.0f,};
                for(int i = 0;i < NUM_WARPS;i++){
                    float4_t temp_value = *((float4_t *)(smem_reduce + i * 16 * (64 + REDUCE_LDS_PAD) + out_r2g_row * (64 + REDUCE_LDS_PAD) + out_r2g_col * 4));
                    sums[0] += temp_value[0];
                    sums[1] += temp_value[1];
                    sums[2] += temp_value[2];
                    sums[3] += temp_value[3];
                }

                scalarx4_t out_sums = {from_float<scalar_t>(sums[0]), from_float<scalar_t>(sums[1]), from_float<scalar_t>(sums[2]), from_float<scalar_t>(sums[3])};
                *((scalarx4_t *)(out_ptr_base + out_row * out_head_stride + our_col)) = out_sums;
            }
             __syncthreads();
        }
        
        // 保存 max sum
        if(valid_num_partition > 1){
            // max, sum 形状为 (batch, seqlen_q, num_q_heads, max_num_partitions)
            if(warp_id == 0 && acc_col == 0){
                for(int m = 0; m < Mloop;m++){
                    int local_row  = m * 16 + acc_row;
                    if(local_row < tile_rows){
                        int global_row = tile_start + local_row;
                        int out_row = (global_row / q_heads_per_kv_head) * (num_kv_heads * q_heads_per_kv_head) + global_row % q_heads_per_kv_head;
                        int offset = batch_idx * num_heads * balanced_seq_len_tile 
                                      + (head_q_index + out_row) * balanced_seq_len_tile 
                                      + partition_idx;
                        float *exp_sums = reinterpret_cast<float*>(out_tmp);
                        // max_tmp_offset 元素单位
                        float *max = reinterpret_cast<float*>(out_tmp + max_tmp_offset);

                        exp_sums[offset] = scores_sum_new[m];
                        // 此处存到 global 的 max 经过 scale_softmax_log2
                        max[offset] = scores_max_new[m]  * scale_softmax_log2;
                    }
                }
            } 
        }
    }

    return;
}

void paged_attention_kernel_opt_launcher(
                                        torch::Tensor& out,                                 // [num_seqs, seqlen_q, num_q_heads, head_dim_v]
                                        torch::Tensor& query,                               // [num_seqs, seqlen_q, num_q_heads, head_dim_qk]
                                        torch::Tensor& key_cache,                           // [num_blocks, num_kv_heads, block_size, head_dim_qk]
                                        torch::Tensor& value_cache,                         // [num_blocks, num_kv_heads, head_dim_v, block_size]
                                        double scale,
                                        torch::Tensor& block_tables,                        // [num_seqs, max_num_blocks_per_seq]
                                        torch::Tensor& seq_lens,                            // [num_seqs]
                                        const c10::optional<torch::Tensor>& alibi_slopes,   // [num_q_heads]
                                        const std::string& kv_cache_dtype,
                                        const c10::optional<torch::Tensor>& q_scale,
                                        const c10::optional<torch::Tensor>& k_scale,
                                        const c10::optional<torch::Tensor>& v_scale,
                                        int max_seq_len,
                                        const c10::optional<at::Tensor> &s_aux,             // [seqlen_q * num_q_heads]
                                        float *out_tmp,
                                        bool print_param
                                    )
{
    (void)scale;
    (void)kv_cache_dtype;
    (void)q_scale;
    (void)k_scale;
    (void)v_scale;

    int batch = query.size(0);
    int mtp = query.size(1);
    int num_q_heads = query.size(2);
    int head_dim_qk = query.size(3);

    int num_blocks = key_cache.size(0);
    int num_kv_heads = key_cache.size(1);
    int block_size = key_cache.size(2);

    int head_dim_v = value_cache.size(2);
    int max_num_blocks_per_seq = block_tables.size(1);
    int num_heads = num_q_heads * mtp;

    int real_reuse_times = num_heads / num_kv_heads;
    int reuse_kv=get_reusekv(num_heads, num_kv_heads);

    static int nums_simd = get_nums_simd();
    int target_blocks = nums_simd;
    int base_blocks = batch * num_kv_heads;
    int balanced_seq_len_tile = MAX_T(1, DIVIDE_ROUND_UP(target_blocks, base_blocks));

    int q_stride = query.stride(0);
    int k_block_stride = key_cache.stride(0);
    int v_block_stride = value_cache.stride(0);

    TORCH_CHECK(0 == (head_dim_qk % 64) && head_dim_qk <= 256, "Page Attention head size must be 64, 128, 192 or 256");
    TORCH_CHECK(0 == (head_dim_v % 64) && head_dim_v <= 256, "Page Attention head size must be 64, 128, 192 or 256");
    TORCH_CHECK(num_heads <= (num_kv_heads * 64), "Page Attention qheads*mtp/kvheads must be smaller than 64");
    TORCH_CHECK(64 == block_size, "Page Attention opt only supprot block_size 64 for now!");

    void *query_ptr = query.data_ptr();
    void *ker_cache_ptr = key_cache.data_ptr();
    void *value_cache_ptr = value_cache.data_ptr();
    void *out_ptr = out.data_ptr();
    int *block_tables_ptr = block_tables.data_ptr<int>();
    int *seq_lens_ptr = seq_lens.data_ptr<int>();

    const void *alibi_slopes_ptr = nullptr;
    if(alibi_slopes.has_value()){
        auto alibi_slopes_value = alibi_slopes.value();
        TORCH_CHECK((torch::kFloat32 == alibi_slopes_value.dtype()), "alibi_slopes must be float!");
        CHECK_SHAPE(alibi_slopes_value, num_q_heads);
        CHECK_DEVICE(alibi_slopes_value);
        CHECK_CONTIGUOUS(alibi_slopes_value);
        alibi_slopes_ptr = alibi_slopes_value.data_ptr();
    }

    const void *s_aux_ptr = nullptr;
    if(s_aux.has_value()){
        auto s_aux_value = s_aux.value();
        TORCH_CHECK(s_aux_value.dtype() == query.dtype(),
                    "s_aux must have the same dtype as query. Got s_aux dtype: ", s_aux_value.dtype(),
                    ", query dtype: ", query.dtype());
        TORCH_CHECK(s_aux_value.dtype() == torch::kFloat16 || s_aux_value.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match query). Got: ", s_aux_value.dtype());
        TORCH_CHECK(num_heads <= 64,
                    "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux_value);
        CHECK_SHAPE(s_aux_value, num_heads);
        CHECK_CONTIGUOUS(s_aux_value);
        s_aux_ptr = s_aux_value.data_ptr();
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    HEADSIZE_SWITCH(head_dim_qk, [&] {
        Output_Type_SWITCH(out.dtype(), [&] {
            Cache_Type_SWITCH(scalar_t, key_cache.dtype(), [&] {
                REUSEKV_SWITCH(reuse_kv, [&] {

                    // using scalar_t = uint16_t;
                    // using cache_t = uint8_t;
                    // constexpr bool is_e4m3 = false;
                    constexpr bool SINK_ENABLE = false;
                    constexpr bool ALIBI_ENABLE = false;
                    constexpr bool MTP_ENABLE = true;

                    constexpr int STAGES = 2;
                    constexpr int BLOCK_SIZE = 64;
                    // HEAD_SIZE scalar_t cache_t is_e4m3 REUSE_KV_TIMES 编译常量已经在宏中定义
                    constexpr static int NUM_THREADS = 256;
                    constexpr static int NUM_WARPS = NUM_THREADS / WARP_SIZE;
                    constexpr int HEAD_SIZE_V = HEAD_SIZE;

                    // Padding 缓解 bc
                    constexpr int Q_LDS_PAD = 8;
                    constexpr int REDUCE_LDS_PAD = 4;

                    TORCH_CHECK(2 == sizeof(scalar_t), "PA only support scalar_t b16 type for now!");
                    TORCH_CHECK(false == is_e4m3, "PA onlyt sopprt b8 is e5m2 for now!");
                    // TORCH_CHECK(0 == PARTITION_SIZE % BLOCK_SIZE);
                    // int max_num_partitions=DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);

                    if constexpr (REUSE_KV_TIMES <= 32){
                        int smem_q_size = REUSE_KV_TIMES * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t);
                        // 每次处理 4 个 16x16 的 acc 结果
                        // smem_reduce 的形状是 [NUM_WARPS, 16 * 16 * 4]
                        int smem_reduce_size = NUM_WARPS * 16 * (16 * 4 + REDUCE_LDS_PAD) * sizeof(float);
                        // 两个 smem 空间复用, 取较大值
                        int smem_size = MAX_T(smem_q_size, smem_reduce_size);
                        
                        dim3 grid_size(num_kv_heads, balanced_seq_len_tile, batch);
                        dim3 kernel_block_size(NUM_THREADS);
                        if(print_param){
                            constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
                            printf("is_e4m3 [%d], is_query_fp16 [%d], HEAD_SIZE [%d], BLOCK_SIZE [%d], REUSE_KV_TIMES [%d], batch [%d], mtp [%d], "
                                    "num_q_heads [%d], head_dim_qk [%d], num_blocks [%d], num_kv_heads [%d], head_dim_v [%d], "
                                    "max_num_blocks_per_seq [%d], max_seq_len [%d], sizeof(scalar_t) [%d], sizeof(cache_t) [%d], "
                                    " balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d]\n", 
                                    is_e4m3,
                                    is_query_fp16,
                                    HEAD_SIZE,
                                    BLOCK_SIZE,
                                    REUSE_KV_TIMES,
                                    batch,
                                    mtp,
                                    num_q_heads,
                                    head_dim_qk,
                                    num_blocks,
                                    num_kv_heads,
                                    head_dim_v,
                                    max_num_blocks_per_seq,
                                    max_seq_len,
                                    sizeof(scalar_t),
                                    sizeof(cache_t),
                                    balanced_seq_len_tile,
                                    grid_size.x,
                                    grid_size.y,
                                    grid_size.z
                                );
                        }
                        paged_attention_kernel_opt<scalar_t, cache_t, is_e4m3, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, NUM_THREADS, REUSE_KV_TIMES, STAGES,
                                                    Q_LDS_PAD, REDUCE_LDS_PAD, SINK_ENABLE, ALIBI_ENABLE, MTP_ENABLE>
                            <<<grid_size, kernel_block_size, smem_size, stream>>>(
                                reinterpret_cast<scalar_t *>(out_ptr),
                                reinterpret_cast<scalar_t *>(out_tmp),
                                reinterpret_cast<scalar_t *>(query_ptr),
                                reinterpret_cast<cache_t *>(ker_cache_ptr),
                                reinterpret_cast<cache_t *>(value_cache_ptr),
                                num_heads,
                                num_kv_heads,
                                block_tables_ptr,
                                seq_lens_ptr,
                                max_num_blocks_per_seq,
                                reinterpret_cast<const float*>(alibi_slopes_ptr),
                                q_stride,
                                k_block_stride,
                                v_block_stride,
                                balanced_seq_len_tile,
                                reinterpret_cast<const scalar_t*>(s_aux_ptr),
                                mtp
                            );
                        if(balanced_seq_len_tile > 1){
                            dim3 reduce_grid_size(num_heads, batch);
                            dim3 reduce_block_size(64);
                            int reduce_smem_size = balanced_seq_len_tile * 2 * sizeof(float);
                            paged_attention_combine_opt<scalar_t, HEAD_SIZE_V, 64, BLOCK_SIZE>
                                <<<reduce_grid_size, reduce_block_size, reduce_smem_size, stream>>>(
                                    reinterpret_cast<scalar_t *>(out_ptr),
                                    reinterpret_cast<scalar_t *>(out_tmp),
                                    seq_lens_ptr,
                                    balanced_seq_len_tile,
                                    num_heads
                                );
                        }
                    }else{
                        // REUSE_KV_TIMES > 32
                        constexpr int NUM_Q_TILES = 2;
                        // 每个Q维度的线程块处理的 REUSE_KV_TIMES
                        constexpr int Q_TILE = DIVIDE_ROUND_UP(REUSE_KV_TIMES, NUM_Q_TILES);
                        
                        TORCH_CHECK(0 == (num_heads / num_kv_heads) % NUM_Q_TILES);

                        const dim3 grid_qtiles(num_kv_heads * NUM_Q_TILES, balanced_seq_len_tile, batch);
                        const dim3 block_qtiles(NUM_THREADS);
                        
                        int smem_q_size = Q_TILE * (HEAD_SIZE + Q_LDS_PAD) * sizeof(scalar_t);
                        int smem_reduce_size = NUM_WARPS * 16 * (16 * 4 + REDUCE_LDS_PAD) * sizeof(float);
                        int tile_smem_size = MAX_T(smem_q_size, smem_reduce_size);

                        if(print_param){
                            constexpr bool is_query_fp16 = std::is_same<scalar_t, _Float16>::value;
                            printf("is_e4m3 [%d], is_query_fp16 [%d], HEAD_SIZE [%d], BLOCK_SIZE [%d], REUSE_KV_TIMES [%d], batch [%d], mtp [%d], "
                                    "num_q_heads [%d], head_dim_qk [%d], num_blocks [%d], num_kv_heads [%d], head_dim_v [%d], "
                                    "max_num_blocks_per_seq [%d], max_seq_len [%d], sizeof(scalar_t) [%d], sizeof(cache_t) [%d], "
                                    " balanced_seq_len_tile [%d], grid_size [%d] [%d] [%d]\n", 
                                    is_e4m3,
                                    is_query_fp16,
                                    HEAD_SIZE,
                                    BLOCK_SIZE,
                                    REUSE_KV_TIMES,
                                    batch,
                                    mtp,
                                    num_q_heads,
                                    head_dim_qk,
                                    num_blocks,
                                    num_kv_heads,
                                    head_dim_v,
                                    max_num_blocks_per_seq,
                                    max_seq_len,
                                    sizeof(scalar_t),
                                    sizeof(cache_t),
                                    balanced_seq_len_tile,
                                    grid_qtiles.x,
                                    grid_qtiles.y,
                                    grid_qtiles.z
                                );
                        }

                        paged_attention_kernel_qtile_opt<scalar_t, cache_t, is_e4m3, HEAD_SIZE, HEAD_SIZE_V, BLOCK_SIZE, NUM_THREADS, Q_TILE, STAGES,
                                Q_LDS_PAD, REDUCE_LDS_PAD, SINK_ENABLE, ALIBI_ENABLE, MTP_ENABLE, NUM_Q_TILES>
                            <<<grid_qtiles, block_qtiles, tile_smem_size, stream>>>(
                                reinterpret_cast<scalar_t *>(out_ptr),
                                reinterpret_cast<scalar_t *>(out_tmp),
                                reinterpret_cast<scalar_t *>(query_ptr),
                                reinterpret_cast<cache_t *>(ker_cache_ptr),
                                reinterpret_cast<cache_t *>(value_cache_ptr),
                                num_heads,
                                num_kv_heads,
                                block_tables_ptr,
                                seq_lens_ptr,
                                max_num_blocks_per_seq,
                                reinterpret_cast<const float*>(alibi_slopes_ptr),
                                q_stride,
                                k_block_stride,
                                v_block_stride,
                                balanced_seq_len_tile,
                                reinterpret_cast<const scalar_t*>(s_aux_ptr),
                                mtp
                            );
                        if(balanced_seq_len_tile > 1){
                            dim3 reduce_grid_size(num_heads, batch);
                            dim3 reduce_block_size(64);
                            int reduce_smem_size = balanced_seq_len_tile * 2 * sizeof(float);
                            paged_attention_combine_opt<scalar_t, HEAD_SIZE_V, 64, BLOCK_SIZE>
                                <<<reduce_grid_size, reduce_block_size, reduce_smem_size, stream>>>(
                                    reinterpret_cast<scalar_t *>(out_ptr),
                                    reinterpret_cast<scalar_t *>(out_tmp),
                                    seq_lens_ptr,
                                    balanced_seq_len_tile,
                                    num_heads
                                );
                        }
                    }
                });
            });
        });
    });

    return;
}
