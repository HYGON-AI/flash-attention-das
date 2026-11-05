// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "paged_attention.h"
#include "paged_attention_window_opt.h"
#include <cmath>
#include <map>
#include <mutex>
#include <utility>
#include <type_traits>

static std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    return raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.
}

const std::string device_name=get_device_name();

static inline int get_env_(const char *env_var) {
  if (char *value = std::getenv(env_var)) {
    return atoi(value);
  }
  return 0;
}

const int PA_PRINT_PARAM = get_env_("PA_PRINT_PARAM");
const int PA_PARTITION_SIZE = get_env_("PA_PARTITION_SIZE");
const int PA_NO_SPLIT_KV = get_env_("PA_NO_SPLIT_KV");
const int PA_REFACTORING_DISABLE = get_env_("PA_REFACTORING_DISABLE");
const int PA936_LEGACY_LAYOUTS = std::getenv("PA936_LEGACY_LAYOUTS") ? get_env_("PA936_LEGACY_LAYOUTS") : 1;
const int PA936_SMALL_SCHEDULER = std::getenv("PA936_SMALL_SCHEDULER") ? get_env_("PA936_SMALL_SCHEDULER") : 1;
// Default to the selected fixed-worker policy; 0/1 remain available for comparisons.
const int PA_GRAPH_COMPACT = std::getenv("PA_GRAPH_COMPACT") ? get_env_("PA_GRAPH_COMPACT") :
    (device_name == "gfx936" ? 1 : 2);
// -1 selects the architecture policy when the actual launch device is known.
const int PA_GRAPH_WORKERS = std::getenv("PA_GRAPH_WORKERS") ? get_env_("PA_GRAPH_WORKERS") : -1;


#include "paged_attention_decode_arch.h"

/*
 * [窗口优化-设备端参数]
 * window_size_left/effective_right 负责 attention 位置 mask；
 * partition_window_size_left 只负责决定压缩后的 active KV page 起点。
 * 两者分开传递，保证 mask 语义与 partition 调度逻辑互不混淆。
 */
template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, bool PERSISTENT = false, bool SMALL_BATCH = false>
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_heads, block_size, head_size]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,const int kv_block_stride,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left,
    bool force_partial = false,               // always write partials to out_tmp, never out
    const int* __restrict__ q_rows = nullptr,
    int num_seqs = 0) {
  using Scheduler = std::conditional_t<SMALL_BATCH, Pa936SmallDecodeScheduler,
                                        Pa936DecodeTileScheduler<PERSISTENT>>;
  Scheduler scheduler(
      seq_lens, num_seqs, num_kv_heads, mtp, partition_window_size_left,
      BLOCK_SIZE, PARTITION_SIZE);
  int seq_idx, scheduled_head, partition_idx;
  for (; scheduler.next(seq_idx, scheduled_head, partition_idx); scheduler.advance()) {
  constexpr int kv_head_stride=BLOCK_SIZE*HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-统一 active range]
   * 通过公共 helper 计算当前 sequence 的活动 page、partition 数量以及
   * local partition 对应的真实 block 范围。这里的 blockIdx.z 是压缩后的
   * local partition 编号，不是原始完整序列的 partition 编号。
   */
  const PagedAttentionActiveRange active_range =
      get_paged_attention_active_range(
          seq_len,
          mtp,
          partition_window_size_left,
          BLOCK_SIZE,
          PARTITION_SIZE,
          partition_idx);
  const int num_seq_blocks = active_range.num_seq_blocks;
  const int num_active_partitions = active_range.num_active_partitions;
  if(num_active_partitions<=partition_idx) {
    if constexpr (PERSISTENT) continue;
    else return;
  }
  constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
  constexpr bool is_fp8 = std::is_same<cache_t, uint8_t>::value;
  constexpr float scale = (HEAD_SIZE==64?0.125f:(HEAD_SIZE==128? 0.0883883476f:(HEAD_SIZE==192?0.0721687836f:0.0625f)))*1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = scheduled_head;
  const int head_idx=num_queries_per_kv/mtp * kv_head_idx;
  constexpr int reuse_group=(REUSE_KV_TIMES-1)/4+1;
  constexpr int Mloop=(REUSE_KV_TIMES-1)/16+1;
  extern __shared__ char shared_mem[];
  using ldstype = _Float16;
  constexpr float lds_scale =  (HEAD_SIZE==128&&REUSE_KV_TIMES==24) ? 512.0f:8.0f ;
  constexpr float qkgemm_scale = scale/lds_scale;
  ldstype* logits_f16 = reinterpret_cast<ldstype*>(shared_mem);
  scalar_t* logits = reinterpret_cast<scalar_t*>(shared_mem);
  float* s_max = reinterpret_cast<float*>(shared_mem + sizeof(scalar_t)*num_queries_per_kv*PARTITION_SIZE);
  float* s_logit = s_max + num_queries_per_kv * NUM_WARPS;
  float* max_out = s_logit+NUM_WARPS;
  float* expsum_out = max_out+num_queries_per_kv;

  // ★ Attention Sinks: load s_aux to shared memory ★
  __shared__ scalar_t smem_s_aux[64];
  if (s_aux_ptr != nullptr) {
    if (thread_idx < num_heads) {
      smem_s_aux[thread_idx] = s_aux_ptr[thread_idx];
    }
    __syncthreads();
  }

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  float alibi_slope[reuse_group]={0.f};
  if (has_abili){
    for(int i=0;i<reuse_group;i++){
      int reuse_kv_idx=rows+i*4;
      if(reuse_kv_idx<num_queries_per_kv) alibi_slope[i]=alibi_slopes[head_idx+reuse_kv_idx]*1.4426950408889634;
    }
  }
  float qk_max[reuse_group];
  for(int i=0;i<reuse_group;i++){
    qk_max[i]=-FLT_MAX;
  }
  half4x2 q_vec[Mloop][HEAD_SIZE/32];
  half4x2 q_zero;
  q_zero.data[0]={0,0,0,0};
  q_zero.data[1]={0,0,0,0};
  scalar_t* s_q = reinterpret_cast<scalar_t*>(shared_mem);
  {
    int head_offset = HEAD_SIZE*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE;i+=NUM_THREADS*8){
      if (q_rows != nullptr) {
        // Gathered queries: mtp slot j of sequence seq_idx is row q_rows[seq_idx*mtp+j]
        // of q (q_stride apart, [rows, num_heads/mtp, head_size]); -1 = zero padding.
        const int j = i / head_offset;
        const int row = q_rows[seq_idx * mtp + j];
        half4x2 val = q_zero;
        if (row >= 0)
          val = *reinterpret_cast<const half4x2*>(q + (int64_t)row * q_stride + head_idx * HEAD_SIZE + i % head_offset);
        *reinterpret_cast<half4x2*>(s_q+i) = val;
        continue;
      }
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;
      *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
    }
  }

  __syncthreads();
  for(int m=0;m<Mloop;m++){
    int head_idx_=rowid+16*m;
    for(int i=0;i<HEAD_SIZE/32;i++){
      if(head_idx_<num_queries_per_kv)q_vec[m][i]=*reinterpret_cast<const half4x2*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*8);
      else q_vec[m][i]=q_zero;
    }
  }
  __syncthreads();
  /*
   * [窗口优化-KV page 映射]
   * 使用 helper 返回的范围，从窗口覆盖到的第一个 page 开始重新划分
   * partition，保证 local partition_idx 直接对应真实 block_table page。
   */
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //comput q*k
  {
    const cache_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const cache_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*HEAD_SIZE;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        half4x2 k_vec[HEAD_SIZE/32];
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/32;i++){
          if constexpr(is_fp8){
            uint8x4x2 k_vec_u8=*reinterpret_cast<const uint8x4x2*>(k_ptr+i*32+rowid*HEAD_SIZE+rows*8);
            scalar_t *p1=(scalar_t*)(k_vec+i);
            uint8_t *p2=(uint8_t*)&k_vec_u8;
            for(int ii=0;ii<8;ii++){
              p1[ii]=uint82half<scalar_t,is_e4m3>(p2[ii]);
            }
          }
          else{
            k_vec[i]=*reinterpret_cast<const half4x2*>(k_ptr+i*32+rowid*HEAD_SIZE+rows*8);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/32;i++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac<is_half>(k_vec[i].data[0],q_vec[m][i].data[0],qk_vec[m]);
            builtin_amdgcn_mmac<is_half>(k_vec[i].data[1],q_vec[m][i].data[1],qk_vec[m]);
          }
        }
        #pragma unroll
        for(int i=0;i<reuse_group;i++){
          int reuse_kv_idx=rows+i*4;
          int m = reuse_kv_idx/16;
          int ii = i%4;
          if(reuse_kv_idx<num_queries_per_kv){
            qk_vec[m][ii]*=qkgemm_scale;
            const int token_idx = block_idx * BLOCK_SIZE+rowid + b;
            if (has_abili){
              float alibi=alibi_slope[i] * (token_idx - seq_len + 1);
              qk_vec[m][ii] += alibi;
            }
            /*
             * [窗口优化-统一 mask]
             * cache_t 可能是 FP16/BF16，也可能是 uint8_t 形式的 FP8。
             * mask 只描述 query/key 的位置关系，不能因为 cache 是 FP8 就被编译掉。
             */
            bool window_masked = false;
            const int query_idx = reuse_kv_idx * mtp / num_queries_per_kv;
            const int query_pos = seq_len - mtp + query_idx;
            const bool mask_left = window_size_left >= 0 &&
                token_idx < query_pos - window_size_left;
            const bool mask_right = effective_right >= 0 &&
                token_idx > query_pos + effective_right;
            if (mask_left || mask_right) {
              qk_vec[m][ii] = -INFINITY;
              window_masked = true;
            }
            if(token_idx >= seq_len) {
              int seq_len_pad=DIVIDE_ROUND_UP(seq_len,8)*8;
              if(token_idx<seq_len_pad) from_float(logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx],-INFINITY);
              else logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx]=0;
            }
            else{
              ldstype temp;
              /*
               * [窗口优化-因果语义]
               * 旧的 mtp>1 隐式 causal 已删除；causal 统一由 host 端的
               * effective_right=0 转换为这里的 right mask。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              /*
               * [窗口优化-softmax]
               * masked score 写入 logits，但不能参与本 partition 的最大值计算。
               */
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_float(temp)*lds_scale);
              }
              // if(partition_idx==0)printf("tid=%d,tokenid=%d,reuse_kv_idx=%d,m=%d,ii=%d,qk=%f\n",thread_idx,token_idx,reuse_kv_idx,m,i,qk_vec[m][ii]);
            }
          }
        }
      }
    }
  }
  // compute max
  #pragma unroll
  for (int mask = 8; mask >= 1; mask /= 2) {
    #pragma unroll
    for(int r=0;r<reuse_group;r++){
      qk_max[r]=fmaxf(qk_max[r],__shfl_xor(qk_max[r],mask));
    }
  }
  #pragma unroll
  for(int r=0;r<reuse_group;r++){
    if(rowid==0&&r*4+rows<num_queries_per_kv){
      s_max[(r*4+rows)*NUM_WARPS+warp_idx] = qk_max[r];
    }
  }
  __syncthreads();
  if(PARTITION_SIZE==256){
    for(int lineid = warp_idx;lineid<REUSE_KV_TIMES/2;lineid+=NUM_WARPS){
      int half_lane = lane%32;
      int which_half = lane/32;
      int real_line=lineid*2+which_half;
      if(real_line<num_queries_per_kv){
        float qk_max_tmp;
        float exp_sum=0;
        if(half_lane==0){
          int smax_offset = real_line*4;
          qk_max_tmp=s_max[smax_offset];
          for(int i=1;i<4;i++){
            qk_max_tmp=fmaxf(qk_max_tmp,s_max[smax_offset+i]);
          }
        }
        qk_max_tmp=__shfl(qk_max_tmp,which_half*32);
        int seq_len_pad = DIVIDE_ROUND_UP(num_tokens,8);
        using float16x8_t = __attribute__( (__vector_size__(8 * sizeof(ldstype)) )) ldstype;
        using f16x8_t = __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(half_lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii])*lds_scale-qk_max_tmp);
            exp_sum+=logit32[ii];
          }
          // printf("tid=%d,logit32=%.4f,%.4f,%.4f,%.4f, %.4f,%.4f,%.4f,%.4f\n",thread_idx,logit32[0],logit32[1],logit32[2],logit32[3],logit32[4],logit32[5],logit32[6],logit32[7]);
        }
        for (int mask = 16; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(half_lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8)=logit16;
          /*
           * [窗口优化-临时结果]
           * 只有存在多个 active partition 时才写 partial softmax 统计量，
           * 并使用压缩后的 partition 编号供 combine 阶段读取。
           */
          if((force_partial||num_active_partitions>1)&&half_lane==0){
            max_out[real_line] = qk_max_tmp;
            expsum_out[real_line] = exp_sum;
          }
        }
      }
    }
  }
  else if(PARTITION_SIZE==512){
    for(int lineid = warp_idx;lineid<num_queries_per_kv;lineid+=NUM_WARPS){
      if(lineid<num_queries_per_kv){
        float qk_max_tmp;
        float exp_sum=0;
        if(lane==0){
          int smax_offset = lineid*4;
          qk_max_tmp=s_max[smax_offset];
          for(int i=1;i<4;i++){
            qk_max_tmp=fmaxf(qk_max_tmp,s_max[smax_offset+i]);
          }
        }
        qk_max_tmp=__shfl(qk_max_tmp,0);
        int seq_len_pad = DIVIDE_ROUND_UP(num_tokens,8);
        using float16x8_t = __attribute__( (__vector_size__(8 * sizeof(ldstype)) )) ldstype;
        using f16x8_t = __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii])*lds_scale-qk_max_tmp);
            exp_sum+=logit32[ii];
          }
          // printf("tid=%d,logit32=%.4f,%.4f,%.4f,%.4f, %.4f,%.4f,%.4f,%.4f\n",thread_idx,logit32[0],logit32[1],logit32[2],logit32[3],logit32[4],logit32[5],logit32[6],logit32[7]);
        }
        for (int mask = 32; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8)=logit16;
          /*
           * [窗口优化-临时结果]
           * PARTITION_SIZE=512 的分支同样依据 active partition 数量写入
           * partial max 和 exp_sum，避免继续使用原始完整 partition 判断。
           */
          if((force_partial||num_active_partitions>1)&&lane==0){
            max_out[lineid] = qk_max_tmp;
            expsum_out[lineid] = exp_sum;
          }
        }
      }
    }
  }
  __syncthreads();
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE, 16*NUM_WARPS);//2
  constexpr int GROUPS=reuse_group*4;
  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float4_t accs[Mloop][NUM_ROWS_PER_THREAD];
  for(int m=0;m<Mloop;m++){
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[m][i] = {0.f,0.f,0.f,0.f};
    }
  }
  constexpr int vecsize=BLOCK_SIZE/16;
  using half4_vec = half4vec<vecsize>;
  using uint8x4_vec = uint8x4vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    half4_vec logits_vec[Mloop];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m].data[i]={0,0,0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        logits_vec[m] = *reinterpret_cast<half4_vec*>(logits + real_row * PARTITION_SIZE+token_idx - start_token_idx);
      }
    }
    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                          kv_head_idx * kv_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*vecsize*4+rowid*BLOCK_SIZE;
        half4_vec v_vec;
        if constexpr(is_fp8){
          uint8x4_vec vecu8 = *reinterpret_cast<const uint8x4_vec*>(v_ptr + offset);
          scalar_t *p1=(scalar_t*)&v_vec;
          uint8_t *p2=(uint8_t*)&vecu8;
          for(int ii=0;ii<vecsize*4;ii++){
            p1[ii]=uint82half<scalar_t,is_e4m3>(p2[ii]);
          }
        }else{
          v_vec=*reinterpret_cast<const half4_vec*>(v_ptr + offset);
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac<is_half>(v_vec.data[ii],logits_vec[m].data[ii],accs[m][i]);
          }
        }
      } 
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*vecsize*4+rowid*BLOCK_SIZE;
        half4_vec v_vec;
        if constexpr(is_fp8){
          uint8x4_vec vecu8 = *reinterpret_cast<const uint8x4_vec*>(v_ptr + offset);
          scalar_t *p1=(scalar_t*)&v_vec;
          uint8_t *p2=(uint8_t*)&vecu8;
          for(int ii=0;ii<vecsize*4;ii++){
            p1[ii]=uint82half<scalar_t,is_e4m3>(p2[ii]);
          }
        }else{
          v_vec=*reinterpret_cast<const half4_vec*>(v_ptr + offset);
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
        scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
        #pragma unroll
        for (int j = 0; j < 4*vecsize; j++) {
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : 0;
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac<is_half>(v_vec.data[ii],logits_vec[m].data[ii],accs[m][i]);
          }
        }
      } 
    }
  }
  {
    scalar_t* out_ptr_base;
    int out_offset;
    /*
     * [窗口优化-输出布局]
     * active partition 多于一个时写 out_tmp；否则直接写最终 out。
     * 这里必须与 host 端的 max_num_partitions 以及 combine 的索引规则一致。
     */
    if(force_partial||num_active_partitions>1){
      out_offset=max_num_partitions*HEAD_SIZE;
      out_ptr_base=out_tmp+out_tmp_offset + seq_idx * num_heads * out_offset + head_idx*out_offset+partition_idx * HEAD_SIZE;
    }
    else{
      out_offset=HEAD_SIZE;
      out_ptr_base=out + seq_idx * num_heads  * HEAD_SIZE + head_idx*HEAD_SIZE;
    }
    int head_offset = num_queries_per_kv/mtp;
    for(int g=0;g<reuse_group;g++){
      int reusekvid=g*4+rows;
      if(reusekvid<num_queries_per_kv){
        int out_head = reusekvid/head_offset*num_kv_heads*head_offset + reusekvid%head_offset;
        scalar_t* out_ptr = out_ptr_base + out_head*out_offset;
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx = rowid+16*warp_idx + i * WARP_SIZE;
          from_float(*(out_ptr + row_idx), accs[reusekvid/16][i][g%4]);
          // if(reusekvid==0)printf("patition=%d,tid=%d,i=%d,g=%d,acc=%f\n",partition_idx,thread_idx,i,g,accs[i][g]);
        }
      }
    }
    /*
     * [窗口优化-统计量索引]
     * exp_sums/max_logits 使用 local partition_idx 写入 compact buffer，
     * combine 阶段也必须按相同的 local 编号读取。
     */
    if ((force_partial||num_active_partitions>1)&&thread_idx < num_queries_per_kv){
      int out_head = thread_idx/head_offset*num_kv_heads*head_offset + thread_idx%head_offset;
      int offset = seq_idx * num_heads * max_num_partitions + (head_idx+out_head) * max_num_partitions + partition_idx;
      float * exp_sums=reinterpret_cast<float*>(out_tmp);
      float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
      *(exp_sums+offset)=expsum_out[thread_idx];
      *(max_logits+offset)=max_out[thread_idx];
    }
  }
  if constexpr (PERSISTENT) __syncthreads();
  }
}

int get_reusekv(int qhead,int kv_head){
  if(qhead>kv_head*48) return 64;
  if(qhead>kv_head*32) return 48;
  if(qhead>kv_head*24) return 32;
  if(qhead>kv_head*16) return 24;
  if(qhead>kv_head*8) return 16;
  if(qhead>kv_head*4)return 8;
  return  4;
}

void paged_attention_refactoring(
                                  torch::Tensor& out,                                 
                                  torch::Tensor& query,                               
                                  torch::Tensor& key_cache,                           
                                  torch::Tensor& value_cache,                         
                                  double scale,
                                  torch::Tensor& block_tables,                        
                                  torch::Tensor& seq_lens,                            
                                  const c10::optional<torch::Tensor>& alibi_slopes,   
                                  const std::string& kv_cache_dtype,
                                  const c10::optional<torch::Tensor>& q_scale,
                                  const c10::optional<torch::Tensor>& k_scale,
                                  const c10::optional<torch::Tensor>& v_scale,
                                  int max_seq_len,
                                  const c10::optional<at::Tensor> &s_aux,             
                                  float *out_tmp,
                                  bool print_param,
                                  const bool use_fp8_compute,
                                  int layout
                                  )
{
  if(use_fp8_compute){
    if(layout == 1){
      paged_attention_kernel_bhsd_fp8_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM
                                );
      return;    
    }else if(layout == 2){
      paged_attention_kernel_bshd_fp8_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM                             
                                );
      return;
    }else{
      paged_attention_kernel_fp8_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM                             
                                );
      return;
    }
  }else{
    if(layout == 1){
      paged_attention_kernel_bhsd_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM                             
                                );
      return;    
    }else if(layout == 2){
      paged_attention_kernel_bshd_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM                             
                                );
      return;
    }else{
      paged_attention_kernel_opt_launcher(out, 
                                  query,
                                  key_cache,
                                  value_cache,
                                  scale,
                                  block_tables,
                                  seq_lens,
                                  alibi_slopes,
                                  kv_cache_dtype,
                                  q_scale,
                                  k_scale,
                                  v_scale,
                                  max_seq_len,
                                  s_aux,
                                  out_tmp,
                                  PA_PRINT_PARAM                             
                                );
      return;
    }
  }
}

// Eager calls reuse a size-checked buffer per device/stream. Capture always
// uses the caller's allocation from that graph's private pool, so independent
// graphs and eager calls on other streams cannot overwrite its partials.
static constexpr int PA_TMP_OUT_MAX_STREAMS = 4;
static float* paged_attention_cached_tmp_out(const torch::Tensor& query, int64_t bytes)
{
  const int device = query.get_device();
  const hipStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  const bool capturing = hipStreamIsCapturing(stream, &capture_status) != hipSuccess || capture_status != hipStreamCaptureStatusNone;
  if (capturing) return nullptr;
  // 同一线程在同一设备、同一流上的重复调用不加锁、不查表
  struct Last { int device; hipStream_t stream; float* ptr; int64_t bytes; };
  static thread_local Last last = {-1, nullptr, nullptr, 0};
  if (last.ptr != nullptr && last.device == device && last.stream == stream && last.bytes >= bytes) return last.ptr;
  static std::mutex lock;
  // 不析构的堆对象: 进程退出时不在 PyTorch 分配器析构之后再释放显存; std::map 的元素地址在插入后不变
  static auto* buffers = new std::map<std::pair<int, hipStream_t>, at::Tensor>();
  static auto* streams_per_device = new std::map<int, int>();
  std::lock_guard<std::mutex> hold(lock);
  auto found = buffers->find({device, stream});
  const at::Tensor* buffer = found != buffers->end() ? &found->second : nullptr;
  if (buffer == nullptr) {
    int& cached_streams = (*streams_per_device)[device];
    if (cached_streams >= PA_TMP_OUT_MAX_STREAMS) return nullptr;
    buffer = &buffers->emplace(std::make_pair(device, stream), at::empty({bytes}, query.options().dtype(at::kByte))).first->second;
    ++cached_streams;
  }
  // Keep cached allocations alive; larger calls receive a separate allocation.
  if (buffer == nullptr || buffer->numel() < bytes) return nullptr;
  float* ptr = reinterpret_cast<float*>(buffer->data_ptr());
  last = {device, stream, ptr, buffer->numel()};
  return ptr;
}

extern "C"
void paged_attention(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, //auto,int8,fp8/fp8_e4m3
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    int layout,
    int window_size_left,
    int window_size_right,
    bool is_causal)
{
  /*
   * [窗口优化-接口语义]
   * host 端保留原始 left/right，同时把 causal 转换为 effective_right=0。
   * 因此 device kernel 只需要统一处理 left 和 effective_right 两个边界。
   */
  const at::cuda::OptionalCUDAGuard entry_device_guard(device_of(query));   // kernel 与临时缓冲区都在 query 所在设备上
  const int effective_right = is_causal ? 0 : window_size_right;
  const int provided_seq_len_upper = max_seq_len;
  int num_seqs = query.size(0);
  int headsize=query.size(3);
  int headsize_v=layout==0?value_cache.size(-2):value_cache.size(-1);
  int block_size=key_cache.size(2);
  int mtp = query.size(1);
  int num_blocks = key_cache.size(0);
  int max_num_blocks_per_seq = block_tables.size(1);
  int num_heads = query.size(2)*mtp;
  int num_kv_heads = key_cache.size(1);
  if(layout==2){
    block_size = num_kv_heads;
    num_kv_heads = key_cache.size(2);
  }
  int real_reuse_times = num_heads/num_kv_heads;
  const bool use_fp8_compute=(device_name=="gfx938"|| device_name == "gfx92a")&&(key_cache.dtype()==torch::kFloat8_e4m3fn);
  const bool use_fp8_lds = use_fp8_compute&&key_cache.dtype()==torch::kFloat8_e4m3fn&&real_reuse_times>16;
  int PARTITION_SIZE=512;
  int reusekv=get_reusekv(num_heads,num_kv_heads);
  if(real_reuse_times>15)PARTITION_SIZE=256;
  if(use_fp8_lds&&real_reuse_times<32)PARTITION_SIZE=512;
  if(PA_PARTITION_SIZE!=0)PARTITION_SIZE=PA_PARTITION_SIZE;
  int temp_out_size=200000000;
  if(device_name=="gfx938")temp_out_size=500000000;
  if (max_seq_len<16||max_seq_len>8192){
    int maxseq = temp_out_size/num_seqs/headsize/num_heads*64;
    if(PARTITION_SIZE==512) maxseq*=2;
    if(max_seq_len>8192)maxseq=MIN(max_seq_len,maxseq);
    max_seq_len=maxseq;
  }
  max_seq_len=MIN(max_num_blocks_per_seq*block_size,max_seq_len);
  if(PA_NO_SPLIT_KV)max_seq_len=512;
  /*
   * [窗口优化-host 端压缩条件]
   * 当前 compaction 覆盖 layout=0/1/2 的 64-token page，并要求 left
   * 窗口有效。FP8 与非 FP8 共用同一套 active partition 计算；非 64-token
   * page 仍使用完整 partition 数量。
   */
  const bool use_window_partition_compaction =
      layout >= 0 && layout <= 2 &&
      block_size == 64 &&
      window_size_left >= 0;
  const int partition_window_size_left =
      use_window_partition_compaction ? window_size_left : -1;
  const int max_num_partitions =
      get_paged_attention_max_num_partitions(
          max_seq_len,
          mtp,
          partition_window_size_left,
          block_size,
          PARTITION_SIZE,
          use_window_partition_compaction);
  // partial 结果的临时缓冲区 (见 paged_attention_cached_tmp_out)。原实现为进程内唯一的静态缓冲区,
  // 同一进程在多张卡上调用时跨设备访问出错, 多个流并发调用时 partial 结果相互覆盖。
  // 不能使用缓存时本次单独申请 (缓存分配器, query 所在设备、当前流; 本函数返回后释放, 同一流上的后续调用可复用)。
  const int64_t tmp_out_bytes = static_cast<int64_t>(temp_out_size) + 10000000;
  at::Tensor tmp_out;
  float* tmp_out_ptr = nullptr;
  auto acquire_native_workspace = [&] {
    tmp_out_ptr = paged_attention_cached_tmp_out(query, tmp_out_bytes);
    if (tmp_out_ptr == nullptr) {
      tmp_out = at::empty({tmp_out_bytes}, query.options().dtype(at::kByte));
      tmp_out_ptr = reinterpret_cast<float*>(tmp_out.data_ptr());
    }
    AT_ASSERTM(max_num_partitions*num_seqs*num_heads*headsize*2<temp_out_size,
               "Out of memory ! Please use smaller batchsize!");
  };

  // Enable small BHSD E4M3 groups on gfx938 and gfx92a; the FP8 entry selects the implementation.
  const bool bhsd_fp8_small_group = layout == 1 && use_fp8_compute && is_causal &&
      query.dtype() == torch::kBFloat16 && (headsize == 128 || headsize == 256) &&
      value_cache.size(3) == headsize && real_reuse_times <= 64 &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value();
  const bool bhsd_fp8_eligible = !(layout == 1 && use_fp8_compute) ||
      (is_causal && !alibi_slopes.has_value() && !s_aux_.has_value() &&
       !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
       std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6 &&
       (out.dtype() == torch::kBFloat16 || out.dtype() == torch::kFloat16) &&
       (query.dtype() == out.dtype() || query.dtype() == torch::kFloat8_e4m3fn));
  // gfx938 / gfx92a: BHSD E4M3 的优化实现覆盖全部组大小与 BF16/FP16/E4M3 Q, 不再要求 reusekv > 16
  // (use_fp8_compute 已限定为这两种架构; FP8 入口再按设备选择 05 的 gfx938 实现或 92a 实现)
  const bool bhsd_fp8_all_groups = layout == 1 && use_fp8_compute &&
      (headsize == 64 || headsize == 128 || headsize == 192 || headsize == 256) &&
      value_cache.size(3) == headsize && real_reuse_times <= 64;
  // BHSD 非 FP8 计算 (全 BF16、全 FP16, 或 BF16/FP16 Q + E5M2 KV): 各架构的 BHSD 优化实现覆盖全部组大小,
  // 不再要求 reusekv > 16; 只用于符合重构 kernel 前提的输入 (因果、无 alibi/sink/缩放张量、默认 softmax scale)。
  // 原工程的 reusekv > 16 条件只在对比用的原配置中保留。
  const bool bhsd_b16_all_groups = layout == 1 && !use_fp8_compute && is_causal &&
      (headsize == 64 || headsize == 128 || headsize == 192 || headsize == 256) &&
      value_cache.size(3) == headsize && real_reuse_times <= 64 &&
      (query.dtype() == torch::kBFloat16 || query.dtype() == torch::kFloat16) && out.dtype() == query.dtype() &&
      key_cache.dtype() == value_cache.dtype() &&
      (key_cache.dtype() == query.dtype() || key_cache.dtype() == torch::kFloat8_e5m2) &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6;
  // gfx92a: BSHD 的优化实现 (BF16/FP16 Q + 同类型 16 位 KV、E5M2 KV 或 E4M3FN KV) 覆盖全部组大小, 不再要求 reusekv > 16;
  // 前提与 bhsd_b16_all_groups 相同。gfx938 见下方 l0_bshd_938_all_groups, 其他架构的 BSHD 保持原条件。
  const bool bshd_92a_all_groups = layout == 2 && device_name == "gfx92a" && is_causal &&
      (headsize == 64 || headsize == 128 || headsize == 192 || headsize == 256) &&
      value_cache.size(3) == headsize && real_reuse_times <= 64 &&
      (query.dtype() == torch::kBFloat16 || query.dtype() == torch::kFloat16) && out.dtype() == query.dtype() &&
      key_cache.dtype() == value_cache.dtype() &&
      (key_cache.dtype() == query.dtype() || key_cache.dtype() == torch::kFloat8_e5m2 ||
       key_cache.dtype() == torch::kFloat8_e4m3fn) &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6;
  // gfx92a: layout=0 (K [blocks, heads, block, D], V [blocks, heads, D, block]) 的优化实现同样覆盖全部组大小, 前提与 BSHD 相同
  const bool l0_92a_all_groups = layout == 0 && device_name == "gfx92a" && is_causal &&
      (headsize == 64 || headsize == 128 || headsize == 192 || headsize == 256) &&
      value_cache.size(2) == headsize && value_cache.size(3) == block_size && real_reuse_times <= 64 &&
      (query.dtype() == torch::kBFloat16 || query.dtype() == torch::kFloat16) && out.dtype() == query.dtype() &&
      key_cache.dtype() == value_cache.dtype() &&
      (key_cache.dtype() == query.dtype() || key_cache.dtype() == torch::kFloat8_e5m2 ||
       key_cache.dtype() == torch::kFloat8_e4m3fn) &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6;
  // gfx938: layout=0 与 BSHD 沿用 git 版本的重构实现, 覆盖全部组大小, 不再要求 reusekv > 16。
  // 只排除重构 kernel 不处理的输入 (alibi、sink、q/k/v scale、非因果、非默认 softmax scale), 这些输入的小组仍走 base。
  const bool l0_bshd_938_all_groups = (layout == 0 || layout == 2) && device_name == "gfx938" && is_causal &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6;
  // gfx936: layout=0 与 BSHD 的优化实现覆盖全部组大小, 不再要求 reusekv > 16 (前提与 bhsd_b16_all_groups 相同)。
  const bool l0_bshd_936 = (layout == 0 || layout == 2) && device_name == "gfx936";
  const bool l0_bshd_936_all_groups = l0_bshd_936 && is_causal &&
      (headsize == 64 || headsize == 128 || headsize == 192 || headsize == 256) &&
      (layout == 0 ? (value_cache.size(2) == headsize && value_cache.size(3) == block_size) : value_cache.size(3) == headsize) &&
      real_reuse_times <= 64 &&
      (query.dtype() == torch::kBFloat16 || query.dtype() == torch::kFloat16) && out.dtype() == query.dtype() &&
      key_cache.dtype() == value_cache.dtype() &&
      (key_cache.dtype() == query.dtype() || key_cache.dtype() == torch::kFloat8_e5m2) &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6;
  // The old D256 long exit uses a different workspace stride from short rows.
  // With multiple sequences, use the existing compatibility kernel for this
  // small-group FP8 case; batch=1 retains the original native implementation.
  const bool bhsd_mixed_batch_workspace_risk = device_name == "gfx92a" && layout == 1 &&
      num_seqs > 1 && headsize == 256 && real_reuse_times <= 16 &&
      out.dtype() == torch::kBFloat16 &&
      (key_cache.dtype() == torch::kFloat8_e5m2 || key_cache.dtype() == torch::kFloat8_e4m3fn);
  if(!PA_REFACTORING_DISABLE && !bhsd_mixed_batch_workspace_risk && is_causal && !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      bhsd_fp8_eligible && ((reusekv > 16 && !l0_bshd_936) || bhsd_fp8_small_group || bhsd_fp8_all_groups || bhsd_b16_all_groups || bshd_92a_all_groups || l0_92a_all_groups || l0_bshd_938_all_groups || l0_bshd_936_all_groups) && (headsize == headsize_v) && (mtp > 1) && (-1 == window_size_left) && (-1 == window_size_right) && (64 == block_size)){
    acquire_native_workspace();
    // Also include the validated small BHSD FP8 query groups.
    // Keep the architecture-specific causal, unwindowed fast paths.
    // Window/compatibility inputs use the window specialization below.
    torch::Tensor query_native = query.is_contiguous() ? query : query.contiguous();
    paged_attention_refactoring(out,
                                query_native,
                                key_cache,
                                value_cache,
                                scale,
                                block_tables,
                                seq_lens,
                                alibi_slopes,
                                kv_cache_dtype,
                                q_scale,
                                k_scale,
                                v_scale,
                                max_seq_len,
                                s_aux_,
                                tmp_out_ptr,
                                PA_PRINT_PARAM,
                                use_fp8_compute,
                                layout
                              );
    return;
  }

  const int partition_capacity = provided_seq_len_upper >= 16 ?
      MIN(max_num_blocks_per_seq * block_size, provided_seq_len_upper) : max_num_blocks_per_seq * block_size;
  const Pa28WindowPlan window_plan = pa28_make_window_plan(
      partition_capacity, mtp, window_size_left,
      PARTITION_SIZE, num_seqs, query.size(2), headsize_v);

  // Small gfx92a decode batches benefit from the existing dense vector loads.
  // Keep windows, E4M3, large batches and all other architectures on their paths.
  const auto aligned16 = [](const torch::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
  };
  const bool decode92a_dense = device_name == "gfx92a" && !PA_REFACTORING_DISABLE &&
      num_seqs > 0 && num_seqs <= 2 && mtp == 1 &&
      partition_capacity >= 8192 &&
      window_size_left == -1 && window_size_right == -1 &&
      layout >= 0 && layout <= 2 && block_size == 64 &&
      headsize == headsize_v && headsize > 0 && headsize % 64 == 0 && headsize <= 256 &&
      real_reuse_times > 0 && real_reuse_times <= 8 && query.size(2) % num_kv_heads == 0 &&
      out.scalar_type() == query.scalar_type() &&
      (query.scalar_type() == torch::kBFloat16 || query.scalar_type() == torch::kFloat16) &&
      key_cache.scalar_type() == value_cache.scalar_type() &&
      (key_cache.scalar_type() == query.scalar_type() || key_cache.scalar_type() == torch::kFloat8_e5m2) &&
      key_cache.is_contiguous() && value_cache.is_contiguous() && out.is_contiguous() &&
      aligned16(query) && aligned16(key_cache) && aligned16(value_cache) && aligned16(out) &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6 &&
      (window_plan.partition_size == 256 || window_plan.partition_size == 512);
  if (decode92a_dense) {
    torch::Tensor query_dense = query.is_contiguous() ? query : query.contiguous();
    if (window_plan.workspace_bytes) {
      tmp_out_ptr = paged_attention_cached_tmp_out(query, window_plan.workspace_bytes);
      if (tmp_out_ptr == nullptr) {
        tmp_out = at::empty({window_plan.workspace_bytes}, query.options().dtype(at::kByte));
        tmp_out_ptr = reinterpret_cast<float*>(tmp_out.data_ptr());
      }
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int parts = window_plan.num_partitions;
    const int part = window_plan.partition_size;
    const dim3 grid(num_kv_heads, num_seqs, parts);
    const int shared = part * 2 * real_reuse_times + (real_reuse_times * 6 + 4) * sizeof(float);
    if (PA_PRINT_PARAM) printf("PA92A_DENSE_DECODE layout=%d parts=%d partition_size=%d blocks=%lld\n",
        layout, parts, part, static_cast<long long>(grid.x) * grid.y * grid.z);
    HEADSIZE_SWITCH(headsize, [&] {
      Output_Type_SWITCH(out.scalar_type(), [&] {
        Cache_Type_SWITCH(scalar_t, key_cache.scalar_type(), [&] {
          auto launch = [&](auto rtag) {
            constexpr int R = decltype(rtag)::value;
            if (layout == 0) {
              pa936_dense_layout0<scalar_t,cache_t,false,HEAD_SIZE,64,256,R>
                <<<grid,256,shared,stream>>>(
                static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                static_cast<const scalar_t*>(query_dense.data_ptr()),
                static_cast<const cache_t*>(key_cache.data_ptr()), static_cast<const cache_t*>(value_cache.data_ptr()),
                num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), max_num_blocks_per_seq,
                nullptr, query_dense.stride(0), key_cache.stride(0), parts, part, nullptr, mtp, false,
                -1, effective_right, -1);
            } else if (layout == 1) {
              pa936_dense_layout1<scalar_t,cache_t,false,HEAD_SIZE,64,256,R>
                <<<grid,256,shared,stream>>>(
                static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                static_cast<const scalar_t*>(query_dense.data_ptr()),
                static_cast<const cache_t*>(key_cache.data_ptr()), static_cast<const cache_t*>(value_cache.data_ptr()),
                num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), max_num_blocks_per_seq,
                nullptr, query_dense.stride(0), key_cache.stride(0), parts, part, nullptr, mtp, false,
                -1, effective_right, -1);
            } else {
              pa936_dense_layout2<scalar_t,cache_t,false,HEAD_SIZE,HEAD_SIZE,64,256,R>
                <<<grid,256,shared,stream>>>(
                static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                static_cast<const scalar_t*>(query_dense.data_ptr()),
                static_cast<const cache_t*>(key_cache.data_ptr()), static_cast<const cache_t*>(value_cache.data_ptr()),
                num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), max_num_blocks_per_seq,
                nullptr, query_dense.stride(0), key_cache.stride(0), value_cache.stride(0), parts, part,
                nullptr, mtp, false, -1, effective_right, -1);
            }
          };
          if (real_reuse_times <= 4) launch(std::integral_constant<int,4>{});
          else launch(std::integral_constant<int,8>{});
          if (parts > 1) {
            paged_attention_combine<scalar_t,HEAD_SIZE,64>
              <<<dim3(num_heads,num_seqs),64,8*parts,stream>>>(
              static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
              seq_lens.data_ptr<int>(), parts, num_heads, part, -1, mtp, 64);
          }
        });
      });
    });
    TORCH_CHECK(hipGetLastError() == hipSuccess, "PA92A: dense decode kernel launch failed");
    return;
  }

  // Preserve the proven contiguous-token V loads for ordinary gfx936 layout0
  // decode/window inputs. Other layouts and compatibility inputs retain PA30.
  const bool decode936_compatible = device_name == "gfx936" && block_size == 64 &&
      headsize == headsize_v && headsize % 64 == 0 && headsize <= 256 &&
      real_reuse_times <= 64 && query.size(2) % num_kv_heads == 0 &&
      out.scalar_type() == query.scalar_type() &&
      (query.scalar_type() == torch::kBFloat16 || query.scalar_type() == torch::kFloat16) &&
      key_cache.scalar_type() == value_cache.scalar_type() &&
      (key_cache.scalar_type() == query.scalar_type() || key_cache.scalar_type() == torch::kFloat8_e5m2) &&
      key_cache.is_contiguous() && value_cache.is_contiguous() && out.is_contiguous() &&
      !alibi_slopes.has_value() && !s_aux_.has_value() &&
      !q_scale.has_value() && !k_scale.has_value() && !v_scale.has_value() &&
      std::abs(scale - 1.0 / std::sqrt(static_cast<double>(headsize))) <= 1e-6 &&
      (window_plan.partition_size == 256 ||
       (window_plan.partition_size == 512 && real_reuse_times <= 15));
  const bool decode936=decode936_compatible && layout==0;
  const bool decode_legacy936=decode936_compatible && (layout==1 || layout==2) &&
      mtp==1 && real_reuse_times<=8 && PA936_LEGACY_LAYOUTS!=0;
  if (window_plan.workspace_bytes) {
    if (decode936 || decode_legacy936) tmp_out_ptr = paged_attention_cached_tmp_out(query, window_plan.workspace_bytes);
    if (tmp_out_ptr == nullptr) {
      tmp_out = at::empty({window_plan.workspace_bytes}, query.options().dtype(at::kByte));
      tmp_out_ptr = reinterpret_cast<float*>(tmp_out.data_ptr());
    }
  }
  if (decode_legacy936) {
    paged_attention_decode_legacy_gfx936(out,query,key_cache,value_cache,block_tables,seq_lens,
        tmp_out_ptr,layout,window_size_left,effective_right,window_plan,
        PA_GRAPH_COMPACT,PA_GRAPH_WORKERS,PA936_SMALL_SCHEDULER!=0);
    return;
  }
  if (decode936) {
    torch::Tensor query_base = query.is_contiguous() ? query : query.contiguous();
    const auto stream = at::cuda::getCurrentCUDAStream();
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    TORCH_CHECK(hipStreamIsCapturing(stream, &status) == hipSuccess, "PA936: cannot query capture status");
    const auto* props = at::cuda::getCurrentDeviceProperties();
    // Small decode batches commonly expose 256 active tiles: 240 workers
    // leave a second iteration on the tail. Larger batches favor 240 workers.
    // An explicit user policy always takes precedence over this default.
    const int workers = PA_GRAPH_WORKERS == -1 && num_seqs <= 2 ? 256 :
        pa28_graph_workers(PA_GRAPH_WORKERS, "gfx936", props->multiProcessorCount);
    const int64_t tasks = static_cast<int64_t>(num_seqs) * num_kv_heads * window_plan.num_partitions;
    const bool compact = status != hipStreamCaptureStatusNone && PA_GRAPH_COMPACT != 0 &&
        (PA_GRAPH_COMPACT == 2 || (tasks >= 8192 && tasks > static_cast<int64_t>(workers) * 4 &&
                                 !(num_seqs >= 8 && tasks <= 8192)));
    TORCH_CHECK(PA_GRAPH_COMPACT >= 0 && PA_GRAPH_COMPACT <= 2 &&
                PA_GRAPH_WORKERS >= -1 && PA_GRAPH_WORKERS <= 65535, "PA936: invalid Graph policy");
    HEADSIZE_SWITCH(headsize, [&] {
      Output_Type_SWITCH(out.scalar_type(), [&] {
        Cache_Type_SWITCH(scalar_t, key_cache.scalar_type(), [&] {
          REUSEKV_SWITCH(reusekv, [&] {
            constexpr int threads = 256;
            const int shared_bytes = window_plan.partition_size * sizeof(scalar_t) * real_reuse_times +
                (real_reuse_times * 4 + 4 + real_reuse_times * 2) * sizeof(float);
            const dim3 grid = compact ? dim3(static_cast<unsigned>(std::min<int64_t>(workers, tasks)), 1, 1) :
                dim3(num_kv_heads, num_seqs, window_plan.num_partitions);
            if (PA_PRINT_PARAM) printf("PA936_LAYOUT0_VECTOR compact=%d parts=%d partition_size=%d launch_blocks=%lld workers=%d\n",
                compact, window_plan.num_partitions, window_plan.partition_size,
                static_cast<long long>(grid.x) * grid.y * grid.z, workers);
            auto launch_decode=[&](auto policy_tag) {
              constexpr int POLICY=decltype(policy_tag)::value;
              if constexpr(POLICY==0) {
                pa936_dense_layout0<scalar_t,cache_t,false,HEAD_SIZE,64,threads,REUSE_KV_TIMES>
                  <<<grid,threads,shared_bytes,stream>>>(
                  static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                  static_cast<const scalar_t*>(query_base.data_ptr()),
                  static_cast<const cache_t*>(key_cache.data_ptr()), static_cast<const cache_t*>(value_cache.data_ptr()),
                  num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), max_num_blocks_per_seq,
                  nullptr, query_base.stride(0), key_cache.stride(0), window_plan.num_partitions,
                  window_plan.partition_size, nullptr, mtp, false, window_size_left, effective_right,
                  window_size_left, false, nullptr);
              } else {
              paged_attention_kernel<scalar_t, cache_t, false, HEAD_SIZE, 64, threads, REUSE_KV_TIMES, POLICY!=0, POLICY==2>
                <<<grid, threads, shared_bytes, stream>>>(
                  static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                  static_cast<const scalar_t*>(query_base.data_ptr()),
                  static_cast<const cache_t*>(key_cache.data_ptr()), static_cast<const cache_t*>(value_cache.data_ptr()),
                  num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), max_num_blocks_per_seq,
                  nullptr, query_base.stride(0), key_cache.stride(0), window_plan.num_partitions,
                  window_plan.partition_size, nullptr, mtp, false, window_size_left, effective_right,
                  window_size_left, false, nullptr, num_seqs);
              }
            };
            if (compact) {
              if constexpr(REUSE_KV_TIMES<=8) {
                if (num_seqs<=2 && PA936_SMALL_SCHEDULER) launch_decode(std::integral_constant<int,2>{});
                else launch_decode(std::integral_constant<int,1>{});
              } else launch_decode(std::integral_constant<int,1>{});
            } else launch_decode(std::integral_constant<int,0>{});
            if (window_plan.num_partitions > 1) {
              paged_attention_combine<scalar_t, HEAD_SIZE, 64>
                <<<dim3(num_heads, num_seqs), 64, 8 * window_plan.num_partitions, stream>>>(
                  static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(tmp_out_ptr),
                  seq_lens.data_ptr<int>(), window_plan.num_partitions, num_heads,
                  window_plan.partition_size, window_size_left, mtp, 64);
            }
          });
        });
      });
    });
    TORCH_CHECK(hipGetLastError() == hipSuccess, "PA936: decode kernel launch failed");
    return;
  }
  paged_attention_window_opt(out, query, key_cache, value_cache, block_tables, seq_lens,
      alibi_slopes, s_aux_, tmp_out_ptr, layout, window_size_left, effective_right, window_plan);
}

/*
 * Partial-output paged attention (bf16/fp16 KV, legacy layout 0, block 64).
 *
 * Every (sequence, kv head, partition) block writes its normalized partial
 * output plus (max, exp_sum) in log2 units into the caller-owned out_tmp
 * (same layout as the internal tmp buffer: exp_sums at 0, max_logits at
 * max_tmp_offset, outputs at out_tmp_offset, all indexed by
 * [seq, head, partition]); the final output is never written and no combine
 * runs. The caller merges partials across partitions / sequences (e.g. for
 * prefix-shared decode, where several requests' queries share one KV chunk)
 * and applies sinks itself.
 *
 * query: [num_seqs, mtp, num_heads, head_size], or with q_rows
 *        [rows, num_heads, head_size] gathered per slot by
 *        q_rows [num_seqs, mtp] int32 (-1 = padding).
 * seq_lens [num_seqs] (<= block_tables.size(1) * 64).
 * Uses the default softmax scale 1/sqrt(head_size); fold other scales into q.
 */
extern "C"
void paged_attention_partial(
    torch::Tensor& out_tmp,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int mtp,
    int partition_size,
    int max_num_partitions,
    const c10::optional<torch::Tensor>& q_rows_)
{
  CHECK_DEVICE(out_tmp); CHECK_DEVICE(query); CHECK_DEVICE(key_cache); CHECK_DEVICE(value_cache);
  CHECK_CONTIGUOUS(out_tmp); CHECK_CONTIGUOUS(block_tables); CHECK_CONTIGUOUS(seq_lens);
  TORCH_CHECK(query.dtype() == torch::kBFloat16 || query.dtype() == torch::kFloat16, "query must be fp16/bf16");
  TORCH_CHECK(key_cache.dtype() == query.dtype() && value_cache.dtype() == query.dtype(),
              "paged_attention_partial: KV cache must have the query dtype");
  TORCH_CHECK(out_tmp.dtype() == query.dtype(), "out_tmp must have the query dtype");
  TORCH_CHECK(block_tables.dtype() == torch::kInt && seq_lens.dtype() == torch::kInt);
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(2) == 64, "layout 0 with block size 64 only");
  TORCH_CHECK(partition_size == 256 || partition_size == 512, "partition_size must be 256 or 512");
  TORCH_CHECK(mtp >= 1 && max_num_partitions >= 1);
  const int* q_rows = nullptr;
  if (q_rows_.has_value()) {
    const auto& qr = q_rows_.value();
    CHECK_DEVICE(qr); CHECK_CONTIGUOUS(qr);
    TORCH_CHECK(qr.dtype() == torch::kInt && qr.numel() == block_tables.size(0) * mtp, "q_rows must be int32 [num_seqs, mtp]");
    TORCH_CHECK(query.dim() == 3, "with q_rows, query is [rows, num_heads, head_size]");
    q_rows = qr.data_ptr<int>();
  } else {
    TORCH_CHECK(query.dim() == 4 && query.size(1) == mtp, "query must be [num_seqs, mtp, num_heads, head_size]");
    TORCH_CHECK(query.size(0) == block_tables.size(0));
  }
  const int num_seqs = block_tables.size(0);
  const int headsize = query.size(-1);
  const int num_heads = query.size(-2) * mtp;
  const int num_kv_heads = key_cache.size(1);
  TORCH_CHECK(value_cache.size(-2) == headsize && value_cache.size(-1) == 64, "value_cache must be [blocks, kv_heads, head_size, 64]");
  TORCH_CHECK(num_heads % num_kv_heads == 0 && num_heads <= num_kv_heads * 64, "qheads*mtp/kvheads must be <= 64");
  AT_ASSERTM(headsize % 64 == 0 && headsize <= 256, "head size must be 64, 128, 192 or 256");
  const int real_reuse_times = num_heads / num_kv_heads;
  TORCH_CHECK(partition_size == 256 || real_reuse_times <= 15, "partition_size 512 needs qheads*mtp/kvheads <= 15");
  TORCH_CHECK((int64_t)num_seqs * num_heads * max_num_partitions <= max_tmp_offset / 2, "too many partials for the stats area");
  TORCH_CHECK(out_tmp.numel() >= (int64_t)out_tmp_offset + (int64_t)num_seqs * num_heads * max_num_partitions * headsize,
              "out_tmp too small");
  const int reusekv = get_reusekv(num_heads, num_kv_heads);
  const int q_stride = query.stride(0);
  const int kv_block_stride = key_cache.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_kv_heads, num_seqs, max_num_partitions);
  HEADSIZE_SWITCH(headsize,[&]{
    Output_Type_SWITCH(query.dtype(),[&]{
      REUSEKV_SWITCH(reusekv,[&] {
        using cache_t = scalar_t;
        constexpr int BLOCK_SIZE = 64;
        constexpr static int NUM_THREADS = 256;
        constexpr static int NUM_WARPS = NUM_THREADS / WARP_SIZE;
        int other_use = (real_reuse_times*NUM_WARPS+NUM_WARPS+ real_reuse_times*2)*sizeof(float);
        int shared_mem_size = partition_size*sizeof(scalar_t)*real_reuse_times+other_use;
        pa936_dense_layout0<scalar_t,cache_t,false,HEAD_SIZE,BLOCK_SIZE,NUM_THREADS,REUSE_KV_TIMES><<<grid,NUM_THREADS,shared_mem_size,stream>>>(
          (scalar_t*)out_tmp.data_ptr(),(scalar_t*)out_tmp.data_ptr(),(const scalar_t*)query.data_ptr(),
          (const cache_t*)key_cache.data_ptr(),(const cache_t*)value_cache.data_ptr(),
          num_heads, num_kv_heads, block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), (int)block_tables.size(1),
          nullptr, q_stride, kv_block_stride, max_num_partitions, partition_size, nullptr, mtp, false,
          -1, -1, -1, /*force_partial=*/true, q_rows);
      });
    });
  });
}
