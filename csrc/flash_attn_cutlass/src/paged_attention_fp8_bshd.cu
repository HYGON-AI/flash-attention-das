#include "paged_attention.h"
  
/*
 * [窗口优化-FP8 BSHD 通用 kernel]
 * BSHD FP8 kernel 的位置 mask 与 FP8 解码路径相互独立，
 * 必须同时保留 left/right/causal 语义。
 */
template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_fp8_bshd(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const q_type* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const uint8_t* __restrict__ k_cache,  // [num_blocks, block_size, num_heads, head_size]
    const uint8_t* __restrict__ v_cache,  // [num_blocks, block_size, num_heads, head_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  // ★ Attention Sinks: [num_heads] scalar_t ★
#if (defined(__gfx938__)||defined(__gfx92a__) )
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride=HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-FP8 BSHD]
   * FP8 BSHD 与非 FP8 共用 active partition 规则。partition_idx 是压缩后的
   * local 编号，helper 负责将其映射到真实 block_table page。
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
  const int num_partitions = active_range.num_active_partitions;
  if(num_partitions<=partition_idx)return ;
  constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
  constexpr bool q_is_fp8 = std::is_same<q_type, uint8_t>::value;
  constexpr float scale = (HEAD_SIZE==64?0.125f:(HEAD_SIZE==128? 0.0883883476f:(HEAD_SIZE==192?0.0721687836f:0.0625f)))*1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  using ldstype = _Float16;
  constexpr float qkgemm_scale = scale;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = blockIdx.x;
  const int head_idx=num_queries_per_kv/mtp * kv_head_idx;
  constexpr int reuse_group=(REUSE_KV_TIMES-1)/4+1;
  constexpr int Mloop=(REUSE_KV_TIMES-1)/16+1;
  extern __shared__ char shared_mem[];
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
  const q_type* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  
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
  intx4 q_vec[Mloop][HEAD_SIZE/64];
  q_type* s_q = reinterpret_cast<q_type*>(shared_mem);
  {
    int head_offset = HEAD_SIZE*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE;i+=NUM_THREADS*8){
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;

      if constexpr (q_is_fp8){
        *reinterpret_cast<intx2*>(s_q+i)=*reinterpret_cast<const intx2*>(q_ptr+qoffset);
      }
      else{
        *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
      }
    }
  }
  __syncthreads();
  for(int m=0;m<Mloop;m++){
    for(int i=0;i<HEAD_SIZE/64;i++){
      int head_idx_=rowid+16*m;
      if(head_idx_<num_queries_per_kv) {
        if constexpr(q_is_fp8){
          q_vec[m][i]=*reinterpret_cast<const intx4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
        }
        else{
          auto q_temp = *reinterpret_cast<const half4x4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
          scalar_t *q_temp_ptr=(scalar_t*)&q_temp;
          q_vec[m][i][0]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[0]),to_float(q_temp_ptr[1]),to_float(q_temp_ptr[2]),to_float(q_temp_ptr[3]));
          q_vec[m][i][1]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[4]),to_float(q_temp_ptr[5]),to_float(q_temp_ptr[6]),to_float(q_temp_ptr[7]));
          q_vec[m][i][2]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[8]),to_float(q_temp_ptr[9]),to_float(q_temp_ptr[10]),to_float(q_temp_ptr[11]));
          q_vec[m][i][3]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[12]),to_float(q_temp_ptr[13]),to_float(q_temp_ptr[14]),to_float(q_temp_ptr[15]));
        }
      }
      else q_vec[m][i]={0,0,0,0};
    }
  }
  __syncthreads();

  /*
   * [窗口优化-FP8 BSHD page 映射]
   * 使用公共 helper 的 block 范围；当前 partition window 为 -1，
   * 因而与原始完整序列映射保持一致。
   */
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //compute q*k
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*num_kv_heads*HEAD_SIZE;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        intx4 k_vec[HEAD_SIZE/64];
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          k_vec[i]=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*num_kv_heads*HEAD_SIZE+rows*16);
        }
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          intx2 *k_vec_2 = (intx2*)(k_vec+i);
          for(int m=0;m<Mloop;m++){
            intx2 *q_vec_2 = (intx2*)(&q_vec[m][i]);
            builtin_amdgcn_mmac_fp8<is_e4m3>(k_vec_2[0],q_vec_2[0],qk_vec[m]);
            builtin_amdgcn_mmac_fp8<is_e4m3>(k_vec_2[1],q_vec_2[1],qk_vec[m]);
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
             * [窗口优化-FP8 BSHD mask]
             * 根据 query 的绝对位置计算窗口边界，并让 masked score 跳过 qk_max。
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
               * [窗口优化-FP8 BSHD 因果语义]
               * 旧的 mtp>1 隐式 causal 已删除；causal 由 dispatcher 转换为
               * effective_right=0。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_float(temp));
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
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
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
          if(num_partitions>1&&half_lane==0){
            /*
             * [FP8 correctness 修复-256 partition]
             * combine 使用 max_logits[i] 作为 softmax 的正向最大值，
             * 必须与 qk_max_tmp 保持同号。原代码写入负值，会在多 partition
             * 的 FP8 layout=2 普通 kernel 中错误地重标定 exp_sum 和 partial output。
             */
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
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
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
          if(num_partitions>1&&lane==0){
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
  constexpr int vecsize=BLOCK_SIZE/32;//2
  using int_vec = int2vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    intx2 logits_vec[Mloop][vecsize];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m][i]={0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        for(int k=0;k<vecsize;k++){
          auto l_temp = *reinterpret_cast<half8_t*>(logits + real_row * PARTITION_SIZE+token_idx - start_token_idx + k*8);
          scalar_t *l_temp_ptr=(scalar_t*)&l_temp;
          logits_vec[m][k][0]=to_f8_from_f32<is_e4m3>(to_float(l_temp_ptr[0]),to_float(l_temp_ptr[1]),to_float(l_temp_ptr[2]),to_float(l_temp_ptr[3]));
          logits_vec[m][k][1]=to_f8_from_f32<is_e4m3>(to_float(l_temp_ptr[4]),to_float(l_temp_ptr[5]),to_float(l_temp_ptr[6]),to_float(l_temp_ptr[7]));
        }
      }
    }
    const uint8_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*HEAD_SIZE*num_kv_heads+rowid;
        int_vec v_vec;
        uint8_t * p = (uint8_t *)&v_vec;
        for(int ii = 0 ; ii<8*vecsize ; ii++){
          p[ii]=v_ptr[offset+ii*HEAD_SIZE*num_kv_heads];
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*HEAD_SIZE*num_kv_heads+rowid;
        int_vec v_vec;
        uint8_t * p = (uint8_t *)&v_vec;
        for(int ii = 0 ; ii<8*vecsize ; ii++){
          p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE*num_kv_heads] : 0;
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
  }
  {
    scalar_t* out_ptr_base;
    int out_offset;
    if(num_partitions>1){
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
    if (num_partitions>1&&thread_idx < num_queries_per_kv){
      int out_head = thread_idx/head_offset*num_kv_heads*head_offset + thread_idx%head_offset;
      int offset = seq_idx * num_heads * max_num_partitions + (head_idx+out_head) * max_num_partitions + partition_idx;
      float * exp_sums=reinterpret_cast<float*>(out_tmp);
      float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
      *(exp_sums+offset)=expsum_out[thread_idx];
      *(max_logits+offset)=max_out[thread_idx];
    }
  }
#endif
}

/*
 * [窗口优化-FP8 BSHD fast kernel]
 * fast kernel 也必须使用统一的窗口参数，避免专用 FP8 路径绕过 mask。
 */
template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int PARTITION_SIZE>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void paged_attention_kernel_fp8_bshd_fast(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const q_type* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const uint8_t* __restrict__ k_cache,  // [num_blocks, block_size, num_heads, head_size]
    const uint8_t* __restrict__ v_cache,  // [num_blocks, block_size, num_heads, head_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  // ★ Attention Sinks: [num_heads] scalar_t ★
#if (defined(__gfx938__)||defined(__gfx92a__) )
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride=HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-FP8 BSHD fast]
   * fast kernel 复用公共范围计算；partition_window_size_left=-1 时
   * 退化为完整 partition，关闭压缩时保持原有临时结果布局。
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
  const int num_partitions = active_range.num_active_partitions;
  if(num_partitions<=partition_idx)return ;
  constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
  constexpr bool q_is_fp8 = std::is_same<q_type, uint8_t>::value;
  constexpr float scale = (HEAD_SIZE==64?0.125f:(HEAD_SIZE==128? 0.0883883476f:(HEAD_SIZE==192?0.0721687836f:0.0625f)))*1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  constexpr float lds_scale = 8.0f;
  constexpr float qkgemm_scale = scale/lds_scale;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = blockIdx.x;
  const int head_idx=num_queries_per_kv/mtp * kv_head_idx;

  constexpr int reuse_group=(REUSE_KV_TIMES-1)/4+1;
  constexpr int Mloop=(REUSE_KV_TIMES-1)/16+1;
  extern __shared__ char shared_mem[];
  uint8_t* logits = reinterpret_cast<uint8_t*>(shared_mem);
  float* s_max = reinterpret_cast<float*>(shared_mem + num_queries_per_kv*PARTITION_SIZE);
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
  const q_type* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  
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
  intx4 q_vec[Mloop][HEAD_SIZE/64];
  q_type* s_q = reinterpret_cast<q_type*>(shared_mem);
  {
    int head_offset = HEAD_SIZE*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE;i+=NUM_THREADS*8){
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;

      if constexpr (q_is_fp8){
        *reinterpret_cast<intx2*>(s_q+i)=*reinterpret_cast<const intx2*>(q_ptr+qoffset);
      }
      else{
        *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
      }
    }
  }
  __syncthreads();
  for(int m=0;m<Mloop;m++){
    for(int i=0;i<HEAD_SIZE/64;i++){
      int head_idx_=rowid+16*m;
      if(head_idx_<num_queries_per_kv) {
        if constexpr(q_is_fp8){
          q_vec[m][i]=*reinterpret_cast<const intx4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
        }
        else{
          auto q_temp = *reinterpret_cast<const half4x4*>(s_q+head_idx_*HEAD_SIZE+(i*4+rows)*16);
          scalar_t *q_temp_ptr=(scalar_t*)&q_temp;
          q_vec[m][i][0]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[0]),to_float(q_temp_ptr[1]),to_float(q_temp_ptr[2]),to_float(q_temp_ptr[3]));
          q_vec[m][i][1]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[4]),to_float(q_temp_ptr[5]),to_float(q_temp_ptr[6]),to_float(q_temp_ptr[7]));
          q_vec[m][i][2]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[8]),to_float(q_temp_ptr[9]),to_float(q_temp_ptr[10]),to_float(q_temp_ptr[11]));
          q_vec[m][i][3]=to_f8_from_f32<is_e4m3>(to_float(q_temp_ptr[12]),to_float(q_temp_ptr[13]),to_float(q_temp_ptr[14]),to_float(q_temp_ptr[15]));
        }
      }
      else q_vec[m][i]={0,0,0,0};
    }
  }
  __syncthreads();

  /*
   * [窗口优化-FP8 BSHD fast page 映射]
   * 普通和 fast FP8 kernel 使用同一套 block 范围公式，当前 -1 表示不压缩。
   */
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //compute q*k
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*num_kv_heads*HEAD_SIZE;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        intx4 k_vec[HEAD_SIZE/64];
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          k_vec[i]=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*num_kv_heads*HEAD_SIZE+rows*16);
        }
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          intx2 *k_vec_2 = (intx2*)(k_vec+i);
          for(int m=0;m<Mloop;m++){
            intx2 *q_vec_2 = (intx2*)(&q_vec[m][i]);
            builtin_amdgcn_mmac_fp8<is_e4m3>(k_vec_2[0],q_vec_2[0],qk_vec[m]);
            builtin_amdgcn_mmac_fp8<is_e4m3>(k_vec_2[1],q_vec_2[1],qk_vec[m]);
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
            /*
             * [窗口优化-FP8 BSHD fast mask]
             * masked score 使用 FP8 的负值编码，同时禁止其参与 qk_max。
             */
            bool window_masked = false;
            if (has_abili){
              float alibi=alibi_slope[i] * (token_idx - seq_len + 1);
              qk_vec[m][ii] += alibi;
            }
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
              if(token_idx<seq_len_pad) logits[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx]=0xfc;
              else logits[PARTITION_SIZE*reuse_kv_idx+token_idx - start_token_idx]=0;
            }
            else{
              uint8_t temp;
              /*
               * [窗口优化-FP8 BSHD fast 因果语义]
               * causal 已由 host 端转换为 effective_right=0，
               * 这里不再恢复旧的无条件 MTP causal 判断。
               */
              temp = window_masked ? static_cast<uint8_t>(0xfc) : to_f8_from_f32<false>(qk_vec[m][ii]);
              logits[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_fp32_from_fp8<false>(temp)*lds_scale);
              }
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

  if constexpr(PARTITION_SIZE==256){
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
        using f8x8_t = __attribute__( (__vector_size__(8 * sizeof(uint8_t)) )) uint8_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(half_lane<seq_len_pad){
          f8x8_t logit8 = *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8);
          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            float4_t temp = to_fp32_from_fp8<false>(logit8x4[ii]);
            for(int kk=0;kk<4;kk++){
              logit32[ii*4+kk] = __builtin_amdgcn_exp2f(temp[kk]*lds_scale-qk_max_tmp);
              exp_sum+=logit32[ii*4+kk];
            }
          }
        }
        for (int mask = 16; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(half_lane<seq_len_pad){
          f8x8_t logit8;
          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            logit8x4[ii]=to_f8_from_f32<is_e4m3>(logit32[ii*4]*inv_sum,logit32[ii*4+1]*inv_sum,logit32[ii*4+2]*inv_sum,logit32[ii*4+3]*inv_sum);
          }
          *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8)=logit8;
          if(num_partitions>1&&half_lane==0){
            max_out[real_line] = qk_max_tmp;
            expsum_out[real_line] = exp_sum;
          }
        }
      }
    }
  }
  else if constexpr(PARTITION_SIZE==512){
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
        using f8x8_t = __attribute__( (__vector_size__(8 * sizeof(uint8_t)) )) uint8_t;
        using f32x8_t = __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
        float sink_contrib = 0.f;
        if (s_aux_ptr != nullptr && partition_idx == 0) {
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  // Convert scalar_t (fp16/bf16) to float
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32; 
        if(lane<seq_len_pad){
          f8x8_t logit8 = *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8);

          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            float4_t temp = to_fp32_from_fp8<false>(logit8x4[ii]);
            for(int kk=0;kk<4;kk++){
              logit32[ii*4+kk] = __builtin_amdgcn_exp2f(temp[kk]*lds_scale-qk_max_tmp);
              exp_sum+=logit32[ii*4+kk];
            }
          }
        }
        for (int mask = 32; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        // printf("tid=%d,exp_sum=%f\n",thread_idx,exp_sum);
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(lane<seq_len_pad){
          f8x8_t logit8;
          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            logit8x4[ii]=to_f8_from_f32<is_e4m3>(logit32[ii*4]*inv_sum,logit32[ii*4+1]*inv_sum,logit32[ii*4+2]*inv_sum,logit32[ii*4+3]*inv_sum);
          }
          *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8)=logit8;
          if(num_partitions>1&&lane==0){
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
  constexpr int vecsize=BLOCK_SIZE/32;//2
  using int_vec = int2vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    intx2 logits_vec[Mloop][vecsize];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m][i]={0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        for(int k=0;k<vecsize;k++){
          logits_vec[m][k] = *reinterpret_cast<intx2*>(logits + real_row * PARTITION_SIZE+token_idx - start_token_idx + k*8);
        }
      }
    }
    const uint8_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*HEAD_SIZE*num_kv_heads+rowid;
        int_vec v_vec;
        uint8_t * p = (uint8_t *)&v_vec;
        for(int ii = 0 ; ii<8*vecsize ; ii++){
          p[ii]=v_ptr[offset+ii*HEAD_SIZE*num_kv_heads];
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*HEAD_SIZE*num_kv_heads+rowid;
        int_vec v_vec;
        uint8_t * p = (uint8_t *)&v_vec;
        for(int ii = 0 ; ii<8*vecsize ; ii++){
          p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE*num_kv_heads] : 0;
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
          }
        }
      } 
    }
  }
  {
    scalar_t* out_ptr_base;
    int out_offset;
    if(num_partitions>1){
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
        }
      }
    }
    if (num_partitions>1&&thread_idx < num_queries_per_kv){
      int out_head = thread_idx/head_offset*num_kv_heads*head_offset + thread_idx%head_offset;
      int offset = seq_idx * num_heads * max_num_partitions + (head_idx+out_head) * max_num_partitions + partition_idx;
      float * exp_sums=reinterpret_cast<float*>(out_tmp);
      float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
      *(exp_sums+offset)=expsum_out[thread_idx];
      *(max_logits+offset)=max_out[thread_idx];
    }
  }
#endif
}


/*
 * [窗口优化-FP8 BSHD launcher]
 * launcher 接收并转发 left/effective right，覆盖普通和 fast 两条 FP8 路径。
 */
void paged_attention_fp8_bshd(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache,// [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    float *tmp_out_ptr,
    int PARTITION_SIZE,
    int window_size_left,
    int effective_right,
    int partition_window_size_left)  // ★ Attention Sinks ★
{
  int max_num_blocks_per_seq = block_tables.size(1);
  int num_seqs = query.size(0);
  int mtp = query.size(1);
  int block_size=key_cache.size(1);
  int num_heads = query.size(2)*mtp;
  int num_kv_heads = key_cache.size(2);
  int head_size = query.size(3);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int num_blocks=key_cache.size(0);
  const float* alibi_slopes_ptr =alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr()):nullptr;
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  auto* out_ptr =  out.data_ptr();
  const float* q_scale_ptr = q_scale? reinterpret_cast<const float*>(q_scale.value().data_ptr()):nullptr;
  const float* k_scale_ptr = k_scale? reinterpret_cast<const float*>(k_scale.value().data_ptr()):nullptr;
  const float* v_scale_ptr = v_scale? reinterpret_cast<const float*>(v_scale.value().data_ptr()):nullptr;

  // Attention Sinks: validate and set s_aux_ptr
  const void* s_aux_ptr = nullptr;
  if (s_aux_.has_value()) {
    auto s_aux = s_aux_.value();
    // ★ s_aux must match Q/K/V dtype (Element type) for mixed precision
    TORCH_CHECK(s_aux.dtype() == query.dtype(),
                "s_aux must have the same dtype as query. Got s_aux dtype: ", s_aux.dtype(),
                ", query dtype: ", query.dtype());
    TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                "s_aux must have dtype float16 or bfloat16 (to match query). Got: ", s_aux.dtype());
    TORCH_CHECK(num_heads <= 64,
                "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
    CHECK_DEVICE(s_aux);
    CHECK_SHAPE(s_aux, num_heads);
    CHECK_CONTIGUOUS(s_aux);
    s_aux_ptr = s_aux.data_ptr();
  }
  auto* query_ptr = query.data_ptr();
  auto* key_cache_ptr = key_cache.data_ptr();
  auto* value_cache_ptr = value_cache.data_ptr();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 reduce_grid(num_heads, num_seqs);
 
  dim3 grid;
  grid.x = num_kv_heads;
  grid.y = num_seqs;
  int reusekv=get_reusekv(num_heads,num_kv_heads);
  int headsize=query.size(3);
  AT_ASSERTM(headsize%64==0 && headsize<=256, "Page Attention head size must be 64, 128, 192 or 256");
  AT_ASSERTM(num_heads<=num_kv_heads*64, "Page Attention qheads*mtp/kvheads must be smaller than 48");
  HEADSIZE_SWITCH(headsize,[&]{
    Output_Type_SWITCH(out.dtype(),[&]{
      Input_Type_SWITCH(scalar_t,query.dtype(),key_cache.dtype(),[&] {
        REUSEKV_SWITCH(reusekv,[&] {
          BOOL_SWITCH(block_size==64,is_block64,[&]{
            // constexpr int BLOCK_SIZE = (is_block64?64:128);
              constexpr int BLOCK_SIZE=64;
              // constexpr int HEAD_SIZE=128;
              // using scalar_t=uint16_t;
              constexpr bool is_e4m3=true;
              // constexpr static int REUSE_KV_TIMES = 8;
              constexpr static int NUM_THREADS = 256;
              constexpr static int NUM_WARPS = NUM_THREADS / WARP_SIZE;
              int real_reuse_times = num_heads/num_kv_heads;
              constexpr bool use_fp8_lds = REUSE_KV_TIMES>=24&&is_e4m3;
              int other_use = (real_reuse_times*NUM_WARPS+NUM_WARPS+ real_reuse_times*2)*sizeof(float);
              int shared_mem_size=PARTITION_SIZE*(2-use_fp8_lds)*real_reuse_times+other_use;
              /*
               * [窗口优化-FP8 BSHD host helper]
               * host 端使用与 device/combine 相同的窗口起点计算 grid.z；
               * partition_window_size_left=-1 时退化为完整 partition 上界。
               */
              const int max_num_partitions =
                  get_paged_attention_max_num_partitions(
                      max_seq_len,
                      mtp,
                      partition_window_size_left,
                      BLOCK_SIZE,
                      PARTITION_SIZE,
                      partition_window_size_left >= 0);
              grid.z = max_num_partitions;
              dim3 block(NUM_THREADS);
              if(PA_PRINT_PARAM)printf("sizeof(q)=%d,shared_mem_size=%d,HEAD_SIZE=%d,num_thread=%d,grid={%d,%d,%d},qhead=%d,kvhead=%d,seq=%d,batch=%d,PARTITION_SIZE=%d,max_num_partitions=%d\n",
                                        sizeof(q_type),shared_mem_size,HEAD_SIZE,NUM_THREADS,grid.x,grid.y,grid.z,num_heads,num_kv_heads,max_seq_len,num_seqs,PARTITION_SIZE,max_num_partitions);
              if constexpr(use_fp8_lds){
                constexpr int patitionsize = REUSE_KV_TIMES>24?256:512;
                /*
                 * [窗口优化-FP8 BSHD fast 参数转发]
                 * 将窗口边界传入 BSHD fast kernel。
                 */
                paged_attention_kernel_fp8_bshd_fast<scalar_t,q_type,is_e4m3,HEAD_SIZE,BLOCK_SIZE,NUM_THREADS,REUSE_KV_TIMES,patitionsize><<<grid,block,shared_mem_size,stream>>>(
                  (scalar_t*)out_ptr,(scalar_t*)tmp_out_ptr, (q_type*)query_ptr,(uint8_t*) key_cache_ptr, (uint8_t*)value_cache_ptr,
                  num_heads, num_kv_heads, block_tables_ptr, seq_lens_ptr,max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride,
                  q_scale_ptr,k_scale_ptr, v_scale_ptr,max_num_partitions,(const scalar_t*)s_aux_ptr,mtp,alibi_slopes_ptr!=nullptr,
                  window_size_left,effective_right,partition_window_size_left);
              }
              else{
                /*
                 * [窗口优化-FP8 BSHD 普通参数转发]
                 * 将窗口边界传入 BSHD 普通 FP8 kernel。
                 */
                paged_attention_kernel_fp8_bshd<scalar_t,q_type,is_e4m3,HEAD_SIZE,BLOCK_SIZE,NUM_THREADS,REUSE_KV_TIMES><<<grid,block,shared_mem_size,stream>>>(
                  (scalar_t*)out_ptr,(scalar_t*)tmp_out_ptr, (q_type*)query_ptr,(uint8_t*) key_cache_ptr, (uint8_t*)value_cache_ptr,
                  num_heads, num_kv_heads, block_tables_ptr, seq_lens_ptr,max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride,
                  q_scale_ptr,k_scale_ptr, v_scale_ptr,max_num_partitions,PARTITION_SIZE,(const scalar_t*)s_aux_ptr,mtp,alibi_slopes_ptr!=nullptr,
                  window_size_left,effective_right,partition_window_size_left);
              }
              if(max_num_partitions>1){
                paged_attention_combine<scalar_t,HEAD_SIZE,64><<<dim3(num_heads,num_seqs),64,4*2*max_num_partitions,stream>>>(
                  (scalar_t*)out_ptr,(scalar_t*)tmp_out_ptr,seq_lens_ptr,max_num_partitions,num_heads,PARTITION_SIZE,
                  partition_window_size_left,mtp,BLOCK_SIZE);
              }
          });
        });
      });
    });
  });
}
 
 
