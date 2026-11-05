#pragma once

// Internal decode implementations shared by gfx936/gfx92a; included after PA_* policies.
// ---- gfx936: paged_attention_decode_scheduler ----


// Decode scheduler: the legacy grid axes are (KV head, sequence, partition).
// All waves derive the same task without a global queue or replay-time reset.
template <bool Persistent>
struct Pa936DecodeTileScheduler {
    const int* lengths;
    int batch, kv_heads, mtp, window_left, page_size, partition_size;
    int group = 0;
    long long task = 0, group_start = 0, inclusive = 0, count = 0, group_count = 0;
    bool loaded = false, done = false;

    __device__ __forceinline__ Pa936DecodeTileScheduler(
        const int* lengths_, int batch_, int kv_heads_, int mtp_,
        int window_left_, int page_size_, int partition_size_)
        : lengths(lengths_), batch(batch_), kv_heads(kv_heads_), mtp(mtp_),
          window_left(window_left_), page_size(page_size_),
          partition_size(partition_size_), task(blockIdx.x) {}

    __device__ __forceinline__ bool next(int& seq, int& head, int& part) {
        if constexpr (!Persistent) {
            seq = blockIdx.y;
            head = blockIdx.x;
            part = blockIdx.z;
            return !done;
        } else {
            const int lane = threadIdx.x % 64;
            while (group < batch) {
                if (!loaded) {
                    const int index = group + lane;
                    const int nparts = index < batch ? get_paged_attention_active_range(
                        lengths[index], mtp, window_left, page_size, partition_size, 0)
                        .num_active_partitions : 0;
                    count = static_cast<long long>(nparts) * kv_heads;
                    inclusive = count;
                    #pragma unroll
                    for (int delta = 1; delta < 64; delta *= 2) {
                        const long long other = __shfl_up(inclusive, delta, 64);
                        if (lane >= delta) inclusive += other;
                    }
                    group_count = __shfl(inclusive, 63, 64);
                    loaded = true;
                }
                const long long relative = task - group_start;
                if (relative < group_count) {
                    const auto matches = __ballot(relative >= inclusive - count && relative < inclusive);
                    const int owner = __ffsll(matches) - 1;
                    const long long first = __shfl(inclusive - count, owner, 64);
                    const long long local = relative - first;
                    seq = __builtin_amdgcn_readfirstlane(group + owner);
                    head = __builtin_amdgcn_readfirstlane(static_cast<int>(local % kv_heads));
                    part = __builtin_amdgcn_readfirstlane(static_cast<int>(local / kv_heads));
                    return true;
                }
                group_start += group_count;
                group += 64;
                loaded = false;
            }
            return false;
        }
    }

    __device__ __forceinline__ void advance() {
        if constexpr (Persistent) task += gridDim.x;
        else done = true;
    }
};

// A batch of one or two sequences needs only uniform scalar prefix lengths.
// Avoid the per-lane 64-sequence scan and its live vector registers.
struct Pa936SmallDecodeScheduler {
    const int* lengths;
    int batch, heads, mtp, left, page, partition;
    int task, first_tasks=0, total_tasks=0;
    bool loaded=false;
    __device__ __forceinline__ Pa936SmallDecodeScheduler(
        const int* p, int b, int h, int m, int l, int ps, int part)
        : lengths(p), batch(b), heads(h), mtp(m), left(l), page(ps),
          partition(part), task(blockIdx.x) {}
    __device__ __forceinline__ bool next(int& seq, int& head, int& part) {
        if (!loaded) {
            const int n0=__builtin_amdgcn_readfirstlane(lengths[0]);
            const int p0=get_paged_attention_active_range(n0,mtp,left,page,partition,0).num_active_partitions;
            first_tasks=__builtin_amdgcn_readfirstlane(p0*heads);
            int second_tasks=0;
            if (batch==2) {
                const int n1=__builtin_amdgcn_readfirstlane(lengths[1]);
                second_tasks=get_paged_attention_active_range(n1,mtp,left,page,partition,0).num_active_partitions*heads;
            }
            total_tasks=__builtin_amdgcn_readfirstlane(first_tasks+second_tasks);
            loaded=true;
        }
        if (task>=total_tasks) return false;
        seq=__builtin_amdgcn_readfirstlane(task>=first_tasks);
        const int local=task-(seq ? first_tasks : 0);
        if ((heads & (heads-1))==0) {
            const int shift=__ffs(heads)-1;
            head=__builtin_amdgcn_readfirstlane(local & (heads-1));
            part=__builtin_amdgcn_readfirstlane(local >> shift);
        } else {
            head=__builtin_amdgcn_readfirstlane(local % heads);
            part=__builtin_amdgcn_readfirstlane(local / heads);
        }
        return true;
    }
    __device__ __forceinline__ void advance() { task+=gridDim.x; }
};

// ---- gfx936: paged_attention_decode_dense_gfx936 ----
// Include after paged_attention.h, whose legacy declarations have no include guard.

// Original ZIP7 arithmetic and straight-line grid mapping, without a persistent loop.

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void pa936_dense_layout0(
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
    const int* __restrict__ q_rows = nullptr) {  // [num_seqs, mtp] query row per mtp slot, -1 = padding
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
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
  if(num_active_partitions<=partition_idx)return ;
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
  const int kv_head_idx = blockIdx.x;
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
}

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES>  // Zero means no partitioning.
/*
 * [窗口优化-BHSD kernel 参数]
 * BHSD kernel 接收 attention mask 的 left/effective right，以及 partition
 * 重编号所需的窗口起点；effective right 已在 host 端完成 causal 转换。
 */
__launch_bounds__(NUM_THREADS) __global__ void pa936_dense_layout1(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_heads, block_size, head_size]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_heads, block_size, head_size]
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
    int partition_window_size_left) {  // ★ Attention Sinks: [num_heads] scalar_t ★
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride=BLOCK_SIZE*HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-BHSD]
   * dispatcher 传入 -1 时保持完整 partition；启用窗口压缩时，helper 从
   * first_active_block 开始重新编号 local partition，避免跳过边界 page。
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
  constexpr bool is_fp8 = std::is_same<cache_t, uint8_t>::value;
  constexpr float scale = (HEAD_SIZE==64?0.125f:(HEAD_SIZE==128? 0.0883883476f:(HEAD_SIZE==192?0.0721687836f:0.0625f)))*1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = blockIdx.x;
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
   * [窗口优化-BHSD page 映射]
   * 使用公共 helper 返回的 block 范围。传入 -1 时等价于原始完整序列
   * 映射；将来打开 BHSD compaction 时只需传入实际 left。
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
             * BHSD 通用 kernel 也可能由 Cache_Type_SWITCH 实例化为 FP8 cache。
             * 因此 left/right/causal mask 必须同时覆盖 FP8 和非 FP8。
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
               * 旧的 mtp>1 隐式 causal 已删除；causal 由 effective_right=0 表示。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              /*
               * [窗口优化-softmax]
               * 被屏蔽的 FP8/非 FP8 score 都不能更新 qk_max。
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
          if(num_partitions>1&&half_lane==0){
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
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*4*vecsize*HEAD_SIZE+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE]);
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=v_ptr[offset+ii*HEAD_SIZE];
          }
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
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*4*vecsize*HEAD_SIZE+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=token_idx + ii < seq_len ? uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE]) : 0;
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE] : 0;
          }
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
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
}

template <typename scalar_t, typename cache_t, bool is_e4m3, int HEAD_SIZE_QK, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void pa936_dense_layout2(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, block_size, num_heads, head_size]
    const cache_t* __restrict__ v_cache,  // [num_blocks, block_size, num_heads, head_size]
                                          // head_size, block_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int k_block_stride, const int v_block_stride,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  // ★ Attention Sinks: [num_heads] scalar_t ★
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int k_head_stride=HEAD_SIZE_QK;
  constexpr int v_head_stride=HEAD_SIZE_V;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-BSHD]
   * dispatcher 传入 -1 时保持完整 partition；启用窗口压缩时，helper 从
   * first_active_block 开始重新编号 local partition。
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
  constexpr bool is_fp8 = std::is_same<cache_t, uint8_t>::value;
  constexpr float qk_scale = HEAD_SIZE_QK == 64 ? 0.125f
                           : HEAD_SIZE_QK == 128 ? 0.0883883476f
                           : HEAD_SIZE_QK == 192 ? 0.0721687836f
                                                 : 0.0625f;
  constexpr float scale = qk_scale * 1.4426950408889634;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane = thread_idx % WARP_SIZE;
  const int rowid = lane%16;
  const int rows = lane/16;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = blockIdx.x;
  const int head_idx=num_queries_per_kv/mtp * kv_head_idx;
  constexpr int reuse_group=(REUSE_KV_TIMES-1)/4+1;
  constexpr int Mloop=(REUSE_KV_TIMES-1)/16+1;
  extern __shared__ char shared_mem[];
  using ldstype = _Float16;
  constexpr float lds_scale =  (HEAD_SIZE_QK==128&&REUSE_KV_TIMES==24) ? 512.0f:8.0f ;
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
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE_QK;
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
  half4x2 q_vec[Mloop][HEAD_SIZE_QK/32];
  half4x2 q_zero;
  q_zero.data[0]={0,0,0,0};
  q_zero.data[1]={0,0,0,0};
  scalar_t* s_q = reinterpret_cast<scalar_t*>(shared_mem);
  {
    int head_offset = HEAD_SIZE_QK*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE_QK;i+=NUM_THREADS*8){
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;
      *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
    }
  }

  __syncthreads();
  for(int m=0;m<Mloop;m++){
    int head_idx_=rowid+16*m;
    for(int i=0;i<HEAD_SIZE_QK/32;i++){
      if(head_idx_<num_queries_per_kv) {
        q_vec[m][i]=*reinterpret_cast<const half4x2*>(s_q+head_idx_*HEAD_SIZE_QK+(i*4+rows)*8);
      }
      else q_vec[m][i]=q_zero;
    }
  }
  __syncthreads();
  /*
   * [窗口优化-BSHD page 映射]
   * 使用公共 helper 返回的 block 范围；当前传入 -1 时与原始完整序列
   * 的 partition 映射等价，并保留 BSHD 自身的 BLOCK_SIZE。
   */
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //comput q*k
  {
    const cache_t* k_ptr_base = k_cache+kv_head_idx * k_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const cache_t* k_ptr=k_ptr_base + physical_block_number * k_block_stride + b*num_kv_heads*HEAD_SIZE_QK;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        half4x2 k_vec[HEAD_SIZE_QK/32];
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE_QK/32;i++){
          if constexpr(is_fp8){
            uint8x4x2 k_vec_u8=*reinterpret_cast<const uint8x4x2*>(k_ptr+i*32+rowid*num_kv_heads*HEAD_SIZE_QK+rows*8);
            scalar_t *p1=(scalar_t*)(k_vec+i);
            uint8_t *p2=(uint8_t*)&k_vec_u8;
            for(int ii=0;ii<8;ii++){
              p1[ii]=uint82half<scalar_t,is_e4m3>(p2[ii]);
            }
          }
          else{
            k_vec[i]=*reinterpret_cast<const half4x2*>(k_ptr+i*32+rowid*num_kv_heads*HEAD_SIZE_QK+rows*8);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE_QK/32;i++){
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
             * BSHD 通用 kernel 可能接收 uint8_t FP8 cache，不能用
             * if constexpr (!is_fp8) 把位置 mask 编译掉。
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
               * 旧的 mtp>1 隐式 causal 已删除；causal 由 effective_right=0 表示。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              /*
               * [窗口优化-softmax]
               * 被 mask 的分数只写入 logits，不参与 qk_max。
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
          if(num_partitions>1&&half_lane==0){
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
          if(num_partitions>1&&lane==0){
            max_out[lineid] = qk_max_tmp;
            expsum_out[lineid] = exp_sum;
          }
        }
      }
    }
  }
  __syncthreads();
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE_V, 16*NUM_WARPS);//2
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
    const cache_t* v_ptr = v_cache + physical_block_number * v_block_stride + kv_head_idx * v_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE_V/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE_V/NUM_ROWS_PER_THREAD/NUM_WARPS+
                   rows*4*vecsize*HEAD_SIZE_V*num_kv_heads+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads]);
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads];
          }
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
        int offset=i*HEAD_SIZE_V/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE_V/NUM_ROWS_PER_THREAD/NUM_WARPS+
                   rows*4*vecsize*HEAD_SIZE_V*num_kv_heads+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=token_idx + ii < seq_len
                ? uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads]) : 0;
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads] : 0;
          }
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
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
    if(num_partitions>1){
      out_offset=max_num_partitions*HEAD_SIZE_V;
      out_ptr_base=out_tmp+out_tmp_offset + seq_idx * num_heads * out_offset + head_idx*out_offset+
                   partition_idx * HEAD_SIZE_V;
    }
    else{
      out_offset=HEAD_SIZE_V;
      out_ptr_base=out + seq_idx * num_heads  * HEAD_SIZE_V + head_idx*HEAD_SIZE_V;
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
}

// ---- gfx936: paged_attention_decode_legacy_gfx936 ----
// Legacy math and layout-specific vector loads from the user's original ZIP7.
// Only task scheduling and shared partition metadata have changed.

template <typename scalar_t, typename cache_t,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, bool PERSISTENT, bool SMALL_BATCH>  // Zero means no partitioning.
/*
 * [窗口优化-BHSD kernel 参数]
 * BHSD kernel 接收 attention mask 的 left/effective right，以及 partition
 * 重编号所需的窗口起点；effective right 已在 host 端完成 causal 转换。
 */
__launch_bounds__(NUM_THREADS) __global__ void pa936_legacy_kernel_bhsd(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_heads, block_size, head_size]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_heads, block_size, head_size]
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
    int partition_window_size_left, int num_seqs) {  // ★ Attention Sinks: [num_heads] scalar_t ★
  using Scheduler=std::conditional_t<SMALL_BATCH, Pa936SmallDecodeScheduler,
                                       Pa936DecodeTileScheduler<PERSISTENT>>;
  Scheduler scheduler(seq_lens,num_seqs,num_kv_heads,mtp,partition_window_size_left,BLOCK_SIZE,PARTITION_SIZE);
  int seq_idx,scheduled_head,partition_idx;
  for (; scheduler.next(seq_idx,scheduled_head,partition_idx); scheduler.advance()) {
  constexpr int kv_head_stride=BLOCK_SIZE*HEAD_SIZE;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-BHSD]
   * dispatcher 传入 -1 时保持完整 partition；启用窗口压缩时，helper 从
   * first_active_block 开始重新编号 local partition，避免跳过边界 page。
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
  if(num_partitions<=partition_idx) { if constexpr(PERSISTENT) continue; else return; }
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
   * [窗口优化-BHSD page 映射]
   * 使用公共 helper 返回的 block 范围。传入 -1 时等价于原始完整序列
   * 映射；将来打开 BHSD compaction 时只需传入实际 left。
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
             * BHSD 通用 kernel 也可能由 Cache_Type_SWITCH 实例化为 FP8 cache。
             * 因此 left/right/causal mask 必须同时覆盖 FP8 和非 FP8。
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
               * 旧的 mtp>1 隐式 causal 已删除；causal 由 effective_right=0 表示。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              /*
               * [窗口优化-softmax]
               * 被屏蔽的 FP8/非 FP8 score 都不能更新 qk_max。
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
          if(num_partitions>1&&half_lane==0){
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
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*4*vecsize*HEAD_SIZE+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE]);
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=v_ptr[offset+ii*HEAD_SIZE];
          }
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
        int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*4*vecsize*HEAD_SIZE+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=token_idx + ii < seq_len ? uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE]) : 0;
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE] : 0;
          }
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
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
  if constexpr(PERSISTENT) __syncthreads();
  }
}

template <typename scalar_t, typename cache_t, bool is_e4m3, int HEAD_SIZE_QK, int HEAD_SIZE_V, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, bool PERSISTENT, bool SMALL_BATCH>  // Zero means no partitioning.
__launch_bounds__(NUM_THREADS) __global__ void pa936_legacy_kernel_bshd(
    scalar_t* __restrict__ out,  // [num_seqs, num_heads,head_size]
    scalar_t* __restrict__ out_tmp,  // [num_seqs, num_heads, max_num_partitions,head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, block_size, num_heads, head_size]
    const cache_t* __restrict__ v_cache,  // [num_blocks, block_size, num_heads, head_size]
                                          // head_size, block_size]
    const int num_heads,
    const int num_kv_heads,               // [num_heads]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int k_block_stride, const int v_block_stride,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left, int num_seqs) {  // ★ Attention Sinks: [num_heads] scalar_t ★
  using Scheduler=std::conditional_t<SMALL_BATCH, Pa936SmallDecodeScheduler,
                                       Pa936DecodeTileScheduler<PERSISTENT>>;
  Scheduler scheduler(seq_lens,num_seqs,num_kv_heads,mtp,partition_window_size_left,BLOCK_SIZE,PARTITION_SIZE);
  int seq_idx,scheduled_head,partition_idx;
  for (; scheduler.next(seq_idx,scheduled_head,partition_idx); scheduler.advance()) {
  constexpr int k_head_stride=HEAD_SIZE_QK;
  constexpr int v_head_stride=HEAD_SIZE_V;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  /*
   * [窗口优化-公共 helper 接入-BSHD]
   * dispatcher 传入 -1 时保持完整 partition；启用窗口压缩时，helper 从
   * first_active_block 开始重新编号 local partition。
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
  if(num_partitions<=partition_idx) { if constexpr(PERSISTENT) continue; else return; }
  constexpr bool is_half = std::is_same<scalar_t, _Float16>::value;
  constexpr bool is_fp8 = std::is_same<cache_t, uint8_t>::value;
  constexpr float qk_scale = HEAD_SIZE_QK == 64 ? 0.125f
                           : HEAD_SIZE_QK == 128 ? 0.0883883476f
                           : HEAD_SIZE_QK == 192 ? 0.0721687836f
                                                 : 0.0625f;
  constexpr float scale = qk_scale * 1.4426950408889634;
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
  constexpr float lds_scale =  (HEAD_SIZE_QK==128&&REUSE_KV_TIMES==24) ? 512.0f:8.0f ;
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
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE_QK;
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
  half4x2 q_vec[Mloop][HEAD_SIZE_QK/32];
  half4x2 q_zero;
  q_zero.data[0]={0,0,0,0};
  q_zero.data[1]={0,0,0,0};
  scalar_t* s_q = reinterpret_cast<scalar_t*>(shared_mem);
  {
    int head_offset = HEAD_SIZE_QK*num_queries_per_kv/mtp;
    for(int i=thread_idx*8;i<num_queries_per_kv*HEAD_SIZE_QK;i+=NUM_THREADS*8){
      int qoffset=i/head_offset;
      qoffset*=num_kv_heads*head_offset;
      qoffset+=i%head_offset;
      *reinterpret_cast<half4x2*>(s_q+i)=*reinterpret_cast<const half4x2*>(q_ptr+qoffset);
    }
  }

  __syncthreads();
  for(int m=0;m<Mloop;m++){
    int head_idx_=rowid+16*m;
    for(int i=0;i<HEAD_SIZE_QK/32;i++){
      if(head_idx_<num_queries_per_kv) {
        q_vec[m][i]=*reinterpret_cast<const half4x2*>(s_q+head_idx_*HEAD_SIZE_QK+(i*4+rows)*8);
      }
      else q_vec[m][i]=q_zero;
    }
  }
  __syncthreads();
  /*
   * [窗口优化-BSHD page 映射]
   * 使用公共 helper 返回的 block 范围；当前传入 -1 时与原始完整序列
   * 的 partition 映射等价，并保留 BSHD 自身的 BLOCK_SIZE。
   */
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  //comput q*k
  {
    const cache_t* k_ptr_base = k_cache+kv_head_idx * k_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const cache_t* k_ptr=k_ptr_base + physical_block_number * k_block_stride + b*num_kv_heads*HEAD_SIZE_QK;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        half4x2 k_vec[HEAD_SIZE_QK/32];
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE_QK/32;i++){
          if constexpr(is_fp8){
            uint8x4x2 k_vec_u8=*reinterpret_cast<const uint8x4x2*>(k_ptr+i*32+rowid*num_kv_heads*HEAD_SIZE_QK+rows*8);
            scalar_t *p1=(scalar_t*)(k_vec+i);
            uint8_t *p2=(uint8_t*)&k_vec_u8;
            for(int ii=0;ii<8;ii++){
              p1[ii]=uint82half<scalar_t,is_e4m3>(p2[ii]);
            }
          }
          else{
            k_vec[i]=*reinterpret_cast<const half4x2*>(k_ptr+i*32+rowid*num_kv_heads*HEAD_SIZE_QK+rows*8);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE_QK/32;i++){
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
             * BSHD 通用 kernel 可能接收 uint8_t FP8 cache，不能用
             * if constexpr (!is_fp8) 把位置 mask 编译掉。
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
               * 旧的 mtp>1 隐式 causal 已删除；causal 由 effective_right=0 表示。
               */
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              /*
               * [窗口优化-softmax]
               * 被 mask 的分数只写入 logits，不参与 qk_max。
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
          if(num_partitions>1&&half_lane==0){
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
          if(num_partitions>1&&lane==0){
            max_out[lineid] = qk_max_tmp;
            expsum_out[lineid] = exp_sum;
          }
        }
      }
    }
  }
  __syncthreads();
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE_V, 16*NUM_WARPS);//2
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
    const cache_t* v_ptr = v_cache + physical_block_number * v_block_stride + kv_head_idx * v_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int offset=i*HEAD_SIZE_V/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE_V/NUM_ROWS_PER_THREAD/NUM_WARPS+
                   rows*4*vecsize*HEAD_SIZE_V*num_kv_heads+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads]);
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads];
          }
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
        int offset=i*HEAD_SIZE_V/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE_V/NUM_ROWS_PER_THREAD/NUM_WARPS+
                   rows*4*vecsize*HEAD_SIZE_V*num_kv_heads+rowid;
        half4_vec v_vec;
        scalar_t *p=(scalar_t*)&v_vec;
        if constexpr(is_fp8){
          for(int ii=0;ii<vecsize*4;ii++){
            p[ii]=token_idx + ii < seq_len
                ? uint82half<scalar_t,is_e4m3>(v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads]) : 0;
          }
        }else{
          for(int ii = 0 ; ii<4*vecsize ; ii++){
            p[ii]=token_idx + ii < seq_len ? v_ptr[offset+ii*HEAD_SIZE_V*num_kv_heads] : 0;
          }
        }
        //这里的if判断会影响一定的性能，因此只有最后一个patition才判断
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
    if(num_partitions>1){
      out_offset=max_num_partitions*HEAD_SIZE_V;
      out_ptr_base=out_tmp+out_tmp_offset + seq_idx * num_heads * out_offset + head_idx*out_offset+
                   partition_idx * HEAD_SIZE_V;
    }
    else{
      out_offset=HEAD_SIZE_V;
      out_ptr_base=out + seq_idx * num_heads  * HEAD_SIZE_V + head_idx*HEAD_SIZE_V;
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
  if constexpr(PERSISTENT) __syncthreads();
  }
}


void paged_attention_decode_legacy_gfx936(
    torch::Tensor& out, torch::Tensor& query_in, torch::Tensor& key,
    torch::Tensor& value, torch::Tensor& table, torch::Tensor& lengths,
    float* workspace, int layout, int left, int right,
    const Pa28WindowPlan& plan, int compact_policy, int configured_workers,
    bool use_small_scheduler)
{
    TORCH_CHECK(layout==1 || layout==2,"PA936 legacy: invalid layout");
    auto query=query_in.is_contiguous() ? query_in : query_in.contiguous();
    const int batch=query.size(0), mtp=query.size(1), heads=query.size(2)*mtp;
    const int kvheads=key.size(layout==2 ? 2 : 1), d=query.size(3);
    const int reuse=heads/kvheads, part=plan.partition_size, parts=plan.num_partitions;
    TORCH_CHECK(mtp==1 && reuse>0 && reuse<=8,"PA936 legacy: query group out of range");
    const auto stream=at::cuda::getCurrentCUDAStream();
    hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
    TORCH_CHECK(hipStreamIsCapturing(stream,&status)==hipSuccess,"PA936 legacy: cannot query capture");
    const int workers=configured_workers==-1 && batch<=2 ? 256 :
        pa28_graph_workers(configured_workers,"gfx936",at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    const int64_t tasks=static_cast<int64_t>(batch)*kvheads*parts;
    const bool compact=status!=hipStreamCaptureStatusNone && compact_policy!=0 &&
        (compact_policy==2 || (tasks>=8192 && tasks>static_cast<int64_t>(workers)*4 &&
                              !(batch<=2 || (d>=192 && batch<=8))));
    const dim3 grid=compact ? dim3(static_cast<unsigned>(std::min<int64_t>(workers,tasks)),1,1) :
                            dim3(kvheads,batch,parts);
    const int shared=part*2*reuse+(reuse*6+4)*sizeof(float);
    if (PA_PRINT_PARAM) printf("PA936_LEGACY_LAYOUT layout=%d compact=%d small=%d workers=%d parts=%d partition_size=%d\n",
        layout,compact,compact && batch<=2 && use_small_scheduler,workers,parts,part);
    HEADSIZE_SWITCH(d,[&]{
      Output_Type_SWITCH(out.scalar_type(),[&]{
        Cache_Type_SWITCH(scalar_t,key.scalar_type(),[&]{
          auto select_reuse=[&](auto rtag) {
            constexpr int R=decltype(rtag)::value;
            auto launch=[&](auto ptag) {
              constexpr int P=decltype(ptag)::value;
              if constexpr(P==0) {
                if (layout==1) {
                  pa936_dense_layout1<scalar_t,cache_t,false,HEAD_SIZE,64,256,R>
                    <<<grid,256,shared,stream>>>(
                    static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                    static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                    static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                    lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),parts,part,
                    nullptr,mtp,false,left,right,left);
                } else {
                  pa936_dense_layout2<scalar_t,cache_t,false,HEAD_SIZE,HEAD_SIZE,64,256,R>
                    <<<grid,256,shared,stream>>>(
                    static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                    static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                    static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                    lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),value.stride(0),
                    parts,part,nullptr,mtp,false,left,right,left);
                }
              } else {
              if (layout==1) {
                pa936_legacy_kernel_bhsd<scalar_t,cache_t,false,HEAD_SIZE,64,256,R,P!=0,P==2>
                  <<<grid,256,shared,stream>>>(
                    static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                    static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                    static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                    lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),parts,part,
                    nullptr,mtp,false,left,right,left,batch);
              } else {
                pa936_legacy_kernel_bshd<scalar_t,cache_t,false,HEAD_SIZE,HEAD_SIZE,64,256,R,P!=0,P==2>
                  <<<grid,256,shared,stream>>>(
                    static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                    static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                    static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                    lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),value.stride(0),
                    parts,part,nullptr,mtp,false,left,right,left,batch);
              }
              }
            };
            if (compact && batch<=2 && use_small_scheduler) launch(std::integral_constant<int,2>{});
            else if (compact) launch(std::integral_constant<int,1>{});
            else launch(std::integral_constant<int,0>{});
          };
          if (reuse<=4) select_reuse(std::integral_constant<int,4>{});
          else select_reuse(std::integral_constant<int,8>{});
          if (parts>1) paged_attention_combine<scalar_t,HEAD_SIZE,64>
            <<<dim3(heads,batch),64,8*parts,stream>>>(static_cast<scalar_t*>(out.data_ptr()),
              reinterpret_cast<scalar_t*>(workspace),lengths.data_ptr<int>(),parts,heads,part,left,mtp,64);
        });
      });
    });
    TORCH_CHECK(hipGetLastError()==hipSuccess,"PA936 legacy: kernel launch failed");
}


// ---- gfx938: layout-specialized decode, including native E4M3 MMAC ----
// Existing gfx936/gfx92a code above is unchanged. FP8 math follows ZIP7,
// with layout addressing specialized at compile time and the 1031 workspace plan.
// The scalar tiny-row route is bounded by the available head parallelism.
// Crossovers were checked against grouped MMAC on both small and large batches.
template<int D>
__device__ __forceinline__ int pa938_tiny_limit(int heads,int kvheads,int batch) {
  const int reuse=heads/kvheads;
  const int parallel_heads=heads*batch;
  if(reuse<=24) return parallel_heads<=512 ? 20 : 0;
  if(reuse<=32) return parallel_heads<=128 ? 24 : (parallel_heads<=512 ? 20 : 0);
  if(reuse<=48) return parallel_heads<=128 ? 48 : (parallel_heads<=512 ? 24 : 0);
  return parallel_heads<=128 ? 72 : (parallel_heads<=512 ? (D==64 ? 48 : 40) : (parallel_heads<=1024 ? 20 : 0));
}

template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int LAYOUT>  
__launch_bounds__(NUM_THREADS) __global__ void pa938_decode_fp8(
    scalar_t* __restrict__ out,  
    scalar_t* __restrict__ out_tmp,  
    const q_type* __restrict__ q,       
    const uint8_t* __restrict__ k_cache,  
    const uint8_t* __restrict__ v_cache,  
    const int num_heads,
    const int num_kv_heads,               
    const int* __restrict__ block_tables,  
    const int* __restrict__ seq_lens,      
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  
#if defined(__gfx938__)
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride = LAYOUT == 2 ? HEAD_SIZE : BLOCK_SIZE * HEAD_SIZE;
  const int kv_token_stride = HEAD_SIZE * (LAYOUT == 2 ? num_kv_heads : 1);
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  if(seq_len<=0) {
    if(blockIdx.x==0 && partition_idx==0) {
      for(int i=threadIdx.x;i<num_heads*HEAD_SIZE;i+=NUM_THREADS)
        from_float(out[seq_idx*num_heads*HEAD_SIZE+i],0.f);
    }
    return;
  }
  if constexpr(REUSE_KV_TIMES>=24) {
    if(mtp==1 && window_size_left<0 &&
       seq_len<=pa938_tiny_limit<HEAD_SIZE>(num_heads,num_kv_heads,gridDim.y)) return;
  }
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
  
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  const bool refine_probs = num_tokens <= 128 || (HEAD_SIZE == 192 && REUSE_KV_TIMES >= 24 && mtp == 1 && window_size_left < 0 && seq_len < 2048) || (window_size_left >= 0 && window_size_left + mtp <= 128);
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*kv_token_stride;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        intx4 k_vec[HEAD_SIZE/64];
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          k_vec[i]=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*kv_token_stride+rows*16);
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
              from_float(temp,qk_vec[m][ii]);
              logits_f16[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_float(temp));
              }
            }
          }
        }
      }
    }
  }
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
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(half_lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
            exp_sum+=logit32[ii];
          }
        }
        for (int mask = 16; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
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
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
            exp_sum+=logit32[ii];
          }
        }
        for (int mask = 32; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
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
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE, 16*NUM_WARPS);
  constexpr int GROUPS=reuse_group*4;
  float4_t accs[Mloop][NUM_ROWS_PER_THREAD];
  for(int m=0;m<Mloop;m++){
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[m][i] = {0.f,0.f,0.f,0.f};
    }
  }
  constexpr int vecsize=BLOCK_SIZE/32;
  using int_vec = int2vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    intx2 logits_vec[Mloop][vecsize];
    intx2 probability_residual[Mloop][vecsize];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m][i]={0,0};
        probability_residual[m][i]={0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        for(int k=0;k<vecsize;k++){
          auto l_temp = *reinterpret_cast<half8_t*>(logits + real_row * PARTITION_SIZE+token_idx - start_token_idx + k*8);
          scalar_t *l_temp_ptr=(scalar_t*)&l_temp;
          logits_vec[m][k][0]=to_f8_from_f32<is_e4m3>(256.0f*to_float(l_temp_ptr[0]),256.0f*to_float(l_temp_ptr[1]),256.0f*to_float(l_temp_ptr[2]),256.0f*to_float(l_temp_ptr[3]));
          logits_vec[m][k][1]=to_f8_from_f32<is_e4m3>(256.0f*to_float(l_temp_ptr[4]),256.0f*to_float(l_temp_ptr[5]),256.0f*to_float(l_temp_ptr[6]),256.0f*to_float(l_temp_ptr[7]));
          if (refine_probs) {
            const auto* hi=reinterpret_cast<const uint8_t*>(&logits_vec[m][k]);
            float residual[8];
            #pragma unroll
            for (int j=0;j<8;++j) residual[j]=256.0f*to_float(l_temp_ptr[j])-uint82float(hi[j]);
            probability_residual[m][k][0]=to_f8_from_f32<is_e4m3>(residual[0],residual[1],residual[2],residual[3]);
            probability_residual[m][k][1]=to_f8_from_f32<is_e4m3>(residual[4],residual[5],residual[6],residual[7]);
          }
        }
      }
    }
    const uint8_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                          kv_head_idx * kv_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int_vec v_vec;
        if constexpr (LAYOUT == 0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes = reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for (int j=0; j<8*vecsize; ++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
            if (refine_probs) builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],probability_residual[m][ii],accs[m][i]);
          }
        }
      }
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int_vec v_vec;
        if constexpr (LAYOUT == 0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes = reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for (int j=0; j<8*vecsize; ++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
        }
        uint8_t* v_vec_ptr = reinterpret_cast<uint8_t*>(&v_vec);
        #pragma unroll
        for (int j = 0; j < 8*vecsize; j++) {
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : 0;
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
            if (refine_probs) builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],probability_residual[m][ii],accs[m][i]);
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
          from_float(*(out_ptr + row_idx), accs[reusekvid/16][i][g%4]*(1.0f/256.0f));
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



// gfx938: compact short-tail precision and high-reuse FP8-LDS.
template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int LAYOUT, int LDS_STRIDE>  
__device__ __forceinline__ void pa938_fp8_short(
    scalar_t* __restrict__ out,  
    scalar_t* __restrict__ out_tmp,  
    const q_type* __restrict__ q,       
    const uint8_t* __restrict__ k_cache,  
    const uint8_t* __restrict__ v_cache,  
    const int num_heads,
    const int num_kv_heads,               
    const int* __restrict__ block_tables,  
    const int* __restrict__ seq_lens,      
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,int PARTITION_SIZE,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  
#if defined(__gfx938__)
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride = LAYOUT == 2 ? HEAD_SIZE : BLOCK_SIZE * HEAD_SIZE;
  const int kv_token_stride = HEAD_SIZE * (LAYOUT == 2 ? num_kv_heads : 1);
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
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
  float* s_max = reinterpret_cast<float*>(shared_mem + sizeof(scalar_t)*num_queries_per_kv*LDS_STRIDE);
  float* s_logit = s_max + num_queries_per_kv * NUM_WARPS;
  float* max_out = s_logit+NUM_WARPS;
  float* expsum_out = max_out+num_queries_per_kv;
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
  
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  constexpr bool refine_probs = true;
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*kv_token_stride;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        intx4 k_vec[HEAD_SIZE/64];
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          k_vec[i]=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*kv_token_stride+rows*16);
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
              if(token_idx<seq_len_pad) from_float(logits_f16[LDS_STRIDE*reuse_kv_idx+token_idx - start_token_idx],-INFINITY);
              else logits_f16[LDS_STRIDE*reuse_kv_idx+token_idx - start_token_idx]=0;
            }
            else{
              ldstype temp;
              from_float(temp,qk_vec[m][ii]);
              logits_f16[LDS_STRIDE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_float(temp));
              }
            }
          }
        }
      }
    }
  }
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
  if(PARTITION_SIZE<=256){
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
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(half_lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+real_line*LDS_STRIDE+half_lane*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
            exp_sum+=logit32[ii];
          }
        }
        for (int mask = 16; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(half_lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+real_line*LDS_STRIDE+half_lane*8)=logit16;
          if(num_partitions>1&&half_lane==0){
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
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(lane<seq_len_pad){
          float16x8_t logit16 = *reinterpret_cast<float16x8_t*>(logits+lineid*LDS_STRIDE+lane*8);
          for(int ii=0;ii<8;ii++){
            logit32[ii]=__builtin_amdgcn_exp2f(to_float(logit16[ii]) - qk_max_tmp);
            exp_sum+=logit32[ii];
          }
        }
        for (int mask = 32; mask >= 1; mask /= 2) {
          exp_sum += __shfl_xor(exp_sum, mask);
        }
        exp_sum += sink_contrib;
        const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
        if(lane<seq_len_pad){
          f16x8_t logit16;
          for(int ii=0;ii<8;ii++){
            scalar_t t;
            from_float(t,logit32[ii]*inv_sum);
            logit16[ii]=t;
          }
          *reinterpret_cast<f16x8_t*>(logits+lineid*LDS_STRIDE+lane*8)=logit16;
          if(num_partitions>1&&lane==0){
            max_out[lineid] = qk_max_tmp;
            expsum_out[lineid] = exp_sum;
          }
        }
      }
    }
  }
  __syncthreads();
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE, 16*NUM_WARPS);
  constexpr int GROUPS=reuse_group*4;
  float4_t accs[Mloop][NUM_ROWS_PER_THREAD];
  for(int m=0;m<Mloop;m++){
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[m][i] = {0.f,0.f,0.f,0.f};
    }
  }
  constexpr int vecsize=BLOCK_SIZE/32;
  using int_vec = int2vec<vecsize>;
  for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx ++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE +rows*(BLOCK_SIZE/4);
    intx2 logits_vec[Mloop][vecsize];
    intx2 probability_residual[Mloop][vecsize];
    for(int m=0;m<Mloop;m++){
      for(int i=0;i<vecsize;i++){
        logits_vec[m][i]={0,0};
        probability_residual[m][i]={0,0};
      }
    }
    for(int m=0;m<Mloop;m++){
      int real_row=rowid+m*16;
      if(real_row<num_queries_per_kv){
        for(int k=0;k<vecsize;k++){
          auto l_temp = *reinterpret_cast<half8_t*>(logits + real_row * LDS_STRIDE+token_idx - start_token_idx + k*8);
          scalar_t *l_temp_ptr=(scalar_t*)&l_temp;
          logits_vec[m][k][0]=to_f8_from_f32<is_e4m3>(256.0f*to_float(l_temp_ptr[0]),256.0f*to_float(l_temp_ptr[1]),256.0f*to_float(l_temp_ptr[2]),256.0f*to_float(l_temp_ptr[3]));
          logits_vec[m][k][1]=to_f8_from_f32<is_e4m3>(256.0f*to_float(l_temp_ptr[4]),256.0f*to_float(l_temp_ptr[5]),256.0f*to_float(l_temp_ptr[6]),256.0f*to_float(l_temp_ptr[7]));
          if (refine_probs) {
            const auto* hi=reinterpret_cast<const uint8_t*>(&logits_vec[m][k]);
            float residual[8];
            #pragma unroll
            for (int j=0;j<8;++j) residual[j]=256.0f*to_float(l_temp_ptr[j])-uint82float(hi[j]);
            probability_residual[m][k][0]=to_f8_from_f32<is_e4m3>(residual[0],residual[1],residual[2],residual[3]);
            probability_residual[m][k][1]=to_f8_from_f32<is_e4m3>(residual[4],residual[5],residual[6],residual[7]);
          }
        }
      }
    }
    const uint8_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                          kv_head_idx * kv_head_stride;
    if(block_idx < num_seq_blocks-1){
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int_vec v_vec;
        if constexpr (LAYOUT == 0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes = reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for (int j=0; j<8*vecsize; ++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
            if (refine_probs) builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],probability_residual[m][ii],accs[m][i]);
          }
        }
      }
    }
    else{
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        int_vec v_vec;
        if constexpr (LAYOUT == 0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec = *reinterpret_cast<const int_vec*>(v_ptr + offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes = reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for (int j=0; j<8*vecsize; ++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
        }
        uint8_t* v_vec_ptr = reinterpret_cast<uint8_t*>(&v_vec);
        #pragma unroll
        for (int j = 0; j < 8*vecsize; j++) {
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : 0;
        }
        for(int ii=0;ii<vecsize;ii++){
          for(int m=0;m<Mloop;m++){
            builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],logits_vec[m][ii],accs[m][i]);
            if (refine_probs) builtin_amdgcn_mmac_fp8<is_e4m3>(v_vec.data[ii],probability_residual[m][ii],accs[m][i]);
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
          from_float(*(out_ptr + row_idx), accs[reusekvid/16][i][g%4]*(1.0f/256.0f));
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

template <typename scalar_t,typename q_type,bool is_e4m3 ,int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int REUSE_KV_TIMES, int PARTITION_SIZE, int LAYOUT>  
__launch_bounds__(NUM_THREADS) __global__ void pa938_decode_fp8_lds(
scalar_t* __restrict__ out,  
    scalar_t* __restrict__ out_tmp,  
    const q_type* __restrict__ q,       
    const uint8_t* __restrict__ k_cache,  
    const uint8_t* __restrict__ v_cache,  
    const int num_heads,
    const int num_kv_heads,               
    const int* __restrict__ block_tables,  
    const int* __restrict__ seq_lens,      
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  
    const int q_stride,const int kv_block_stride,
    const float* q_scale_ptr, const float* k_scale_ptr, const float* v_scale_ptr,
    int max_num_partitions,
    const scalar_t* __restrict__ s_aux_ptr,int mtp,bool has_abili,
    int window_size_left, int effective_right,
    int partition_window_size_left) {  
#if defined(__gfx938__)
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  constexpr int kv_head_stride=LAYOUT==2 ? HEAD_SIZE : BLOCK_SIZE*HEAD_SIZE;
  const int kv_token_stride=HEAD_SIZE*(LAYOUT==2 ? num_kv_heads : 1);
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  // Tiny rows expose more parallelism by query head.  Larger short rows
  // reuse K/V through precise 128-token matrix partitions instead.
  if (seq_len <= pa938_tiny_limit<HEAD_SIZE>(num_heads,num_kv_heads,gridDim.y)) return;
  constexpr int precise_cutoff = HEAD_SIZE<=192 ? 2048 : 512;
  const bool short_sequence = seq_len < precise_cutoff;
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
  if(!short_sequence && num_partitions<=partition_idx)return ;
  // A short tail uses precise FP16 logits and FP8 probability residuals.
  // Its compressed row stride occupies the same LDS as a full FP8 row.
  if (short_sequence || MIN(active_range.end_block_idx*BLOCK_SIZE,seq_len)-active_range.start_block_idx*BLOCK_SIZE <= 128) {
    pa938_fp8_short<scalar_t,q_type,is_e4m3,HEAD_SIZE,BLOCK_SIZE,NUM_THREADS,
                       REUSE_KV_TIMES,LAYOUT,PARTITION_SIZE/2>(
        out,out_tmp,q,k_cache,v_cache,num_heads,num_kv_heads,block_tables,seq_lens,
        max_num_blocks_per_seq,alibi_slopes,q_stride,kv_block_stride,
        q_scale_ptr,k_scale_ptr,v_scale_ptr,max_num_partitions,short_sequence ? 128 : PARTITION_SIZE,
        s_aux_ptr,mtp,has_abili,window_size_left,effective_right,partition_window_size_left);
    return;
  }

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
  
  const int start_block_idx = active_range.start_block_idx;
  const int end_block_idx = active_range.end_block_idx;
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  {
    const uint8_t* k_ptr_base = k_cache+kv_head_idx * kv_head_stride;
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      #pragma unroll
      for(int b=0;b<BLOCK_SIZE;b+=16){
        const uint8_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride + b*kv_token_stride;
        float4_t qk_vec[Mloop];
        for(int m=0;m<Mloop;m++){
          qk_vec[m]={0,0,0,0};
        }
        intx4 k_vec[HEAD_SIZE/64];
         __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i=0;i<HEAD_SIZE/64;i++){
          k_vec[i]=*reinterpret_cast<const intx4*>(k_ptr+i*64+rowid*kv_token_stride+rows*16);
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
              temp = window_masked ? static_cast<uint8_t>(0xfc) : to_f8_from_f32<true>(qk_vec[m][ii]);
              logits[PARTITION_SIZE*reuse_kv_idx+token_idx- start_token_idx]=temp;
              if (!window_masked) {
                qk_max[i] = fmaxf(qk_max[i], to_fp32_from_fp8<true>(temp)*lds_scale);
              }
            }
          }
        }
      }
    }
  }
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
          float s_aux_val = to_float(smem_s_aux[head_idx+real_line]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(half_lane<seq_len_pad){
          f8x8_t logit8 = *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*2*PARTITION_SIZE+thread_idx*8);
          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            float4_t temp = to_fp32_from_fp8<true>(logit8x4[ii]);
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
        const float inv_sum = __fdividef(256.f, exp_sum + 1e-6f);
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
          float s_aux_val = to_float(smem_s_aux[head_idx+lineid]);  
          sink_contrib = __builtin_amdgcn_exp2f(s_aux_val*1.4426950408889634 - qk_max_tmp);
        }
        f32x8_t logit32;
        if(lane<seq_len_pad){
          f8x8_t logit8 = *reinterpret_cast<f8x8_t*>(logits+lineid/NUM_WARPS*NUM_WARPS*PARTITION_SIZE+thread_idx*8);
          int *logit8x4=(int*)&logit8;
          for(int ii=0;ii<2;ii++){
            float4_t temp = to_fp32_from_fp8<true>(logit8x4[ii]);
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
        const float inv_sum = __fdividef(256.f, exp_sum + 1e-6f);
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
  constexpr int NUM_ROWS_PER_THREAD =DIVIDE_ROUND_UP(HEAD_SIZE, 16*NUM_WARPS);
  constexpr int GROUPS=reuse_group*4;
  float4_t accs[Mloop][NUM_ROWS_PER_THREAD];
  for(int m=0;m<Mloop;m++){
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[m][i] = {0.f,0.f,0.f,0.f};
    }
  }
  constexpr int vecsize=BLOCK_SIZE/32;
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
        int_vec v_vec;
        if constexpr(LAYOUT==0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec=*reinterpret_cast<const int_vec*>(v_ptr+offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes=reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for(int j=0;j<8*vecsize;++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
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
        int_vec v_vec;
        if constexpr(LAYOUT==0) {
          const int offset=i*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*BLOCK_SIZE*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize+rowid*BLOCK_SIZE;
          v_vec=*reinterpret_cast<const int_vec*>(v_ptr+offset);
        } else {
          const int offset=i*HEAD_SIZE/NUM_ROWS_PER_THREAD+warp_idx*HEAD_SIZE/NUM_ROWS_PER_THREAD/NUM_WARPS+rows*8*vecsize*kv_token_stride+rowid;
          auto* bytes=reinterpret_cast<uint8_t*>(&v_vec);
          #pragma unroll
          for(int j=0;j<8*vecsize;++j) bytes[j]=v_ptr[offset+j*kv_token_stride];
        }
        uint8_t* v_vec_ptr = reinterpret_cast<uint8_t*>(&v_vec);
        #pragma unroll
        for (int j = 0; j < 8*vecsize; j++) {
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : 0;
        }
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
          from_float(*(out_ptr + row_idx), accs[reusekvid/16][i][g%4]*(1.0f/256.0f));
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

template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__device__ __forceinline__ void pa938_combine_wave(
    scalar_t* __restrict__ out,            
    scalar_t* out_tmp,  
    const int* __restrict__ seq_lens,      
    const int max_num_partitions,
    int num_heads,
    int PARTITION_SIZE,
    int partition_window_size_left = -1,
    int mtp = 1,
    int block_size = 64) {
  extern __shared__ char shared_mem[];
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  const int lane = threadIdx.x;
  const PagedAttentionActiveRange active_range =
      get_paged_attention_active_range(
          seq_len,
          mtp,
          partition_window_size_left,
          block_size,
          PARTITION_SIZE,
          0);
  const int num_partitions = active_range.num_active_partitions;
  if(num_partitions<=1)return;
  float* shared_exp_sums=reinterpret_cast<float*>(shared_mem);
  float* shared_max_logits=shared_exp_sums+num_partitions;
  float max_logit = -FLT_MAX;
  float global_exp_sum = 0.0f;
  int offset = seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  const float * exp_sums=reinterpret_cast<float*>(out_tmp);
  const float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
  const float* max_logits_ptr = max_logits + offset;
  const float* exp_sums_ptr = exp_sums + offset;
  const scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  const scalar_t* tmp_out_ptr = out_tmp + out_tmp_offset + offset* HEAD_SIZE;
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l; 
    max_logit = fmaxf(max_logit,l);
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    float rescaled_exp_sum = exp_sums_ptr[i] * __builtin_amdgcn_exp2f(shared_max_logits[i] - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    global_exp_sum += __shfl_xor(global_exp_sum, mask);
  }
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);
  constexpr int vec_size_o=HEAD_SIZE/64;
  constexpr int vec_size = vec_size_o==3?4:vec_size_o;
  using half_vec= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  using float_vec= __attribute__( (__vector_size__(vec_size * sizeof(float)) )) float;
  float_vec acc = {0.0f};
  half_vec acc_half;
  if(lane<HEAD_SIZE/vec_size){
    for (int j = 0; j < num_partitions; ++j) {
      half_vec tout= *(half_vec*)(tmp_out_ptr + j * HEAD_SIZE + lane * vec_size);
      float temp_sum=shared_exp_sums[j]*inv_global_exp_sum;
      #pragma unroll
      for(int i=0;i<vec_size;i++){
        acc[i] += to_float(tout[i])*temp_sum;
      }
    }
    #pragma unroll
    for(int i=0;i<vec_size;i++){
      scalar_t temp;
      from_float(temp,acc[i]);
      acc_half[i]=temp;
    }
    *(half_vec*)(out_ptr+lane*vec_size)=acc_half;
  }
}


// Short and long rows share the same scratch stride.  The partition width
// must match the device-side decision in pa938_decode_fp8_lds at every replay.
template<typename scalar_t,int D>
__launch_bounds__(256) __global__ void pa938_fp8_finalize(
    scalar_t* out,scalar_t* tmp,const scalar_t* q,const uint8_t* k,const uint8_t* v,
    int heads,int kvheads,const int* table,const int* lengths,int table_stride,
    int q_stride,int kv_stride,int parts,int part,int precise_cutoff,int layout) {
#if defined(__gfx938__)
  const int b=blockIdx.y,h=blockIdx.x,tid=threadIdx.x;
  const int n=__builtin_amdgcn_readfirstlane(lengths[b]);
  if(n<=0) {
    if(tid<D) from_float(out[(b*heads+h)*D+tid],0.f);
    return;
  }
  if(precise_cutoff==0 || n>pa938_tiny_limit<D>(heads,kvheads,gridDim.y)) {
    if(tid<64) pa938_combine_wave<scalar_t,D,64>(out,tmp,lengths,parts,heads,
                                             n<precise_cutoff ? 128 : part,-1,1,64);
    return;
  }
  extern __shared__ char shared_mem[];
  float* sq=reinterpret_cast<float*>(shared_mem);
  float* prob=sq+D;float* scratch=prob+72;
  if(tid<D) {
    // Preserve the original 16-bit Q on tiny rows; requantizing it to FP8
    // can exceed the accuracy tolerance when only a few keys contribute.
    sq[tid]=to_float(q[b*q_stride+h*D+tid]);
  }
  __syncthreads();
  const int lane=tid%64,warp=tid/64,kh=h/(heads/kvheads);
  const int kt=layout==2 ? kvheads*D : D;
  const int kh_stride=layout==2 ? D : 64*D;
  const int vt=layout==0 ? 1 : kt,vc=layout==0 ? 64 : 1;
  constexpr float scale=(D==64 ? .125f : D==128 ? .0883883476f : D==192 ? .0721687836f : .0625f)*1.4426950408889634f;
  for(int t=warp;t<n;t+=4) {
    const int physical=table[b*table_stride+t/64];
    const uint8_t* kp=k+static_cast<int64_t>(physical)*kv_stride+kh*kh_stride+(t%64)*kt;
    float sum=0.f;
    for(int d=lane;d<D;d+=64) sum+=sq[d]*uint82float(kp[d]);
    #pragma unroll
    for(int step=32;step;step/=2) sum+=__shfl_xor(sum,step);
    if(lane==0) prob[t]=sum*scale;
  }
  __syncthreads();
  float maximum=-INFINITY;
  for(int t=tid;t<n;t+=256) maximum=fmaxf(maximum,prob[t]);
  #pragma unroll
  for(int step=32;step;step/=2) maximum=fmaxf(maximum,__shfl_xor(maximum,step));
  if(lane==0) scratch[warp]=maximum;
  __syncthreads();
  maximum=fmaxf(fmaxf(scratch[0],scratch[1]),fmaxf(scratch[2],scratch[3]));
  float sum=0.f;
  for(int t=tid;t<n;t+=256) {float p=__builtin_amdgcn_exp2f(prob[t]-maximum);prob[t]=p;sum+=p;}
  #pragma unroll
  for(int step=32;step;step/=2) sum+=__shfl_xor(sum,step);
  if(lane==0) scratch[warp]=sum;
  __syncthreads();
  const float inverse=1.f/(scratch[0]+scratch[1]+scratch[2]+scratch[3]+1e-6f);
  if(tid<D) {
    float result=0.f;
    for(int t=0;t<n;++t) {
      const int physical=table[b*table_stride+t/64];
      const uint8_t* vp=v+static_cast<int64_t>(physical)*kv_stride+kh*kh_stride+(t%64)*vt+tid*vc;
      result+=prob[t]*uint82float(*vp);
    }
    from_float(out[(b*heads+h)*D+tid],result*inverse);
  }
#endif
}

// Dispatch only the window/compatibility call in paged_attention.cu. Its native
// MTP path returns before this entry. gfx936/gfx92a forward to the original
// function, preserving their existing dispatch, arithmetic and workspaces.
static inline void pa938_window_dispatch(
    torch::Tensor& out, torch::Tensor& query_in, torch::Tensor& key,
    torch::Tensor& value, torch::Tensor& table, torch::Tensor& lengths,
    const c10::optional<torch::Tensor>& alibi,
    const c10::optional<at::Tensor>& sink, float* workspace,
    int layout, int left, int right, const Pa28WindowPlan& plan)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
    const auto aligned16 = [](const torch::Tensor& t) {
        return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    const bool is938 = device_name == "gfx938" &&
        std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName).substr(0,6) == "gfx938";
    if (!is938) {
        paged_attention_window_opt(out,query_in,key,value,table,lengths,
                                  alibi,sink,workspace,layout,left,right,plan);
        return;
    }
    const int d = query_in.size(3), mtp = query_in.size(1);
    const int batch = query_in.size(0), heads = query_in.size(2) * mtp;
    const int kvheads = key.size(layout == 2 ? 2 : 1);
    const int page = key.size(layout == 2 ? 1 : 2);
    const int dv = value.size(layout == 0 ? 2 : 3);
    const int reuse = kvheads > 0 ? heads / kvheads : 0;
    const int part = plan.partition_size, parts = plan.num_partitions;
    const bool e4m3 = key.scalar_type() == torch::kFloat8_e4m3fn;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    TORCH_CHECK(hipStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture) == hipSuccess,
                "PA938: cannot query capture status");
    // Preserve the online/compacted route where the dense mapping loses:
    // high-reuse 16-bit BHSD/BSHD and very large Graph capacity grids.
    // Dense arithmetic with device-side active-task scheduling avoids both
    // the online kernel's long-context cost and empty capacity partitions.
    const bool thin_bshd = layout == 2 && batch <= 2 && kvheads == 1 && reuse <= 8;
    const bool small_dense = !thin_bshd && !e4m3 && layout != 0 && batch <= 2 &&
        reuse <= 16 && left < 0 && parts >= 256 && PA_GRAPH_COMPACT != 0;
    const bool dense_graph = !thin_bshd && !e4m3 && layout != 0 &&
        (reuse <= 8 || (batch <= 2 && reuse <= 16 && left < 0)) &&
        (batch <= 2 || layout == 1) &&
        capture != hipStreamCaptureStatusNone && parts >= 256 && PA_GRAPH_COMPACT != 0;
    // Small batches favor the straight dense grid; the device scheduler pays
    // off for BHSD when more sequences amplify empty capacity tasks.
    const bool compact_dense = dense_graph && batch > 2;
    const bool prefer_window = layout != 0 &&
        ((!e4m3 && reuse > 8 && left < 0 && !small_dense) ||
         (capture != hipStreamCaptureStatusNone && parts >= 256 && !dense_graph));
    const int shared = part * 2 * reuse + (reuse * 6 + 4) * sizeof(float);
    const bool eligible = is938 && !PA_REFACTORING_DISABLE && !prefer_window &&
        layout >= 0 && layout <= 2 && page == 64 &&
        d == dv && d >= 64 && d <= 256 && d % 64 == 0 &&
        batch > 0 && mtp > 0 && kvheads > 0 && query_in.size(2) % kvheads == 0 &&
        reuse > 0 && reuse <= 64 && left >= -1 && right >= -1 &&
        (part == 256 || part == 512) && parts > 0 &&
        shared <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock &&
        out.scalar_type() == query_in.scalar_type() &&
        (out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat16) &&
        key.scalar_type() == value.scalar_type() &&
        (key.scalar_type() == out.scalar_type() || e4m3 || key.scalar_type() == torch::kFloat8_e5m2) &&
        key.is_contiguous() && value.is_contiguous() && out.is_contiguous() &&
        aligned16(key) && aligned16(value) && aligned16(out) && !alibi && !sink;
    if (eligible) {
        auto query = query_in.is_contiguous() && aligned16(query_in) ? query_in : query_in.clone(at::MemoryFormat::Contiguous);
        TORCH_CHECK(parts == 1 || workspace != nullptr, "PA938: missing partition workspace");
        const auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t capacity_tasks=static_cast<int64_t>(kvheads)*batch*parts;
        const int workers=pa28_graph_workers(PA_GRAPH_WORKERS,"gfx938",
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
        const dim3 grid=compact_dense ?
            dim3(static_cast<unsigned>(std::min<int64_t>(workers,capacity_tasks)),1,1) :
            dim3(kvheads,batch,parts);
        if (PA_PRINT_PARAM) printf("PA938_DECODE layout=%d fp8_mmac=%d parts=%d partition_size=%d rows=%d\n",
            layout, e4m3, parts, part, reuse);
        const int reuse_tag = get_reusekv(heads, kvheads);
        HEADSIZE_SWITCH(d, [&] {
          Output_Type_SWITCH(out.scalar_type(), [&] {
            REUSEKV_SWITCH(reuse_tag, [&] {
              bool fp8_finalized=false;
              if (e4m3) {
                auto launch = [&](auto tag) {
                  constexpr int L = decltype(tag)::value;
                  if constexpr(REUSE_KV_TIMES>=24) {
                    constexpr int P=REUSE_KV_TIMES==24 ? 512 : 256;
                    if(part==P && left<0 && mtp==1 && parts>=16) {
                      const int fast_shared=std::max(P,HEAD_SIZE*2)*reuse+(reuse*6+4)*sizeof(float);
                      pa938_decode_fp8_lds<scalar_t,scalar_t,true,HEAD_SIZE,64,256,REUSE_KV_TIMES,P,L>
                        <<<grid,256,fast_shared,stream>>>(
                          static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                          static_cast<const scalar_t*>(query.data_ptr()),
                          static_cast<const uint8_t*>(key.data_ptr()),static_cast<const uint8_t*>(value.data_ptr()),
                          heads,kvheads,table.data_ptr<int>(),lengths.data_ptr<int>(),table.size(1),nullptr,
                          query.stride(0),key.stride(0),nullptr,nullptr,nullptr,parts,
                          nullptr,mtp,false,left,right,left);
                      const int merge_shared=std::max(8*parts,(72+HEAD_SIZE+4)*int(sizeof(float)));
                      pa938_fp8_finalize<scalar_t,HEAD_SIZE>
                        <<<dim3(heads,batch),256,merge_shared,stream>>>(
                          static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                          static_cast<const scalar_t*>(query.data_ptr()),static_cast<const uint8_t*>(key.data_ptr()),
                          static_cast<const uint8_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                          lengths.data_ptr<int>(),table.size(1),query.stride(0),key.stride(0),parts,part,
                          HEAD_SIZE<=192 ? 2048 : 512,L);
                      fp8_finalized=true;
                      return;
                    }
                  }
                  pa938_decode_fp8<scalar_t, scalar_t, true, HEAD_SIZE, 64, 256, REUSE_KV_TIMES, L>
                    <<<grid, 256, shared, stream>>>(
                      static_cast<scalar_t*>(out.data_ptr()), reinterpret_cast<scalar_t*>(workspace),
                      static_cast<const scalar_t*>(query.data_ptr()),
                      static_cast<const uint8_t*>(key.data_ptr()), static_cast<const uint8_t*>(value.data_ptr()),
                      heads, kvheads, table.data_ptr<int>(), lengths.data_ptr<int>(), table.size(1), nullptr,
                      query.stride(0), key.stride(0), nullptr, nullptr, nullptr, parts, part,
                      nullptr, mtp, false, left, right, left);
                  // Preserve V3's empty-row clearing when a small workspace
                  // selects the original precise producer.  Its
                  // partition width stays P, so disable the 128-token merge.
                  if constexpr(REUSE_KV_TIMES>=24) {
                    constexpr int P=REUSE_KV_TIMES==24 ? 512 : 256;
                    if(left<0 && mtp==1) {
                      pa938_fp8_finalize<scalar_t,HEAD_SIZE>
                        <<<dim3(heads,batch),256,std::max(8*parts,(72+HEAD_SIZE+4)*int(sizeof(float))),stream>>>(
                          static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                          static_cast<const scalar_t*>(query.data_ptr()),static_cast<const uint8_t*>(key.data_ptr()),
                          static_cast<const uint8_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                          lengths.data_ptr<int>(),table.size(1),query.stride(0),key.stride(0),parts,part,-1,L);
                      fp8_finalized=true;
                    }
                  }
                };
                if (layout == 0) launch(std::integral_constant<int,0>{});
                else if (layout == 1) launch(std::integral_constant<int,1>{});
                else launch(std::integral_constant<int,2>{});
              } else {
                Cache_Type_SWITCH(scalar_t, key.scalar_type(), [&] {
                  if constexpr(REUSE_KV_TIMES<=8) {
                    if(compact_dense) {
                      auto persistent=[&](auto small_tag) {
                        constexpr bool SMALL=decltype(small_tag)::value;
                        if(layout==1) {
                          pa936_legacy_kernel_bhsd<scalar_t,cache_t,false,HEAD_SIZE,64,256,REUSE_KV_TIMES,true,SMALL>
                            <<<grid,256,shared,stream>>>(
                              static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                              static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                              static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),lengths.data_ptr<int>(),
                              table.size(1),nullptr,query.stride(0),key.stride(0),parts,part,nullptr,mtp,false,left,right,left,batch);
                        } else {
                          pa936_legacy_kernel_bshd<scalar_t,cache_t,false,HEAD_SIZE,HEAD_SIZE,64,256,REUSE_KV_TIMES,true,SMALL>
                            <<<grid,256,shared,stream>>>(
                              static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                              static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                              static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),lengths.data_ptr<int>(),
                              table.size(1),nullptr,query.stride(0),key.stride(0),value.stride(0),parts,part,nullptr,mtp,false,left,right,left,batch);
                        }
                      };
                      if(batch<=2) persistent(std::true_type{});
                      else persistent(std::false_type{});
                      return;
                    }
                  }
                  if (layout == 0) {
                    pa936_dense_layout0<scalar_t,cache_t,false,HEAD_SIZE,64,256,REUSE_KV_TIMES>
                      <<<grid,256,shared,stream>>>(
                        static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                        static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                        static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                        lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),parts,part,
                        nullptr,mtp,false,left,right,left);
                  } else if (layout == 1) {
                    pa936_dense_layout1<scalar_t,cache_t,false,HEAD_SIZE,64,256,REUSE_KV_TIMES>
                      <<<grid,256,shared,stream>>>(
                        static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                        static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                        static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                        lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),parts,part,
                        nullptr,mtp,false,left,right,left);
                  } else {
                    pa936_dense_layout2<scalar_t,cache_t,false,HEAD_SIZE,HEAD_SIZE,64,256,REUSE_KV_TIMES>
                      <<<grid,256,shared,stream>>>(
                        static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                        static_cast<const scalar_t*>(query.data_ptr()),static_cast<const cache_t*>(key.data_ptr()),
                        static_cast<const cache_t*>(value.data_ptr()),heads,kvheads,table.data_ptr<int>(),
                        lengths.data_ptr<int>(),table.size(1),nullptr,query.stride(0),key.stride(0),value.stride(0),
                        parts,part,nullptr,mtp,false,left,right,left);
                  }
                });
              }
              if (parts > 1 && !fp8_finalized) paged_attention_combine<scalar_t,HEAD_SIZE,64>
                <<<dim3(heads,batch),64,8*parts,stream>>>(
                  static_cast<scalar_t*>(out.data_ptr()),reinterpret_cast<scalar_t*>(workspace),
                  lengths.data_ptr<int>(),parts,heads,part,left,mtp,64);
            });
          });
        });
        TORCH_CHECK(hipGetLastError() == hipSuccess, "PA938: decode kernel launch failed");
        return;
    }
#endif
    paged_attention_window_opt(out,query_in,key,value,table,lengths,
                              alibi,sink,workspace,layout,left,right,plan);
}

// This header is included only by paged_attention.cu, after the original
// window entry declaration. Keep the integration in this one requested file:
// the alias redirects that TU's call, never the definition in the BSHD TU.
#define paged_attention_window_opt pa938_window_dispatch
