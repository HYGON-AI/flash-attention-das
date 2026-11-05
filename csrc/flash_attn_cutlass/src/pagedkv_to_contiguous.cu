// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "paged_attention.h"

#define Type_SWITCH(in_type,out_type, ...)                                         \
  [&] {                                                                            \
    if(in_type==torch::kFloat8_e5m2&&out_type==torch::kFloat8_e5m2){               \
      using cache_t = uint8_t;                                                     \
      using scalar_t = uint8_t;                                                    \
      constexpr bool is_e4m3=false;                                                \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat8_e4m3fn&&out_type==torch::kFloat8_e4m3fn){      \
      using cache_t = uint8_t;                                                     \
      using scalar_t = uint8_t;                                                    \
      constexpr bool is_e4m3 = true;                                               \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat8_e5m2&&out_type==torch::kFloat16){              \
      using cache_t = uint8_t;                                                     \
      using scalar_t = _Float16;                                                   \
      constexpr bool is_e4m3 = false;                                              \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat8_e5m2&&out_type==torch::kBFloat16){             \
      using cache_t = uint8_t;                                                     \
      using scalar_t = uint16_t;                                                   \
      constexpr bool is_e4m3 = false;                                              \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat8_e4m3fn&&out_type==torch::kFloat16){            \
      using cache_t = uint8_t;                                                     \
      using scalar_t = _Float16;                                                   \
      constexpr bool is_e4m3 = true;                                               \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat8_e4m3fn&&out_type==torch::kBFloat16){           \
      using cache_t = uint8_t;                                                     \
      using scalar_t = uint16_t;                                                   \
      constexpr bool is_e4m3 = true;                                               \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else if(in_type==torch::kFloat16&&out_type==torch::kFloat16){                  \
      using cache_t = _Float16;                                                    \
      using scalar_t = _Float16;                                                   \
      constexpr bool is_e4m3 = false;                                              \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    else{                                                                          \
      using cache_t = uint16_t;                                                    \
      using scalar_t = uint16_t;                                                   \
      constexpr bool is_e4m3 = false;                                              \
      return __VA_ARGS__();                                                        \
    }                                                                              \
  }()

static constexpr int kPagedKvToContiguousPartitionTokens = 512;

template <typename scalar_t, typename cache_t,int vec_size, bool is_e4m3>
inline __device__ void write_one_block_bhsd(int lane_id,int kv_head_stride,int head_size,int block_idx,
                                int seq_len,int num_head,const cache_t* cache_ptr,const scalar_t* out_ptr,int PAGE_BLOCK_SIZE)
{
  using cache_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(cache_t)) )) cache_t;
  using out_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  constexpr bool io_same = std::is_same<scalar_t, cache_t>::value;
  for(int i=lane_id*vec_size;i<kv_head_stride;i+=WARP_SIZE*vec_size){
    int token_idx=block_idx * PAGE_BLOCK_SIZE+i/head_size;
    if(token_idx<seq_len){
      cache_vec_t kv_vec=*(cache_vec_t*)(cache_ptr+i);
      if constexpr(io_same){
        *(cache_vec_t*)(out_ptr+token_idx*num_head*head_size+i%head_size)=kv_vec;
      }
      else{
        out_vec_t kv_vec_o;
        for(int ii=0;ii<vec_size;ii++){
          kv_vec_o[ii]=uint82half<scalar_t,is_e4m3>(kv_vec[ii]);
        }
        *(out_vec_t*)(out_ptr+token_idx*num_head*head_size+i%head_size)=kv_vec_o;
      }
    }
  }
}

template < typename scalar_t, typename cache_t,int vec_size, bool is_e4m3>
inline __device__ void write_one_block_bshd(int lane_id,int head_size,int block_idx,
                                int seq_len,int num_head,const cache_t* cache_ptr,const scalar_t* out_ptr,int PAGE_BLOCK_SIZE)
{
  using cache_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(cache_t)) )) cache_t;
  using out_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  constexpr bool io_same = std::is_same<scalar_t, cache_t>::value;
  int total = head_size*PAGE_BLOCK_SIZE*num_head;
  int thread_per_dim = head_size/vec_size;
  int lane_id_offset = (lane_id/thread_per_dim)*num_head*head_size + (lane_id%thread_per_dim)*vec_size;
  for(int i=lane_id_offset;i<total;i+=WARP_SIZE/thread_per_dim*num_head*head_size){
    int token_idx = block_idx * PAGE_BLOCK_SIZE + i/(num_head*head_size);
    if(token_idx<seq_len){
      cache_vec_t kv_vec=*(cache_vec_t*)(cache_ptr+i);
      if constexpr(io_same){
        *(cache_vec_t*)(out_ptr+i)=kv_vec;
      }
      else{
        out_vec_t kv_vec_o;
        for(int ii=0;ii<vec_size;ii++){
          kv_vec_o[ii]=uint82half<scalar_t,is_e4m3>(kv_vec[ii]);
        }
        *(out_vec_t*)(out_ptr+i)=kv_vec_o;
      }
    }
  }
}

template <typename scalar_t, typename cache_t,bool is_e4m3,int NUM_THREADS>
__launch_bounds__(NUM_THREADS) __global__ void  pagedkv_to_contiguouskv_kernel(
  scalar_t* k_ptr_out,
  scalar_t* v_ptr_out,
  cache_t* k_cache_ptr,
  cache_t* v_cache_ptr,
  int *block_tables_ptr,
  int *seq_lens_ptr,
  int *cu_seqlens_k_ptr,
  int head_size,
  int head_size_v,
  int num_head,
  int layout,
  int max_num_blocks_per_seq,
  int kv_block_stride,
  int v_block_stride,
  int PAGE_BLOCK_SIZE
  )
{
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens_ptr[seq_idx]);
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, kPagedKvToContiguousPartitionTokens);
  if(num_partitions<=partition_idx)return ;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __builtin_amdgcn_readfirstlane(thread_idx / WARP_SIZE);
  const int lane_id = thread_idx%64;
  constexpr int NUM_WARPS = NUM_THREADS/WARP_SIZE;
  constexpr int vec_size = 16/sizeof(cache_t);
  __shared__ cache_t shared_mem[NUM_THREADS*vec_size];
  int seq_offset=0;
  for(int i=thread_idx;i<seq_idx;i+=NUM_THREADS){
    seq_offset+=seq_lens_ptr[i];
  }
  int start_offset = __builtin_amdgcn_readfirstlane(block_sum<int,NUM_WARPS>((int*)shared_mem,seq_offset));
  //write cu_seqlens_k
  if(thread_idx==0&&partition_idx==0&&head_idx==0){
    cu_seqlens_k_ptr[seq_idx+1]=start_offset+seq_len;
    if(seq_idx==0)cu_seqlens_k_ptr[0]=0;
  }
  __syncthreads();
  cache_t* warp_shared = shared_mem+vec_size*WARP_SIZE*warp_idx;
  constexpr bool io_same = std::is_same<scalar_t, cache_t>::value;
  const int* block_table = block_tables_ptr + seq_idx * max_num_blocks_per_seq;
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, PAGE_BLOCK_SIZE);
  const int start_block_idx = partition_idx * kPagedKvToContiguousPartitionTokens / PAGE_BLOCK_SIZE;
  const int end_block_idx =MIN(start_block_idx + kPagedKvToContiguousPartitionTokens / PAGE_BLOCK_SIZE, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  const int start_token_idx = start_block_idx * PAGE_BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * PAGE_BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  int kv_head_stride;
  if(layout!=2)kv_head_stride=PAGE_BLOCK_SIZE*head_size;
  else kv_head_stride=head_size;
  int v_head_stride = layout!=2 ? PAGE_BLOCK_SIZE*head_size_v : head_size_v;
  const cache_t* k_ptr_base = k_cache_ptr + head_idx * kv_head_stride;
  const cache_t* v_ptr_base = v_cache_ptr + head_idx * v_head_stride;
  scalar_t* k_out = k_ptr_out+(start_offset*num_head+head_idx)*head_size;
  scalar_t* v_out = v_ptr_out+(start_offset*num_head+head_idx)*head_size_v;
  using cache_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(cache_t)) )) cache_t;
  using out_vec_t= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  if(layout==0){
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      const cache_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride;
      const cache_t* v_ptr=v_ptr_base + physical_block_number * v_block_stride;
      write_one_block_bhsd<scalar_t,cache_t,vec_size,is_e4m3>(lane_id,kv_head_stride,head_size,block_idx,
                                                              seq_len,num_head,k_ptr,k_out,PAGE_BLOCK_SIZE);
      if (PAGE_BLOCK_SIZE == 64){
        int dim=0;
        for(int i=lane_id*vec_size;i<v_head_stride;i+=WARP_SIZE*vec_size){
          int token_idx=block_idx * PAGE_BLOCK_SIZE+lane_id;
          *(cache_vec_t*)(warp_shared+lane_id*vec_size)=*(cache_vec_t*)(v_ptr+i);
          if(token_idx<seq_len){
            cache_vec_t v_vec;
            #pragma unroll
            for(int ii=0;ii<vec_size;ii++){
              v_vec[ii] = warp_shared[lane_id+ii*PAGE_BLOCK_SIZE];
            }
            if constexpr(io_same){
              *(cache_vec_t*)(v_out+token_idx*num_head*head_size_v+dim)=v_vec;
            }
            else{
              out_vec_t v_vec_o;
              for(int ii=0;ii<vec_size;ii++){
                v_vec_o[ii]=uint82half<scalar_t,is_e4m3>(v_vec[ii]);
              }
              *(out_vec_t*)(v_out+token_idx*num_head*head_size_v+dim)=v_vec_o;
            }
          }
          dim+=vec_size;
        }
      }
      else{
        // The custom V cache is [head_dim, page].  Transpose page128 as
        // two independent vec_size-dimension x 64-token wave tiles so each global
        // load and packed-output store remains 16-byte vectorized.
        constexpr int WAVE_TILE_TOKENS = WARP_SIZE;
        int TOKEN_TILES = PAGE_BLOCK_SIZE / WAVE_TILE_TOKENS;
        for(int dim=0;dim<head_size_v;dim+=vec_size){
          for(int token_tile=0;token_tile<TOKEN_TILES;++token_tile){
            int wave_offset=lane_id*vec_size;
            int load_dim=dim+wave_offset/WAVE_TILE_TOKENS;
            int load_token=token_tile*WAVE_TILE_TOKENS+wave_offset%WAVE_TILE_TOKENS;
            int load_offset=load_dim*PAGE_BLOCK_SIZE+load_token;
            *(cache_vec_t*)(warp_shared+wave_offset)=*(cache_vec_t*)(v_ptr+load_offset);
            // This is only a compiler scheduling barrier.  No CTA-wide
            // synchronization is needed because every wave owns a disjoint
            // LDS slice and executes this transpose tile in wave lockstep.
            __builtin_amdgcn_sched_barrier(0);
            int token_idx=block_idx*PAGE_BLOCK_SIZE+token_tile*WAVE_TILE_TOKENS+lane_id;
            if(token_idx<seq_len){
              cache_vec_t v_vec;
              #pragma unroll
              for(int ii=0;ii<vec_size;ii++){
                v_vec[ii]=warp_shared[lane_id+ii*WAVE_TILE_TOKENS];
              }
              if constexpr(io_same){
                *(cache_vec_t*)(v_out+token_idx*num_head*head_size_v+dim)=v_vec;
              }
              else{
                out_vec_t v_vec_o;
                #pragma unroll
                for(int ii=0;ii<vec_size;ii++){
                  v_vec_o[ii]=uint82half<scalar_t,is_e4m3>(v_vec[ii]);
                }
                *(out_vec_t*)(v_out+token_idx*num_head*head_size_v+dim)=v_vec_o;
              }
            }
            // Prevent the compiler from moving the next tile's LDS writes
            // ahead of the current tile's reads; this is not an s_barrier.
            __builtin_amdgcn_sched_barrier(0);
          }
        }
      }
    }
  }
  else if(layout==1){
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      const cache_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride;
      const cache_t* v_ptr=v_ptr_base + physical_block_number * v_block_stride;
      write_one_block_bhsd<scalar_t,cache_t,vec_size,is_e4m3>(lane_id,kv_head_stride,head_size,block_idx,
                                                              seq_len,num_head,k_ptr,k_out,PAGE_BLOCK_SIZE);
      write_one_block_bhsd<scalar_t,cache_t,vec_size,is_e4m3>(lane_id,v_head_stride,head_size_v,block_idx,
                                                        seq_len,num_head,v_ptr,v_out,PAGE_BLOCK_SIZE);
    }
  }
  else if(layout==2){
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;block_idx += NUM_WARPS) {
      const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
      const cache_t* k_ptr=k_ptr_base + physical_block_number * kv_block_stride;
      const cache_t* v_ptr=v_ptr_base + physical_block_number * v_block_stride;
      int out_offset = block_idx * PAGE_BLOCK_SIZE * num_head * head_size;
      int v_out_offset = block_idx * PAGE_BLOCK_SIZE * num_head * head_size_v;
      write_one_block_bshd<scalar_t,cache_t,vec_size,is_e4m3>(lane_id,head_size,block_idx,
                                                              seq_len,num_head,k_ptr,k_out+out_offset,PAGE_BLOCK_SIZE);
      write_one_block_bshd<scalar_t,cache_t,vec_size,is_e4m3>(lane_id,head_size_v,block_idx,
                                                        seq_len,num_head,v_ptr,v_out+v_out_offset,PAGE_BLOCK_SIZE);
    }
  }
}
extern "C"
void pagedkv_to_contiguouskv(
    torch::Tensor& k,             // [total_tokens, num_heads, head_size]
    torch::Tensor& v,             // [total_tokens, num_heads, head_size]
    torch::Tensor& key_cache,     // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    torch::Tensor& cu_seqlens_k,   // [num_seqs+1]
    int max_seq_len,
    int layout)
{
  int num_seqs = seq_lens.size(0);
  int head_size=key_cache.size(-1);
  int block_size=key_cache.size(2);
  
  int num_blocks = key_cache.size(0);
  int max_num_blocks_per_seq = block_tables.size(1);
  int num_kv_heads = key_cache.size(1);
  if(layout<0||layout>2)layout=0;
  int head_size_v = layout==0 ? value_cache.size(-2) : value_cache.size(-1);
  if(layout==2){
    block_size = num_kv_heads;
    num_kv_heads = key_cache.size(2);
  }
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  int* cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int>();

  auto* key_cache_ptr = key_cache.data_ptr();
  auto* value_cache_ptr = value_cache.data_ptr();
  auto* k_ptr = k.data_ptr();
  auto* v_ptr = v.data_ptr();
  int kv_block_stride = key_cache.stride(0);
  int v_block_stride = value_cache.stride(0);
  constexpr static int NUM_THREADS = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if(max_seq_len>max_num_blocks_per_seq*block_size)max_seq_len=max_num_blocks_per_seq*block_size;
  int max_partitions = DIVIDE_ROUND_UP(max_seq_len,kPagedKvToContiguousPartitionTokens);
  AT_ASSERTM(block_size==32||block_size==64||block_size==128, "Only support block_size 64 or 128");
  AT_ASSERTM(head_size%64==0, "Only support head_size divisible by 64");
  Type_SWITCH(key_cache.dtype(),k.dtype(),[&]{
    pagedkv_to_contiguouskv_kernel<scalar_t,cache_t,is_e4m3,NUM_THREADS><<<dim3(num_kv_heads,num_seqs,max_partitions),NUM_THREADS,0,stream>>>(
      (scalar_t*)k_ptr,(scalar_t*)v_ptr,(cache_t*)key_cache_ptr,(cache_t*)value_cache_ptr,block_tables_ptr,seq_lens_ptr,cu_seqlens_k_ptr,
      head_size,head_size_v,num_kv_heads,layout,max_num_blocks_per_seq,kv_block_stride,v_block_stride,block_size
    );
  });
}
