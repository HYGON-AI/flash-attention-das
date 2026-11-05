/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#pragma once

#include <hute/tensor.hpp>

namespace flash {

using namespace hute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                           const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // 计算块内线程位置
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + lane_id / 16;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 4;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j * stride_between_each_thread;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_continuous(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                           const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // 计算块内线程位置
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 1;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j * stride_between_each_thread;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + lane_id / 16;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 4;

    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        const int row_idx = row_idx_base;
        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j * stride_between_each_thread;
                if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local_continuous(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 1;

    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        const int row_idx = row_idx_base;
        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j * stride_between_each_thread;
                if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}
template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal_continuous(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local_continuous</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}



template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_trans(Tensor<Engine, Layout> &tensor, const int max_seqlen_q,
                                           const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // 计算块内线程位置
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 1;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j * stride_between_each_thread;
            if (col_idx >= max_seqlen_q) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local_trans(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    

    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    // static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // const int lane_id = threadIdx.x % 64;
    // const int col_idx_offset = col_idx_offset_ + lane_id / 16;
    // const int stride_between_each_repeat = 16;
    // const int stride_between_each_thread = 4;
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 64;
    const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
    const int stride_between_each_repeat = 16;
    const int stride_between_each_thread = 1;

    if constexpr (HasWSLeft) {
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j * stride_between_each_thread;
                const int row_idx_limit_up = std::max(0, col_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                const int row_idx_limit_down = std::min(max_seqlen_k, col_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    const int row_idx = row_idx_base;
                    // int tidx = threadIdx.x;
                    // if (tidx < 64)
                    // {
                    //     printf("col_idx = %d row_idx_limit_up = %d row_idx_limit_down = %d\n", col_idx, row_idx_limit_up, row_idx_limit_down);
                    // }
                    if (row_idx < row_idx_limit_up || row_idx >= row_idx_limit_down) {
                        tensor(mi, make_coord(j, nj)) = -INFINITY;
                    }
                }
            
            }
        }
    }
    else {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            const int row_idx = row_idx_base;
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_q - max_seqlen_k - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_q - max_seqlen_k + window_size_right);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                #pragma unroll  
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j * stride_between_each_thread;
                    
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                    //     printf("tid = %d col_idx_limit_left = %d col_idx_limit_right = %d col_idx = %d row_idx = %d max_seqlen_k = %d max_seqlen_q = %d\n", threadIdx.x, col_idx_limit_left, col_idx_limit_right, col_idx, row_idx, 
                    //         max_seqlen_k, max_seqlen_q);
                    // }
                    
                    if (col_idx + 1 < col_idx_limit_left) {
                        tensor(mi, make_coord(j, nj)) = -INFINITY;
                    }
                    // if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                    //     tensor(mi, make_coord(j, nj)) = -INFINITY;
                    // }
                }
            }
        }
    }
}


template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal_trans(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local_trans</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}


template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            /*
            查看acc的指令格式
            */
            // 0_15 = 0 16_31 = 1 32_47=2 48~63=4
            const int lane_id = threadIdx.x & 63;
            const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
            const int stride_between_each_repeat = 16;
            const int stride_between_each_thread = 4;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    // 沿着N方向重复，间隔为16
                    const int col_idx_base = col_idx_offset + (nj << 4);
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        /*
                        每个线程4个元素，其间隔为4
                        因为格式是
                        t0 t16 t32 t48 | t0 t16 t32 t48
                        */
                        const int col_idx = col_idx_base + (j << 2) ;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    const int row_idx = row_idx_offset + mi * warp_row_stride;
                    const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                    const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + (nj << 4);
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            // t0的第0个元素与t0的第1个元素间隔4
                            const int col_idx = col_idx_base + (j << 2);
                            if constexpr (Has_alibi) {
                                if constexpr (Is_causal) {
                                    tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                                } else {
                                    tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                }
                            }

                            if constexpr (Causal_mask) {
                                if (col_idx >= col_idx_limit_right) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                                // else {
                                //     if constexpr (!Has_alibi && !Is_local) {
                                //         return;
                                //     }
                                // }
                            }
                            if constexpr (Is_local) {
                                if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                            // #if 1
                            // if (cute::thread0())
                            // {
                            //     printf("in mask Is_even_MN = %d\n", Is_even_MN);
                            // }
                            // #enfif
                            // if causal情况下mn也不是整数
                            if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                // Causal and Local already handles MN masking
                                if (col_idx >= max_seqlen_k) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask_continuous(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            /*
            查看acc的指令格式
            */
            // 0_15 = 0 16_31 = 4 32_47=8 48~63=12
            const int lane_id = threadIdx.x % 64;
            const int col_idx_offset = col_idx_offset_ + ((lane_id >> 4) << 2);
            const int stride_between_each_repeat = 16;
            const int stride_between_each_thread = 4;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    // 沿着N方向重复，间隔为16
                    const int col_idx_base = col_idx_offset + (nj << 4);
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        /* 每个线程4个元素，其间隔为1
                        t0 t1 t2 t3 | t4 t5 t6 t7 */
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    const int row_idx = row_idx_offset + mi * warp_row_stride;
                    const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                    const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + (nj << 4);
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            // t0的第0个元素与t0的第1个元素间隔1
                            const int col_idx = col_idx_base + j;
                            if constexpr (Has_alibi) {
                                if constexpr (Is_causal) {
                                    tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                                } else {
                                    tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                }
                            }

                            if constexpr (Causal_mask) {
                                if (col_idx >= col_idx_limit_right) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                                // else {
                                //     if constexpr (!Has_alibi && !Is_local) {
                                //         return;
                                //     }
                                // }
                            }
                            if constexpr (Is_local) {
                                if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                            // #if 1
                            // if (cute::thread0())
                            // {
                            //     printf("in mask Is_even_MN = %d\n", Is_even_MN);
                            // }
                            // #enfif
                            // if causal情况下mn也不是整数
                            if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                // Causal and Local already handles MN masking
                                if (col_idx >= max_seqlen_k) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            }
        }
    };
	
	
	 template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask_continuous_fp8(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 8, "First dimension must be 8");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            /*
            查看acc的指令格式
            */
            // 0_15 = 0 16_31 = 4 32_47=8 48~63=12
            const int lane_id = threadIdx.x % 64;
            const int col_idx_offset = col_idx_offset_ + ((lane_id >> 4) << 3);
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {//2
                    // 沿着N方向重复，间隔为16
                    const int col_idx_base = col_idx_offset + (nj << 5);
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {//8
                        /* 每个线程8个元素，其间隔为1
                        t0 t1 t2 t3 | t4 t5 t6 t7 */
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    const int row_idx = row_idx_offset + mi * warp_row_stride;
                    const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                    const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {//2
                        const int col_idx_base = col_idx_offset + (nj << 5);
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {//8
                            // t0的第0个元素与t0的第1个元素间隔1
                            const int col_idx = col_idx_base + j;
                            if constexpr (Has_alibi) {
                                if constexpr (Is_causal) {
                                    tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                                } else {
                                    tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                }
                            }

                            if constexpr (Causal_mask) {
                                if (col_idx >= col_idx_limit_right) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                                // else {
                                //     if constexpr (!Has_alibi && !Is_local) {
                                //         return;
                                //     }
                                // }
                            }
                            if constexpr (Is_local) {
                                if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                            // #if 1
                            // if (cute::thread0())
                            // {
                            //     printf("in mask Is_even_MN = %d\n", Is_even_MN);
                            // }
                            // #enfif
                            // if causal情况下mn也不是整数
                            if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                // Causal and Local already handles MN masking
                                if (col_idx >= max_seqlen_k) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            }
        }
    };


};

} // namespace flash
