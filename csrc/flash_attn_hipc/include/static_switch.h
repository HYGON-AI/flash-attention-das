// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once
#include <cassert>
#include "numeric_types.h"
#include "splitkv.h"

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = Float16;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = BFloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define LAYOUT_SWITCH(layout, ...)         \
  [&] {                                    \
    if (layout == 0) {                     \
      constexpr static int Layout = 0;     \
      return __VA_ARGS__();                \
    } else if (layout == 1) {              \
      constexpr static int Layout = 1;     \
      return __VA_ARGS__();                \
    }                                      \
  }()

// #define ElementType_SWITCH(is_bf16, is_e4m3, ...)   \
//   [&] {                                      \
//     if (is_bf16) {                           \
//       using elem_type = BFloat16;            \
//       return __VA_ARGS__();                  \
//     } else if (is_e4m3) {                    \
//       using elem_type = Float8_e4m3_t;       \
//       return __VA_ARGS__();                  \
//     } else {                                 \
//       using elem_type = Float16;             \
//       return __VA_ARGS__();                  \
//     }                                        \
//   }()

#define ElementType_SWITCH(is_bf16, is_e4m3, ...)   \
[&] {                                      \
  if (is_bf16) {                           \
    using elem_type = BFloat16;            \
    return __VA_ARGS__();                  \
  } else if (is_e4m3) {                    \
    printf("fa bwd does not support fp8 yet");\
  } else {                                 \
    using elem_type = Float16;             \
    return __VA_ARGS__();                  \
  }                                        \
}()

#define HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 256) {             \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define ALL_HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    }  else if (HEADDIMQ <= 256) {            \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define M_MMAC_COUNT_SWITCH(COND, M_MMAC_COUNT, ...)    \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static int M_MMAC_COUNT = 2;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int M_MMAC_COUNT = 1;    \
      return __VA_ARGS__();                     \
    }                                           \
  }()


#define REUSEKV_SWITCH(seqlen_q , ...)                  \
[&] {                                                   \
    if (seqlen_q == 16) {                               \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 8) {                         \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    } else if (seqlen_q == 4) {                         \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    } else{                                             \
        constexpr static int REUSE_KV_TIMES = 0;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()



#define PA_MTP_REUSEKV_SWITCH(pa_mtp , ...)             \
[&] {                                                   \
    if (pa_mtp == 8) {                                  \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 16) {                          \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 24) {                          \
        constexpr static int REUSE_KV_TIMES = 24;       \
        return __VA_ARGS__();                           \
    } else if (pa_mtp == 32) {                          \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    } else {                                            \
        constexpr static int REUSE_KV_TIMES = 0;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()


#define PERMUTE_DWORD_SWITCH(COND, DWORD_PER_TX, ...)    \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static int DWORD_PER_TX = 4;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int DWORD_PER_TX = 1;    \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define PERMUTE_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM == 64) {                   \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM < 256) {            \
      constexpr static int kHeadDim = 0;   \
      return __VA_ARGS__();                \
    }  else if (HEADDIM == 256) {          \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()


// PA 实际编译进来的 head dim 组合 (见 src/target/*split_hdim*.cpp); HEADDIM_*_ONLY 构建只含子集。
#if defined(HEADDIM_128_ONLY)
  #define PA_HEADDIM_SUPPORTED(HEADDIMQ, HEADDIMV) \
    ((HEADDIMQ) == 128 && (HEADDIMV) == 128)
#elif defined(HEADDIM_192_128_ONLY)
  #define PA_HEADDIM_SUPPORTED(HEADDIMQ, HEADDIMV) \
    ((HEADDIMQ) == 192 && (HEADDIMV) == 128)
#else
  #define PA_HEADDIM_SUPPORTED(HEADDIMQ, HEADDIMV)                      \
    (((HEADDIMQ) == 128 && (HEADDIMV) == 128) ||                        \
     ((HEADDIMQ) == 192 && ((HEADDIMV) == 128 || (HEADDIMV) == 192)) || \
     ((HEADDIMQ) == 256 && (HEADDIMV) == 256))
#endif


// 不支持的 head dim 直接报错, 否则会被映射到不匹配的 kHeadDim 模板或什么都不发射。
#define PA_HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)                           \
  [&] {                                                                      \
    if (not PA_HEADDIM_SUPPORTED(HEADDIMQ, HEADDIMV)) {                      \
      printf("\x1b[31mPagedAttention: unsupported head dim %d/%d, only "     \
             "128/128, 192/128, 192/192 and 256/256 are compiled!\033[0m\n", \
             (int)(HEADDIMQ), (int)(HEADDIMV));                              \
      assert(false && "unsupported head dimension for PagedAttention");      \
      return;                                                                \
    }                                                                        \
    if (HEADDIMQ <= 128) {                    \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 256) {             \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()


#define PA_PAGEBLOCKSIZE_SWITCH(page_block_size, ...)                      \
  [&] {                                                                    \
    if (page_block_size % 32 == 0 and page_block_size % 64 != 0) {         \
      constexpr static int kBlockN = 32;                                   \
      return __VA_ARGS__();                                                \
    } else if (page_block_size % 64 == 0 and page_block_size % 128 != 0) { \
      constexpr static int kBlockN = 64;                                   \
      return __VA_ARGS__();                                                \
    } else if (page_block_size % 128 == 0) {                               \
      constexpr static int kBlockN = 128;                                  \
      return __VA_ARGS__();                                                \
    }                                                                      \
  }()

