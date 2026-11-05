# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)

project(flash-attention  VERSION 1.0.0)

if (DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
else()
    set(ROCM_PATH /opt/rocm)
endif()
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

# 用 HIP ? Thunk ?
option(USE_HIP "use hip" ON)
MESSAGE(STATUS "USE_HIP: ${USE_HIP}")

# 设置编译器路径
if(EXISTS "${ROCM_PATH}/aillvm")
    message(STATUS "Use compiler of aillvm")
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/aillvm/bin/clang++)
    set(CC_WARPPER aicc)
else()
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/llvm/bin/clang++)
    set(CC_WARPPER hipcc)
endif()

# 设置架构号
SET(GFX_VERSION "928" CACHE STRING "i.e. cmake .. -DGFX_VERSION=936")
if (GFX_VERSION MATCHES "(928|936|938|92a|9210|946)")
    message(STATUS "GFX_VERSION: ${GFX_VERSION}")
else()
    message(FATAL_ERROR "Unsupported arch: ${GFX_VERSION}, supported: 928, 936, 938, 92a, 9210, 946")
endif()
if (GFX_VERSION STREQUAL "9210")
    set(GFX_VERSION "92a")
endif()
string(REPLACE ";" "," target_part "${GFX_VERSION}")
string(REPLACE ";" ",gfx" offload_arch_part "${GFX_VERSION}")
set(offload_arch_part "--offload-arch=gfx${offload_arch_part}")
set(GFX_TARGET_STRING ${offload_arch_part} -DTARGET=${target_part})
if (GFX_VERSION MATCHES "(928|92a|946)")
    set(GFX_MAX_VGPRS 512)
elseif (GFX_VERSION MATCHES "(936|938)")
    set(GFX_MAX_VGPRS 768)
endif()
message(STATUS "GFX_MAX_VGPRS: ${GFX_MAX_VGPRS}")

# 如果编译多个目标, 则不编译出汇编
string(REPLACE ";" "" NO_SEMICOLON "${GFX_VERSION}")
option(BUILD_ASM "Build kernel asm" ON)
if(NOT "${GFX_VERSION}" STREQUAL "${NO_SEMICOLON}")
    set (BUILD_ASM OFF)
endif()
MESSAGE(STATUS "BUILD_ASM: ${BUILD_ASM}")

set(HIP_COMPILE_FLAGS ${HIP_COMPILE_FLAGS} ${GFX_TARGET_STRING})

# 设置当前激活的 Conda 环境路径
set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
if (CONDA_PREFIX_PATH MATCHES "conda")
    message(STATUS "CONDA_PREFIX_PATH: ${CONDA_PREFIX_PATH}")
    set(PYTHON_PACKAGE_NAME "site-packages")
else()
    set(CONDA_PREFIX_PATH "/usr/local")
    set(PYTHON_PACKAGE_NAME "dist-packages")
    message(STATUS "CONDA not found")
endif()
# 将 Conda 环境路径添加到 CMAKE_PREFIX_PATH 中
list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX_PATH})
find_package(Python3 COMPONENTS Interpreter Development)

# 是否只编译 C 接口, 简化编译流程
# 常用在 migraphx 中, 编译命令: ../build.sh 1 "-DHEADDIM_128_ONLY=ON -DBUILD_C_INTERFACE=ON -DBUILD_ASM=OFF"
option(BUILD_C_INTERFACE "whether compile c interface only" OFF)

# 检查是否找到了 Python
if (Python3_FOUND)
    # 设置 PYTHON_INC 变量
    set(PYTHON_INC ${Python3_INCLUDE_DIRS})
    set(LIB_PATH_PYTHON ${Python3_LIBRARIES})
    set(PYTHON_VERSION "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
    message(STATUS "Python Version: ${PYTHON_VERSION}")
elseif (BUILD_C_INTERFACE)
    set(PYTHON_INC "${ROCM_PATH}/include") # placeholder to avoid compile error
    set(LIB_PATH_PYTHON "${ROCM_PATH}/lib")
    message(STATUS "Python3 is not needed for BUILD_C_INTERFACE.")
else ()
    message(FATAL_ERROR "Python3 development libraries not found.")
endif ()

# 设置头文件路径
set(ROCM_INC ${ROCM_PATH}/include)
set(PROJECT_INC ${CMAKE_CURRENT_LIST_DIR}/include)
set(SOURCE_INC ${CMAKE_CURRENT_LIST_DIR}/src)
set(THIRD_PARTY_INC ${CONDA_PREFIX_PATH}/lib/${PYTHON_VERSION}/${PYTHON_PACKAGE_NAME}/torch/include)
set(PYTORCH_INC ${CONDA_PREFIX_PATH}/lib/${PYTHON_VERSION}/${PYTHON_PACKAGE_NAME}/torch/include/torch/csrc/api/include)
message(STATUS "Torch_INCLUDE_DIRS: ${PYTORCH_INC}")

# 添加cu_dir目录下的kernel文件
set(PROJECT_CU_DIR ${CMAKE_CURRENT_LIST_DIR}/src/target)
set(CUDA_SOURCES "")

# 暂时过滤一些文件, 加速临时编译
set(BUILD_FA_OPTIONS "")

# 决定 rocm5.7 or rocm6.3.3
option(ROCM_5_7 "whether compile with rocm5.7" OFF)
set(ROCM_VERSION_FILE "${ROCM_PATH}/.info/version")
if (EXISTS ${ROCM_VERSION_FILE})
    file(READ ${ROCM_VERSION_FILE} FILE_CONTENT LIMIT 100 OFFSET 0)
    string(REGEX MATCH "^5.7.0" MATCH_RESULT ${FILE_CONTENT})
    MESSAGE(STATUS "ROCM_VERSION: ${MATCH_RESULT}")
    if (MATCH_RESULT)
        set(ROCM_5_7 ON)
    endif()
endif()
MESSAGE(STATUS "ROCM_5_7: ${ROCM_5_7}")
if (ROCM_5_7)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DROCM_5_7)
endif()

# 是否编译 C 接口, 无 torch 依赖, 无调试信息生成
MESSAGE(STATUS "BUILD_C_INTERFACE: ${BUILD_C_INTERFACE}")
if (BUILD_C_INTERFACE)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DBUILD_C_INTERFACE)
else()
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -g)
endif()

# 是否只编译 headdim 128 的 case
option(HEADDIM_128_ONLY "whether compile cases of headdim 128 only" OFF)
MESSAGE(STATUS "HEADDIM_128_ONLY: ${HEADDIM_128_ONLY}")

# 是否只编译 FlashAttention fwd/bwd 的 fp16/bf16 target, 用于 merge gate 快速迭代
option(BUILD_FA_F16_BF16_ONLY "whether compile only fp16/bf16 FlashAttention targets" OFF)
set(FA_BUILD_HEAD_DIMS "" CACHE STRING "semicolon-separated head dims to compile, e.g. 64;80;96;128")
MESSAGE(STATUS "BUILD_FA_F16_BF16_ONLY: ${BUILD_FA_F16_BF16_ONLY}")
MESSAGE(STATUS "FA_BUILD_HEAD_DIMS: ${FA_BUILD_HEAD_DIMS}")

MESSAGE(STATUS "HEADDIM_192_128_ONLY: ${HEADDIM_192_128_ONLY}")
if (HEADDIM_128_ONLY)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DHEADDIM_128_ONLY)
endif()
if (HEADDIM_192_128_ONLY)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DHEADDIM_192_128_ONLY)
endif()


# 是否编译前向的 case
option(BUILD_FA_FWD "whether compile fwd cases" OFF)
MESSAGE(STATUS "BUILD_FA_FWD: ${BUILD_FA_FWD}")
if (BUILD_FA_FWD)
    if (HEADDIM_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*fwd_hdim128_*.cpp )
    elseif (HEADDIM_192_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*fwd_hdimqk192_hdimv128_*.cpp)
    else()
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*fwd_hdim*.cpp)
    endif()
    set(CUDA_SOURCES ${CUDA_SOURCES} ${TARGET_CUDA_SOURCES})
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DBUILD_FA_FWD)
endif()

# 是否编译反向的 case
option(BUILD_FA_BWD "whether compile bwd cases" OFF)
MESSAGE(STATUS "BUILD_FA_BWD: ${BUILD_FA_BWD}")
if (BUILD_FA_BWD)
    if (HEADDIM_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*bwd_hdim128_*.cpp)
    elseif (HEADDIM_192_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*bwd_hdimqk192_hdimv128_*.cpp)
    else()
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*bwd*.cpp)
    endif()
    set(CUDA_SOURCES ${CUDA_SOURCES} ${TARGET_CUDA_SOURCES})
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DBUILD_FA_BWD)
endif()

# 是否编译 KV Cache 的 case
option(BUILD_FA_KVCACHE "whether compile bwd cases of kvcache" OFF)
MESSAGE(STATUS "BUILD_FA_KVCACHE: ${BUILD_FA_KVCACHE}")
if (BUILD_FA_KVCACHE)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DBUILD_FA_KVCACHE)
    if (HEADDIM_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*split_hdim128_*.cpp)
    elseif (HEADDIM_192_128_ONLY)
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*split_hdimqk*_hdimv*_*.cpp)
    else()
        file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*split_hdim*.cpp)
    endif()
    set(CUDA_SOURCES ${CUDA_SOURCES} ${TARGET_CUDA_SOURCES})
endif()

# 是否编译 FA 前向 的 permute
option(BUILD_FA_PERMUTE "whether compile permute cases of fa fwd varlen" OFF)
MESSAGE(STATUS "BUILD_FA_PERMUTE: ${BUILD_FA_PERMUTE}")
if (BUILD_FA_PERMUTE)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DBUILD_FA_PERMUTE)
    file(GLOB TARGET_CUDA_SOURCES ${PROJECT_CU_DIR}/*fwd_permute*.cpp)
    set(CUDA_SOURCES ${CUDA_SOURCES} ${TARGET_CUDA_SOURCES})
endif()

# 是否做 kernel 的单独机时, 观察框架 API 的影响
option(FA_KERNEL_TIMER "whether benchmark time for kernel only" OFF)
MESSAGE(STATUS "FA_KERNEL_TIMER: ${FA_KERNEL_TIMER}")
if (FA_KERNEL_TIMER)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DFA_KERNEL_TIMER)
endif()

# 是否开启 debug
option(FA_DEBUG "Open Debug" OFF)
MESSAGE(STATUS "FA_DEBUG: ${FA_DEBUG}")
if (FA_DEBUG)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DFA_DEBUG)
endif()

# 是否编译多个 page block size, PA
option(PA_PAGE_BLOCK_SIZE "Open Debug" OFF)
MESSAGE(STATUS "PA_PAGE_BLOCK_SIZE: ${PA_PAGE_BLOCK_SIZE}")
if (PA_PAGE_BLOCK_SIZE)
    set(BUILD_FA_OPTIONS ${BUILD_FA_OPTIONS} -DPA_PAGE_BLOCK_SIZE)
endif()


function(filter_merge_gate_fa_sources OUT_VAR)
    set(FILTERED_SOURCES "")
    foreach(SRC_FILE ${ARGN})
        get_filename_component(SRC_NAME ${SRC_FILE} NAME)
        set(KEEP_SOURCE ON)
        set(IS_LINK_COMPAT_SOURCE OFF)

        if (SRC_NAME MATCHES "^(flash_fwd_hdim512_(fp16|bf16)|flash_fwd_hdim128_prefix_prefill_(fp16|bf16))\\.cpp$")
            set(IS_LINK_COMPAT_SOURCE ON)
        endif()

        if (BUILD_FA_F16_BF16_ONLY)
            if (NOT IS_LINK_COMPAT_SOURCE AND SRC_NAME MATCHES "(fp8|prefix|prefill|permute|kvcache|paged|_pa_|_pa\\.)")
                set(KEEP_SOURCE OFF)
            endif()
        endif()

        if (KEEP_SOURCE AND NOT IS_LINK_COMPAT_SOURCE AND NOT "${FA_BUILD_HEAD_DIMS}" STREQUAL "")
            set(KEEP_DIM OFF)
            foreach(HEAD_DIM ${FA_BUILD_HEAD_DIMS})
                if (SRC_NAME MATCHES "hdim${HEAD_DIM}(_|\\.|$)" OR
                    SRC_NAME MATCHES "hdimqk${HEAD_DIM}(_|\\.|$)" OR
                    SRC_NAME MATCHES "hdimv${HEAD_DIM}(_|\\.|$)")
                    set(KEEP_DIM ON)
                endif()
            endforeach()
            if (NOT KEEP_DIM)
                set(KEEP_SOURCE OFF)
            endif()
        endif()

        if (KEEP_SOURCE)
            list(APPEND FILTERED_SOURCES ${SRC_FILE})
        endif()
    endforeach()
    set(${OUT_VAR} ${FILTERED_SOURCES} PARENT_SCOPE)
endfunction()

if (BUILD_FA_F16_BF16_ONLY OR NOT "${FA_BUILD_HEAD_DIMS}" STREQUAL "")
    list(LENGTH CUDA_SOURCES CUDA_SOURCES_BEFORE_FILTER)
    filter_merge_gate_fa_sources(CUDA_SOURCES ${CUDA_SOURCES})
    list(LENGTH CUDA_SOURCES CUDA_SOURCES_AFTER_FILTER)
    message(STATUS "CUDA_SOURCES filtered for merge gate: ${CUDA_SOURCES_BEFORE_FILTER} -> ${CUDA_SOURCES_AFTER_FILTER}")
endif()

if (BUILD_FA_F16_BF16_ONLY AND EXISTS "${PROJECT_CU_DIR}/flash_fp8_fwd_hdim128_fp16.cpp")
    set(MERGE_GATE_LINK_STUBS "${CMAKE_CURRENT_BINARY_DIR}/merge_gate_link_stubs.cpp")
    file(WRITE "${MERGE_GATE_LINK_STUBS}" [=[
#include "numeric_types.h"
#include "flash.h"
#include <stdexcept>

namespace {
void throw_merge_gate_disabled_fp8() {
    throw std::runtime_error("FP8 FlashAttention entry is disabled in BUILD_FA_F16_BF16_ONLY merge-gate builds");
}
}

template<>
void run_fp8_mha_fwd_<Float16, 128, 128>(Flash_fwd_params &, hipStream_t) {
    throw_merge_gate_disabled_fp8();
}

template<>
void run_fp8_mha_fwd_<BFloat16, 128, 128>(Flash_fwd_params &, hipStream_t) {
    throw_merge_gate_disabled_fp8();
}

template<>
void run_fp8_mha_fwd_prefix_prefill_<Float16, 128, 128>(Flash_fwd_params &, hipStream_t) {
    throw_merge_gate_disabled_fp8();
}

template<>
void run_fp8_mha_fwd_prefix_prefill_<BFloat16, 128, 128>(Flash_fwd_params &, hipStream_t) {
    throw_merge_gate_disabled_fp8();
}
]=])
    list(APPEND CUDA_SOURCES "${MERGE_GATE_LINK_STUBS}")
endif()

message(STATUS "BUILD_FA_OPTIONS: ${BUILD_FA_OPTIONS}")
message(STATUS "CUDA_SOURCES: ${CUDA_SOURCES}")

# 添加CMAKE_CURRENT_SOURCE_DIR目录下的.cpp文件
file(GLOB CPP_SOURCES *.cpp)

# 设置输出目录为CMAKE_CURRENT_BINARY_DIR
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# 设置LIB_PATH
set(LIB_PATH_TORCH ${CONDA_PREFIX_PATH}/lib/${PYTHON_VERSION}/${PYTHON_PACKAGE_NAME}/torch/lib)
set(LIB_PATH_PYTHON ${CONDA_PREFIX_PATH}/lib/)
message(STATUS "Python Lib Dir: ${LIB_PATH_PYTHON}")
message(STATUS "Torch  Lib Dir: ${LIB_PATH_TORCH}")

# 通用编译选项
set(HIP_COMPILE_FLAGS -I${ROCM_INC} -I${PROJECT_INC} -I${SOURCE_INC} ${HIP_COMPILE_FLAGS})
set(HIP_COMPILE_FLAGS ${HIP_COMPILE_FLAGS} -fPIC -O3 -std=c++17 -ffast-math -fno-finite-math-only -fno-gpu-rdc)
set(HIP_COMPILE_FLAGS ${HIP_COMPILE_FLAGS} ${BUILD_FA_OPTIONS} -DNDEBUGING)

# target specific 的后端编译选项
set(LLVM_COMPILE_FLAGS -mcode-object-version=5)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-support-${GFX_MAX_VGPRS}-vgprs=true)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-disable-machine-sink)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-disable-code-sink)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-amdgpu-enable-rewrite-partial-reg-uses=false)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-allow-gvn-convergent-call=true)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-disallow-uniform-vmed3-combine=true)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-amdgpu-early-inline-all=true)
set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-amdgpu-function-calls=false)
if (GFX_VERSION MATCHES "(92a|946)")
    set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-co-issue-vgpr-size=256)
endif()
# set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-hcu-pre-emit-load-store-opt=false)
# set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-greedy-reverse-local-assignment=true)
# set(LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} -mllvm=-bundle-kernel-args-loads-top-orderly=true)

# asm / IR 编译选项
set(ASM_COMPILE_FLAGS -fno-strict-return --cuda-device-only -fomit-frame-pointer -x hip)

# libtorch, pybind 接口文件的编译选项
set(BIND_COMPILE_FLAGS -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DCUDA_HAS_FP16=1 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1016"' -D_GLIBCXX_USE_CXX11_ABI=1)
set(BIND_COMPILE_FLAGS ${BIND_COMPILE_FLAGS} -I${THIRD_PARTY_INC} -I${PYTORCH_INC} -I${PYTHON_INC})

# 编译kernel文件
foreach(CU_FILE ${CUDA_SOURCES})
    get_filename_component(CU_FILE_WE ${CU_FILE} NAME_WE)
    set(CU_OBJECT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CU_FILE_WE}.o")
    if (${BUILD_ASM})
        set(CU_ASM "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CU_FILE_WE}.asm")
        add_custom_command(
            OUTPUT ${CU_OBJECT}
            COMMAND ${CC_WARPPER} -c ${CU_FILE} -o ${CU_OBJECT} ${HIP_COMPILE_FLAGS} ${LLVM_COMPILE_FLAGS}
            DEPENDS ${CU_FILE}
            COMMAND ${CMAKE_CXX_COMPILER} -DBUILD_ASM=ON ${ASM_COMPILE_FLAGS} -S ${CU_FILE}  -o ${CU_ASM} ${HIP_COMPILE_FLAGS} ${LLVM_COMPILE_FLAGS}
            DEPENDS ${CU_FILE}
            COMMENT "Building ${CU_OBJECT} from ${CU_FILE}"
        )
    else()
        add_custom_command(
            OUTPUT ${CU_OBJECT}
            COMMAND ${CC_WARPPER} -c ${CU_FILE} -o ${CU_OBJECT} ${HIP_COMPILE_FLAGS} ${LLVM_COMPILE_FLAGS}
            DEPENDS ${CU_FILE}
            COMMENT "Building ${CU_OBJECT} from ${CU_FILE}"
        )
    endif()
    set(CUDA_OBJECTS ${CUDA_OBJECTS} ${CU_OBJECT})
endforeach()

# 编译c++文件
foreach(CPP_FILE ${CPP_SOURCES})
    get_filename_component(CPP_FILE_WE ${CPP_FILE} NAME_WE)
    set(CPP_OBJECT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CPP_FILE_WE}.o")
    add_custom_command(
        OUTPUT ${CPP_OBJECT}
        COMMAND ${CC_WARPPER} ${HIP_COMPILE_FLAGS} ${BIND_COMPILE_FLAGS} -c ${CPP_FILE} -o ${CPP_OBJECT}
        DEPENDS ${CPP_FILE}
        COMMENT "Building ${CPP_OBJECT} from ${CPP_FILE}"
    )
    set(CPP_OBJECTS ${CPP_OBJECTS} ${CPP_OBJECT})
endforeach()

if (BUILD_C_INTERFACE)
    set(COMPILE_DEPENDENCY ${CUDA_OBJECTS} ${CPP_OBJECTS} )
else()
    set(COMPILE_DEPENDENCY ${CUDA_OBJECTS} ${CPP_OBJECTS} -L${LIB_PATH_TORCH} -L${LIB_PATH_PYTHON} -Wl,--no-as-needed -ltorch_hip -ltorch_cpu -ltorch_python -lc10 -l${PYTHON_VERSION})
endif()
if(NOT "${HIP_COMPILE_FLAGS}" MATCHES "-DNDEBUG")
    add_custom_target(flash_attention ALL
                    COMMAND ${CC_WARPPER}  ${HIP_COMPILE_FLAGS} -o flash_attention ${COMPILE_DEPENDENCY}
                    DEPENDS ${CUDA_OBJECTS} ${CPP_OBJECTS})
endif()

add_custom_target(libflash_attention.so ALL
                  COMMAND ${CC_WARPPER}  -shared ${HIP_COMPILE_FLAGS} -o libflash_attention.so ${COMPILE_DEPENDENCY}
                  DEPENDS ${CUDA_OBJECTS} ${CPP_OBJECTS})
