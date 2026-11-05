# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

set(MHA_CODEGEN_ONLY OFF CACHE BOOL
  "Generate pure HIP MHA sources without constructing the shared library")
set(MHA_HIP_ROOT "$ENV{ROCM_PATH}" CACHE PATH "DTK/HIP installation root")
if(NOT MHA_HIP_ROOT)
  set(MHA_HIP_ROOT "/opt/dtk" CACHE PATH "DTK/HIP installation root" FORCE)
endif()
if(EXISTS "${MHA_HIP_ROOT}/bin/aicc")
  set(_MHA_DEFAULT_HIP_COMPILER "${MHA_HIP_ROOT}/bin/aicc")
else()
  set(_MHA_DEFAULT_HIP_COMPILER "${MHA_HIP_ROOT}/bin/hipcc")
endif()
set(MHA_HIP_COMPILER "${_MHA_DEFAULT_HIP_COMPILER}" CACHE FILEPATH
  "Compiler used for pure HIP FlashAttention MHA sources")
unset(_MHA_DEFAULT_HIP_COMPILER)
set(MHA_HIPIFY_PERL "${MHA_HIP_ROOT}/hip/bin/hipify/hipify-perl" CACHE FILEPATH
  "DTK hipify-perl executable")
set(MHA_GPU_ARCHS "gfx938;gfx936;gfx928;gfx92a" CACHE STRING
  "GPU architectures compiled into libmulti-head-attention")
set(MHA_KERNEL_LIMIT "" CACHE STRING
  "Test-only limit for the number of generated kernels to compile")

if(NOT MHA_CODEGEN_ONLY AND NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${MHA_HIP_COMPILER}" CACHE FILEPATH
    "C++ compiler selected before project()" FORCE)
endif()

project(flash_attention_mha_hip VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT EXISTS "${MHA_HIP_COMPILER}")
  message(FATAL_ERROR "MHA_HIP_COMPILER does not exist: ${MHA_HIP_COMPILER}")
endif()
if(NOT EXISTS "${MHA_HIPIFY_PERL}")
  message(FATAL_ERROR "MHA_HIPIFY_PERL does not exist: ${MHA_HIPIFY_PERL}")
endif()

find_package(hip CONFIG REQUIRED
  PATHS
    "${MHA_HIP_ROOT}/lib/cmake/hip"
    "${MHA_HIP_ROOT}/hip/lib/cmake/hip"
  NO_DEFAULT_PATH)

include(cmake/FlashAttentionMHA.cmake)
set(MHA_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn_cutlass")
set(MHA_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated/flash_attn_cutlass")
flash_attention_configure_mha_sources(
  SOURCE_ROOT "${MHA_SOURCE_ROOT}"
  OUTPUT_ROOT "${MHA_GENERATED_ROOT}"
  HIPIFY_PERL "${MHA_HIPIFY_PERL}"
  OUT_HOST MHA_GENERATED_HOST
  OUT_KERNELS MHA_GENERATED_KERNELS
  OUT_HEADERS MHA_GENERATED_HEADERS
  OUT_TARGET MHA_CODEGEN_TARGET)

add_custom_target(generate-mha-sources)
add_dependencies(generate-mha-sources "${MHA_CODEGEN_TARGET}")

if(MHA_CODEGEN_ONLY)
  message(STATUS "MHA_CODEGEN_ONLY=ON: configured generate-mha-sources only")
  return()
endif()

set(MHA_COMPILE_OPTIONS
  -O3
  -w
  -std=c++17
  -fPIC
  -DNDEBUG
  -DUSE_ROCM
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math
  -mno-fma
  -DHCU_ASM
  -DFLASH_ATTENTION_BF16_TYPE=0
  -DFLASHATTENTION_DISABLE_BACKWARD
  -DFLASHATTENTION_DISABLE_DROPOUT
  -DFLASHATTENTION_DISABLE_ALIBI
  -DFLASHATTENTION_DISABLE_SOFTCAP
  -DFLASHATTENTION_DISABLE_UNEVEN_K
  -DFLASHATTENTION_DISABLE_LOCAL
  -DBWDSEPARATE
  -DBF162FLOAT_USE_ASM
  -mllvm
  -enable-num-vgprs-512=true
  -ftemplate-backtrace-limit=0)
foreach(gpu_architecture IN LISTS MHA_GPU_ARCHS)
  list(APPEND MHA_COMPILE_OPTIONS "--offload-arch=${gpu_architecture}")
endforeach()

set(MHA_INCLUDE_DIRECTORIES
  "${MHA_GENERATED_ROOT}"
  "${MHA_GENERATED_ROOT}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include"
  "${MHA_HIP_ROOT}/include"
  "${MHA_HIP_ROOT}/hip/include")
set(MHA_INCLUDE_OPTIONS)
foreach(include_directory IN LISTS MHA_INCLUDE_DIRECTORIES)
  list(APPEND MHA_INCLUDE_OPTIONS "-I${include_directory}")
endforeach()

set(MHA_KERNEL_SOURCES_TO_COMPILE ${MHA_GENERATED_KERNELS})
if(MHA_KERNEL_LIMIT)
  if(NOT MHA_KERNEL_LIMIT MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "MHA_KERNEL_LIMIT must be a positive integer")
  endif()
  list(LENGTH MHA_KERNEL_SOURCES_TO_COMPILE MHA_KERNEL_COUNT)
  if(MHA_KERNEL_LIMIT LESS MHA_KERNEL_COUNT)
    list(SUBLIST MHA_KERNEL_SOURCES_TO_COMPILE 0 ${MHA_KERNEL_LIMIT}
      MHA_KERNEL_SOURCES_TO_COMPILE)
  endif()
  message(WARNING
    "MHA_KERNEL_LIMIT=${MHA_KERNEL_LIMIT} is test-only and reduces compiled functionality")
endif()

set(MHA_KERNEL_OBJECTS)
foreach(generated_kernel IN LISTS MHA_KERNEL_SOURCES_TO_COMPILE)
  get_filename_component(kernel_stem "${generated_kernel}" NAME_WE)
  set(kernel_object "${CMAKE_CURRENT_BINARY_DIR}/mha_objects/${kernel_stem}.o")
  add_custom_command(
    OUTPUT "${kernel_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/mha_objects"
    COMMAND "${MHA_HIP_COMPILER}"
      ${MHA_COMPILE_OPTIONS}
      ${MHA_INCLUDE_OPTIONS}
      -c "${generated_kernel}"
      -o "${kernel_object}"
    DEPENDS "${generated_kernel}" ${MHA_GENERATED_HEADERS}
    COMMENT "Building ${kernel_stem}.o"
    VERBATIM)
  list(APPEND MHA_KERNEL_OBJECTS "${kernel_object}")
endforeach()

add_library(multi-head-attention SHARED
  "${MHA_GENERATED_HOST}"
  ${MHA_KERNEL_OBJECTS})
add_dependencies(multi-head-attention "${MHA_CODEGEN_TARGET}")
target_compile_features(multi-head-attention PRIVATE cxx_std_17)
target_compile_options(multi-head-attention PRIVATE ${MHA_COMPILE_OPTIONS})
target_include_directories(multi-head-attention
  PRIVATE ${MHA_INCLUDE_DIRECTORIES}
  INTERFACE
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/flash-attention>")
target_link_libraries(multi-head-attention PUBLIC hip::host)
if(NOT MHA_KERNEL_LIMIT)
  target_link_options(multi-head-attention PRIVATE "-Wl,--no-undefined")
endif()
set_target_properties(multi-head-attention PROPERTIES
  OUTPUT_NAME multi-head-attention
  POSITION_INDEPENDENT_CODE ON)

install(TARGETS multi-head-attention
  EXPORT FlashAttentionTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(MHA_PUBLIC_HEADERS
  "${MHA_GENERATED_ROOT}/flash_c_api.h"
  "${MHA_GENERATED_ROOT}/src/flash_attnmask.h"
  "${MHA_GENERATED_ROOT}/src/flash.h"
  "${MHA_GENERATED_ROOT}/src/flash_dtk_macros.h"
  "${MHA_GENERATED_ROOT}/src/flash_mha_runtime.h")
foreach(generated_header IN LISTS MHA_PUBLIC_HEADERS)
  file(RELATIVE_PATH header_relative_path "${MHA_GENERATED_ROOT}" "${generated_header}")
  get_filename_component(header_directory "${header_relative_path}" DIRECTORY)
  install(FILES "${generated_header}"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flash-attention/${header_directory}")
endforeach()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlashAttentionConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FlashAttentionConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlashAttention")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FlashAttentionConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(EXPORT FlashAttentionTargets
  FILE FlashAttentionTargets.cmake
  NAMESPACE FlashAttention::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlashAttention")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/FlashAttentionConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FlashAttentionConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlashAttention")
