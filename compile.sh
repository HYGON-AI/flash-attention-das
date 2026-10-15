#!/usr/bin/env bash
set -euo pipefail

usage() {
  command cat <<'EOF'
Build and install the pure HIP FlashAttention MHA shared library.

Environment:
  ROCM_PATH          DTK/HIP root (default: /opt/dtk)
  MHA_HIP_COMPILER   HIP compiler (default: aicc when available, otherwise hipcc)
  MHA_GPU_ARCHS      Semicolon-separated GPU architectures
  MHA_BUILD_DIR      Out-of-source build directory (default: build-mha-hip)
  MHA_INSTALL_PREFIX Install prefix (default: $MHA_BUILD_DIR/install)
  MAX_JOBS           Parallel build jobs (default: detected CPU count)
  MHA_CODEGEN_ONLY   Set to 1 to generate sources without compiling/installing

Options:
  -h, --help         Show this help
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    usage >&2
    exit 2
    ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=${script_dir}
dtk_env=${ROCM_PATH:-/opt/dtk}/env.sh
if [[ ! -r "${dtk_env}" ]]; then
  echo "error: DTK environment script not found or not readable: ${dtk_env}" >&2
  exit 1
fi
set +u
source "${dtk_env}"
set -u
hip_root=${ROCM_PATH:-/opt/dtk}
if [[ -n "${MHA_HIP_COMPILER:-}" ]]; then
  hip_compiler=${MHA_HIP_COMPILER}
elif [[ -x "${hip_root}/bin/aicc" ]]; then
  hip_compiler=${hip_root}/bin/aicc
else
  hip_compiler=${hip_root}/bin/hipcc
fi
gpu_archs=${MHA_GPU_ARCHS:-gfx938;gfx936;gfx928;gfx92a}
build_dir=${MHA_BUILD_DIR:-${source_dir}/build-mha-hip}
install_prefix=${MHA_INSTALL_PREFIX:-${build_dir}/install}
max_jobs=${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN)}
codegen_only=${MHA_CODEGEN_ONLY:-0}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "error: required executable not found or not executable: $1" >&2
    exit 1
  fi
}

require_command cmake
require_command python3
require_executable "${hip_compiler}"
require_executable "${hip_root}/hip/bin/hipify/hipify-perl"

if command -v ninja >/dev/null 2>&1; then
  cmake_generator=Ninja
elif command -v make >/dev/null 2>&1; then
  cmake_generator='Unix Makefiles'
else
  echo "error: neither ninja nor make is available" >&2
  exit 1
fi

if [[ ! "${max_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MAX_JOBS must be a positive integer: ${max_jobs}" >&2
  exit 1
fi
if [[ "${codegen_only}" != 0 && "${codegen_only}" != 1 ]]; then
  echo "error: MHA_CODEGEN_ONLY must be 0 or 1: ${codegen_only}" >&2
  exit 1
fi

cmake_arguments=(
  -S "${source_dir}"
  -B "${build_dir}"
  -G "${cmake_generator}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER="${hip_compiler}"
  -DMHA_HIP_ROOT="${hip_root}"
  -DMHA_HIP_COMPILER="${hip_compiler}"
  -DMHA_GPU_ARCHS="${gpu_archs}"
  -DCMAKE_INSTALL_PREFIX="${install_prefix}"
)

if [[ "${codegen_only}" == 1 ]]; then
  cmake_arguments+=( -DMHA_CODEGEN_ONLY=ON )
  cmake "${cmake_arguments[@]}"
  cmake --build "${build_dir}" --target generate-mha-sources -j "${max_jobs}"
  exit 0
fi

cmake "${cmake_arguments[@]}"
cmake --build "${build_dir}" --target multi-head-attention -j "${max_jobs}"
cmake --install "${build_dir}"
