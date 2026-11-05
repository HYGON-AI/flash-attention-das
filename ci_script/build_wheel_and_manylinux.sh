#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building flashattn wheel in ${repo_root}"
cd "${repo_root}"

source /opt/dtk/env.sh

export PYTHON="${PYTHON:-python}"
PYTHON_VERSION="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

declare -A NUMPY_VERSIONS=(
  ["3.8"]="1.21.6"
  ["3.9"]="1.22.4"
  ["3.10"]="1.24.3"
  ["3.11"]="1.26.2"
  ["3.12"]="1.26.2"
  ["3.13"]="2.1.2"
  ["3.14"]="2.3.4"
)
declare -A PANDAS_VERSIONS=(
  ["3.8"]="2.0.3"
  ["3.9"]="2.2.3"
  ["3.10"]="2.2.3"
  ["3.11"]="2.2.3"
  ["3.12"]="2.2.3"
)

NUMPY_VERSION="${NUMPY_VERSIONS[$PYTHON_VERSION]:-}"
PANDAS_VERSION="${PANDAS_VERSIONS[$PYTHON_VERSION]:-}"
echo "Python ${PYTHON_VERSION}: numpy ${NUMPY_VERSION:-auto}, pandas ${PANDAS_VERSION:-auto}"

PIP_CONSTRAINT="${PIP_CONSTRAINT:-$(mktemp)}"
export PIP_CONSTRAINT
# Keep these constraints visible to later pip commands in the parent CI shell.
printf '%s\n' "numpy${NUMPY_VERSION:+==${NUMPY_VERSION}}" "pandas${PANDAS_VERSION:+==${PANDAS_VERSION}}" > "${PIP_CONSTRAINT}"
if [[ -n "${TORCH_VERSION:-}" ]]; then
  printf 'torch==%s\n' "${TORCH_VERSION}" >> "${PIP_CONSTRAINT}"
fi
echo "Dependency constraints: $(tr '\n' ' ' < "${PIP_CONSTRAINT}")"

"${PYTHON}" -m pip install "pandas${PANDAS_VERSION:+==${PANDAS_VERSION}}" -i https://pypi.tuna.tsinghua.edu.cn/simple/
"${PYTHON}" -m pip install "numpy${NUMPY_VERSION:+==${NUMPY_VERSION}}" --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/
"${PYTHON}" -m pip install pyyaml -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install hypothesis -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install Ninja -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install MonkeyType -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install wheel -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install packaging -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install einops -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install pytest -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install auditwheel patchelf -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
"${PYTHON}" -m pip install setuptools==79.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall

"${PYTHON}" -m pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple/

"${PYTHON}" -c "import numpy, pandas; pandas.DataFrame([{'ok': 1}]); print(f'numpy {numpy.__version__} / pandas {pandas.__version__} installed successfully')"

hipcc --version

"${PYTHON}" - <<'PY'
import torch

if torch.version.hip is None:
    raise RuntimeError("Expected a HIP-enabled Torch build")
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
PY

git submodule update --init
"${PYTHON}" setup.py bdist_wheel

wheel_count="$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l)"
if [[ "${wheel_count}" -ne 1 ]]; then
  echo "Expected exactly one wheel in dist/ after build_wheel_and_manylinux.sh, found ${wheel_count}" >&2
  exit 1
fi

shopt -s nullglob
wheels=(dist/flash_attn*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one flashattn wheel in dist/, found ${#wheels[@]}" >&2
  exit 1
fi

torch_path="$("${PYTHON}" -c "import torch, os; print(os.path.dirname(torch.__file__))")"
if [[ -z "${torch_path}" || ! -d "${torch_path}/lib" ]]; then
  echo "Unable to locate the PyTorch library directory" >&2
  exit 1
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${torch_path}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${torch_path}/lib"
fi

platform="manylinux_$(ldd --version | awk 'NR==1 {print $NF}' | tr '.' '_')_x86_64"
echo "Repairing ${wheels[0]} for ${platform}"
auditwheel show "${wheels[0]}"

excludes=(
  --exclude "libgalaxyhip.so*"
  --exclude "libMIOpen.so*"
  --exclude "librccl.so*"
  --exclude "libhipblas.so*"
  --exclude "libhipfft.so*"
  --exclude "libhiprand.so*"
  --exclude "libhipsolver.so*"
  --exclude "libhipsparse.so*"
  --exclude "libhipnn.so*"
  --exclude "librocblas.so*"
  --exclude "librocsolver.so*"
  --exclude "librocfft.so*"
  --exclude "librocrand.so*"
  --exclude "librocsparse.so*"
  --exclude "librocm_smi64.so*"
  --exclude "librocfft-device-*.so*"
  --exclude "libc10.so*"
  --exclude "libc10_hip.so*"
  --exclude "libtorch_cpu.so*"
  --exclude "libtorch_hip.so*"
  --exclude "libtorch_python.so*"
  --exclude "libtorch.so*"
  --exclude "libtorchaudio*"
  --exclude "_torchaudio*"
  --exclude "libavcodec*"
  --exclude "libavdevice*"
  --exclude "libavfilter*"
  --exclude "libavformat*"
  --exclude "libavutil*"
  --exclude "libsox*"
  --exclude "libhipblaslt.so*"
  --exclude "libnuma.so*"
)

auditwheel repair --plat "${platform}" --strip "${excludes[@]}" -w dist "${wheels[0]}"

repaired=(dist/*manylinux*.whl)
if [[ "${#repaired[@]}" -ne 1 ]]; then
  echo "Expected exactly one repaired manylinux flashattn wheel in dist/, found ${#repaired[@]}" >&2
  exit 1
fi
