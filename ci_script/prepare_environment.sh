#!/usr/bin/env bash

: "${DTK_VERSION:?DTK_VERSION must be set}"
: "${DTK_PIP_INDEX_URL:?DTK_PIP_INDEX_URL must be set}"

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

# Install the newest AICC package published on the AICC nightly server
# AICC filename format:
#   dtk_llvm_aicc-<pkg_version>+<dtk_tag>.<YYMMDDHHMM>.<commit>-x86_64.run
#
# <dtk_tag> names the DTK release and does not follow DTK_VERSION: dtk2610
# already has builds under its own tag, the transitional dtk26042 is only a
# fallback for as long as it is published.
#
# Every line below is bracketed or prefixed so that one run's whole AICC story
# can be pulled back out of a job log with:  grep -n 'AICC' job.log
aicc_banner() {
  local rule
  rule="$(printf '=%.0s' {1..96})"
  echo "${rule}"
  printf '= %-92s =\n' "$1"
  echo "${rule}"
}
aicc_log() { echo "[AICC] $*"; }

if [[ -z "${AICC_BASE_URL}" ]]; then
  aicc_banner "AICC result: FAILED, AICC_BASE_URL is not set"
  exit 1
fi
case "${DTK_VERSION}" in
  26.04) aicc_tags=(dtk2604) ;;
  26.10) aicc_tags=(dtk2610 dtk26042) ;;
  *)
    aicc_banner "AICC result: FAILED, no tag configured for DTK ${DTK_VERSION}"
    exit 1
    ;;
esac

aicc_banner "AICC install: DTK ${DTK_VERSION}, tag(s): ${aicc_tags[*]}"
aicc_log "base URL: ${AICC_BASE_URL}"

if ! aicc_listing="$(wget -qO- -t 3 -T 30 "${AICC_BASE_URL}/")"; then
  aicc_banner "AICC result: FAILED, cannot list ${AICC_BASE_URL}/"
  exit 1
fi
aicc_log "listing OK"

aicc_candidates=()
for aicc_tag in "${aicc_tags[@]}"; do
  mapfile -t aicc_tag_candidates < <(
    printf '%s\n' "${aicc_listing}" \
      | grep -oE "dtk_llvm_aicc-[^\"%]*%2B${aicc_tag}\.[0-9]{8,12}\.[^\"]*\.run" \
      | sed -E "s/.*%2B${aicc_tag}\.([0-9]{8,12})\..*/\1 &/" \
      | awk '!seen[$0]++' \
      | sort -nr | cut -d' ' -f2- | head -n 3
  )
  aicc_log "tag ${aicc_tag}: ${#aicc_tag_candidates[@]} candidate(s)"
  aicc_candidates+=("${aicc_tag_candidates[@]}")
done

cd /opt || exit 1
aicc_installer=""

for aicc_index in "${!aicc_candidates[@]}"; do
  aicc_url_name="${aicc_candidates[$aicc_index]}"
  aicc_installer="${aicc_url_name//%2B/+}"
  aicc_wget_log="${aicc_installer}.wget.log"
  aicc_log "downloading $((aicc_index + 1))/${#aicc_candidates[@]}: ${aicc_installer}"

  if wget -nv -c -t 3 -T 300 -O "${aicc_installer}.part" \
       "${AICC_BASE_URL}/${aicc_url_name}" 2>"${aicc_wget_log}" \
     && mv -f "${aicc_installer}.part" "${aicc_installer}" \
     && [[ -s "${aicc_installer}" ]]; then
    aicc_log "download OK: $(stat -c '%s' "${aicc_installer}") bytes"
    rm -f "${aicc_wget_log}"
    break
  fi

  aicc_log "download FAILED, falling back to the next candidate: ${aicc_installer}"
  sed 's/^/[AICC]   wget: /' "${aicc_wget_log}" | tail -n 20 >&2 || true
  rm -f "${aicc_installer}.part" "${aicc_installer}" "${aicc_wget_log}"
  aicc_installer=""
done

if [[ -n "${aicc_installer}" ]]; then
  aicc_log "installing ${aicc_installer}"
  chmod +x "${aicc_installer}"

  set +o pipefail
  yes | bash "${aicc_installer}"
  aicc_status="${PIPESTATUS[1]}"
  set -o pipefail

  if [[ "${aicc_status}" -ne 0 ]]; then
    aicc_banner "AICC result: INSTALL FAILED (exit ${aicc_status})"
    exit "${aicc_status}"
  fi

  aicc_banner "AICC result: INSTALLED ${aicc_installer}"
else
  aicc_banner "AICC result: SKIPPED, no usable installer for ${aicc_tags[*]}"
fi

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

python -m pip install pyyaml -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install hypothesis -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install Ninja -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install MonkeyType -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install wheel -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install packaging -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install einops -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install pytest -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install auditwheel patchelf -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install optest -i "${DTK_PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST:-10.16.1.201}" 
python -m pip install ciupload -i "${DTK_PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST:-10.16.1.201}" 
python -m pip install setuptools==79.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall

export PYTHON_VERSION="$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')"

declare -A NUMPY_VERSIONS=(
  ["3.8"]="1.21.6"
  ["3.9"]="1.22.4"
  ["3.10"]="1.24.3"
  ["3.11"]="1.26.2"
  ["3.12"]="1.26.2"
  ["3.13"]="2.1.2"
  ["3.14"]="2.3.4"
)

NUMPY_VERSION="${NUMPY_VERSIONS[$PYTHON_VERSION]}"
if [[ -z "${NUMPY_VERSION}" ]]; then
  echo "Warning: no numpy version configured for Python ${PYTHON_VERSION}; using numpy==1.24.3"
  NUMPY_VERSION="1.24.3"
fi

echo "Installing numpy==${NUMPY_VERSION} ..."
python -m pip uninstall -y numpy 2>/dev/null || true
python -m pip install numpy=="${NUMPY_VERSION}" \
  --force-reinstall \
  --no-deps \
  --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

python -c "import numpy; print(f'numpy {numpy.__version__} installed successfully')"

hipcc --version

python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip)"
