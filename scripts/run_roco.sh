#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cache_root="$(cd -- "${repo_root}/.." && pwd)"

cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/.uv-cache}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ -z "${DISPLAY:-}" ]]; then
  export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
fi

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

exec uv run python task/run_pick_place.py "$@"
