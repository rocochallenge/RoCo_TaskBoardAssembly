#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="$(cd "${repo_root}/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"

ckpt="${1:-${PI05_CKPT:-}}"
if [[ -z "${ckpt}" ]]; then
  echo "usage: $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi

export PI05_CKPT="${ckpt}"
export PI05_SERVER_PY="${PI05_SERVER_PY:-${lerobot_root}/.venv/bin/python}"
export PI05_CUDA_VISIBLE_DEVICES="${PI05_CUDA_VISIBLE_DEVICES:-1}"
export PI05_DEVICE="${PI05_DEVICE:-cuda}"
export HF_HOME="${HF_HOME:-${models_root}/.hf-cache}"
if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(<"${HF_HOME}/token")"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

extra_args=()
if [[ -n "${PI05_EVAL_MAX_STEPS:-}" ]]; then
  extra_args+=(--max-steps "${PI05_EVAL_MAX_STEPS}")
fi
if [[ -n "${PI05_EVAL_MAX_SIM_SECONDS:-}" ]]; then
  extra_args+=(--max-sim-seconds "${PI05_EVAL_MAX_SIM_SECONDS}")
fi
if [[ -n "${PI05_EVAL_MAX_PARTS:-}" ]]; then
  extra_args+=(--max-parts "${PI05_EVAL_MAX_PARTS}")
fi

cd "${repo_root}"
exec ./scripts/run_roco.sh \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video "${PI05_EVAL_VIDEO:-artifacts/pi05_eval_head.mp4}" \
  --record-video-camera "${PI05_EVAL_CAMERA:-head}" \
  --record-video-fps "${PI05_EVAL_FPS:-15}" \
  --results-json "${PI05_EVAL_RESULTS:-artifacts/pi05_eval_results.json}" \
  "${extra_args[@]}"
