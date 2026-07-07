# pi0.5 Evaluation

This repo can evaluate a LeRobot pi0.5 checkpoint through an Isaac Sim
policy adapter. The adapter starts a small sidecar process for pi0.5
inference and sends observations from the RoCo task runner to it.

## Prerequisites

- RoCo repo dependencies installed with `uv sync`.
- A LeRobot checkout with pi0.5 dependencies installed. By default the
  eval script expects it at `../lerobot_roco_pi05` relative to this repo.
- Hugging Face auth available through `HF_HOME/token` or `HF_TOKEN`.
- A pi0.5 checkpoint directory containing `config.json` and
  `model.safetensors`, for example a LeRobot `pretrained_model` checkpoint.

The adapter expects the RoCo dataset action layout:

```text
left xyz + left xyz Euler + left gripper + right xyz + right xyz Euler + right gripper
```

The current task runner still holds the R arm fixed, so only the left 7-D
slice is executed during rollout.

## Smoke Eval

Use a short simulated-time limit when you only need to verify the code path
and produce a rollout video/JSON:

```bash
HF_HOME=/path/to/.hf-cache \
ISAACSIM_HEADLESS=1 \
ISAACSIM_ACTIVE_GPU=2 \
ISAACSIM_PHYSICS_GPU=2 \
PI05_CUDA_VISIBLE_DEVICES=1 \
PI05_EVAL_MAX_SIM_SECONDS=10 \
PI05_EVAL_VIDEO=artifacts/pi05_smoke_head.mp4 \
PI05_EVAL_RESULTS=artifacts/pi05_smoke_results.json \
./scripts/eval_pi05_roco.sh /path/to/checkpoint/pretrained_model
```

The command writes an MP4 video and a JSON result file. When a short limit is
used, the JSON contains metadata such as:

```json
{
  "completion_reason": "max_sim_seconds",
  "sim_time_s": 10.01
}
```

## Full Eval

Omit the short-run limit to run the full task:

```bash
HF_HOME=/path/to/.hf-cache \
ISAACSIM_HEADLESS=1 \
ISAACSIM_ACTIVE_GPU=2 \
ISAACSIM_PHYSICS_GPU=2 \
PI05_CUDA_VISIBLE_DEVICES=1 \
PI05_EVAL_VIDEO=artifacts/pi05_eval_head.mp4 \
PI05_EVAL_RESULTS=artifacts/pi05_eval_results.json \
./scripts/eval_pi05_roco.sh /path/to/checkpoint/pretrained_model
```

## Useful Environment Variables

- `LEROBOT_ROOT`: path to the LeRobot checkout.
- `PI05_CUDA_VISIBLE_DEVICES`: GPU visible to the pi0.5 sidecar.
- `ISAACSIM_ACTIVE_GPU`, `ISAACSIM_PHYSICS_GPU`: GPUs used by Isaac Sim.
- `PI05_EVAL_CAMERA`: one of `head`, `L_wrist`, or `R_wrist`.
- `PI05_EVAL_FPS`: output video FPS.
- `PI05_EVAL_MAX_SIM_SECONDS`: stop after this many simulated seconds.
- `PI05_EVAL_MAX_STEPS`: stop after this many task-control steps.
- `PI05_EVAL_MAX_PARTS`: stop after this many completed parts.

Isaac Sim may print `Failed to create change watch ... errno=28` when the
system inotify watch limit is exhausted. In headless smoke eval this is noisy
but not fatal if the process continues into rollout and writes the artifacts.
