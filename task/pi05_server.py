"""Pi0.5 inference server for the LeRobot environment.

Runs outside Isaac Sim's Python environment. The Isaac-side policy talks to
this process through a length-prefixed pickle protocol.
"""
# ruff: noqa: E402
from __future__ import annotations

import os
import pickle
import struct
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION


if len(sys.argv) != 2:
    raise SystemExit("usage: python pi05_server.py /path/to/checkpoint/pretrained_model")

CKPT = sys.argv[1]
DEV = os.environ.get("PI05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TASK = os.environ.get("PI05_TASK", "assemble parts onto the task board")

policy = PI05Policy.from_pretrained(CKPT)
policy.eval().to(DEV)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    pretrained_revision=getattr(policy.config, "pretrained_revision", None),
    preprocessor_overrides={"device_processor": {"device": DEV}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)

sys.stderr.write(f"[pi05_server] loaded {CKPT} on {DEV}\n")
sys.stderr.flush()

_in = sys.stdin.buffer
_out = sys.stdout.buffer


def _read():
    header = _in.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack(">I", header)[0]
    buf = b""
    while len(buf) < size:
        chunk = _in.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(obj):
    payload = pickle.dumps(obj)
    _out.write(struct.pack(">I", len(payload)) + payload)
    _out.flush()


def _img(arr):
    # HxWx3 uint8/float -> 3xHxW float32 in [0, 1].
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return t.float().div(255.0)
    t = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1)
    return t.float().clamp(0.0, 1.0)


while True:
    msg = _read()
    if msg is None:
        break
    if msg.get("cmd") == "reset":
        policy.reset()
        _write({"ok": True})
        continue

    obs = {
        "observation.state": torch.as_tensor(msg["state"], dtype=torch.float32),
        "observation.images.head": _img(msg["head"]),
        "observation.images.left_hand": _img(msg["left"]),
        "observation.images.right_hand": _img(msg["right"]),
        "task": msg.get("task", TASK),
    }

    with torch.inference_mode():
        batch = preprocessor(obs)
        action = policy.select_action(batch)
        action = postprocessor(action)
    if isinstance(action, dict):
        action = action[ACTION]
    action_np = action.squeeze(0).float().cpu().numpy().reshape(-1)
    if action_np.shape != (14,):
        raise RuntimeError(f"expected 14-D pi0.5 action, got shape {action_np.shape}")
    if not np.isfinite(action_np).all():
        raise RuntimeError("pi0.5 action contains non-finite values")
    _write({"action": action_np.tolist()})
