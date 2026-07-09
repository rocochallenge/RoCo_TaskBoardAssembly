"""Worked example: deploy a trained LeRobot Diffusion Policy in the harness.

This is a REFERENCE you adapt to your own policy — it shows the integration
pattern, not a plug-and-play checkpoint. The `Policy` interface makes no
assumption about how `act` produces its action, so a trained network drives
the same `L.forward(pos, quat, grip)` call the scripted baseline uses.

Two things make a modern learned policy awkward to run in-process, and this
example works around both:

  * A torch / lerobot stack usually can't share Isaac Sim's interpreter (numpy
    1.26 pin, Isaac's own torch build). So the model lives in a SEPARATE
    process (its own venv) and this adapter talks to it over a stdin/stdout
    pickle pipe (see `dp_server.py`); lerobot is never imported here.
  * Each step it rebuilds the training-time `observation.state` + 3 RGB images,
    gets a 14-D action back, takes the left-arm slice
    [xyz + rotvec + gripper], converts rotvec->quat, and drives the L-arm IK.

Self-contained by design: it depends only on the public `policy_api` surface
(`EnvInfo`, `Observation`) plus `param_config`, so it drops into the devkit
with no other new modules.

NOTE ON THE RIGHT ARM: the public `EnvInfo` exposes `L_controller` but not the
right-arm end effector, so the 7 state dims for the right EE pose (Rp, Rq) are
zeroed here unless the harness happens to provide an `R_controller` attribute
(picked up opportunistically via getattr). A policy trained on the full 44-D
state that includes the right EE pose should reconstruct those dims from the
right-arm joints (forward kinematics) — swap `_build_state` accordingly.

Env:
  DP_CKPT         path to the checkpoint pretrained_model dir (required)
  DP_SERVER_PY    python for the model's venv (default /lrv/venv/bin/python)
Run the harness with camera output enabled so `Observation.rgb` is populated.
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys

import numpy as np

import param_config as pc  # noqa: E402  (kept for parity with other policies)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Constants inlined so this file has no extra module dependencies. ---
GRIPPER_OPEN_LIMIT = 0.6649704       # L_gripper joint value at fully open
_IMG_H, _IMG_W = 240, 320            # RGB size the model was trained at


def _resize_rgb(img):
    """HxWx3 uint8 (any size) -> (240,320,3) uint8. None -> zeros."""
    if img is None:
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.shape[0] == _IMG_H and a.shape[1] == _IMG_W:
        return a.astype(np.uint8)
    try:
        import cv2
        return cv2.resize(a, (_IMG_W, _IMG_H),
                          interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        ys = max(1, a.shape[0] // _IMG_H)
        xs = max(1, a.shape[1] // _IMG_W)
        out = a[::ys, ::xs, :3]
        return out[:_IMG_H, :_IMG_W].astype(np.uint8)


def _rotvec_to_quat_wxyz(rx, ry, rz):
    """Decode the dataset-style rotation-vector orientation output."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_rotvec([rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class DiffusionLeRobotPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        # Right-arm controller is not part of the public EnvInfo contract; use
        # it only if the harness exposes it (see the RIGHT ARM note up top).
        self.R = getattr(env_info, "R_controller", None)
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("DP_CKPT")
        if not ckpt:
            raise ValueError("set DP_CKPT to the checkpoint pretrained_model dir")
        server_py = os.environ.get("DP_SERVER_PY", "/lrv/venv/bin/python")
        # CLEAN env: the harness runs in Isaac's interpreter which exports
        # PYTHONPATH / LD_LIBRARY_PATH / CARB_* pointing at Isaac packages. If
        # the model's venv subprocess inherits those it loads Isaac's torch /
        # libs and crashes. Pass only what the server needs.
        keep = ("HOME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES", "UV_PYTHON_INSTALL_DIR")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin",
               "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        for k in keep:
            if k in os.environ:
                env[k] = os.environ[k]
        _log_path = os.environ.get(
            "DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            [server_py, os.path.join(_TASK_DIR, "dp_server.py"), ckpt],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp] spawned inference server (ckpt={ckpt})", flush=True)
        # No handshake; give it a moment to load and check it's alive.
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server died on startup (exit {self._proc.returncode}); "
                f"see {_log_path}")
        self._last_action = None

    # ---- length-prefixed pickle pipe ----
    def _send(self, obj):
        b = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(b)) + b)
        self._proc.stdin.flush()

    def _recv(self):
        h = self._proc.stdout.read(4)
        if len(h) < 4:
            raise RuntimeError("dp_server closed")
        n = struct.unpack(">I", h)[0]
        buf = b""
        while len(buf) < n:
            buf += self._proc.stdout.read(n - len(buf))
        return pickle.loads(buf)

    # ---- Policy API ----
    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._send({"cmd": "reset"})
        self._recv()

    def _build_state(self, obs: Observation) -> np.ndarray:
        """Rebuild the 44-D training-time observation.state from `obs`."""
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        # L EE pose: prefer the controller (byte-for-byte what collection used);
        # fall back to the Observation field the public API guarantees.
        if self.L is not None:
            Lp, Lq = self.L.end_effector.get_world_pose()
        else:
            Lp, Lq = obs.ee_pose_L
        # R EE pose: not in the public EnvInfo -> zero unless R_controller given.
        if self.R is not None:
            Rp, Rq = self.R.end_effector.get_world_pose()
        else:
            Rp, Rq = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        ratio = lambda v: float(np.clip(v / GRIPPER_OPEN_LIMIT, 0, 1))
        return np.concatenate([
            np.asarray(Lp).reshape(-1)[:3], np.asarray(Lq).reshape(-1)[:4],
            np.asarray(Rp).reshape(-1)[:3], np.asarray(Rq).reshape(-1)[:4],
            q[self._Li], q[self._Ri], qd[self._Li], qd[self._Ri],
            [ratio(q[self._Lg])],
            [ratio(q[self._Rg]) if self._Rg is not None else 0.0],
        ]).astype(np.float32)

    def act(self, obs: Observation):
        state = self._build_state(obs)
        self._send({
            "state": state,
            "head": _resize_rgb(obs.rgb.get("head")),
            "left": _resize_rgb(obs.rgb.get("L_wrist")),
            "right": _resize_rgb(obs.rgb.get("R_wrist")),
        })
        a = np.asarray(self._recv()["action"], np.float64)
        self._last_action = a
        # left arm slice: [x, y, z, rx, ry, rz, gripper_ratio]
        pos = a[:3]
        quat = _rotvec_to_quat_wxyz(a[3], a[4], a[5])
        grip = float(np.clip(a[6], 0, 1)) * GRIPPER_OPEN_LIMIT
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        return False  # harness advances on snap fire / per-part timeout

    def __del__(self):
        try:
            self._proc.terminate()
        except Exception:
            pass
