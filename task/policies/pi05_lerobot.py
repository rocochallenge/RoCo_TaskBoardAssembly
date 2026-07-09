"""Deploy a LeRobot Pi0.5 policy in the Isaac Sim harness."""
from __future__ import annotations

import os
import pickle
import struct
import subprocess

import numpy as np

from policy_api import EnvInfo, Observation, PartTarget, Policy

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRIPPER_OPEN_LIMIT = 0.6649704
_IMG_H, _IMG_W = 240, 320


def _resize_rgb(img):
    if img is None:
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[0] == _IMG_H and a.shape[1] == _IMG_W:
        return a.astype(np.uint8, copy=False)
    try:
        import cv2

        return cv2.resize(a, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        ys = max(1, a.shape[0] // _IMG_H)
        xs = max(1, a.shape[1] // _IMG_W)
        out = a[::ys, ::xs, :3]
        return out[:_IMG_H, :_IMG_W].astype(np.uint8)


def _rotvec_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_rotvec([rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class Pi05LeRobotPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self._proc = None
        self._err = None
        self.L = env_info.L_controller
        self.R = getattr(env_info, "R_controller", None)
        if self.L is None:
            raise ValueError("Pi05LeRobotPolicy requires env_info.L_controller")

        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("PI05_CKPT")
        if not ckpt:
            raise ValueError("set PI05_CKPT to a LeRobot Pi0.5 checkpoint")
        server_py = os.environ.get("PI05_SERVER_PY")
        if not server_py:
            raise ValueError("set PI05_SERVER_PY to the LeRobot venv python")
        server_script = os.environ.get("PI05_SERVER", os.path.join(_TASK_DIR, "pi05_server.py"))

        keep = (
            "HOME",
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "PI05_DEVICE",
            "PI05_TASK",
            "TOKENIZERS_PARALLELISM",
        )
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for key in keep:
            if key in os.environ:
                env[key] = os.environ[key]
        cuda_devices = os.environ.get("PI05_CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        log_path = os.environ.get("PI05_SERVER_LOG", os.path.join(_TASK_DIR, "pi05_server.log"))
        self._err = open(log_path, "w")
        self._proc = subprocess.Popen(
            [server_py, server_script, ckpt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            env=env,
            cwd=_TASK_DIR,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            self.close()
            raise RuntimeError("failed to open pi0.5 sidecar pipes")
        print(f"[pi05] spawned inference server (ckpt={ckpt})", flush=True)
        import time

        time.sleep(2)
        if self._proc.poll() is not None:
            self.close()
            raise RuntimeError(
                f"pi05_server died on startup (exit {self._proc.returncode}); see {log_path}"
            )

    def _send(self, obj):
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("pi05_server is not running")
        payload = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(payload)) + payload)
        self._proc.stdin.flush()

    def _recv(self):
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("pi05_server is not running")
        header = self._proc.stdout.read(4)
        if len(header) < 4:
            raise RuntimeError("pi05_server closed")
        size = struct.unpack(">I", header)[0]
        buf = b""
        while len(buf) < size:
            chunk = self._proc.stdout.read(size - len(buf))
            if not chunk:
                raise RuntimeError("pi05_server closed while sending response")
            buf += chunk
        return pickle.loads(buf)

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._send({"cmd": "reset", "task": os.environ.get("PI05_TASK")})
        reply = self._recv()
        if not reply.get("ok"):
            raise RuntimeError(f"pi05 reset failed: {reply!r}")

    def _build_state(self, obs: Observation) -> np.ndarray:
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        if self.L is not None:
            Lp, Lq = self.L.end_effector.get_world_pose()
        else:
            Lp, Lq = obs.ee_pose_L
        if self.R is not None:
            Rp, Rq = self.R.end_effector.get_world_pose()
        else:
            Rp, Rq = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def ratio(v):
            return float(np.clip(v / GRIPPER_OPEN_LIMIT, 0, 1))

        return np.concatenate(
            [
                np.asarray(Lp).reshape(-1)[:3],
                np.asarray(Lq).reshape(-1)[:4],
                np.asarray(Rp).reshape(-1)[:3],
                np.asarray(Rq).reshape(-1)[:4],
                q[self._Li],
                q[self._Ri],
                qd[self._Li],
                qd[self._Ri],
                [ratio(q[self._Lg])],
                [ratio(q[self._Rg]) if self._Rg is not None else 0.0],
            ]
        ).astype(np.float32)

    def act(self, obs: Observation):
        self._send(
            {
                "state": self._build_state(obs),
                "head": _resize_rgb(obs.rgb.get("head")),
                "left": _resize_rgb(obs.rgb.get("L_wrist")),
                "right": _resize_rgb(obs.rgb.get("R_wrist")),
                "task": os.environ.get("PI05_TASK", "assemble parts onto the task board"),
            }
        )
        action = np.asarray(self._recv()["action"], np.float64).reshape(-1)
        if action.size < 7:
            raise RuntimeError(f"pi0.5 action has {action.size} values, expected at least 7")
        if not np.isfinite(action[:7]).all():
            raise RuntimeError(f"pi0.5 action contains non-finite values: {action[:7]}")
        pos = action[:3]
        quat = _rotvec_to_quat_wxyz(action[3], action[4], action[5])
        grip = float(np.clip(action[6], 0, 1)) * GRIPPER_OPEN_LIMIT
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        return False

    def close(self):
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._proc = None
        try:
            if self._err is not None:
                self._err.close()
        except Exception:
            pass
        self._err = None

    def __del__(self):
        self.close()
