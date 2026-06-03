"""Policy interface for the IROS 2026 vega_1u assembly challenge.

Participants subclass `Policy` and implement `reset`, `act`, and `is_done`.
The harness (`task/run_pick_place.py`) drives the simulator and calls into
the policy as follows:

    1. Policy(env_info)                # once at sim startup
    2. for each part in pc.part_order:
           policy.reset(obs, target)   # part becomes active
           while True:
               action = policy.act(obs)
               # harness applies action, steps physics, builds next obs
               if policy.is_done(obs) or env_done:
                   break
       # harness scores the part, then advances

`env_done` is a harness-side condition: snap fired (snap-mode parts only)
or step count exceeded `pc.PER_PART_TIMEOUT_STEPS`. The harness will always
advance on these even if `is_done` keeps returning False.

See `policies/baseline_scripted.py` for the reference implementation and
`policies/template.py` for the participant stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvInfo:
    """One-shot info handed to `Policy.__init__` at sim startup.

    Indices into `dof_names` are stable for the lifetime of the run.
    """
    dof_names: List[str]
    """Full articulation dof name list (L arm, R arm, torso, lift, grippers)."""

    L_arm_joints: List[str]
    """Names of the seven L arm joints (j1..j7), in order."""

    R_arm_joints: List[str]
    """Names of the seven R arm joints (j1..j7), in order."""

    L_gripper_joint: str
    """Primary L gripper dof name (mimic joint commanded automatically by sim)."""

    L_arm_init_q: np.ndarray
    """Snapshot of L arm c-space at startup, after `INIT_JOINT_TARGETS` applied."""

    physics_dt: float
    """Seconds per physics step."""

    enable_camera_output: bool
    """If False, all entries in `Observation.rgb` / `depth` / `intrinsics` are None."""

    L_controller: Optional[Any] = None
    """Reserved for BaselinePolicy: the harness sets this to the L-arm
    `EEPoseController` instance so the baseline can wrap it in
    `EEPathFollower`. Participant policies that compute their own joint
    targets should ignore this — it's not part of the public Policy
    contract."""


@dataclass
class PartTarget:
    """Per-part info handed to `Policy.reset` at the start of each part.

    Coordinates are stage-frame, metres. Quaternions are wxyz.
    """
    name: str
    """Part name (matches a key in `param_config.PART_CONFIG`)."""

    release_mode: str
    """Either "snap" (part is rigid-joined to its slot once close enough) or
    "open" (part is dropped and gravity-settles)."""

    pick_pos: Optional[np.ndarray] = None
    """World-frame gripper target above the part's spawn pose. None means
    the part is not in the scene's init extract and the policy must locate
    it itself (rare; see PART_INIT_POSES coverage)."""

    spawn_orn: Optional[np.ndarray] = None
    """Spawn orientation of the part (wxyz)."""

    place_pos: Optional[np.ndarray] = None
    """World-frame gripper target where the part should be released."""

    grade_pos: Optional[np.ndarray] = None
    """Expected settled position used by the scorer for `open` parts (often
    differs from `place_pos` because the part falls/rolls after release).
    `None` for snap parts (snap fired/not-fired is the success signal)."""

    snap_target_pos: Optional[np.ndarray] = None
    """Mesh-frame world position the part must land within `snap_pos_tol` of
    to trigger the snap. Only set when `release_mode == 'snap'`."""

    snap_target_rot: Optional[np.ndarray] = None
    """Mesh-frame world rotation (wxyz) the part must reach within
    `snap_rot_tol_deg` of. Only set when `release_mode == 'snap'`."""

    snap_pos_tol: Optional[np.ndarray] = None
    """Per-axis position tolerance (3-vec, metres) for the snap gate."""

    snap_rot_tol_deg: Optional[float] = None
    """Rotation tolerance for the snap gate (degrees). Negative or None
    disables the rotation gate (axis-symmetric parts)."""

    gripper_open: float = 0.0
    """L gripper joint value commanded when "open" (part can be released)."""

    gripper_close: float = 0.0
    """L gripper joint value commanded when "close" (gripping the part)."""

    ee_orientation: Optional[np.ndarray] = None
    """Recommended EE orientation (wxyz) for picking/placing this part."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Full `PART_CONFIG` dict for this part. Provided verbatim so advanced
    policies can read any field the typed slots above don't surface
    (e.g. `ee_offset`, `init_height`, `transit_steps`, `sequence`)."""


@dataclass
class Observation:
    """Per-step snapshot passed to `Policy.act` and `Policy.is_done`.

    Arrays are read-only views into the harness's buffers — do not mutate.
    """
    step_idx: int
    """Physics step counter since sim start. Resets to 0 on `World.reset()`."""

    joint_positions: np.ndarray
    """Full dof vector (radians for revolute, metres for prismatic).
    Index with `env_info.dof_names`."""

    joint_velocities: np.ndarray
    """Full dof velocity vector. Same indexing as `joint_positions`."""

    L_gripper_position: float
    """Current value of the primary L gripper dof."""

    ee_pose_L: Tuple[np.ndarray, np.ndarray]
    """L end-effector pose in world frame, from Lula FK:
    (position[3], quaternion_wxyz[4])."""

    rgb: Dict[str, Optional[np.ndarray]]
    """Camera RGB frames as `{name: HxWx3 uint8}` for
    `name in {'head', 'L_wrist', 'R_wrist'}`. Value is None when
    `env_info.enable_camera_output` is False or the camera has no frame yet."""

    depth: Dict[str, Optional[np.ndarray]]
    """Camera depth frames as `{name: HxW float32}` in metres.
    Same keys / None semantics as `rgb`. None when `env_info.enable_camera_output`
    is False or the camera has no frame yet."""

    intrinsics: Dict[str, Optional[np.ndarray]]
    """Camera intrinsics as `{name: 3x3 float64}`. Same keys / None semantics."""

    snap_fired: bool = False
    """True if the env-side snap detector has already fired for the current
    part. Only meaningful for snap-mode parts; always False for `open` parts.
    The baseline policy uses this to advance past its snap_wait waypoint;
    other policies can ignore it (the harness will still advance the episode
    on snap fire)."""

    target_part: Optional[str] = None
    """Name of the part currently being placed (matches the most recent
    `reset()`'s `target.name`). `None` between parts."""


class Policy:
    """Base class for participant policies.

    Subclasses MUST override `reset`, `act`, and `is_done`. The default
    `__init__` just stores `env_info`; override it to load weights, parse
    configs, etc.

    The harness creates exactly one Policy instance per run. `reset` is
    called once per part (not per episode-restart) — if you need
    per-episode setup, do it in `reset` and key on `obs.step_idx == 0`.
    """

    def __init__(self, env_info: EnvInfo) -> None:
        self.env_info = env_info

    def reset(self, obs: Observation, target: PartTarget) -> None:
        """Called when a new part becomes the active target.

        Use this to plan trajectories, reset internal state, etc. The
        observation reflects the world state at the moment the part became
        active (gripper at home pose, arm at init c-space for the first
        part; arm at the previous part's end pose for subsequent parts).
        """
        raise NotImplementedError

    def act(self, obs: Observation):
        """Return the next L-arm ArticulationAction.

        The action's `joint_positions` should be a list of length
        `len(env_info.dof_names)` with `None` for dofs the policy doesn't
        command this step (R arm dofs are held by the harness regardless).
        Returning `None` is equivalent to "no command" — the previous
        commanded targets persist.
        """
        raise NotImplementedError

    def is_done(self, obs: Observation) -> bool:
        """True when the policy considers the current part finished.

        The harness will then call `reset()` with the next part (or end
        the episode). Returning False doesn't block the harness — snap
        fire (snap parts) and `PER_PART_TIMEOUT_STEPS` (any part) also
        advance the episode.
        """
        raise NotImplementedError
