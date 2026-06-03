"""Participant policy template for the IROS 2026 vega_1u assembly challenge.

Copy this file to `policies/<your_name>.py` and fill in the methods. Run with:

    ${ISAAC_SIM}/python.sh task/run_pick_place.py --policy policies.<your_name>.MyPolicy

See `policy_api.py` for the full Observation / PartTarget / EnvInfo schema
and `policies/baseline_scripted.py` for the reference implementation.
"""
from __future__ import annotations

import os.path
import sys

# Ensure `task/` is on sys.path so `policy_api` imports work.
_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402


class MyPolicy(Policy):
    """Replace this with your policy.

    The harness instantiates exactly one of these per run and calls:

        __init__(env_info)               # once at sim startup
        for each part in part_order:
            reset(obs, target)           # part becomes active
            while not done:
                action = act(obs)        # every physics step (~60 Hz)
                done = is_done(obs)

    Returning False from `is_done` will not hang the harness — it also
    advances on snap-fired (snap parts) or per-part timeout.
    """

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        # One-time setup: cache joint name → dof index, load model weights,
        # build planners, etc.
        self._L_arm_idx = np.array(
            [env_info.dof_names.index(j) for j in env_info.L_arm_joints],
            dtype=np.int64,
        )
        self._L_gripper_idx = env_info.dof_names.index(env_info.L_gripper_joint)
        self._n_dof = len(env_info.dof_names)
        self._step_in_part = 0

    def reset(self, obs: Observation, target: PartTarget) -> None:
        """Called when a new part becomes the active target.

        `target` carries the part name, pick / place / grade poses, the
        release mode, snap target (if any), and per-part gripper open /
        close values. See `policy_api.PartTarget` for the full schema.
        """
        self._step_in_part = 0
        self._current_target = target
        # Plan your trajectory here, reset internal state, etc.

    def act(self, obs: Observation) -> ArticulationAction:
        """Produce the next L arm + L gripper action.

        Return an ArticulationAction whose `joint_positions` is a list of
        length `len(env_info.dof_names)` with `None` for dofs you don't
        command this step. The harness merges your L action with its R-arm
        hold pose, so R dofs in your action are ignored.

        This stub holds the arm at the current pose — replace with your
        own logic.
        """
        self._step_in_part += 1

        joint_targets = [None] * self._n_dof
        for idx in self._L_arm_idx:
            joint_targets[idx] = float(obs.joint_positions[idx])
        joint_targets[self._L_gripper_idx] = self._current_target.gripper_open

        return ArticulationAction(joint_positions=joint_targets)

    def is_done(self, obs: Observation) -> bool:
        """True when you consider the current part complete.

        For the stub, never "complete" on our own — let the harness's
        per-part timeout advance us. Replace with a real condition (e.g.
        gripper at place_pos, settled, etc.).
        """
        return False
