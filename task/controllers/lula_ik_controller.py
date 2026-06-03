import os
import numpy as np

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction


# Joint order matches the `cspace:` list in each description yaml.
# Key: (side, owns_lift, owns_torso). On this rig torso_flip sits ABOVE
# Lift in the URDF chain, so torso ownership implies lift ownership;
# owns_torso=True with owns_lift=False is rejected at construction time.
_CSPACE_JOINT_NAMES = {
    ("L", True,  True):  ["Lift", "torso_flip"] + [f"L_arm_j{i}" for i in range(1, 8)],
    ("L", True,  False): ["Lift"]                + [f"L_arm_j{i}" for i in range(1, 8)],
    ("L", False, False):                            [f"L_arm_j{i}" for i in range(1, 8)],
    ("R", True,  True):  ["Lift", "torso_flip"] + [f"R_arm_j{i}" for i in range(1, 8)],
    ("R", True,  False): ["Lift"]                + [f"R_arm_j{i}" for i in range(1, 8)],
    ("R", False, False):                            [f"R_arm_j{i}" for i in range(1, 8)],
}


# URDF<->USD frame offset on the vega_1u EE link.
# Empirical relationship from the runtime diag:
#   stage_ee_world_orient = lula_target_orientation * R_offset
# with R_offset = (0, 0, 0, -1)  (180 deg about gripper local Z, the
# approach axis). We pre-multiply the user's requested target by the
# inverse of this offset before sending to Lula, so the stage prim ends up
# at exactly the user-requested orientation:
#   q_for_lula = q_user_target * R_offset_inverse
# R_offset is its own inverse rotation in SO(3); the conjugate (used here
# as the multiplicative inverse) is (0, 0, 0, +1).
_STAGE_OFFSET_INV = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _quat_mul(q1, q2):
    """Hamilton product of two quaternions in (w, x, y, z) form."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


class LulaIKController(BaseController):
    """Per-step Lula IK controller for one arm of the vega_1u robot.

    Drop-in replacement for the previous RMPFlowController: same
    ``forward(target_end_effector_position, target_end_effector_orientation)``
    interface, returns an ArticulationAction aligned to the full /vega_1u
    articulation DOF order (None for joints this controller does not own).
    """

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        side: str = "L",
        owns_torso: bool = True,
        owns_lift: bool = True,
        base_translation_offset: np.ndarray = None,
    ) -> None:
        super().__init__(name=name)
        side = side.upper()
        if side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        if owns_torso and not owns_lift:
            raise ValueError(
                "owns_torso=True requires owns_lift=True (torso_flip is "
                "above Lift in the URDF chain on this rig)."
            )

        # Suffix picks the right description yaml:
        #   ""          → 9-DOF (Lift + torso + arm) — owns_lift & owns_torso
        #   "_liftonly" → 8-DOF (Lift + arm)         — owns_lift only
        #   "_armonly"  → 7-DOF (arm only)           — neither
        if owns_torso:
            suffix = ""
        elif owns_lift:
            suffix = "_liftonly"
        else:
            suffix = "_armonly"
        base_dir = os.path.abspath(os.path.dirname(__file__))
        robot_description_path = os.path.join(
            base_dir, f"vega_1u_{side}_arm_description{suffix}.yaml"
        )
        urdf_path = os.path.abspath(
            os.path.join(base_dir, "..", "..", "robot", "vega_1u_gripper.urdf")
        )

        self._ee_frame = f"{side}_ee_link_gripper_link"
        self._ik = mg.LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
        )
        # Tolerances must be at least as tight as the follower's POS_TOL /
        # ORN_TOL — otherwise IK reports success at, say, 5 mm and the
        # follower waits forever for 1 mm convergence the IK never tried
        # for. 0.3 deg / 0.5 mm was too strict and Lula returned
        # success=False on reachable hover targets, so 1 mm / 3 deg is the
        # current middle ground. If you see ik_ok=False on reachable
        # waypoints, loosen these (and POS_TOL / ORN_TOL with them).
        self._position_tolerance = 1e-3      # 1 mm
        self._orientation_tolerance = 5e-2   # ~3 deg

        self._articulation = robot_articulation
        self._cspace_joint_names = _CSPACE_JOINT_NAMES[(side, owns_lift, owns_torso)]

        dof_names = list(robot_articulation.dof_names)
        self._dof_index = {n: i for i, n in enumerate(dof_names)}
        self._n_dof = len(dof_names)
        self._cspace_dof_indices = np.array(
            [self._dof_index[j] for j in self._cspace_joint_names], dtype=np.int64
        )

        self._default_position, self._default_orientation = (
            robot_articulation.get_world_pose()
        )
        # Optional translational correction for the base pose passed to Lula
        # (e.g. to compensate for a stage-level ground/parent shift that
        # robot.get_world_pose() doesn't reflect).
        if base_translation_offset is not None:
            self._default_position = (
                np.asarray(self._default_position, dtype=np.float64)
                + np.asarray(base_translation_offset, dtype=np.float64).reshape(-1)
            )
        self._ik.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

        self._last_full = [None] * self._n_dof
        # Warm-start the next IK solve from the *current actual* joint state
        # so the solver is bounded to a solution near where the robot
        # physically is. Using the previous IK solution as warm-start drifts
        # along the redundant null space near the workspace edge — the
        # solver iteratively walks the elbow through different
        # configurations even at the same target, and the PD chases each
        # one, causing wild EE swings. Seeding from physical state damps
        # that drift at the cost of some IK lag when physics is slow.
        self._last_cspace_q = None
        # Diagnostic state.
        self.ik_ok = True
        self._last_target_position = None
        self._last_target_orientation = None

    def reset(self) -> None:
        super().reset()
        self._ik.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
        self._last_full = [None] * self._n_dof
        self._last_cspace_q = None

    def _current_cspace_q(self) -> np.ndarray:
        q_full = np.asarray(self._articulation.get_joint_positions()).reshape(-1)
        return q_full[self._cspace_dof_indices].astype(np.float64)

    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: np.ndarray = None,
    ) -> ArticulationAction:
        warm_start = self._current_cspace_q()
        target_orn = (
            np.asarray(target_end_effector_orientation, dtype=np.float64)
            if target_end_effector_orientation is not None
            else None
        )
        # Pre-compose the inverse URDF<->USD frame offset so the stage prim
        # (what end_effector.get_world_pose() reports) ends up at exactly
        # `target_orn`, not at `target_orn * R_offset`.
        target_orn_for_lula = (
            _quat_mul(target_orn, _STAGE_OFFSET_INV)
            if target_orn is not None
            else None
        )
        action, success = self._ik.compute_inverse_kinematics(
            frame_name=self._ee_frame,
            target_position=np.asarray(target_end_effector_position, dtype=np.float64),
            target_orientation=target_orn_for_lula,
            warm_start=warm_start,
            position_tolerance=self._position_tolerance,
            orientation_tolerance=self._orientation_tolerance,
        )
        self.ik_ok = bool(success) and action is not None
        if not self.ik_ok:
            return ArticulationAction(joint_positions=list(self._last_full))

        if isinstance(action, np.ndarray):
            q_cspace = np.asarray(action, dtype=np.float64).reshape(-1)
        else:
            q_cspace = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)

        full = [None] * self._n_dof
        for jname, val in zip(self._cspace_joint_names, q_cspace.tolist()):
            full[self._dof_index[jname]] = float(val)
        self._last_full = list(full)
        self._last_cspace_q = q_cspace.astype(np.float64)
        self._last_target_position = np.asarray(target_end_effector_position, dtype=np.float64).copy()
        self._last_target_orientation = (
            target_orn.copy() if target_orn is not None else None
        )
        return ArticulationAction(joint_positions=full)

    def solve(self, target_position, target_orientation=None, seed=None):
        """One-shot IK, returns (q_cspace ndarray, success). Does NOT mutate
        ``_last_cspace_q`` / ``_last_full`` — use when you need a probe
        solution (e.g., the path follower's joint-lerp endpoint anchor)
        without disturbing the main IK chain."""
        target_orn = (
            np.asarray(target_orientation, dtype=np.float64)
            if target_orientation is not None
            else None
        )
        target_orn_for_lula = (
            _quat_mul(target_orn, _STAGE_OFFSET_INV)
            if target_orn is not None
            else None
        )
        warm = (
            np.asarray(seed, dtype=np.float64)
            if seed is not None
            else self._current_cspace_q()
        )
        action, success = self._ik.compute_inverse_kinematics(
            frame_name=self._ee_frame,
            target_position=np.asarray(target_position, dtype=np.float64),
            target_orientation=target_orn_for_lula,
            warm_start=warm,
            position_tolerance=self._position_tolerance,
            orientation_tolerance=self._orientation_tolerance,
        )
        if not success or action is None:
            return None, False
        if isinstance(action, np.ndarray):
            q = action
        else:
            q = action.joint_positions
        return np.asarray(q, dtype=np.float64).reshape(-1), True

    @property
    def cspace_joint_names(self):
        return list(self._cspace_joint_names)

    def current_cspace_q(self) -> np.ndarray:
        return self._current_cspace_q()

    def fk_for_last_command(self):
        """Lula FK on the last commanded q. Returns (pos, orn_quat_wxyz) or (None, None)."""
        if self._last_cspace_q is None:
            return None, None
        fk_fn = getattr(self._ik, "compute_forward_kinematics", None)
        if fk_fn is None:
            return None, None
        pos, rot = fk_fn(self._ee_frame, self._last_cspace_q)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        rot = np.asarray(rot, dtype=np.float64)
        if rot.shape != (3, 3):
            return pos, rot.reshape(-1)[:4].astype(np.float64)
        m = rot
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return pos, np.array([w, x, y, z], dtype=np.float64)
