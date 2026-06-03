"""End-effector pose controller and path follower.

Replaces the pick-and-place state machine with a simpler abstraction: the
caller supplies a target EE pose + gripper command and the controller asks
Lula IK for the joint configuration that places the EE there. Each call
returns one ArticulationAction; PD physics drives the joints toward it
over the next few steps. A waypoint follower is layered on top to walk
through a list of (pos, orn, gripper) tuples, advancing when the actual
EE pose enters position AND orientation tolerance.

Coordinate convention: positions are world-frame xyz; orientations are
unit quaternions in (w, x, y, z) form.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Optional

import numpy as np

from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.types import ArticulationAction

from .lula_ik_controller import LulaIKController


Waypoint = namedtuple(
    "Waypoint",
    ["pos", "orn", "gripper", "settle_steps", "name", "lock_pose",
     "joint_lerp_t", "advance_when", "timeout_steps", "cspace_target",
     "cspace_tol"],
    defaults=(False, None, None, None, None, None),
)
Waypoint.__doc__ = (
    "EE pose waypoint. ``settle_steps`` is the number of EEPathFollower.step()"
    " calls the EE must remain in pos+orn tolerance before the follower"
    " advances; default 0 means advance on the first in-tolerance step."
    " ``lock_pose=True`` tells the follower to ignore ``pos`` and instead"
    " snapshot the actual EE pose on the first step at this waypoint, using"
    " the snapshot as the IK target for the duration (so the arm holds still"
    " while e.g. the gripper closes)."
    " ``joint_lerp_t`` (None or float in (0, 1)) marks the waypoint as a"
    " joint-space-interpolated transit step: the follower computes q_A from"
    " the actual robot state when entering the segment, solves IK once at"
    " the first non-lerp waypoint after the segment to get q_B, and emits"
    " ``q(t) = (1 - t) q_A + t q_B`` directly (no IK at this waypoint). This"
    " guarantees c-space continuity at the cost of a non-straight cartesian"
    " EE path. ``pos``/``orn`` on a joint-lerp waypoint are kept only for"
    " diagnostics (they show the cartesian lerp midpoint)."
    " ``advance_when`` (Callable[[], bool] or None) replaces the pos/orn"
    " tolerance gate with a custom predicate, evaluated once per step()."
    " When set, the waypoint advances on the first step the callable returns"
    " True (after ``settle_steps`` additional in-gate steps, if > 0). Use"
    " together with ``lock_pose=True`` to hold the EE still while waiting"
    " for an external event (e.g., a snap_attach FixedJoint authoring)."
    " ``timeout_steps`` (int or None) caps how long the follower waits at"
    " an ``advance_when``-gated waypoint: after this many total step() calls,"
    " the follower logs a warning and advances anyway. None = wait forever."
    " ``cspace_target`` (numpy 1-D float array or None) bypasses IK entirely:"
    " the follower emits ``forward_raw_q(cspace_target, gripper)`` each step"
    " and advances when actual c-space q is within ``cspace_tolerance`` of"
    " the target. ``pos`` / ``orn`` are ignored when ``cspace_target`` is"
    " set (pass any dummy). Used for return-to-home between parts so every"
    " part's first IK call gets the same c-space seed."
    " ``cspace_tol`` (float or None) overrides the follower's default"
    " ``cspace_tolerance`` for this waypoint only. Used on the return_home"
    " waypoint to enforce a much tighter gate (so the IK seed at hover_pick"
    " is bit-close to the standalone-sim seed) without tightening the"
    " joint_lerp transit gates, which need to stay loose."
)


def _quat_angle(q1, q2):
    """Smallest rotation angle (radians) between two unit quaternions (wxyz)."""
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return 2.0 * float(np.arccos(d))


def _approach_axis_world(quat_wxyz):
    """Return the gripper's local +Z axis expressed in world coords.

    For a quat (w, x, y, z), gripper +Z rotates to:
        (2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2))
    For a top-down grasp this should be roughly (0, 0, -1).
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    w, x, y, z = q
    return np.array([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], dtype=np.float64)


def _approach_axis_angle(q1, q2):
    """Angle (radians) between the gripper +Z axes implied by q1 and q2.

    Use this instead of _quat_angle when only the approach direction
    matters (e.g., grasping a cylindrically symmetric object); it ignores
    wrist roll about the approach axis.
    """
    a = _approach_axis_world(q1)
    b = _approach_axis_world(q2)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(d))


class EEPoseController(BaseController):
    """One-shot EE-pose + gripper controller.

    forward(target_pos, target_orn, gripper_cmd) returns an
    ArticulationAction whose joint_positions are aligned to the full
    /vega_1u articulation DOF order. None entries mean "don't drive this
    joint"; the PD controller will hold its previous target.

    gripper_cmd:
      - "open"  -> gripper joints commanded to opened positions.
      - "close" -> gripper joints commanded to closed positions.
      - float v -> primary gripper joint commanded to v (radians); the
                   USD mimic constraint propagates to the secondary joint.
      - None    -> no gripper change (gripper joints left at None).
    """

    def __init__(
        self,
        name: str,
        robot_articulation,
        side: str = "L",
        owns_torso: bool = True,
        owns_lift: bool = True,
        base_translation_offset=None,
    ) -> None:
        super().__init__(name=name)
        side = side.upper()
        if side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        self._side = side
        self._robot = robot_articulation
        self._gripper = getattr(robot_articulation, "gripper", None)

        self._ik = LulaIKController(
            name=name + "_ik",
            robot_articulation=robot_articulation,
            side=side,
            owns_torso=owns_torso,
            owns_lift=owns_lift,
            base_translation_offset=base_translation_offset,
        )

        dof_names = list(robot_articulation.dof_names)
        self._dof_names = dof_names
        self._dof_index = {n: i for i, n in enumerate(dof_names)}
        self._n_dof = len(dof_names)

        self._gripper_joint_names = [f"{side}_gripper_joint", f"{side}_gripper_joint_01"]
        self._gripper_dof_indices = [
            self._dof_index[j] for j in self._gripper_joint_names if j in self._dof_index
        ]

    # ----- accessors for diagnostics -----
    @property
    def ik(self) -> LulaIKController:
        return self._ik

    @property
    def end_effector(self):
        return self._robot.end_effector

    # ----- joint-lerp helpers (used by EEPathFollower transit segments) -----
    def current_cspace_q(self) -> np.ndarray:
        """Actual current c-space joint vector (size = len(cspace_joint_names))."""
        return self._ik.current_cspace_q()

    def cspace_joint_names(self):
        """Order in which raw c-space q must be supplied to forward_raw_q."""
        return self._ik.cspace_joint_names

    def solve_q(self, target_position, target_orientation=None, seed=None):
        """One-shot IK probe; returns (q, success) without mutating IK state."""
        return self._ik.solve(target_position, target_orientation, seed)

    def forward_raw_q(self, q_cspace, gripper_cmd=None) -> ArticulationAction:
        """Emit an ArticulationAction from raw c-space joint values, bypassing
        IK entirely. Used for joint-space-interpolated transit waypoints
        where running IK at each midpoint would let the solver pick
        different solution branches (breaking c-space continuity)."""
        q_cspace = np.asarray(q_cspace, dtype=np.float64).reshape(-1)
        full = [None] * self._n_dof
        for jname, val in zip(self._ik.cspace_joint_names, q_cspace.tolist()):
            full[self._dof_index[jname]] = float(val)

        if gripper_cmd is None:
            return ArticulationAction(joint_positions=full)

        if isinstance(gripper_cmd, str):
            cmd = gripper_cmd.lower()
            if cmd == "open":
                gripper_action = self._gripper.forward(action="open")
            elif cmd == "close":
                gripper_action = self._gripper.forward(action="close")
            else:
                raise ValueError(
                    f"gripper_cmd string must be 'open' or 'close'; got {gripper_cmd!r}"
                )
            gp = list(gripper_action.joint_positions or [])
            n = min(len(gp), self._n_dof)
            for i in range(n):
                if gp[i] is not None:
                    full[i] = float(gp[i])
        else:
            try:
                val = float(gripper_cmd)
            except (TypeError, ValueError):
                raise ValueError(
                    f"gripper_cmd must be 'open', 'close', float, or None; got {gripper_cmd!r}"
                )
            for idx in self._gripper_dof_indices:
                full[idx] = val
        return ArticulationAction(joint_positions=full)

    # ----- forward -----
    def forward(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = None,
        gripper_cmd=None,
    ) -> ArticulationAction:
        ik_action = self._ik.forward(target_position, target_orientation)
        if gripper_cmd is None:
            return ik_action

        # Build/merge gripper joint commands on top of IK joints.
        ik_positions = list(ik_action.joint_positions or [None] * self._n_dof)

        if isinstance(gripper_cmd, str):
            cmd = gripper_cmd.lower()
            if cmd == "open":
                gripper_action = self._gripper.forward(action="open")
            elif cmd == "close":
                gripper_action = self._gripper.forward(action="close")
            else:
                raise ValueError(
                    f"gripper_cmd string must be 'open' or 'close'; got {gripper_cmd!r}"
                )
            gp = list(gripper_action.joint_positions or [])
            # Gripper actions from ParallelGripper are typically aligned to the
            # full articulation DOF order; merge non-None entries onto IK.
            n = min(len(gp), self._n_dof)
            for i in range(n):
                if gp[i] is not None:
                    ik_positions[i] = float(gp[i])
        else:
            try:
                val = float(gripper_cmd)
            except (TypeError, ValueError):
                raise ValueError(
                    f"gripper_cmd must be 'open', 'close', float, or None; got {gripper_cmd!r}"
                )
            for idx in self._gripper_dof_indices:
                ik_positions[idx] = val

        return ArticulationAction(joint_positions=ik_positions)

    def reset(self) -> None:
        super().reset()
        self._ik.reset()


class EEPathFollower:
    """Walks an EE pose path; advances waypoints by pos AND orn tolerance.

    Waypoints are 3-tuples ``(target_position, target_orientation,
    gripper_cmd)``. A waypoint is considered reached when the actual EE
    world pose is within ``position_tolerance`` (meters) and
    ``orientation_tolerance`` (radians) of the target — at which point the
    next waypoint becomes active on the following ``step()`` call.

    The controller's IK keeps targeting the *current* waypoint each step;
    physics converges over many steps. Once all waypoints are reached the
    follower marks itself done and emits a no-op action (all-None joints).
    """

    def __init__(
        self,
        ee_controller: EEPoseController,
        position_tolerance: float = 0.01,   # 1 cm
        orientation_tolerance: float = 0.05,  # ~3 deg
        cspace_tolerance: float = 0.05,     # ~3 deg per-joint; for joint-lerp wps
        default_timeout_steps: Optional[int] = None,
    ) -> None:
        self._ctrl = ee_controller
        self._pos_tol = float(position_tolerance)
        self._orn_tol = float(orientation_tolerance)
        self._cspace_tol = float(cspace_tolerance)
        # Per-step fallback timeout applied to any waypoint whose own
        # ``timeout_steps`` is None — protects against PD never quite
        # tracking the IK target (gate pos/orn within tol per FK but not
        # per physics) from deadlocking the follower. None = no fallback.
        self._default_timeout_steps = (
            int(default_timeout_steps) if default_timeout_steps is not None
            else None
        )
        self._waypoints: List[Waypoint] = []
        self._idx = 0
        self._done = False
        # Number of consecutive step() calls the EE has been within both
        # tolerances at the current waypoint (resets on advance / drift).
        self._steps_in_tol = 0
        # Total step() calls at the current waypoint regardless of gate
        # state. Used to enforce ``timeout_steps`` on advance_when-gated
        # waypoints; resets on advance and on path set/reset.
        self._steps_at_wp = 0
        # Per-step diagnostic state.
        self.last_pos_err = None
        self.last_orn_err = None
        # Snapshotted target for lock_pose waypoints (cleared on advance).
        self._locked_pos = None
        self._locked_orn = None
        self._locked_idx = -1
        # Joint-lerp segment anchors. Held across all consecutive joint_lerp
        # waypoints; cleared when the follower advances to a non-lerp wp.
        self._lerp_q_A = None
        self._lerp_q_B = None
        self._lerp_seg_idx = -1   # index of the first joint_lerp wp in the
                                  # current segment, -1 if no active segment.

    def set_path(self, waypoints) -> None:
        self._waypoints = [self._normalize_wp(w) for w in waypoints]
        self._idx = 0
        self._done = len(self._waypoints) == 0
        self._steps_in_tol = 0
        self._steps_at_wp = 0
        self.last_pos_err = None
        self.last_orn_err = None
        self._locked_pos = None
        self._locked_orn = None
        self._locked_idx = -1
        self._lerp_q_A = None
        self._lerp_q_B = None
        self._lerp_seg_idx = -1

    def reset(self) -> None:
        self._idx = 0
        self._done = len(self._waypoints) == 0
        self._steps_in_tol = 0
        self._steps_at_wp = 0
        self.last_pos_err = None
        self.last_orn_err = None
        self._locked_pos = None
        self._locked_orn = None
        self._locked_idx = -1
        self._lerp_q_A = None
        self._lerp_q_B = None
        self._lerp_seg_idx = -1
        self._ctrl.reset()

    @staticmethod
    def _normalize_wp(w) -> Waypoint:
        if isinstance(w, Waypoint):
            return w
        if isinstance(w, dict):
            jlt = w.get("joint_lerp_t")
            timeout = w.get("timeout_steps")
            cspace = w.get("cspace_target")
            cs_tol = w.get("cspace_tol")
            return Waypoint(
                pos=np.asarray(w["pos"], dtype=np.float64),
                orn=(np.asarray(w["orn"], dtype=np.float64)
                     if w.get("orn") is not None else None),
                gripper=w.get("gripper"),
                settle_steps=int(w.get("settle_steps", 0)),
                name=w.get("name"),
                lock_pose=bool(w.get("lock_pose", False)),
                joint_lerp_t=(None if jlt is None else float(jlt)),
                advance_when=w.get("advance_when"),
                timeout_steps=(None if timeout is None else int(timeout)),
                cspace_target=(None if cspace is None
                               else np.asarray(cspace, dtype=np.float64)),
                cspace_tol=(None if cs_tol is None else float(cs_tol)),
            )
        # Tuple forms: 2/3/4/5/6/7 elements.
        if not (2 <= len(w) <= 7):
            raise ValueError(
                f"waypoint must be Waypoint, dict, or tuple of length 2-7; got {w!r}"
            )
        pos = np.asarray(w[0], dtype=np.float64)
        orn = np.asarray(w[1], dtype=np.float64) if w[1] is not None else None
        gripper = w[2] if len(w) >= 3 else None
        settle = int(w[3]) if len(w) >= 4 else 0
        name = w[4] if len(w) >= 5 else None
        lock_pose = bool(w[5]) if len(w) >= 6 else False
        joint_lerp_t = (None if len(w) < 7 or w[6] is None else float(w[6]))
        return Waypoint(pos=pos, orn=orn, gripper=gripper,
                        settle_steps=settle, name=name, lock_pose=lock_pose,
                        joint_lerp_t=joint_lerp_t)

    def is_done(self) -> bool:
        return self._done

    def current_index(self) -> int:
        return self._idx

    def current_waypoint(self) -> Optional[Waypoint]:
        if self._done or self._idx >= len(self._waypoints):
            return None
        return self._waypoints[self._idx]

    def num_waypoints(self) -> int:
        return len(self._waypoints)

    def steps_in_tolerance(self) -> int:
        return self._steps_in_tol

    # ---------------- joint-lerp segment helpers ----------------
    def _seg_first_idx(self, idx: int) -> int:
        """First joint_lerp waypoint of the segment containing idx."""
        i = idx
        while i > 0 and self._waypoints[i - 1].joint_lerp_t is not None:
            i -= 1
        return i

    def _seg_anchor_after(self, idx: int):
        """First non-joint_lerp waypoint at-or-after idx that has pos/orn we
        can run IK against (anchor wp for q_B). Returns (anchor_idx, wp) or
        (None, None) if no such waypoint exists."""
        n = len(self._waypoints)
        i = idx
        while i < n and self._waypoints[i].joint_lerp_t is not None:
            i += 1
        if i >= n:
            return None, None
        return i, self._waypoints[i]

    def _ensure_lerp_anchors(self, wp: Waypoint) -> bool:
        """On entry to a new joint_lerp segment, snapshot q_A (current
        robot state) and solve IK once at the segment's exit anchor to get
        q_B. Returns False if no anchor exists or IK at q_B fails."""
        seg_idx = self._seg_first_idx(self._idx)
        if self._lerp_seg_idx == seg_idx and self._lerp_q_A is not None:
            return True
        q_A = np.asarray(self._ctrl.current_cspace_q(), dtype=np.float64).reshape(-1)
        anchor_idx, anchor_wp = self._seg_anchor_after(self._idx)
        if anchor_wp is None:
            return False
        q_B, ok = self._ctrl.solve_q(anchor_wp.pos, anchor_wp.orn, seed=q_A)
        if not ok or q_B is None:
            return False
        self._lerp_q_A = q_A
        self._lerp_q_B = np.asarray(q_B, dtype=np.float64).reshape(-1)
        self._lerp_seg_idx = seg_idx
        return True

    def _clear_lerp_state(self) -> None:
        self._lerp_q_A = None
        self._lerp_q_B = None
        self._lerp_seg_idx = -1

    def step(self) -> ArticulationAction:
        if self._done:
            return ArticulationAction(joint_positions=[None] * self._ctrl._n_dof)

        wp = self._waypoints[self._idx]

        ee = self._ctrl.end_effector
        actual_pos, actual_orn = ee.get_world_pose()
        actual_pos = np.asarray(actual_pos, dtype=np.float64).reshape(-1)
        actual_orn = np.asarray(actual_orn, dtype=np.float64).reshape(-1)

        # ---------------- cspace-target branch ----------------
        # Waypoints with cspace_target set bypass IK entirely: the
        # follower drives the named c-space joints directly to the target
        # vector via forward_raw_q. Used for return-to-home between parts
        # so the next part's first IK call gets the same c-space seed it
        # would get if that part were run alone — fixes the IK-branch
        # divergence that otherwise makes part 2 reach a worse posture
        # than running it standalone.
        if wp.cspace_target is not None:
            q_target = np.asarray(wp.cspace_target,
                                  dtype=np.float64).reshape(-1)
            q_actual = np.asarray(self._ctrl.current_cspace_q(),
                                  dtype=np.float64).reshape(-1)
            cspace_err = float(np.max(np.abs(q_actual - q_target)))
            # Repurpose pos_err for the cspace gap so diagnostics stay
            # populated; orn_err is irrelevant for a c-space target.
            self.last_pos_err = cspace_err
            self.last_orn_err = 0.0
            self._steps_at_wp += 1
            tol = wp.cspace_tol if wp.cspace_tol is not None else self._cspace_tol
            gate_ok = cspace_err <= tol
            # Timeout fall-through: if the tolerance is set below the PD
            # controller's steady-state floor the gate would never fire and
            # the follower would deadlock. wp.timeout_steps (snap_wait
            # etc.) wins if set; otherwise self._default_timeout_steps acts
            # as a global safety net.
            eff_timeout = (wp.timeout_steps if wp.timeout_steps is not None
                           else self._default_timeout_steps)
            if (not gate_ok
                    and eff_timeout is not None
                    and self._steps_at_wp > eff_timeout):
                print(
                    f"[follower] WARN: waypoint {wp.name!r} cspace gate "
                    f"timed out at {self._steps_at_wp} steps "
                    f"(err={cspace_err:.2e} > tol={tol:.2e}); advancing anyway."
                )
                gate_ok = True
            if gate_ok:
                self._steps_in_tol += 1
                if self._steps_in_tol > wp.settle_steps:
                    self._idx += 1
                    self._steps_in_tol = 0
                    self._steps_at_wp = 0
                    if self._idx >= len(self._waypoints):
                        self._done = True
                        return ArticulationAction(
                            joint_positions=[None] * self._ctrl._n_dof
                        )
            else:
                self._steps_in_tol = 0
            return self._ctrl.forward_raw_q(q_target, wp.gripper)

        # ---------------- joint-lerp branch ----------------
        # Transit waypoints with joint_lerp_t set: bypass IK and emit a raw
        # c-space lerp between the actual q at segment entry (q_A) and the
        # IK solution at the segment's exit anchor (q_B). Advance is gated
        # on per-joint cspace tolerance.
        if wp.joint_lerp_t is not None:
            if not self._ensure_lerp_anchors(wp):
                # Anchor IK failed or no anchor — fall back to standard IK
                # on the cartesian lerp midpoint so we don't hang the path.
                return self._ctrl.forward(wp.pos, wp.orn, wp.gripper)

            t = float(wp.joint_lerp_t)
            q_target = (1.0 - t) * self._lerp_q_A + t * self._lerp_q_B
            q_actual = np.asarray(self._ctrl.current_cspace_q(),
                                  dtype=np.float64).reshape(-1)
            cspace_err = float(np.max(np.abs(q_actual - q_target)))
            # Diagnostic fields: keep last_pos_err/last_orn_err meaningful
            # by reporting against the cartesian lerp pos. Won't converge
            # to zero (cartesian path isn't straight), but it's a useful
            # progress indicator.
            self.last_pos_err = float(np.linalg.norm(actual_pos - wp.pos))
            self.last_orn_err = (
                _quat_angle(actual_orn, wp.orn) if wp.orn is not None else 0.0
            )

            if cspace_err <= self._cspace_tol:
                self._steps_in_tol += 1
                if self._steps_in_tol > wp.settle_steps:
                    self._idx += 1
                    self._steps_in_tol = 0
                    self._steps_at_wp = 0
                    if self._idx >= len(self._waypoints):
                        self._done = True
                        self._clear_lerp_state()
                        return ArticulationAction(
                            joint_positions=[None] * self._ctrl._n_dof
                        )
                    next_wp = self._waypoints[self._idx]
                    if next_wp.joint_lerp_t is None:
                        # Leaving the lerp segment; clear anchors so the
                        # next segment (if any) re-snapshots.
                        self._clear_lerp_state()
            else:
                self._steps_in_tol = 0

            return self._ctrl.forward_raw_q(q_target, wp.gripper)

        # Non-lerp waypoint: clear any stale lerp anchors so a subsequent
        # transit segment will re-snapshot from the actual robot state.
        if self._lerp_seg_idx != -1:
            self._clear_lerp_state()

        # On first step at a lock_pose waypoint, snapshot the actual EE
        # pose and treat that as the target for both error checks and IK.
        # Holds the arm still while gripper acts (e.g., phases 3/4).
        if wp.lock_pose and self._locked_idx != self._idx:
            self._locked_pos = actual_pos.copy()
            self._locked_orn = (
                actual_orn.copy() if wp.orn is not None else None
            )
            self._locked_idx = self._idx

        target_pos = self._locked_pos if wp.lock_pose else wp.pos
        target_orn = (
            self._locked_orn if (wp.lock_pose and self._locked_orn is not None)
            else wp.orn
        )

        pos_err = float(np.linalg.norm(actual_pos - target_pos))
        orn_err = (
            _quat_angle(actual_orn, target_orn) if target_orn is not None else 0.0
        )
        self.last_pos_err = pos_err
        self.last_orn_err = orn_err

        # Determine the advance gate. ``advance_when`` (e.g. snap_wait
        # waiting on a SnapAttacher) replaces the pos/orn-tolerance gate;
        # pos/orn errors remain in the diagnostic fields but don't gate
        # advance. ``timeout_steps`` is the safety fall-through.
        self._steps_at_wp += 1
        if wp.advance_when is not None:
            gate_ok = bool(wp.advance_when())
            if (not gate_ok
                    and wp.timeout_steps is not None
                    and self._steps_at_wp > wp.timeout_steps):
                print(
                    f"[follower] WARN: waypoint {wp.name!r} timed out at "
                    f"{self._steps_at_wp} steps (gate never fired); "
                    f"advancing anyway."
                )
                gate_ok = True
        else:
            gate_ok = (pos_err <= self._pos_tol and orn_err <= self._orn_tol)
            # Same timeout fall-through as cspace branch: if FK reports the
            # IK target as hit but the physics EE drifts outside POS_TOL /
            # ORN_TOL forever (steady-state PD lag), force-advance after
            # the effective timeout. wp.timeout_steps wins if set.
            eff_timeout = (wp.timeout_steps if wp.timeout_steps is not None
                           else self._default_timeout_steps)
            if (not gate_ok
                    and eff_timeout is not None
                    and self._steps_at_wp > eff_timeout):
                print(
                    f"[follower] WARN: waypoint {wp.name!r} pos/orn gate "
                    f"timed out at {self._steps_at_wp} steps "
                    f"(pos_err={pos_err * 1000:.2f} mm, "
                    f"orn_err={np.degrees(orn_err):.2f} deg); advancing anyway."
                )
                gate_ok = True

        if gate_ok:
            self._steps_in_tol += 1
            if self._steps_in_tol > wp.settle_steps:
                # Advance to the next waypoint.
                self._idx += 1
                self._steps_in_tol = 0
                self._steps_at_wp = 0
                self._locked_pos = None
                self._locked_orn = None
                # Lookahead: keep consuming consecutive waypoints whose
                # advance_when gate fires immediately. Typical case:
                # after a snap_search cell fires, every remaining cell's
                # gate returns True on first call (because snap_when()
                # is now True). Without this we'd burn one follower step
                # per leftover cell. Only safe for waypoints that:
                #   - use advance_when (custom gate);
                #   - require no dwell (settle_steps == 0);
                #   - aren't joint_lerp / cspace_target / lock_pose
                #     (those have special handling we shouldn't skip).
                # Prefer the gate's side-effect-free ``peek_skip()`` when
                # exposed (snap_search gates do this) so we don't burn
                # per-cell dwell counters during lookahead.
                while self._idx < len(self._waypoints):
                    nxt = self._waypoints[self._idx]
                    if (nxt.advance_when is None
                            or nxt.settle_steps > 0
                            or nxt.joint_lerp_t is not None
                            or nxt.cspace_target is not None
                            or nxt.lock_pose):
                        break
                    peek = getattr(nxt.advance_when, "peek_skip", None)
                    if peek is not None:
                        if not peek():
                            break
                    elif not bool(nxt.advance_when()):
                        break
                    self._idx += 1
                if self._idx >= len(self._waypoints):
                    self._done = True
                    return ArticulationAction(joint_positions=[None] * self._ctrl._n_dof)
                wp = self._waypoints[self._idx]
                if wp.joint_lerp_t is not None:
                    # Entering a joint-lerp segment on the same step — let
                    # the lerp branch handle the next call cleanly.
                    return self._ctrl.forward_raw_q(
                        np.asarray(self._ctrl.current_cspace_q(),
                                   dtype=np.float64).reshape(-1),
                        wp.gripper,
                    )
                if wp.lock_pose:
                    # Snapshot for the new wp on the same step so we don't
                    # send one IK call to the old target before locking.
                    self._locked_pos = actual_pos.copy()
                    self._locked_orn = (
                        actual_orn.copy() if wp.orn is not None else None
                    )
                    self._locked_idx = self._idx
                    target_pos = self._locked_pos
                    target_orn = self._locked_orn if wp.orn is not None else wp.orn
                else:
                    target_pos = wp.pos
                    target_orn = wp.orn
        else:
            # Gate not satisfied — restart the dwell counter for this wp.
            self._steps_in_tol = 0

        return self._ctrl.forward(target_pos, target_orn, wp.gripper)


# ---------------------------------------------------------------------------
# Higher-level path builder: 8-phase pick-and-place from a few key inputs.
# ---------------------------------------------------------------------------
def _make_search_gate(snap_when, dwell_steps):
    """Per-cell gate for the snap_search XY sweep. Returns True on the
    first step ``snap_when()`` is True (early termination once the snap
    fires), otherwise advances the cell after exactly ``dwell_steps``
    calls. Each cell uses a fresh closure with its own counter.

    Exposes ``gate.peek_skip()`` for the follower's lookahead — a
    side-effect-free check that returns True ONLY if the snap has fired.
    Crucial when ``dwell_steps == 1``: without peek_skip, lookahead would
    call ``gate()`` per cell, increment each counter to 1, and consume
    every cell in one step regardless of whether the snap actually fired.
    """
    counter = 0
    def gate():
        nonlocal counter
        if snap_when():
            return True
        counter += 1
        return counter >= dwell_steps
    gate.peek_skip = lambda: bool(snap_when())
    return gate


def build_pick_place_phases(
    pick_pos=None,
    pick_orn=None,
    place_pos=None,
    place_orn=None,
    init_height: float = 0.10,
    include_close: bool = True,
    include_open: bool = True,
    settle_close_steps: int = 15,
    settle_open_steps: int = 15,
    settle_hover_place_steps: int = 0,
    settle_descend_place_steps: int = 0,
    gripper_open_value=None,
    gripper_close_value=None,
    transit_steps: int = 0,
    descend_pick_steps: int = 0,
    descend_place_steps: int = 0,
    release_mode: str = "open",
    snap_advance_when=None,
    snap_timeout_steps=None,
    snap_search_n: int = 0,
    snap_search_extent_xy=(0.002, 0.002),
    snap_search_dwell_steps: int = 1,
    return_home_q=None,
    return_home_gripper=None,
    return_home_settle_steps: int = 20,
    return_home_cspace_tol=None,
    final_height: float = None,
) -> List[Waypoint]:
    """Generate a (subset of the) canonical 8-phase pick-and-place path.

    Inputs (all phase-related ones may be None to skip the corresponding
    block — see PICK_PLACE_PHASES_CHEATSHEET.md):

      pick_pos, pick_orn      : EE world pose at the grasp.
                                If either is None, phases 1-4 are skipped.
      place_pos, place_orn    : EE world pose at the release.
                                If either is None, phases 5-8 are skipped.
      init_height             : delta-z added above pick/place to get the
                                hover pose for approach / lift / traversal.
      include_close           : if False, skip phase 3 (gripper close cmd).
      include_open            : if False, skip phase 7 (gripper open cmd).
      settle_*_steps          : number of EEPathFollower.step() calls each
                                gripper phase must dwell at its waypoint
                                before advancing (~10 Hz forward => 15 ~=
                                1.5 sec). ``settle_hover_place_steps``
                                holds the arm at the post-transit hover
                                so it can settle from the lerp before
                                descending — set > 0 if the transit
                                arrives at hover_place with momentum and
                                the descend kicks in before the arm is
                                stable.
      gripper_open_value      : float (rad) used in the gripper field of the
                                "open" waypoints (phase 1 hover_pick and
                                phase 7 release). If None, falls back to
                                the string "open" so the gripper's baked-in
                                joint_opened_position is used. Pass a part-
                                specific float for per-part finger spread.
      gripper_close_value     : same idea for the "close" waypoint (phase 3).
      transit_steps           : number of joint-space-interpolated transit
                                waypoints between the pick block's
                                `lift_pick` and the place block's
                                `hover_place`, when both blocks are present.
                                Each is marked `joint_lerp_t=k/(N+1)`; the
                                follower snapshots q_A from the actual robot
                                at segment entry, runs IK once at
                                `hover_place` for q_B, and emits raw
                                `q(t) = (1 - t) q_A + t q_B` for the
                                midpoints (no IK on the way through). 0 =
                                no transit (one big cartesian jump where
                                the IK is free to pick a different solution
                                branch and swing the arm out).
      descend_pick_steps      : number of joint-space-interpolated waypoints
                                inserted between `hover_pick` and
                                `descend_pick`. Same mechanism as
                                `transit_steps` — paces the descent across N
                                ticks instead of snapping the IK target
                                straight to the bottom. 0 disables (one-step
                                descend, existing behavior).
      descend_place_steps     : same idea for the descent from `hover_place`
                                to `descend_place`. Useful for precise
                                placement where the one-step descent is too
                                fast for the PD controller to track
                                smoothly. 0 disables.
      release_mode            : "open"  → only the gripper-open phase
                                releases the part (default).
                                "snap"  → a snap_wait waypoint is inserted
                                between descend_place and open. The arm
                                holds at descend_place (lock_pose=True,
                                gripper still closed) and the follower
                                gates the advance on ``snap_advance_when``
                                — typically ``lambda: attacher.attached``,
                                where ``attacher`` is a SnapAttacher driven
                                each physics step by the runner. The
                                ``open`` phase still emits after the snap
                                fires; with the part already pinned by a
                                FixedJoint, opening the fingers is safe.
      snap_advance_when       : callable returning True when the snap has
                                fired. Required when release_mode == "snap".
                                Evaluated once per follower step at
                                snap_wait.
      snap_timeout_steps      : if the snap never fires, the follower logs
                                a warning and advances past snap_wait after
                                this many step() calls. None = wait forever.
                                Only consulted when snap_search_n == 0.
      snap_search_n           : if > 0, the single snap_wait waypoint is
                                replaced by an N x N XY grid sweep around
                                place_pos. Cells are ordered from the
                                center outward and each advances on
                                (snap fired) OR (snap_search_dwell_steps
                                elapsed) so early termination kicks in as
                                soon as the snap triggers. 0 = legacy
                                single snap_wait behavior.
      snap_search_extent_xy   : (ex, ey) in meters. Each grid axis spans
                                [-ex, +ex]. Set to match the snap's
                                pos_tol_axes[0:2] so the sweep covers
                                exactly the snap tolerance box.
      snap_search_dwell_steps : per-cell fallback timeout. After this many
                                follower steps without snap firing, the
                                cell advances without warning. Default 1 =
                                one step per cell (fast scan). Raise if
                                the part needs more time to settle before
                                checking the tolerance box.
      return_home_q           : 1-D c-space joint vector. When set, a
                                ``return_home`` waypoint is prepended to
                                the path: the follower drives the joints
                                directly to this q via forward_raw_q
                                (bypassing IK) before starting hover_pick.
                                Used between parts to make every part's
                                first IK call start from the same c-space
                                seed it would get when the part is run
                                alone. Pass None for the first part of a
                                sequence (no return needed — robot is
                                already at init).
      return_home_gripper     : gripper command applied during return_home.
                                Typically the next part's gripper_open
                                value, so the gripper is at the correct
                                opening by the time hover_pick is reached.
      return_home_settle_steps: dwell at return_home after the joints
                                reach the target before advancing to
                                hover_pick. Default 20.
      return_home_cspace_tol  : per-waypoint override of
                                ``EEPathFollower.cspace_tolerance`` for the
                                return_home gate. The follower's default
                                tolerance has to stay loose for joint-lerp
                                transit waypoints; the return_home gate has
                                to be tight (its job is to deliver an IK
                                seed that matches a fresh-sim seed to many
                                decimals — a few degrees of seed drift
                                lets Lula IK pick a different solution
                                branch at hover_pick). Pass a tight value
                                (e.g. 5e-4 rad ≈ 0.03°) here. None defers
                                to the follower's default.

    Returns a list of Waypoint(pos, orn, gripper, settle_steps, name).
    """
    open_cmd  = float(gripper_open_value)  if gripper_open_value  is not None else "open"
    close_cmd = float(gripper_close_value) if gripper_close_value is not None else "close"

    dz = np.array([0.0, 0.0, float(init_height)], dtype=np.float64)
    # Separate delta-z for the post-open lift_place waypoint. If
    # final_height is None, use init_height (existing behavior — the
    # gripper retracts to the same hover height it came from). Set
    # explicitly when the post-release lift should be lower (e.g. fast
    # exit just above the part) or higher (e.g. clear a tall obstacle).
    dz_final = (np.array([0.0, 0.0, float(final_height)], dtype=np.float64)
                if final_height is not None else dz)
    out: List[Waypoint] = []

    # Optional return-to-home prepended before any pick/place motion. The
    # cspace_target waypoint drives joints back to a known c-space seed
    # so the part's first IK call resolves to the same branch it would
    # when running this part alone. ``pos`` is a placeholder (ignored by
    # the cspace-target branch in the follower).
    if return_home_q is not None:
        out.append(Waypoint(
            np.zeros(3, dtype=np.float64),  # dummy pos, unused
            None,                            # orn unused
            return_home_gripper,
            int(return_home_settle_steps),
            "return_home",
            False,
            None,
            None,
            None,
            np.asarray(return_home_q, dtype=np.float64).reshape(-1),
            (None if return_home_cspace_tol is None
             else float(return_home_cspace_tol)),
        ))

    pick_ok = pick_pos is not None and pick_orn is not None
    place_ok = place_pos is not None and place_orn is not None

    hover_pick = None
    pick_orn_arr = None
    if pick_ok:
        p_pick = np.asarray(pick_pos, dtype=np.float64).reshape(-1)
        pick_orn_arr = np.asarray(pick_orn, dtype=np.float64).reshape(-1)
        hover_pick = p_pick + dz
        out.append(Waypoint(hover_pick, pick_orn_arr, open_cmd, 0,            "hover_pick"))
        # Optional joint-space-lerped pacing for the descend. With
        # descend_pick_steps=N, the follower lerps the commanded q from
        # q_A (at hover_pick) to q_B (IK at descend_pick) across N
        # midpoints — same mechanism as transit, applied vertically.
        #
        # The lerp parameter uses an ease-out profile: t = sin(π/2 · k/(N+1)).
        # This remaps uniform k/(N+1) so that waypoints are bunched near
        # the bottom — joint motion is fast at the top (departure from
        # hover) and slows smoothly to zero at the bottom (gentle landing
        # at descend_pick). Linear lerp would give constant velocity with
        # a velocity discontinuity at both ends.
        if descend_pick_steps > 0:
            for k in range(1, descend_pick_steps + 1):
                t = float(np.sin(0.5 * np.pi * k / (descend_pick_steps + 1)))
                mid_pos = (1.0 - t) * hover_pick + t * p_pick
                out.append(Waypoint(
                    mid_pos, pick_orn_arr, None, 0,
                    f"descend_pick_{k}/{descend_pick_steps}",
                    False, t,
                ))
        out.append(Waypoint(p_pick,     pick_orn_arr, None,     0,            "descend_pick"))
        # The close waypoint locks to the actual EE pose at entry: the
        # descend phase advances as soon as pos_err <= POS_TOL, so the EE is
        # usually a few mm above the nominal target. Without locking, IK
        # keeps pulling the arm down while the gripper closes, dragging the
        # EE lower during the grasp.
        if include_close:
            out.append(Waypoint(p_pick, pick_orn_arr, close_cmd, settle_close_steps, "close", True))
        out.append(Waypoint(hover_pick, pick_orn_arr, None,     0,            "lift_pick"))

    place_orn_arr = None
    hover_place = None
    if place_ok:
        p_place = np.asarray(place_pos, dtype=np.float64).reshape(-1)
        place_orn_arr = np.asarray(place_orn, dtype=np.float64).reshape(-1)
        hover_place = p_place + dz

    # Joint-space-interpolated transit between lift_pick and hover_place.
    # Cartesian interpolation here let the IK pick a different solution
    # branch as the target moved across the workspace, which swung the arm
    # behind the back. The follower runs IK once at hover_place to obtain
    # q_B, snapshots q_A from the actual robot at segment entry, and emits
    # ``q(t) = (1 - t) q_A + t q_B`` directly for each transit waypoint —
    # no IK at the midpoints. ``pos`` here is still the cartesian lerp
    # midpoint, retained only for diagnostics (the actual EE traces a
    # non-straight curve through that region).
    if (pick_ok and place_ok and transit_steps > 0
            and hover_pick is not None and hover_place is not None):
        for k in range(1, transit_steps + 1):
            t = k / (transit_steps + 1)
            mid_pos = (1.0 - t) * hover_pick + t * hover_place
            out.append(Waypoint(
                mid_pos, pick_orn_arr, None, 0, f"transit_{k}/{transit_steps}",
                False, t,
            ))

    if place_ok:
        out.append(Waypoint(hover_place, place_orn_arr, None, settle_hover_place_steps, "hover_place"))
        # Optional joint-space-lerped pacing for the place descend. Same
        # mechanism as descend_pick_steps; the segment anchor for q_B is
        # the descend_place waypoint, so IK is solved once at the bottom
        # and the commanded q lerps from current down to that solution.
        # Ease-out profile (see descend_pick block above): velocity is
        # max at the top and decays to zero at the bottom for a gentle
        # landing — preferred for precise placement.
        if descend_place_steps > 0:
            for k in range(1, descend_place_steps + 1):
                t = float(np.sin(0.5 * np.pi * k / (descend_place_steps + 1)))
                mid_pos = (1.0 - t) * hover_place + t * p_place
                out.append(Waypoint(
                    mid_pos, place_orn_arr, None, 0,
                    f"descend_place_{k}/{descend_place_steps}",
                    False, t,
                ))
        # In snap mode, let descend_place ALSO advance the moment snap
        # fires — without this, the FixedJoint authored by the snap can
        # drag the gripper a few mm off the descend_place target, kicking
        # the pos/orn gate out of tolerance forever and forcing the
        # follower to wait for its 500-step "advance anyway" timeout.
        # In open mode, no advance_when (just settle + pos/orn gate).
        descend_place_gate = (snap_advance_when
                              if release_mode == "snap" else None)
        out.append(Waypoint(
            p_place, place_orn_arr, None, settle_descend_place_steps,
            "descend_place",
            False,                    # lock_pose
            None,                     # joint_lerp_t
            descend_place_gate,       # advance_when
        ))
        # release_mode == "snap": hold at the descend pose with the gripper
        # still closed while a SnapAttacher (driven by the runner each
        # physics step) waits for the part's world pose to enter its snap
        # tolerance and authors a FixedJoint. ``snap_advance_when`` returns
        # True once the joint is authored; the follower advances to
        # ``open``, which now releases onto an already-pinned part.
        if release_mode == "snap":
            if snap_advance_when is None:
                raise ValueError(
                    "release_mode='snap' requires snap_advance_when (a "
                    "callable returning True when the snap has fired)."
                )
            if snap_search_n and snap_search_n > 0:
                # XY grid sweep: visit n*n cells offset from place_pos by
                # +-extent in each axis, ordered by distance-from-center
                # (spiral-ish) so the most likely cells hit first. Each
                # cell advances on (snap fired) OR (dwell_steps elapsed);
                # the path falls through naturally if the snap never fires.
                n = int(snap_search_n)
                ex = float(snap_search_extent_xy[0])
                ey = float(snap_search_extent_xy[1])
                step_x = (2.0 * ex / (n - 1)) if n > 1 else 0.0
                step_y = (2.0 * ey / (n - 1)) if n > 1 else 0.0
                center = (n - 1) / 2.0
                cells = sorted(
                    ((i, j) for i in range(n) for j in range(n)),
                    key=lambda c: (c[0] - center) ** 2 + (c[1] - center) ** 2,
                )
                total = len(cells)
                for k, (i, j) in enumerate(cells, start=1):
                    dx = (i - center) * step_x
                    dy = (j - center) * step_y
                    cell_pos = p_place + np.array(
                        [dx, dy, 0.0], dtype=np.float64
                    )
                    out.append(Waypoint(
                        cell_pos, place_orn_arr, None, 0,
                        f"snap_search_{k}/{total}",
                        False,  # lock_pose: actually move to the cell
                        None,
                        _make_search_gate(snap_advance_when,
                                          snap_search_dwell_steps),
                        None,   # no per-cell timeout warning
                    ))
            else:
                out.append(Waypoint(
                    p_place, place_orn_arr, None, 0, "snap_wait",
                    True,           # lock_pose: hold actual EE pose at entry
                    None,           # joint_lerp_t
                    snap_advance_when,
                    snap_timeout_steps,
                ))
        # The open waypoint locks pose at entry for the same reason as
        # close: keep the arm height fixed while the gripper releases.
        if include_open:
            out.append(Waypoint(p_place, place_orn_arr, open_cmd, settle_open_steps, "open", True))
        lift_place_pos = p_place + dz_final
        out.append(Waypoint(lift_place_pos, place_orn_arr, None,    0,           "lift_place"))

    # Number names sequentially based on what actually made it into the
    # path, so skipped phases never leave gaps (e.g. "1.hover_pick" through
    # "N.lift_place" with no 7.open if INCLUDE_OPEN=False).
    out = [wp._replace(name=f"{i+1}.{wp.name}") for i, wp in enumerate(out)]
    return out
