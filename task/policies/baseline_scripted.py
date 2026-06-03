"""Reference scripted policy for the IROS 2026 vega_1u assembly challenge.

Wraps the original `EEPathFollower`-driven pick-and-place: per part, build
a 9-phase EE-pose path from `PART_CONFIG`, drive Lula IK to follow it,
gate the snap_wait waypoint on `obs.snap_fired`.

Output should be byte-identical to the pre-refactor `run_pick_place.py` when
the harness selects this policy. Participants should not modify this file;
copy `template.py` instead.
"""
from __future__ import annotations

import os.path
import sys

# Ensure `task/` is on sys.path so absolute imports of `param_config` and
# `controllers.*` work whether the policy is loaded as a top-level module
# or as `policies.baseline_scripted`.
_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

import param_config as pc  # noqa: E402
from controllers.ee_pose_controller import (  # noqa: E402
    EEPathFollower,
    build_pick_place_phases,
)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402


def make_l_path_for_part(part_name, snap_advance_when=None,
                         snap_timeout_steps=None, return_home_q=None):
    """Build the 9-phase L path for one part using its PART_CONFIG entry.

    Pick/place world OBJECT positions are offset by `cfg["ee_offset"]` to
    get the EE-frame target positions. Orientations and per-part gripper
    open/close joint values are pulled from cfg too. Returns the (possibly
    `MAX_PHASES`-truncated) waypoint list, or `[]` if both pick_pos and
    place_pos are None.

    When `cfg["release_mode"] == "snap"`, `snap_advance_when` must be a
    callable returning True once the snap fires; the phase builder inserts
    a snap_wait waypoint between descend_place and open and gates the
    advance on it. `snap_timeout_steps` is the timeout fall-through.

    `return_home_q` (when set) prepends a `return_home` waypoint that
    drives the L arm joints directly to that c-space vector before
    starting hover_pick. The gripper is commanded to this part's
    `gripper_open` value during the return so it's at the right opening
    by the time the new pick begins.
    """
    cfg = pc.get_part_config(part_name)
    ee_off = np.asarray(cfg["ee_offset"], dtype=np.float64)
    orn = np.asarray(cfg["ee_orientation"], dtype=np.float64)
    pick_pos_ee = (
        None if cfg.get("pick_pos") is None
        else np.asarray(cfg["pick_pos"], dtype=np.float64) + ee_off
    )
    place_pos_ee = (
        None if cfg.get("place_pos") is None
        else np.asarray(cfg["place_pos"], dtype=np.float64) + ee_off
    )
    init_height = (cfg.get("init_height")
                   if cfg.get("init_height") is not None
                   else pc.INIT_HEIGHT)
    transit_steps = (cfg.get("transit_steps")
                     if cfg.get("transit_steps") is not None
                     else pc.TRANSIT_STEPS)
    final_height = (cfg.get("final_height")
                    if cfg.get("final_height") is not None
                    else getattr(pc, "FINAL_HEIGHT", None))
    full = build_pick_place_phases(
        pick_pos=pick_pos_ee,
        pick_orn=orn if pick_pos_ee is not None else None,
        place_pos=place_pos_ee,
        place_orn=orn if place_pos_ee is not None else None,
        init_height=init_height,
        final_height=final_height,
        include_close=pc.INCLUDE_CLOSE,
        include_open=pc.INCLUDE_OPEN,
        settle_close_steps=pc.SETTLE_CLOSE,
        settle_open_steps=pc.SETTLE_OPEN,
        settle_hover_place_steps=pc.SETTLE_HOVER_PLACE,
        settle_descend_place_steps=pc.SETTLE_DESCEND_PLACE,
        transit_steps=int(transit_steps),
        descend_pick_steps=pc.DESCEND_PICK_STEPS,
        descend_place_steps=pc.DESCEND_PLACE_STEPS,
        gripper_open_value=cfg.get("gripper_open"),
        gripper_close_value=cfg.get("gripper_close"),
        release_mode=cfg.get("release_mode", "open"),
        snap_advance_when=snap_advance_when,
        snap_timeout_steps=snap_timeout_steps,
        snap_search_n=int(((cfg.get("snap") or {}).get("search") or {})
                          .get("n", 0)),
        snap_search_extent_xy=tuple(((cfg.get("snap") or {}).get("search") or {})
                                    .get("extent_xy", (0.002, 0.002))),
        snap_search_dwell_steps=int(((cfg.get("snap") or {}).get("search") or {})
                                    .get("dwell_steps", 1)),
        return_home_q=return_home_q,
        return_home_gripper=cfg.get("gripper_open"),
        return_home_cspace_tol=getattr(pc, "RETURN_HOME_CSPACE_TOL", None),
        return_home_settle_steps=getattr(pc, "RETURN_HOME_SETTLE_STEPS", 20),
    )
    if pc.MAX_PHASES is not None and pc.MAX_PHASES < len(full):
        return full[:int(pc.MAX_PHASES)]
    return full


class BaselinePolicy(Policy):
    """Scripted EE-path follower over each part's 9-phase pick-place plan.

    Requires `env_info.L_controller` (the harness sets it automatically).
    Other policies do not need this controller — they can produce joint
    targets directly.
    """

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        L_controller = getattr(env_info, "L_controller", None)
        if L_controller is None:
            raise ValueError(
                "BaselinePolicy requires env_info.L_controller (an "
                "EEPoseController). The harness sets this by default; "
                "if you see this error, you may be running the policy "
                "outside the provided harness."
            )
        self._L_controller = L_controller
        self._follower = EEPathFollower(
            L_controller,
            position_tolerance=getattr(pc, "POS_TOL", 0.005),
            orientation_tolerance=getattr(pc, "ORN_TOL", 0.05),
            default_timeout_steps=getattr(pc, "WAYPOINT_TIMEOUT_STEPS", None),
        )
        self._is_first_part = True
        # _last_obs is read by the snap_advance_when closure, which is
        # invoked by EEPathFollower.step() to decide whether to advance
        # past the snap_wait waypoint. Updated every act() call.
        self._last_obs: Observation = None  # type: ignore[assignment]

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._last_obs = obs

        snap_advance_when = None
        snap_timeout_steps = None
        if target.release_mode == "snap":
            snap_advance_when = self._read_snap_fired
            snap_cfg = target.extra.get("snap") if target.extra else None
            if snap_cfg is not None:
                snap_timeout_steps = snap_cfg.get("timeout_steps")

        return_home_q = (None if self._is_first_part
                         else self.env_info.L_arm_init_q)
        self._is_first_part = False

        path = make_l_path_for_part(
            target.name,
            snap_advance_when=snap_advance_when,
            snap_timeout_steps=snap_timeout_steps,
            return_home_q=return_home_q,
        )
        self._follower.reset()
        self._follower.set_path(path)

    def act(self, obs: Observation):
        self._last_obs = obs
        return self._follower.step()

    def is_done(self, obs: Observation) -> bool:
        self._last_obs = obs
        return self._follower.is_done()

    def _read_snap_fired(self) -> bool:
        return bool(self._last_obs is not None
                    and self._last_obs.snap_fired)

    # Diagnostic accessors used by the harness's stuck detector.
    @property
    def current_waypoint(self):
        return self._follower.current_waypoint()

    @property
    def current_index(self) -> int:
        return self._follower.current_index()
