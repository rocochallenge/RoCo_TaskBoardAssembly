import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Per-part data lives in part_init_poses.json next to this file. ONE source
# of truth — edit values there, not inline in PART_CONFIG below.
#
# Schema (per part):
#   pos      : [x, y, z]  spawn pose, mesh world; auto-extracted by
#                         extract_part_poses.py.
#   orn      : [w, x, y, z] spawn orientation, wxyz quat (mesh world).
#   pick_z   : float (optional) — gripper-target z. pick x/y inherit from
#              pos. If absent, falls back to pos[2].
#   path / mesh_path / rb_* : informational; not consumed by the runner.
# ---------------------------------------------------------------------------
_PART_INIT_POSES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "part_init_poses.json"
)


def _load_part_init_poses() -> dict:
    """Return {name: {'pick_pos': ndarray, 'pos': ndarray, 'orn': ndarray}}.

    Builds pick_pos by composing the spawn pose's x/y with pick_z (z). If
    pick_z is absent for a part, pick_pos falls back to pos verbatim.
    Missing file returns {}. Keys starting with '_' are ignored.
    """
    if not os.path.isfile(_PART_INIT_POSES_JSON):
        return {}
    with open(_PART_INIT_POSES_JSON) as f:
        data = json.load(f)
    out = {}
    for name, entry in data.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        if "pos" not in entry:
            continue
        pos = np.asarray(entry["pos"], dtype=np.float64)
        rec = {"pos": pos}
        if "orn" in entry:
            rec["orn"] = np.asarray(entry["orn"], dtype=np.float64)
        pick_z = entry.get("pick_z")
        pick_pos = pos.copy()
        if pick_z is not None:
            pick_pos[2] = float(pick_z)
        rec["pick_pos"] = pick_pos
        out[name] = rec
    return out


PART_INIT_POSES = _load_part_init_poses()


# ===========================================================================
# SCENE / ROBOT SETUP  (consumed by setup_pick_place_sim in vega_1u_setup.py)
# ===========================================================================
# Pre-built scene loaded as the stage by open_scene_and_world(). Path is
# resolved relative to this file's directory (task/), so "../foo.usd" points
# at assets/foo.usd.
#
# scene_init.usd already contains all parts at their hand-placed positions,
# so import_missing_parts() finds them present and spawns nothing — the
# runtime parts (pin/hdmi/usb*/ethernet/batteries) keep the exact poses
# authored in the file instead of being re-spawned from part_init_poses.json
# (which would re-introduce the prim-origin vs mesh offset). pick_pos for
# those parts is pinned in PART_CONFIG to the mesh world xy read from this
# same file.
SCENE_USD = "../scene_init.usd"
L_object_prim_path = "/World/task_board/task_board_color"   # static
R_object_prim_path = "/World/task_board/task_board_color"

# Optional path to dump a flattened USD snapshot of the stage when the
# part iteration finishes (i.e. after all parts in part_order have been
# placed). Used to test post-placement physics — re-open the file and
# press play, no gripper interaction, and watch whether parts stay put
# or fall. None disables the dump. Path can be absolute or relative to
# task/.
SAVE_FINAL_STAGE_PATH = "scene_final.usd"

# Optional path to dump a JSON with per-part pass/fail and the aggregate
# score after the iteration finishes. None disables; --results-json on the
# CLI overrides. Path can be absolute or relative to task/.
RESULTS_JSON_PATH = None

# Failsafe: maximum number of physics steps the harness will wait for a
# policy to finish a single part before forcibly advancing to the next.
# Prevents a stuck / buggy policy from hanging the eval. The baseline
# scripted policy typically completes a part in 600-1500 steps.
PER_PART_TIMEOUT_STEPS = 3000

# Physics warmup: number of my_world.step() iterations to run before the
# pick-and-place task starts (after each reset / first play). Lets PhysX
# finish cooking colliders — especially SDF meshes, which are slow to
# bake on the very first parse — and lets joint drives settle to their
# init targets before the gripper tries to grasp anything. Without this,
# the first run after a fresh launch often fails to grip (SDF cooking
# hasn't completed), but a stop-and-replay works because PhysX caches
# the cooked SDF across resets within a session. 60 steps ≈ 0.3 s at
# 200 Hz. Set to 0 to disable.
WARMUP_STEPS = 60

# When True, the runner prints [STUCK] diagnostic blocks for every
# follower stall longer than STUCK_LOG_STEPS in run_pick_place.py.
# Default False — normal long-settle phases (.close, snap_search,
# descend_place) routinely cross the threshold and would spam the
# log. Flip to True to debug a real hang (no advance, snap never
# fires, IK failing, gripper slipping).
VERBOSE_STUCK = False

# Camera config. The three cameras live in the robot USD now:
#   headcam     at /World/robotics/vega_1u_gripper/zed_depth_frame/headcam
#   L_wristcam  at /World/robotics/vega_1u_gripper/L_ee_link/gripper_link/L_wristcam
#   R_wristcam  at /World/robotics/vega_1u_gripper/R_ee_link/gripper_link/R_wristcam
# Their poses are authored in the USD; nothing here overrides them.
enable_camera_viewports = True   # show the 3-tile viewport layout in Kit UI
enable_camera_output    = False   # bind sensors so RGB/depth are readable from Python
HEAD_DEPTH_CAMERA_FOCAL_LENGTH = float(os.getenv("TASK_HEAD_DEPTH_FOCAL_LENGTH", "10"))

# IK c-space size per arm. Three modes per side, picked by the
# (OWNS_LIFT, OWNS_TORSO) pair:
#   (False, False) → 7-DOF arm-only (`*_armonly.yaml`).
#                    Lift + torso_flip held by the PD controller.
#   (True,  False) → 8-DOF lift+arm  (`*_liftonly.yaml`).
#                    Lift in cspace, torso_flip fixed.
#   (True,  True ) → 9-DOF full      (`*_arm_description.yaml`).
#                    Lift + torso_flip both in cspace.
# (OWNS_TORSO=True with OWNS_LIFT=False is rejected — torso_flip sits above
# Lift in the URDF chain, so torso ownership requires lift ownership.)
#
# Watch for torso_flip swinging around on cross-body transits in 9-DOF mode;
# mitigation is a positive bias on default_q[torso_flip] in the description
# yaml.
OWNS_LIFT_L  = False
OWNS_TORSO_L = False
OWNS_LIFT_R  = False
OWNS_TORSO_R = False

# When OWNS_TORSO_<side>=False, Lift is held at this value (meters) by the
# PD controller. URDF Lift joint limit is [0.0, 0.4], so 0.4 = full extension
# upward (torso at its highest = max workspace reach). MUST match the
# `Lift` fixed-value entry in both *_armonly description yamls so Lula's
# IK kinematic chain stays consistent with the runtime joint state.
LIFT_INIT_M = 0.0

# R-arm resting pose. True = tucked ([-90, 0, 0, 0, 0, 0, 0] deg, R_arm_j1
# folded out of the way); False = forward USDA pose from vega_1u_gripper.usda
# ([-15, -20, 0, 0, 0, 0, 0] deg). Keep True when L handles every pick on its
# own — gives L more workspace clearance. Flip to False when running the L→R
# handoff flow (R needs to be reach-ready for picks L can't make). The L
# descriptor yamls' R_arm_j* fixed values currently assume TUCKED — flipping
# to False without updating those will leave Lula's collision spheres for R
# slightly off (acceptable while R isn't actively doing IK).
R_ARM_TUCKED = True

# Joint targets applied at sim start AND on every restart (stop+play).
# Overrides whatever USDA-driven joint values World.reset() restores to —
# guarantees the IK warm_start is identical across runs and prevents Lula
# from latching onto a bad solution branch on the first waypoint after a
# stop-and-replay. Values in RADIANS for revolute joints, METERS for
# prismatic. Joints not listed are left untouched.
#
# Defaults below mirror vega_1u_gripper.usda's L arm init (-15, 10, 0, 5,
# 0, 30, -30 deg) plus Lift = LIFT_INIT_M.  Edit values here without
# touching the USDA.
INIT_JOINT_TARGETS = {
    "Lift":      LIFT_INIT_M,
    # L_arm values mirror the drive targets authored in scene_base.usd
    # (so the post-reset PD target matches what the scene drives toward,
    # eliminating the immediate snap-back to the scene pose that would
    # otherwise pull the arm out of the gripper USDA's neutral pose).
    "L_arm_j1": -0.52359878,   #  -30 deg
    "L_arm_j2":  1.04719755,   #  +60 deg
    "L_arm_j3":  1.74532925,   # +100 deg
    "L_arm_j4": -1.74532925,   # -100 deg
    "L_arm_j5": -0.17453293,   #  -10 deg
    "L_arm_j6": -0.17453293,   #  -10 deg
    "L_arm_j7": -1.04719755,   #  -60 deg
    "head_j1":   0.95993109,   #  +55 deg, selected head camera pitch
    "head_j2":   0.0,
    "head_j3":   0.0,
}

# R-arm values derived from R_ARM_TUCKED so flipping the flag above
# automatically reconfigures the post-reset R pose — no edits needed here.
if R_ARM_TUCKED:
    # j1 = -90 deg, rest = 0  (folded out of the way).
    INIT_JOINT_TARGETS.update({
        "R_arm_j1": -1.57079633,
        "R_arm_j2":  0.0,
        "R_arm_j3":  0.0,
        "R_arm_j4":  0.0,
        "R_arm_j5":  0.0,
        "R_arm_j6":  0.0,
        "R_arm_j7":  0.0,
    })
else:
    # USDA forward pose: j1=-15, j2=-20, rest=0.
    INIT_JOINT_TARGETS.update({
        "R_arm_j1": -0.26179939,
        "R_arm_j2": -0.34906585,
        "R_arm_j3":  0.0,
        "R_arm_j4":  0.0,
        "R_arm_j5":  0.0,
        "R_arm_j6":  0.0,
        "R_arm_j7":  0.0,
    })


# ===========================================================================
# PHASE CONFIG  ── edit this block to choose what L IK targets to run.
# Set None / False to skip phases (see PICK_PLACE_PHASES_CHEATSHEET.md).
# Use MAX_PHASES to truncate to the first N waypoints (None = all).
# ===========================================================================

# ---------------------------------------------------------------------------
# Per-part config. Values that vary by object live here, indexed by the
# part name used in part_order. PART_DEFAULTS provides fallbacks; missing
# keys in a per-part dict fall through.
#
# ee_orientation  (wxyz quat): top-down approach. Quat (0, 1, 0, 0) is
#                 the stage-reported quat for a top-down grasp on this rig
#                 (LulaIKController.forward() applies the URDF<->USD
#                 180-deg offset correction for you).
# ee_offset       (xyz, m): gripper-tip-to-EE-frame offset.
#                 y = lateral offset of fingers vs EE frame (gripper geom).
#                 z = 0.15 base + 0.066 measured; tweak by eye if fingers
#                 don't enclose the object in phase 3.
# gripper_open    (rad): finger spread for "open" command at this part.
#                 Smaller = tighter fit. ~0.15 sized for the rod_16mm.
# gripper_close   (rad): finger spread for "close" command at this part.
# ---------------------------------------------------------------------------
PART_DEFAULTS = {
    "ee_orientation": np.array([0.0, 1.0, 0.0, 0.0]),
    "ee_offset":      np.array([0.0, 0.016, 0.196]),
    "gripper_open":   0.15,
    "gripper_close":  0.0,
    "pick_pos":       None,   # required per-part; None = skip phases 1-5
    "place_pos":      None,   # required per-part; None = skip phases 6-9
    # Release mode at the place block.
    #   "open"  — phase 7 just opens the gripper (default).
    #   "snap"  — insert a snap_wait waypoint between descend_place and
    #             open; the runner drives a SnapAttacher each step and
    #             gates the wait on attacher.attached. Requires a "snap"
    #             dict on the part (see pin / hdmi / usb_a below).
    "release_mode":   "open",
    "snap":           None,
    # Per-part hover delta-z above the grasp / place EE pose (m). None =
    # fall back to the global INIT_HEIGHT below.
    "init_height":    None,
    # Per-part delta-z above the place pose for the post-open lift_place
    # waypoint (the retract after the gripper releases). None = fall back
    # to the global FINAL_HEIGHT (and if that's None too, init_height).
    # Set to a value like 0.05 if you want a low retract that hugs the
    # part, or 0.15 to clear a tall obstacle on the way out.
    "final_height":   None,
    # Per-part joint-lerp transit waypoint count between lift_pick and
    # hover_place. None = fall back to the global TRANSIT_STEPS below. Set
    # explicitly (e.g. 0) when a specific part needs a different count than
    # the global default — e.g. usb_a is sensitive to transit interpolation
    # and wants 0 even when other parts use several.
    "transit_steps":  None,
    # End-of-iteration grading target (world XYZ, m). The final settled
    # pose the part should occupy after release + gravity + contact (NOT
    # the gripper's release pose, which is place_pos). Only used by
    # _grade_task for position-graded parts (release_mode == "open").
    # None = fall back to place_pos (i.e. assume the part lands right
    # where the gripper let go). Set explicitly for parts that sink /
    # roll / settle into a socket after release.
    "grade_pos":      None,
    # Spawn orientation (wxyz quat) used when the runner injects this part
    # into the stage from ../parts/<name>.usdc because the prim is absent
    # from the loaded scene. None = fall back to part_poses.json's
    # init.<name>.orn (the orientation the part had in the original scene
    # extract). Set explicitly per-part to override.
    "spawn_orn":      None,
    # Per-part overrides applied only when this part is run as part of a
    # multi-part sequence (len(part_order) > 1). Any keys here replace the
    # base values returned by get_part_config — typically you'll re-tune
    # pick_pos / ee_offset / gripper_* to compensate for the slight drift
    # vs. a fresh-sim standalone run. None or {} = no overrides; the
    # standalone values are used in sequence too. Nested dicts (e.g. snap)
    # are replaced wholesale, not deep-merged.
    "sequence":       None,
}

PART_CONFIG = {
    # Each entry only sets keys that differ from PART_DEFAULTS. pick_pos /
    # place_pos are world OBJECT positions (bottom-mesh, scraped from
    # part_poses.json). Tune ee_offset / gripper_open / gripper_close per
    # part by adding the override key when you start testing that part.
    "rod_16mm": {
        "gripper_close":  0.04,
        "place_pos":      np.array([ 0.24681,  0.16982, 1.057]),
        "ee_offset":      np.array([0.0, 0.016, 0.21]),
        "release_mode":   "snap",
        "snap": {
            "movable_path":     "/World/parts/rod_16mm",
            "debug":            False,  # log snap gate pos/rot err each ~30 ticks
            "parent_body_path": "/World/task_board/task_board_color/root_001/_188_028",
            "target_pos":       (0.24681, 0.16982, 1.057),    # = place_pos (mesh-frame)
            "target_rot":       (0.7071, 0.7071, 0.0, 0.0),  # wxyz, from extract 'final' orn
            "pos_tol_axes":     (0.0025, 0.0025, 0.005),        # WORLD frame
            "rot_tol_deg":      -1,                        # skip rot gate (axis-symmetric)
            "set_kinematic":    False,
            "timeout_steps":    300,
            # Joint anchor = same as proximity target (= place_pos), per
            # user request. snap_attach converts mesh-frame target →
            # body-frame anchor internally via mesh_local_in_body.
            "connect_pos":      (0.24681, 0.16982, 1.045),
            "connect_rot":      (0.7071, 0.7071, 0.0, 0.0),  # wxyz, from extract 'final' orn
            # connect_rot omitted → defaults to target_rot above.
            # Fine XY grid sweep at the place pose. dwell_steps=1 →
            # 1 follower step per cell (fast scan).
            "search": {
                "n":          5,               # 5x5 = 25 cells
                "extent_xy":  (0.002, 0.002),  # match pos_tol_axes[0:2]
                "dwell_steps": 1,
            },
        },
    },
    "battery_size1": {
        # pick_pos xy = actual mesh world pos from scene_init.usd (the part
        # prim spawns at JSON pos, but the mesh sits offset from the prim
        # origin); z = pick_z. Overrides the JSON-derived pick_pos.
        "pick_pos":       np.array([ 0.03362,  0.15554, 1.047]),
        "gripper_open":  0.2,
        "gripper_close":  0.07,
        "place_pos":      np.array([ 0.135,  0.15639, 1.06423]),
        # grade_pos is the AABB MIDPOINT (world-axis-aligned bounds of
        # the mesh) extracted from scene_final.usd. Rotation about the
        # battery's long axis doesn't move this value (axis-symmetric).
        "grade_pos":      np.array([+0.12914, +0.15615, +1.04601]),
        "grade_use_aabb": True,
        "ee_offset":      np.array([0, 0.015,0.185]),
        #"transit_steps":  12,
        #"init_height":    0.01,
    },
    "battery_size5": {
        # pick_pos xy = mesh world pos from scene_init.usd; z = pick_z.
        "pick_pos":       np.array([-0.01071,  0.16490, 1.04]),
        "gripper_open":  0.11,
        "gripper_close":  0.05,
        "place_pos":      np.array([ 0.09029,  0.168, 1.06]),
        # AABB midpoint from scene_final.usd.
        "grade_pos":      np.array([+0.08701, +0.16742, +1.03578]),
        "grade_use_aabb": True,
        "ee_offset":      np.array([-0.001, 0.015,0.1875]),
    },
    "bolt_8mm": {
        "gripper_open":   0.06,
        "gripper_close":  0.04,
        "pick_pos":       np.array([ 0.04415, 0.00093, 1.039]),
        "place_pos":      np.array([ 0.21531,  0.13135, 1.06]),
        "ee_offset":      np.array([0.0, 0.0147, 0.2045]),
        "init_height":    0.05, 
        "final_height":   0.1,
        "release_mode":   "snap",
        "snap": {
            "movable_path":     "/World/parts/bolt_8mm",
            "debug":            False,  # log snap gate pos/rot err each ~30 ticks
            # Shared rack slot _188_028 with rod_16mm.
            "parent_body_path": "/World/task_board/task_board_color/root_001/_188_028",
            "target_pos":       (0.21531, 0.13135, 1.06),  # = place_pos (mesh-frame)
            "target_rot":       (0.6892, 0.6892, 0.0, 0.0),   # wxyz
            "pos_tol_axes":     (0.002, 0.002, 0.005),         # WORLD frame
            "rot_tol_deg":      -1,                      
            "set_kinematic":    False,
            "timeout_steps":    300,
            "connect_pos":      (0.21531, 0.13135, 1.04),  # = place_pos
            # Force the joint to anchor at exactly target_rot. Without this,
            # _joint_anchor_matrix defaults to the mesh's CURRENT rotation
            # at snap-fire time — and with rot_tol_deg=10 that can be up to
            # 10° off the socket axis, leaving the bolt visibly tilted.
            "connect_rot":      (0.6892, 0.6892, 0.0, 0.0),  # = target_rot
            "search": {
                "n":          6,
                "extent_xy":  (0.006, 0.006),
                "dwell_steps": 1,
            },
        },

    },

    "gear_20teeth": {
        "gripper_open":   0.12,
        "gripper_close":  0.065,
        # Lands in the gear_60teeth slot (gear_60teeth was deleted from
        # scene_base.usd; gear_20teeth substitutes for it).
        "pick_pos":       np.array([0.14366, -0.043, 1.04]),
        "place_pos":      np.array([ 0.1972314984643313, -0.09598882384960386, 1.05398]),
        # Final settled pose for grading (gear sinks onto the rack post
        # after release; values measured from a known-good run).
        "grade_pos":      np.array([ 0.1972314984643313,
                                    -0.09598882384960386,
                                     1.033770322353139]),
        "ee_offset":      np.array([0.0, 0.016, 0.197]),
        "init_height":    0.05,
        "final_height":   0.1,
        # SDF collider — gear teeth are too concave for convexDecomposition
        # to capture cleanly; SDF preserves the tooth profile for stable
        # meshing against the rack post.
        "collision_approximation": "sdf",
        # Teleport the gear to its target pose right before the gripper
        # opens, if it lands within these loose tolerances. No joint —
        # gear stays dynamic and settles onto the rack post naturally.

    },
    "gear_60teeth": {
        "gripper_open":   0.2,
        "gripper_close":  0.15,
        "ee_offset":      np.array([0.0, 0.016, 0.2]),
        "place_pos":      np.array([ 0.20822, -0.05629, 1.05293]),
        # Final settled pose for grading (gear sinks onto the rack post
        # after release; values measured from a known-good run).
        "grade_pos":      np.array([ 0.20853747092278196,
                                    -0.056146127605838425,
                                     1.0315798358785877]),
        "init_height":    0.08,
        "final_height":   0.15,
        "collision_approximation": "sdf",
    },
    "hdmi": {
        # pick_pos xy = mesh world pos from scene_init.usd; z = pick_z.
        "pick_pos":       np.array([ 0.27285, -0.01949, 1.03854]),
        "gripper_open":   0.1,
        "gripper_close":  0.04,
        "place_pos":      np.array([ 0.27469,  0.049, 1.055]),
        "release_mode":   "snap",
        "snap": {
            "movable_path":     "/World/parts/hdmi",
            "parent_body_path": "/World/task_board/task_board_color",
            "target_pos":       (0.27469, 0.049, 1.055),   # = place_pos (mesh-frame)
            "target_rot":       (0.5, 0.5, 0.5, 0.5),         # wxyz, from part_poses.json 'final'
            "pos_tol_axes":     (0.002, 0.002, 0.005),        # WORLD frame
            "rot_tol_deg":      10,
            "set_kinematic":    False,
            "timeout_steps":    300,
            "connect_pos":      (0.27469, 0.049, 1.039),
            # Force the joint to anchor at exactly target_rot. Without this,
            # _joint_anchor_matrix defaults to the mesh's CURRENT rotation
            # at snap-fire time — and with rot_tol_deg=10 that can be up to
            # 10° off the socket axis, leaving the connector visibly tilted.
            "connect_rot":      (0.5, 0.5, 0.5, 0.5),  # = target_rot
            "search": {
                "n":          5,               # 5x5 = 25 cells
                "extent_xy":  (0.002, 0.002),  # match pos_tol_axes[0:2]
                "dwell_steps": 1,
            },
        },
    },
    "pin": {
        # pick_pos xy = mesh world pos from scene_init.usd; z = pick_z.
        "pick_pos":       np.array([ 0.18600, -0.01476, 1.04437]),
        "gripper_close": 0.055,
        "place_pos":      np.array([ 0.24945,  0.00616, 1.065]),
        "ee_offset":      np.array([0.0, 0.017, 0.185]),
        # TODO(sync): the snap dict below is the dev-loop copy. Once these
        # values stabilize, mirror them into author_snap_targets.SNAP_CONFIGS
        # ["pin"] and re-run author_snap_targets.py so the USD-authored
        # `snap:*` attrs match what runtime uses.
        "release_mode":   "snap",
        "snap": {
            "movable_path":     "/World/parts/pin",
            "parent_body_path": "/World/task_board/task_board_color/_188_001",
            "target_pos":       (0.24945,  0.00616, 1.065),#(+0.066, +0.00616, +1.05000),
            "target_rot":       (+0.7071, +0.7071, +0.0000, +0.0000),  # wxyz
            "pos_tol_axes":     (0.002, 0.002, 0.01),  # WORLD frame
            "rot_tol_deg":      -1.0,                   # axis-symmetric, skip rot gate
            "set_kinematic":    False,
            "timeout_steps":    300,
            # Exact world pose the MESH should end up at after the snap fires.
            # snap_attach pulls the rigid body to whatever pose makes the mesh
            # land here (using the cached mesh-to-body local offset). MESH
            # frame, consistent with target_pos. Value = pin mesh world pose
            # in scene_final.usd (extract_part_poses.py 'pos' field).
            # connect_rot omitted → defaults to target_rot.
            "connect_pos":      (0.2494453614671286,
                                 0.006163426226573593,
                                 1.0567751179777043),
            # Fine XY grid sweep at the place pose: replace the single
            # snap_wait with n*n cells spanning [-extent, +extent] each
            # axis, ordered center-out. dwell_steps=1 → 1 follower step
            # per cell (fast scan); raise if the part needs more settle
            # time to land inside the tolerance box per cell.
            "search": {
                "n":          5,               # 5x5 = 25 cells
                "extent_xy":  (0.002, 0.002),  # match pos_tol_axes[0:2]
                "dwell_steps": 1,
            },
        },
    },
    "usb_a": {
        # pick_pos xy = mesh world pos from scene_init.usd; z = pick_z.
        "pick_pos":       np.array([ 0.02506, -0.07027, 1.04]),
        "place_pos":      np.array([ 0.23768,  0.05143, 1.05]),
        "gripper_close": 0.04,
        "init_height":    0.02,
        "release_mode":   "snap",
        "snap": {
            "movable_path":     "/World/parts/usb_a",
            "parent_body_path": "/World/task_board/task_board_color",
            "target_pos":       (0.23768, 0.05143, 1.05),  # = place_pos
            "target_rot":       (-0.5, -0.5, 0.5, 0.5),       # wxyz, from part_poses.json 'final'
            "pos_tol_axes":     (0.003, 0.003, 0.005),        # WORLD frame
            "rot_tol_deg":      10,
            "set_kinematic":    False,
            "timeout_steps":    300,
            "connect_pos":      (0.23768, 0.05143, 1.04),
            "connect_rot":      (-0.5, -0.5, 0.5, 0.5),  # = target_rot
            "search": {
                "n":          5,
                "extent_xy":  (0.002, 0.002),
                "dwell_steps": 1,
            },
        },
    },
}

# Order in which parts will be picked-and-placed when run_pick_place.py
# iterates. Each name must have an entry in PART_CONFIG (or fall back to
# PART_DEFAULTS for any missing keys).
#
# scene_base.usd has rod_16mm / bolt_8mm pre-assembled as PERMANENT_PARTS
# (the harness refreshes them in place rather than re-picking), and the
# gears authored at their pick positions so PhysX bakes their SDF colliders
# at stage-load time. Everything else in part_order spawns at runtime from
# ../parts/<name>.usdc at the pose recorded in part_init_poses.json.
part_order = (
    "gear_20teeth",
    "gear_60teeth",
    "rod_16mm",
    "bolt_8mm",
    "usb_a",
    "hdmi",
    "pin",
    "battery_size1",
    "battery_size5",
)

# Parts already assembled on the board. Their
# pick/place poses come from PART_CONFIG unchanged. Everything else is a
# "swap" part: it spawns at given positions, gets picked from
# there, and dropped at its own configured place_pos.
PERMANENT_PARTS = frozenset({"rod_16mm", "bolt_8mm"})

def get_part_config(name: str) -> dict:
    """Return PART_CONFIG[name] merged on top of PART_DEFAULTS.

    Merge order (later wins):
      1. PART_DEFAULTS
      2. pick_pos from PART_INIT_POSES[name] (composed from pos.xy + pick_z)
      3. PART_CONFIG[name]
      4. PART_CONFIG[name]["sequence"]  (only if len(part_order) > 1)

    The spawn pose (pos / orn) from PART_INIT_POSES is NOT pushed into cfg
    — read pc.PART_INIT_POSES[name]['pos' / 'orn'] directly for spawning
    (see import_missing_parts in run_pick_place.py).

    The ``sequence`` key itself is stripped from the returned config.
    """
    cfg = dict(PART_DEFAULTS)
    pose = PART_INIT_POSES.get(name)
    if pose is not None and "pick_pos" in pose:
        cfg["pick_pos"] = pose["pick_pos"]
    cfg.update(PART_CONFIG.get(name, {}))
    if len(part_order) > 1:
        seq = cfg.get("sequence")
        if isinstance(seq, dict) and seq:
            cfg.update(seq)
    cfg.pop("sequence", None)
    return cfg

# Hover delta-z above the grasp/place EE pose (m).
INIT_HEIGHT      = 0.1

# Post-open lift_place delta-z above the place pose (m). The gripper
# retracts to (place_pos + (0, 0, FINAL_HEIGHT)) after opening at the
# place location. None = use INIT_HEIGHT (existing behavior — symmetric
# hover/lift). Set to a smaller value for a low retract that doesn't
# disturb stacked parts, or larger to clear obstacles overhead.
FINAL_HEIGHT     = None

# End-of-iteration grader (see _grade_task in run_pick_place.py).
# Position tolerance (m) used for parts whose release_mode == "open" —
# pass iff mesh world XYZ is within this distance of place_pos. No
# orientation check (batteries / gears are axis-symmetric).
# Parts whose release_mode == "snap" are graded by snap-fired only.
GRADE_POS_TOL_M = 0.01   # 10 mm

# Per-phase include flags.
INCLUDE_CLOSE    = True     # phase 3
INCLUDE_OPEN     = True     # phase 7 (release at descend_place)

# Gripper dwell budgets (number of forward() calls; ~10 Hz => 5 ~= 0.5s).
SETTLE_CLOSE     = 10
SETTLE_OPEN      = 1
# Hold dwell at hover_place after the transit lerp arrives, before the
# descend kicks in. Lets the arm settle from any post-transit momentum so
# the descend starts from a stable hover.
SETTLE_HOVER_PLACE = 10
# Hold dwell at descend_place before the gripper opens, so the release
# happens with the EE actually settled at the place target (not just
# crossing the advance tolerance for one step).
SETTLE_DESCEND_PLACE = 15

# Dwell at the ``return_home`` waypoint after the cspace gate fires, before
# advancing to the next part's hover_pick. During this dwell the runner
# zeros joint velocities every step (see run_pick_place.py), so a longer
# value gives the arm more steady-state v=0 ticks before the next part's
# first IK call. 10 ≈ 1 s at the rig's default physics rate.
RETURN_HOME_SETTLE_STEPS = 10

# Joint-space-interpolated transit between lift_pick and hover_place.
# Inserts this many waypoints lerped in c-space (see PICK_PLACE_PHASES_
# CHEATSHEET.md for why this bypasses IK at the midpoints). 0 disables.
TRANSIT_STEPS    = 0 #usb-a needs 0.

# Joint-space-interpolated descend pacing. When > 0, inserts N intermediate
# joint-lerp waypoints between hover_pick / descend_pick (DESCEND_PICK_STEPS)
# and hover_place / descend_place (DESCEND_PLACE_STEPS). The follower
# lerps the commanded joint vector linearly across the segment instead of
# snapping the IK target straight to the bottom — same mechanism as
# TRANSIT_STEPS, but for the short vertical descend. Use this to slow down
# the descent for visual inspection or to reduce overshoot. 0 = one-step
# descend (existing behavior). Typical values: 4–8 for a smooth ~1 s drop
# at the rig's default physics rate.
DESCEND_PICK_STEPS  = 5
DESCEND_PLACE_STEPS = 5

# Path-follower advance tolerances.
POS_TOL          = 0.004
ORN_TOL          = 0.05

# Per-waypoint safety timeout (physics steps). If a waypoint's own
# `timeout_steps` is None and the gate doesn't fire within this many
# steps, the follower force-advances and logs a [follower] WARN. Set to
# None to disable. Tune up if you have legitimately long settling
# waypoints. Typical PD lag on this rig is well under 200 steps; 500 is
# a comfortable safety margin.
WAYPOINT_TIMEOUT_STEPS = 500


# Truncate the generated path to the first N waypoints (None = run all).
# Set to 2 for "phase 1 + phase 2" (hover_pick + descend_pick) only.
MAX_PHASES       = None
