# Eval harness for the IROS 2026 vega_1u assembly challenge.
#
# Iterates pc.part_order. For each part: optionally spawns it, creates a
# snap-attacher (success detector) if the release_mode is "snap", asks the
# loaded Policy to drive the L arm via act(obs) each physics step, and
# advances on policy.is_done() / snap fire / per-part timeout. Scores at
# the end via _grade_task (pass/fail per part, optionally written to JSON).
#
# Select the policy with --policy module.path.ClassName. Default:
# policies.baseline_scripted.BaselinePolicy (the reference scripted solver).
# Participants subclass policy_api.Policy and pass --policy <their module>.
#
# R arm holds its init joint pose every step.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import importlib
import json
import os
import sys
import numpy as np

import param_config as pc
from controllers.vega_1u_setup import (
    restore_scene_part_xforms, setup_pick_place_sim,
)
from controllers.part_from_usd import DynamicPart
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from policy_api import EnvInfo, Observation, PartTarget

# Physics material prim authored in the scene USD; bound to every spawned
# DynamicPart so newly imported parts share the same friction/restitution
# profile as rod_16mm / bolt_8mm (which are already in scene_base.usd).
_PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterial"

# snap_attach.py lives one directory up from Task_test, alongside the
# USD scene. Add that directory to sys.path so the runner can import it.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import omni.physx
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from snap_attach import SnapAttacher, _quat


# Collision approximation applied to each spawned part's Mesh descendants.
# convexDecomposition handles concave shapes well at the cost of cooking
# time. Override per-part via PART_CONFIG[name]["collision_approximation"]
# if a particular part wants e.g. "convexHull" (faster) or "sdf" (tighter).
_DEFAULT_COLLISION_APPROXIMATION = "convexDecomposition"

# Stuck detector: print one diagnostic block when the follower stalls on a
# single waypoint for this many physics steps without advancing.
# ~100 steps ≈ 0.5 s at 200 Hz physics. Re-armed on the next wp_idx change.
STUCK_LOG_STEPS = 100

# URDF<->USD frame offset on the EE link. Mirror of _STAGE_OFFSET_INV in
# controllers/lula_ik_controller.py — used here to convert Lula's FK output
# (URDF frame) into the stage-frame orientation we can compare directly
# against wp.orn. R_offset = (0, 0, 0, -1) (180° about Z). Without this
# composition the stuck-diagnostic comparison shows a fake ~180° error.
_R_OFFSET_FK_TO_STAGE = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)


def _quat_mul(q1, q2):
    """Hamilton product (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ===========================================================================
# Joint-ownership masks for the L+R action merge.
# ===========================================================================
R_ARM_JOINT_NAMES = [f"R_arm_j{i}" for i in range(1, 8)]
L_OWNED_JOINTS = {
    "Lift", "torso_flip",
    "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
    "L_arm_j5", "L_arm_j6", "L_arm_j7",
    "L_gripper_joint", "L_gripper_joint_01",
}
R_OWNED_JOINTS = {
    "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
    "R_arm_j5", "R_arm_j6", "R_arm_j7",
    "R_gripper_joint", "R_gripper_joint_01",
}


def _value_or_none(seq, i):
    try:
        v = seq[i]
    except (TypeError, IndexError):
        return None
    return v


def merge_bimanual_actions(L_action, R_action, dof_names):
    n = len(dof_names)

    def _vec(action):
        v = getattr(action, "joint_positions", None)
        if v is None:
            return [None] * n
        return list(v)

    Lp = _vec(L_action)
    Rp = _vec(R_action)
    merged = [None] * n
    for i, jname in enumerate(dof_names):
        lv = _value_or_none(Lp, i)
        rv = _value_or_none(Rp, i)
        if jname in L_OWNED_JOINTS:
            merged[i] = lv if lv is not None else rv
        elif jname in R_OWNED_JOINTS:
            merged[i] = rv if rv is not None else lv
        else:
            merged[i] = lv if lv is not None else rv
    return ArticulationAction(joint_positions=merged)


def build_snap_attacher(stage, part_name, snap_cfg):
    """Construct a SnapAttacher from a param_config snap dict.

    Converts target_pos / target_rot tuples to Gf types and picks a joint
    path unique to the part. ``joint_path`` defaults to
    ``/World/_snap_joint_<part_name>`` so concurrent or sequential snaps
    don't collide on the same prim path.
    """
    target_pos = Gf.Vec3d(*snap_cfg["target_pos"])
    w, x, y, z = snap_cfg["target_rot"]
    target_rot = _quat(w, x, y, z)
    connect_rot_tup = snap_cfg.get("connect_rot")
    connect_rot = _quat(*connect_rot_tup) if connect_rot_tup is not None else None
    connect_offset_rot_tup = snap_cfg.get("connect_offset_rot")
    connect_offset_rot = (_quat(*connect_offset_rot_tup)
                          if connect_offset_rot_tup is not None else None)
    return SnapAttacher(
        stage,
        movable_path=snap_cfg["movable_path"],
        parent_body_path=snap_cfg["parent_body_path"],
        target_pos=target_pos,
        target_rot=target_rot,
        pos_tol=snap_cfg.get("pos_tol", 0.005),
        pos_tol_axes=snap_cfg.get("pos_tol_axes"),
        rot_tol_deg=snap_cfg.get("rot_tol_deg", 5.0),
        joint_path=snap_cfg.get("joint_path",
                                f"/World/_snap_joint_{part_name}"),
        debug=snap_cfg.get("debug", False),
        debug_every=snap_cfg.get("debug_every", 30),
        set_kinematic_on_snap=snap_cfg.get("set_kinematic", False),
        mesh_path=snap_cfg.get("mesh_path"),
        author_joint_on_snap=snap_cfg.get("author_joint", True),
        connect_pos=snap_cfg.get("connect_pos"),
        connect_rot=connect_rot,
        connect_offset_pos=snap_cfg.get("connect_offset_pos"),
        connect_offset_rot=connect_offset_rot,
    )


# Directory holding the per-part USD files (../parts relative to this script).
_PARTS_USD_DIR = os.path.join(_PARENT_DIR, "parts")


def _apply_mesh_colliders(stage, prim_path, approximation):
    """Walk every Mesh descendant of `prim_path` and apply
    UsdPhysics.CollisionAPI + MeshCollisionAPI(approximation=...).

    Most exported part USDs ship visual meshes only — no PhysicsCollisionAPI.
    Without this pass, DynamicPart's set_collision_approximation() call on the
    root xform has nothing to act on and the spawned part has zero physical
    collider, so the gripper passes through it. Idempotent: a mesh that
    already has CollisionAPI gets its approximation refreshed and nothing
    else. Returns the count of meshes touched.
    """
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return 0
    n = 0
    for p in Usd.PrimRange(root):
        if p.GetTypeName() != "Mesh":
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(p)
        mesh_api.CreateApproximationAttr().Set(approximation)
        n += 1
    return n


def import_missing_parts():
    """Populate the stage with every part recorded in part_init_poses.json
    that isn't already present, then handle any remaining pc.part_order
    parts not covered by the JSON.

    Spawn pose precedence per part:
      1. pc.PART_INIT_POSES[name]['pos' / 'orn'] (primary — bulk spawn).
      2. cfg['pick_pos'] for position and cfg['spawn_orn'] (or identity)
         for orientation (fallback for pc.part_order parts not in
         part_init_poses.json).

    Parts already present in the loaded scene USD are left alone.
    """
    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()

    # Wrap the scene-authored physics material so we can bind it to each
    # spawned part. If the prim is missing (older scene), fall through to
    # DynamicPart's auto-created default.
    shared_phys_mat = None
    if is_prim_path_valid(_PHYSICS_MATERIAL_PATH):
        shared_phys_mat = PhysicsMaterial(prim_path=_PHYSICS_MATERIAL_PATH)
    else:
        print(f"[setup] WARNING: {_PHYSICS_MATERIAL_PATH} not in scene — "
              f"spawned parts will get DynamicPart's default material.")

    def _spawn(name, pos, orn, source):
        prim_path = f"/World/parts/{name}"
        usd_path = os.path.join(_PARTS_USD_DIR, f"{name}.usdc")
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(
                f"part {name!r} missing from stage and no USD at {usd_path}"
            )
        # Add the reference FIRST so the DynamicPart constructor sees an
        # existing prim. The "prim already valid" branch in
        # VisualPart.__init__ skips the default gray PreviewSurface override
        # that would otherwise mask the part USD's authored materials.
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        # Apply CollisionAPI + approximation to the Mesh descendants before
        # DynamicPart wraps the prim. Without this, the part has no physical
        # collider and the gripper passes through it.
        cfg = pc.get_part_config(name) if name in pc.PART_CONFIG else {}
        approximation = cfg.get(
            "collision_approximation", _DEFAULT_COLLISION_APPROXIMATION
        )
        n_meshes = _apply_mesh_colliders(stage, prim_path, approximation)
        DynamicPart(
            prim_path=prim_path,
            name=name,
            position=np.asarray(pos, dtype=np.float64),
            orientation=np.asarray(orn, dtype=np.float64),
            physics_material=shared_phys_mat,
        )

    # Pass 1: every part recorded in part_init_poses.json. Parts already
    # present in the scene USD (e.g. gears authored in scene_base.usd
    # so PhysX bakes their SDF collider at stage-load time) are left
    # alone here. JSON XY overrides for scene-resident parts already
    # happened pre-World in vega_1u_setup._override_scene_part_xy_inplace
    # (must run before task wrappers snapshot the "default" pose,
    # otherwise World.reset() reverts the override).
    for name, entry in pc.PART_INIT_POSES.items():
        if "pos" not in entry or "orn" not in entry:
            continue
        prim_path = f"/World/parts/{name}"
        if is_prim_path_valid(prim_path):
            continue
        _spawn(name, entry["pos"], entry["orn"], "part_init_poses.json")

    # Pass 2: anything still missing that pc.part_order asks for — falls back
    # to PART_CONFIG values (covers parts not in part_init_poses.json).
    for name in pc.part_order:
        if is_prim_path_valid(f"/World/parts/{name}"):
            continue
        cfg = pc.get_part_config(name)
        pos = cfg.get("pick_pos")
        if pos is None:
            raise ValueError(
                f"part {name!r} not in part_init_poses.json and has no "
                f"pick_pos in PART_CONFIG — cannot spawn."
            )
        spawn_orn = cfg.get("spawn_orn")
        if spawn_orn is None:
            spawn_orn = identity_q
        _spawn(name, pos, spawn_orn, "PART_CONFIG")


def _save_stage_snapshot(out_path):
    """Save the current Isaac Sim stage to a flattened USD file.

    Output is a single self-contained file that can be re-opened in Isaac
    Sim (or any USD viewer) and played to test post-placement physics
    (e.g. whether a part stays put or falls). Path is resolved relative
    to this script's directory if not absolute. Parent dirs are created
    if missing.
    """
    if not out_path:
        return
    abs_path = (out_path if os.path.isabs(out_path)
                else os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), out_path)))
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print(f"[setup] WARN: no stage to save to {abs_path}")
        return
    # stage.Export() flattens all sublayers + references into one file.
    try:
        stage.Export(abs_path)
        print(f"[setup] saved final stage snapshot -> {abs_path}")
    except Exception as e:
        print(f"[setup] WARN: failed to save stage to {abs_path}: {e}")


def _override_scene_part_xy(stage, prim_path, target_xy, name):
    """Surgically update the XY of a scene-resident part. Z preserved,
    scale and rotation preserved. Handles both authoring conventions:
      1. xformOp:translate / orient / scale (Isaac Sim default for
         spawned parts) — update the translate op's value directly.
      2. xformOp:transform (single matrix, Composer's default when you
         drag in a reference) — pull the translation out of the matrix,
         swap XY, write the matrix back. Rotation/scale rows are
         untouched.
    """
    target_x, target_y = (float(target_xy[0]), float(target_xy[1]))
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        print(f"[setup] {name}: prim {prim_path} not in stage — no XY override.")
        return

    # Pick the rigid-body prim if there is one; otherwise the root.
    body_prim = root
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prim = p
            break

    xform = UsdGeom.Xformable(body_prim)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    transform_op = None
    for op in ops:
        t = op.GetOpType()
        if t == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif t == UsdGeom.XformOp.TypeTransform and transform_op is None:
            transform_op = op

    if translate_op is not None:
        cur = translate_op.Get()
        cur_z = float(cur[2]) if cur is not None else 0.0
        translate_op.Set(Gf.Vec3d(target_x, target_y, cur_z))
        print(f"[setup] {name}: overrode xformOp:translate XY to "
              f"({target_x:+.5f}, {target_y:+.5f}) on {body_prim.GetPath()} "
              f"(Z preserved at {cur_z:+.5f}).")
        return

    if transform_op is not None:
        mat = transform_op.Get()
        if mat is None:
            print(f"[setup] {name}: xformOp:transform has no authored value.")
            return
        old_t = mat.ExtractTranslation()
        cur_z = float(old_t[2])
        # SetTranslateOnly preserves the rotation/scale rows of the matrix.
        new_mat = Gf.Matrix4d(mat)
        new_mat.SetTranslateOnly(Gf.Vec3d(target_x, target_y, cur_z))
        transform_op.Set(new_mat)
        print(f"[setup] {name}: overrode xformOp:transform translation XY to "
              f"({target_x:+.5f}, {target_y:+.5f}) on {body_prim.GetPath()} "
              f"(Z preserved at {cur_z:+.5f}).")
        return

    op_names = [op.GetName() for op in ops]
    print(f"[setup] {name}: no xformOp:translate OR xformOp:transform on "
          f"{body_prim.GetPath()} (found: {op_names}) — can't override XY.")


def _grade_task(stage, snap_fired_parts, results_json_path=None):
    """End-of-iteration summary: pass/fail for every name in pc.part_order.

    Grading rule per part:
      release_mode == "snap"  -> pass iff name in snap_fired_parts.
      release_mode == "open"  -> pass iff the part's MESH world position
                                 is within GRADE_POS_TOL_M of place_pos.
                                 (No orientation check — batteries / gears
                                 are axis-symmetric.)

    If ``results_json_path`` is non-None (or pc.RESULTS_JSON_PATH is set),
    the per-part outcome is also written to that path as JSON for offline
    aggregation.
    """
    GRADE_POS_TOL_M = float(getattr(pc, "GRADE_POS_TOL_M", 0.01))
    print("=" * 72)
    print(f"[grade] task summary (pos tol = {GRADE_POS_TOL_M * 1000:.1f} mm):")
    n_pass = 0
    n_fail = 0
    n_missing = 0
    per_part_results = []
    for part in pc.part_order:
        if isinstance(part, str) and part.startswith("<"):
            continue
        cfg = pc.get_part_config(part)
        release_mode = cfg.get("release_mode", "open")
        if release_mode == "snap":
            fired = part in snap_fired_parts
            status = "pass" if fired else "FAIL"
            print(f"  {part:<16}  snap={'fired' if fired else 'NOT fired':<10}  "
                  f"-> {status}")
            per_part_results.append({
                "name": part,
                "release_mode": "snap",
                "snap_fired": bool(fired),
                "pass": bool(fired),
            })
            if fired:
                n_pass += 1
            else:
                n_fail += 1
            continue

        # Position-only grade. Prefer cfg["grade_pos"] (final settled
        # pose, post-release) over place_pos (gripper release pose) —
        # they often differ when the part sinks / rolls / settles after
        # the gripper opens.
        place_pos = cfg.get("grade_pos")
        if place_pos is None:
            place_pos = cfg.get("place_pos")
        if place_pos is None:
            print(f"  {part:<16}  no grade_pos / place_pos -> SKIP")
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "no grade_pos / place_pos",
            })
            continue
        prim = stage.GetPrimAtPath(f"/World/parts/{part}")
        if not prim or not prim.IsValid():
            print(f"  {part:<16}  prim missing from stage -> MISSING")
            n_missing += 1
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "prim missing",
            })
            continue
        deepest_mesh = None
        deepest_d = -1
        for p in Usd.PrimRange(prim):
            if p.GetTypeName() != "Mesh":
                continue
            d = p.GetPath().pathString.count("/")
            if d > deepest_d:
                deepest_d = d
                deepest_mesh = p
        if deepest_mesh is None:
            print(f"  {part:<16}  no Mesh descendant -> MISSING")
            n_missing += 1
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "no mesh descendant",
            })
            continue
        # Choose between mesh-translation and AABB-midpoint as the
        # "actual position" reading. For axis-symmetric parts like
        # batteries, the mesh local origin may sit anywhere — and a
        # rotation about the symmetry axis moves mesh_world_t even
        # though the part is geometrically in the same place. AABB
        # midpoint (world-axis-aligned bounds, midpoint of min/max per
        # axis) is invariant under that rotation and gives a fair
        # position-only comparison. Opt in per-part via cfg["grade_use_aabb"]
        # = True; grade_pos must also be expressed as the AABB midpoint
        # (not the mesh local origin) for the comparison to be apples-to-
        # apples.
        if cfg.get("grade_use_aabb", False):
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_],
            )
            bbox = bbox_cache.ComputeWorldBound(deepest_mesh)
            aabb = bbox.ComputeAlignedRange()
            mid = aabb.GetMidpoint()
            cur = np.array([float(mid[0]), float(mid[1]), float(mid[2])],
                           dtype=np.float64)
            measure = "AABBmid"
        else:
            m = UsdGeom.XformCache().GetLocalToWorldTransform(deepest_mesh)
            t = m.ExtractTranslation()
            cur = np.array([float(t[0]), float(t[1]), float(t[2])],
                           dtype=np.float64)
            measure = "meshT"
        target = np.asarray(place_pos, dtype=np.float64)
        delta = cur - target  # 3D per-axis error
        err = float(np.linalg.norm(delta))
        ok = err < GRADE_POS_TOL_M
        status = "pass" if ok else "FAIL"
        print(f"  {part:<16}  pos_err={err * 1000:7.2f} mm "
              f"d=({delta[0]*1000:+6.2f}, {delta[1]*1000:+6.2f}, "
              f"{delta[2]*1000:+6.2f}) mm ({measure})  -> {status}")
        # On FAIL, dump full actual/target so you can re-baseline grade_pos.
        if not ok:
            print(f"    actual=({float(cur[0]):.6f}, {float(cur[1]):.6f}, "
                  f"{float(cur[2]):.6f})")
            print(f"    target=({float(target[0]):.6f}, "
                  f"{float(target[1]):.6f}, {float(target[2]):.6f})")
        n_pass += 1 if ok else 0
        n_fail += 0 if ok else 1
        per_part_results.append({
            "name": part,
            "release_mode": "open",
            "measure": measure,
            "pos_err_m": err,
            "tolerance_m": GRADE_POS_TOL_M,
            "actual": [float(cur[0]), float(cur[1]), float(cur[2])],
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "pass": bool(ok),
        })

    print(f"[grade] summary: pass={n_pass}  fail={n_fail}  missing={n_missing}")
    print("=" * 72)

    # Optional JSON dump for offline aggregation.
    out_path = (results_json_path
                if results_json_path is not None
                else getattr(pc, "RESULTS_JSON_PATH", None))
    if out_path:
        abs_path = (out_path if os.path.isabs(out_path)
                    else os.path.abspath(os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), out_path)))
        try:
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            payload = {
                "pos_tol_m": GRADE_POS_TOL_M,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_missing": n_missing,
                "per_part": per_part_results,
            }
            with open(abs_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[grade] wrote results JSON -> {abs_path}")
        except Exception as e:
            print(f"[grade] WARN: failed to write results JSON to {abs_path}: {e}")


def _parse_args():
    """CLI: --policy and --results-json overrides."""
    parser = argparse.ArgumentParser(
        description="vega_1u assembly challenge eval harness."
    )
    parser.add_argument(
        "--policy",
        default="policies.baseline_scripted.BaselinePolicy",
        help="Dotted import path to a Policy subclass "
             "(default: policies.baseline_scripted.BaselinePolicy).",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Override pc.RESULTS_JSON_PATH. If set, _grade_task writes the "
             "per-part pass/fail summary to this file at the end of the run.",
    )
    # SimulationApp consumes argv too; tolerate unknown args so the runner
    # can be launched as ${ISAAC_SIM}/python.sh run_pick_place.py --policy ...
    return parser.parse_known_args()[0]


def _load_policy_class(dotted_path: str):
    """Resolve `module.path.ClassName` -> the class object."""
    if "." not in dotted_path:
        raise ValueError(
            f"--policy must be dotted (module.ClassName), got {dotted_path!r}"
        )
    module_name, _, class_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"could not import policy module {module_name!r}: {e}"
        ) from e
    if not hasattr(module, class_name):
        raise AttributeError(
            f"policy module {module_name!r} has no attribute {class_name!r}"
        )
    return getattr(module, class_name)


def main():
    args = _parse_args()

    # The task signature still requires L/R object prim paths. Point both
    # at a STATIC prim so the task's SingleRigidPrim wrapper never aliases
    # a part that snap_attach later joint-locks — that aliasing was what
    # invalidated the physics tensor view mid-snap. L/R_target_position
    # are stored as observation labels we never query, so a dummy zero
    # vector is fine.
    _DUMMY_TARGET = np.zeros(3, dtype=np.float64)
    (my_world, my_controller, my_robots,
     head_depth_camera, L_wrist_camera, R_wrist_camera,
     articulation_controller, task_params, reset_needed) = setup_pick_place_sim(
        L_object_prim_path=pc.L_object_prim_path,
        R_object_prim_path=pc.R_object_prim_path,
        L_target_position=_DUMMY_TARGET,
        R_target_position=_DUMMY_TARGET,
        joint_opened_position=np.array([pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=pc.enable_camera_viewports,
        enable_camera_output=pc.enable_camera_output,
    )

    # Spawn any pc.part_order entries that aren't already in the loaded scene.
    import_missing_parts()

    L_controller = my_controller["L"]
    R_controller = my_controller["R"]
    L_robot = my_robots["L"]
    R_robot = my_robots["R"]

    dof_names = list(L_robot.dof_names)
    R_arm_dof_indices = np.array(
        [dof_names.index(j) for j in R_ARM_JOINT_NAMES], dtype=np.int64
    )
    L_gripper_dof_index = dof_names.index("L_gripper_joint")
    L_arm_joint_names = [j for j in dof_names if j.startswith("L_arm_j")]

    def _apply_init_joint_targets():
        """Override the live joint state with pc.INIT_JOINT_TARGETS.

        Called at startup and after every World.reset() (stop+play). Velocities
        are zeroed too so PD doesn't carry residual motion through the teleport.
        """
        targets = getattr(pc, "INIT_JOINT_TARGETS", None)
        if not targets:
            return
        full_q = np.asarray(L_robot.get_joint_positions(),
                            dtype=np.float64).copy()
        for jname, val in targets.items():
            if jname in dof_names:
                full_q[dof_names.index(jname)] = float(val)
        L_robot.set_joint_positions(full_q)
        L_robot.set_joint_velocities(np.zeros(len(dof_names)))

    _apply_init_joint_targets()

    # R: latch init pose, command those joints every step.
    R_arm_hold_q = np.asarray(L_robot.get_joint_positions())[R_arm_dof_indices].astype(np.float64)

    # Snapshot the L arm's c-space joint vector at startup. The baseline
    # policy uses this as the return-home target between parts; other
    # policies can use it for whatever (or ignore it).
    L_arm_init_q = np.asarray(
        L_controller.current_cspace_q(), dtype=np.float64
    ).copy()

    # Build EnvInfo and load the chosen policy. `L_controller` is stashed
    # on env_info so the BaselinePolicy can wrap it in EEPathFollower;
    # participant policies should ignore that attribute.
    env_info = EnvInfo(
        dof_names=dof_names,
        L_arm_joints=L_arm_joint_names,
        R_arm_joints=list(R_ARM_JOINT_NAMES),
        L_gripper_joint="L_gripper_joint",
        L_arm_init_q=L_arm_init_q.copy(),
        physics_dt=1.0 / 200.0,
        enable_camera_output=pc.enable_camera_output,
        L_controller=L_controller,
    )

    policy_class = _load_policy_class(args.policy)
    policy = policy_class(env_info)
    print(f"[setup] policy: {policy_class.__module__}.{policy_class.__name__}")

    # Snap attacher lifecycle (env-owned success detector).
    stage = omni.usd.get_context().get_stage()
    physx_iface = omni.physx.get_physx_interface()

    parts_iter = iter(pc.part_order)
    current_part = None
    current_snap_attacher = None
    current_snap_sub = None
    snap_fired_parts = set()
    part_step_count = 0
    PER_PART_TIMEOUT_STEPS = int(getattr(pc, "PER_PART_TIMEOUT_STEPS", 3000))

    def _clear_snap_state():
        nonlocal current_snap_attacher, current_snap_sub
        # Drop the subscription first so the dying attacher can't be ticked
        # by a stray physx event between the two None assignments.
        current_snap_sub = None
        current_snap_attacher = None

    def _build_observation():
        full_q = np.asarray(L_robot.get_joint_positions(), dtype=np.float64)
        try:
            full_qd = np.asarray(L_robot.get_joint_velocities(), dtype=np.float64)
        except Exception:
            full_qd = np.zeros_like(full_q)

        # EE pose via Lula FK at the last commanded q, composed with the
        # URDF<->stage frame offset so we return stage-frame quaternions.
        ee_pos = np.zeros(3, dtype=np.float64)
        ee_orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        try:
            ik = getattr(L_controller, "_ik", None)
            fk_fn = getattr(ik, "fk_for_last_command", None) if ik else None
            if fk_fn is not None:
                p, o = fk_fn()
                if p is not None:
                    ee_pos = np.asarray(p, dtype=np.float64)
                if o is not None:
                    ee_orn = np.asarray(
                        _quat_mul(o, _R_OFFSET_FK_TO_STAGE), dtype=np.float64
                    )
        except Exception:
            pass

        rgb = {"head": None, "L_wrist": None, "R_wrist": None}
        depth = {"head": None, "L_wrist": None, "R_wrist": None}
        intrinsics = {"head": None, "L_wrist": None, "R_wrist": None}
        if pc.enable_camera_output:
            for key, cam in (("head", head_depth_camera),
                             ("L_wrist", L_wrist_camera),
                             ("R_wrist", R_wrist_camera)):
                if cam is None:
                    continue
                try:
                    rgba = cam.get_rgba()
                    if rgba is not None and rgba.size > 0:
                        rgb[key] = np.asarray(rgba[..., :3])
                except Exception:
                    pass
                try:
                    frame = cam.get_current_frame()
                    if frame and frame.get("distance_to_image_plane") is not None:
                        depth[key] = np.asarray(frame["distance_to_image_plane"],
                                                dtype=np.float32)
                except Exception:
                    pass
                try:
                    K = cam.get_intrinsics_matrix()
                    if K is not None:
                        intrinsics[key] = np.asarray(K, dtype=np.float64)
                except Exception:
                    pass

        snap_fired = bool(current_snap_attacher is not None
                          and current_snap_attacher.attached)

        return Observation(
            step_idx=int(my_world.current_time_step_index),
            joint_positions=full_q,
            joint_velocities=full_qd,
            L_gripper_position=float(full_q[L_gripper_dof_index]),
            ee_pose_L=(ee_pos, ee_orn),
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            snap_fired=snap_fired,
            target_part=current_part if isinstance(current_part, str) else None,
        )

    def _build_part_target(name):
        cfg = pc.get_part_config(name)
        snap = cfg.get("snap") or {}
        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).copy()
        return PartTarget(
            name=name,
            release_mode=cfg.get("release_mode", "open"),
            pick_pos=_arr(cfg.get("pick_pos")),
            spawn_orn=_arr(cfg.get("spawn_orn")),
            place_pos=_arr(cfg.get("place_pos")),
            grade_pos=_arr(cfg.get("grade_pos")),
            snap_target_pos=_arr(snap.get("target_pos")),
            snap_target_rot=_arr(snap.get("target_rot")),
            snap_pos_tol=_arr(snap.get("pos_tol_axes")),
            snap_rot_tol_deg=(None if snap.get("rot_tol_deg") is None
                              else float(snap["rot_tol_deg"])),
            gripper_open=float(cfg.get("gripper_open", 0.0)),
            gripper_close=float(cfg.get("gripper_close", 0.0)),
            ee_orientation=_arr(cfg.get("ee_orientation")),
            extra=dict(cfg),
        )

    def _finalize_iteration():
        _grade_task(stage, snap_fired_parts,
                    results_json_path=args.results_json)
        save_path = getattr(pc, "SAVE_FINAL_STAGE_PATH", None)
        if save_path:
            _save_stage_snapshot(save_path)

    def _start_next_part():
        """Advance to the next part: build snap attacher, call policy.reset()."""
        nonlocal current_part, current_snap_attacher, current_snap_sub
        nonlocal part_step_count

        # Record previous part's snap status before clearing.
        if (current_part is not None
                and current_snap_attacher is not None
                and current_snap_attacher.attached):
            snap_fired_parts.add(current_part)
        _clear_snap_state()

        try:
            current_part = next(parts_iter)
        except StopIteration:
            current_part = None
            print("[setup] All parts done.")
            _finalize_iteration()
            return None

        cfg = pc.get_part_config(current_part)
        release_mode = cfg.get("release_mode", "open")
        snap_cfg = cfg.get("snap")
        if release_mode == "snap":
            if snap_cfg is None:
                raise ValueError(
                    f"part {current_part!r} has release_mode='snap' but no "
                    f"'snap' config dict in PART_CONFIG."
                )
            current_snap_attacher = build_snap_attacher(
                stage, current_part, snap_cfg,
            )
            attacher = current_snap_attacher
            current_snap_sub = physx_iface.subscribe_physics_step_events(
                lambda dt, a=attacher: a.update()
            )
            print(f"[setup] {current_part}: snap mode  "
                  f"movable={snap_cfg['movable_path']}  "
                  f"target_pos={snap_cfg['target_pos']}")

        obs = _build_observation()
        target = _build_part_target(current_part)
        policy.reset(obs, target)
        part_step_count = 0
        print(f"now working on the part: {current_part}", flush=True)
        return current_part

    def _restart_iteration():
        nonlocal parts_iter, current_part
        _clear_snap_state()
        snap_fired_parts.clear()
        # Remove any FixedJoints that snap_attach authored on previous
        # iterations. Joints live in USD and persist across my_world.stop()
        # / play(), so without cleanup the bolt (and any other snap part)
        # stays anchored to wherever the previous run's snap pinned it.
        _stage = omni.usd.get_context().get_stage()
        if _stage is not None:
            for _name in pc.PART_CONFIG.keys():
                _joint_path = f"/World/_snap_joint_{_name}"
                if is_prim_path_valid(_joint_path):
                    _stage.RemovePrim(_joint_path)
                    print(f"[setup] removed stale snap joint at {_joint_path}")
        # Restore scene-resident parts' xformOps to startup snapshot.
        restore_scene_part_xforms()
        parts_iter = iter(pc.part_order)
        current_part = None
        _start_next_part()

    _restart_iteration()

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if my_world.is_stopped():
                reset_needed = True
            continue

        if reset_needed:
            my_world.reset()
            reset_needed = False
            _apply_init_joint_targets()
            L_controller.reset()
            R_controller.reset()
            _restart_iteration()

        if my_world.current_time_step_index == 0:
            _apply_init_joint_targets()
            L_controller.reset()
            R_controller.reset()
            _restart_iteration()

        # Warmup: skip task logic until PhysX has had time to cook
        # colliders (SDF meshes in particular) and joints have settled to
        # their init drive targets.
        _warmup_steps = int(getattr(pc, "WARMUP_STEPS", 0))
        if (_warmup_steps > 0
                and my_world.current_time_step_index < _warmup_steps):
            _apply_init_joint_targets()
            if my_world.current_time_step_index == _warmup_steps - 1:
                print(f"[setup] warmup done ({_warmup_steps} steps); "
                      f"starting task.")
            continue

        if current_part is None:
            continue

        obs = _build_observation()

        # Latch snap-fired into the per-iteration record as soon as the
        # attacher reports it (so a snap that fires on the very last tick
        # before timeout still counts as pass at grading time).
        if (current_snap_attacher is not None
                and current_snap_attacher.attached):
            snap_fired_parts.add(current_part)

        cfg = pc.get_part_config(current_part)
        is_snap_done = (cfg.get("release_mode") == "snap"
                        and current_snap_attacher is not None
                        and current_snap_attacher.attached)
        is_timeout = part_step_count >= PER_PART_TIMEOUT_STEPS

        if policy.is_done(obs) or is_snap_done or is_timeout:
            if is_timeout:
                print(f"[setup] {current_part}: per-part timeout "
                      f"({PER_PART_TIMEOUT_STEPS} steps) — advancing.")
            _start_next_part()
            continue

        L_action = policy.act(obs)

        R_action_positions = [None] * len(dof_names)
        for j_idx, val in zip(R_arm_dof_indices, R_arm_hold_q.tolist()):
            R_action_positions[j_idx] = float(val)
        R_action = ArticulationAction(joint_positions=R_action_positions)

        merged = merge_bimanual_actions(L_action, R_action, dof_names)
        articulation_controller.apply_action(merged)

        part_step_count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
