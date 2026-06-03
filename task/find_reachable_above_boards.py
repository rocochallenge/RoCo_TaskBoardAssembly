"""Find L-arm IK-reachable poses in a 3D grid above the two boards.

Builds a grid of XYZ points covering the AABB of the two board prims, with
Z values stepping upward from a small offset above the boards' top surface.
Probes Lula IK at each (target_position, ee_orn) pair using the same L-arm
descriptor + URDF the runtime controller uses (selected by
param_config.OWNS_LIFT_L / OWNS_TORSO_L). Writes:

  <prefix>_reachable.ply    — points where IK succeeded
  <prefix>_unreachable.ply  — points where IK failed (handy to see "gaps")
  <prefix>_boards_aabb.ply  — 16 corner markers of the two board AABBs
                              (red = board #1, green = board #2)

Open all three in MeshLab / CloudCompare / Open3D simultaneously to see
the reachable workspace overlaid on the board volume.

This script must be run with Isaac Sim's python (it imports Lula):
    ${ISAAC_SIM}/python.sh find_reachable_above_boards.py

Examples
--------
# Defaults: scene from param_config, boards autodetected by name match,
# 20x20 XY grid x 6 Z layers from +1 cm to +25 cm above board top.
python find_reachable_above_boards.py

# Override the EE orientation / offset (e.g. to match pin's grasp):
python find_reachable_above_boards.py --part pin

# Tighter grid, more Z steps:
python find_reachable_above_boards.py --xy-res 40 --z-steps 12
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import param_config as pc  # noqa: E402


ROBOT_PRIM_PATH = "/World/robotics/vega_1u_gripper"

# Tolerances mirror controllers/lula_ik_controller.py so the OK/FAIL verdict
# matches the actual controller's IK success/failure.
IK_POS_TOL = 1e-3
IK_ORN_TOL = 5e-2

# URDF<->USD frame offset on the vega_1u EE link — copied verbatim from
# controllers/lula_ik_controller.py so probes match runtime IK behavior.
_STAGE_OFFSET_INV = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _orn_for_lula(orn):
    if orn is None:
        return None
    return _quat_mul(np.asarray(orn, dtype=np.float64), _STAGE_OFFSET_INV)


def get_world_pose(prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = get_prim_at_path(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    imag = rot.GetImaginary()
    pos = np.array([t[0], t[1], t[2]], dtype=np.float64)
    quat_wxyz = np.array([rot.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)
    return pos, quat_wxyz


def _world_aabb(prim: Usd.Prim) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """World-frame AABB of `prim` (including descendants). Returns (min, max)
    or None if no bound-able geometry under it."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    return (np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
            np.array([mx[0], mx[1], mx[2]], dtype=np.float64))


def _autodetect_board_paths(stage: Usd.Stage) -> List[str]:
    """Find prim paths whose name ends in '_board' under /World."""
    out = []
    for p in stage.Traverse():
        path = p.GetPath().pathString
        name = p.GetName().lower()
        # Only top-level board prims, not deep children.
        if path.startswith("/World/") and path.count("/") == 2 and "board" in name:
            out.append(path)
    return out


def write_ply(path: str, points, rgb=None) -> None:
    """ASCII PLY. `rgb` optional: either one (r,g,b) for all, or a list
    matching `points`."""
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    n = len(pts)
    has_color = rgb is not None
    if has_color and isinstance(rgb, tuple) and len(rgb) == 3:
        rgb_list = [rgb] * n
    else:
        rgb_list = list(rgb) if has_color else None
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if has_color:
                r, g, b = rgb_list[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"  wrote {n} pts -> {path}")


def _aabb_corners(mn: np.ndarray, mx: np.ndarray) -> List[np.ndarray]:
    return [
        np.array([x, y, z]) for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1])
                            for z in (mn[2], mx[2])
    ]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_scene = os.path.abspath(os.path.join(_HERE, pc.SCENE_USD))
    ap.add_argument("--scene", default=default_scene,
                    help="Scene USD path. Default: param_config.SCENE_USD.")
    ap.add_argument("--boards", nargs="+", default=None,
                    metavar="PATH",
                    help="One or more board prim paths. Default: "
                         "autodetect any /World/*_board prims (case-insensitive).")
    ap.add_argument("--part", default=None, metavar="NAME",
                    help="If set, pull ee_orientation / ee_offset from "
                         "pc.get_part_config(NAME). Otherwise uses PART_DEFAULTS.")
    ap.add_argument("--ee-orn", nargs=4, type=float, default=None,
                    metavar=("W", "X", "Y", "Z"),
                    help="Override EE orientation (wxyz quat).")
    ap.add_argument("--ee-offset", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Override ee_offset.")
    ap.add_argument("--xy-res", type=int, default=20,
                    help="XY grid resolution per axis. Default: 20.")
    ap.add_argument("--xy-margin", type=float, default=0.0,
                    help="Extra meters added to each side of the boards' "
                         "XY AABB before gridding. Default: 0.")
    ap.add_argument("--z-min-above", type=float, default=0.01,
                    help="Z offset above the boards' top (max-Z) for the "
                         "lowest grid layer (m). Default: 0.01.")
    ap.add_argument("--z-max-above", type=float, default=0.25,
                    help="Z offset above the boards' top for the highest "
                         "grid layer (m). Default: 0.25.")
    ap.add_argument("--z-steps", type=int, default=6,
                    help="Number of Z layers between z-min-above and "
                         "z-max-above (inclusive endpoints). Default: 6.")
    ap.add_argument("--z-inspection-below", type=float, default=0.01,
                    help="Add one extra Z layer at (board top - this) for "
                         "inspecting reachability AT or below the board "
                         "surface. Default: 0.01 (1 cm below top). Set to "
                         "<=0 to disable the extra layer.")
    ap.add_argument("--target-is-ee-pos", action="store_true",
                    help="Interpret each grid point as the EE world target "
                         "directly (no ee_offset added). Default: treat each "
                         "grid point as a fingertip target and add ee_offset.")
    ap.add_argument("--output-prefix", default=None,
                    help="Path prefix for output PLYs. Default: "
                         "reachable_above_boards next to this script.")
    # Use parse_known_args so Isaac Sim's own runtime flags (e.g.
    # --/rtx/verifyDriverVersion/enabled=false) don't trip our parser.
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[info] ignoring unknown args (likely Isaac Sim runtime): {unknown}")
    return args


def _resolve_ee_pose(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.part is not None:
        cfg = pc.get_part_config(args.part)
        ee_orn = np.asarray(cfg["ee_orientation"], dtype=np.float64)
        ee_offset = np.asarray(cfg["ee_offset"], dtype=np.float64)
    else:
        ee_orn = np.asarray(pc.PART_DEFAULTS["ee_orientation"], dtype=np.float64)
        ee_offset = np.asarray(pc.PART_DEFAULTS["ee_offset"], dtype=np.float64)
    if args.ee_orn is not None:
        ee_orn = np.asarray(args.ee_orn, dtype=np.float64)
    if args.ee_offset is not None:
        ee_offset = np.asarray(args.ee_offset, dtype=np.float64)
    return ee_orn, ee_offset


def _l_descriptor_path() -> str:
    owns_lift = bool(getattr(pc, "OWNS_LIFT_L",
                             getattr(pc, "OWNS_TORSO_L", False)))
    owns_torso = bool(getattr(pc, "OWNS_TORSO_L", False))
    if owns_torso:
        suffix = ""
    elif owns_lift:
        suffix = "_liftonly"
    else:
        suffix = "_armonly"
    return os.path.join(_HERE, "controllers",
                        f"vega_1u_L_arm_description{suffix}.yaml")


def main() -> None:
    args = _parse_args()

    print(f"scene: {args.scene}")
    if not os.path.isfile(args.scene):
        raise SystemExit(f"scene USD not found: {args.scene}")
    open_stage(usd_path=args.scene)

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise SystemExit("failed to open stage")

    # --- Find board prims & compute combined AABB.
    board_paths = args.boards or _autodetect_board_paths(stage)
    if not board_paths:
        raise SystemExit(
            "no board prims found. Pass --boards /World/<path> explicitly."
        )
    print(f"boards: {board_paths}")

    board_aabbs: List[Tuple[np.ndarray, np.ndarray]] = []
    for bp in board_paths:
        prim = stage.GetPrimAtPath(bp)
        if not prim or not prim.IsValid():
            print(f"  WARNING: {bp} not in stage, skipping")
            continue
        aabb = _world_aabb(prim)
        if aabb is None:
            print(f"  WARNING: {bp} empty AABB, skipping")
            continue
        board_aabbs.append(aabb)
        print(f"  {bp} AABB: min={np.round(aabb[0], 4).tolist()} "
              f"max={np.round(aabb[1], 4).tolist()}")
    if not board_aabbs:
        raise SystemExit("no valid board AABBs")

    combined_min = np.min(np.stack([a[0] for a in board_aabbs]), axis=0)
    combined_max = np.max(np.stack([a[1] for a in board_aabbs]), axis=0)
    combined_min[0] -= args.xy_margin
    combined_min[1] -= args.xy_margin
    combined_max[0] += args.xy_margin
    combined_max[1] += args.xy_margin
    print(f"combined XY: x=[{combined_min[0]:.4f}, {combined_max[0]:.4f}]  "
          f"y=[{combined_min[1]:.4f}, {combined_max[1]:.4f}]  "
          f"top-z={combined_max[2]:.4f}")

    # --- Build grid.
    xs = np.linspace(combined_min[0], combined_max[0], int(args.xy_res))
    ys = np.linspace(combined_min[1], combined_max[1], int(args.xy_res))
    z0 = combined_max[2] + float(args.z_min_above)
    z1 = combined_max[2] + float(args.z_max_above)
    zs = list(np.linspace(z0, z1, int(args.z_steps)))
    # Optional inspection layer below the board top.
    if float(args.z_inspection_below) > 0:
        z_inspect = combined_max[2] - float(args.z_inspection_below)
        zs = [z_inspect] + zs
        print(f"  inspection layer added at z={z_inspect:.4f} "
              f"({args.z_inspection_below:.3f} m below board top)")
    zs = np.asarray(zs, dtype=np.float64)
    print(f"grid: {len(xs)}x{len(ys)} XY × {len(zs)} Z layers = "
          f"{len(xs) * len(ys) * len(zs)} probes")
    print(f"  z values: {np.round(zs, 4).tolist()}")

    # --- Robot base + IK solver.
    robot_pos, robot_orn = get_world_pose(ROBOT_PRIM_PATH)
    L_desc = _l_descriptor_path()
    urdf_path = os.path.abspath(os.path.join(_HERE, "..", "robot",
                                             "vega_1u_gripper.urdf"))
    print(f"robot base pos: {np.round(robot_pos, 4).tolist()}")
    print(f"L descriptor : {L_desc}")
    print(f"URDF         : {urdf_path}")

    ee_orn, ee_offset = _resolve_ee_pose(args)
    print(f"ee_orn (wxyz): {np.round(ee_orn, 4).tolist()}")
    print(f"ee_offset    : {np.round(ee_offset, 4).tolist()}  "
          f"(applied: {not args.target_is_ee_pos})")

    ik = LulaKinematicsSolver(
        robot_description_path=L_desc,
        urdf_path=urdf_path,
    )
    ik.set_robot_base_pose(
        robot_position=robot_pos,
        robot_orientation=robot_orn,
    )
    ee_frame = "L_ee_link_gripper_link"
    target_orn_lula = _orn_for_lula(ee_orn)

    # --- Probe.
    reachable: List[np.ndarray] = []
    unreachable: List[np.ndarray] = []
    n = 0
    for z in zs:
        for y in ys:
            for x in xs:
                fingertip = np.array([x, y, z], dtype=np.float64)
                target_pos = (fingertip if args.target_is_ee_pos
                              else fingertip + ee_offset)
                _, ok = ik.compute_inverse_kinematics(
                    frame_name=ee_frame,
                    target_position=target_pos,
                    target_orientation=target_orn_lula,
                    warm_start=None,
                    position_tolerance=IK_POS_TOL,
                    orientation_tolerance=IK_ORN_TOL,
                )
                # Older / newer Isaac Sim returns (q, success) or
                # (action, success); ok is a bool either way.
                if bool(ok):
                    reachable.append(fingertip)
                else:
                    unreachable.append(fingertip)
                n += 1
                if n % 200 == 0:
                    print(f"  probed {n}/{len(xs)*len(ys)*len(zs)}  "
                          f"reachable so far: {len(reachable)}")

    total = len(reachable) + len(unreachable)
    print(f"\nresults: {len(reachable)}/{total} reachable "
          f"({100*len(reachable)/total:.1f}%)")

    # --- Write outputs.
    prefix = args.output_prefix or os.path.join(_HERE, "reachable_above_boards")
    write_ply(prefix + "_reachable.ply", reachable, rgb=(0, 200, 0))
    write_ply(prefix + "_unreachable.ply", unreachable, rgb=(200, 50, 50))
    corners = []
    corner_rgb = []
    palette = [(255, 0, 0), (0, 0, 255), (255, 128, 0), (0, 200, 200)]
    for i, (mn, mx) in enumerate(board_aabbs):
        c = _aabb_corners(mn, mx)
        corners.extend(c)
        corner_rgb.extend([palette[i % len(palette)]] * len(c))
    write_ply(prefix + "_boards_aabb.ply", corners, rgb=corner_rgb)

    simulation_app.close()


if __name__ == "__main__":
    main()
