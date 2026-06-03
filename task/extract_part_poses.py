"""Dump world poses of every prim under /World/parts in scene_base.usd and
write them to part_init_poses.json (preserving any hand-tuned `pick_z`
already in the file).

Output:
  1. Table of (part_name, world position, world orient quat) for the scene.
  2. Updated part_init_poses.json that the runner loads via
     param_config._load_part_init_poses().

Run:
    ${ISAAC_SIM}/python.sh extract_part_poses.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(HERE, ".."))
SCENES = [
    ("init",  os.path.join(PARENT, "scene_base.usd")),
]
PARTS_ROOT = "/World/parts"


def world_pose(prim) -> tuple:
    """Returns (translation_xyz, quat_wxyz) for a prim using XformCache."""
    cache = UsdGeom.XformCache()
    mat: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()  # Gf.Quatd: real + imaginary[3]
    imag = rot.GetImaginary()
    quat_wxyz = (rot.GetReal(), imag[0], imag[1], imag[2])
    return (float(t[0]), float(t[1]), float(t[2])), tuple(float(v) for v in quat_wxyz)


def _first_rigid_body(prim):
    """Return the first prim at-or-under `prim` carrying UsdPhysics.
    RigidBodyAPI — same logic snap_attach.resolve_rigid_body uses. None
    if no rigid body is found in the subtree."""
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
    return None


def _find_mesh_descendants(prim) -> list:
    """Return all Mesh prims at any depth under `prim` (DFS order)."""
    out = []
    stack = [prim]
    while stack:
        p = stack.pop()
        if p.GetTypeName() == "Mesh":
            out.append(p)
        # Children pushed in reverse so DFS order matches scene order.
        children = list(p.GetChildren())
        for c in reversed(children):
            stack.append(c)
    return out


def collect_parts(scene_path: str) -> dict:
    """Returns {part_name: {"path", "pos", "orn", "mesh_path"?}}.

    Walks /World/parts recursively. For each immediate child (the "part
    group", e.g. ``rod_16mm`` or a grouping like ``ICRA2022_practice``),
    finds Mesh descendants and reports the world pose of the *deepest*
    Mesh (which is the actual rendered object after all parent xformOps
    compose). If a group has multiple Meshes (e.g. visuals + collisions),
    each is listed under "<group>__<mesh-name>" so they don't collide.
    Groups with no Mesh fall back to the group's own world pose.
    """
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(scene_path)
    stage = Usd.Stage.Open(scene_path)
    if stage is None:
        raise RuntimeError(f"failed to open {scene_path}")
    root = stage.GetPrimAtPath(PARTS_ROOT)
    if not root or not root.IsValid():
        print(f"  WARNING: {PARTS_ROOT} not found in {scene_path}")
        return {}

    out = {}
    for top in root.GetChildren():
        name = top.GetName()
        # Rigid-body pose: what snap_attach.resolve_rigid_body sees. May
        # differ from the mesh pose if the mesh has a local xformOp that
        # offsets it from the body's frame.
        rb_prim = _first_rigid_body(top)
        if rb_prim is not None:
            try:
                rb_pos, rb_orn = world_pose(rb_prim)
                rb_info = {
                    "rb_path": rb_prim.GetPath().pathString,
                    "rb_pos":  rb_pos,
                    "rb_orn":  rb_orn,
                }
            except Exception as e:
                print(f"  rigid-body pose error for {rb_prim.GetPath()}: {e}")
                rb_info = {"rb_path": rb_prim.GetPath().pathString,
                           "rb_pos": None, "rb_orn": None}
        else:
            rb_info = {"rb_path": None, "rb_pos": None, "rb_orn": None}

        meshes = _find_mesh_descendants(top)
        if not meshes:
            # No mesh under this group — record the group's own pose so the
            # entry doesn't disappear. (Maybe it's a pure xform group, or the
            # mesh prim type differs from "Mesh".)
            try:
                pos, orn = world_pose(top)
            except Exception as e:
                print(f"  pose error for {top.GetPath()}: {e}")
                continue
            out[name] = {
                "path": top.GetPath().pathString,
                "pos": pos,
                "orn": orn,
                "mesh_path": None,
                **rb_info,
            }
            continue

        # One or more Meshes — sort deepest-first (longest path), then by name.
        meshes.sort(key=lambda p: (-p.GetPath().pathString.count("/"),
                                   p.GetPath().pathString))
        for i, m in enumerate(meshes):
            try:
                pos, orn = world_pose(m)
            except Exception as e:
                print(f"  pose error for {m.GetPath()}: {e}")
                continue
            key = name if i == 0 else f"{name}__{m.GetName()}"
            out[key] = {
                "path": top.GetPath().pathString,
                "mesh_path": m.GetPath().pathString,
                "pos": pos,
                "orn": orn,
                **rb_info,
            }
    return out


def fmt_xyz(t):  return f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"
def fmt_quat(q): return f"({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})"


def main():
    scene_data = {}
    for label, path in SCENES:
        print(f"\n========== {label.upper()}: {path} ==========")
        if not os.path.isfile(path):
            print(f"  MISSING: {path}")
            scene_data[label] = {}
            continue
        scene_data[label] = collect_parts(path)
        print(f"  {len(scene_data[label])} part entries under {PARTS_ROOT}")
        print(f"  {'name':<28} {'world pos':<32} {'world orn (wxyz)':<36} mesh_path")
        for name in sorted(scene_data[label]):
            v = scene_data[label][name]
            mp = v.get("mesh_path") or "<no mesh>"
            print(f"  {name:<28} {fmt_xyz(v['pos']):<32} "
                  f"{fmt_quat(v['orn']):<36} {mp}")

        # Rigid-body world poses: what snap_attach.resolve_rigid_body sees.
        # Only list parts whose rb pose differs from the mesh pose by > 1mm
        # (those are the cases where snap target_pos must use rb_pos, not
        # mesh pos — otherwise the snap proximity check is off-frame).
        rb_rows = []
        for name in sorted(scene_data[label]):
            v = scene_data[label][name]
            rb_pos = v.get("rb_pos")
            if rb_pos is None:
                continue
            dx = rb_pos[0] - v["pos"][0]
            dy = rb_pos[1] - v["pos"][1]
            dz = rb_pos[2] - v["pos"][2]
            if (dx * dx + dy * dy + dz * dz) ** 0.5 > 1e-3:
                rb_rows.append((name, v))
        if rb_rows:
            print(f"\n  ---- rigid-body poses (differ from mesh by > 1 mm) ----")
            print(f"  {'name':<28} {'rb world pos':<32} {'rb world orn (wxyz)':<36} rb_path")
            for name, v in rb_rows:
                print(f"  {name:<28} {fmt_xyz(v['rb_pos']):<32} "
                      f"{fmt_quat(v['rb_orn']):<36} {v.get('rb_path')}")

    # Side-by-side init -> final.
    print("\n========== INIT -> FINAL (delta in m) ==========")
    init = scene_data.get("init", {})
    final = scene_data.get("final", {})
    common = sorted(set(init) & set(final))
    init_only = sorted(set(init) - set(final))
    final_only = sorted(set(final) - set(init))
    if common:
        print(f"  {'name':<24} {'init pos':<32} {'final pos':<32} {'|d|':>8}")
        for name in common:
            ip = np.asarray(init[name]["pos"])
            fp = np.asarray(final[name]["pos"])
            d = float(np.linalg.norm(fp - ip))
            print(f"  {name:<24} {fmt_xyz(ip):<32} {fmt_xyz(fp):<32} {d:>8.4f}")
    if init_only:
        print(f"  Only in INIT  : {init_only}")
    if final_only:
        print(f"  Only in FINAL : {final_only}")

    # Python-dict block for easy pasting.
    print("\n========== Python config block (paste into your script) ==========")
    print("PART_POSES = {")
    for name in sorted(set(init) | set(final)):
        ip = init.get(name, {}).get("pos")
        ipo = init.get(name, {}).get("orn")
        fp = final.get(name, {}).get("pos")
        fpo = final.get(name, {}).get("orn")
        irb = init.get(name, {}).get("rb_pos")
        irbo = init.get(name, {}).get("rb_orn")
        frb = final.get(name, {}).get("rb_pos")
        frbo = final.get(name, {}).get("rb_orn")
        print(f"    {name!r}: {{")
        print(f"        'init':  {{'pos': {ip}, 'orn': {ipo},")
        print(f"                  'rb_pos': {irb}, 'rb_orn': {irbo}}},")
        print(f"        'final': {{'pos': {fp}, 'orn': {fpo},")
        print(f"                  'rb_pos': {frb}, 'rb_orn': {frbo}}},")
        print(f"    }},")
    print("}")
    print("# NOTE: for snap_attach target_pos/target_rot, use rb_pos/rb_orn —")
    print("# that is the frame snap_attach.resolve_rigid_body actually compares.")

    # Write the merged per-part record consumed by param_config.py /
    # run_pick_place.py. Flat schema: {<name>: {pos, orn, mesh_path, ...}}.
    # Preserves any hand-tuned `pick_z` from the existing file so re-running
    # this script never clobbers gripper-target tuning.
    json_out = os.path.join(HERE, "part_init_poses.json")
    existing_pick_z = {}
    if os.path.isfile(json_out):
        try:
            with open(json_out) as f:
                prior = json.load(f)
            for k, v in prior.items():
                if k.startswith("_") or not isinstance(v, dict):
                    continue
                if "pick_z" in v:
                    existing_pick_z[k] = v["pick_z"]
        except Exception as e:
            print(f"  WARNING: could not parse existing {json_out}: {e}")

    init = scene_data.get("init", {})
    merged = {"_comment": (
        "Per-part data for run_pick_place.py. ONE source of truth — auto-"
        "extracted spawn pose (pos/orn/mesh_path/rb_*) from "
        "extract_part_poses.py, plus hand-tuned pick_z (gripper target z; "
        "x/y inherit from pos). Re-running extract_part_poses.py preserves "
        "pick_z."
    )}
    for name in sorted(init):
        src = init[name]
        rec = {}
        if "path" in src:      rec["path"] = src["path"]
        if "mesh_path" in src: rec["mesh_path"] = src["mesh_path"]
        rec["pos"] = src["pos"]
        rec["orn"] = src["orn"]
        if name in existing_pick_z:
            rec["pick_z"] = existing_pick_z[name]
        for k in ("rb_path", "rb_pos", "rb_orn"):
            if k in src and src[k] is not None:
                rec[k] = src[k]
        merged[name] = rec

    with open(json_out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote {json_out}  ({len(init)} parts, "
          f"{len(existing_pick_z)} pick_z preserved)")

    simulation_app.close()


if __name__ == "__main__":
    main()
