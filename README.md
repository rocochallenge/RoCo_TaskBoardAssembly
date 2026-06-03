# 🏭 Industrial Task Board Assembly Track — 2nd RoCo Challenge @ IROS 2026

Official scene assets, robot description files, per-part USDs, and the Isaac
Sim evaluation harness for the **Industrial Task Board Assembly** Track of the
**2nd RoCo Challenge @ IROS 2026**.
The harness walks a fixed `part_order`, hands each part off to the
participant's `Policy`, and either snaps the part rigidly into its slot
(connectors / pin / rod) or lets gravity settle it on a slot (gears /
batteries). Scoring is `pass / total` over the 9 parts in `part_order`.

Participants implement one class — see
[`task/policy_api.py`](task/policy_api.py) and copy
[`task/policies/template.py`](task/policies/template.py).

## 📁 Repository Structure

```
AssemblyTask/
├── README.md                 # this file (organizer-facing overview)
├── PARTS.md                  # participant-facing parts/scoring reference
├── pyproject.toml            # uv project: Isaac Sim 5.1.0 + numpy deps, resolver settings
├── uv.lock                   # pinned dependency graph (uv sync reproduces .venv/ from this)
├── .python-version           # pins CPython 3.11 for uv
├── snap_attach.py            # proximity-triggered FixedJoint helper (SnapAttacher)
├── scene_init.usd            # init scene loaded by the runner (all parts pre-placed)
├── scene_base.usd            # base scene the init was authored from
├── parts/                    # per-part USDs (all parts now pre-resident in scene_init.usd)
│   ├── battery_size{1,5}.usdc
│   ├── bolt_8mm.usdc, bolt_rack.usdc
│   ├── hdmi.usdc, usb_a.usdc
│   ├── gear_20teeth.usdc, gear_60teeth.usdc
│   ├── pin.usdc, rod_16mm.usdc
│   ├── task_board.usdc
│   └── *_color.usdc, *_color_v2.usdc      # colored board variants
│   (note: battery_size7 / usb_c / ethernet / part_board are no longer
│    part of the task; their .usdc files may still exist but are unused)
├── robot/
│   ├── vega_1u_gripper.urdf               # source URDF
│   ├── vega_1u_gripper.usda               # USD authored from URDF (cameras baked in)
│   ├── vega_1u_gripper_collision.yaml
│   ├── gripper.usdc, gripper1.usdc        # flattened gripper geometry payloads
│   ├── meshes/                            # visual + collision meshes
│   ├── configuration/                     # Lula / IK config
│   └── sharpa_north/             # alternate URDF→USD robot export (not used by the runner)
├── table/
│   ├── OakTableLarge_meters.usd
│   ├── materials/, Textures/
└── task/                                  # evaluation harness + reference policy
    ├── run_pick_place.py                  # main entry point (the runner / grader)
    ├── policy_api.py                      # Policy / Observation / PartTarget / EnvInfo
    ├── policies/
    │   ├── baseline_scripted.py           # reference EEPathFollower policy
    │   └── template.py                    # participant stub
    ├── param_config.py                    # SCENE_USD, INIT_JOINT_TARGETS, PART_CONFIG, part_order, ...
    ├── part_init_poses.json               # auto-extracted spawn poses + hand-tuned pick_z
    ├── extract_part_poses.py              # rebuilds part_init_poses.json from the loaded scene
    ├── find_reachable_above_boards.py     # IK feasibility sweep (L arm only)
    ├── controllers/                       # EEPathFollower, Lula IK, snap helpers, Lula descriptor yamls (L + R)
    ├── scene_final.usd, demo.mp4          # reference run output
    └── README.md                          # runner notes + gotchas
```

The harness is single-arm by default — L picks every part, R stays
tucked at `INIT_JOINT_TARGETS`. The runner's controllers
(`task/controllers/`) already wrap **both** arms (`PickPlace_scene_bimanual`,
`_bimanual_robots`, L+R IK descriptors, L+R wrist cameras), so a
participant policy can command R as well by addressing the R arm dof
indices via `EnvInfo.R_arm_joints`.

## 🧩 Task Part Checklist

`task/param_config.py::part_order` is the source of truth for runtime
order. Each entry has a `PART_CONFIG` block. Release mode is what
distinguishes a snap-anchored placement from a gravity-settled drop.

See [`PARTS.md`](PARTS.md) for the full per-part table (snap targets,
grade poses, tolerances). Summary:

| # | part            | release | parent slot (snap)                       | notes                                       |
|---|-----------------|---------|------------------------------------------|---------------------------------------------|
| 1 | `gear_20teeth`  | open    | —                                        | sdf collider; settles on rack post          |
| 2 | `gear_60teeth`  | open    | —                                        | sdf collider; settles on rack post          |
| 3 | `rod_16mm` ★    | snap    | `task_board_color/root_001/_188_028`     | axis-symmetric; shares slot with bolt_8mm   |
| 4 | `bolt_8mm` ★    | snap    | `task_board_color/root_001/_188_028`     | axis-symmetric; shares slot with rod_16mm   |
| 5 | `usb_a`         | snap    | `task_board_color` (board root)          | transit-sensitive (`transit_steps` tuned)   |
| 6 | `hdmi`          | snap    | `task_board_color` (board root)          | connector inserted ~16 mm below target_pos  |
| 7 | `pin`           | snap    | `task_board_color/_188_001`              | axis-symmetric (rot gate off)               |
| 8 | `battery_size1` | open    | —                                        | graded by AABB midpoint                     |
| 9 | `battery_size5` | open    | —                                        | graded by AABB midpoint                     |

★ `PERMANENT_PARTS = {bolt_8mm, rod_16mm}` is still **defined** in
`param_config.py` but is **no longer consumed by the runner** (vestigial).
In this scene `rod_16mm` and `bolt_8mm` sit at ordinary pick positions in
`scene_init.usd` and are picked and snapped like every other snap part.

The board was renamed `right_board` → `task_board`, so full parent paths
are `/World/task_board/task_board_color/...`. `usb_a` / `hdmi` anchor at the
board root because their original per-connector sockets (`_188_032`, and
`_188_021` for the removed `ethernet`) were lost when the board was
restructured. The task board is static collision geometry, so a board-root
anchor pins the connector at `target_pos` exactly as a per-socket anchor
would. (Three parts — `battery_size7`, `usb_c`, `ethernet` — were removed
from the task entirely.)

All 9 parts are **pre-resident** in `scene_init.usd`; `import_missing_parts`
finds every prim already present and spawns nothing. `_apply_mesh_colliders()`
applied `UsdPhysics.CollisionAPI` + the configured approximation when the
scene was authored, so the parts are physical on load.

## 🧲 Placement Mechanics

Two release strategies live side by side, picked by
`PART_CONFIG[name]["release_mode"]`:

### 🔒 `release_mode = "snap"` — Proximity-Triggered FixedJoint

For connectors, bolts, the pin, and the rod, the gripper rarely lands
the part within the millimetre-scale tolerance of a real socket. Even
when it does, the part will rotate, vibrate, or be nudged when the
fingers release. To make these placements visually clean and
reproducible:

1. The runner drives a `SnapAttacher` (`snap_attach.py`) each physics
   step during the `snap_search` waypoint. The attacher compares the
   **mesh's** world pose against `target_pos` / `target_rot` from
   `PART_CONFIG[name]["snap"]`. Position is gated per-axis
   (`pos_tol_axes`, world frame); rotation is gated by `rot_tol_deg`
   (set negative to skip it for axis-symmetric parts).
2. When the part lands inside the box, the attacher
   - teleports the rigid body so the mesh exactly hits the joint
     anchor (`connect_pos` / `connect_rot`, mesh-frame; defaults to
     `target_pos` and the mesh's current rotation), then
   - authors a `UsdPhysics.FixedJoint` between the part's rigid body
     and `parent_body_path` (a sub-prim of `task_board_color`, or the
     board root itself).
3. The gripper opens and retracts. The joint keeps the part rigidly
   pinned to the board.

Why both halves are needed:

- The **teleport** removes the residual mm-scale offset the gripper
  couldn't close on its own, and removes the rotational yank PhysX
  would otherwise apply when the joint constraint solver kicks in
  (visible as a sudden tilt up to `rot_tol_deg`).
- The **FixedJoint** anchors the part so subsequent picks, the
  gripper finger sweeping past, or contact from a later-placed part
  cannot dislodge it. Without it, even a few-mm initial offset can
  let gravity walk the part out of the socket over the next 5 s.
- The proximity gate (rather than a fixed phase trigger) means the
  snap only fires once the gripper has actually delivered the part —
  if IK fails or the gripper drops it, the snap won't paste it into
  the slot anyway.
- An optional XY `search` grid (e.g. `n=5, extent_xy=(2 mm, 2 mm)`)
  sweeps cells center-out across the place pose, so a part that
  delivered slightly off-center still trips the gate on a nearby cell
  rather than getting force-advanced by `WAYPOINT_TIMEOUT_STEPS`.

Configure per-part inside `PART_CONFIG[name]["snap"]`:

```python
"snap": {
    "movable_path":     "/World/parts/<name>",
    "parent_body_path": "/World/task_board/task_board_color/<slot>",  # board root if the per-part socket is gone
    "target_pos":       (x, y, z),               # mesh-frame world pose
    "target_rot":       (w, x, y, z),
    "pos_tol_axes":     (0.002, 0.002, 0.005),   # world frame, per-axis
    "rot_tol_deg":      10,                       # < 0 disables rot gate
    "set_kinematic":    False,                    # leave dynamic for FixedJoint
    "timeout_steps":    300,
    "connect_pos":      (x, y, z),                # anchor (often = target_pos - insert depth)
    "connect_rot":      (w, x, y, z),             # always set explicitly — see task/README.md gotcha #6
    "search": {"n": 5, "extent_xy": (0.002, 0.002), "dwell_steps": 1},
},
```

### 🍃 `release_mode = "open"` — Natural Fall-Off

For the gears and batteries, the slot geometry does the work. The
gripper opens at `place_pos`, the part falls a few millimetres, and
contact with the rack post / battery cradle settles it into a
repeatable final pose. No joint is authored, the part stays fully
dynamic, and subsequent grasps can re-pick it if needed.

Two consequences worth flagging:

- The grade is by **landed pose**, not gripper-release pose:
  `PART_CONFIG[name]["grade_pos"]` records the measured settled
  position (AABB midpoint when `grade_use_aabb=True`), and
  `_grade_task` checks `mesh_world_xyz` against it with
  `GRADE_POS_TOL_M = 10 mm`. `place_pos` is just where the gripper
  lets go; gravity moves the part from there.
- `collision_approximation` matters here. Gears use `sdf` because
  convexDecomposition smears the tooth profile and the gear walks off
  the rack post. Convex parts default to `convexDecomposition` —
  `convexHull` only for purely convex shapes. SDF + snap FixedJoint is
  a known crash combo (see `task/README.md` gotcha #5); SDF is fine
  here precisely because no joint is authored.

## 📷 Camera Interfaces

Three cameras are **baked into `robot/vega_1u_gripper.usda`** (and
therefore into `scene_init.usd`) as Camera prims on the robot
hierarchy — the runner does **not** instantiate them at runtime
anymore:

| camera        | USD prim path                                                                | gaze            |
|---------------|------------------------------------------------------------------------------|-----------------|
| `headcam`     | `/World/robotics/vega_1u_gripper/zed_depth_frame/headcam`                    | along parent −Y |
| `L_wristcam`  | `/World/robotics/vega_1u_gripper/L_ee_link/gripper_link/L_wristcam`          | along parent +Z |
| `R_wristcam`  | `/World/robotics/vega_1u_gripper/R_ee_link/gripper_link/R_wristcam`          | along parent +Z |

All three are 640×480 with Isaac Sim defaults for intrinsics. The
runner's `setup_pick_place_sim` looks them up by path and (optionally)
binds `Camera` wrappers around them.

Two flags in `param_config.py` control runtime camera behaviour:

```python
enable_camera_viewports = True   # show the 3-tile viewport layout in Kit UI
enable_camera_output    = False  # bind sensors so RGB/depth are readable from Python
```

`enable_camera_output = False` skips the sensor binding entirely (no
RGB/depth bound; viewport tiles still draw if `enable_camera_viewports`
is True). When `True`, the harness exposes `head_depth_camera`,
`L_wrist_camera`, `R_wrist_camera` handles to the policy via
`Observation.rgb` / `Observation.depth` / `Observation.intrinsics`. To
dump frames to disk, do it inside your policy (e.g. via PIL for PNG or
numpy for `.npy`); the harness does not save frames itself.

Relocating a camera is done by editing the prim's `xformOp:translate` /
`xformOp:orient` in `robot/vega_1u_gripper.usda` (or `scene_init.usd`)
— there are no Python-side overrides anymore.

## 📦 Git LFS

This repository contains large simulation assets. Before adding USD, USDC,
OBJ, or MP4 files to Git, install Git LFS and configure the repository from
the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_git_lfs.ps1
```

The script tracks `*.usd`, `*.usdc`, `*.obj`, and `*.mp4` through Git LFS.
Commit the generated `.gitattributes` file before staging matching assets, so
large files are stored as LFS objects instead of ordinary Git blobs.

## ⚙️ Environment Setup

The Python environment is managed with [`uv`](https://docs.astral.sh/uv/).
Isaac Sim ships as pip packages, so no separate Omniverse-launcher
install is needed — `uv sync` builds a self-contained `.venv/` with the
pinned **Isaac Sim 5.1.0** stack on **CPython 3.11**.

```bash
# from the repo root
uv sync                       # builds .venv/ from uv.lock (~18 GB download)
```

uv drives everything from three checked-in files at the repo root:

- `pyproject.toml` — dependencies (`isaacsim[all,extscache]==5.1.0.0`,
  `numpy<2`) plus resolver settings (linux / x86_64 only; numpy pinned to
  1.26.4, which Isaac Sim requires).
- `.python-version` — pins the interpreter to CPython 3.11.
- `uv.lock` — the fully resolved, hash-pinned graph. `uv sync` reproduces
  the exact environment from it; keep it committed.

Run anything inside the env with `uv run`, or activate the venv directly:

```bash
uv run python -c "import isaacsim; print('ready')"
# or
source .venv/bin/activate
```

**First import accepts the NVIDIA Omniverse EULA.** Importing `isaacsim`
prints the EULA and waits for acceptance. Answer `Yes` interactively, or
accept it non-interactively before running:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

**Disk note.** The Isaac Sim stack is ~18 GB. uv caches downloaded wheels
under `~/.cache/uv` and hardlinks them into `.venv/`. If your home
filesystem is small, point the cache at a larger disk **on the same
filesystem as the repo** (so uv hardlinks instead of copying — otherwise
you pay for the ~18 GB twice):

```bash
export UV_CACHE_DIR=/path/to/big-disk/.uv-cache
uv sync
```

## 🚀 Running the Harness

With the env synced (see [Environment setup](#environment-setup)) and the
EULA accepted, launch the harness with `uv run` from the repo root — the
runner fixes up `sys.path` so `param_config` / `controllers` / `policies`
import correctly:

```bash
uv run python task/run_pick_place.py                                  # baseline policy
uv run python task/run_pick_place.py --policy policies.my_team.MyPolicy
uv run python task/run_pick_place.py --results-json out/results.json  # dump per-part pass/fail
```

If you instead have a standalone Omniverse-launcher Isaac Sim, its
bundled interpreter still works:
`${ISAAC_SIM}/python.sh task/run_pick_place.py`.

CLI flags:

- `--policy <dotted-path>` — Policy subclass to instantiate. Default is
  `policies.baseline_scripted.BaselinePolicy`. Resolved by Python
  import; the module must be on `sys.path` (the `policies/` directory
  next to `policy_api.py` is added automatically by the template /
  baseline boilerplate).
- `--results-json <path>` — overrides `pc.RESULTS_JSON_PATH`. The
  harness writes a JSON of per-part pass/fail (with measured / target
  positions and tolerances) at the end of the run.

Edit `task/param_config.py::part_order` to run a subset for debugging.
See `task/README.md` for the per-runner gotchas (scene drive targets,
the `SingleRigidPrim` ↔ FixedJoint tensor-view crash, SDF + FixedJoint
crash, snap `connect_rot` requirement, etc.).
