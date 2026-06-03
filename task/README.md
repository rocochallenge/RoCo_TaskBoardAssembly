# task/ — vega_1u pick-and-place eval harness

The runner side of the challenge. `run_pick_place.py` loads
`../scene_init.usd` (all 9 parts pre-resident), walks `param_config.part_order`,
hands each part to the participant's `Policy`, and grades the result
(per-part pass/fail with optional JSON dump).

## File map

| File | Role |
|------|------|
| `run_pick_place.py`       | Eval harness. Sets up sim, spawns missing parts, walks `part_order`, instantiates the selected `Policy`, drives it through each part, then grades. Has the stuck detector (`_diagnose_stuck`) and the `_grade_task` summary. |
| `policy_api.py`           | Participant-facing contract: `Policy` ABC + `EnvInfo` / `PartTarget` / `Observation` dataclasses. |
| `policies/baseline_scripted.py` | Reference scripted policy (`BaselinePolicy`). Original EEPathFollower-driven pick-and-place — byte-identical to the pre-Policy-API runner output. |
| `policies/template.py`    | Participant stub. Copy to `policies/<your_team>.py`, fill in `reset` / `act` / `is_done`, run with `--policy policies.<your_team>.MyPolicy`. |
| `param_config.py`         | Single source of truth for non-pose config — `PART_DEFAULTS` / `PART_CONFIG`, `INIT_JOINT_TARGETS`, IK descriptor flags, phase-tuning constants, `SCENE_USD`, `enable_camera_*`, `PER_PART_TIMEOUT_STEPS`, `RESULTS_JSON_PATH`. **No `pick_pos` lives here** — see `part_init_poses.json`. |
| `part_init_poses.json`    | Per-part spawn pose (`pos`, `orn`) + hand-tuned `pick_z`. Pick x/y come from `pos`. Re-running `extract_part_poses.py` preserves `pick_z`. |
| `extract_part_poses.py`   | Scrapes per-part mesh / rigid-body world poses from the loaded scene; writes `part_init_poses.json` (merging existing `pick_z`). |
| `find_reachable_above_boards.py`   | L-arm IK-feasibility sweep above the board AABBs. Writes reachable / unreachable PLYs for visualization. Must run with Isaac Sim's python. |
| `controllers/`            | EEPoseController, EEPathFollower, LulaIKController, snap helpers, pick-place task, Lula descriptor yamls for both arms (L + R). |
| `controllers/vega_1u_L_arm_description*.yaml` | Lula descriptors for L arm — `default_q` mirrors the scene init pose (see *Gotchas* below). Three variants picked by `pc.OWNS_LIFT_L` / `OWNS_TORSO_L`. |
| `controllers/vega_1u_R_arm_description*.yaml` | Same, R arm. Available so a policy can do bimanual IK; the default single-arm runner just holds R at `INIT_JOINT_TARGETS`. |
| `scene_final.usd`, `demo.mp4` | Reference output of a passing baseline run. |

## Running

The environment is uv-managed (see the root `README.md` for `uv sync`).
Run from the repo root — `uv run` finds `pyproject.toml` automatically:

```bash
uv run python task/run_pick_place.py                                  # baseline policy
uv run python task/run_pick_place.py --policy policies.my_team.MyPolicy
uv run python task/run_pick_place.py --results-json out/results.json  # dump per-part pass/fail
```

A standalone Omniverse-launcher Isaac Sim still works:
`${ISAAC_SIM}/python.sh task/run_pick_place.py`.

`--policy` defaults to `policies.baseline_scripted.BaselinePolicy`.
`--results-json` overrides `pc.RESULTS_JSON_PATH`.

Edit `pc.part_order` to choose which parts to run. All 9 parts are
pre-resident in `scene_init.usd`; any `part_order` entry that *isn't* in
the loaded scene would be spawned at runtime from `../parts/<name>.usdc`
at the pose recorded in `part_init_poses.json`.

## Policy contract (quick reference)

Participants subclass `policy_api.Policy`:

```python
class MyPolicy(Policy):
    def __init__(self, env_info: EnvInfo): ...   # called once at sim startup
    def reset(self, obs, target: PartTarget): ...# called at the start of each part
    def act(self, obs) -> ArticulationAction:    # called every physics step
        ...
    def is_done(self, obs) -> bool:              # called every physics step
        ...
```

Per-part loop:

1. Harness publishes `Observation` (joint state, EE pose, snap state,
   optional RGB/depth).
2. `policy.reset(obs, PartTarget(name, release_mode, pick_pos, place_pos, snap_target))`.
3. While `not is_done(obs) and not env_done`: harness applies `act(obs)`
   and steps physics.
4. `env_done` = snap fired (snap-mode only) or step count exceeded
   `pc.PER_PART_TIMEOUT_STEPS`. The harness always advances on these
   even if `is_done` keeps returning False.
5. `_grade_task` scores the part.

Full schema in `policy_api.py`; reference implementation in
`policies/baseline_scripted.py`.

## Config primer (`param_config.py`)

**Scene** — `SCENE_USD = "../scene_init.usd"`, resolved relative to
`task/`. All 9 parts are pre-resident in it, so `import_missing_parts`
spawns nothing.

**Cameras** — three cameras are baked into the robot USD (head + L/R
wrist). Two flags here just toggle viewports and sensor binding:

```python
enable_camera_viewports = True   # show the 3-tile viewport layout in Kit UI
enable_camera_output    = False  # bind sensors so RGB/depth are readable from Python
```

When `enable_camera_output = True`, the harness binds the three cameras
and surfaces frames via `Observation.rgb` / `depth` / `intrinsics` to
the policy.

**IK descriptor mode** — three modes per arm via the (`OWNS_LIFT_*`,
`OWNS_TORSO_*`) flag pair:

| flags | yaml suffix | DOFs in cspace |
|---|---|---|
| `(False, False)` | `_armonly` | j1..j7 only |
| `(True, False)`  | `_liftonly` | Lift + j1..j7 |
| `(True, True)`   | `` (full) | Lift + torso + j1..j7 |

`OWNS_TORSO=True with OWNS_LIFT=False` is rejected.

**R-arm rest pose** — `R_ARM_TUCKED` (True = j1 folded at −90°, False =
USDA forward `-15, -20, 0, ...`). Also sets the **L descriptor's**
`R_arm_j*` fixed values; flipping it without re-authoring the L yamls
leaves Lula's R collision spheres slightly off.

**Startup pose** — `INIT_JOINT_TARGETS` is applied at sim start and on
every Stop+Play via `set_joint_positions`. Currently mirrors
`scene_init.usd`'s authored drive targets so PD doesn't fight the
override.

**Phase tuning** — `INIT_HEIGHT`, `TRANSIT_STEPS`, `DESCEND_PICK_STEPS`,
`DESCEND_PLACE_STEPS`, `POS_TOL`, `ORN_TOL`, `SETTLE_*`, `MAX_PHASES`,
`WAYPOINT_TIMEOUT_STEPS`, `PER_PART_TIMEOUT_STEPS`. `init_height` and
`transit_steps` are per-part overridable in `PART_CONFIG[name]`.

**Per-part config** — `PART_CONFIG[name]` keys override `PART_DEFAULTS`.
Common keys: `ee_orientation`, `ee_offset`, `gripper_open`,
`gripper_close`, `place_pos`, `release_mode` (`"open"` | `"snap"`),
`snap` (dict), `collision_approximation`, `transit_steps`,
`init_height`, `final_height`, `spawn_orn`, `sequence` (multi-part
overrides).

## Pose / collider data flow

```
extract_part_poses.py  reads scene_base.usd
        ↓ (preserves pick_z)
part_init_poses.json   ← single source of truth for spawn pose + pick z
        ↓
param_config._load_part_init_poses()
        ↓
pc.PART_INIT_POSES[name] = {pos, orn, pick_pos}
        ↓                ↓
run_pick_place           pc.get_part_config(name) → {..., pick_pos}
.import_missing_parts                  ↓
   spawns part at        Policy.act(obs) (every physics step)
   (pos, orn)                ↓
                         articulation action → IK → physics
```

**Spawn flow** (`import_missing_parts`):
1. Pass 1: every name in `pc.PART_INIT_POSES` not already in the stage
   gets a `DynamicPart` referenced in at `pos`/`orn`.
2. Pass 2: any `pc.part_order` name not in the JSON falls back to
   `cfg.pick_pos` and `cfg.spawn_orn`.

Each spawn also runs `_apply_mesh_colliders()` which walks the
referenced part's Mesh descendants and applies `UsdPhysics.CollisionAPI`
+ the configured approximation. Without this the spawned part has no
collider and the gripper passes through it.

## Gotchas / lessons learned

### 1. Scene drive targets override USDA defaults

`scene_init.usd` authors `drive:angular:physics:targetPosition` on every
L arm joint, overriding the gripper USDA's USDA-neutral pose. The
scene's L pose is the *contorted* `[-30, +60, +100, -100, -10, -10,
-60]°`. Two consequences:

- `INIT_JOINT_TARGETS["L_arm_j*"]` must equal the **scene** drive
  targets, not the USDA defaults — otherwise PD fights `set_joint_
  positions` after every World.reset().
- `default_q` in the L description yamls must match the scene pose
  too, so Lula's null-space rest-bias doesn't pull solutions toward a
  pose the PD has long since left.

Both are currently in sync. If the scene pose changes, update *three*
places: `param_config.INIT_JOINT_TARGETS["L_arm_j*"]`, all three
`vega_1u_L_arm_description*.yaml` `default_q`s, and (if R IK is ever
used) the R yamls' `cspace_to_urdf_rules` L-fixed values.

### 2. SingleRigidPrim task wrappers vs snap_attach FixedJoint

The task (`PickPlaceTask_scene_bimanual.set_up_scene`) wraps
`L_object_prim_path` / `R_object_prim_path` as `SingleRigidPrim`s for
observation tracking. If either path points at a part that
`snap_attach` later authors a `FixedJoint` on, the two views fight,
PhysX rebuilds the simulation tensor view, and the next call to
`get_dof_positions()` / `get_world_pose()` throws `Failed to get …
from backend`.

**Fix:** point both paths at a *static* prim that snap will never
target. `param_config.py` currently ships pointing both at
`/World/parts/rod_16mm` (`L_object_prim_path` / `R_object_prim_path`,
marked `# static`).

⚠️ **Stale in the 9-part task:** `rod_16mm` is no longer a
`PERMANENT_PART` — it is now picked and snapped like every other snap
part, so `snap_attach` authors a `FixedJoint` on it. That makes it
exactly the kind of prim this gotcha warns against. Point both paths at
the static task board instead — `/World/task_board/task_board_color`
(static collision geometry, never snapped).

### 3. DomeLight `color_0C0C0C.exr` texture errors

USD-authored DomeLights sometimes carry a phantom
`inputs:texture:file = ./textures/color_0C0C0C.exr` synthesized by a
color picker and never written to disk. The renderer logs a noisy
error. Fix: clear the attribute. Done once for `scene_init.usd` and
for the per-part USDs in `../parts/*.usdc`.

### 4. Part USDs ship without CollisionAPI

Most part USDs in `../parts/` author visual meshes only. Without
runtime `UsdPhysics.CollisionAPI` + `MeshCollisionAPI`, the gripper
passes through the part. `_apply_mesh_colliders()` in
`run_pick_place.py` adds the API at spawn time. Approximation defaults
to `convexDecomposition`; override per-part via
`PART_CONFIG[name]["collision_approximation"]`.

### 5. SDF colliders + runtime FixedJoint = tensor view crash

Switching a part to `"collision_approximation": "sdf"` *and* having
snap_attach author a runtime FixedJoint on it can invalidate physics
tensor views (similar mechanism to #2). `convexDecomposition` is also
better for threaded bolts in rigid-body contact anyway (smooth contact
strip vs SDF's crest-only contacts). Use `convexHull` for chunky/convex
parts, `sdf` only for insertions when you really need accurate cavity
contact.

### 6. Snap `connect_rot` must be set explicitly

If `snap.connect_rot` is omitted, the joint anchor uses the mesh's
*current* (potentially rot-tol-degrees-off) rotation at snap-fire time
— part lands visibly tilted. Always set `connect_rot = target_rot` (or
the desired final rotation).

### 7. Stuck detector orn err can look like 180°

`_diagnose_stuck` in `run_pick_place.py` reads Lula's FK output and
composes it with `R_OFFSET = (0, 0, 0, -1)` before comparing to wp.orn
— otherwise the URDF↔USD 180°-about-Z offset shows up as a fake
~180° "error" every time.

### 8. Per-part overrides via `sequence`

When a part runs as part of a multi-part sequence (len(part_order) > 1),
its `PART_CONFIG[name]["sequence"]` sub-dict's keys are merged in on
top of the standalone values. Used to re-tune `pick_pos` /
`ee_offset` / `gripper_*` to compensate for the drift that
accumulates in chained runs. Stripped from `get_part_config()`'s
return.

## Quick recipes

**Write a new policy:**
1. Copy `policies/template.py` to `policies/<your_team>.py`.
2. Implement `reset` / `act` / `is_done`. The `act` return type is
   `omni.isaac.core.utils.types.ArticulationAction` — `joint_positions`
   sized to `len(env_info.dof_names)`, with NaN in any dof you don't
   want to command.
3. Run: `uv run python task/run_pick_place.py --policy policies.<your_team>.MyPolicy`.

**Add a new part to pick:**
1. Drop `<name>.usdc` into `../parts/`.
2. Add a `PART_CONFIG[<name>]` entry with at least `gripper_open`,
   `gripper_close`, `place_pos`, and (if snap-mode) a `snap` dict.
3. If the part is in the scene's `init` extract, its spawn `pos/orn`
   and `pick_z` go into `part_init_poses.json` via
   `extract_part_poses.py`. Otherwise add a `pick_pos` fallback in the
   `PART_CONFIG` entry.

**Probe IK reachability over the board volume (L arm only):**
```bash
uv run python task/find_reachable_above_boards.py --part pin
```
The script loads the L descriptor (picked by `pc.OWNS_LIFT_L` /
`pc.OWNS_TORSO_L`) and probes the `L_ee_link_gripper_link` frame — there
is no R-arm path. To evaluate R reachability, mirror the script with the
R descriptor + R EE frame.

**Re-extract spawn poses after editing the scene:**
```bash
uv run python task/extract_part_poses.py
```
`pick_z` values you've tuned by hand are preserved. (Note: this script
reads `../scene_base.usd`, not the runtime `scene_init.usd` — see the
data-flow note above.)

## Debugging

- Stuck waypoint? Watch for `[STUCK] step=… no advance for N steps …`
  blocks in stdout (gated by `pc.VERBOSE_STUCK`). Lists `ik_ok`, EE FK
  pose (stage frame), pos/orn error, gripper actual vs commanded, snap
  state. Threshold `STUCK_LOG_STEPS = 100` at the top of
  `run_pick_place.py`.
- IK failed at startup? Confirm `default_q` in the active L yaml
  matches the runtime joint state — if `OWNS_LIFT_L = False` the yaml
  is `vega_1u_L_arm_description_armonly.yaml`.
- Snap not firing? Lower `rot_tol_deg`, widen `pos_tol_axes`, increase
  `search.n`, or check `target_pos` is actually where the part ends up
  in `scene_final.usd`.
- Gripper tilts after grip? Friction/grip-force/collider — not
  generally an IK issue. Try `convexHull` only for purely convex
  parts, raise `dynamic_friction` on `/World/PhysicsMaterial`, lower
  `gripper_close` for a tighter squeeze.
- Per-part hang? `PER_PART_TIMEOUT_STEPS` (default 3000) caps the per-
  part loop. The harness logs the timeout and advances; the part is
  graded on whatever state it ended in.
