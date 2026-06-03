# Parts Inventory

Source of truth: `task/param_config.py` — `part_order`, `PART_CONFIG`.
This file is a participant-facing summary and is kept in sync by hand, so
if it disagrees with `param_config.py`, trust the code.

Scene loaded at runtime: **`scene_init.usd`** (`SCENE_USD = "../scene_init.usd"`).
All parts are already authored in that file at their hand-placed positions,
so `import_missing_parts` spawns nothing — it just reuses the prims in the
loaded stage.

## What changed in this simulation

- **`part_board` removed.** The source-side board (`left_board` /
  `part_board_color`) was deleted from the scene. The single remaining board
  is the target board, now named **`task_board`** (renamed from `right_board`).
  Its prim path is `/World/task_board/task_board_color`.
- **Parts removed from the task (12 → 9):**
  - `battery_size7` — deleted entirely (gone from `PART_CONFIG`, `part_order`,
    and not picked or scored).
  - `usb_c`, `ethernet` — removed from `part_order` and `PART_CONFIG` (no
    longer picked or scored). They still exist as inert prims in
    `scene_init.usd`, but the runner ignores them.

## The 9 parts (order of `part_order`)

The harness walks `part_order` top-to-bottom. Each row is one episode of
`Policy.act` / `Policy.is_done`.

| # | name            | source USD                 | release | snap target slot (parent body)                  | scoring                             |
|---|-----------------|----------------------------|---------|-------------------------------------------------|-------------------------------------|
| 1 | `gear_20teeth`  | `parts/gear_20teeth.usdc`  | open    | —                                               | `grade_pos` (mesh translate, ≤10 mm)|
| 2 | `gear_60teeth`  | `parts/gear_60teeth.usdc`  | open    | —                                               | `grade_pos` (mesh translate, ≤10 mm)|
| 3 | `rod_16mm` ★    | `parts/rod_16mm.usdc`      | snap    | `task_board_color/root_001/_188_028` (w/ bolt)  | snap fired                          |
| 4 | `bolt_8mm` ★    | `parts/bolt_8mm.usdc`      | snap    | `task_board_color/root_001/_188_028` (w/ rod)   | snap fired                          |
| 5 | `usb_a`         | `parts/usb_a.usdc`         | snap    | `task_board_color` (board root)                 | snap fired                          |
| 6 | `hdmi`          | `parts/hdmi.usdc`          | snap    | `task_board_color` (board root)                 | snap fired                          |
| 7 | `pin`           | `parts/pin.usdc`           | snap    | `task_board_color/_188_001`                     | snap fired                          |
| 8 | `battery_size1` | `parts/battery_size1.usdc` | open    | —                                               | `grade_pos` (AABB midpoint, ≤10 mm) |
| 9 | `battery_size5` | `parts/battery_size5.usdc` | open    | —                                               | `grade_pos` (AABB midpoint, ≤10 mm) |

★ `PERMANENT_PARTS = {bolt_8mm, rod_16mm}` is still **defined** in
`param_config.py` but is **no longer consumed by the runner** (vestigial). In
this scene `rod_16mm` and `bolt_8mm` sit at ordinary pick positions in
`scene_init.usd` and are picked and snapped like every other snap part.

`part_order` is being actively tuned — it currently holds these 9 names in the
order above. Trimming it for a single-part test is fine; just keep the trailing
comma so a one-element `part_order` stays a tuple
(`("gear_20teeth",)`, not `("gear_20teeth")`).

## What's in `scene_init.usd`

Every part is pre-resident — the runner does **not** spawn anything at startup
(`import_missing_parts` finds all prims already present and skips them).

Prims under `/World/parts` in the loaded scene:

- Picked & scored (in `part_order`): `gear_20teeth`, `gear_60teeth`,
  `rod_16mm`, `bolt_8mm`, `usb_a`, `hdmi`, `pin`, `battery_size1`,
  `battery_size5`.
- Present but **not** in `part_order` (inert scenery, not scored): `usb_c`,
  `ethernet`.

`pick_pos` for each part is pinned in `PART_CONFIG` (kept in sync with the
part's mesh world XY in `scene_init.usd`).

## Files in `parts/` that are NOT loaded into the active task

| file                                | role / status                                                                |
|-------------------------------------|------------------------------------------------------------------------------|
| `parts/task_board*.usdc`            | target board — referenced by the scene (active)                              |
| `parts/part_board*.usdc`            | source-side board — **no longer in the scene** (the `part_board` was removed); files remain but are unreferenced |
| `parts/bolt_rack.usdc`              | small rack the bolts sit in                                                  |
| `parts/usb_c.usdc`, `ethernet.usdc` | still referenced by `scene_init.usd` as inert prims, but not in `part_order` |
| `parts/battery_size7.usdc`          | removed from the task; file remains but is unused                            |

## Snap-mode targets (`release_mode == "snap"`)

The success detector fires once the part's mesh lands within the per-axis
position tolerance AND within the rotation tolerance of the target pose. On
fire, the harness teleports the mesh to `connect_pos` and authors a
`UsdPhysics.FixedJoint` anchoring it to `parent_body_path`. The part is then
rigidly pinned for the rest of the run.

All values below come from `PART_CONFIG[name]["snap"]`. Coordinates are
stage-frame metres; quaternions are wxyz. Position tolerances are world-frame
per-axis; rotation tolerance is degrees (`OFF` / `-1` = axis-symmetric,
rotation gate disabled).

| part       | target_pos (m)               | target_rot (wxyz)        | pos_tol (mm) x,y,z | rot_tol | timeout | search grid |
|------------|------------------------------|--------------------------|--------------------|---------|---------|-------------|
| `rod_16mm` | `(0.24681, 0.16982, 1.057)`  | `(0.7071, 0.7071, 0, 0)` | `2.5, 2.5, 5`      | OFF     | 300     | 5×5 @ 2 mm  |
| `bolt_8mm` | `(0.21531, 0.13135, 1.06)`   | `(0.6892, 0.6892, 0, 0)` | `2, 2, 5`          | OFF     | 300     | 6×6 @ 6 mm  |
| `usb_a`    | `(0.23768, 0.05143, 1.05)`   | `(-0.5, -0.5, 0.5, 0.5)` | `3, 3, 5`          | 10°     | 300     | 5×5 @ 2 mm  |
| `hdmi`     | `(0.27469, 0.049,   1.055)`  | `(0.5, 0.5, 0.5, 0.5)`   | `2, 2, 5`          | 10°     | 300     | 5×5 @ 2 mm  |
| `pin`      | `(0.24945, 0.00616, 1.065)`  | `(0.7071, 0.7071, 0, 0)` | `2, 2, 10`         | OFF     | 300     | 5×5 @ 2 mm  |

Parent body for each (the prim the FixedJoint anchors to):

| part       | parent_body_path                                                        |
|------------|-------------------------------------------------------------------------|
| `rod_16mm` | `/World/task_board/task_board_color/root_001/_188_028` (shared w/ bolt) |
| `bolt_8mm` | `/World/task_board/task_board_color/root_001/_188_028` (shared w/ rod)  |
| `usb_a`    | `/World/task_board/task_board_color` (board root)                       |
| `hdmi`     | `/World/task_board/task_board_color` (board root)                       |
| `pin`      | `/World/task_board/task_board_color/_188_001`                           |

Notes:
- **Board renamed.** All parent paths moved from `/World/right_board/...` to
  `/World/task_board/...` when the board was renamed.
- **Board-root anchors.** `usb_a` and `hdmi` anchor at the board root because
  their original per-connector sockets (`_188_032`, and `_188_021` for the
  now-removed `ethernet`) were lost in the board restructure. The task board is
  static collision geometry (no rigid body), so a board-root anchor pins the
  connector at `target_pos` exactly the same as a per-socket anchor would.
- **Search grid**: center-out `n × n` XY sweep of `extent_xy` around
  `target_pos` while waiting for the snap, giving a few mm of margin.
- **`rot_tol = OFF` (`-1`)**: rotation gate disabled (pin / rod / bolt are
  axis-symmetric about the insertion axis).
- **`connect_pos`** (not shown) is where the mesh is teleported on fire;
  typically `target_pos` shifted down by the insertion depth (e.g. HDMI
  inserts ~16 mm).

## Open-mode targets (`release_mode == "open"`)

Pass iff the part's settled position (after release + gravity/contact) is
within `GRADE_POS_TOL_M` (10 mm) of `grade_pos`. For gears, `grade_pos` is the
mesh translation from a known-good run. For batteries, it's the AABB midpoint
(`"grade_use_aabb": True`).

| part            | grade_pos (m)                  | measure        | tol   |
|-----------------|--------------------------------|----------------|-------|
| `gear_20teeth`  | `(0.19723, -0.09599, 1.03377)` | mesh translate | 10 mm |
| `gear_60teeth`  | `(0.20854, -0.05615, 1.03158)` | mesh translate | 10 mm |
| `battery_size1` | `(0.12914,  0.15615, 1.04601)` | AABB midpoint  | 10 mm |
| `battery_size5` | `(0.08701,  0.16742, 1.03578)` | AABB midpoint  | 10 mm |

## Scoring summary

- **snap** parts: pass iff `snap_attach` fires (mesh within `snap.pos_tol_axes`
  of `snap.target_pos` and within `snap.rot_tol_deg` of `snap.target_rot`,
  per-axis). The harness authors a `FixedJoint` on fire; the part is then
  rigidly anchored for the rest of the run.
- **open** parts: pass iff the part's final settled position is within
  `GRADE_POS_TOL_M` (10 mm) of `grade_pos`. Position-only — no orientation
  check (axis-symmetric).

Aggregate score per run: `pass / total` across the entries in `part_order`
(currently 9).
