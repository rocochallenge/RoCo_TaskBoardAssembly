"""Proximity-triggered FixedJoint: snap a movable part onto a target world pose.

Construct a `SnapAttacher` with the stage + target pose, then call `.update()`
once per simulation step. When the movable mesh's world pose lands inside the
configured (pos_tol, rot_tol_deg) box of the target, the attacher teleports
the part exactly onto the target pose and authors a UsdPhysics.FixedJoint
between the part's rigid body and the parent body so the assembly holds.

Assumes the movable part has UsdPhysics.RigidBodyAPI and the parent body is
rigid or treated as a static world body.

Used by `task/run_pick_place.py` as the snap-fired success detector for
connector / pin / rod parts.
"""

import math
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


# ---------- helpers ---------------------------------------------------------

def _world_xform(stage, path):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(path)))


def _pose_error(m_cur, m_tgt):
    pos = (m_cur.ExtractTranslation() - m_tgt.ExtractTranslation()).GetLength()
    q_cur = m_cur.ExtractRotationQuat().GetNormalized()
    q_tgt = m_tgt.ExtractRotationQuat().GetNormalized()
    dot = abs(q_cur.GetReal() * q_tgt.GetReal()
              + Gf.Dot(q_cur.GetImaginary(), q_tgt.GetImaginary()))
    return pos, math.degrees(2.0 * math.acos(min(1.0, dot)))


def _quat(w, x, y, z):
    """Portable Gf.Quatd builder — Quatd(real, Vec3d(i, j, k))."""
    return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))


def _find_descendant_with(prim, predicate):
    """Return `prim` if predicate(prim), else first descendant satisfying it."""
    if prim and predicate(prim):
        return prim
    for desc in Usd.PrimRange(prim):
        if desc != prim and predicate(desc):
            return desc
    return None


# ---------- reusable physics utilities --------------------------------------
# Drop these into your own pick-and-place / gripper flows: resolve the
# rigid body under a parent Xform, build a target world matrix, teleport
# the body, and author a FixedJoint pinning it to an anchor.

def resolve_rigid_body(stage, path):
    """Path (str) of first prim with `UsdPhysics.RigidBodyAPI` at-or-under
    `path`. "" if `path` is empty/missing or no rigid body is found.
    """
    if not path:
        return ""
    root = stage.GetPrimAtPath(Sdf.Path(path))
    rb = _find_descendant_with(root,
                               lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
    return str(rb.GetPath()) if rb else ""


def resolve_anchor_body(stage, path):
    """Path (str) of first prim with `RigidBodyAPI` (preferred) or
    `CollisionAPI` (fallback) at-or-under `path`. Empty `path` is valid
    and yields "" — interpreted as world-anchored by `author_fixed_joint`.
    """
    if not path:
        return ""
    root = stage.GetPrimAtPath(Sdf.Path(path))
    pb = _find_descendant_with(root,
                               lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
    if pb is None:
        pb = _find_descendant_with(root,
                                   lambda p: p.HasAPI(UsdPhysics.CollisionAPI))
    return str(pb.GetPath()) if pb else ""


def build_world_matrix(translation, rotation):
    """`Gf.Matrix4d` from world translation and rotation.

    translation: 3-element iterable (Gf.Vec3d, tuple, list).
    rotation:    Gf.Quatd / Gf.Quatf, or 4-element iterable as wxyz.
    """
    if isinstance(rotation, (Gf.Quatd, Gf.Quatf)):
        q = Gf.Quatd(rotation)
    else:
        q = _quat(rotation[0], rotation[1], rotation[2], rotation[3])
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Rotation(q))
    m.SetTranslateOnly(
        Gf.Vec3d(translation[0], translation[1], translation[2]))
    return m


def snap_to_pose(stage, movable_path, m_tgt, set_kinematic=False):
    """Teleport the rigid body under `movable_path` to world matrix `m_tgt`.

    Authors a single transform op so the body's world transform equals
    `m_tgt`. If `set_kinematic=True`, also sets
    `physics:kinematicEnabled = True` on the body — pairing this with a
    static anchor will cause PhysX to reject a subsequent FixedJoint, so
    keep it False unless the anchor is dynamic/kinematic.

    Returns the resolved rigid-body path (str), or "" if not found.
    """
    resolved = resolve_rigid_body(stage, movable_path)
    if not resolved:
        return ""
    movable = stage.GetPrimAtPath(Sdf.Path(resolved))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    m_parent = cache.GetLocalToWorldTransform(movable.GetParent())
    m_local = m_tgt * m_parent.GetInverse()
    xf = UsdGeom.Xformable(movable)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m_local)

    if set_kinematic:
        attr = movable.GetAttribute("physics:kinematicEnabled")
        if not attr:
            attr = movable.CreateAttribute(
                "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(True)
    return resolved


def author_fixed_joint(stage, movable_path, parent_path, m_tgt, joint_path):
    """Author a `UsdPhysics.FixedJoint` pinning the movable to the anchor
    at world pose `m_tgt`.

    Resolves `movable_path` to the first RigidBodyAPI prim under it.
    Resolves `parent_path` to RigidBodyAPI (preferred) or CollisionAPI.
    Empty `parent_path` → world-anchored joint.

    Returns (joint, resolved_movable, resolved_parent). `joint` is None
    if no rigid body resolves under `movable_path`.

    PhysX requires at least one body to be dynamic. Static anchor +
    kinematic movable → "cannot create a joint between static bodies".
    """
    resolved_movable = resolve_rigid_body(stage, movable_path)
    if not resolved_movable:
        return None, "", ""
    resolved_parent = resolve_anchor_body(stage, parent_path)

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody1Rel().SetTargets([Sdf.Path(resolved_movable)])
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    if resolved_parent:
        body0 = stage.GetPrimAtPath(Sdf.Path(resolved_parent))
        m_body0 = cache.GetLocalToWorldTransform(body0)
        local0 = m_tgt * m_body0.GetInverse()
        joint.CreateBody0Rel().SetTargets([Sdf.Path(resolved_parent)])
    else:
        local0 = Gf.Matrix4d(m_tgt)

    p = local0.ExtractTranslation()
    q = local0.ExtractRotationQuat().GetNormalized()
    joint.CreateLocalPos0Attr(Gf.Vec3f(p[0], p[1], p[2]))
    joint.CreateLocalRot0Attr(Gf.Quatf(
        q.GetReal(), q.GetImaginary()[0],
        q.GetImaginary()[1], q.GetImaginary()[2]))
    joint.CreateExcludeFromArticulationAttr(True)
    return joint, resolved_movable, resolved_parent


# ---------- core ------------------------------------------------------------

class SnapAttacher:
    """Per-tick proximity snap. Call update() once per simulation step.

    When the movable prim's world pose is within (pos_tol, rot_tol_deg) of
    the target pose, snap it exactly to the target and author a FixedJoint
    between the movable and the parent body so the assembly holds.
    """

    def __init__(
        self,
        stage,
        movable_path="/World/parts/pin",
        target_pos=Gf.Vec3d(0.0, -0.13878, 1.23394),
        target_rot=None,                  # default below: (0.5, 0.5, -0.5, -0.5)
        parent_body_path="/World/right_board/task_board_color",
        pos_tol=0.005,                    # 5 mm scalar (used if pos_tol_axes is None)
        pos_tol_axes=None,                # (tx, ty, tz) in WORLD frame; overrides pos_tol
        rot_tol_deg=5.0,                  # set < 0 to skip the rotation gate
        joint_path="/World/_snap_joint",
        debug=False,
        debug_every=30,
        set_kinematic_on_snap=True,
        mesh_path=None,                   # primary Mesh prim used for the
                                          # proximity comparison (what the
                                          # user sees in the viewport). If
                                          # None, the first Mesh descendant
                                          # of movable_path is auto-picked.
        author_joint_on_snap=True,        # if False, the proximity gate
                                          # still fires (self.attached ->
                                          # True) but no FixedJoint is
                                          # created. Use when the snap is
                                          # just a phase trigger and the
                                          # part should remain free.
        connect_pos=None,                 # ABSOLUTE world translation of the
                                          # MESH's final pose (consistent with
                                          # target_pos's mesh-frame semantic
                                          # after Design B). snap_attach
                                          # converts this to the body-frame
                                          # joint anchor internally using
                                          # the cached mesh_local_in_body.
                                          # None → fall back to
                                          # connect_offset_* or, if those
                                          # are also None, to the body's
                                          # current world pose on fire.
        connect_rot=None,                 # ABSOLUTE world rotation (wxyz)
                                          # of the MESH's final pose. None →
                                          # uses the mesh's CURRENT rotation
                                          # at fire time (the delivered
                                          # orientation), NOT target_rot.
                                          # Anchoring at the current rotation
                                          # eliminates the rotational yank
                                          # PhysX would otherwise apply up to
                                          # the rot-gate tolerance.
        connect_offset_pos=None,          # RELATIVE translation, applied in
                                          # target's local frame, that maps
                                          # target_pos → mesh final pose.
                                          # Body-frame conversion is done
                                          # internally. Honored only when
                                          # connect_pos is None.
        connect_offset_rot=None,          # RELATIVE rotation (wxyz), applied
                                          # in target's local frame on top
                                          # of connect_offset_pos. Same
                                          # gating as above.
    ):
        if target_rot is None:
            target_rot = _quat(0.5, 0.5, -0.5, -0.5)
        self.stage = stage
        self.movable_path = movable_path
        self.target_pos = Gf.Vec3d(target_pos)
        self.target_rot = Gf.Quatd(target_rot).GetNormalized()
        self.parent_body_path = parent_body_path
        self.pos_tol = float(pos_tol)
        self.pos_tol_axes = (Gf.Vec3d(float(pos_tol_axes[0]),
                                      float(pos_tol_axes[1]),
                                      float(pos_tol_axes[2]))
                             if pos_tol_axes is not None else None)
        self.rot_tol_deg = float(rot_tol_deg)
        self.joint_path = joint_path
        self.debug = bool(debug)
        self.debug_every = max(1, int(debug_every))
        self.set_kinematic_on_snap = bool(set_kinematic_on_snap)
        self.mesh_path = mesh_path       # explicit override or None for auto-pick
        self.author_joint_on_snap = bool(author_joint_on_snap)
        self.connect_pos = (Gf.Vec3d(connect_pos)
                            if connect_pos is not None else None)
        self.connect_rot = (Gf.Quatd(connect_rot).GetNormalized()
                            if connect_rot is not None else None)
        self.connect_offset_pos = (Gf.Vec3d(connect_offset_pos)
                                   if connect_offset_pos is not None else None)
        self.connect_offset_rot = (Gf.Quatd(connect_offset_rot).GetNormalized()
                                   if connect_offset_rot is not None else None)
        self.attached = False
        self._tick = 0
        self._resolved_movable = None    # set lazily on first update()
        self._resolved_parent = None     # set lazily on first update()
        self._resolved_mesh = None       # first Mesh under movable_path; the
                                         # proximity-check reference frame.
        # Post-fire telemetry: baselines captured at the moment the joint
        # is authored so [snap.post] can report yank/drift as deltas, not
        # raw coordinates. _post_tick counts physics steps since fire so
        # "tick=1" lines up with the very first frame after authoring.
        self._post_baseline_mesh = None
        self._post_baseline_body = None
        self._post_tick = 0

    # --- internals --------------------------------------------------------

    def _resolve_paths(self):
        if self._resolved_movable is not None:
            return
        self._resolved_movable = resolve_rigid_body(
            self.stage, self.movable_path)
        self._resolved_parent = resolve_anchor_body(
            self.stage, self.parent_body_path)
        # Mesh resolution: explicit mesh_path wins; else walk under
        # movable_path and pick the first Mesh prim. Falls back to the
        # rigid body itself if no mesh is found (legacy Design-A behavior).
        if self.mesh_path:
            self._resolved_mesh = self.mesh_path
        else:
            root = self.stage.GetPrimAtPath(Sdf.Path(self.movable_path))
            m = _find_descendant_with(root,
                                      lambda p: p.GetTypeName() == "Mesh")
            self._resolved_mesh = str(m.GetPath()) if m else self._resolved_movable
        # Cache mesh's local-to-body transform. Used to convert connect_pos
        # / connect_rot (specified in MESH frame, consistent with target_pos)
        # into the body-frame anchor that author_fixed_joint actually wants.
        # USD convention: M_mesh_world = mesh_local_in_body * M_body_world
        # → mesh_local_in_body = M_mesh_world * inv(M_body_world).
        if self._resolved_mesh and self._resolved_mesh != self._resolved_movable:
            m_body_world = _world_xform(self.stage, self._resolved_movable)
            m_mesh_world = _world_xform(self.stage, self._resolved_mesh)
            self._mesh_local_in_body = m_mesh_world * m_body_world.GetInverse()
        else:
            self._mesh_local_in_body = Gf.Matrix4d(1.0)  # identity fallback
        if getattr(self, "debug", False):
            print(f"[snap] resolved movable: '{self.movable_path}' → '{self._resolved_movable}'")
            print(f"[snap] resolved parent : '{self.parent_body_path}' → '{self._resolved_parent}'")
            print(f"[snap] resolved mesh   : '{self.mesh_path or '<auto>'}' → '{self._resolved_mesh}'")

    def _target_world_matrix(self):
        return build_world_matrix(self.target_pos, self.target_rot)

    def _snap_pose(self, m_tgt):
        snap_to_pose(self.stage, self._resolved_movable, m_tgt,
                     set_kinematic=self.set_kinematic_on_snap)

    def _author_fixed_joint(self, m_tgt):
        author_fixed_joint(self.stage, self._resolved_movable,
                           self._resolved_parent, m_tgt, self.joint_path)

    def _log(self, pos_err, rot_err, pos_ok, rot_ok, m_cur, dp=None):
        cp = m_cur.ExtractTranslation()
        cq = m_cur.ExtractRotationQuat().GetNormalized()
        tq = self.target_rot
        if dp is not None:
            ax = self.pos_tol_axes
            pos_str = (f"dx={dp[0]*1000:+6.2f}[<{ax[0]*1000:.1f}] "
                       f"dy={dp[1]*1000:+6.2f}[<{ax[1]*1000:.1f}] "
                       f"dz={dp[2]*1000:+6.2f}[<{ax[2]*1000:.1f}]mm world "
                       f"[{'PASS' if pos_ok else 'fail'}]")
        else:
            pos_str = (f"pos_err={pos_err*1000:7.2f}mm "
                       f"[{'PASS' if pos_ok else 'fail'}<{self.pos_tol*1000:.1f}]")
        if self.rot_tol_deg < 0:
            rot_str = f"rot_err={rot_err:6.2f}deg [SKIP]"
        else:
            rot_str = (f"rot_err={rot_err:6.2f}deg "
                       f"[{'PASS' if rot_ok else 'fail'}<{self.rot_tol_deg:.1f}]")
        print(
            f"[snap.debug] tick={self._tick:5d}  {pos_str}  {rot_str}  "
            f"cur=({cp[0]:+.4f},{cp[1]:+.4f},{cp[2]:+.4f}) "
            f"wxyz=({cq.GetReal():+.3f},{cq.GetImaginary()[0]:+.3f},{cq.GetImaginary()[1]:+.3f},{cq.GetImaginary()[2]:+.3f})  "
            f"tgt=({self.target_pos[0]:+.4f},{self.target_pos[1]:+.4f},{self.target_pos[2]:+.4f}) "
            f"wxyz=({tq.GetReal():+.3f},{tq.GetImaginary()[0]:+.3f},{tq.GetImaginary()[1]:+.3f},{tq.GetImaginary()[2]:+.3f})"
        )

    # --- public -----------------------------------------------------------

    def update(self):
        if self.attached:
            # Post-fire telemetry: watch what PhysX does to the bolt after
            # the joint is authored. Logs every tick for the first 5 frames
            # (the yank window — joint constraint solver kicks in here),
            # then thins to `debug_every` for subsequent drift.
            #
            #   yank  = mesh - mesh_at_fire  (PhysX's correction to satisfy
            #           the joint; if the body was offset from the anchor
            #           when authored, the solver yanks it into alignment)
            #   d_tgt = mesh - target_pos    (absolute residual vs the pose
            #           the snap was gated on; non-zero post-yank means the
            #           joint anchor disagrees with target_pos)
            if (self.debug
                    and self._resolved_mesh
                    and self._resolved_movable):
                self._post_tick += 1
                if (self._post_tick <= 5
                        or self._post_tick % self.debug_every == 0):
                    m_mesh = _world_xform(self.stage, self._resolved_mesh)
                    m_body = _world_xform(self.stage, self._resolved_movable)
                    m_tgt = self._target_world_matrix()
                    mp = m_mesh.ExtractTranslation()
                    bp = m_body.ExtractTranslation()
                    tp = m_tgt.ExtractTranslation()
                    if self._post_baseline_mesh is not None:
                        base_p = self._post_baseline_mesh.ExtractTranslation()
                        yk = mp - base_p
                    else:
                        yk = Gf.Vec3d(0.0, 0.0, 0.0)
                    dt = mp - tp
                    _, rot_err = _pose_error(m_mesh, m_tgt)
                    print(
                        f"[snap.post] tick={self._post_tick:4d}  "
                        f"mesh=({mp[0]:+.4f},{mp[1]:+.4f},{mp[2]:+.4f})  "
                        f"body=({bp[0]:+.4f},{bp[1]:+.4f},{bp[2]:+.4f})  "
                        f"yank=({yk[0]*1000:+6.2f},{yk[1]*1000:+6.2f},{yk[2]*1000:+6.2f})mm  "
                        f"d_tgt=({dt[0]*1000:+6.2f},{dt[1]*1000:+6.2f},{dt[2]*1000:+6.2f})mm  "
                        f"rot_err={rot_err:5.2f}deg",
                        flush=True,
                    )
            return True
        self._resolve_paths()
        if not self._resolved_movable:
            if self.debug:
                print(f"[snap] no RigidBodyAPI under {self.movable_path}; skipping")
            return False
        # Design B option (b): compare the visible MESH's world pose against
        # target_pos/target_rot. This is what you see in the viewport and
        # what extract_part_poses.py reports as `pos`. The rigid body may
        # sit at a different world pose because of a body↔mesh local xform.
        m_cur = _world_xform(self.stage, self._resolved_mesh)
        m_tgt = self._target_world_matrix()
        pos_err, rot_err = _pose_error(m_cur, m_tgt)

        # Position gate: per-axis in WORLD frame if pos_tol_axes set, else
        # scalar Euclidean.
        if self.pos_tol_axes is not None:
            dp = m_cur.ExtractTranslation() - m_tgt.ExtractTranslation()
            pos_ok = (abs(dp[0]) < self.pos_tol_axes[0]
                      and abs(dp[1]) < self.pos_tol_axes[1]
                      and abs(dp[2]) < self.pos_tol_axes[2])
        else:
            dp = None
            pos_ok = pos_err < self.pos_tol

        # Rotation gate: rot_tol_deg < 0 disables it (axis-symmetric movables).
        rot_ok = True if self.rot_tol_deg < 0 else (rot_err < self.rot_tol_deg)

        self._tick += 1
        if self.debug:
            if self._tick % self.debug_every == 0:
                self._log(pos_err, rot_err, pos_ok, rot_ok, m_cur, dp)

        if pos_ok and rot_ok:
            # Option (b): no teleport. The body stays where the gripper put
            # it; the FixedJoint anchors at one of three places, in priority
            # order:
            #   1. ABSOLUTE: (connect_pos, connect_rot) supplied as world
            #      pose. connect_rot defaults to target_rot if absent.
            #   2. RELATIVE: m_joint = connect_offset * m_tgt — target_pos
            #      with a local-frame offset applied. Defaults to identity
            #      offset for the dimension not supplied.
            #   3. DEFAULT: body's current world pose (no decoupling) — the
            #      mesh lands within pos_tol_axes of target_pos and is
            #      frozen there.
            if self.author_joint_on_snap:
                m_joint = self._joint_anchor_matrix(m_tgt, m_cur)
                # Teleport body to the joint anchor before authoring the
                # joint so PhysX sees matching transforms (no "disjointed
                # body transforms" warning, no rotational yank up to the
                # rot-gate tolerance).
                self._snap_pose(m_joint)
                self._author_fixed_joint(m_joint)
            self.attached = True
            # Baseline for post-fire telemetry. Captured from the same
            # XformCache the gate evaluated against so yank=0 on tick 0.
            self._post_baseline_mesh = m_cur
            self._post_baseline_body = _world_xform(
                self.stage, self._resolved_movable)
            self._post_tick = 0
            tag = "attached + joint" if self.author_joint_on_snap else "gate fired (no joint)"
            print(f"[snap] {tag}  pos_err={pos_err*1000:.2f}mm  rot_err={rot_err:.2f}deg")
            return True
        return False

    def _joint_anchor_matrix(self, m_tgt, m_cur):
        """World matrix to pass to author_fixed_joint as the BODY's target.

        connect_pos / connect_rot / connect_offset_* are specified in MESH
        frame (consistent with target_pos / target_rot after the Design-B
        switch). This method computes the mesh-frame world target first,
        then converts to the body-frame target the joint actually wants:

            M_body_target = inv(mesh_local_in_body) * M_mesh_target

        Priority:
          1. Absolute: connect_pos / connect_rot (mesh-frame world pose).
             When connect_rot is None, the rotation defaults to the mesh's
             CURRENT rotation (`m_cur`), not target_rot — anchoring at the
             delivered orientation keeps PhysX from yanking the body by the
             rot-gate's tolerance, which would otherwise sweep the mesh
             through a big arc when the body origin sits far from the mesh.
          2. Relative: connect_offset_* composed with m_tgt (= target_pos).
             Same current-rotation default when connect_offset_rot is None.
          3. Default: no override → joint anchors at the body's CURRENT
             world pose (option-b default; no mesh→body conversion needed
             because the body's current pose is already in body frame).
        """
        if self.connect_pos is not None or self.connect_rot is not None:
            cp = self.connect_pos if self.connect_pos is not None \
                 else self.target_pos
            cr = (self.connect_rot if self.connect_rot is not None
                  else m_cur.ExtractRotationQuat().GetNormalized())
            m_mesh_target = build_world_matrix(cp, cr)
            return self._mesh_local_in_body.GetInverse() * m_mesh_target
        if self.connect_offset_pos is not None or self.connect_offset_rot is not None:
            op = self.connect_offset_pos if self.connect_offset_pos is not None \
                 else Gf.Vec3d(0.0, 0.0, 0.0)
            if self.connect_offset_rot is not None:
                orot = self.connect_offset_rot
                m_offset = build_world_matrix(op, orot)
                m_mesh_target = m_offset * m_tgt
            else:
                # No rotation offset → keep mesh's current rotation, only
                # translate by op in target's local frame. Same yank-avoidance
                # rationale as the connect_pos/connect_rot branch above.
                m_translate = build_world_matrix(op, _quat(1.0, 0.0, 0.0, 0.0))
                m_mesh_pos_target = m_translate * m_tgt
                m_mesh_target = Gf.Matrix4d(m_cur)
                m_mesh_target.SetTranslateOnly(
                    m_mesh_pos_target.ExtractTranslation())
            return self._mesh_local_in_body.GetInverse() * m_mesh_target
        return _world_xform(self.stage, self._resolved_movable)
