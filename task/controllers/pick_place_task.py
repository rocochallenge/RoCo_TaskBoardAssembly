from typing import Optional

import numpy as np
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

from .pick_place import PickPlace_scene_bimanual


# Robot articulation root inside scene_base.usd.
ROBOT_PRIM_PATH = "/World/robotics/vega_1u_gripper"


def _gripper_paths(side: str):
    """Returns (end_effector_prim_path, gripper_joint_name) for the given arm side.

    The EE prim path points at the URDF-chain link {side}_ee (= URDF link
    L_ee/R_ee). The gripper sub-asset lives under {side}_ee_link/gripper_link
    in the USD with its own authored transform; pointing at that nested prim
    instead introduces a constant ~12 cm tool offset between Lula's URDF FK
    and the SingleManipulator's end_effector pose. Pointing at {side}_ee
    keeps both kinematic chains aligned. Per-grasp finger-tip offsets should
    be added via param_config.{L,R}_end_effector_offset (consumed by
    pick_place_base_controller).
    """
    side = side.upper()
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    ee = f"{ROBOT_PRIM_PATH}/{side}_ee"
    joint = f"{side}_gripper_joint"
    return ee, joint


def _make_manipulator(side: str, name: str, joint_opened, joint_closed) -> SingleManipulator:
    ee_path, joint_name = _gripper_paths(side)
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=[joint_name],
        joint_opened_positions=joint_opened,
        joint_closed_positions=joint_closed,
        use_mimic_joints=True,
    )
    return SingleManipulator(
        prim_path=ROBOT_PRIM_PATH,
        name=name,
        end_effector_prim_path=ee_path,
        gripper=gripper,
    )


def _bimanual_robots(joint_opened_position, joint_closed_position):
    return (
        _make_manipulator(
            side="L",
            name="vega_1u_L_arm",
            joint_opened=joint_opened_position,
            joint_closed=joint_closed_position,
        ),
        _make_manipulator(
            side="R",
            name="vega_1u_R_arm",
            joint_opened=joint_opened_position,
            joint_closed=joint_closed_position,
        ),
    )


class PickPlaceTask_scene_bimanual(PickPlace_scene_bimanual):
    """Bimanual scene-based pick-and-place task for vega_1u.

    Wraps the robot articulation + two existing pick-object prims from the
    loaded scene_base.usd; spawns no new geometry.
    """

    def __init__(
        self,
        name: str = "vega_1u_pick_place_scene_bimanual",
        L_object_prim_path: str = "/World/L_object",
        R_object_prim_path: str = "/World/R_object",
        L_target_position: Optional[np.ndarray] = None,
        R_target_position: Optional[np.ndarray] = None,
        joint_opened_position: Optional[np.ndarray] = None,
        joint_closed_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            name=name,
            L_object_prim_path=L_object_prim_path,
            R_object_prim_path=R_object_prim_path,
            L_target_position=L_target_position,
            R_target_position=R_target_position,
            offset=offset,
        )
        self._joint_opened_position = joint_opened_position
        self._joint_closed_position = joint_closed_position

    def set_robots(self):
        return _bimanual_robots(self._joint_opened_position, self._joint_closed_position)
