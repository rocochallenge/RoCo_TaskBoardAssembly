# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scene-based bimanual pick-and-place task: the robot, ground, table, and the
# two pick objects are ALREADY present in the loaded USDA (scene_base.usd).
# This task class only wraps existing prims (no ground/table/object creation)
# so observations + poses are available to the per-arm PickPlaceControllers.

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.grippers import ParallelGripper


class PickPlace_scene_bimanual(ABC, BaseTask):
    """Bimanual pick-and-place over a pre-built scene.

    The robot and the two pick objects must already exist in the stage at the
    prim paths supplied to __init__. Sub-classes implement set_robots() to
    return two SingleManipulator wrappers on the existing robot articulation.
    """

    def __init__(
        self,
        name: str,
        L_object_prim_path: str,
        R_object_prim_path: str,
        L_target_position: Optional[np.ndarray] = None,
        R_target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot_L = None
        self._robot_R = None
        self._object_L = None
        self._object_R = None
        self._L_object_prim_path = L_object_prim_path
        self._R_object_prim_path = R_object_prim_path
        self._L_target_position = (np.array([0.5, 0.30, 0.10]) if L_target_position is None else np.asarray(L_target_position)) + self._offset
        self._R_target_position = (np.array([0.5, -0.30, 0.10]) if R_target_position is None else np.asarray(R_target_position)) + self._offset

    @abstractmethod
    def set_robots(self):
        """Return (manipulator_L, manipulator_R) bound to the existing /vega_1u articulation."""
        raise NotImplementedError

    def set_up_scene(self, scene: Scene) -> None:
        # No ground plane / table / object creation: the scene USDA provides them.
        super().set_up_scene(scene)

        if not is_prim_path_valid(self._L_object_prim_path):
            raise ValueError(f"L_object_prim_path is not in the stage: {self._L_object_prim_path}")
        if not is_prim_path_valid(self._R_object_prim_path):
            raise ValueError(f"R_object_prim_path is not in the stage: {self._R_object_prim_path}")

        L_name = find_unique_string_name(initial_name="object_L", is_unique_fn=lambda x: not scene.object_exists(x))
        R_name = find_unique_string_name(initial_name="object_R", is_unique_fn=lambda x: not scene.object_exists(x))
        self._object_L = scene.add(SingleRigidPrim(prim_path=self._L_object_prim_path, name=L_name))
        self._object_R = scene.add(SingleRigidPrim(prim_path=self._R_object_prim_path, name=R_name))
        self._task_objects[self._object_L.name] = self._object_L
        self._task_objects[self._object_R.name] = self._object_R

        self._robot_L, self._robot_R = self.set_robots()
        scene.add(self._robot_L)
        scene.add(self._robot_R)
        self._task_objects[self._robot_L.name] = self._robot_L
        self._task_objects[self._robot_R.name] = self._robot_R

        self._move_task_objects_to_their_frame()

    def get_params(self) -> dict:
        L_pos, L_ori = self._object_L.get_local_pose()
        R_pos, R_ori = self._object_R.get_local_pose()
        return {
            "L_object_position":    {"value": L_pos, "modifiable": True},
            "L_object_orientation": {"value": L_ori, "modifiable": True},
            "L_target_position":    {"value": self._L_target_position, "modifiable": True},
            "L_object_name":        {"value": self._object_L.name, "modifiable": False},
            "L_robot_name":         {"value": self._robot_L.name, "modifiable": False},
            "R_object_position":    {"value": R_pos, "modifiable": True},
            "R_object_orientation": {"value": R_ori, "modifiable": True},
            "R_target_position":    {"value": self._R_target_position, "modifiable": True},
            "R_object_name":        {"value": self._object_R.name, "modifiable": False},
            "R_robot_name":         {"value": self._robot_R.name, "modifiable": False},
        }

    def get_observations(self) -> dict:
        L_pos, L_ori = self._object_L.get_local_pose()
        R_pos, R_ori = self._object_R.get_local_pose()
        L_ee, _ = self._robot_L.end_effector.get_local_pose()
        R_ee, _ = self._robot_R.end_effector.get_local_pose()
        joints_state = self._robot_L.get_joints_state()
        return {
            self._object_L.name: {"position": L_pos, "orientation": L_ori, "target_position": self._L_target_position},
            self._object_R.name: {"position": R_pos, "orientation": R_ori, "target_position": self._R_target_position},
            self._robot_L.name: {"joint_positions": joints_state.positions, "end_effector_position": L_ee},
            self._robot_R.name: {"joint_positions": joints_state.positions, "end_effector_position": R_ee},
        }

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        return

    def post_reset(self) -> None:
        for robot in (self._robot_L, self._robot_R):
            if isinstance(robot.gripper, ParallelGripper):
                robot.gripper.set_joint_positions(robot.gripper.joint_opened_positions)

    def calculate_metrics(self) -> dict:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError
