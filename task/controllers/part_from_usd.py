# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Sequence

import numpy as np
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.api.materials.visual_material import VisualMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from pxr import Gf, UsdGeom



class VisualPart(SingleGeometryPrim):
    """High level wrapper to create/encapsulate a visual part

    .. note::

        Visual cuboids (part shape) have no collisions (Collider API) or rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "visual_part".
        part_usd_path (Optional[str], optional): path of the usd file for import the part
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.api.objects import VisualPart
        >>> import numpy as np
        >>>
        >>> # create a red visual part at the given path
        >>> prim = VisualPart(prim_path="/World/Xform/part", color=np.array([1.0, 0.0, 0.0]))
        >>> prim
        <isaacsim.core.api.objects.cuboid.VisualPart object at 0x7f12e756fa00>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "visual_part",
        part_usd_path: Optional[str] = None,
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        visual_material: Optional[VisualMaterial] = None,
    ) -> None:
        if is_prim_path_valid(prim_path):
            prim = get_prim_at_path(prim_path)
            if not prim.IsA(UsdGeom.Xformable):
                raise Exception("The prim at path {} cannot be parsed as a part object".format(prim_path))
            visible = prim.GetAttribute("visibility").Get() != "invisible"
        ## if prim path not valid, we need to import it
        else:
            if part_usd_path is None:
                raise ValueError("Part_usd_path must be provided to create a new VisualPart.")
            add_reference_to_stage(
                usd_path=part_usd_path,
                prim_path=prim_path # Use the input prim_path for consistency
            ) 
            if visible is None:
                visible = True
            if visual_material is None:
                if color is None:
                    color = np.array([0.5, 0.5, 0.5])
                visual_prim_path = find_unique_string_name(
                    initial_name="/World/Looks/visual_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
                visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
            
        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            collision=False,
        )
        self._prim = get_prim_at_path(prim_path)
        if visual_material is not None:
            VisualPart.apply_visual_material(self, visual_material)
        return

    def get_dimensions_and_extent(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the world-scaled dimensions of the part and returns 
        the raw local min/max coordinates from the extent attribute.

        Returns:
            tuple: (final_dimensions, min_coords, max_coords)
                - final_dimensions (np.ndarray, shape=(3,)): World-scaled LWH.
                - min_coords (np.ndarray, shape=(3,)): Raw local minimum coordinates.
                - max_coords (np.ndarray, shape=(3,)): Raw local maximum coordinates.
        """
        boundable = UsdGeom.Boundable(self._prim) 
        extent_value = boundable.GetExtentAttr().Get()
        
        if extent_value is None or extent_value.size == 0:
            zero_array = np.zeros(3, dtype=np.float32)
            return zero_array, zero_array, zero_array

        min_coords = extent_value[0]
        max_coords = extent_value[1]
        
        dimensions_span = max_coords - min_coords

        scale_factor = self.get_local_scale() 
        final_dimensions = dimensions_span * scale_factor
        
        return final_dimensions, min_coords, max_coords

class FixedPart(VisualPart):
    """High level wrapper to create/encapsulate a fixed part

    .. note::

        Fixed parts (part shape) have collisions (Collider API) but no rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_part".
        part_usd_path (Optional[str], optional): path of the usd file for import the part
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.api.objects import FixedPart
        >>> import numpy as np
        >>>
        >>> # create a red fixed part at the given path
        >>> prim = FixedPart(prim_path="/World/Xform/part", color=np.array([1.0, 0.0, 0.0]))
        >>> prim
        <isaacsim.core.api.objects.cuboid.FixedPart object at 0x7f7b4d91da80>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "fixed_part",
        part_usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        set_offsets = False
        if not is_prim_path_valid(prim_path):
            # set default values if no physics material given
            if physics_material is None:
                static_friction = 0.2
                dynamic_friction = 1.0
                restitution = 0.0
                physics_material_path = find_unique_string_name(
                    initial_name="/World/Physics_Materials/physics_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                physics_material = PhysicsMaterial(
                    prim_path=physics_material_path,
                    dynamic_friction=dynamic_friction,
                    static_friction=static_friction,
                    restitution=restitution,
                )
            set_offsets = True
        VisualPart.__init__(
            self,
            prim_path=prim_path,
            name=name,
            part_usd_path=part_usd_path,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            visual_material=visual_material,
        )
        SingleGeometryPrim.set_collision_enabled(self, True)
        FixedPart.set_collision_approximation(self,"convexDecomposition")
        if physics_material is not None:
            FixedPart.apply_physics_material(self, physics_material)
        if set_offsets:
            ## set_offset values inhereted from the cube, default: rest_offset 0, contact_offset 0.1, 
            ##                              torsional_patch_radius 1, min_torsional_patch_radisu 0.8
            FixedPart.set_rest_offset(self, 0)
            FixedPart.set_contact_offset(self, 0.1)
            FixedPart.set_torsional_patch_radius(self, 1.0)
            FixedPart.set_min_torsional_patch_radius(self, 0.8)
        return


class DynamicPart(SingleRigidPrim, FixedPart):
    """High level wrapper to create/encapsulate a dynamic part

    .. note::

        Dynamic parts (part shape) have collisions (Collider API) and rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_part".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.
        mass (Optional[float], optional): mass in kg. Defaults to None.
        density (Optional[float], optional): density. Defaults to None.
        linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
        angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.api.objects import DynamicPart
        >>> import numpy as np
        >>>
        >>> # create a red dynamic part of mass 1kg at the given path
        >>> prim = DynamicPart(prim_path="/World/Xform/part", color=np.array([1.0, 0.0, 0.0]), mass=1.0)
        >>> prim
        <isaacsim.core.api.objects.cuboid.DynamicPart object at 0x7ff14c04d990>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "dynamic_part",
        part_usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            if mass is None:
                mass = 0.02
        FixedPart.__init__(
            self,
            prim_path=prim_path,
            name=name,
            part_usd_path=part_usd_path,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            visual_material=visual_material,
            physics_material=physics_material,
        )
        SingleRigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            density=density,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
        self._prim = get_prim_at_path(prim_path)
        return