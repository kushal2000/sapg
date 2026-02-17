from __future__ import annotations

import time
from collections import defaultdict
from typing import Literal

import mujoco
import numpy as np
import viser
import viser.transforms as vtf

from isaacgymenvs.utils.utils import get_repo_root_dir
from misc.viser_mujoco.viser_conversions import get_body_name, is_fixed_body, merge_geoms


def create_robot_control_sliders(
    server: viser.ViserServer, viser_mj_model: ViserMJModel
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_mj_model.joint_limits.items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_mj_model.update_cfg(
                q_dict={
                    name: slider.value
                    for name, slider in zip(viser_mj_model.joint_names, slider_handles)
                },
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


class ViserMJModel:
    # ############################################################
    # Initialization
    # ############################################################
    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        root_node_name: str = "/",
    ):
        self.server = server
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.root_node_name = root_node_name
        self.visual_handles = self._create_mesh_handles(
            mesh_type="visual", visible=True
        )
        self.collision_handles = self._create_mesh_handles(
            mesh_type="collision", visible=False
        )
        self._print_model_info()

    def _print_model_info(self) -> None:
        print()
        print("Model info:")
        print(f"  Number of joints (njnt): {self.mj_model.njnt}")
        print(f"  Number of DOFs (nv): {self.mj_model.nv}")
        print(f"  Number of actuators (nu): {self.mj_model.nu}")
        print(f"  data.qpos shape: {self.mj_data.qpos.shape}")
        print(f"  data.qvel shape: {self.mj_data.qvel.shape}")
        print(f"  data.ctrl shape: {self.mj_data.ctrl.shape}")
        print(f"  Joint names: {self.joint_names}")
        print(f"  Actuator names: {self.actuator_names}")
        print(f"  Body names: {self.body_names}")
        print(f"  Joint limits: {self.joint_limits}")
        print()

    def _create_mesh_handles(
        self, mesh_type: Literal["visual", "collision"], visible: bool
    ) -> dict[int, viser.BatchedGlbHandle]:
        return self._create_mesh_handles_static(
            server=self.server,
            mj_model=self.mj_model,
            mesh_type=mesh_type,
            root_node_name=self.root_node_name,
            visible=visible,
        )

    @staticmethod
    def _create_mesh_handles_static(
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        mesh_type: Literal["visual", "collision"],
        visible: bool,
        root_node_name: str = "/",
        batch_size: int = 1,
    ) -> dict[int, viser.BatchedGlbHandle]:
        assert batch_size == 1, f"batch_size: {batch_size}, expected: 1 for now"

        # Group geoms by body
        body_geoms: defaultdict[int, list[int]] = defaultdict(list)

        for i in range(mj_model.ngeom):
            body_id = mj_model.geom_bodyid[i]
            is_collision = (
                mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0
            )

            # Add geom to body's list if it matches the type we're looking for
            if (mesh_type == "collision" and is_collision) or (
                mesh_type == "visual" and not is_collision
            ):
                body_geoms[body_id].append(i)

        handles = {}
        with server.atomic():
            for body_id, geom_indices in body_geoms.items():
                # Skip fixed world geometry
                if is_fixed_body(mj_model, body_id):
                    # We include fixed geometry to see things like the first link of robot arm
                    INCLUDE_FIXED_GEOMETRY = True
                    if not INCLUDE_FIXED_GEOMETRY:
                        continue

                # Get body name
                body_name = get_body_name(mj_model, body_id)

                # Merge geoms into a single mesh
                mesh = merge_geoms(mj_model, geom_indices)
                lod_ratio = 1000.0 / mesh.vertices.shape[0]

                # Create handle
                if root_node_name.endswith("/"):
                    root_node_name = root_node_name[:-1]
                handle = server.scene.add_batched_meshes_trimesh(
                    f"{root_node_name}/{body_name}/{mesh_type}",
                    mesh,
                    batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
                        batch_size, axis=0
                    ),
                    batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
                        batch_size, axis=0
                    ),
                    lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
                    visible=visible,
                )
                handles[body_id] = handle

        return handles

    # ############################################################
    # Updating viser
    # ############################################################
    def _update_viser(
        self,
    ) -> None:
        body_xpos = self.mj_data.xpos
        body_xmat = self.mj_data.xmat
        body_xmat = body_xmat.reshape(body_xmat.shape[0], 3, 3)
        self._update_viser_static(
            server=self.server,
            visual_handles=self.visual_handles,
            collision_handles=self.collision_handles,
            body_xpos=body_xpos[None],
            body_xmat=body_xmat[None],
        )

    @staticmethod
    def _update_viser_static(
        server: viser.ViserServer,
        visual_handles: dict[int, viser.BatchedGlbHandle],
        collision_handles: dict[int, viser.BatchedGlbHandle],
        body_xpos: np.ndarray,
        body_xmat: np.ndarray,
    ) -> None:
        batch_size, num_bodies = body_xpos.shape[:2]
        assert batch_size == 1, f"batch_size: {batch_size}, expected: 1 for now"
        assert body_xpos.shape == (batch_size, num_bodies, 3), (
            f"body_xpos.shape: {body_xpos.shape}, expected: ({batch_size}, {num_bodies}, 3)"
        )
        assert body_xmat.shape == (batch_size, num_bodies, 3, 3), (
            f"body_xmat.shape: {body_xmat.shape}, expected: ({batch_size}, {num_bodies}, 3, 3)"
        )

        with server.atomic():
            body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

            # Update both visual and collision handles symmetrically
            for handles_dict in [visual_handles, collision_handles]:
                for body_id, handle in handles_dict.items():
                    # Skip if handle is not visible
                    if not handle.visible:
                        continue

                    # Show all environments
                    handle.batched_positions = body_xpos[..., body_id, :]
                    handle.batched_wxyzs = body_xquat[..., body_id, :]

        server.flush()

    def update_cfg(
        self, q_dict: dict[str, float], actuator_dict: dict[str, float] | None = None
    ) -> None:
        assert set(q_dict.keys()).issubset(set(self.joint_names)), (
            f"q_dict.keys(): {q_dict.keys()}, expected subset of {self.joint_names} for joint positions"
        )

        # Set joint positions
        for name, value in q_dict.items():
            joint_id = self.mj_model.joint(name=name).id
            self.mj_data.qpos[joint_id] = value

        # Set actuator positions
        if actuator_dict is not None:
            assert set(actuator_dict.keys()).issubset(set(self.actuator_names)), (
                f"actuator_dict.keys(): {actuator_dict.keys()}, expected subset of {self.actuator_names} for actuator commands"
            )
            for name, value in actuator_dict.items():
                actuator_id = self.mj_model.actuator(name=name).id
                self.mj_data.ctrl[actuator_id] = value

        # Step simulation to update the state
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Update viser
        self._update_viser()

    # ############################################################
    # Getting body poses and simulation state
    # ############################################################
    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self.mj_model.body(name=body_name).id
        pos = self.mj_data.xpos[body_id]  # (3,) world position of body frame
        quat_wxyz = self.mj_data.xquat[
            body_id
        ]  # (4,) world orientation quaternion (w, x, y, z)
        return pos, quat_wxyz

    def get_sim_state(
        self,
    ) -> tuple[
        dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float], dict[str, float]
    ]:
        body_dict = {name: self.get_body_pose(name) for name in self.body_names}
        joint_dict = {
            name: self.mj_data.qpos[joint_id]
            for name, joint_id in zip(self.joint_names, self.joint_ids)
        }
        actuator_dict = {
            name: self.mj_data.ctrl[actuator_id]
            for name, actuator_id in zip(self.actuator_names, self.actuator_ids)
        }
        return body_dict, joint_dict, actuator_dict

    # ############################################################
    # Properties
    # ############################################################
    @property
    def joint_names(self) -> list[str]:
        return [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]

    @property
    def actuator_names(self) -> list[str]:
        return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

    @property
    def body_names(self) -> list[str]:
        return [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]

    @property
    def joint_ids(self) -> list[int]:
        return [self.mj_model.joint(name=name).id for name in self.joint_names]

    @property
    def actuator_ids(self) -> list[int]:
        return [self.mj_model.actuator(name=name).id for name in self.actuator_names]

    @property
    def body_ids(self) -> list[int]:
        return [self.mj_model.body(name=name).id for name in self.body_names]

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {
            name: (
                self.mj_model.jnt_range[joint_id][0],
                self.mj_model.jnt_range[joint_id][1],
            )
            for name, joint_id in zip(self.joint_names, self.joint_ids)
        }

    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return all(handle.visible for handle in self.visual_handles.values())

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        for handle in self.visual_handles.values():
            handle.visible = visible
        self._update_viser()

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return all(handle.visible for handle in self.collision_handles.values())

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        for handle in self.collision_handles.values():
            handle.visible = visible
        self._update_viser()


def main():
    iiwa_xml_path = get_repo_root_dir() / "assets/mjcf/kuka_iiwa_14/scene.xml"
    assert iiwa_xml_path.exists(), f"IIWA XML path does not exist: {iiwa_xml_path}"
    mj_model = mujoco.MjModel.from_xml_path(str(iiwa_xml_path))
    mj_data = mujoco.MjData(mj_model)
    server = viser.ViserServer()
    viser_mj_model = ViserMJModel(
        server=server,
        mj_model=mj_model,
        mj_data=mj_data,
        root_node_name="/robot",
    )

    joint_names = viser_mj_model.joint_names
    _body_names = viser_mj_model.body_names

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_mj_model
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_mj_model.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_mj_model.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_mj_model.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_mj_model.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = len(viser_mj_model.visual_handles) > 0
    show_collision_meshes_cb.visible = len(viser_mj_model.collision_handles) > 0

    # Set initial robot configuration.
    viser_mj_model.update_cfg(
        q_dict={joint_names[i]: initial_config[i] for i in range(len(joint_names))}
    )

    # Create grid.
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        body_dict, joint_dict, actuator_dict = viser_mj_model.get_sim_state()
        link_pose = body_dict["link7"]
        print(f"link_pose: {link_pose}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
