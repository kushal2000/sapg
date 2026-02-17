import time
from collections import defaultdict
from typing import Literal

import mujoco
import numpy as np
import viser
import viser.transforms as vtf

from eval.sim2sim.mujoco_sim.mujoco_sim import MujocoSim, MujocoSimConfig
from misc.viser_mujoco.viser_conversions import get_body_name, is_fixed_body, merge_geoms


class ViserMujocoSim:
    def __init__(self, server: viser.ViserServer, sim: MujocoSim):
        self.server = server
        self.sim = sim
        self.visual_handles = self._create_mesh_handles(
            mesh_type="visual", visible=True
        )
        self.collision_handles = self._create_mesh_handles(
            mesh_type="collision", visible=False
        )

    def _create_mesh_handles(
        self, mesh_type: Literal["visual", "collision"], visible: bool
    ) -> dict[int, viser.BatchedGlbHandle]:
        return self._create_mesh_handles_static(
            server=self.server,
            mj_model=self.sim.mj_model,
            mesh_type=mesh_type,
            visible=visible,
        )

    @staticmethod
    def _create_mesh_handles_static(
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        mesh_type: Literal["visual", "collision"],
        visible: bool,
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
                handle = server.scene.add_batched_meshes_trimesh(
                    f"/bodies/{body_name}/{mesh_type}",
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

    def _update_viser(
        self,
    ) -> None:
        body_xpos = self.sim.mj_data.xpos
        body_xmat = self.sim.mj_data.xmat
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

    def run(self):
        loop_no_sleep_dts, loop_dts = [], []

        while self.sim._continue_running():
            start_loop_no_sleep_time = time.time()

            # Step simulation
            self.sim.sim_step()

            # Get simulation state
            sim_state_dict = self.sim.get_sim_state()

            PRINT_SIM_STATE = False
            if PRINT_SIM_STATE:
                for key, value in sim_state_dict.items():
                    print(f"{key}: {value}")
                print()

            # Update viewer
            self._update_viser()

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.sim.config.sim_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                print(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.sim.config.sim_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            # Get FPS
            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(
                PRINT_FPS_EVERY_N_SECONDS / self.sim.config.sim_dt
            )
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []


def main():
    sim = MujocoSim(MujocoSimConfig(enable_viewer=False))
    server = viser.ViserServer()
    viser_mujoco_sim = ViserMujocoSim(server, sim)
    viser_mujoco_sim.run()


if __name__ == "__main__":
    main()
