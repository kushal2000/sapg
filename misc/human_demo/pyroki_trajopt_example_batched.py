"""Trajectory Optimization (Batched)

Demonstrates solving N trajectory optimization problems in parallel using JAX vmap.
"""

import time
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


def main(robot_name: Literal["ur5", "panda"] = "panda"):
    if robot_name == "ur5":
        urdf = load_robot_description("ur5_description")
        down_wxyz = np.array([0.707, 0, 0.707, 0])
        target_link_name = "ee_link"
        default_cfg = np.zeros(6)
        default_cfg[1] = -1.308
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)
    elif robot_name == "panda":
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
        down_wxyz = np.array([0, 0, 1, 0]) 
        robot = pk.Robot.from_urdf(urdf)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # --- BATCH SETUP ---
    timesteps, dt = 25, 0.02
    N = 10  # Batch size
    
    # We will solve N problems simultaneously.
    # Let's vary the start/end positions along the Y-axis to see different paths.
    base_start_pos = np.array([0.5, -0.3, 0.2])
    base_end_pos = np.array([0.5, 0.3, 0.2])
    
    # Create (N, 3) arrays
    y_offsets = np.linspace(-0.2, 0.2, N)
    
    start_pos_batch = np.tile(base_start_pos, (N, 1))
    start_pos_batch[:, 1] += y_offsets  # Spread starts
    
    end_pos_batch = np.tile(base_end_pos, (N, 1))
    end_pos_batch[:, 1] += y_offsets    # Spread ends
    
    # Rotations are constant across the batch, but must be tiled to (N, 4)
    start_wxyz_batch = np.tile(down_wxyz, (N, 1))
    end_wxyz_batch = np.tile(down_wxyz, (N, 1))
    # -------------------

    # Define obstacles (Same as original)
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # Wall
    wall_height = 0.4
    wall_width = 0.1
    wall_length = 0.4
    wall_intervals = np.arange(start=0.3, stop=wall_length + 0.3, step=0.05)
    translation = np.concatenate(
        [
            wall_intervals.reshape(-1, 1),
            np.full((wall_intervals.shape[0], 1), 0.0),
            np.full((wall_intervals.shape[0], 1), wall_height / 2),
        ],
        axis=1,
    )
    wall_coll = pk.collision.Capsule.from_radius_height(
        position=translation,
        radius=np.full((translation.shape[0], 1), wall_width / 2),
        height=np.full((translation.shape[0], 1), wall_height),
    )
    world_coll = [ground_coll, wall_coll]

    print(f"Solving {N} trajectories in parallel...")
    start_time = time.time()
    
    # 1. Resolve string to index OUTSIDE the batched function
    target_link_index = robot.links.names.index(target_link_name)
    
    # 2. Call batched solver
    traj_batch = pks.solve_trajopt_batched(
        robot,
        robot_coll,
        world_coll,
        target_link_index,
        start_pos_batch,
        start_wxyz_batch,
        end_pos_batch,
        end_wxyz_batch,
        timesteps,
        dt,
    )
    traj_batch = np.array(traj_batch) # Convert to numpy for visualization
    print(f"Solved in {time.time() - start_time:.4f}s")
    
    # Traj batch shape: (N, Timesteps, DOF)

    # Visualize!
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    
    # Visualise Wall
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(wall_length, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0.5, 0.0, wall_height / 2])
            ),
        ),
    )

    # Visualize Start/End frames for the *current* batch index
    def update_frames(batch_idx):
        server.scene.add_frame(
            "/start",
            position=start_pos_batch[batch_idx],
            wxyz=start_wxyz_batch[batch_idx],
            axes_length=0.05,
            axes_radius=0.01,
        )
        server.scene.add_frame(
            "/end",
            position=end_pos_batch[batch_idx],
            wxyz=end_wxyz_batch[batch_idx],
            axes_length=0.05,
            axes_radius=0.01,
        )

    # GUI
    batch_slider = server.gui.add_slider(
        "Batch ID", min=0, max=N - 1, step=1, initial_value=0
    )
    time_slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    update_frames(0)

    while True:
        # Update frames if batch ID changed
        update_frames(batch_slider.value)
        
        if playing.value:
            time_slider.value = (time_slider.value + 1) % timesteps

        # Index into the batch, then the timestep
        cfg = traj_batch[batch_slider.value, time_slider.value]
        urdf_vis.update_cfg(cfg)
        
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)