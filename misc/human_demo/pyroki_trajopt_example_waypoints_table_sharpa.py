"""Trajectory Optimization with Waypoints (Fixed for 29-DOF Sharpa+Kuka IIWA)"""


# Cache
# Seems like 37s instead of 57s
USE_CACHE = True
if USE_CACHE:
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import time
import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki_snippets as pks

def main():
    # --- Setup Robot & Scene ---
    # urdf = load_robot_description("panda_description")
    import yourdfpy
    # Sharpa
    urdf = yourdfpy.URDF.load("assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    
    # [FIX] Inspect DOF to handle the gripper joints
    print(f"Robot Active DOF: {robot.joints.num_actuated_joints}")
    print(f"Robot Joint Names: {robot.joints.actuated_names}")

    # Standard 7-DOF Arm Configuration
    HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571 - np.deg2rad(10), -0.000, 1.376 + np.deg2rad(10), -0.000, 1.485, 1.308])
    HOME_JOINT_POS_SHARPA = np.zeros(22)
    HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])

    assert len(HOME_JOINT_POS) == robot.joints.num_actuated_joints, (
        f"HOME_JOINT_POS.shape: {HOME_JOINT_POS.shape}, expected: ({robot.joints.num_actuated_joints},)"
    )
    
    start_cfg = HOME_JOINT_POS
    
    # Define Obstacle (Table)
    table_size = np.array([0.475, 0.4, 0.3])
    table_center = np.array([0.0, -0.8, 0.38])
    
    # Pyroki only works with halfspaces, capsules, and spheres
    # Boxes are not supported
    table_coll = pk.collision.Capsule.from_radius_height(
        position=table_center,
        radius=np.array([table_size[0]/2]),
        height=np.array([table_size[2]])
    )
    # table_coll = pk.collision.Box.from_extents(
    #     position=table_center,
    #     extents=table_size
    # )
    world_coll = [table_coll]

    # --- Define Problem ---
    timesteps, dt = 250, 0.05
    total_time = timesteps * dt
    print(f"Creating a trajectory with {timesteps} waypoints, each {dt} seconds apart, for a total time of {total_time} seconds")
    palm_down_wxyz = np.array([0.5, 0.5, 0.5, -0.5])
    
    early_pos = np.array([0.0, -0.6, 0.8]) # High Z to clear wall
    mid_pos = np.array([0.4, -0.6, 0.8])  # One side
    end_pos = np.array([-0.4, -0.6, 0.8])  # Other side

    # --- WAYPOINT API ---
    HOLD_STEPS = 20
    waypoints = {}
    waypoints.update({
        60 + i: (early_pos, palm_down_wxyz)
        for i in range(HOLD_STEPS)
    })
    waypoints.update({
        120 + i: (mid_pos, palm_down_wxyz)
        for i in range(HOLD_STEPS)
    })
    waypoints.update({
        180 + i: (end_pos, palm_down_wxyz)
        for i in range(HOLD_STEPS)
    })

    print("Solving trajectory with waypoints...")
    start_time = time.time()
    
    # Make sure to run this on GPU (should happen automatically now)
    traj = pks.solve_waypoint_trajopt(
        robot=robot,
        robot_coll=robot_coll,
        world_coll=world_coll,
        target_link_name="left_hand_C_MC",
        start_cfg=start_cfg,
        waypoints=waypoints,
        timesteps=timesteps,
        dt=dt
    )
    
    # Block until ready to measure real execution time
    # (Checking traj.shape implicitly blocks)
    print(f"Solved in {time.time() - start_time:.4f}s")

    # --- Visualization ---
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2)
    
    server.scene.add_mesh_trimesh(
        "table",
        # trimesh.creation.box(
        #     extents=table_size,
        #     transform=trimesh.transformations.translation_matrix(table_center)
        # )
        # trimesh.creation.capsule(
        #     radius=table_size[0]/2,
        #     height=table_size[2],
        #     transform=trimesh.transformations.translation_matrix(table_center)
        # )
        table_coll.to_trimesh()
    )
    server.scene.add_mesh_trimesh(
        "robot_coll",
        robot_coll.at_config(robot, start_cfg).to_trimesh()
    )

    for t, (pos, wxyz) in waypoints.items():
        server.scene.add_frame(
            f"/waypoint_t{t}",
            position=pos,
            wxyz=wxyz,
            axes_length=0.1,
            axes_radius=0.01,
        )

    slider = server.gui.add_slider("Timestep", 0, timesteps-1, 1, 0)
    playing = server.gui.add_checkbox("Playing", True)
    
    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        
        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(dt)

if __name__ == "__main__":
    tyro.cli(main)
