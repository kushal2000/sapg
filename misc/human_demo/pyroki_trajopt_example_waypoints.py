"""Trajectory Optimization with Waypoints (Fixed for 8-DOF Panda)"""

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
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    
    # [FIX] Inspect DOF to handle the gripper joints
    print(f"Robot Active DOF: {robot.joints.num_actuated_joints}")
    print(f"Robot Joint Names: {robot.joints.actuated_names}")

    # Standard 7-DOF Arm Configuration
    arm_cfg = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # [FIX] Pad start_cfg if the robot includes gripper joints (8 DOF)
    if robot.joints.num_actuated_joints > len(arm_cfg):
        padding = np.zeros(robot.joints.num_actuated_joints - len(arm_cfg))
        # Default gripper width (0.04 is usually open-ish for Panda)
        padding[:] = 0.04 
        start_cfg = np.concatenate([arm_cfg, padding])
        print(f"Padded start_cfg to {len(start_cfg)} (added gripper vars)")
    else:
        start_cfg = arm_cfg
    
    # Define Obstacle (Table/Wall)
    wall_height = 0.4
    wall_width = 0.2
    wall_center = np.array([0.5, 0.0, wall_height/2])
    
    wall_coll = pk.collision.Capsule.from_radius_height(
        position=np.array([[0.5, 0.0, wall_height/2]]),
        radius=np.array([[wall_width/2]]),
        height=np.array([[wall_height]])
    )
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    world_coll = [ground_coll, wall_coll]

    # --- Define Problem ---
    timesteps, dt = 250, 0.05
    total_time = timesteps * dt
    print(f"Creating a trajectory with {timesteps} waypoints, each {dt} seconds apart, for a total time of {total_time} seconds")
    down_wxyz = np.array([0, 1, 0, 0]) 
    
    early_pos = np.array([0.5, 0.0, 0.6]) # High Z to clear wall
    mid_pos = np.array([0.5, -0.4, 0.2])  # One side
    end_pos = np.array([0.5, 0.4, 0.2])  # Other side

    # --- WAYPOINT API ---
    waypoints = {
        20: (early_pos, down_wxyz),
        125: (mid_pos, down_wxyz),
        249: (end_pos, down_wxyz)
    }

    print("Solving trajectory with waypoints...")
    start_time = time.time()
    
    # Make sure to run this on GPU (should happen automatically now)
    traj = pks.solve_waypoint_trajopt(
        robot=robot,
        robot_coll=robot_coll,
        world_coll=world_coll,
        target_link_name="panda_hand",
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
        "wall",
        trimesh.creation.box(
            extents=(0.2, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(wall_center)
        )
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