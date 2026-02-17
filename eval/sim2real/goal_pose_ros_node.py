#!/usr/bin/env python
from copy import deepcopy
from pathlib import Path
import json
import time
from typing import Literal
import torch
from isaacgymenvs.utils.observation_action_utils_sharpa import _compute_keypoint_positions
from isaacgymenvs.utils.utils import get_repo_root_dir
from isaacgymenvs.utils.objects import (
    NAME_TO_OBJECT,
)

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from termcolor import colored

def info(message: str):
    print(colored(message, "green"))

def warn(message: str):
    print(colored(message, "yellow"))

def warn_every(message: str, n_seconds: float, key=None):
    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def keypoint_distance(pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray) -> float:
    """Compute the distance between two keypoints."""
    object_keypoint_positions = _compute_keypoint_positions(
        pose=pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    N_KEYPOINTS = 4
    N = 1
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    keypoint_distances_l2 = np.linalg.norm(keypoints_rel_goal, axis=-1).max(axis=-1)
    return keypoint_distances_l2


class GoalPoseNode:
    def __init__(self, goal_object_pose_file: Path, object_scales: np.ndarray, success_threshold: float, success_steps: int):
        # ROS setup
        rospy.init_node("goal_pose_ros_node")

        KEYPOINT_SCALE = 1.5
        self.object_scales = object_scales
        self.success_threshold = success_threshold
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SCALE
        self.success_steps = success_steps
        self.current_success_steps = 0

        # Goal object pose 
        # Assumes xyzw quat convention
        assert goal_object_pose_file.exists(), f"File does not exist: {goal_object_pose_file}"
        assert goal_object_pose_file.suffix == ".json", f"Expected JSON file, got {goal_object_pose_file}"
        with open(goal_object_pose_file, "r") as f:
            self.goal_object_poses = np.array(json.load(f))
        N, D = self.goal_object_poses.shape
        assert D == 7, f"Expected 7 dimensions, got {D}"
        assert N > 0, f"Expected at least 1 goal object pose, got {N}"

        # State
        self.current_goal_object_pose_index = 0

        # ROS msgs
        self.latest_current_object_pose = None

        # Publisher and subscriber
        self.goal_object_pose_pub = rospy.Publisher("/robot_frame/goal_object_pose", Pose, queue_size=1)
        self.current_object_pose_sub = rospy.Subscriber("/robot_frame/current_object_pose", PoseStamped, self.current_object_pose_callback, queue_size=1)

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def current_object_pose_callback(self, msg: PoseStamped):
        """Callback to update the current object pose."""
        self.latest_current_object_pose = msg.pose

    def update_goal_object_pose(self):
        """Update the goal object pose."""
        latest_current_object_pose = deepcopy(self.latest_current_object_pose)
        p = latest_current_object_pose

        current_object_pose_xyzw = np.array([p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        current_goal_object_pose_xyzw = self.goal_object_poses[self.current_goal_object_pose_index]

        distance = keypoint_distance(
            pose1_xyzw=current_object_pose_xyzw, pose2_xyzw=current_goal_object_pose_xyzw, object_scales=self.object_scales
        )
        print(f"Distance: {distance}")

        if distance < self.keypoint_success_threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                info(f"Success threshold reached, updating goal object pose index to {self.current_goal_object_pose_index + 1}")
                self.current_success_steps = 0
                self.current_goal_object_pose_index += 1
                if self.current_goal_object_pose_index >= self.goal_object_poses.shape[0]:
                    self.current_goal_object_pose_index = self.goal_object_poses.shape[0] - 1
            else:
                info(f"Success threshold reached, at {self.current_success_steps} of {self.success_steps} steps")

    def publish_goal_object_pose(self):
        """Publish the goal object pose."""
        current_goal_object_pose_xyzw = self.goal_object_poses[self.current_goal_object_pose_index]
        goal_object_pose_msg = Pose()
        goal_object_pose_msg.position.x = current_goal_object_pose_xyzw[0]
        goal_object_pose_msg.position.y = current_goal_object_pose_xyzw[1]
        goal_object_pose_msg.position.z = current_goal_object_pose_xyzw[2]
        goal_object_pose_msg.orientation.x = current_goal_object_pose_xyzw[3]
        goal_object_pose_msg.orientation.y = current_goal_object_pose_xyzw[4]
        goal_object_pose_msg.orientation.z = current_goal_object_pose_xyzw[5]
        goal_object_pose_msg.orientation.w = current_goal_object_pose_xyzw[6]
        self.goal_object_pose_pub.publish(goal_object_pose_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""

        # Wait for the current object pose to be received
        while not rospy.is_shutdown():
            if self.latest_current_object_pose is None:
                warn_every("Waiting for current object pose", n_seconds=1.0)
                time.sleep(0.1)
            else:
                info("Current object pose received, starting goal pose node")
                break  # All messages received, exit loop

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the goal object pose
            self.update_goal_object_pose()

            # Publish the goal object pose
            self.publish_goal_object_pose()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
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
    # Load trajectory
    # This makes it easier to change object and trajectory

    # object_type = "hammer"
    # object_name = "mallet"
    # object_name = "hammer_2"
    # trajectory_name = "vertical_swing"
    # trajectory_name = "horizontal_swing"
    # trajectory_name = "horizontal_swing_higher"
    # trajectory_name = "horizontal_swing_human"
    # trajectory_name = "horizontal_swing_human_closer"

    # object_type = "spatula"
    # object_name = "black_spatula"
    # trajectory_name = "pick_and_place_human"
    # trajectory_name = "pick_and_place_human_hardinit"

    object_type = "screwdriver"
    object_name = "real_flat_screwdriver"
    # trajectory_name = "top_down_screwing_human_easyinit"
    # trajectory_name = "top_down_screwing_human"
    trajectory_name = "top_down_screwing_closer"

    # object_type = "eraser"
    # object_name = "whiteboard_eraser"
    # trajectory_name = "wipe_left_human"
    # trajectory_name = "wipe_left_human_2"

    # object_type = "marker"
    # object_name = "040_large_marker"
    # trajectory_name = "draw_circle_human"
    # trajectory_name = "draw_circle_human_hardinit"

    # object_type = "brush"
    # object_name = "green_brush"
    # object_name = "red_brush"
    # trajectory_name = "simple"
    # trajectory_name = "complex"

    trajectory_path = get_repo_root_dir() / "dex_tool_bench/evaluation_trajectories" / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Account for robot to world frame
    goals_world_frame = traj_data["goals"]
    goals_robot_frame = [[x, y - 0.8, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goals_world_frame]

    tmp_file = Path("tmp.json")
    with open(tmp_file, "w") as f:
        json.dump(goals_robot_frame, f)

    try:
        # Create and run the GoalPoseNode
        node = GoalPoseNode(
            # goal_object_pose_file=Path("dummy_goals.json"),
            # goal_object_pose_file=Path("dummy_goals_flat.json"),
            # goal_object_pose_file=Path("goal_pose_listener_ros_node_output/mallet/2025-11-17_15-39-20.json"),
            # goal_object_pose_file=Path("goal_pose_listener_ros_node_output/mallet/2025-11-17_15-40-29.json"),
            # goal_object_pose_file=Path("goal_pose_listener_ros_node_output/mallet/2025-11-17_15-44-33.json"),
            # goal_object_pose_file=Path("goal_pose_listener_ros_node_output/mallet/2025-11-17_15-59-04.json"),
            # goal_object_pose_file=Path("goal_pose_listener_ros_node_output/mallet/2025-11-17_16-17-57.json"),
            # goal_object_pose_file=Path("goal_poses_around_z_axis.json"),
            # goal_object_pose_file=Path("goal_poses_around_y_axis.json"),
            # goal_object_pose_file=Path("hammer_trajectory_2.json"),
            # goal_object_pose_file=Path("hammer_trajectory.json"),
            # goal_object_pose_file=Path("real_flat_screwdriver_trajectory.json"),
            # goal_object_pose_file=Path("040_large_marker_trajectory.json"),
            # goal_object_pose_file=Path("whiteboard_eraser_trajectory.json"),
            goal_object_pose_file=tmp_file,
            # object_scales=np.array([5.0, 0.9375, 1.25]),
            # object_scales=np.array([0.24, 0.03, 0.02]) * 25,  # Mallet
            # object_scales=np.array([0.25, 0.03, 0.02]) * 25,  # scanned hammer 2
            # object_scales=np.array([0.1, 0.03, 0.02]) * 25,  # real flat screwdriver
            # object_scales=np.array([0.121277, 0.019341, 0.021183]) * 25,  # 040 large marker
            # object_scales=np.array(NAME_TO_OBJECT[object_name].scale),
            # object_scales=np.array(NAME_TO_OBJECT[object_name].scale),
            object_scales=np.array([0.141, 0.03025, 0.0271]) * 25,  # fixed size
            # object_scales=np.array([0.12965531, 0.0337145 , 0.06038587]) * 25,  # whiteboard eraser
            # object_scales=np.array([0.15954332, 0.0777093 , 0.01231273]) * 25,  # iphone15pro
            # success_threshold=0.0,
            # success_threshold=0.01,
            success_threshold=0.02,
            # success_threshold=0.05,
            success_steps=1,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()