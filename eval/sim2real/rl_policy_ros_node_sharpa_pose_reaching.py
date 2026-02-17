#!/usr/bin/env python

import copy
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pytorch_kinematics as pk
import rospy
import torch
from geometry_msgs.msg import Pose, PoseStamped
from rl_player import RlPlayer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from termcolor import colored


from isaacgymenvs.utils.observation_action_utils_sharpa_pose_reaching import (
    compute_joint_pos_targets,
    compute_observation,
    Q_LOWER_LIMITS_restricted_np as Q_LOWER_LIMITS_np,
    Q_UPPER_LIMITS_restricted_np as Q_UPPER_LIMITS_np,
)


T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])


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


def info(message: str):
    print(colored(message, "green"))


def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"


def get_ros_loop_rate_str(
    start_time: rospy.Time,
    before_sleep_time: rospy.Time,
    after_sleep_time: rospy.Time,
    node_name: Optional[str] = None,
) -> str:
    max_rate_dt = (before_sleep_time - start_time).to_sec()
    max_rate_hz = 1 / max_rate_dt
    actual_rate_dt = (after_sleep_time - start_time).to_sec()
    actual_rate_hz = 1 / actual_rate_dt
    loop_rate_str = f"Max rate: {np.round(max_rate_hz, 1)} Hz ({np.round(max_rate_dt * 1000, 1)} ms), Actual rate: {np.round(actual_rate_hz, 1)} Hz"
    return f"{node_name} {loop_rate_str}" if node_name is not None else loop_rate_str


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


class RLPolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("rl_policy_node_sharpa_pose_reaching")

        # Publisher for iiwa and sharpa joint commands
        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=1
        )
        self.sharpa_joint_cmd_pub = rospy.Publisher(
            "/sharpa/joint_cmd", JointState, queue_size=1
        )

        # Variables to store the latest messages
        self.iiwa_joint_state_msg = None
        self.sharpa_joint_state_msg = None
        self.joint_targets_msg = None

        # Subscribers
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback,
            queue_size=1
        )
        self.sharpa_joint_state_sub = rospy.Subscriber(
            "/sharpa/joint_states", JointState, self.sharpa_joint_state_callback,
            queue_size=1
        )
        self.joint_targets_sub = rospy.Subscriber(
            "/joint_targets", JointState, self.joint_targets_callback,
            queue_size=1
        )

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 109  # Update this number based on actual dimensions
        self.num_actions = 29

        CONFIG_PATH = Path(
            # "/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml"
            # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/config.yaml"
            # "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa/config.yaml"
            # "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa_hammer_2/config.yaml"
            "/home/tylerlum/github_repos/sapg/closed_loop_testing_pose_reaching/config.yaml"
        )
        assert Path(CONFIG_PATH).exists()
        CHECKPOINT_PATH = Path(
            # Fast
            # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-12_sharpa_hammer_2_coacd/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/runs/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/last/model.pth"
            # Slow
            # "/juno/u/kedia/sapg/train_dir/checkpoints/SLOW_CUBOID/model.pth"
            # "/juno/u/kedia/sapg/train_dir/checkpoints/dr_hammer_slow.pth"
            # "/juno/u/kedia/sapg/train_dir/checkpoints/hammer_slowest.pth"

            # DR 4.075 speed
            # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/hammer_dr_4.075/00_DR_REAL_FINETUNING_SLOW_2025-11-15_13-49-55.pth"

            # NODR 2.5 speed
            # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/hammer_nodr_2.5/00_REAL_FINETUNING_SLOW_2025-11-15_13-51-31.pth"

            # Cuboid
            # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/cuboid_nodr_5/00_SLOW_CUBOID_2025-11-14_11-59-02.pth"

            # Pose reaching
            "/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching.pth"
        )
        assert CHECKPOINT_PATH.exists()

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            device=self.device,
        )

        # ROS rate
        self.rate_hz = 60
        self.dt = 1.0 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # State: prev_targets
        self.prev_targets = None

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def sharpa_joint_state_callback(self, msg: JointState):
        self.sharpa_joint_state_msg = msg

    def joint_targets_callback(self, msg: JointState):
        self.joint_targets_msg = msg

    def create_observation(self) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.sharpa_joint_state_msg is None
            or self.joint_targets_msg is None
        ):
            warn_every(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, sharpa_joint_state_msg: {var_to_is_none_str(self.sharpa_joint_state_msg)}, joint_targets_msg: {var_to_is_none_str(self.joint_targets_msg)}",
                n_seconds=1.0,
            )
            return None, None

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        sharpa_joint_state_msg = copy.copy(self.sharpa_joint_state_msg)
        joint_targets_msg = copy.copy(self.joint_targets_msg)

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        sharpa_position = np.array(sharpa_joint_state_msg.position)
        sharpa_velocity = np.array(sharpa_joint_state_msg.velocity)

        joint_targets = np.array(joint_targets_msg.position)

        q = np.concatenate([iiwa_position, sharpa_position])
        qd = np.concatenate([iiwa_velocity, sharpa_velocity])

        # HACK: Rearrange joint order
        observation = compute_observation(
            q=torch.from_numpy(q).float().to(self.device)[None],
            qd=torch.from_numpy(qd).float().to(self.device)[None],
            joint_targets=torch.from_numpy(joint_targets).float().to(self.device)[None],
        )
        assert_equals(
            observation.shape,
            (
                1,
                self.num_observations,
            ),
        )

        DEBUG = False
        if DEBUG:
            print(f"q: {q}")
            print(f"qd: {qd}")
            print(f"joint_targets: {joint_targets}")
            breakpoint()

        return observation, q

    def publish_targets(self, joint_pos_targets: torch.Tensor):
        assert_equals(joint_pos_targets.shape, (1, self.num_actions))
        joint_pos_targets = joint_pos_targets.squeeze(dim=0)
        joint_pos_targets = joint_pos_targets.cpu().numpy()

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.header.frame_id = ""
        iiwa_msg.name = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        iiwa_msg.position = joint_pos_targets[:7].tolist()
        self.iiwa_joint_cmd_pub.publish(iiwa_msg)
        sharpa_msg = JointState()
        sharpa_msg.header.stamp = rospy.Time.now()
        sharpa_msg.header.frame_id = ""
        sharpa_msg.name = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_16.0",
            "joint_17.0",
            "joint_18.0",
            "joint_19.0",
            "joint_20.0",
            "joint_21.0",
        ]
        sharpa_msg.position = joint_pos_targets[7:].tolist()
        self.sharpa_joint_cmd_pub.publish(sharpa_msg)

    def run(self):
        first_observations_received = False

        loop_no_sleep_dts, loop_dts = [], []

        # CURRENT_STEP = 0
        while not rospy.is_shutdown():
            # print(f"Current step: {CURRENT_STEP}")
            # if CURRENT_STEP > 1500:
            #     print("Exiting")
            #     import sys

            #     sys.exit(0)
            # CURRENT_STEP += 1

            start_time = rospy.Time.now()

            # Create observation from the latest messages
            obs, q = self.create_observation()

            if obs is not None and q is not None:
                if not first_observations_received:
                    info("=" * 100)
                    info("First observations received, starting to publish sim state")
                    info("=" * 100)
                    first_observations_received = True

                if self.prev_targets is None:
                    self.prev_targets = torch.from_numpy(q).float().to(self.device)[None]

                assert_equals(obs.shape, (1, self.num_observations))

                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(
                    obs=obs,
                    deterministic_actions=True,
                    # obs=obs, deterministic_actions=True
                )
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                HAND_MOVING_AVERAGE = 0.1
                ARM_MOVING_AVERAGE = 0.01
                HAND_DOF_SPEED_SCALE = 10.0
                DT = 1 / 60
                joint_pos_targets = compute_joint_pos_targets(
                    actions=normalized_action,
                    prev_targets=self.prev_targets,
                    hand_moving_average=HAND_MOVING_AVERAGE,
                    arm_moving_average=ARM_MOVING_AVERAGE,
                    hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
                    dt=DT,
                )
                assert_equals(joint_pos_targets.shape, (1, self.num_actions))

                # Clamp
                joint_pos_targets = torch.clip(
                    joint_pos_targets,
                    min=torch.from_numpy(Q_LOWER_LIMITS_np).float().to(self.device)[None],
                    max=torch.from_numpy(Q_UPPER_LIMITS_np).float().to(self.device)[None],
                )

                # Publish the targets
                self.publish_targets(joint_pos_targets)
                # print(f"CURRENT_STEP: {CURRENT_STEP}")
                # print(f"joint_pos_targets: {joint_pos_targets}")
                # print()
                self.prev_targets = joint_pos_targets.clone()

            # Sleep to maintain loop rate
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
            # if True:
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


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
