from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
import numpy as np
from misc.recorded_data_scripts.recorded_data_sharpa import RecordedData
import time
from pathlib import Path
from typing import Tuple

from eval.sim2real.rl_player import RlPlayer

import torch  # isort:skip
import pytorch_kinematics as pk
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
)
from eval.sim2sim.isaac_sim.isaac_env import create_env

N_OBS = 117
N_ACT = 29

ACT_MOVING_AVERAGE = 0.1
HAND_DOF_SPEED_SCALE = 0.5


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNoRosJointPosTargetsFromFile:
    def __init__(
        self,
        env: AllegroKukaBase,
        control_dt: float,
        device: str,
    ):
        self.env = env
        self.control_dt = control_dt
        self.device = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return obs["obs"], reward, done, info

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((1, N_ACT), device=self.device))
        return obs["obs"]

    def step_with_joint_pos_targets(
        self, joint_pos_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(
            torch.zeros((1, N_ACT), device=self.device), joint_pos_targets=joint_pos_targets
        )
        return obs["obs"], reward, done, info


def main():
    CONTROL_DT = 1.0 / 60.0
    CONFIG_PATH = Path(
        "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path(
        "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/last/model.pth"
    )
    assert CHECKPOINT_PATH.exists()

    RECORDED_DATA_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-48-58_sin_wave_hand_10-0s_1-0s_0-2rad.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-11-06_17-09-47_None_550.npz"  # Slow sliced
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-07_13-43-59_slowpolicyopenloop.npz"  # Real world policy open loop
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-07_14-07-41_slowpolicytargets.npz"  # Real world policy targets
        "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-09_15-22-31_sharpa_sin_wave.npz"  # Sharpa sin wave
    )
    assert RECORDED_DATA_PATH.exists()
    recorded_data = RecordedData.from_file(RECORDED_DATA_PATH)
    joint_pos_targets_array = recorded_data.robot_joint_pos_targets_array
    T = joint_pos_targets_array.shape[0]
    assert joint_pos_targets_array.shape == (T, N_ACT), f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}, expected: ({T}, {N_ACT})"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
        episode_length=T*2,  # Make it not reset before finishing the trajectory
        overrides={
            "task.env.asset.kukaAllegro": "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
            "task.task.randomize": False,
            "task.env.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT": True,
        },
        # overrides={"task.env.asset.kukaAllegro": "urdf/kuka_allegro_description/iiwa14_left_sharpa.urdf"},
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(CHECKPOINT_PATH)
    env_state = checkpoint[0]["env_state"]
    env.set_env_state(env_state)

    isaac_env_no_ros_joint_pos_targets_from_file = IsaacEnvNoRosJointPosTargetsFromFile(
        env=env,
        control_dt=CONTROL_DT,
        device=DEVICE,
    )
    observation = isaac_env_no_ros_joint_pos_targets_from_file.reset()
    joint_pos_history = []
    joint_vel_history = []
    # joint_pos_history.append(isaac_env_no_ros_joint_pos_targets_from_file.env.arm_hand_dof_pos.clone().cpu().numpy()[0])

    idx = 0
    while True:
        start_time = time.time()
        # print(f"idx: {idx}")
        observation, _, done, _ = (
            isaac_env_no_ros_joint_pos_targets_from_file.step_with_joint_pos_targets(torch.from_numpy(joint_pos_targets_array[idx]).to(DEVICE).float().unsqueeze(0))
        )
        joint_pos_history.append(isaac_env_no_ros_joint_pos_targets_from_file.env.arm_hand_dof_pos.clone().cpu().numpy()[0])
        joint_vel_history.append(isaac_env_no_ros_joint_pos_targets_from_file.env.arm_hand_dof_vel.clone().cpu().numpy()[0])
        idx += 1
        # idx -= 1   #HACK
        if idx >= T:
            joint_pos_history = np.array(joint_pos_history)
            joint_vel_history = np.array(joint_vel_history)
            print(f"Reached end of trajectory!")
            print(f"joint_pos_history.shape: {joint_pos_history.shape}")
            print(f"joint_vel_history.shape: {joint_vel_history.shape}")
            print(f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}")
            assert joint_pos_history.shape == joint_pos_targets_array.shape, f"joint_pos_history.shape: {joint_pos_history.shape}, expected: {joint_pos_targets_array.shape}"
            assert joint_vel_history.shape == joint_pos_targets_array.shape, f"joint_vel_history.shape: {joint_vel_history.shape}, expected: {joint_pos_targets_array.shape}"
            robot_root_states_array = np.zeros((T, 13))
            robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
            object_root_states_array = np.zeros((T, 13))
            object_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
            robot_joint_names = [
                "iiwa_joint_1",
                "iiwa_joint_2",
                "iiwa_joint_3",
                "iiwa_joint_4",
                "iiwa_joint_5",
                "iiwa_joint_6",
                "iiwa_joint_7",
                "sharpa_joint_0",
                "sharpa_joint_1",
                "sharpa_joint_2",
                "sharpa_joint_3",
                "sharpa_joint_4",
                "sharpa_joint_5",
                "sharpa_joint_6",
                "sharpa_joint_7",
                "sharpa_joint_8",
                "sharpa_joint_9",
                "sharpa_joint_10",
                "sharpa_joint_11",
                "sharpa_joint_12",
                "sharpa_joint_13",
                "sharpa_joint_14",
                "sharpa_joint_15",
                "sharpa_joint_16",
                "sharpa_joint_17",
                "sharpa_joint_18",
                "sharpa_joint_19",
                "sharpa_joint_20",
                "sharpa_joint_21",
            ]
            time_array = np.arange(T) * CONTROL_DT
            new_recorded_data = RecordedData(
                robot_root_states_array=robot_root_states_array,
                object_root_states_array=object_root_states_array,
                robot_joint_positions_array=joint_pos_history,
                robot_joint_velocities_array=joint_vel_history,
                time_array=time_array,
                robot_joint_names=robot_joint_names,
                robot_joint_pos_targets_array=joint_pos_targets_array,
            )
            output_path = RECORDED_DATA_PATH.parent / f"{RECORDED_DATA_PATH.stem}_isaac.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving recorded data to {output_path}")
            new_recorded_data.to_file(output_path)
            print(f"Saved recorded data to {output_path}")
            breakpoint()
        if done.item():
            # idx = 0
            observation = isaac_env_no_ros_joint_pos_targets_from_file.reset()
        end_time = time.time()
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
