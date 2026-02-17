import time
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import colored
import yourdfpy

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from eval.sim2real.rl_player import RlPlayer
from eval.sim2sim.mujoco_sim.mujoco_sim_sharpa import (
    MujocoSim,
    MujocoSimConfig,
)

N_OBS = 140
N_ACT = 29


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class MujocoEnvNoRosSharpa:
    def __init__(
        self,
        sim: MujocoSim,
        object_scales: np.ndarray,
        urdf: yourdfpy.URDF,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        control_dt: float,
        device: str,
        obs_list: list[str],
    ):
        self.sim = sim
        self.object_scales = object_scales
        self.urdf = urdf
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device
        self.obs_list = obs_list

    def compute_observation(self) -> torch.Tensor:
        sim_state = self.sim.get_sim_state()

        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
        object_pose_W = np.concatenate([object_pos, object_quat_xyzw])

        goal_object_pos = sim_state["goal_object_pos"]
        goal_object_quat_wxyz = sim_state["goal_object_quat_wxyz"]
        goal_object_quat_xyzw = goal_object_quat_wxyz[[1, 2, 3, 0]]
        goal_object_pose_W = np.concatenate([goal_object_pos, goal_object_quat_xyzw])

        q = sim_state["joint_positions"]
        qd = sim_state["joint_velocities"]

        observation = compute_observation(
            q=q[None],
            qd=qd[None],
            prev_action_targets=self.sim.robot_joint_pos_targets[None],
            object_pose=object_pose_W[None],
            goal_object_pose=goal_object_pose_W[None],
            object_scales=self.object_scales[None],
            urdf=self.urdf,
            obs_list=self.obs_list,
        )
        observation = torch.from_numpy(observation).float().to(self.device)

        assert observation.shape == (
            1,
            N_OBS,
        ), f"observation.shape: {observation.shape}, expected: (1, {N_OBS})"
        return observation

    def step(self, action: torch.Tensor) -> None:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=self.sim.robot_joint_pos_targets[None],
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            hand_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
        )
        self.sim.set_robot_joint_pos_targets(joint_pos_targets[0])

        for _ in range(self.sim_steps_per_control_step):
            self.sim.sim_step()

            if self.sim.config.enable_viewer:
                self.sim.viewer.sync()

        return

    @property
    def sim_steps_per_control_step(self) -> int:
        return int(self.control_dt / self.sim.config.sim_dt)


def main():
    # Parameters
    SIM_DT = 1.0 / 600.0  # Mujoco sim step (needs to be small to get stable physics)
    CONTROL_DT = 1.0 / 60.0  # Control loop frequency (policy loop rate)
    HAND_MOVING_AVERAGE = 0.1
    ARM_MOVING_AVERAGE = 0.05
    HAND_DOF_SPEED_SCALE = 2.5

    # Cuboid
    # OBJECT_SCALES = np.array([4.0000, 0.7500, 1.0000])
    # OBJECT_SCALES = np.array([4.0000, 0.7500, 1.0000]) * 1.25
    OBJECT_SCALES = np.array([0.24, 0.03, 0.02]) * 25

    CONFIG_PATH = Path(
        "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path(
        # "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/newGains.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/noisyInput.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/2025-12-11_newGains/cleanInputs.pth"
        "/juno/u/kedia/sapg/train_dir/checkpoints/2025-12-11_newGains/noisyInputs.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/cleanInputsFinetuned.pth"
    )
    assert CHECKPOINT_PATH.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="cuboid_4_0.75_1"))
    # sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="cuboid_5_0.9375_1.25"))
    sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="cuboidal_mallet", object_start_pos=np.array([0.0, 0.0, 0.58]), object_start_quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0]), goal_object_start_pos=np.array([0.0, 0.0, 0.78]), goal_object_start_quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0])))
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")

    obs_list = policy.cfg["task"]["env"]["obsList"]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"obs_list: {obs_list}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    mujoco_env_no_ros = MujocoEnvNoRosSharpa(
        sim=sim,
        object_scales=OBJECT_SCALES,
        urdf=urdf,
        hand_moving_average=HAND_MOVING_AVERAGE,
        arm_moving_average=ARM_MOVING_AVERAGE,
        hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
        control_dt=CONTROL_DT,
        device=device,
        obs_list=obs_list,
    )

    while True:
        start_time = time.time()
        # Get observation, get action, step simulation
        observation = mujoco_env_no_ros.compute_observation()
        action = policy.get_normalized_action(
            observation, deterministic_actions=True
        )
        mujoco_env_no_ros.step(action)
        end_time = time.time()

        # Sleep to maintain control loop frequency
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
