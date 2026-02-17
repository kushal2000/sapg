import time
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa_pose_reaching import (
    compute_joint_pos_targets,
    compute_observation,
)
from isaacgymenvs.utils.observation_action_utils_sharpa import Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np
from eval.sim2real.rl_player import RlPlayer
from eval.sim2sim.mujoco_sim.mujoco_sim_sharpa import (
    MujocoSim,
    MujocoSimConfig,
)

N_OBS = 133
N_ACT = 29

FORCE_QD_ZERO = False

def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


NOMINAL_JOINT_TARGETS = np.array([
    -1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0
])
assert len(NOMINAL_JOINT_TARGETS) == 29
np.random.seed(42)

# SAMPLED_JOINT_TARGETS = NOMINAL_JOINT_TARGETS + np.random.normal(0, 0.1, 29)
SAMPLED_JOINT_TARGETS = np.array([
    -1.271,  1.871,  0.3  ,  1.676,  0.3  ,  1.785,  1.608,  0.3  ,
    0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,
    0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,
    0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3
])
JOINT_TARGETS = np.clip(SAMPLED_JOINT_TARGETS, Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JOINT_TARGETS = torch.from_numpy(JOINT_TARGETS).float().to(DEVICE)

class MujocoEnvNoRosSharpa:
    def __init__(
        self,
        sim: MujocoSim,
        object_scales: np.ndarray,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        control_dt: float,
        device: str,
    ):
        self.sim = sim
        self.object_scales = object_scales
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device

    def compute_observation(self) -> torch.Tensor:
        sim_state = self.sim.get_sim_state()

        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_pose_R = np.concatenate([object_pos, object_quat_wxyz])

        table_pos = sim_state["table_pos"]
        table_quat_wxyz = sim_state["table_quat_wxyz"]
        goal_object_pos = table_pos + np.array([0.0, 0.0, 0.5])
        goal_object_quat_wxyz = table_quat_wxyz
        goal_object_pose_R = np.concatenate([goal_object_pos, goal_object_quat_wxyz])

        q = sim_state["joint_positions"]
        qd = sim_state["joint_velocities"]

        observation = compute_observation(
            q=torch.from_numpy(q).float().to(self.device)[None],
            qd=torch.from_numpy(qd if not FORCE_QD_ZERO else qd * 0).float().to(self.device)[None],
            joint_targets=JOINT_TARGETS[None],
            reward=torch.zeros(1, device=self.device)[None],
        )
        assert observation.shape == (
            1,
            N_OBS,
        ), f"observation.shape: {observation.shape}, expected: (1, {N_OBS})"
        return observation

    def step(self, action: torch.Tensor) -> None:
        joint_cmd = compute_joint_pos_targets(
            actions=action,
            prev_targets=torch.from_numpy(self.sim.robot_joint_pos_targets)
            .float()
            .to(self.device)[None],
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            hand_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
        )
        self.sim.set_robot_joint_pos_targets(joint_cmd.squeeze(dim=0).cpu().numpy())

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
    # ARM_MOVING_AVERAGE = 0.01
    # ARM_MOVING_AVERAGE = 0.02
    # HAND_DOF_SPEED_SCALE = 10.0
    # HAND_DOF_SPEED_SCALE = 4.075
    HAND_DOF_SPEED_SCALE = 2.5
    # OBJECT_SCALES = np.array([0.1, 0.035, 0.025]) * 20
    # OBJECT_SCALES = np.array([3.0, 0.5, 0.5])

    # blue_cuboid
    # OBJECT_SCALES = np.array([4.0, 1.0, 0.75])

    # Hammer 2
    OBJECT_SCALES = np.array([3.0, 0.25, 0.2])

    CONFIG_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml"
        # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/config.yaml"
        # "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa_hammer_2/config.yaml"
        "/juno/u/kedia/sapg/closed_loop_testing/pose_reaching.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    # CHECKPOINT_PATH = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-22_slow-action-obs-randomize-all_slower-curriculum/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/runs/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/last/model.pth")
    # CHECKPOINT_PATH = Path("/juno/u/kedia/sapg/train_dir/checkpoints/hammer/absoluteControl_0.5.pth")
    CHECKPOINT_NAME = "baseline_best"
    # CHECKPOINT_NAME = "joint_vel_best"
    CHECKPOINT_PATH = Path(
        # "/juno/u/kedia/sapg/train_dir/checkpoints/hammer/relativeControl_5.pth"
        # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/last/model.pth"

        # DR 4.075 speed
        # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/hammer_dr_4.075/00_DR_REAL_FINETUNING_SLOW_2025-11-15_13-49-55.pth"

        # NODR 2.5 speed
        # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/hammer_nodr_2.5/00_REAL_FINETUNING_SLOW_2025-11-15_13-51-31.pth"

        # Cuboid
        # "/juno/u/kedia/sapg/train_dir/checkpoints/2025_11_17_checkpoints/cuboid_nodr_5/00_SLOW_CUBOID_2025-11-14_11-59-02.pth"

        # Pose reaching
        f"/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching/{CHECKPOINT_NAME}.pth"
        # f"/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching/baseline_best.pth"
        # f"/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching/joint_vel_best.pth"
    )
    assert CHECKPOINT_PATH.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="scanned_hammer_2"))
    # sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="mallet"))
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )


    mujoco_env_no_ros = MujocoEnvNoRosSharpa(
        sim=sim,
        object_scales=OBJECT_SCALES,
        hand_moving_average=HAND_MOVING_AVERAGE,
        arm_moving_average=ARM_MOVING_AVERAGE,
        hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
        control_dt=CONTROL_DT,
        device=device,
    )

    data = {
        'robot_joint_positions_array': [],
        'robot_joint_velocities_array': [],
        'robot_joint_accelerations_array': [],
        'robot_joint_pos_targets_array': [],
        'hand_joint_velocities_array': [],
        'hand_joint_accelerations_array': [],
    }
    prev_joint_velocities = np.zeros(29)
    joint_targets = JOINT_TARGETS
    current_step = 0
    while True:
        start_time = time.time()
        # Get observation, get action, step simulation
        observation = mujoco_env_no_ros.compute_observation()
        joint_accelerations = (observation[0][29:58].cpu().numpy() - prev_joint_velocities) / CONTROL_DT
        data['robot_joint_positions_array'].append(observation[0][:29].cpu().numpy())
        data['robot_joint_velocities_array'].append(observation[0][29:58].cpu().numpy())
        data['robot_joint_accelerations_array'].append(joint_accelerations[:])
        data['hand_joint_velocities_array'].append(observation[0][36:65].cpu().numpy())
        data['hand_joint_accelerations_array'].append(joint_accelerations[7:])
        prev_joint_velocities = observation[0][29:58].cpu().numpy()
        error = (joint_targets - observation[0][:29]).abs().cpu().numpy()
        mean_kuka_mse_error = np.mean(error[:7]**2)
        mean_hand_mse_error = np.mean(error[7:]**2)

        action = policy.get_normalized_action(
            observation, deterministic_actions=True
        )  # Careful about deterministic_actions=True here!
        data['robot_joint_pos_targets_array'].append(mujoco_env_no_ros.sim.robot_joint_pos_targets)
        mujoco_env_no_ros.step(action)
        end_time = time.time()
        current_step += 1
        if current_step == 500:
            current_step = 0
            break

        # Sleep to maintain control loop frequency
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )

    data['robot_joint_positions_array'] = np.array(data['robot_joint_positions_array'])
    data['robot_joint_velocities_array'] = np.array(data['robot_joint_velocities_array'])
    data['robot_joint_accelerations_array'] = np.array(data['robot_joint_accelerations_array'])
    data['robot_joint_pos_targets_array'] = np.array(data['robot_joint_pos_targets_array'])
    data['hand_joint_velocities_array'] = np.array(data['hand_joint_velocities_array'])
    data['hand_joint_accelerations_array'] = np.array(data['hand_joint_accelerations_array'])
    # print mean squared joint velocities and accelerations
    # breakpoint()
    print(f"Mean Squared Joint Velocities = {np.mean(data['robot_joint_velocities_array']**2)}")
    print(f"Mean Squared Joint Accelerations = {np.mean(data['robot_joint_accelerations_array']**2)}")
    print(f"Mean Squared Hand Joint Velocities = {np.mean(data['hand_joint_velocities_array']**2)}")
    print(f"Mean Squared Hand Joint Accelerations = {np.mean(data['hand_joint_accelerations_array']**2)}")
    hand_mean_mse_error = np.sqrt(np.mean(data['hand_joint_accelerations_array']**2))
    rounded_hand_mean_mse_error = round(hand_mean_mse_error, 1)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # npz_dir = f'/juno/u/kedia/sapg/recorded_robot_states/pose_reaching_FINAL/{CHECKPOINT_NAME}_{STIFFNESS_MULTIPLIER}_{DAMPING_MULTIPLIER}'
    # npz_dir = f'/juno/u/tylerlum/sapg/recorded_robot_states/pose_reaching_FINAL/{CHECKPOINT_NAME}_{STIFFNESS_MULTIPLIER}_{DAMPING_MULTIPLIER}'
    npz_dir = f'recorded_robot_states/pose_reaching_FINAL/mujoco/{CHECKPOINT_NAME}'
    import os
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f'{rounded_hand_mean_mse_error}.npz'), **data)
    print(f"Saved data to {os.path.join(npz_dir, f'{rounded_hand_mean_mse_error}.npz')}")


if __name__ == "__main__":
    main()
