from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
import time
from pathlib import Path
from typing import Tuple

from eval.sim2real.rl_player import RlPlayer

import torch  # isort:skip
import pytorch_kinematics as pk
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa_pose_reaching import (
    compute_joint_pos_targets,
    compute_observation,
)
from isaacgymenvs.utils.observation_action_utils_sharpa import Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np
from eval.sim2sim.isaac_sim.isaac_env import create_env_from_cfg
import numpy as np
from datetime import datetime
import os
from eval.sim2real.rl_player_utils import (
    read_cfg_omegaconf,
)

N_OBS = 133
N_ACT = 29

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.05
HAND_DOF_SPEED_SCALE = 2.5

FORCE_QD_ZERO = False


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNoRosJointPosTargets:
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
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        joint_targets = self.env.joint_targets

        DEBUG = False
        if DEBUG:
            print(f"q = {q}")
            print(f"qd = {qd}")
            breakpoint()

        new_obs = compute_observation(
            q=q,
            qd=qd if not FORCE_QD_ZERO else qd * 0,
            joint_targets=joint_targets,
            reward=reward * 0,
        )
        return new_obs, reward, done, info

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((1, N_ACT), device=self.device))
        if FORCE_QD_ZERO:
            obs['obs'][29:58] = 0.0
        return obs["obs"]

    def step_with_joint_pos_targets(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action,
            prev_targets=self.env.prev_targets[:, :self.env.num_hand_arm_dofs],
            hand_moving_average=HAND_MOVING_AVERAGE,
            arm_moving_average=ARM_MOVING_AVERAGE,
            hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
            dt=self.control_dt,
        )

        obs, reward, done, info = self.env.step(
            action, joint_pos_targets=joint_pos_targets
        )
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        joint_targets = self.env.joint_targets

        DEBUG = False
        if DEBUG:
            print(f"q = {q}")
            print(f"qd = {qd}")
            breakpoint()

        new_obs = compute_observation(
            q=q,
            qd=qd if not FORCE_QD_ZERO else qd * 0,
            joint_targets=joint_targets,
            reward=reward * 0,
        )

        DEBUG = False
        if DEBUG:
            diff= (obs['obs'] - new_obs).abs()[0]
            print(f"diff = {diff}")
            print(f"diff.max() = {diff.max()}")
            print(f"diff.argsort() = {diff.argsort()}")

            from isaacgymenvs.utils.observation_action_utils_sharpa import OBS_NAMES
            idxs = diff.argsort()
            for idx in idxs:
                print(f"OBS_NAMES[{idx}] = {OBS_NAMES[idx]}")
                print(f"obs['obs'][{idx}] = {obs['obs'][0, idx]}")
                print(f"new_obs[{idx}] = {new_obs[0, idx]}")
                print(f"diff[{idx}] = {diff[idx]}")
                print(f"--------------------------------")

            breakpoint()
        return new_obs, reward, done, info, joint_pos_targets


def main():
    CONTROL_DT = 1.0 / 60.0
    CONFIG_PATH = Path(
        "/juno/u/kedia/sapg/closed_loop_testing/pose_reaching.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_NAME = "joint_vel_best"
    # CHECKPOINT_NAME = "baseline_best"
    CHECKPOINT_PATH = Path(
        # Pose reaching
        # "/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching/baseline_best.pth"
        f"/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching/{CHECKPOINT_NAME}.pth"
    )
    STIFFNESS_MULTIPLIER, DAMPING_MULTIPLIER = 1, 1
    # STIFFNESS_MULTIPLIER, DAMPING_MULTIPLIER = 1.3, 0.7
    assert CHECKPOINT_PATH.exists()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = read_cfg_omegaconf(config_path=str(CONFIG_PATH), device=DEVICE)
    # breakpoint()
    cfg.task.env.allegroDamping = DAMPING_MULTIPLIER * cfg.task.env.allegroDamping
    cfg.task.env.allegroStiffness = STIFFNESS_MULTIPLIER * cfg.task.env.allegroStiffness

    cfg.task.env.episodeLength = 1000000  # Don't end early
    cfg.task.env.successSteps = 10000  # Hold position for long time

    env = create_env_from_cfg(
        cfg=cfg,
        headless=False,
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(CHECKPOINT_PATH)
    env_state = checkpoint[0]["env_state"]
    # HACK
    # env.set_env_state(env_state)

    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )

    isaac_env_no_ros_joint_pos_targets = IsaacEnvNoRosJointPosTargets(
        env=env,
        control_dt=CONTROL_DT,
        device=DEVICE,
    )
    nominal_joint_targets = np.array([
        -1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0
    ])
    assert len(nominal_joint_targets) == 29

    # sampled_joint_targets = nominal_joint_targets + np.random.normal(0, 0.1, 29)
    sampled_joint_targets = np.array([
        -1.271,  1.871,  0.3  ,  1.676,  0.3  ,  1.785,  1.608,  0.3  ,
        0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,
        0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3  ,
        0.3  ,  0.3  ,  0.3  ,  0.3  ,  0.3
    ])
    # sampled_joint_targets = np.array([-1.52132858, 1.55717357, 0.06476885, 1.52830299, -0.02341534, 1.4615863,
    #   2.51592128, 0.07674347, -0.04694744, 0.054256, -0.04634177, -0.04657298,
    #   0.02419623, -0.19132802, -0.17249178, -0.05622875, -0.10128311, 0.03142473,
    #   -0.09080241, -0.14123037, 0.14656488, -0.02257763, 0.00675282, -0.14247482,
    #   -0.05443827, 0.01109226, -0.11509936, 0.0375698, -0.06006387])
    joint_targets = np.clip(sampled_joint_targets, Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np)
    joint_targets = torch.from_numpy(joint_targets).float().to(DEVICE)
    
    isaac_env_no_ros_joint_pos_targets.env.joint_targets = joint_targets[None].clone()
    observation = isaac_env_no_ros_joint_pos_targets.reset()
    current_step = 0

    data = {
        'robot_joint_positions_array': [],
        'robot_joint_velocities_array': [],
        'robot_joint_accelerations_array': [],
        'robot_joint_pos_targets_array': [],
        'hand_joint_velocities_array': [],
        'hand_joint_accelerations_array': [],
    }
    prev_joint_velocities = np.zeros(29)
    while True:
        isaac_env_no_ros_joint_pos_targets.env.joint_targets = joint_targets[None].clone()
        start_time = time.time()
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
        
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, done, info, joint_pos_targets = (
            isaac_env_no_ros_joint_pos_targets.step_with_joint_pos_targets(action)
        )
        data['robot_joint_pos_targets_array'].append(joint_pos_targets[0][:29].cpu().numpy())
        success = (mean_kuka_mse_error < 0.01 and mean_hand_mse_error < 0.05) or info['successes'].item()
        # print(f"Mean Kuka MSE Error = {mean_kuka_mse_error}")
        # print(f"Mean Hand MSE Error = {mean_hand_mse_error}")
        # print(f"Step {current_step}, Success = {success}")
        current_step += 1
        # if success:
        # if False:
        if current_step == 500:
            current_step = 0
            break
        end_time = time.time()
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            pass
            # print(
            #     f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            # )

    data['robot_joint_positions_array'] = np.array(data['robot_joint_positions_array'])
    data['robot_joint_velocities_array'] = np.array(data['robot_joint_velocities_array'])
    data['robot_joint_accelerations_array'] = np.array(data['robot_joint_accelerations_array'])
    data['robot_joint_pos_targets_array'] = np.array(data['robot_joint_pos_targets_array'])
    data['hand_joint_velocities_array'] = np.array(data['hand_joint_velocities_array'])
    data['hand_joint_accelerations_array'] = np.array(data['hand_joint_accelerations_array'])
    # print mean squared joint velocities and accelerations
    # breakpoint()
    print(f"CHECKPOINT_NAME = {CHECKPOINT_NAME} Stiffness Multiplier = {STIFFNESS_MULTIPLIER} Damping Multiplier = {DAMPING_MULTIPLIER}")
    print(f"Mean Squared Joint Velocities = {np.mean(data['robot_joint_velocities_array']**2)}")
    print(f"Mean Squared Joint Accelerations = {np.mean(data['robot_joint_accelerations_array']**2)}")
    print(f"Mean Squared Hand Joint Velocities = {np.mean(data['hand_joint_velocities_array']**2)}")
    print(f"Mean Squared Hand Joint Accelerations = {np.mean(data['hand_joint_accelerations_array']**2)}")
    hand_mean_mse_error = np.sqrt(np.mean(data['hand_joint_accelerations_array']**2))
    rounded_hand_mean_mse_error = round(hand_mean_mse_error, 1)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # npz_dir = f'/juno/u/kedia/sapg/recorded_robot_states/pose_reaching_FINAL/{CHECKPOINT_NAME}_{STIFFNESS_MULTIPLIER}_{DAMPING_MULTIPLIER}'
    npz_dir = f'recorded_robot_states/pose_reaching_FINAL/isaac/{CHECKPOINT_NAME}_{STIFFNESS_MULTIPLIER}_{DAMPING_MULTIPLIER}'
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f'{rounded_hand_mean_mse_error}.npz'), **data)
    print(f"Saved data to {os.path.join(npz_dir, f'{rounded_hand_mean_mse_error}.npz')}")
    # close the environment
    del env
    del isaac_env_no_ros_joint_pos_targets
    del policy
    # breakpoint()

if __name__ == "__main__":
    main()
