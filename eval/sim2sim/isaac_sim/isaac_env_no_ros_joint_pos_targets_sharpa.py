from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
from isaacgymenvs.utils.utils import get_repo_root_dir
import json
import time
from pathlib import Path
from typing import Tuple

import yourdfpy
from eval.sim2real.rl_player import RlPlayer

import torch  # isort:skip
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from eval.sim2sim.isaac_sim.isaac_env import create_env

N_OBS = 140
N_ACT = 29

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.05
HAND_DOF_SPEED_SCALE = 2.5


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
        urdf: yourdfpy.URDF,
    ):
        self.env = env
        self.control_dt = control_dt
        self.device = device
        self.urdf = urdf

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((self.env.num_envs, N_ACT), device=self.device))
        return obs["obs"]

    def step_with_joint_pos_targets(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=self.env.prev_targets.cpu().numpy(),
            hand_moving_average=HAND_MOVING_AVERAGE,
            arm_moving_average=ARM_MOVING_AVERAGE,
            hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
            dt=self.control_dt,
        )
        joint_pos_targets = torch.from_numpy(joint_pos_targets).float().to(self.device)

        obs, reward, done, info = self.env.step(
            action, joint_pos_targets=joint_pos_targets
        )
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        object_pose = self.env.object_pose
        goal_object_pose = self.env.goal_pose
        object_scales = self.env.object_scales

        DEBUG = False
        if DEBUG:
            print(f"q = {q}")
            print(f"qd = {qd}")
            print(f"object_pose = {object_pose}")
            print(f"goal_object_pose = {goal_object_pose}")
            print(f"object_scales = {object_scales}")
            breakpoint()

        # # HACK: Overwrite
        # goal_object_pose[:] = torch.tensor([0.,  0.,  0.88, 0.,  0.,  0.,  1.], device=self.device)[None]
        # object_scales[:] = torch.tensor([5.0, 0.9375, 1.25], device=self.device)[None]

        new_obs = compute_observation(
            q=q.cpu().numpy(),
            qd=qd.cpu().numpy(),
            prev_action_targets=self.env.prev_targets.cpu().numpy(),
            object_pose=object_pose.cpu().numpy(),
            goal_object_pose=goal_object_pose.cpu().numpy(),
            object_scales=object_scales.cpu().numpy(),
            urdf=self.urdf,
            obs_list=self.env.obs_list,
        )
        new_obs = torch.from_numpy(new_obs).float().to(self.device)

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
        return new_obs, reward, done, info


def main():
    CONTROL_DT = 1.0 / 60.0
    CONFIG_PATH = Path(
        # "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/config.yaml"
        "/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_slowSpeed/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path(
        # "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/newGains.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/noisyInput.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/2025-12-11_newGains/cleanInputs.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/cleanInputsFinetuned.pth",
        # "/juno/u/kedia/sapg/train_dir/checkpoints/FINETUNED/finetuned_o1t0.pth",
        # "/juno/u/kedia/sapg/train_dir/checkpoints/FINETUNED/finetuned_o1t1.pth",
        # "/juno/u/kedia/sapg/train_dir/checkpoints/FINETUNED/finetuned_o0t0.pth",
        "/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_slowSpeed/model.pth"
    )
    assert CHECKPOINT_PATH.exists()

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

    # object_type = "screwdriver"
    # object_name = "real_flat_screwdriver"
    # trajectory_name = "top_down_screwing_human_easyinit"
    # trajectory_name = "top_down_screwing_human"

    # object_type = "eraser"
    # object_name = "whiteboard_eraser"
    # trajectory_name = "wipe_left_human"
    # trajectory_name = "wipe_left_human_2"

    object_type = "marker"
    object_name = "040_large_marker"
    trajectory_name = "draw_circle_human"
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

    # NOTE: cpu has different physics than training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.resetRotationNoise": 0.0,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            # "task.env.object_type": "blue_cuboid",
            # "task.env.object_type": "tyler_handle_head",
            # "task.env.object_type": "blue_cuboid_fake_hammer",
            "task.env.object_type": object_name,
            # "task.env.forceNoReset": True,
            "task.env.randomizeObjectRotation": False,
            # "task.env.objectStartPose": [0.,  0.,  0.58, 0.,  0.,  0.,  1.],  # x, y, z, qx, qy, qz, qw
            # "task.env.objectStartPose": [0.,  0.,  0.58, 0.,  0.,  1.,  0.],  # x, y, z, qx, qy, qz, qw
            # "task.env.goalObjectPose": [0.,  0.,  0.88, 0.,  0.,  0.,  1.],  # x, y, z, qx, qy, qz, qw
            # "task.env.use_fixed_set_of_goal_states": True,
            # "task.env.use_fixed_set_of_goal_states": False,
            "task.env.objectStartPose": traj_data["start_pose"],
            "task.env.use_fixed_set_of_goal_states": True,
            "task.env.fixedGoalStates": traj_data["goals"],
            "task.env.forceScale": 0.0,
            "task.env.numEnvs": 1,  # The compute functions use CPU, so parallelizing to many envs will be very slow here
            "task.env.envSpacing": 0.4,
            "task.env.tableResetZRange": 0.0,
            "task.env.capture_video": False,
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.resetWhenDropped": False,
            "task.env.armMovingAverage": 0.1,
            "task.env.evalSuccessTolerance": 0.02,
            "task.env.successSteps": 1,
            "task.env.fixedSizeKeypointReward": True,
            "task.env.forceOnlyWhenLifted": True,
            "task.env.startArmHigher": True,
        },
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(CHECKPOINT_PATH)
    env_state = checkpoint[0]["env_state"]
    env.set_env_state(env_state)

    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        num_envs=env.num_envs,
    )

    urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")

    isaac_env_no_ros_joint_pos_targets = IsaacEnvNoRosJointPosTargets(
        env=env,
        control_dt=CONTROL_DT,
        device=DEVICE,
        urdf=urdf,
    )
    observation = isaac_env_no_ros_joint_pos_targets.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, done, _ = (
            isaac_env_no_ros_joint_pos_targets.step_with_joint_pos_targets(action)
        )
        # if done.item():
        #     observation = isaac_env_no_ros_joint_pos_targets.reset()
        end_time = time.time()
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            warn(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
