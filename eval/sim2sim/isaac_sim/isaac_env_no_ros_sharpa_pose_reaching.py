from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
import time
from pathlib import Path
from typing import Tuple

from eval.sim2real.rl_player import RlPlayer

import torch  # isort:skip
from termcolor import colored

from eval.sim2sim.isaac_sim.isaac_env import create_env

N_OBS = 109
N_ACT = 29


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNoRos:
    def __init__(self, env: AllegroKukaBase, device: str):
        self.env = env
        self.device = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        DEBUG = False
        if DEBUG:
            q = self.env.arm_hand_dof_pos
            qd = self.env.arm_hand_dof_vel
            object_pose = self.env.object_pose
            goal_object_pose = self.env.goal_pose
            object_scales = self.env.object_scales
            print(f"q = {q}")
            print(f"qd = {qd}")
            print(f"object_pose = {object_pose}")
            print(f"goal_object_pose = {goal_object_pose}")
            print(f"object_scales = {object_scales}")
            breakpoint()
        return obs["obs"], reward, done, info

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((1, N_ACT), device=self.device))
        return obs["obs"]


def main():
    CONTROL_DT = 1.0 / 60.0  # Control loop frequency (policy loop rate)
    CONFIG_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa/config.yaml"
        # "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa_hammer_2/config.yaml"
        "/juno/u/kedia/sapg/closed_loop_testing/pose_reaching.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path(
        # Fast
        # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-12_sharpa_hammer_2_coacd/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/runs/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/last/model.pth"
        # Slow
        # "/juno/u/kedia/sapg/train_dir/checkpoints/SLOW_CUBOID/model.pth"
        # "/juno/u/kedia/sapg/train_dir/checkpoints/dr_hammer_slow.pth"

        # Pose reaching
        "/juno/u/kedia/sapg/train_dir/checkpoints/pose_reaching.pth"
    )
    assert CHECKPOINT_PATH.exists()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
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

    isaac_env_no_ros = IsaacEnvNoRos(env=env, device=DEVICE)
    observation = isaac_env_no_ros.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(
            observation, deterministic_actions=True
        )  # Careful about deterministic_actions=True here!
        observation, _, _, _ = isaac_env_no_ros.step(action)
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
