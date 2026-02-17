import time
from pathlib import Path
from misc.recorded_data_scripts.recorded_data_sharpa import RecordedData

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
from eval.sim2sim.mujoco_sim.mujoco_sim_sharpa import (
    MujocoSim,
    MujocoSimConfig,
)

N_OBS = 133
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
        object_pose_R = np.concatenate([object_pos, object_quat_wxyz])

        table_pos = sim_state["table_pos"]
        table_quat_wxyz = sim_state["table_quat_wxyz"]
        goal_object_pos = table_pos + np.array([0.0, 0.0, 0.5])
        goal_object_quat_wxyz = table_quat_wxyz
        goal_object_pose_R = np.concatenate([goal_object_pos, goal_object_quat_wxyz])

        q = sim_state["joint_positions"]
        qd = sim_state["joint_velocities"]

        observation = compute_observation(
            q=q[None],
            qd=qd[None],
            prev_action_targets=self.sim.robot_joint_pos_targets[None],
            object_pose=object_pose_R[None],
            goal_object_pose=(
                goal_object_pose_R[None]
            ),
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

    def step(self, joint_pos_targets: np.ndarray) -> None:
        self.sim.set_robot_joint_pos_targets(joint_pos_targets)

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
    SIM_DT = 1.0 / 6_000.0  # Mujoco sim step (needs to be small to get stable physics)
    CONTROL_DT = 1.0 / 60.0  # Control loop frequency (policy loop rate)
    HAND_MOVING_AVERAGE = 0.1
    ARM_MOVING_AVERAGE = 0.1
    HAND_DOF_SPEED_SCALE = 0.5
    OBJECT_SCALES = np.array([0.1, 0.035, 0.025]) * 20

    CONFIG_PATH = Path(
        "/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    # CHECKPOINT_PATH = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-22_slow-action-obs-randomize-all_slower-curriculum/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/runs/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/last/model.pth")
    # CHECKPOINT_PATH = Path("/juno/u/kedia/sapg/train_dir/checkpoints/hammer/absoluteControl_0.5.pth")
    CHECKPOINT_PATH = Path(
        "/juno/u/kedia/sapg/train_dir/checkpoints/hammer/relativeControl_5.pth"
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

    joint_positions_array = recorded_data.robot_joint_positions_array
    T = joint_pos_targets_array.shape[0]
    assert joint_pos_targets_array.shape == (T, N_ACT), f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}, expected: ({T}, {N_ACT})"
    assert joint_positions_array.shape == (T, N_ACT), f"joint_positions_array.shape: {joint_positions_array.shape}, expected: ({T}, {N_ACT})"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = MujocoSim(MujocoSimConfig(enable_viewer=True, sim_dt=SIM_DT, object_name="hairbrush"))

    urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")

    mujoco_env_no_ros = MujocoEnvNoRosSharpa(
        sim=sim,
        object_scales=OBJECT_SCALES,
        urdf=urdf,
        hand_moving_average=HAND_MOVING_AVERAGE,
        arm_moving_average=ARM_MOVING_AVERAGE,
        hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
        control_dt=CONTROL_DT,
        device=device,
        obs_list=["joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "palm_vel", "object_rot", "object_vel", "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales", "closest_keypoint_max_dist", "closest_fingertip_dist", "lifted_object", "progress", "successes", "reward"],
    )
    mujoco_env_no_ros.sim.set_robot_joint_positions(joint_positions_array[0])
    mujoco_env_no_ros.sim.set_robot_joint_pos_targets(joint_pos_targets_array[0])
    # mujoco_env_no_ros.sim.set_object_position(recorded_data.object_root_states_array[0, :3] + np.array([0.0, -0.5, 0.0]))
    mujoco_env_no_ros.sim.set_object_position(recorded_data.object_root_states_array[0, :3])
    mujoco_env_no_ros.sim.set_object_quat_wxyz(recorded_data.object_root_states_array[0, 3:7][[3, 0, 1, 2]])

    NO_MOVE_FIRST_N_STEPS = 100
    print(f"No moving for {NO_MOVE_FIRST_N_STEPS} steps")
    for _ in range(NO_MOVE_FIRST_N_STEPS):
        start_time = time.time()
        mujoco_env_no_ros.step(joint_positions_array[0])
        end_time = time.time()
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )
    print(f"Done no moving for {NO_MOVE_FIRST_N_STEPS} steps")

    joint_pos_history = []
    joint_vel_history = []
    robot_base_root_states_history = []
    object_root_states_history = []
    table_root_states_history = []
    idx = 0
    while True:
        start_time = time.time()
        # Get observation, step simulation
        observation = mujoco_env_no_ros.compute_observation()
        mujoco_env_no_ros.step(joint_pos_targets_array[idx])
        sim_state = mujoco_env_no_ros.sim.get_sim_state()
        joint_pos_history.append(sim_state["joint_positions"])
        joint_vel_history.append(sim_state["joint_velocities"])

        robot_base_pos = sim_state["robot_base_pos"]
        robot_base_quat_wxyz = sim_state["robot_base_quat_wxyz"]
        robot_base_quat_xyzw = robot_base_quat_wxyz[[1, 2, 3, 0]]
        robot_base_root_states = np.concatenate([robot_base_pos, robot_base_quat_xyzw, np.zeros(6)])
        robot_base_root_states_history.append(robot_base_root_states)
        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
        object_root_states = np.concatenate([object_pos, object_quat_xyzw, np.zeros(6)])
        object_root_states_history.append(object_root_states)
        table_pos = sim_state["table_pos"]
        table_quat_wxyz = sim_state["table_quat_wxyz"]
        table_quat_xyzw = table_quat_wxyz[[1, 2, 3, 0]]
        table_root_states = np.concatenate([table_pos, table_quat_xyzw, np.zeros(6)])
        table_root_states_history.append(table_root_states)
        idx += 1
        if idx >= T:
            joint_pos_history = np.array(joint_pos_history)
            joint_vel_history = np.array(joint_vel_history)
            robot_base_root_states_history = np.array(robot_base_root_states_history)
            object_root_states_history = np.array(object_root_states_history)
            table_root_states_history = np.array(table_root_states_history)
            print(f"Reached end of trajectory!")
            print(f"joint_pos_history.shape: {joint_pos_history.shape}")
            print(f"joint_vel_history.shape: {joint_vel_history.shape}")
            print(f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}")
            assert joint_pos_history.shape == joint_pos_targets_array.shape, f"joint_pos_history.shape: {joint_pos_history.shape}, expected: {joint_pos_targets_array.shape}"
            assert joint_vel_history.shape == joint_pos_targets_array.shape, f"joint_vel_history.shape: {joint_vel_history.shape}, expected: {joint_pos_targets_array.shape}"
            # robot_root_states_array = np.zeros((T, 13))
            # robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
            # object_root_states_array = np.zeros((T, 13))
            # object_root_states_array[:, 6] = 1.0
            from isaacgymenvs.utils.observation_action_utils_sharpa import JOINT_NAMES_ISAACGYM
            robot_joint_names = JOINT_NAMES_ISAACGYM
            time_array = np.arange(T) * CONTROL_DT
            new_recorded_data = RecordedData(
                robot_root_states_array=robot_base_root_states_history,
                object_root_states_array=object_root_states_history,
                robot_joint_positions_array=joint_pos_history,
                robot_joint_velocities_array=joint_vel_history,
                time_array=time_array,
                robot_joint_names=robot_joint_names,
                robot_joint_pos_targets_array=joint_pos_targets_array,
                table_root_states_array=table_root_states_history,
                object_name=sim.config.object_name,
            )
            # output_path = RECORDED_DATA_PATH.parent / f"{RECORDED_DATA_PATH.stem}_mujoco.npz"
            output_path = RECORDED_DATA_PATH.parent / f"{RECORDED_DATA_PATH.stem}_mujoco.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving recorded data to {output_path}")
            new_recorded_data.to_file(output_path)
            print(f"Saved recorded data to {output_path}")
            breakpoint()
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
