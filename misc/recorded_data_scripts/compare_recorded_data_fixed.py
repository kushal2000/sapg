import numpy as np
from pathlib import Path
from misc.recorded_data_scripts.plot_recorded_data_comparison import plot_grid_of_values
from misc.recorded_data_scripts.recorded_data_sharpa import JOINT_NAMES_ISAACGYM, RecordedData

# ###############
# Load data
# ###############
# Read in recorded data
# sim = RecordedData.from_file(Path("recorded_robot_inputs/isaac/2025-12-12_19-41-34_cleanInputs_arm0.05.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/real_world_replay_from_isaac/2025-12-12_21-05-02_replay_2025-12-12_19-41-34_cleanInputs_arm0.05.npz"))

# sim = RecordedData.from_file(Path("recorded_robot_inputs/mujoco/2025-12-12_19-45-23_noisyInputs_arm0.05.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/real_world_replay_from_mujoco/2025-12-12_21-10-11_replay_2025-12-12_19-45-23_noisyInputs_arm0.05.npz"))

# sim = RecordedData.from_file(Path("recorded_robot_inputs/isaac/2025-12-12_19-40-52_noisyInputs_arm0.01.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/real_world_replay_from_isaac/2025-12-12_20-49-48_replay_2025-12-12_19-40-52_noisyInputs_arm0.01.npz"))

# sim = RecordedData.from_file(Path("recorded_robot_inputs/isaac/2025-12-12_19-43-50_cleanInputs_arm0.01.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/real_world_replay_from_isaac/2025-12-12_20-59-59_replay_2025-12-12_19-43-50_cleanInputs_arm0.01.npz"))

# sim = RecordedData.from_file(Path("recorded_robot_inputs/isaac/2025-12-12_19-40-13_noisyInputs_arm0.05.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/isaac_replay_from_isaac/2025-12-14_18-59-30_replay_2025-12-12_19-40-13_noisyInputs_arm0.05.npz"))

sim = RecordedData.from_file(Path("recorded_robot_inputs/2025-12-16_real_world/2025-12-16_16-04-49_cleanInputsFinetuned_arm0.05.npz"))
# real = RecordedData.from_file(Path("recorded_robot_inputs/2025-12-16_real_world/2025-12-16_16-19-23_cleanInputsFinetuned_arm0.03.npz"))
real = RecordedData.from_file(Path("recorded_robot_inputs/2025-12-16_real_world/2025-12-16_16-34-56_noisyInputs_arm0.05.npz"))

# Extract data from recorded data
sim_object_pose_array = sim.object_root_states_array[:, :7]
real_object_pose_array = real.object_root_states_array[:, :7]

sim_joint_positions_array = sim.robot_joint_positions_array
real_joint_positions_array = real.robot_joint_positions_array

sim_joint_pos_targets_array = sim.robot_joint_pos_targets_array
real_joint_pos_targets_array = real.robot_joint_pos_targets_array

sim_joint_velocities_array = sim.robot_joint_velocities_array
real_joint_velocities_array = real.robot_joint_velocities_array

sim_robot_ee_pose_array = sim.robot_ee_pose_array
real_robot_ee_pose_array = real.robot_ee_pose_array

sim_robot_target_ee_pose_array = sim.robot_target_ee_pose_array
real_robot_target_ee_pose_array = real.robot_target_ee_pose_array

sim_time_array = sim.time_array
real_time_array = real.time_array

# ###############
# Object Pose
# ###############
NAME_TO_OBJECT_POSE_OVER_TIME = {
    "pose_sim": (sim_time_array, sim_object_pose_array),
    "pose_real": (real_time_array, real_object_pose_array),
}
POSE_NAMES = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
OBJECT_POSE_TITLE_TO_NAME_TO_X_Y = {
    f"Object_Pose_{i}_{object_pose_name}": {
        name: (time_array, object_pose[:, i])
        for name, (time_array, object_pose) in NAME_TO_OBJECT_POSE_OVER_TIME.items()
    }
    for i, object_pose_name in enumerate(POSE_NAMES)
}
plot_grid_of_values(
    title_to_name_to_x_y=OBJECT_POSE_TITLE_TO_NAME_TO_X_Y, grid_name="Object_Pose"
)

# ###############
# Joint Positions
# ###############
NAME_TO_JOINT_POS_OVER_TIME = {
    "pos_sim": (sim_time_array, sim_joint_positions_array),
    "pos_real": (real_time_array, real_joint_positions_array),
    "target_sim": (sim_time_array, sim_joint_pos_targets_array),
    "target_real": (real_time_array, real_joint_pos_targets_array),
}
JOINT_POS_TITLE_TO_NAME_TO_X_Y = {
    f"Joint_Pos_{i}_{name}": {
        name: (time_array, joint_pos[:, i])
        for name, (time_array, joint_pos) in NAME_TO_JOINT_POS_OVER_TIME.items()
    }
    for i, name in enumerate(JOINT_NAMES_ISAACGYM)
}
plot_grid_of_values(
    title_to_name_to_x_y=JOINT_POS_TITLE_TO_NAME_TO_X_Y, grid_name="Joint_Positions"
)

# ###############
# Joint Velocities
# ###############
NAME_TO_JOINT_VEL_OVER_TIME = {
    "vel_sim": (sim_time_array, sim_joint_velocities_array),
    "vel_real": (real_time_array, real_joint_velocities_array),
}
JOINT_VEL_TITLE_TO_NAME_TO_X_Y = {
    f"Joint_Vel_{i}_{name}": {
        name: (time_array, joint_vel[:, i])
        for name, (time_array, joint_vel) in NAME_TO_JOINT_VEL_OVER_TIME.items()
    }
    for i, name in enumerate(JOINT_NAMES_ISAACGYM)
}
plot_grid_of_values(
    title_to_name_to_x_y=JOINT_VEL_TITLE_TO_NAME_TO_X_Y, grid_name="Joint_Velocities"
)

# ###############
# Robot EE Pose
# ###############
NAME_TO_ROBOT_EE_POSE_OVER_TIME = {
    "ee_pose_sim": (sim_time_array, sim_robot_ee_pose_array),
    "ee_pose_real": (real_time_array, real_robot_ee_pose_array),
    "target_ee_pose_sim": (sim_time_array, sim_robot_target_ee_pose_array),
    "target_ee_pose_real": (real_time_array, real_robot_target_ee_pose_array),
}
ROBOT_EE_POSE_TITLE_TO_NAME_TO_X_Y = {
    f"Robot_EE_Pose_{i}_{pose_name}": {
        name: (time_array, ee_pose[:, i])
        for name, (time_array, ee_pose) in NAME_TO_ROBOT_EE_POSE_OVER_TIME.items()
    }
    for i, pose_name in enumerate(POSE_NAMES)
}
plot_grid_of_values(
    title_to_name_to_x_y=ROBOT_EE_POSE_TITLE_TO_NAME_TO_X_Y, grid_name="Robot_EE_Pose"
)

# ###############
# EE pos z and Object pos z
# ###############

EE_POS_Z_AND_OBJECT_POS_Z_TITLE_TO_NAME_TO_X_Y = {
    "Pos_Z": {
        "ee_pos_z_sim": (sim_time_array, sim_robot_ee_pose_array[:, 2]),
        "ee_pos_z_real": (real_time_array, real_robot_ee_pose_array[:, 2]),
        "target_ee_pos_z_sim": (sim_time_array, sim_robot_target_ee_pose_array[:, 2]),
        "target_ee_pos_z_real": (real_time_array, real_robot_target_ee_pose_array[:, 2]),
        "object_pos_z_sim": (sim_time_array, sim_object_pose_array[:, 2]),
        "object_pos_z_real": (real_time_array, real_object_pose_array[:, 2]),
    }
}
plot_grid_of_values(
    title_to_name_to_x_y=EE_POS_Z_AND_OBJECT_POS_Z_TITLE_TO_NAME_TO_X_Y, grid_name="Pos_Z"
)

# ###############
# EE vel z and Object vel z
# ###############
EE_VEL_Z_AND_OBJECT_VEL_Z_TITLE_TO_NAME_TO_X_Y = {
    "Vel_Z": {
        "ee_vel_z_sim": (sim_time_array[:-1], np.diff(sim_robot_ee_pose_array[:, 2])),
        "ee_vel_z_real": (real_time_array[:-1], np.diff(real_robot_ee_pose_array[:, 2])),
        "target_ee_vel_z_sim": (sim_time_array[:-1], np.diff(sim_robot_target_ee_pose_array[:, 2])),
        "target_ee_vel_z_real": (real_time_array[:-1], np.diff(real_robot_target_ee_pose_array[:, 2])),
        "object_vel_z_sim": (sim_time_array[:-1], np.diff(sim_object_pose_array[:, 2])),
        "object_vel_z_real": (real_time_array[:-1], np.diff(real_object_pose_array[:, 2])),
    }
}
plot_grid_of_values(
    title_to_name_to_x_y=EE_VEL_Z_AND_OBJECT_VEL_Z_TITLE_TO_NAME_TO_X_Y, grid_name="Vel_Z"
)
