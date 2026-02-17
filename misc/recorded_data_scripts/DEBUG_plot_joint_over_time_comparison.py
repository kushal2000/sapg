import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import plot_grid_of_values
from misc.recorded_data_scripts.recorded_data import JOINT_NAMES_ISAACGYM

N_JOINT = len(JOINT_NAMES_ISAACGYM)
T = 100
NAME_TO_JOINT_POS_OVER_TIME = {
    "v1": np.random.randn(T, N_JOINT),
    "v2": np.random.randn(T, N_JOINT),
    "v1_target": np.random.randn(T, N_JOINT),
    "v2_target": np.random.randn(T, N_JOINT),
}

JOINT_POS_TITLE_TO_NAME_TO_X_Y = {
    f"Joint_Pos_{i}_{joint_pos_name}": {
        name: (np.arange(T), joint_pos[:, i])
        for name, joint_pos in NAME_TO_JOINT_POS_OVER_TIME.items()
    }
    for i, joint_pos_name in enumerate(JOINT_NAMES_ISAACGYM)
}

plot_grid_of_values(
    title_to_name_to_x_y=JOINT_POS_TITLE_TO_NAME_TO_X_Y, grid_name="Joint_Positions"
)
