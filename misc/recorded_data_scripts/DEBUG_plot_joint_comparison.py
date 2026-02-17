import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import (
    plot_per_idx_comparison,
    visualize_fig,
)
from misc.recorded_data_scripts.recorded_data import JOINT_NAMES_ISAACGYM

N_JOINT = len(JOINT_NAMES_ISAACGYM)
NAME_TO_JOINT_POS = {
    "v1": np.random.randn(N_JOINT),
    "v2": np.random.randn(N_JOINT),
    "v1_target": np.random.randn(N_JOINT),
    "v2_target": np.random.randn(N_JOINT),
}

joint_pos_fig = plot_per_idx_comparison(
    name_to_y={name: joint_pos for name, joint_pos in NAME_TO_JOINT_POS.items()},
    title="Joint Positions",
    yaxis_title="Joint Position Value",
    y_names=JOINT_NAMES_ISAACGYM,
)
visualize_fig(joint_pos_fig, save_html=True)
