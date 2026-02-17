import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import (
    plot_per_idx_comparison,
    visualize_fig,
)
from misc.recorded_data_scripts.recorded_data import ACTION_NAMES

N_ACT = len(ACTION_NAMES)
NAME_TO_ACTIONS = {
    "v1": np.random.randn(N_ACT),
    "v2": np.random.randn(N_ACT),
}

act_fig = plot_per_idx_comparison(
    name_to_y={name: act for name, act in NAME_TO_ACTIONS.items()},
    title="Actions",
    yaxis_title="Action Value",
    y_names=ACTION_NAMES,
)
visualize_fig(act_fig, save_html=True)
