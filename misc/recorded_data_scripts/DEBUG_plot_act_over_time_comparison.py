import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import plot_grid_of_values
from misc.recorded_data_scripts.recorded_data import ACTION_NAMES

N_ACT = len(ACTION_NAMES)
T = 100
NAME_TO_ACTIONS_OVER_TIME = {
    "v1": np.random.randn(T, N_ACT),
    "v2": np.random.randn(T, N_ACT),
}

ACTION_TITLE_TO_NAME_TO_X_Y = {
    f"Action_{i}_{action_name}": {
        name: (np.arange(T), actions[:, i])
        for name, actions in NAME_TO_ACTIONS_OVER_TIME.items()
    }
    for i, action_name in enumerate(ACTION_NAMES)
}

plot_grid_of_values(
    title_to_name_to_x_y=ACTION_TITLE_TO_NAME_TO_X_Y, grid_name="Actions"
)
