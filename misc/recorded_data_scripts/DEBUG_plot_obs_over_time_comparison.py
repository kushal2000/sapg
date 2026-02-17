import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import plot_grid_of_values
from misc.recorded_data_scripts.recorded_data import OBS_NAMES

N_OBS = len(OBS_NAMES)
T = 100
NAME_TO_OBSERVATIONS_OVER_TIME = {
    "v1": np.random.randn(T, N_OBS),
    "v2": np.random.randn(T, N_OBS),
}

OBS_TITLE_TO_NAME_TO_X_Y = {
    f"Obs_{i}_{obs_name}": {
        name: (np.arange(T), observations[:, i])
        for name, observations in NAME_TO_OBSERVATIONS_OVER_TIME.items()
    }
    for i, obs_name in enumerate(OBS_NAMES)
}

plot_grid_of_values(
    title_to_name_to_x_y=OBS_TITLE_TO_NAME_TO_X_Y, grid_name="Observations"
)
