import numpy as np

from misc.recorded_data_scripts.plot_recorded_data_comparison import (
    plot_per_idx_comparison,
    visualize_fig,
)
from misc.recorded_data_scripts.recorded_data import OBS_NAMES

N_OBS = len(OBS_NAMES)
NAME_TO_OBSERVATIONS = {
    "v1": np.random.randn(N_OBS),
    "v2": np.random.randn(N_OBS),
}

for obs in NAME_TO_OBSERVATIONS.values():
    assert obs.shape == (len(OBS_NAMES),), (
        f"Expected observations to have shape (len(OBS_NAMES),) = ({len(OBS_NAMES)},), got {obs.shape}"
    )

obs_fig = plot_per_idx_comparison(
    name_to_y={name: obs for name, obs in NAME_TO_OBSERVATIONS.items()},
    title="Observations",
    yaxis_title="Obs Value",
    y_names=OBS_NAMES,
)
visualize_fig(obs_fig, save_html=True)
