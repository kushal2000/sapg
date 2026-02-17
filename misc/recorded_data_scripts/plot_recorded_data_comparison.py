from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import viser
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from misc.recorded_data_scripts.recorded_data import RecordedData

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def figures_to_grid_html(
    figures: List[go.Figure],
    filename: str,
    one_legend_per_fig: bool,
    cols: Optional[int] = None,
    base_width: int = 400,
    base_height: int = 300,
) -> tuple[Path, go.Figure]:
    """
    Arrange a list of plotly figures into a grid and save as an HTML file.

    Args:
        figures: List of plotly figures to arrange into a grid.
        filename: Name of the HTML file to save the grid to.
        one_legend_per_fig: Whether to show one legend per figure.
                            True: separate legend per subplot (each trace has its own toggle with consistent colors)
                            False: single legend shared across all subplots (linked visibility with consistent colors)
        cols: Number of columns in the grid. If None, will be determined automatically.
        base_width: Base width of each figure in pixels.
        base_height: Base height of each figure in pixels.
    """
    if len(figures) == 0:
        raise ValueError("No figures provided")
    n = len(figures)
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Use each figure's title as the title of the subplot if present, otherwise "Plot i"
    titles = []
    for i, fig in enumerate(figures):
        if getattr(fig.layout, "title", None) is not None:
            title = fig.layout.title.text
        else:
            title = f"Plot {i}"
        titles.append(title)

    subplots_fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
    )

    # If single legend: assign consistent colors manually
    base_colors = list(pio.templates["plotly"].layout.colorway)
    extra_colors = list(
        qualitative.Light24
    )  # Extra colors in case we need more than 10 from base_colors
    color_cycle = itertools.cycle(base_colors + extra_colors)

    color_map = {}
    seen_groups = set()

    for i, fig in enumerate(figures):
        row = i // cols + 1
        col = i % cols + 1

        for trace in fig.data:
            name = trace.name

            # Assign a consistent color per name
            if name not in color_map:
                color_map[name] = next(color_cycle)
            trace.line.color = color_map[name]
            trace.marker.color = color_map[name]

            if one_legend_per_fig:
                trace.showlegend = True
                trace.legendgroup = None  # Don't group by name
            else:
                trace.legendgroup = name
                if name in seen_groups:
                    trace.showlegend = False  # Don't show legend for this trace if it's already in a group
                else:
                    trace.showlegend = True
                    seen_groups.add(name)

            subplots_fig.add_trace(trace, row=row, col=col)

        x_axis = getattr(fig.layout, "xaxis", None)
        y_axis = getattr(fig.layout, "yaxis", None)
        subplots_fig.update_xaxes(
            title_text=(x_axis.title.text if x_axis and x_axis.title else None),
            type=(x_axis.type if x_axis and x_axis.type else None),
            range=(x_axis.range if x_axis and x_axis.range else None),
            row=row,
            col=col,
        )
        subplots_fig.update_yaxes(
            title_text=(y_axis.title.text if y_axis and y_axis.title else None),
            type=(y_axis.type if y_axis and y_axis.type else None),
            range=(y_axis.range if y_axis and y_axis.range else None),
            row=row,
            col=col,
        )

    subplots_fig.update_layout(
        width=cols * base_width + 250,  # Add space for legend
        height=rows * base_height,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,  # push outside the plot
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )

    assert filename.endswith(".html"), f"Filename must end with .html, got {filename}"
    html_path = Path("plots") / f"{DATETIME_STR}_grid" / filename
    html_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {html_path}")
    pio.write_html(
        subplots_fig,
        file=html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )
    return html_path, subplots_fig


def visualize_fig(
    fig: go.Figure,
    server: viser.ViserServer | None = None,
    save_html: bool = False,
) -> None:
    """
    Visualize a plotly figure in a viser server or save as an HTML file.

    Args:
        fig: The plotly figure to visualize.
        server: The viser server to visualize the figure in. If None, the figure will not be visualized.
        save_html: Whether to save the figure as an HTML file.
    """
    if server is not None:
        server.gui.add_plotly(figure=fig, aspect=0.5)

    if save_html:
        fig_title = fig.layout.title.text.replace(" ", "_").lower()
        output_path = Path("plots") / f"{DATETIME_STR}_individual" / f"{fig_title}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {output_path}")
        fig.write_html(output_path)


def plot_per_idx_comparison(
    name_to_y: Dict[str, np.ndarray],
    title: str,
    yaxis_title: str,
    y_names: List[str] | None = None,
) -> go.Figure:
    """
    Plot a comparison of values at each index for each name.
    X axis is indices and Y axis is values.
    Different colors for each name.

    Args:
        name_to_y: A dictionary mapping names to numpy arrays of values.
        title: The title of the figure.
        yaxis_title: The title of the y axis.
        y_names: The names of the y values. If None, will be generated.
    """
    names = list(name_to_y.keys())
    n_names = len(names)
    length = len(next(iter(name_to_y.values())))
    for name, y in name_to_y.items():
        assert y.shape == (length,), (
            f"Expected y for {name} to have shape (length,), got {y.shape} for length {length}"
        )

    if y_names is None:
        y_names = [f"y_{i}" for i in range(n_names)]
    assert len(y_names) == length, (
        f"Expected y_names to have length {length}, got {len(y_names)}"
    )

    fig = go.Figure()
    offsets = np.linspace(-0.2, 0.2, n_names)
    for i, name in enumerate(names):
        y = name_to_y[name]
        # x-position: the index for the point + small offset per file for visibility
        offset = offsets[i]
        x = np.arange(length) + offset
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(size=8),
            )
        )

    min_y = np.min(np.stack([y for y in name_to_y.values()]))
    max_y = np.max(np.stack([y for y in name_to_y.values()]))

    # Add text annotations for each y_name
    for idx, y_name in enumerate(y_names):
        # Text above line midpoint
        text_x = idx
        above_y = interpolate(init=min_y, final=max_y, alpha=0.8)
        slightly_above_y = interpolate(init=min_y, final=max_y, alpha=0.6)
        slightly_below_y = interpolate(init=min_y, final=max_y, alpha=0.4)
        below_y = interpolate(init=min_y, final=max_y, alpha=0.2)
        # Alternate above/below for visibility
        if idx % 4 == 0:
            text_y = above_y
        elif idx % 4 == 1:
            text_y = slightly_above_y
        elif idx % 4 == 2:
            text_y = slightly_below_y
        else:
            text_y = below_y

        fig.add_annotation(
            x=text_x,
            y=text_y,
            text=y_name,
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    # Dotted vertical lines between each idx
    for idx in range(length - 1):
        x_pos = idx + 0.5
        fig.add_shape(
            type="line",
            x0=x_pos,
            y0=min_y,
            x1=x_pos,
            y1=max_y,
            line=dict(color="black", width=1, dash="dot"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Index (offset per name)",
        yaxis_title=yaxis_title,
        legend_title="Name",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def interpolate(init, final, alpha: float) -> float:
    assert 0 <= alpha <= 1, f"alpha must be between 0 and 1, got {alpha}"
    return init + (final - init) * alpha


def plot_grid_of_values(
    title_to_name_to_x_y: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    grid_name: str,
) -> go.Figure:
    # e.g.,
    # {
    #     "Obs 0 iiwa7_joint_1": {
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [1, 1.1, 1.2]),
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [0.9, 0.91, 0.92]),
    #     },
    #     "Obs 1 iiwa7_joint_2": {
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [1, 1.1, 1.2]),
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [0.9, 0.91, 0.92]),
    #     },
    #     ...
    # }

    # Create figures
    figures = []
    for title, name_to_x_y in title_to_name_to_x_y.items():
        fig = go.Figure()

        for name, (x, y) in name_to_x_y.items():
            assert len(x) == len(y), (
                f"Expected x and y to have the same length, got {len(x)} and {len(y)} for {name}"
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=name,
                )
            )

        # Enforce min y axis range to make it clear when values are unchanging
        MIN_Y_AXIS_RANGE = 0.2

        all_y_values = [v for trace in fig.data for v in trace.y]
        if len(all_y_values) == 0:
            raise ValueError(f"No y values found for {title}")
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        y_range = y_max - y_min
        if y_range < MIN_Y_AXIS_RANGE:
            y_center = (y_min + y_max) / 2
            y_min_adjusted = y_center - MIN_Y_AXIS_RANGE / 2
            y_max_adjusted = y_center + MIN_Y_AXIS_RANGE / 2
            fig.update_yaxes(range=[y_min_adjusted, y_max_adjusted])

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Value",
            legend_title="Name",
        )

        figures.append(fig)

    # Visualize
    for fig in figures:
        visualize_fig(fig, save_html=True)

    figures_to_grid_html(
        figures, filename=f"values_over_time_{grid_name}.html", one_legend_per_fig=False
    )


def plot_values_at_one_time_index(
    filename_to_recorded_data: Dict[str, RecordedData],
    time_index: int,
) -> go.Figure:
    any_recorded_data = next(iter(filename_to_recorded_data.values()))
    obs_fig = plot_per_idx_comparison(
        name_to_y={
            filename: recorded_data.observations_array[time_index, :]
            for filename, recorded_data in filename_to_recorded_data.items()
        },
        title=f"Observations at time index {time_index}",
        yaxis_title="Obs Value at Time Index",
        y_names=any_recorded_data.observation_names,
    )
    action_fig = plot_per_idx_comparison(
        name_to_y={
            filename: recorded_data.actions_array[time_index, :]
            for filename, recorded_data in filename_to_recorded_data.items()
        },
        title=f"Actions at time index {time_index}",
        yaxis_title="Action Value at Time Index",
        y_names=any_recorded_data.action_names,
    )
    joint_pos_target_fig = plot_per_idx_comparison(
        name_to_y={
            filename: recorded_data.robot_joint_pos_targets_array[time_index, :]
            for filename, recorded_data in filename_to_recorded_data.items()
        },
        title=f"Joint Pos Targets at time index {time_index}",
        yaxis_title="Joint Pos Target Value at Time Index",
        y_names=any_recorded_data.robot_joint_names,
    )
    joint_position_fig = plot_per_idx_comparison(
        name_to_y={
            filename: recorded_data.robot_joint_positions_array[time_index, :]
            for filename, recorded_data in filename_to_recorded_data.items()
        },
        title=f"Joint Positions at time index {time_index}",
        yaxis_title="Joint Position Value at Time Index",
        y_names=any_recorded_data.robot_joint_names,
    )
    one_time_index_figs = [
        obs_fig,
        action_fig,
        joint_pos_target_fig,
        joint_position_fig,
    ]
    for fig in one_time_index_figs:
        visualize_fig(fig, save_html=True)
    return


def plot_values_over_time(
    filename_to_recorded_data: Dict[str, RecordedData],
) -> go.Figure:
    any_recorded_data = next(iter(filename_to_recorded_data.values()))
    # e.g.,
    # {
    #     "Obs 0 iiwa7_joint_1": {
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [1, 1.1, 1.2]),
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [0.9, 0.91, 0.92]),
    #     },
    #     "Obs 1 iiwa7_joint_2": {
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [1, 1.1, 1.2]),
    #         "2025-10-20_14-32-39_None_310.npz": ([0, 0.1, 0.2], [0.9, 0.91, 0.92]),
    #     },
    #     ...
    # }
    OBSERVATION_title_to_name_to_x_y = {
        f"Obs_{i}_{obs_name}": {
            filename: (recorded_data.time_array, recorded_data.observations_array[:, i])
            for filename, recorded_data in filename_to_recorded_data.items()
        }
        for i, obs_name in enumerate(any_recorded_data.observation_names)
    }

    ACTION_title_to_name_to_x_y = {
        f"Action_{i}_{action_name}": {
            filename: (recorded_data.time_array, recorded_data.actions_array[:, i])
            for filename, recorded_data in filename_to_recorded_data.items()
        }
        for i, action_name in enumerate(any_recorded_data.action_names)
    }
    JOINT_POS_TARGET_title_to_name_to_x_y = {
        f"Joint_Pos_Target_{i}_{joint_pos_target_name}": {
            filename: (
                recorded_data.time_array,
                recorded_data.robot_joint_pos_targets_array[:, i],
            )
            for filename, recorded_data in filename_to_recorded_data.items()
        }
        for i, joint_pos_target_name in enumerate(any_recorded_data.robot_joint_names)
    }
    JOINT_POSITION_title_to_name_to_x_y = {
        f"Joint_Position_{i}_{joint_position_name}": {
            filename: (
                recorded_data.time_array,
                recorded_data.robot_joint_positions_array[:, i],
            )
            for filename, recorded_data in filename_to_recorded_data.items()
        }
        for i, joint_position_name in enumerate(any_recorded_data.robot_joint_names)
    }
    JOINT_POSITION_AND_TARGET_title_to_name_to_x_y = {
        f"Joint_Pos_and_Target_{i}_{joint_position_and_target_name}": {
            **{
                f"{filename}_Pos": (
                    recorded_data.time_array,
                    recorded_data.robot_joint_positions_array[:, i],
                )
                for filename, recorded_data in filename_to_recorded_data.items()
            },
            **{
                f"{filename}_Target": (
                    recorded_data.time_array,
                    recorded_data.robot_joint_pos_targets_array[:, i],
                )
                for filename, recorded_data in filename_to_recorded_data.items()
            },
        }
        for i, joint_position_and_target_name in enumerate(
            any_recorded_data.robot_joint_names
        )
    }
    JOINT_VELOCITY_title_to_name_to_x_y = {
        f"Joint_Velocity_{i}_{joint_velocity_name}": {
            filename: (
                recorded_data.time_array,
                recorded_data.robot_joint_velocities_array[:, i],
            )
            for filename, recorded_data in filename_to_recorded_data.items()
        }
        for i, joint_velocity_name in enumerate(any_recorded_data.robot_joint_names)
    }
    JOINT_VELOCITY_AND_FD_title_to_name_to_x_y = {
        f"Joint_Vel_and_FD_{i}_{joint_velocity_and_fd_name}": {
            **{
                f"{filename}_Vel": (
                    recorded_data.time_array,
                    recorded_data.robot_joint_velocities_array[:, i],
                )
                for filename, recorded_data in filename_to_recorded_data.items()
            },
            **{
                f"{filename}_Vel_FD1": (
                    recorded_data.time_array,
                    recorded_data.robot_joint_velocities_array_fd1[:, i],
                )
                for filename, recorded_data in filename_to_recorded_data.items()
            },
            **{
                f"{filename}_Vel_FD2": (
                    recorded_data.time_array,
                    recorded_data.robot_joint_velocities_array_fd2[:, i],
                )
                for filename, recorded_data in filename_to_recorded_data.items()
            },
        }
        for i, joint_velocity_and_fd_name in enumerate(
            any_recorded_data.robot_joint_names
        )
    }

    for name, title_to_name_to_x_y in [
        ("Observations", OBSERVATION_title_to_name_to_x_y),
        ("Actions", ACTION_title_to_name_to_x_y),
        ("Joint_Pos_Targets", JOINT_POS_TARGET_title_to_name_to_x_y),
        ("Joint_Positions", JOINT_POSITION_title_to_name_to_x_y),
        ("Joint_Pos_and_Targets", JOINT_POSITION_AND_TARGET_title_to_name_to_x_y),
        ("Joint_Velocities", JOINT_VELOCITY_title_to_name_to_x_y),
        ("Joint_Vel_and_FD", JOINT_VELOCITY_AND_FD_title_to_name_to_x_y),
    ]:
        plot_grid_of_values(title_to_name_to_x_y=title_to_name_to_x_y, grid_name=name)

    return


def main():
    filepaths = [
        # Path("/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"),
        # Path(
        #     "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39_None_310.npz"
        # ),
        # Path("/home/tylerlum/github_repos/sapg/recorded_data/2025-11-06_17-09-47.npz"),  # Slow
        Path("/home/tylerlum/github_repos/sapg/recorded_data/2025-11-06_17-09-47_None_550.npz"),  # Slow sliced
    ]
    for filepath in filepaths:
        assert filepath.exists(), f"File {filepath} does not exist"
    filename_to_recorded_data = {
        filepath.stem: RecordedData.from_file(filepath) for filepath in filepaths
    }

    PLOT_VALUES_AT_ONE_TIME_INDEX = True
    if PLOT_VALUES_AT_ONE_TIME_INDEX:
        plot_values_at_one_time_index(filename_to_recorded_data, time_index=0)

    print("=" * 100)
    PLOT_VALUES_OVER_TIME = True
    if PLOT_VALUES_OVER_TIME:
        plot_values_over_time(filename_to_recorded_data)


if __name__ == "__main__":
    main()
