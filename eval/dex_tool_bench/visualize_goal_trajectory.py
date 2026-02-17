import json
import time
from pathlib import Path

import numpy as np
import viser
from termcolor import colored
from viser.extras import ViserUrdf

from isaacgymenvs.utils.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


class GoalTrajectoryVisualizer:
    def __init__(self, trajectory_data, object_name, port=8080):
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.start_pose = np.array(trajectory_data["start_pose"])
        self.goals = [np.array(g) for g in trajectory_data["goals"]]
        self._setup_scene(object_name)

    def _setup_scene(self, object_name):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.5, 1.5)
            client.camera.look_at = (0.0, -0.5, 0.7)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        table_urdf = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
        self.server.scene.add_frame("/table", position=(0, 0, 0.38), wxyz=(1, 0, 0, 0), show_axes=False)
        ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 0.5))

        object_urdf = NAME_TO_OBJECT[object_name].filepath
        self.start_frame = self.server.scene.add_frame(
            "/start", position=self.start_pose[:3], wxyz=quat_xyzw_to_wxyz(self.start_pose[3:7]),
            show_axes=True, axes_length=0.1, axes_radius=0.002,
        )
        ViserUrdf(self.server, object_urdf, root_node_name="/start")

        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=True, axes_length=0.1, axes_radius=0.002)
        ViserUrdf(self.server, object_urdf, root_node_name="/goal", mesh_color_override=(0, 255, 0, 0.5))

        self.server.gui.add_markdown(f"**{object_name}** - {len(self.goals)} goals")
        self.slider = self.server.gui.add_slider("Goal", min=0, max=len(self.goals) - 1, step=1, initial_value=0)
        self.slider.on_update(lambda _: self._update_goal())
        self._update_goal()

    def _update_goal(self):
        pose = self.goals[int(self.slider.value)]
        self.goal_frame.position = pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(pose[3:7])

    def run(self):
        print(colored(f"Viser: http://localhost:{self.port}", "green"))
        while True:
            time.sleep(1.0)


def main():
    object_type = "hammer"
    object_name = "scanned_hammer_2"
    trajectory_name = "horizontal_swing"
    trajectory_path = get_repo_root_dir() / "dex_tool_bench/evaluation_trajectories" / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory file does not exist: {trajectory_path}"
    with open(trajectory_path) as f:
        GoalTrajectoryVisualizer(json.load(f), object_name).run()


if __name__ == "__main__":
    main()

