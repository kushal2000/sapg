import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from isaacgymenvs.utils.utils import get_repo_root_dir
from eval.sim2sim.mujoco_sim.mujoco_sim_sharpa import INIT_JOINT_POS, MujocoSim, MujocoSimConfig
from misc.viser_mujoco.viser_sim import ViserMujocoSim


def main():
    sim = MujocoSim(MujocoSimConfig(enable_viewer=False))
    server = viser.ViserServer()
    _viser_urdf_frame = server.scene.add_frame(
        "/robot",
        position=(0, 0.8, 0),
        wxyz=(1, 0, 0, 0),
    )
    urdf_path = (
        get_repo_root_dir() / "assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        root_node_name="/robot",
    )
    viser_urdf.update_cfg(INIT_JOINT_POS)
    viser_mujoco_sim = ViserMujocoSim(server, sim)
    viser_mujoco_sim.sim.sim_step()
    viser_mujoco_sim._update_viser()
    while True:
        print("Sleeping for 1.0 seconds")
        time.sleep(1.0)
        print(f"viser urdf qpos: {np.round(viser_urdf._urdf._cfg, 3)}")
        print(
            f"viser mujoco sim qpos: {np.round(viser_mujoco_sim.sim.get_sim_state()['joint_positions'], 3)}"
        )
        breakpoint()


if __name__ == "__main__":
    main()
