"""MuJoCo backend — runs N independent MjData instances on the CPU.

This backend loads an MJCF scene (IIWA arm + Sharpa hand + objects)
and batches N independent simulations by stacking their states into
``(N, ...)`` torch tensors.

Reference implementation:
    ``sim2sim/mujoco_sim/mujoco_sim_sharpa.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from train.sapg.envs.backends.base_backend import SimBackend


class MuJoCoBackend(SimBackend):
    """MuJoCo backend using ``mujoco`` Python bindings.

    Each of the ``num_envs`` environments runs its own ``mujoco.MjData``
    instance against a shared ``mujoco.MjModel``.
    """

    def __init__(self) -> None:
        self._device_str: str = "cpu"
        self._dt_val: float = 0.001
        self._num_envs_val: int = 0
        self._num_dofs_val: int = 0
        self._num_bodies_val: int = 0
        self._num_actors_val: int = 0

        self._model = None  # mujoco.MjModel (shared)
        self._datas: List = []  # list[mujoco.MjData]
        self._renderer = None

        self._depth_image_size: int = 224
        self._depth_image_buf: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, cfg: dict, num_envs: int, device: str) -> None:
        import mujoco

        self._device_str = device  # MuJoCo always runs on CPU
        self._num_envs_val = num_envs

        sim_cfg = cfg.get("sim", {})
        self._dt_val = sim_cfg.get("dt", 0.001)

        mjcf_path = sim_cfg.get("mjcf_path", None)
        if mjcf_path is None:
            # Default: build IIWA + Sharpa scene (like mujoco_sim_sharpa.py)
            asset_root = Path(__file__).resolve().parents[3] / "assets" / "mjcf"
            mjcf_path = str(asset_root / "kuka_iiwa_14" / "scene.xml")

        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._model.opt.timestep = self._dt_val

        # Create N independent data instances
        self._datas = [mujoco.MjData(self._model) for _ in range(num_envs)]

        self._num_dofs_val = self._model.nq  # generalized positions
        self._num_bodies_val = self._model.nbody
        # MuJoCo doesn't have the same actor concept; we treat the
        # whole model as one actor per env.
        self._num_actors_val = 1

        # Initialize all envs
        for d in self._datas:
            mujoco.mj_resetData(self._model, d)
            mujoco.mj_forward(self._model, d)

        print(
            f"[MuJoCoBackend] Created {num_envs} envs, "
            f"{self._num_dofs_val} qpos, {self._num_bodies_val} bodies"
        )

    def step(self) -> None:
        import mujoco

        for d in self._datas:
            mujoco.mj_step(self._model, d)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        import mujoco

        for idx in env_ids.cpu().tolist():
            mujoco.mj_resetData(self._model, self._datas[idx])
            mujoco.mj_forward(self._model, self._datas[idx])

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_dof_positions(self) -> torch.Tensor:
        positions = np.stack([d.qpos.copy() for d in self._datas], axis=0)
        return torch.from_numpy(positions).float().to(self._device_str)

    def get_dof_velocities(self) -> torch.Tensor:
        velocities = np.stack([d.qvel.copy() for d in self._datas], axis=0)
        return torch.from_numpy(velocities).float().to(self._device_str)

    def get_rigid_body_states(self) -> torch.Tensor:
        """Return ``(N, num_bodies, 13)`` — pos(3), quat(4), linvel(3), angvel(3)."""
        N = self._num_envs_val
        B = self._num_bodies_val
        states = np.zeros((N, B, 13), dtype=np.float32)
        for i, d in enumerate(self._datas):
            states[i, :, 0:3] = d.xpos[:B]
            states[i, :, 3:7] = d.xquat[:B]  # wxyz in MuJoCo
            # Body velocities need to be computed via mj_objectVelocity
            # For simplicity, we approximate with zeros (can be extended)
        return torch.from_numpy(states).to(self._device_str)

    def get_root_states(self) -> torch.Tensor:
        """Return ``(N, 1, 13)`` — single root actor per env."""
        N = self._num_envs_val
        states = np.zeros((N, 1, 13), dtype=np.float32)
        for i, d in enumerate(self._datas):
            # Root body (body 0 is worldbody, body 1 is first mobile body)
            if self._model.nbody > 1:
                states[i, 0, 0:3] = d.xpos[1]
                states[i, 0, 3:7] = d.xquat[1]
        return torch.from_numpy(states).to(self._device_str)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_dof_position_targets(self, targets: torch.Tensor) -> None:
        targets_np = targets.cpu().numpy()
        for i, d in enumerate(self._datas):
            d.ctrl[:] = targets_np[i, : len(d.ctrl)]

    def set_dof_states(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        pos_np = positions.cpu().numpy()
        vel_np = velocities.cpu().numpy()
        for j, idx in enumerate(env_ids.cpu().tolist()):
            self._datas[idx].qpos[:] = pos_np[j, : len(self._datas[idx].qpos)]
            self._datas[idx].qvel[:] = vel_np[j, : len(self._datas[idx].qvel)]

    def set_root_states(
        self, states: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        # MuJoCo doesn't have a direct "root state" API like Isaac Gym;
        # free joints can be set via qpos of the free joint.
        pass

    # ------------------------------------------------------------------
    # Rendering / Depth
    # ------------------------------------------------------------------

    def setup_depth_cameras(self, cfg: dict) -> None:
        import mujoco

        env_cfg = cfg.get("env", cfg)
        self._depth_image_size = env_cfg.get("depthImageSize", 224)

        self._renderer = mujoco.Renderer(
            self._model,
            height=self._depth_image_size,
            width=self._depth_image_size,
        )

        self._depth_image_buf = torch.zeros(
            (self._num_envs_val, 1, self._depth_image_size, self._depth_image_size),
            device=self._device_str,
            dtype=torch.float32,
        )
        print(
            f"[MuJoCoBackend] Setup depth cameras: "
            f"{self._depth_image_size}x{self._depth_image_size}"
        )

    def get_depth_images(self) -> torch.Tensor:
        import mujoco

        for i, d in enumerate(self._datas):
            self._renderer.update_scene(d)
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            self._depth_image_buf[i, 0] = torch.from_numpy(depth.copy()).float()

        return self._depth_image_buf

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def apply_rigid_body_forces(
        self, forces: torch.Tensor, torques: torch.Tensor
    ) -> None:
        forces_np = forces.cpu().numpy()
        torques_np = torques.cpu().numpy()
        for i, d in enumerate(self._datas):
            # xfrc_applied has shape (nbody, 6): [torque(3), force(3)]
            B = min(forces_np.shape[1], d.xfrc_applied.shape[0])
            d.xfrc_applied[:B, 3:6] = forces_np[i, :B]
            d.xfrc_applied[:B, 0:3] = torques_np[i, :B]

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_tensors(self) -> None:
        import mujoco

        for d in self._datas:
            mujoco.mj_forward(self._model, d)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._dt_val

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def num_envs(self) -> int:
        return self._num_envs_val

    @property
    def num_dofs(self) -> int:
        return self._num_dofs_val

    @property
    def num_bodies(self) -> int:
        return self._num_bodies_val

    @property
    def num_actors(self) -> int:
        return self._num_actors_val

    @property
    def model(self):
        """Direct access to the shared MjModel."""
        return self._model

    @property
    def datas(self) -> list:
        """Direct access to the list of MjData instances."""
        return self._datas
