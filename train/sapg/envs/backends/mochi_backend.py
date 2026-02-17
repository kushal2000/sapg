"""Mochi backend — uses HybridVectorEnv for multi-process vectorization.

Mochi environments are CPU-based and vectorized via
``mochi_gym.utils.vector.HybridVectorEnv``, which distributes envs
across async workers where each worker runs multiple envs sequentially
(allowing scene sharing).

Default robot: FrankaMetahandEnv (Franka arm + Metahand).
The observation space is ``{"pose": (num_dofs,), "vel": (num_dofs,)}``.

Reference files:
    /juno/u/satvik/mochi/mochi_gym/mochi_gym/envs/mochi_env.py
    /juno/u/satvik/mochi/mochi_gym/mochi_gym/envs/robots/franka_metahand_env.py
    /juno/u/satvik/mochi/mochi_gym/mochi_gym/utils/vector.py
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np
import torch

from train.sapg.envs.backends.base_backend import SimBackend


class MochiBackend(SimBackend):
    """Mochi simulator backend using ``HybridVectorEnv``.

    The backend wraps N Mochi gym environments via the HybridVectorEnv
    (async workers × sync envs-per-worker).  State queries aggregate
    individual env observations into ``(N, ...)`` tensors.
    """

    def __init__(self) -> None:
        self._device_str: str = "cpu"
        self._dt_val: float = 0.05  # 1/control_freq
        self._num_envs_val: int = 0
        self._num_dofs_val: int = 0
        self._num_bodies_val: int = 0
        self._num_actors_val: int = 1

        self._vec_env = None  # HybridVectorEnv
        self._latest_obs: Optional[np.ndarray] = None

        self._depth_image_buf: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, cfg: dict, num_envs: int, device: str) -> None:
        import os

        self._device_str = device  # Mochi runs on CPU
        self._num_envs_val = num_envs

        sim_cfg = cfg.get("sim", {})
        control_freq = sim_cfg.get("control_frequency", 20)
        sim_freq = sim_cfg.get("simulation_frequency", 100)
        self._dt_val = 1.0 / control_freq

        num_workers = sim_cfg.get("num_workers", 8)
        num_envs_per_worker = sim_cfg.get("num_envs_per_worker", max(1, num_envs // num_workers))

        # Set MOCHI_ASSETS_PATH if not already in env
        mochi_assets_path = sim_cfg.get("mochi_assets_path")
        if mochi_assets_path and "MOCHI_ASSETS_PATH" not in os.environ:
            os.environ["MOCHI_ASSETS_PATH"] = str(mochi_assets_path)

        # Try to import mochi_gym
        mochi_gym_path = sim_cfg.get(
            "mochi_gym_path", "/juno/u/satvik/mochi/mochi_gym"
        )
        if mochi_gym_path not in sys.path:
            sys.path.insert(0, mochi_gym_path)

        try:
            from mochi_gym.utils.vector import HybridVectorEnv
            from mochi_gym.envs.robots.franka_metahand_env import (
                FrankaMetahandEnv,
                FrankaMetahandEnvCfg,
            )
        except ImportError as e:
            raise ImportError(
                f"Cannot import mochi_gym. Ensure it is installed or "
                f"set sim.mochi_gym_path in config. Error: {e}"
            )

        # Build env factory functions using FrankaMetahandEnv
        # (Franka arm + Metahand / Allegro-style hand)
        env_cfg = FrankaMetahandEnvCfg(
            control_frequency=control_freq,
            simulation_frequency=sim_freq,
            steps_per_episode=cfg.get("env", {}).get("episodeLength", 1000),
        )

        def make_env_fn():
            return FrankaMetahandEnv(env_cfg)

        env_fns = [make_env_fn for _ in range(num_envs)]

        self._vec_env = HybridVectorEnv(
            env_fns=env_fns,
            num_envs_per_worker=num_envs_per_worker,
        )

        # Infer dimensions from a single env's spaces
        sample_obs_space = self._vec_env.single_observation_space
        sample_act_space = self._vec_env.single_action_space

        # Flatten obs to get total dim
        if hasattr(sample_obs_space, "shape"):
            obs_dim = sample_obs_space.shape[0]
        else:
            obs_dim = sum(
                np.prod(v.shape) for v in sample_obs_space.spaces.values()
            )

        self._num_dofs_val = sample_act_space.shape[0] if hasattr(sample_act_space, "shape") else 29
        self._num_bodies_val = 1  # Mochi doesn't expose body-level state directly
        self._num_actors_val = 1

        # Initial reset
        obs, _info = self._vec_env.reset()
        self._latest_obs = obs

        print(
            f"[MochiBackend] Created {num_envs} envs via HybridVectorEnv "
            f"({num_workers} workers × {num_envs_per_worker} envs/worker), "
            f"obs_dim={obs_dim}, act_dim={self._num_dofs_val}"
        )

    def step(self) -> None:
        # Mochi step is done in apply_action + internal simulate
        # The actual step is managed by the vec_env.step() call
        # which is triggered from the env's step method
        pass

    def step_with_actions(self, actions: np.ndarray) -> Tuple:
        """Step the vectorized Mochi environment.

        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        result = self._vec_env.step(actions)
        self._latest_obs = result[0]
        return result

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        # HybridVectorEnv handles auto-resets internally
        pass

    # ------------------------------------------------------------------
    # State queries — map mochi structured obs to our interface
    # ------------------------------------------------------------------

    def get_dof_positions(self) -> torch.Tensor:
        """Extract joint positions from latest observation.

        FrankaMetahandEnv returns ``{"pose": (num_dofs,), "vel": (num_dofs,)}``.
        """
        if self._latest_obs is None:
            return torch.zeros(
                (self._num_envs_val, self._num_dofs_val), device=self._device_str
            )
        obs = self._latest_obs
        if isinstance(obs, dict) and "pose" in obs:
            return torch.from_numpy(np.asarray(obs["pose"])).float().to(self._device_str)
        arr = np.asarray(obs)
        return torch.from_numpy(arr[:, : self._num_dofs_val]).float().to(self._device_str)

    def get_dof_velocities(self) -> torch.Tensor:
        if self._latest_obs is None:
            return torch.zeros(
                (self._num_envs_val, self._num_dofs_val), device=self._device_str
            )
        obs = self._latest_obs
        if isinstance(obs, dict) and "vel" in obs:
            return torch.from_numpy(np.asarray(obs["vel"])).float().to(self._device_str)
        return torch.zeros(
            (self._num_envs_val, self._num_dofs_val), device=self._device_str
        )

    def get_rigid_body_states(self) -> torch.Tensor:
        return torch.zeros(
            (self._num_envs_val, self._num_bodies_val, 13),
            device=self._device_str,
        )

    def get_root_states(self) -> torch.Tensor:
        return torch.zeros(
            (self._num_envs_val, self._num_actors_val, 13),
            device=self._device_str,
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_dof_position_targets(self, targets: torch.Tensor) -> None:
        # Targets are applied via step_with_actions
        pass

    def set_dof_states(
        self, positions: torch.Tensor, velocities: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        # Mochi doesn't support direct state setting through the gym API
        pass

    def set_root_states(
        self, states: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # Rendering / Depth
    # ------------------------------------------------------------------

    def setup_depth_cameras(self, cfg: dict) -> None:
        env_cfg = cfg.get("env", cfg)
        size = env_cfg.get("depthImageSize", 224)
        self._depth_image_buf = torch.zeros(
            (self._num_envs_val, 1, size, size),
            device=self._device_str,
            dtype=torch.float32,
        )
        print(
            f"[MochiBackend] Depth cameras setup ({size}x{size}). "
            f"Note: Mochi depth rendering is limited."
        )

    def get_depth_images(self) -> torch.Tensor:
        # Mochi rendering is done via mochi.viewer in RGB_ARRAY mode;
        # depth is not directly supported. Return zeros as placeholder.
        if self._depth_image_buf is None:
            raise RuntimeError("Call setup_depth_cameras first")
        return self._depth_image_buf

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def apply_rigid_body_forces(
        self, forces: torch.Tensor, torques: torch.Tensor
    ) -> None:
        # Not directly supported through the Mochi gym interface
        pass

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_tensors(self) -> None:
        # Mochi obs are already up-to-date after step
        pass

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
    def vec_env(self):
        """Direct access to the underlying HybridVectorEnv."""
        return self._vec_env

    def close(self) -> None:
        if self._vec_env is not None:
            self._vec_env.close()
