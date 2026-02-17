"""Abstract interface that each simulator backend must implement.

The backend handles all simulator-specific concerns: environment creation,
physics stepping, state queries, control application, and depth rendering.
All tensor outputs must be on ``self.device`` and have a leading batch
dimension of ``self.num_envs``.
"""

from abc import ABC, abstractmethod

import torch


class SimBackend(ABC):
    """Interface that each simulator must implement."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def create(self, cfg: dict, num_envs: int, device: str) -> None:
        """Create the sim, load assets, and build ``num_envs`` environments.

        After this call, :pyattr:`num_envs`, :pyattr:`device`, and
        :pyattr:`dt` must be valid.
        """

    @abstractmethod
    def step(self) -> None:
        """Advance physics by one simulation step."""

    @abstractmethod
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset the environments at the given indices.

        The backend is only responsible for resetting the *simulator state*
        (body poses, dof states).  Task-level resets (goals, counters) are
        handled by the env.
        """

    # ------------------------------------------------------------------
    # State queries – all return tensors on self.device
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dof_positions(self) -> torch.Tensor:
        """Return joint positions, shape ``(N, num_dofs)``."""

    @abstractmethod
    def get_dof_velocities(self) -> torch.Tensor:
        """Return joint velocities, shape ``(N, num_dofs)``."""

    @abstractmethod
    def get_rigid_body_states(self) -> torch.Tensor:
        """Return rigid-body states, shape ``(N, num_bodies, 13)``.

        Each 13-vector is ``[pos(3), quat(4), linvel(3), angvel(3)]``.
        """

    @abstractmethod
    def get_root_states(self) -> torch.Tensor:
        """Return actor root states, shape ``(N, num_actors, 13)``."""

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    @abstractmethod
    def set_dof_position_targets(self, targets: torch.Tensor) -> None:
        """Set PD position targets for all DOFs, shape ``(N, num_dofs)``."""

    @abstractmethod
    def set_dof_states(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Directly set DOF positions and velocities for ``env_ids``."""

    @abstractmethod
    def set_root_states(
        self, states: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        """Directly set actor root states for ``env_ids``.

        ``states`` has shape ``(len(env_ids), num_actors, 13)``.
        """

    # ------------------------------------------------------------------
    # Rendering / Depth
    # ------------------------------------------------------------------

    @abstractmethod
    def setup_depth_cameras(self, cfg: dict) -> None:
        """Create depth-camera sensors in each environment."""

    @abstractmethod
    def get_depth_images(self) -> torch.Tensor:
        """Render and return depth images, shape ``(N, 1, H, W)``."""

    # ------------------------------------------------------------------
    # Force / torque application
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_rigid_body_forces(
        self, forces: torch.Tensor, torques: torch.Tensor
    ) -> None:
        """Apply external forces and torques to rigid bodies.

        ``forces``/``torques`` have shape ``(N, num_bodies, 3)``.
        """

    # ------------------------------------------------------------------
    # Refresh (Isaac Gym needs explicit refresh calls after stepping)
    # ------------------------------------------------------------------

    @abstractmethod
    def refresh_tensors(self) -> None:
        """Refresh any cached GPU tensor views after a sim step.

        For Isaac Gym this calls ``refresh_*_tensor()``.  Other backends
        can no-op.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation timestep in seconds."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Torch device string, e.g. ``'cuda:0'`` or ``'cpu'``."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""

    @property
    @abstractmethod
    def num_dofs(self) -> int:
        """Number of DOFs per environment."""

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of rigid bodies per environment."""

    @property
    @abstractmethod
    def num_actors(self) -> int:
        """Number of actors per environment."""
