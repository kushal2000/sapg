"""Isaac Gym backend — wraps the existing gymapi/gymtorch calls.

Isaac Gym is natively vectorized on the GPU: a single ``gym.create_sim()``
runs N environments in parallel, so no additional vectorization is needed.
All state queries return ``(N, ...)`` GPU tensors directly.
"""

from typing import Optional

import torch

from train.sapg.envs.backends.base_backend import SimBackend


class IsaacGymBackend(SimBackend):
    """Wraps Isaac Gym (``gymapi`` / ``gymtorch``) behind the :class:`SimBackend` interface.

    This backend delegates asset loading, env creation, and the full
    environment setup to the existing ``AllegroKukaBase`` code.  It is
    designed to be used *alongside* the original env class during the
    migration period and can also be used as a standalone backend once
    the task logic is fully in ``DexManipEnv``.
    """

    def __init__(self) -> None:
        self._gym = None
        self._sim = None
        self._device: str = "cpu"
        self._dt: float = 0.0
        self._num_envs: int = 0
        self._num_dofs: int = 0
        self._num_bodies: int = 0
        self._num_actors: int = 0

        # Wrapped gymtorch tensor views (set in create())
        self._dof_state = None
        self._rigid_body_states = None
        self._root_state_tensor = None

        # Per-env handles
        self._envs: list = []
        self._depth_camera_handles: list = []
        self._depth_image_buf: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, cfg: dict, num_envs: int, device: str) -> None:
        from isaacgym import gymapi, gymtorch

        self._device = device
        self._num_envs = num_envs

        # Acquire gym instance
        self._gym = gymapi.acquire_gym()

        # Parse sim params
        sim_cfg = cfg["sim"]
        sim_params = self._parse_sim_params(cfg.get("physics_engine", "physx"), sim_cfg)
        self._dt = sim_params.dt

        physics_engine = (
            gymapi.SIM_PHYSX
            if cfg.get("physics_engine", "physx") == "physx"
            else gymapi.SIM_FLEX
        )

        # Device parsing
        split = device.split(":")
        device_type = split[0]
        device_id = int(split[1]) if len(split) > 1 else 0

        graphics_device_id = cfg.get("graphics_device_id", device_id)
        headless = cfg.get("headless", True)
        enable_camera = cfg.get("env", {}).get("enableCameraSensors", False)
        if not enable_camera and headless:
            graphics_device_id = -1

        self._sim = self._gym.create_sim(
            device_id, graphics_device_id, physics_engine, sim_params
        )
        if self._sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self._gym.add_ground(self._sim, plane_params)

        # Env creation is delegated to the caller (the task env) because it
        # involves task-specific asset loading.  The backend exposes
        # ``gym`` and ``sim`` so the task can call create_actor() etc.
        # After env creation, the task should call ``finalize()`` so the
        # backend can wrap GPU tensors.

    def finalize(self) -> None:
        """Call after all envs and actors have been created.

        Wraps GPU tensor views for fast state queries.
        """
        from isaacgym import gymtorch

        self._gym.prepare_sim(self._sim)

        # Wrap GPU tensors
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        rb_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        self._rigid_body_states_raw = gymtorch.wrap_tensor(rb_tensor)

        root_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        self._root_state_tensor_raw = gymtorch.wrap_tensor(root_tensor)

        # Compute per-env shapes
        total_dofs = self._dof_state.shape[0]
        self._num_dofs = total_dofs // self._num_envs

        total_bodies = self._rigid_body_states_raw.shape[0]
        self._num_bodies = total_bodies // self._num_envs

        total_actors = self._root_state_tensor_raw.shape[0]
        self._num_actors = total_actors // self._num_envs

        # Reshaped views
        self._dof_state_view = self._dof_state.view(self._num_envs, -1, 2)
        self._rigid_body_states = self._rigid_body_states_raw.view(
            self._num_envs, self._num_bodies, 13
        )
        self._root_state_tensor = self._root_state_tensor_raw.view(
            self._num_envs, self._num_actors, 13
        )

    def step(self) -> None:
        self._gym.simulate(self._sim)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        # Resetting is handled by the task env via set_dof_states / set_root_states
        pass

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_dof_positions(self) -> torch.Tensor:
        return self._dof_state_view[..., 0]

    def get_dof_velocities(self) -> torch.Tensor:
        return self._dof_state_view[..., 1]

    def get_rigid_body_states(self) -> torch.Tensor:
        return self._rigid_body_states

    def get_root_states(self) -> torch.Tensor:
        return self._root_state_tensor

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_dof_position_targets(self, targets: torch.Tensor) -> None:
        from isaacgym import gymtorch

        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(targets)
        )

    def set_dof_states(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        from isaacgym import gymtorch

        # Write into the wrapped tensor, then push indexed
        self._dof_state_view[env_ids, :, 0] = positions
        self._dof_state_view[env_ids, :, 1] = velocities
        actor_indices = env_ids.to(torch.int32)
        self._gym.set_dof_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(env_ids),
        )

    def set_root_states(
        self, states: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        from isaacgym import gymtorch

        self._root_state_tensor[env_ids] = states
        flat_root = self._root_state_tensor.view(-1, 13)
        # Compute flat actor indices
        actor_indices = (
            env_ids.unsqueeze(1) * self._num_actors
            + torch.arange(self._num_actors, device=env_ids.device).unsqueeze(0)
        ).flatten().to(torch.int32)
        unique_indices = torch.unique(actor_indices)
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(flat_root),
            gymtorch.unwrap_tensor(unique_indices),
            len(unique_indices),
        )

    # ------------------------------------------------------------------
    # Rendering / Depth
    # ------------------------------------------------------------------

    def setup_depth_cameras(self, cfg: dict) -> None:
        from isaacgym import gymapi

        env_cfg = cfg.get("env", cfg)
        img_size = env_cfg.get("depthImageSize", 224)
        mount_type = env_cfg.get("depthCameraMount", "fixed")

        cam_props = gymapi.CameraProperties()
        cam_props.width = img_size
        cam_props.height = img_size
        cam_props.enable_tensors = True

        self._depth_camera_handles = []
        for i, env in enumerate(self._envs):
            cam_handle = self._gym.create_camera_sensor(env, cam_props)

            if mount_type == "fixed":
                pos = env_cfg.get("depthCameraPos", [0.5, 0.0, 0.8])
                target = env_cfg.get("depthCameraTarget", [0.0, 0.0, 0.5])
                self._gym.set_camera_location(
                    cam_handle,
                    env,
                    gymapi.Vec3(*pos),
                    gymapi.Vec3(*target),
                )
            elif mount_type == "wrist":
                wrist_body = env_cfg.get("depthCameraWristBody", "iiwa14_link_7")
                actor_handle = self._gym.get_actor_handle(env, 0)
                body_handle = self._gym.find_actor_rigid_body_handle(
                    env, actor_handle, wrist_body
                )
                local_transform = gymapi.Transform()
                local_transform.p = gymapi.Vec3(0.05, 0.0, 0.0)
                self._gym.attach_camera_to_body(
                    cam_handle,
                    env,
                    body_handle,
                    local_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
            else:
                raise ValueError(f"Unknown depth camera mount: {mount_type}")

            self._depth_camera_handles.append(cam_handle)

        self._depth_image_size = img_size
        self._depth_image_buf = torch.zeros(
            (self._num_envs, 1, img_size, img_size),
            device=self._device,
            dtype=torch.float32,
        )
        print(
            f"[IsaacGymBackend] Initialized {len(self._depth_camera_handles)} "
            f"depth cameras ({mount_type}, {img_size}x{img_size})"
        )

    def get_depth_images(self) -> torch.Tensor:
        from isaacgym import gymapi, gymtorch

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        for i, env in enumerate(self._envs):
            depth_tensor = self._gym.get_camera_image_gpu_tensor(
                self._sim,
                env,
                self._depth_camera_handles[i],
                gymapi.IMAGE_DEPTH,
            )
            torch_depth = gymtorch.wrap_tensor(depth_tensor)
            self._depth_image_buf[i, 0] = -torch_depth.clamp(-10.0, 0.0)

        self._gym.end_access_image_tensors(self._sim)
        return self._depth_image_buf

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def apply_rigid_body_forces(
        self, forces: torch.Tensor, torques: torch.Tensor
    ) -> None:
        from isaacgym import gymtorch

        self._gym.apply_rigid_body_force_tensors(
            self._sim,
            gymtorch.unwrap_tensor(forces.view(-1, 3)),
            gymtorch.unwrap_tensor(torques.view(-1, 3)),
            0,  # gymapi.ENV_SPACE
        )

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_tensors(self) -> None:
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_actors(self) -> int:
        return self._num_actors

    @property
    def gym(self):
        """Direct access to the gymapi gym instance (for task-level setup)."""
        return self._gym

    @property
    def sim(self):
        """Direct access to the sim handle (for task-level setup)."""
        return self._sim

    @property
    def envs(self) -> list:
        return self._envs

    @envs.setter
    def envs(self, value: list) -> None:
        self._envs = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sim_params(physics_engine: str, config_sim: dict):
        from isaacgym import gymapi

        sim_params = gymapi.SimParams()

        if config_sim.get("up_axis", "z") == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim.get("use_gpu_pipeline", True)
        sim_params.substeps = config_sim.get("substeps", 2)
        sim_params.gravity = gymapi.Vec3(*config_sim.get("gravity", [0, 0, -9.81]))

        if physics_engine == "physx" and "physx" in config_sim:
            for opt, val in config_sim["physx"].items():
                if opt == "contact_collection":
                    setattr(
                        sim_params.physx,
                        opt,
                        gymapi.ContactCollection(val),
                    )
                else:
                    setattr(sim_params.physx, opt, val)
        elif physics_engine == "flex" and "flex" in config_sim:
            for opt, val in config_sim["flex"].items():
                setattr(sim_params.flex, opt, val)

        return sim_params
