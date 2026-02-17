"""Simulator-agnostic dexterous manipulation environment.

This module contains ALL task logic (observation computation, rewards, resets,
goal sampling, curriculum) and delegates simulator-specific operations to a
:class:`SimBackend`.

Usage::

    from train.sapg.envs.backends.isaacgym_backend import IsaacGymBackend

    backend = IsaacGymBackend()
    env = DexManipEnv(cfg, backend)

    obs = env.reset()
    for _ in range(1000):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
from gym import spaces


class DexManipEnv:
    """Simulator-agnostic dexterous manipulation task.

    All task logic lives here.  The backend handles simulator-specific
    concerns (env creation, stepping, state queries, depth rendering).

    This class implements the interface that rl_games expects:
    ``step``, ``reset``, ``observation_space``, ``action_space``,
    ``num_envs``, ``get_env_info``.
    """

    def __init__(self, cfg: dict, backend: "SimBackend") -> None:
        from train.sapg.envs.backends.base_backend import SimBackend

        self.cfg = cfg
        self.backend: SimBackend = backend

        env_cfg = cfg["env"]

        # ---- Device / pipeline ----
        self.device = cfg.get("device", "cuda:0")
        self.rl_device = cfg.get("rl_device", self.device)

        # ---- Robot configuration ----
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_allegro_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_allegro_fingertips

        use_sharpa = env_cfg.get("asset", {}).get("kukaAllegro", "").find("sharpa") >= 0
        if use_sharpa:
            self.num_allegro_fingertips = 5
            self.num_hand_dofs = 22
            self.num_finger_dofs = None
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        # ---- Actions ----
        self.privileged_actions = env_cfg.get("privilegedActions", False)
        num_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            num_actions += 3

        # ---- Observations ----
        self.num_keypoints = 8  # default cube corners
        self.obs_type_size_dict = {
            "joint_pos": self.num_hand_arm_dofs,
            "joint_vel": self.num_hand_arm_dofs,
            "prev_action_targets": self.num_hand_arm_dofs,
            "palm_pos": 3,
            "palm_rot": 4,
            "palm_vel": 6,
            "object_rot": 4,
            "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * self.num_allegro_fingertips,
            "keypoints_rel_palm": 3 * self.num_keypoints,
            "keypoints_rel_goal": 3 * self.num_keypoints,
            "object_scales": 3,
            "closest_keypoint_max_dist": 1,
            "closest_fingertip_dist": self.num_allegro_fingertips,
            "lifted_object": 1,
            "progress": 1,
            "successes": 1,
            "reward": 1,
        }

        self.image_obs_types = {"depth_image"}
        self.state_list = env_cfg.get("stateList", [])
        self.obs_list = env_cfg.get("obsList", [])
        self.flat_obs_list = [k for k in self.obs_list if k not in self.image_obs_types]
        self.image_obs_list = [k for k in self.obs_list if k in self.image_obs_types]
        self.has_image_obs = len(self.image_obs_list) > 0
        self.use_depth = "depth_image" in self.obs_list

        self.full_state_size = sum(
            self.obs_type_size_dict[k] for k in self.state_list
        )
        self.flat_obs_size = sum(
            self.obs_type_size_dict[k] for k in self.flat_obs_list
        )
        self.num_observations = self.flat_obs_size
        self.num_states = self.full_state_size
        self.num_actions = num_actions

        # ---- Spaces (rl_games compatible) ----
        if self.has_image_obs:
            depth_size = env_cfg.get("depthImageSize", 224)
            self.obs_space = spaces.Dict({
                "proprio": spaces.Box(
                    -np.inf * np.ones(self.num_observations),
                    np.inf * np.ones(self.num_observations),
                ),
                "depth_image": spaces.Box(
                    low=0.0, high=10.0,
                    shape=(1, depth_size, depth_size),
                    dtype=np.float32,
                ),
            })
        else:
            self.obs_space = spaces.Box(
                -np.inf * np.ones(self.num_observations),
                np.inf * np.ones(self.num_observations),
            )
        self.state_space = spaces.Box(
            -np.inf * np.ones(self.num_states),
            np.inf * np.ones(self.num_states),
        )
        self.act_space = spaces.Box(
            -np.ones(self.num_actions), np.ones(self.num_actions)
        )

        # ---- Timing ----
        self.control_freq_inv = env_cfg.get("controlFrequencyInv", 1)
        self.max_episode_length = env_cfg.get("episodeLength", 600)

        # ---- Clipping ----
        self.clip_obs = env_cfg.get("clipObservations", np.inf)
        self.clip_actions = env_cfg.get("clipActions", np.inf)

        # ---- Reward scales (read from config) ----
        self.distance_delta_rew_scale = env_cfg.get("distanceDeltaRewScale", 50.0)
        self.lifting_rew_scale = env_cfg.get("liftingRewScale", 20.0)
        self.lifting_bonus = env_cfg.get("liftingBonus", 300.0)
        self.lifting_bonus_threshold = env_cfg.get("liftingBonusThreshold", 0.15)
        self.keypoint_rew_scale = env_cfg.get("keypointRewScale", 200.0)
        self.kuka_actions_penalty_scale = env_cfg.get("kukaActionsPenaltyScale", 0.003)
        self.allegro_actions_penalty_scale = env_cfg.get("allegroActionsPenaltyScale", 0.0003)
        self.object_lin_vel_penalty_scale = env_cfg.get("objectLinVelPenaltyScale", 0.0)
        self.object_ang_vel_penalty_scale = env_cfg.get("objectAngVelPenaltyScale", 0.0)
        self.reach_goal_bonus = env_cfg.get("reachGoalBonus", 1000.0)
        self.fall_dist = env_cfg.get("fallDistance", 0.24)
        self.fall_penalty = env_cfg.get("fallPenalty", 0.0)
        self.keypoint_scale = env_cfg.get("keypointScale", 1.5)

        # ---- Success / curriculum ----
        self.initial_tolerance = env_cfg.get("successTolerance", 0.075)
        self.target_tolerance = env_cfg.get("targetSuccessTolerance", 0.01)
        self.success_tolerance = self.initial_tolerance
        self.tolerance_curriculum_increment = env_cfg.get("toleranceCurriculumIncrement", 0.9)
        self.tolerance_curriculum_interval = env_cfg.get("toleranceCurriculumInterval", 3000)
        self.max_consecutive_successes = env_cfg.get("maxConsecutiveSuccesses", 50)
        self.success_steps = env_cfg.get("successSteps", 1)

        # ---- Counters ----
        self.total_train_env_frames: int = 0
        self.control_steps: int = 0

        # ---- Backend creates the sim ----
        self._num_envs = env_cfg.get("numEnvs", 8192)

        # NOTE: The backend.create() and env-creation are typically done
        # by the caller (e.g. the IsaacGym-specific wrapper that calls
        # _create_envs).  For the fully-decoupled path, the backend
        # handles everything.  This class allocates task-level buffers
        # *after* the backend is ready.

    # ------------------------------------------------------------------
    # Buffer allocation (called after backend is fully initialized)
    # ------------------------------------------------------------------

    def allocate_buffers(self) -> None:
        """Allocate task-level torch buffers.  Call after backend is ready."""
        N = self._num_envs
        d = self.device

        self.obs_buf = torch.zeros((N, self.num_observations), device=d, dtype=torch.float)
        self.states_buf = torch.zeros((N, self.num_states), device=d, dtype=torch.float)
        self.rew_buf = torch.zeros(N, device=d, dtype=torch.float)
        self.reset_buf = torch.ones(N, device=d, dtype=torch.long)
        self.timeout_buf = torch.zeros(N, device=d, dtype=torch.long)
        self.progress_buf = torch.zeros(N, device=d, dtype=torch.long)
        self.reset_goal_buf = torch.zeros(N, device=d, dtype=torch.long)

        self.successes = torch.zeros(N, device=d, dtype=torch.float)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.true_objective = torch.zeros(N, device=d, dtype=torch.float)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)
        self.near_goal_steps = torch.zeros(N, device=d, dtype=torch.long)

        self.prev_targets = torch.zeros(
            (N, self.num_hand_arm_dofs), device=d, dtype=torch.float
        )
        self.cur_targets = torch.zeros(
            (N, self.num_hand_arm_dofs), device=d, dtype=torch.float
        )

        self.extras: Dict[str, Any] = {}
        self.obs_dict: Dict[str, Any] = {}

        if self.use_depth:
            depth_size = self.cfg["env"].get("depthImageSize", 224)
            self.depth_obs = torch.zeros(
                (N, 1, depth_size, depth_size), device=d, dtype=torch.float
            )

    # ------------------------------------------------------------------
    # rl_games interface
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> gym.Space:
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        return self.act_space

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_obs(self) -> int:
        return self.num_observations

    @property
    def num_acts(self) -> int:
        return self.num_actions

    def get_env_info(self) -> dict:
        """Return env info dict expected by rl_games."""
        info = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
        }
        if self.num_states > 0:
            info["state_space"] = self.state_space
        return info

    def set_train_info(self, env_frames: int, *args, **kwargs) -> None:
        self.total_train_env_frames = env_frames

    def get_env_state(self):
        return None

    def set_env_state(self, env_state):
        pass

    # ------------------------------------------------------------------
    # Step / Reset
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment.

        Returns:
            (obs_dict, rewards, resets, extras)
        """
        action_tensor = torch.clamp(actions.to(self.device), -self.clip_actions, self.clip_actions)

        # Pre-physics: apply actions, handle resets
        self._pre_physics_step(action_tensor)

        # Step physics
        for _ in range(self.control_freq_inv):
            self.backend.step()

        # Post-physics: refresh state, compute reward, compute obs
        self.backend.refresh_tensors()
        self._post_physics_step()

        self.control_steps += 1

        # Timeout
        self.timeout_buf = (
            (self.progress_buf >= self.max_episode_length - 1)
            & (self.reset_buf != 0)
        )
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        # Build output obs dict
        return self._build_obs_dict(), self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset(self) -> Dict[str, torch.Tensor]:
        """Called once at the start to provide initial observations."""
        return self._build_obs_dict()

    def _build_obs_dict(self) -> Dict[str, torch.Tensor]:
        flat_obs = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        if self.has_image_obs:
            obs_out = {"proprio": flat_obs}
            if hasattr(self, "depth_obs"):
                obs_out["depth_image"] = self.depth_obs.to(self.rl_device)
            self.obs_dict["obs"] = obs_out
        else:
            self.obs_dict["obs"] = flat_obs

        if self.num_states > 0:
            self.obs_dict["states"] = torch.clamp(
                self.states_buf, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

        return self.obs_dict

    # ------------------------------------------------------------------
    # Pre-physics: action processing, resets
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions and handle pending resets.

        This method mirrors the logic from
        ``AllegroKukaBase.pre_physics_step``.  Subclasses can override
        to add task-specific behaviour.
        """
        # Subclass or the Isaac Gym wrapper implements the full
        # pre_physics_step logic (action smoothing, target computation,
        # reset handling, random forces, etc.)
        raise NotImplementedError(
            "_pre_physics_step must be implemented by the task-specific "
            "subclass or the migration wrapper."
        )

    # ------------------------------------------------------------------
    # Post-physics: state extraction, reward, observations
    # ------------------------------------------------------------------

    def _post_physics_step(self) -> None:
        """Compute reward and observations after physics step.

        This method mirrors ``AllegroKukaBase.post_physics_step``.
        """
        raise NotImplementedError(
            "_post_physics_step must be implemented by the task-specific "
            "subclass or the migration wrapper."
        )

    # ------------------------------------------------------------------
    # Observation computation (simulator-agnostic)
    # ------------------------------------------------------------------

    def compute_observations(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        prev_targets: torch.Tensor,
        palm_state: torch.Tensor,
        palm_center_pos: torch.Tensor,
        object_state: torch.Tensor,
        fingertip_pos_rel_palm: torch.Tensor,
        keypoints_rel_palm: torch.Tensor,
        keypoints_rel_goal: torch.Tensor,
        object_scales: torch.Tensor,
        closest_keypoint_max_dist: torch.Tensor,
        closest_fingertip_dist: torch.Tensor,
        lifted_object: torch.Tensor,
        dof_lower: torch.Tensor,
        dof_upper: torch.Tensor,
    ) -> None:
        """Fill ``obs_buf`` and ``states_buf`` from raw sim state.

        This implements the core of ``populate_obs_and_states_buffers``
        but is simulator-agnostic — it only operates on torch tensors.
        """
        N = dof_pos.shape[0]
        obs_dict: Dict[str, torch.Tensor] = {}

        # Unscale joint positions to [-1, 1]
        obs_dict["joint_pos"] = self._unscale(dof_pos, dof_lower, dof_upper)
        obs_dict["joint_vel"] = dof_vel
        obs_dict["prev_action_targets"] = prev_targets.clone()
        obs_dict["palm_pos"] = palm_center_pos
        obs_dict["palm_rot"] = palm_state[:, 3:7]
        obs_dict["palm_vel"] = palm_state[:, 7:13]
        obs_dict["object_rot"] = object_state[:, 3:7]
        obs_dict["object_vel"] = object_state[:, 7:13]
        obs_dict["fingertip_pos_rel_palm"] = fingertip_pos_rel_palm.reshape(N, -1)
        obs_dict["keypoints_rel_palm"] = keypoints_rel_palm.reshape(N, -1)
        obs_dict["keypoints_rel_goal"] = keypoints_rel_goal.reshape(N, -1)
        obs_dict["object_scales"] = object_scales
        obs_dict["closest_keypoint_max_dist"] = closest_keypoint_max_dist.unsqueeze(-1)
        obs_dict["closest_fingertip_dist"] = closest_fingertip_dist
        obs_dict["lifted_object"] = lifted_object.float().unsqueeze(-1)
        obs_dict["progress"] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        obs_dict["successes"] = torch.log(self.successes + 1).unsqueeze(-1)
        obs_dict["reward"] = 0.01 * self.rew_buf.unsqueeze(-1) if self.rew_buf.dim() == 1 else 0.01 * self.rew_buf

        # Critic states (clean)
        self.states_buf = torch.cat(
            [obs_dict[k].reshape(N, -1) for k in self.state_list], dim=-1
        )

        # Actor observations (may have noise/delay applied by caller)
        self.obs_buf = torch.cat(
            [obs_dict[k].reshape(N, -1) for k in self.flat_obs_list], dim=-1
        )

        # Depth images
        if self.use_depth:
            self.depth_obs = self.backend.get_depth_images()

    # ------------------------------------------------------------------
    # Reward computation (simulator-agnostic)
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        object_pos: torch.Tensor,
        object_init_z: torch.Tensor,
        object_linvel: torch.Tensor,
        object_angvel: torch.Tensor,
        keypoints_max_dist: torch.Tensor,
        closest_keypoint_max_dist: torch.Tensor,
        curr_fingertip_distances: torch.Tensor,
        closest_fingertip_dist: torch.Tensor,
        furthest_hand_dist: torch.Tensor,
        lifted_object: torch.Tensor,
        arm_dof_vel: torch.Tensor,
        hand_dof_vel: torch.Tensor,
        finger_rew_coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reward and detect successes.

        Returns ``(reward_buf, is_success)`` tensors.
        """
        N = object_pos.shape[0]

        # --- Lifting reward ---
        z_lift = 0.05 + object_pos[:, 2] - object_init_z
        lifting_rew = torch.clip(z_lift, 0, 0.5)
        just_lifted = (z_lift > self.lifting_bonus_threshold) & ~lifted_object
        lifted_object_new = (z_lift > self.lifting_bonus_threshold) | lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted
        lifting_rew *= ~lifted_object_new

        # --- Fingertip delta reward ---
        fingertip_deltas = torch.clip(
            closest_fingertip_dist - curr_fingertip_distances, 0, 10
        )
        fingertip_deltas *= finger_rew_coeffs
        fingertip_delta_rew = fingertip_deltas.sum(dim=-1) * ~lifted_object_new

        # --- Keypoint reward ---
        kp_deltas = torch.clip(
            closest_keypoint_max_dist - keypoints_max_dist, 0, 100
        )
        keypoint_rew = kp_deltas * lifted_object_new

        # --- Action penalties ---
        kuka_penalty = -(arm_dof_vel.abs().sum(dim=-1)) * self.kuka_actions_penalty_scale
        allegro_penalty = -(hand_dof_vel.abs().sum(dim=-1)) * self.allegro_actions_penalty_scale

        # --- Success detection ---
        kp_tol = self.success_tolerance * self.keypoint_scale
        near_goal = keypoints_max_dist <= kp_tol
        self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        is_success = self.near_goal_steps >= self.success_steps
        self.successes += is_success
        self.reset_goal_buf[:] = is_success

        bonus_rew = near_goal.float() * (self.reach_goal_bonus / self.success_steps)

        # --- Velocity penalties ---
        obj_lin_penalty = -object_linvel.square().sum(dim=-1) * self.object_lin_vel_penalty_scale
        obj_ang_penalty = -object_angvel.square().sum(dim=-1) * self.object_ang_vel_penalty_scale

        # --- Total ---
        reward = (
            fingertip_delta_rew * self.distance_delta_rew_scale
            + lifting_rew * self.lifting_rew_scale
            + lift_bonus_rew
            + keypoint_rew * self.keypoint_rew_scale
            + kuka_penalty
            + allegro_penalty
            + bonus_rew
            + obj_lin_penalty
            + obj_ang_penalty
        )

        self.rew_buf[:] = reward
        return self.rew_buf, is_success

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - lower) / (upper - lower) - 1.0

    def zero_actions(self) -> torch.Tensor:
        return torch.zeros(
            (self._num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.rl_device,
        )
