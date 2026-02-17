"""Evaluation script for dexterous manipulation with viser visualization."""

# IsaacGym must be imported before torch
from isaacgym import gymapi  # noqa: F401 isort:skip

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
import viser
from termcolor import colored
from viser.extras import ViserUrdf

from isaacgymenvs.utils.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir
from eval.sim2real.rl_player import RlPlayer
from eval.sim2sim.isaac_sim.isaac_env import create_env

TABLE_Z = 0.38
# TABLE_Z = 0.37


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz format."""
    return (q[3], q[0], q[1], q[2])

def log_info(text):
    print(colored(text, "cyan"))

def log_success(text):
    print(colored(text, "green"))

def log_warn(text):
    print(colored(text, "yellow"))


class ViserServer:
    """Viser-based visualization server for robot manipulation."""

    def __init__(self, object_name: str, trajectory_name: str, num_keypoints: int, table_urdf: str, port: int = 8080):
        self.port = port
        self.num_keypoints = num_keypoints
        self.is_paused = False
        self.show_keypoints = True
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.table_urdf = table_urdf
        self._setup_scene(object_name, trajectory_name)

    def _setup_scene(self, object_name: str, trajectory_name: str):
        """Initialize the 3D scene with robot, table, object, and GUI elements."""
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        # Ground grid
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Robot
        robot_urdf = get_repo_root_dir() / "assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        self.server.scene.add_frame("/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False)
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(np.zeros(29))

        # Table
        table_urdf = get_repo_root_dir() / "assets" / self.table_urdf
        self.server.scene.add_frame("/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False)
        # ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 0.5))
        ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 1.0))

        # Object and goal
        object_urdf = NAME_TO_OBJECT[object_name].filepath
        self.object_frame = self.server.scene.add_frame("/object", show_axes=True, axes_length=0.1, axes_radius=0.001)
        ViserUrdf(self.server, object_urdf, root_node_name="/object")
        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=True, axes_length=0.1, axes_radius=0.001)
        ViserUrdf(self.server, object_urdf, root_node_name="/goal", mesh_color_override=(0, 255, 0, 0.5))

        # Keypoint spheres (red for object, green for goal)
        self.obj_keypoint_spheres = []
        self.goal_keypoint_spheres = []
        self.obj_keypoint_spheres_fixed_size = []
        self.goal_keypoint_spheres_fixed_size = []
        for i in range(self.num_keypoints):
            self.obj_keypoint_spheres.append(
                self.server.scene.add_icosphere(f"/obj_keypoint_{i}", radius=0.01, color=(255, 0, 0))
            )
            self.goal_keypoint_spheres.append(
                self.server.scene.add_icosphere(f"/goal_keypoint_{i}", radius=0.01, color=(0, 255, 0))
            )
            self.obj_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(f"/obj_keypoint_{i}_fixed_size", radius=0.01, color=(255, 0, 0), opacity=0.6)
            )
            self.goal_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(f"/goal_keypoint_{i}_fixed_size", radius=0.01, color=(0, 255, 0), opacity=0.6)
            )

        # GUI elements
        self.server.gui.add_markdown(f"**Task:** {trajectory_name}")
        self.server.gui.add_markdown(f"**Object:** {object_name}")
        self.server.gui.add_markdown("---")
        self.progress_text = self.server.gui.add_markdown("**Progress:** --")
        self.stats_text = self.server.gui.add_markdown("**Stats:** No episodes completed")
        self.object_state_text = self.server.gui.add_markdown("**Object State:** --")
        self.server.gui.add_markdown("---")

        # Controls
        self.keypoint_toggle = self.server.gui.add_checkbox("Show Keypoints", initial_value=True)
        self.keypoint_toggle.on_update(lambda _: self._toggle_keypoints())
        self.keypoint_toggle_fixed_size = self.server.gui.add_checkbox("Show Keypoints Fixed Size", initial_value=False)
        self.keypoint_toggle_fixed_size.on_update(lambda _: self._toggle_keypoints_fixed_size())

    def _toggle_keypoints(self):
        """Toggle visibility of keypoint spheres."""
        self.show_keypoints = self.keypoint_toggle.value
        for sphere in self.obj_keypoint_spheres + self.goal_keypoint_spheres:
            sphere.visible = self.show_keypoints

    def _toggle_keypoints_fixed_size(self):
        """Toggle visibility of keypoint spheres fixed size."""
        self.show_keypoints_fixed_size = self.keypoint_toggle_fixed_size.value
        for sphere in self.obj_keypoint_spheres_fixed_size + self.goal_keypoint_spheres_fixed_size:
            sphere.visible = self.show_keypoints_fixed_size

    def _toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.pause_button.name = "Resume" if self.is_paused else "Pause"
        log_info(f"Paused: {self.is_paused}")

    def add_controls(self, run_callback):
        """Add run and pause buttons."""
        self.server.gui.add_button("Run Episode").on_click(lambda _: run_callback())
        self.pause_button = self.server.gui.add_button("Pause")
        self.pause_button.on_click(lambda _: self._toggle_pause())

    def update(self, joint_pos, object_pose, goal_pose, obj_keypoints=None, goal_keypoints=None, obj_keypoints_fixed_size=None, goal_keypoints_fixed_size=None):
        """Update visualization with current state."""
        self.robot.update_cfg(joint_pos)
        self.object_frame.position = object_pose[:3]
        self.object_frame.wxyz = quat_xyzw_to_wxyz(object_pose[3:7])
        self.goal_frame.position = goal_pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

        if obj_keypoints is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres):
                sphere.position = tuple(obj_keypoints[i])
        if goal_keypoints is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres):
                sphere.position = tuple(goal_keypoints[i])
        if obj_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres_fixed_size):
                sphere.position = tuple(obj_keypoints_fixed_size[i])
        if goal_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres_fixed_size):
                sphere.position = tuple(goal_keypoints_fixed_size[i])

    def update_progress(self, current: int, total: int, timestep: int, control_hz: float = 60.0):
        """Update progress display."""
        pct = 100 * current / total if total > 0 else 0
        self.progress_text.content = f"**Time:** {timestep / control_hz:.1f}s | **Goal:** {current}/{total} ({pct:.0f}%)"

    def update_object_state(self, object_state: np.ndarray):
        object_pos = object_state[:3]
        self.object_state_text.content = f"**Object State:** {object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}"

    def update_stats(self, num_episodes: int, avg_goal_pct: float, avg_time_sec: float):
        """Update statistics display."""
        self.stats_text.content = f"**Episodes:** {num_episodes} | **Avg Goal:** {avg_goal_pct:.1f}% | **Avg Time:** {avg_time_sec:.1f}s"

    def get_frame(self) -> np.ndarray:
        """Capture current view as image."""
        clients = list(self.server.get_clients().values())
        if clients:
            return clients[0].camera.get_render(height=480, width=640)
        return np.zeros((480, 640, 3), dtype=np.uint8)


class EvalRunner:
    """Runs policy evaluation with viser visualization."""

    def __init__(self, env, config_path: Path, checkpoint_path: Path,
                 object_name: str, trajectory_name: str, table_urdf: str, output_dir: Optional[Path] = None):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_act = 29
        self.control_hz = 60.0
        self.control_dt = 1.0 / self.control_hz
        self.record_fps = 10
        self.record_interval = int(self.control_hz / self.record_fps)

        # Joint limits for denormalization
        self.joint_lower = env.arm_hand_dof_lower_limits[:self.n_act].cpu().numpy()
        self.joint_upper = env.arm_hand_dof_upper_limits[:self.n_act].cpu().numpy()

        # Load policy
        self.env.set_env_state(torch.load(checkpoint_path)[0]["env_state"])
        self.policy = RlPlayer(140, self.n_act, config_path, checkpoint_path, self.device, env.num_envs)

        # Recording setup
        self.record_video = output_dir is not None
        self.episode_count = 0
        self.episode_goal_pcts = []
        self.episode_lengths = []
        if self.record_video:
            self.session_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            log_info(f"Recording to: {self.session_dir}")

        # Visualization
        self.viser = ViserServer(object_name, trajectory_name, env.num_keypoints, table_urdf)
        self.obs = self._reset()

    def _reset(self):
        """Reset environment and return initial observation."""
        obs, _, _, _ = self.env.step(torch.zeros((self.env.num_envs, self.n_act), device=self.device))
        return obs["obs"]

    def _step(self, action) -> Tuple[torch.Tensor, bool]:
        """Step environment with action."""
        obs, _, done, _ = self.env.step(action)
        return obs["obs"], done[0].item()

    def _get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract current state for visualization."""
        obs_np = self.obs[0].cpu().numpy()
        joint_pos = 0.5 * (obs_np[:29] + 1.0) * (self.joint_upper - self.joint_lower) + self.joint_lower
        return (
            joint_pos,
            self.env.object_state[0, :7].cpu().numpy(),
            self.env.goal_pose[0].cpu().numpy(),
            self.env.obj_keypoint_pos[0].cpu().numpy(),
            self.env.goal_keypoint_pos[0].cpu().numpy(),
            self.env.obj_keypoint_pos_fixed_size[0].cpu().numpy(),
            self.env.goal_keypoint_pos_fixed_size[0].cpu().numpy(),
        )

    def _sim_step(self, timestep: int) -> bool:
        """Execute one simulation step with timing control."""
        t0 = time.time()
        self.viser.update(*self._get_state())
        self.obs, done = self._step(self.policy.get_normalized_action(self.obs, deterministic_actions=True))
        self.viser.update_progress(int(self.env.successes[0].item()), self.env.max_consecutive_successes, timestep, self.control_hz)
        self.viser.update_object_state(self.env.object_state[0].cpu().numpy())

        elapsed = time.time() - t0
        if (sleep_time := self.control_dt - elapsed) > 0:
            time.sleep(sleep_time)
        return done

    def _render_video(self, states: list, path: Path):
        """Render recorded states to video file."""
        log_info(f"Rendering {len(states)} frames...")
        frames = []
        for i, state in enumerate(states):
            self.viser.update(*state)
            time.sleep(0.05)
            frames.append(self.viser.get_frame())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(states)}")
        imageio.mimsave(str(path), frames, fps=self.record_fps)
        log_success(f"Saved: {path}")

    def _run_episode(self):
        """Run a single evaluation episode."""

        # Don't let this function be called twice at the same time
        if not hasattr(self, "_run_episode_in_progress"):
            self._run_episode_in_progress = False
        if self._run_episode_in_progress:
            log_warn("Episode already in progress. Skipping...")
            return
        self._run_episode_in_progress = True

        self.policy.reset()
        log_info("Reset...")
        self.obs = self._reset()
        self.viser.update(*self._get_state())

        log_success(f"Running{' (+ recording)' if self.record_video else ''}...")
        states, step, done = [], 0, False

        while not done:
            # Handle pause
            while self.viser.is_paused:
                time.sleep(0.1)

            if self.record_video and step % self.record_interval == 0:
                states.append(tuple(x.copy() for x in self._get_state()))
            done = self._sim_step(step)
            step += 1

        # Update stats
        goal_pct = 100 * int(self.env.successes[0].item()) / self.env.max_consecutive_successes
        self.episode_goal_pcts.append(goal_pct)
        self.episode_lengths.append(step)
        self.episode_count += 1
        avg_goal_pct = sum(self.episode_goal_pcts) / len(self.episode_goal_pcts)
        avg_time_sec = sum(self.episode_lengths) / len(self.episode_lengths) / self.control_hz
        self.viser.update_stats(self.episode_count, avg_goal_pct, avg_time_sec)

        if states and self.record_video:
            self._render_video(states, self.session_dir / f"{self.episode_count}.mp4")

        log_success(f"Done: {step / self.control_hz:.1f}s, {goal_pct:.0f}% goals")
        self._run_episode_in_progress = False

    def run(self):
        """Start the evaluation loop."""
        self.viser.add_controls(self._run_episode)
        log_info(f"Open http://localhost:{self.viser.port}")
        log_info("Click 'Run Episode' to start.")
        while True:
            time.sleep(1.0)


def parse_checkpoint_dir(path: Path) -> Tuple[Path, Path]:
    """Extract config and model paths from checkpoint directory."""
    run_dir = sorted(d for d in (path / "runs").iterdir() if d.is_dir())[0]
    return run_dir / "config.yaml", run_dir / "best" / "model.pth"


def main():
    # Configuration
    # checkpoint_dir = Path("/share/portal/kk837/sapg/train_dir/FINAL_ASYMMETRIC_RUNS/FINETUNE_8x/O0T0_tyler_branch_2026-01-05_02-16-57")

    # checkpoint_dir = Path("/share/portal/kk837/sapg/train_dir/WHAT_MAKES_TRAINING_SLOW/FINETUNE4x_SLOWSPEED_2026-01-05_02-17-46")

    # checkpoint_dir = Path("/share/portal/kk837/sapg/train_dir/customPretraining/FINETUNE_5x/FINETUNE_5x_SLOW_SPEED_ADD_ACTION_DELAY_2026-01-05_02-10-22")

    # object_type = "hammer"
    # object_name = "hammer_2"
    # object_name = "mallet"
    # trajectory_name = "horizontal_swing_nail"
    # trajectory_name = "horizontal_swing_rotated"
    # trajectory_name = "vertical_swing"
    # trajectory_name = "vertical_swing_2"
    # trajectory_name = "horizontal_swing_human"

    # object_type = "spatula"
    # object_name = "black_spatula"
    # object_name = "small_spatula"
    # object_name = "large_spatula"
    # trajectory_name = "flip_from_left"
    # trajectory_name = "pick_and_place_human"

    # object_type = "eraser"
    # object_name = "whiteboard_eraser"
    # trajectory_name = "wipe_right"
    # trajectory_name = "wipe_left"
    # trajectory_name = "wipe_left_slanted"
    # trajectory_name = "wipe_left_slanted_higher"
    # trajectory_name = "wipe_left_vertical"
    # trajectory_name = "wipe_left_vertical_farther"
    # trajectory_name = "wipe_left_slanted_higher_farther"
    # trajectory_name = "wipe_left_human"
    # trajectory_name = "wipe_left_human_2"

    # object_type = "screwdriver"
    # object_name = "real_flat_screwdriver"
    # object_name = "cylindrical_screwdriver"
    # trajectory_name = "top_down_screwing"
    # trajectory_name = "top_down_screwing_closer"
    # trajectory_name = "top_down_screwing_closer_lower"
    # trajectory_name = "top_down_screwing_closer_lower_hole"
    # trajectory_name = "top_down_screwing_human"
    # trajectory_name = "top_down_screwing_human_easyinit"

    # object_type = "marker"
    # object_name = "040_large_marker"
    # trajectory_name = "write_circle_whiteboard"
    # trajectory_name = "write_circle_whiteboard_adjusted"
    # trajectory_name = "draw_circle_human"
    # trajectory_name = "draw_circle_human_hardinit"

    # object_type = "knife"
    # object_name = "kitchen_knife"
    # trajectory_name = "knife_on_cutting_board"

    object_type = "spatula"
    object_name = "black_spatula"
    # trajectory_name = "pick_and_place"
    # trajectory_name = "pick_and_place_hardinit"
    # trajectory_name = "pick_and_place_hardinit2"
    trajectory_name = "pick_and_place_human"
    # trajectory_name = "pick_and_place_human_hardinit"

    # object_type = "brush"
    # object_name = "green_brush"
    # object_name = "red_brush"
    # trajectory_name = "simple"
    # trajectory_name = "complex"

    output_dir = None  # Set to Path("videos") to enable recording

    TABLE_URDF = "urdf/table_narrow.urdf"
    TABLE_WHITEBOARD_URDF = "urdf/table_narrow_whiteboard.urdf"
    TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
    TABLE_SCREWDRIVER_HOLE_URDF = "urdf/table_narrow_screwdriver_hole.urdf"
    TABLE_CUTTINGBOARD_URDF = "urdf/table_narrow_cuttingboard.urdf"
    TABLE_BOWL_PLATE_URDF = "urdf/table_narrow_bowl_plate.urdf"

    SELECTED_TABLE_URDF = TABLE_URDF
    # SELECTED_TABLE_URDF = TABLE_NAIL_URDF
    # SELECTED_TABLE_URDF = TABLE_WHITEBOARD_URDF
    # SELECTED_TABLE_URDF = TABLE_SCREWDRIVER_HOLE_URDF
    # SELECTED_TABLE_URDF = TABLE_CUTTINGBOARD_URDF
    # SELECTED_TABLE_URDF = TABLE_BOWL_PLATE_URDF

    # Load trajectory
    trajectory_path = get_repo_root_dir() / "dex_tool_bench/evaluation_trajectories" / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Create environment
    # config_path, checkpoint_path = parse_checkpoint_dir(checkpoint_dir)

    # folder_path = Path("/juno/u/kedia/sapg/train_dir/latest_checkpoints/o0t0_fullSpeed")
    folder_path = Path("/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_slowSpeed")
    # folder_path = Path("/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_fastSpeed")
    config_path = folder_path / "config.yaml"
    checkpoint_path = folder_path / "model.pth"

    # Original one we tried in eral
    # config_path = Path("/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/config.yaml")
    # checkpoint_path = Path("/juno/u/kedia/sapg/train_dir/checkpoints/FINETUNED/finetuned_o0t0.pth")

    env = create_env(
        config_path=str(config_path),
        # headless=True,
        headless=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.resetRotationNoise": 0.0,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.object_type": object_name,
            "task.env.randomizeObjectRotation": False,
            # "task.env.numEnvs": 1,
            "task.env.numEnvs": 5,
            "task.env.envSpacing": 0.4,
            "task.env.tableResetZRange": 0.0,
            # "task.env.tableResetZ": 0.38 + 0.02,
            "task.env.capture_video": False,
            "task.env.use_fixed_set_of_goal_states": True,
            "task.env.fixedGoalStates": traj_data["goals"],
            # "task.env.fixedGoalStates": None,
            "task.env.objectStartPose": traj_data["start_pose"],
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.resetWhenDropped": False,
            # "task.env.armMovingAverage": 0.05,
            "task.env.armMovingAverage": 0.1,
            # "task.env.evalSuccessTolerance": 0.0075,
            # "task.env.evalSuccessTolerance": 0.01,
            "task.env.evalSuccessTolerance": 0.02,
            # "task.env.evalSuccessTolerance": 0.025,
            # "task.env.evalSuccessTolerance": 0.03,
            # "task.env.successSteps": 3,
            "task.env.successSteps": 1,
            "task.env.asset.table": str(SELECTED_TABLE_URDF),
            "task.env.tableResetZ": TABLE_Z,
            "task.env.fixedSizeKeypointReward": True,
            "task.env.startArmHigher": True,

            # Forces
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            # "task.env.forceScale": 2.0,  # Default
            # "task.env.torqueScale": 0.2,  # 10x smaller than forceScale because objects are ~0.1m
            # "task.env.linVelImpulseScale": 0.2,  # 20cm/s
            # "task.env.angVelImpulseScale": 2.0,  # 2rad/s = 114.6 deg/s

            "task.env.forceOnlyWhenLifted": True,
            "task.env.torqueOnlyWhenLifted": True,
            "task.env.linVelImpulseOnlyWhenLifted": True,
            "task.env.angVelImpulseOnlyWhenLifted": True,

            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        },
    )

    EvalRunner(env, config_path, checkpoint_path, object_name, trajectory_name, SELECTED_TABLE_URDF, output_dir).run()


if __name__ == "__main__":
    main()
