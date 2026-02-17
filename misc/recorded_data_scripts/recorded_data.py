from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

OBSERVATIONS_DIM = 117
ACTIONS_DIM = 23
JOINT_NAMES_ISAACGYM = [
    "iiwa7_joint_1",
    "iiwa7_joint_2",
    "iiwa7_joint_3",
    "iiwa7_joint_4",
    "iiwa7_joint_5",
    "iiwa7_joint_6",
    "iiwa7_joint_7",
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]

USE_SHARPA = False
if USE_SHARPA:
    ACTIONS_DIM = 29
    JOINT_NAMES_ISAACGYM = [
        "iiwa7_joint_1",
        "iiwa7_joint_2",
        "iiwa7_joint_3",
        "iiwa7_joint_4",
        "iiwa7_joint_5",
        "iiwa7_joint_6",
        "iiwa7_joint_7",
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
        "joint_16.0",
        "joint_17.0",
        "joint_18.0",
        "joint_19.0",
        "joint_20.0",
        "joint_21.0",
    ]

OBS_NAME_TO_NAMES = {
    "q": [f"{name}_q" for name in JOINT_NAMES_ISAACGYM],
    "qd": [f"{name}_qd" for name in JOINT_NAMES_ISAACGYM],
    "palm_center_pos": [f"palm_center_pos_{x}" for x in "xyz"],
    "palm_rot": [f"palm_rot_{x}" for x in "xyzw"],
    "palm_linvel": [f"palm_linvel_{x}" for x in "xyz"],
    "palm_angvel": [f"palm_angvel_{x}" for x in "xyz"],
    "object_rot": [f"object_rot_{x}" for x in "xyzw"],
    "object_linvel": [f"object_linvel_{x}" for x in "xyz"],
    "object_angvel": [f"object_angvel_{x}" for x in "xyz"],
    "fingertip_rel_pos": [
        f"fingertip_rel_pos_{finger}_{x}"
        for finger in ["index", "middle", "ring", "thumb"]
        for x in "xyz"
    ],
    "keypoints_rel_palm": [
        f"keypoints_rel_palm_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "keypoints_rel_goal": [
        f"keypoints_rel_goal_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "object_scales": [f"object_scales_{x}" for x in "xyz"],
    "closest_keypoint_max_dist": ["closest_keypoint_max_dist"],
    "closest_fingertip_dist": [
        f"closest_fingertip_dist_{finger}"
        for finger in ["index", "middle", "ring", "thumb"]
    ],
    "lifted_object": ["lifted_object"],
    "progress_obs": ["progress_obs"],
    "successes": ["successes_obs"],
    "reward_obs": ["reward_obs"],
}
OBS_NAMES = sum(OBS_NAME_TO_NAMES.values(), [])
ACTION_NAMES = JOINT_NAMES_ISAACGYM


@dataclass
class RecordedData:
    robot_root_states_array: np.ndarray
    object_root_states_array: np.ndarray
    robot_joint_positions_array: np.ndarray
    time_array: np.ndarray
    robot_joint_names: list[str]

    table_root_states_array: Optional[np.ndarray] = None
    goal_root_states_array: Optional[np.ndarray] = None
    robot_joint_velocities_array: Optional[np.ndarray] = None
    robot_joint_pos_targets_array: Optional[np.ndarray] = None

    observations_array: Optional[np.ndarray] = None
    actions_array: Optional[np.ndarray] = None

    object_name: Optional[str] = None

    def __post_init__(self):
        T = self.T
        J = self.J
        ROOT_STATE_DIM = 13  # xyz, xyzw, linvel, angvel
        assert self.robot_root_states_array.shape == (T, ROOT_STATE_DIM), (
            f"Expected robot root states array to be (T, {ROOT_STATE_DIM}), got {self.robot_root_states_array.shape}"
        )
        assert self.object_root_states_array.shape == (T, ROOT_STATE_DIM), (
            f"Expected object root states array to be (T, {ROOT_STATE_DIM}), got {self.object_root_states_array.shape}"
        )
        assert self.robot_joint_positions_array.shape == (T, J), (
            f"Expected robot joint positions array to be (T, J), got {self.robot_joint_positions_array.shape}"
        )
        assert self.time_array.shape == (T,), (
            f"Expected time array to be (T,), got {self.time_array.shape}"
        )
        assert len(self.robot_joint_names) == J, (
            f"Expected robot joint names to have length J, got {len(self.robot_joint_names)} and {J}"
        )

        if self.table_root_states_array is not None:
            assert self.table_root_states_array.shape == (T, ROOT_STATE_DIM), (
                f"Expected table root states array to be (T, {ROOT_STATE_DIM}), got {self.table_root_states_array.shape}"
            )
        if self.goal_root_states_array is not None:
            assert self.goal_root_states_array.shape == (T, ROOT_STATE_DIM), (
                f"Expected goal root states array to be (T, {ROOT_STATE_DIM}), got {self.goal_root_states_array.shape}"
            )
        if self.robot_joint_velocities_array is not None:
            assert self.robot_joint_velocities_array.shape == (T, J), (
                f"Expected robot joint velocities array to be (T, J), got {self.robot_joint_velocities_array.shape}"
            )
        if self.robot_joint_pos_targets_array is not None:
            assert self.robot_joint_pos_targets_array.shape == (T, J), (
                f"Expected robot joint pos targets array to be (T, J), got {self.robot_joint_pos_targets_array.shape}"
            )

        if self.observations_array is not None:
            assert self.observations_array.shape == (T, self.observations_dim), (
                f"Expected observations array to be (T, {self.observations_dim}), got {self.observations_array.shape}"
            )
        if self.actions_array is not None:
            assert self.actions_array.shape == (T, self.actions_dim), (
                f"Expected actions array to be (T, {self.actions_dim}), got {self.actions_array.shape}"
            )

    def to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            file_path,
            robot_root_states_array=self.robot_root_states_array,
            object_root_states_array=self.object_root_states_array,
            robot_joint_positions_array=self.robot_joint_positions_array,
            time_array=self.time_array,
            robot_joint_names=self.robot_joint_names,
            table_root_states_array=self.table_root_states_array,
            goal_root_states_array=self.goal_root_states_array,
            robot_joint_velocities_array=self.robot_joint_velocities_array,
            robot_joint_pos_targets_array=self.robot_joint_pos_targets_array,
            observations_array=self.observations_array,
            actions_array=self.actions_array,
            object_name=self.object_name,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> RecordedData:
        assert file_path.exists(), f"File {file_path} does not exist"
        recorded_data = np.load(file_path, allow_pickle=True)

        def maybe_none(x):
            if (
                isinstance(x, np.ndarray)
                and x.shape == ()
            ):
                # If 0-D array, just get the item
                # Handles case of None and string and other types
                item = x.item()
                return item
            return x

        return cls(
            robot_root_states_array=recorded_data["robot_root_states_array"],
            object_root_states_array=recorded_data["object_root_states_array"],
            robot_joint_positions_array=recorded_data["robot_joint_positions_array"],
            time_array=recorded_data["time_array"],
            robot_joint_names=recorded_data["robot_joint_names"],
            table_root_states_array=maybe_none(
                recorded_data["table_root_states_array"] if "table_root_states_array" in recorded_data else None
            ),
            goal_root_states_array=maybe_none(recorded_data["goal_root_states_array"] if "goal_root_states_array" in recorded_data else None),
            robot_joint_velocities_array=maybe_none(
                recorded_data["robot_joint_velocities_array"] if "robot_joint_velocities_array" in recorded_data else None
            ),
            robot_joint_pos_targets_array=maybe_none(
                recorded_data["robot_joint_pos_targets_array"] if "robot_joint_pos_targets_array" in recorded_data else None
            ),
            observations_array=maybe_none(recorded_data["observations_array"] if "observations_array" in recorded_data else None),
            actions_array=maybe_none(recorded_data["actions_array"] if "actions_array" in recorded_data else None),
            object_name=maybe_none(recorded_data["object_name"] if "object_name" in recorded_data else None),
        )

    def slice(
        self, start: int | None = None, end: int | None = None, reset_time: bool = True
    ) -> RecordedData:
        if start is None and end is None:
            raise ValueError("start and end cannot both be None")

        if start is None:
            start = 0
        if end is None:
            end = self.T

        return RecordedData(
            robot_root_states_array=self.robot_root_states_array[start:end],
            object_root_states_array=self.object_root_states_array[start:end],
            robot_joint_positions_array=self.robot_joint_positions_array[start:end],
            time_array=(
                self.time_array[start:end] - self.time_array[start]
                if reset_time
                else self.time_array[start:end]
            ),
            robot_joint_names=self.robot_joint_names,
            table_root_states_array=self.table_root_states_array[start:end] if self.table_root_states_array is not None else None,
            goal_root_states_array=self.goal_root_states_array[start:end] if self.goal_root_states_array is not None else None,
            robot_joint_velocities_array=self.robot_joint_velocities_array[start:end] if self.robot_joint_velocities_array is not None else None,
            robot_joint_pos_targets_array=self.robot_joint_pos_targets_array[start:end] if self.robot_joint_pos_targets_array is not None else None,
            observations_array=self.observations_array[start:end] if self.observations_array is not None else None,
            actions_array=self.actions_array[start:end] if self.actions_array is not None else None,
            object_name=self.object_name,
        )

    def __len__(self) -> int:
        return self.T

    def robot_joint_positions_reordered(
        self,
        to_order: list[str],
    ) -> np.ndarray:
        return self.change_joint_order(
            q=self.robot_joint_positions_array,
            from_order=self.robot_joint_names,
            to_order=to_order,
        )

    def robot_joint_pos_targets_reordered(
        self,
        to_order: list[str],
    ) -> np.ndarray:
        return self.change_joint_order(
            q=self.robot_joint_pos_targets_array,
            from_order=self.robot_joint_names,
            to_order=to_order,
        )

    # ###############
    # Complex Properties
    # ###############
    @cached_property
    def robot_joint_velocities_array_fd2(self) -> np.ndarray:
        q = self.robot_joint_positions_array
        t = self.time_array

        qd = np.zeros_like(q)

        # Interior points: 2nd order finite difference
        # q_i = (q_{i+1} - q_{i-1}) / (t_{i+1} - t_{i-1})
        qd[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[..., None]

        # First point: 1st order finite difference
        # q_0 = (q_1 - q_0) / (t_1 - t_0)
        qd[0] = (q[1] - q[0]) / (t[1] - t[0])

        # Last point: 1st order finite difference
        # q_N = (q_N - q_{N-1}) / (t_N - t_{N-1})
        qd[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
        return qd

    @cached_property
    def robot_joint_velocities_array_fd1(self) -> np.ndarray:
        q = self.robot_joint_positions_array
        t = self.time_array
        qd = np.zeros_like(q)

        # Forward difference for all but the last point
        # q_i = (q_{i+1} - q_i) / (t_{i+1} - t_i)
        qd[1:] = (q[1:] - q[:-1]) / (t[1:] - t[:-1])[..., None]

        # Backward difference for the last point
        # q_N = (q_N - q_{N-1}) / (t_N - t_{N-1})
        qd[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
        return qd

    # ###############
    # Hardcoded Properties
    # ###############
    @cached_property
    def observation_names(self) -> list[str]:
        names = OBS_NAMES
        assert len(names) == OBSERVATIONS_DIM, (
            f"Expected {len(names)} observation names, got {OBSERVATIONS_DIM}"
        )
        return names

    @cached_property
    def action_names(self) -> list[str]:
        names = ACTION_NAMES
        assert len(names) == ACTIONS_DIM, (
            f"Expected {len(names)} action names, got {ACTIONS_DIM}"
        )
        return names

    # ###############
    # Simple Properties
    # ###############
    @cached_property
    def T(self) -> int:
        return self.robot_root_states_array.shape[0]

    @cached_property
    def dt(self) -> float:
        dt = self.time_array[1] - self.time_array[0]
        assert np.allclose(np.diff(self.time_array), dt), (
            f"Expected time array to be evenly spaced, got {self.time_array}"
        )
        return dt

    @cached_property
    def J(self) -> int:
        return len(self.robot_joint_names)

    @cached_property
    def total_time(self) -> float:
        return self.time_array[-1] - self.time_array[0]

    @cached_property
    def observations_dim(self) -> int:
        assert self.observations_array.shape[-1] == OBSERVATIONS_DIM, (
            f"Expected observations array to have shape (..., {OBSERVATIONS_DIM}), got {self.observations_array.shape}"
        )
        return OBSERVATIONS_DIM

    @cached_property
    def actions_dim(self) -> int:
        assert self.actions_array.shape[-1] == ACTIONS_DIM, (
            f"Expected actions array to have shape (..., {ACTIONS_DIM}), got {self.actions_array.shape}"
        )
        return ACTIONS_DIM

    # ###############
    # Static methods
    # ###############
    @staticmethod
    def change_joint_order(
        q: np.ndarray,
        from_order: list[str],
        to_order: list[str],
        require_all_joints: bool = True,
    ) -> np.ndarray:
        J = len(from_order)
        assert q.ndim in [1, 2], (
            f"Expected q to be either (N,) or (N, J), got {q.shape}"
        )
        assert q.shape[-1] == J, (
            f"Expected q to have the same length as from_order, got {q.shape[-1]} and {J}"
        )

        if require_all_joints:
            assert len(to_order) == J, (
                f"Expected to_order to have the same length as from_order, got {len(to_order)} and {len(from_order)}. If you don't want to require all joints, set require_all_joints to False."
            )

        assert set(to_order).issubset(set(from_order)), (
            f"Expected to_order to be a subset of from_order, got to_order: {to_order} and from_order: {from_order}. Only in to_order: {set(to_order) - set(from_order)}"
        )

        # q is given in the from_order
        joint_name_to_value = {from_order[i]: q[..., i] for i in range(J)}
        new_q = np.stack([joint_name_to_value[name] for name in to_order], axis=-1)

        assert new_q.shape == (q.shape[:-1] + (len(to_order),)), (
            f"Expected new_q to be {q.shape[:-1] + (len(to_order),)}, got {new_q.shape}"
        )
        if require_all_joints:
            assert new_q.shape == q.shape, (
                f"Expected new_q to be {q.shape}, got {new_q.shape}"
            )
        return new_q

    @staticmethod
    def pose_to_T(pose: np.ndarray) -> np.ndarray:
        assert pose.ndim in [1, 2], (
            f"Expected pose to be either (7,) or (N, 7), got {pose.shape}"
        )
        assert pose.shape[-1] == 7, (
            f"Expected pose to be (7,) or (N, 7), got {pose.shape}"
        )
        xyz = pose[..., :3]
        xyzw = pose[..., 3:7]
        T = (
            np.eye(4)
            if pose.ndim == 1
            else np.eye(4)[None, ...].repeat(repeats=pose.shape[0], axis=0)
        )
        T[..., :3, :3] = R.from_quat(xyzw).as_matrix()
        T[..., :3, 3] = xyz
        return T

    @staticmethod
    def T_to_pose(T: np.ndarray) -> np.ndarray:
        assert T.ndim in [2, 3], (
            f"Expected T to be either (4, 4) or (N, 4, 4), got {T.shape}"
        )
        assert T.shape[-2:] == (4, 4), (
            f"Expected T to be (4, 4) or (N, 4, 4), got {T.shape}"
        )
        xyz = T[..., :3, 3]
        xyzw = R.from_matrix(T[..., :3, :3]).as_quat()
        pose = np.concatenate([xyz, xyzw], axis=-1)
        assert pose.shape == (T.shape[:-2] + (7,)), (
            f"Expected pose to be {T.shape[:-2] + (7,)}, got {pose.shape}"
        )
        return pose
