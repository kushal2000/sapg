from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import rerun as rr
import yourdfpy
from tqdm import tqdm

from misc.recorded_data_scripts.recorded_data import RecordedData

# ###########
# Constants
# ###########
AXES_LENGTH = 0.2


# ###########
# Rerun helper functions
# ###########
def build_joint_paths(urdf: yourdfpy.URDF, prefix: str) -> dict[str, str]:
    all_joints = urdf.joint_map  # include revolute + fixed + others
    rev = {n: j for n, j in all_joints.items() if j.type == "revolute"}  # final keys
    child_to_joint = {j.child: j for j in all_joints.values()}

    cache: dict[str, str] = {}

    def build(child: str) -> str:
        if child in cache:
            return cache[child]
        j = child_to_joint.get(child)
        if j is None:
            path = child  # root link
        else:
            path = f"{build(j.parent)}/{j.name}/{child}"
        cache[child] = path
        return path

    # Only return revolute joints, but paths include all intermediate links/joints
    return {name: f"{prefix}/{build(j.child)}" for name, j in rev.items()}


def update_joints(
    joint_name_to_pos: dict[str, float],
    joint_paths: dict[str, str],
    urdf: yourdfpy.URDF,
):
    for name, pos in joint_name_to_pos.items():
        if name not in joint_paths:
            continue
        path = joint_paths[name]
        j = urdf.joint_map[name]
        axis = j.axis.tolist()
        rr.log(
            path,
            rr.Transform3D(
                rotation_axis_angle=rr.RotationAxisAngle(
                    axis=axis,
                    angle=pos,
                ),
            ),
        )


def update_joints_array(
    joint_name_to_pos_array: dict[str, np.ndarray],
    joint_paths: dict[str, str],
    urdf: yourdfpy.URDF,
    time_array: np.ndarray,
):
    time_indexes = [
        rr.TimeColumn(
            "tick",
            duration=[timedelta(seconds=float(s - time_array[0])) for s in time_array],
        )
    ]
    for name, pos_array in joint_name_to_pos_array.items():
        if name not in joint_paths:
            continue
        path = joint_paths[name]
        j = urdf.joint_map[name]
        axis = j.axis.tolist()
        T = pos_array.shape[0]
        rotation_axis_angle_list = [
            rr.RotationAxisAngle(
                axis=axis,
                angle=pos_array[i],
            )
            for i in range(T)
        ]
        rr.send_columns(
            path,
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                rotation_axis_angle=rotation_axis_angle_list
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()
    file_path = Path(args.file_path)

    # ###########
    # Load recorded data
    # ###########
    # file_path = Path(
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-43-04.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-42-41.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-30-37.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-09.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-31.npz"
    #     "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_17-18-32.npz"
    # )
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)

    # ###########
    # Load recorded data
    # ###########
    # file_path = Path(
    #     "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-16_09-08-27.npz"
    # )
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)

    # ###########
    # Setup rerun objects
    # ###########
    APPLICATION_ID = "rerun_visualize"
    rr.init(application_id=APPLICATION_ID, spawn=True)
    rr.set_time("tick", duration=timedelta(seconds=0))

    # Add world frame
    rr.log(
        "world",
        rr.Transform3D(clear=False, axis_length=AXES_LENGTH),
    )

    # Load assets into rerun
    KUKA_ALLEGRO_URDF_PATH = Path(
        "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_real.urdf"
    )
    assert KUKA_ALLEGRO_URDF_PATH.exists(), (
        f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
    )
    from isaacgymenvs.utils.objects import NAME_TO_OBJECT
    OBJECT_URDF_PATH = NAME_TO_OBJECT["044_flat_screwdriver"].filepath
    # OBJECT_URDF_PATH = Path(
    #     "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf"
    # )
    assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
    ALLEGRO_URDF_PATH = Path(
        "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf"
    )
    assert ALLEGRO_URDF_PATH.exists(), (
        f"ALLEGRO_URDF_PATH not found: {ALLEGRO_URDF_PATH}"
    )
    TABLE_URDF_PATH = Path(
        "/home/tylerlum/github_repos/sapg/assets/urdf/table_narrow.urdf"
    )
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    # Load urdfs
    kuka_allegro_urdf = yourdfpy.URDF.load(KUKA_ALLEGRO_URDF_PATH)
    allegro_urdf = yourdfpy.URDF.load(ALLEGRO_URDF_PATH)

    # Robot
    rr.log_file_from_path(
        KUKA_ALLEGRO_URDF_PATH,
        entity_path_prefix="/kuka_allegro",
    )

    # Object
    rr.log_file_from_path(
        OBJECT_URDF_PATH,
        entity_path_prefix="/object",
    )

    # Table
    if recorded_data.table_root_states_array is not None:
        rr.log_file_from_path(
            TABLE_URDF_PATH,
            entity_path_prefix="/table",
        )

    # Goal
    if recorded_data.goal_root_states_array is not None:
        rr.log("goal", rr.Transform3D(clear=False, axis_length=AXES_LENGTH))

    # Palm
    rr.log("palm", rr.Transform3D(clear=False, axis_length=AXES_LENGTH))

    # Floating allegro hand
    rr.log_file_from_path(
        ALLEGRO_URDF_PATH,
        entity_path_prefix="/allegro",
    )

    # Object relative to floating allegro hand
    rr.log_file_from_path(
        OBJECT_URDF_PATH,
        entity_path_prefix="/allegro/object",
    )

    # Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
    kuka_allegro_joint_names = kuka_allegro_urdf.actuated_joint_names
    allegro_joint_names = allegro_urdf.actuated_joint_names

    # ###########
    # Build joint paths
    # ###########
    # BRITTLE: This is the most brittle part of the code, since it relies on the urdf structure
    # Need to check that the joint paths created here match what is created in the rerun viewer
    # Currently, prefix is /{entity_path_prefix}/{urdf_robot_name}
    # Where entity_path_prefix is what we pass into rr.log_file_from_path
    # and urdf_robot_name is the name of the robot urdf (defined in the urdf file)
    # Debug this by breakpointing after creating the joint paths and then printing out the joint paths here
    # Then in rerun viewer, click into the tree of links deeply, right click, and copy the path to the clipboard
    # Compare the copied path with the joint paths here and modify as needed
    kuka_allegro_joint_paths = build_joint_paths(
        kuka_allegro_urdf, prefix="/kuka_allegro/kuka_allegro"
    )
    allegro_joint_paths = build_joint_paths(allegro_urdf, prefix="/allegro/allegro")

    # ###########
    # Run forward kinematics and compute relative poses
    # ###########
    # Run forward kinematics to get palm poses over time
    kuka_allegro_joint_positions_reordered = (
        recorded_data.robot_joint_positions_reordered(to_order=kuka_allegro_joint_names)
    )
    T_R_P_list = []
    for t in tqdm(
        range(len(recorded_data.time_array)), desc="Running forward kinematics"
    ):
        kuka_allegro_urdf.update_cfg(kuka_allegro_joint_positions_reordered[t, :])
        palm_pose_R = kuka_allegro_urdf.get_transform(frame_to="allegro_mount").copy()
        assert palm_pose_R.shape == (
            4,
            4,
        ), f"palm_pose_R.shape: {palm_pose_R.shape}"
        T_R_P = palm_pose_R
        T_R_P_list.append(T_R_P)
    T_R_Ps = np.stack(T_R_P_list, axis=0)
    T_W_Rs = RecordedData.pose_to_T(recorded_data.robot_root_states_array[:, :7])
    T_W_Ps = T_W_Rs @ T_R_Ps
    palm_xyz_xyzw = RecordedData.T_to_pose(T_W_Ps)

    # By default MOVE_FLOATING_ALLEGRO_HAND = False so we can see how the object is moving wrt a fixed allegro hand
    # Can set to True to debug and make sure that everything aligns
    MOVE_FLOATING_ALLEGRO_HAND = False
    if MOVE_FLOATING_ALLEGRO_HAND:
        floating_allegro_hand_position = palm_xyz_xyzw[:, :3]
        floating_allegro_hand_quat_xyzw = palm_xyz_xyzw[:, 3:7]
    else:
        floating_allegro_hand_position = recorded_data.robot_root_states_array[
            :, :3
        ] + np.array([0.5, -0.8, 0.7])[None].repeat(
            len(recorded_data.time_array), axis=0
        )
        floating_allegro_hand_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])[None].repeat(
            len(recorded_data.time_array), axis=0
        )

    # Compute object poses wrt palm frame
    T_W_Os = RecordedData.pose_to_T(recorded_data.object_root_states_array[:, :7])
    T_P_Ws = np.linalg.inv(T_W_Ps)
    T_P_Os = T_P_Ws @ T_W_Os
    object_xyz_xyzw_P = RecordedData.T_to_pose(T_P_Os)

    # ###########
    # Log data
    # ###########

    # Columns is batched and fast
    # Log is easier to use but slow
    from typing import Literal

    MODE: Literal["columns", "log"] = "log"
    if MODE == "columns":
        # Time
        time_indexes = [
            rr.TimeColumn(
                "tick",
                duration=[
                    timedelta(seconds=float(s - recorded_data.time_array[0]))
                    for s in recorded_data.time_array
                ],
            )
        ]

        # Robot
        rr.send_columns(
            "kuka_allegro",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    recorded_data.robot_root_states_array[t, :3]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=recorded_data.robot_root_states_array[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
                axis_length=[AXES_LENGTH for _ in range(len(recorded_data.time_array))],
            ),
        )
        kuka_allegro_joint_name_to_pos_array = {
            name: recorded_data.robot_joint_positions_array[:, i]
            for i, name in enumerate(recorded_data.robot_joint_names)
        }
        update_joints_array(
            joint_name_to_pos_array=kuka_allegro_joint_name_to_pos_array,
            joint_paths=kuka_allegro_joint_paths,
            urdf=kuka_allegro_urdf,
            time_array=recorded_data.time_array,
        )

        # Object
        rr.send_columns(
            "object",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    recorded_data.object_root_states_array[t, :3]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=recorded_data.object_root_states_array[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
                axis_length=[AXES_LENGTH for _ in range(len(recorded_data.time_array))],
            ),
        )

        # Table
        if recorded_data.table_root_states_array is not None:
            rr.send_columns(
                "table",
                indexes=time_indexes,
                columns=rr.Transform3D.columns(
                    translation=[
                        recorded_data.table_root_states_array[t, :3]
                        for t in range(len(recorded_data.time_array))
                    ],
                    quaternion=[
                        rr.Quaternion(
                            xyzw=recorded_data.table_root_states_array[t, 3:7],
                        )
                        for t in range(len(recorded_data.time_array))
                    ],
                    axis_length=[
                        AXES_LENGTH for _ in range(len(recorded_data.time_array))
                    ],
                ),
            )

        # Goal
        if recorded_data.goal_root_states_array is not None:
            rr.send_columns(
                "goal",
                indexes=time_indexes,
                columns=rr.Transform3D.columns(
                    translation=[
                        recorded_data.goal_root_states_array[t, :3]
                        for t in range(len(recorded_data.time_array))
                    ],
                    quaternion=[
                        rr.Quaternion(
                            xyzw=recorded_data.goal_root_states_array[t, 3:7],
                        )
                        for t in range(len(recorded_data.time_array))
                    ],
                    axis_length=[
                        AXES_LENGTH for _ in range(len(recorded_data.time_array))
                    ],
                ),
            )

        # Floating allegro hand
        rr.send_columns(
            "allegro",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    floating_allegro_hand_position[t, :]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=floating_allegro_hand_quat_xyzw[t, :],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
                axis_length=[AXES_LENGTH for _ in range(len(recorded_data.time_array))],
            ),
        )
        allegro_joint_name_to_pos_array = {
            name: recorded_data.robot_joint_positions_array[:, i]
            for i, name in enumerate(recorded_data.robot_joint_names)
            if name in allegro_joint_names
        }
        update_joints_array(
            joint_name_to_pos_array=allegro_joint_name_to_pos_array,
            joint_paths=allegro_joint_paths,
            urdf=allegro_urdf,
            time_array=recorded_data.time_array,
        )

        # Palm
        rr.send_columns(
            "palm",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    palm_xyz_xyzw[t, :3] for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=palm_xyz_xyzw[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
                axis_length=[AXES_LENGTH for _ in range(len(recorded_data.time_array))],
            ),
        )

        # Object relative to floating allegro hand
        rr.send_columns(
            "/allegro/object",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    object_xyz_xyzw_P[t, :3]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=object_xyz_xyzw_P[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
                axis_length=[AXES_LENGTH for _ in range(len(recorded_data.time_array))],
            ),
        )

    elif MODE == "log":
        rr_recording = rr.get_global_data_recording() or rr.RecordingStream(
            application_id=APPLICATION_ID,
        )
        for t in tqdm(range(len(recorded_data.time_array)), desc="Logging data"):
            # Time
            time_seconds = float(
                recorded_data.time_array[t] - recorded_data.time_array[0]
            )
            rr_recording.set_time("tick", duration=timedelta(seconds=time_seconds))

            # Robot
            rr.log(
                "kuka_allegro",
                rr.Transform3D(
                    clear=False,
                    translation=recorded_data.robot_root_states_array[t, :3],
                    quaternion=rr.Quaternion(
                        xyzw=recorded_data.robot_root_states_array[t, 3:7],
                    ),
                    axis_length=AXES_LENGTH,
                ),
            )
            kuka_allegro_joint_name_to_pos = {
                name: recorded_data.robot_joint_positions_array[t, i]
                for i, name in enumerate(recorded_data.robot_joint_names)
            }
            update_joints(
                joint_name_to_pos=kuka_allegro_joint_name_to_pos,
                joint_paths=kuka_allegro_joint_paths,
                urdf=kuka_allegro_urdf,
            )

            # Table
            if recorded_data.table_root_states_array is not None:
                rr.log(
                    "table",
                    rr.Transform3D(
                        clear=False,
                        translation=recorded_data.table_root_states_array[t, :3],
                        quaternion=rr.Quaternion(
                            xyzw=recorded_data.table_root_states_array[t, 3:7],
                        ),
                        axis_length=AXES_LENGTH,
                    ),
                )

            # Goal
            if recorded_data.goal_root_states_array is not None:
                rr.log(
                    "goal",
                    rr.Transform3D(
                        clear=False,
                        translation=recorded_data.goal_root_states_array[t, :3],
                        quaternion=rr.Quaternion(
                            xyzw=recorded_data.goal_root_states_array[t, 3:7],
                        ),
                        axis_length=AXES_LENGTH,
                    ),
                )

            # Object
            rr.log(
                "object",
                rr.Transform3D(
                    clear=False,
                    translation=recorded_data.object_root_states_array[t, :3],
                    quaternion=rr.Quaternion(
                        xyzw=recorded_data.object_root_states_array[t, 3:7],
                    ),
                    axis_length=AXES_LENGTH,
                ),
            )

            # Floating allegro hand
            rr.log(
                "allegro",
                rr.Transform3D(
                    clear=False,
                    translation=floating_allegro_hand_position[t, :],
                    quaternion=rr.Quaternion(
                        xyzw=floating_allegro_hand_quat_xyzw[t, :],
                    ),
                    axis_length=AXES_LENGTH,
                ),
            )
            allegro_joint_name_to_pos = {
                name: recorded_data.robot_joint_positions_array[t, i]
                for i, name in enumerate(recorded_data.robot_joint_names)
                if name in allegro_joint_names
            }
            update_joints(
                joint_name_to_pos=allegro_joint_name_to_pos,
                joint_paths=allegro_joint_paths,
                urdf=allegro_urdf,
            )

            # Palm
            rr.log(
                "palm",
                rr.Transform3D(
                    clear=False,
                    translation=palm_xyz_xyzw[t, :3],
                    quaternion=rr.Quaternion(
                        xyzw=palm_xyz_xyzw[t, 3:7],
                    ),
                    axis_length=AXES_LENGTH,
                ),
            )

            # Object relative to floating allegro hand
            rr.log(
                "/allegro/object",
                rr.Transform3D(
                    clear=False,
                    translation=object_xyz_xyzw_P[t, :3],
                    quaternion=rr.Quaternion(
                        xyzw=object_xyz_xyzw_P[t, 3:7],
                    ),
                    axis_length=AXES_LENGTH,
                ),
            )
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    print("Done logging data: View the rerun viewer now.")
    breakpoint()


if __name__ == "__main__":
    main()
