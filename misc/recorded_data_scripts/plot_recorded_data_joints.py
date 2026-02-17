import math
from pathlib import Path

import matplotlib.pyplot as plt

from misc.recorded_data_scripts.recorded_data import RecordedData

filepath = Path(
    "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39_None_310.npz"
)
assert filepath.exists(), f"File {filepath} does not exist"
recorded_data = RecordedData.from_file(filepath)

joint_names = recorded_data.robot_joint_names
joint_positions_array = recorded_data.robot_joint_positions_array
joint_pos_targets_array = recorded_data.robot_joint_pos_targets_array

joint_name_to_joint_positions_array = {
    joint_names[i]: joint_positions_array[:, i] for i in range(len(joint_names))
}
joint_name_to_joint_pos_targets_array = {
    joint_names[i]: joint_pos_targets_array[:, i] for i in range(len(joint_names))
}

nrows = math.ceil(math.sqrt(len(joint_names)))
ncols = math.ceil(len(joint_names) / nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
axes = axes.flatten()
for i, joint_name in enumerate(joint_names):
    axes[i].plot(joint_name_to_joint_positions_array[joint_name], label=joint_name)
    axes[i].plot(
        joint_name_to_joint_pos_targets_array[joint_name], label=joint_name + " target"
    )
    axes[i].legend()
plt.tight_layout()
plt.show()
