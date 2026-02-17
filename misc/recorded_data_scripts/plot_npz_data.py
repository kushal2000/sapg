import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import base64
import io

# filepath = Path("/juno/u/kedia/sapg/recorded_robot_state/2025-11-22_12-10-53.npz")
# filepath = Path("/juno/u/kedia/sapg/recorded_robot_state/sim_rollout_2025-11-24_17-34-52.npz")
# filepath = Path("/juno/u/kedia/sapg/recorded_robot_state/sim_rollout_2025-11-24_17-47-54.npz")
# filepath = Path("/juno/u/kedia/sapg/recorded_robot_state/sim_rollout_2025-11-24_17-46-31.npz")
filepath = Path("/juno/u/kedia/sapg/recorded_robot_state/sim_rollout_2025-11-24_17-53-22.npz")
assert filepath.exists(), f"File {filepath} does not exist"

joint_data = np.load(filepath, allow_pickle=True)

# Using correct keys based on file inspection
joint_positions = joint_data["robot_joint_positions_array"]
joint_targets = joint_data["robot_joint_pos_targets_array"]

# breakpoint()

num_joints = joint_positions.shape[1]
print(f"Found {num_joints} joints.")

html_content = [
    "<html>",
    "<head><title>Joint Plots</title></head>",
    "<body>",
    "<h1>Joint Plots</h1>"
]

for i in range(num_joints):
    print(f"Plotting joint {i}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(joint_positions[:, i], label="Position")
    if joint_targets is not None:
        ax.plot(joint_targets[:, i], label="Target", linestyle="--")
    
    ax.set_title(f"Joint: {i}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    # Save plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    
    html_content.append(f"<h2>{i}</h2>")
    html_content.append(f'<img src="data:image/png;base64,{img_str}" />')
    html_content.append("<hr>")

html_content.append("</body></html>")

output_path = filepath.with_suffix(".html")
with open(output_path, "w") as f:
    f.write("\n".join(html_content))

print(f"Saved plots to {output_path}")
