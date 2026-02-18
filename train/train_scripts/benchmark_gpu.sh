#!/bin/bash
#
# GPU Utilization Benchmark Harness
#
# Measures GPU SM utilization and training timing during RL training.
# Runs as a wrapper: starts nvidia-smi dmon in a separate process,
# launches training for a configurable number of epochs, then parses results.
#
# Usage:
#   bash train/train_scripts/benchmark_gpu.sh [NUM_GPUS] [MAX_EPOCHS]
#
# Examples:
#   bash train/train_scripts/benchmark_gpu.sh 1 15    # 1 GPU, 15 epochs
#   bash train/train_scripts/benchmark_gpu.sh 2 15    # 2 GPUs, 15 epochs
#
# Output: prints summary table of GPU utilization and timing stats.

set -uo pipefail

# Activate conda environment
source /juno/u/satvik/miniconda3/etc/profile.d/conda.sh
conda activate sapg

cd /juno/u/satvik/sapg

NUM_GPUS="${1:-1}"
MAX_EPOCHS="${2:-15}"
WARMUP_EPOCHS="${3:-3}"

# Output directory
BENCH_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)_${NUM_GPUS}gpu"
mkdir -p "$BENCH_DIR"

DMON_LOG="$BENCH_DIR/gpu_dmon.csv"
TRAIN_LOG="$BENCH_DIR/train_stdout.log"
SUMMARY="$BENCH_DIR/summary.txt"

echo "=== GPU Utilization Benchmark ==="
echo "  GPUs:         $NUM_GPUS"
echo "  Max epochs:   $MAX_EPOCHS"
echo "  Warmup:       $WARMUP_EPOCHS"
echo "  Output dir:   $BENCH_DIR"
echo ""

# --- Step 1: Start nvidia-smi dmon in background ---
# -s u: utilization metrics (SM%, mem%, enc%, dec%)
# -d 1: sample every 1 second
# -o DT: include date+time in output
nvidia-smi dmon -s u -d 1 -o DT > "$DMON_LOG" 2>&1 &
DMON_PID=$!
echo "Started nvidia-smi dmon (PID=$DMON_PID), logging to $DMON_LOG"

# Cleanup dmon on exit
cleanup() {
    if kill -0 "$DMON_PID" 2>/dev/null; then
        kill "$DMON_PID" 2>/dev/null || true
        wait "$DMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Step 2: Launch training ---
echo "Launching training for $MAX_EPOCHS epochs..."

# Build launcher command
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS}"
    MULTI_GPU_FLAG="True"
else
    LAUNCHER="python"
    MULTI_GPU_FLAG="False"
fi

NUM_ENVS=24576
MINIBATCH_SIZE=98304
speed=1.5
robotFriction=0.5

EXPERIMENT_NAME="00_bench_$(date +%s)"
HYDRA_RUN_DIR="$BENCH_DIR/hydra_run"

# Run training, capturing stdout+stderr
${LAUNCHER} -m isaacgymenvs.train \
    task/env=reorientation \
    ++task.env.useSparseReward=False \
    headless=True \
    task.env.numEnvs=${NUM_ENVS} \
    train.params.config.minibatch_size=${MINIBATCH_SIZE} \
    multi_gpu=${MULTI_GPU_FLAG} \
    train.params.config.good_reset_boundary=0 \
    task.env.goodResetBoundary=0 \
    train.params.config.use_others_experience=lf \
    train.params.config.off_policy_ratio=1.0 \
    train.params.config.expl_type=mixed_expl_learn_param \
    train.params.config.expl_reward_type=entropy \
    train.params.config.expl_coef_block_size=4096 \
    train.params.config.expl_reward_coef_scale=0.005 \
    train.params.network.space.continuous.fixed_sigma=coef_cond \
    wandb_activate=False \
    seed=0 \
    experiment=${EXPERIMENT_NAME} \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    task.env.object_type=tyler_handle_head \
    task.env.useRelativeControl=False \
    task.task.randomize=False \
    task=AllegroKukaLSTMAsymmetric \
    task.env.objectBaseSize=0.04 \
    task.env.kukaActionsPenaltyScale=0.03 \
    task.env.allegroActionsPenaltyScale=0.003 \
    task.env.stateList=["joint_pos","joint_vel","prev_action_targets","palm_pos","palm_rot","palm_vel","object_rot","object_vel","fingertip_pos_rel_palm","keypoints_rel_palm","keypoints_rel_goal","object_scales","closest_keypoint_max_dist","closest_fingertip_dist","lifted_object","progress","successes","reward"] \
    task.env.obsList=["joint_pos","joint_vel","prev_action_targets","palm_pos","palm_rot","object_rot","fingertip_pos_rel_palm","keypoints_rel_palm","keypoints_rel_goal","object_scales"] \
    task.env.use_fixed_set_of_goal_states=False \
    task.env.controlFrequencyInv=1 \
    task.env.useObsDelay=True \
    task.env.useActionDelay=False \
    task.env.useObjectStateDelayNoise=True \
    task.env.jointVelocityObsNoiseStd=0.01 \
    task.env.successSteps=10 \
    task.env.goalSamplingType=delta \
    task.env.dofSpeedScale=${speed} \
    task.env.robotFriction=${robotFriction} \
    task.env.tableResetZRange=0.025 \
    task.env.resetWhenDropped=True \
    task.env.armMovingAverage=0.1 \
    train.params.config.max_epochs=${MAX_EPOCHS} \
    2>&1 | tee "$TRAIN_LOG"

# Stop dmon
kill "$DMON_PID" 2>/dev/null || true
wait "$DMON_PID" 2>/dev/null || true

echo ""
echo "=== Parsing Results ==="

# --- Step 3: Parse GPU utilization from dmon log ---
# dmon format: date time gpu sm mem enc dec
# Skip header lines (start with #) and warmup period
python3 - "$DMON_LOG" "$WARMUP_EPOCHS" "$TRAIN_LOG" "$SUMMARY" <<'PYEOF'
import sys
import re
import numpy as np

dmon_log = sys.argv[1]
warmup_epochs = int(sys.argv[2])
train_log = sys.argv[3]
summary_file = sys.argv[4]

# Parse training timings from stdout
play_times = []
update_times = []
total_times = []

with open(train_log) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'Play time\s*:\s*([\d.]+)', line)
        if m:
            play_times.append(float(m.group(1)))
        m = re.match(r'Update time\s*:\s*([\d.]+)', line)
        if m:
            update_times.append(float(m.group(1)))
        m = re.match(r'Time to train epoch\s*:\s*([\d.]+)', line)
        if m:
            total_times.append(float(m.group(1)))

# Skip warmup epochs
play_times = play_times[warmup_epochs:]
update_times = update_times[warmup_epochs:]
total_times = total_times[warmup_epochs:]

# Parse GPU SM utilization from dmon log
sm_values = {}  # gpu_id -> list of values
with open(dmon_log) as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.split()
        # Format: date time gpu sm mem enc dec
        if len(parts) >= 5:
            try:
                gpu_id = int(parts[2])
                sm = int(parts[3])
                if gpu_id not in sm_values:
                    sm_values[gpu_id] = []
                sm_values[gpu_id].append(sm)
            except (ValueError, IndexError):
                continue

# Skip warmup samples (rough: first 30% of samples correspond to warmup epochs)
for gpu_id in sm_values:
    total = len(sm_values[gpu_id])
    skip = int(total * warmup_epochs / (warmup_epochs + len(total_times))) if total_times else 0
    sm_values[gpu_id] = sm_values[gpu_id][skip:]

lines = []
lines.append("=" * 60)
lines.append("GPU UTILIZATION BENCHMARK RESULTS")
lines.append("=" * 60)
lines.append("")

# GPU utilization summary
lines.append("GPU SM Utilization (after warmup):")
lines.append("-" * 40)
for gpu_id in sorted(sm_values.keys()):
    vals = np.array(sm_values[gpu_id])
    if len(vals) > 0:
        lines.append(f"  GPU {gpu_id}:")
        lines.append(f"    Mean:   {vals.mean():.1f}%")
        lines.append(f"    Median: {np.median(vals):.1f}%")
        lines.append(f"    Min:    {vals.min():.1f}%")
        lines.append(f"    Max:    {vals.max():.1f}%")
        lines.append(f"    Std:    {vals.std():.1f}%")
        lines.append(f"    Samples: {len(vals)}")

lines.append("")
lines.append("Training Timing (per epoch, after warmup):")
lines.append("-" * 40)

if play_times:
    lines.append(f"  Play time:   mean={np.mean(play_times):.3f}s, std={np.std(play_times):.3f}s")
if update_times:
    lines.append(f"  Update time: mean={np.mean(update_times):.3f}s, std={np.std(update_times):.3f}s")
if total_times:
    lines.append(f"  Total time:  mean={np.mean(total_times):.3f}s, std={np.std(total_times):.3f}s")
if play_times and update_times:
    lines.append(f"  Update/Total ratio: {np.mean(update_times)/np.mean(total_times)*100:.1f}%")

lines.append("")
lines.append(f"  Measured epochs: {len(total_times)}")
lines.append("=" * 60)

output = "\n".join(lines)
print(output)

with open(summary_file, 'w') as f:
    f.write(output + "\n")

print(f"\nFull results saved to: {summary_file}")
PYEOF
