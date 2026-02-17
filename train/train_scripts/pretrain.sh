#!/bin/bash
#
# Pretraining script for SAPG dexterous manipulation.
#
# Supports two entry points:
#   1. isaacgymenvs.train  — original Hydra-based Isaac Gym pipeline
#   2. train.sapg.train    — new simulator-agnostic pipeline
#
# Set SIMULATOR to choose backend: isaacgym (default), mujoco, mochi

# ---- Simulator selection ----
SIMULATOR="${SIMULATOR:-isaacgym}"

# ---- Multi-GPU ----
# Set NUM_GPUS>1 to enable multi-GPU training (e.g. NUM_GPUS=4)
NUM_GPUS="${NUM_GPUS:-1}"

# ---- Task parameters ----
speed=1.5
robotFriction=0.5
tableResetZRange=0.025
resetWhenDropped=True

CUSTOM_EXPERIMENT_NAME="FINETUNE_5x_SLOW_SPEED_NO_ACTION_DELAY"
WANDB_GROUP="FINETUNE_5x"

WANDB_ENTITY="samratsahoo-stanford-university"
WANDB_PROJECT="simpretrain"
OBJECT_TYPE="tyler_handle_head"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

# ---- Depth obs (set USE_DEPTH=1 to enable) ----
USE_DEPTH="${USE_DEPTH:-0}"
if [ "$USE_DEPTH" = "1" ]; then
    DEPTH_FLAGS=(
        "env.useDepthObservation=True"
        "env.obsList=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,depth_image]"
    )
    # Reduce envs when rendering depth cameras
    NUM_ENVS=4096
    MINIBATCH_SIZE=16384
else
    DEPTH_FLAGS=()
    NUM_ENVS=24576
    MINIBATCH_SIZE=98304
fi

# ---- Launcher: torchrun for multi-GPU, plain python otherwise ----
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS}"
    MULTI_GPU_FLAG="True"
else
    LAUNCHER="python"
    MULTI_GPU_FLAG="False"
fi

if [ "$SIMULATOR" = "isaacgym" ]; then
    # ================================================================
    # Original Isaac Gym path (Hydra-based, fully backward compatible)
    # ================================================================
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
    wandb_project=${WANDB_PROJECT} \
    wandb_entity=${WANDB_ENTITY} \
    wandb_activate=True \
    wandb_group=${WANDB_GROUP} \
    wandb_tags=[] \
    ++wandb_notes='' \
    seed=0 \
    experiment=00_${EXPERIMENT_NAME} \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    task.env.object_type=${OBJECT_TYPE} \
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
    task.env.tableResetZRange=${tableResetZRange} \
    task.env.resetWhenDropped=${resetWhenDropped} \
    task.env.armMovingAverage=0.1 \
    train.params.config.max_frames=100_000_000_000_000

else
    # ================================================================
    # New simulator-agnostic path (MuJoCo, Mochi, or Isaac Gym via train.sapg)
    # ================================================================

    # Mochi needs the assets path set
    if [ "$SIMULATOR" = "mochi" ]; then
        export MOCHI_ASSETS_PATH="${MOCHI_ASSETS_PATH:-/juno/u/satvik/mochi/assets}"
    fi

    # MuJoCo may need osmesa for headless rendering
    if [ "$SIMULATOR" = "mujoco" ]; then
        export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
    fi

    ${LAUNCHER} -m train.sapg.train \
    simulator=${SIMULATOR} \
    multi_gpu=${MULTI_GPU_FLAG} \
    env.numEnvs=${NUM_ENVS} \
    env.objectBaseSize=0.04 \
    env.kukaActionsPenaltyScale=0.03 \
    env.allegroActionsPenaltyScale=0.003 \
    env.stateList=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,palm_vel,object_rot,object_vel,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,closest_keypoint_max_dist,closest_fingertip_dist,lifted_object,progress,successes,reward] \
    env.obsList=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales] \
    env.controlFrequencyInv=1 \
    env.useObsDelay=True \
    env.useActionDelay=False \
    env.useObjectStateDelayNoise=True \
    env.jointVelocityObsNoiseStd=0.01 \
    env.successSteps=10 \
    env.useRelativeControl=False \
    env.dofSpeedScale=${speed} \
    env.robotFriction=${robotFriction} \
    env.tableResetZRange=${tableResetZRange} \
    env.resetWhenDropped=${resetWhenDropped} \
    env.armMovingAverage=0.1 \
    train.params.config.minibatch_size=${MINIBATCH_SIZE} \
    train.params.config.max_frames=100_000_000_000_000 \
    wandb_activate=True \
    wandb_project=${WANDB_PROJECT} \
    wandb_entity=${WANDB_ENTITY} \
    wandb_group=${WANDB_GROUP} \
    seed=0 \
    headless=True \
    "${DEPTH_FLAGS[@]}"
fi

# ---- Usage examples ----
# Isaac Gym (default, single GPU):
#   bash train_scripts/pretrain.sh
#
# Isaac Gym multi-GPU (4 GPUs):
#   NUM_GPUS=4 bash train_scripts/pretrain.sh
#
# MuJoCo:
#   SIMULATOR=mujoco bash train_scripts/pretrain.sh
#
# Mochi:
#   SIMULATOR=mochi bash train_scripts/pretrain.sh
#
# Isaac Gym with depth observations:
#   USE_DEPTH=1 bash train_scripts/pretrain.sh
#
# MuJoCo with depth observations:
#   SIMULATOR=mujoco USE_DEPTH=1 bash train_scripts/pretrain.sh
