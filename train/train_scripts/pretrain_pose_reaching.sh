#!/bin/bash

CUSTOM_EXPERIMENT_NAME="LOGGING_WITH_JOINT_ACCELERATION_PENALTY_HIGHER"
WANDB_GROUP="POSE_REACHING_NO_FORCE"

WANDB_ENTITY="tylerlum"
WANDB_PROJECT="sapg_pose_reaching"
OBJECT_TYPE="cuboid"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/pose_reaching/${WANDB_GROUP}/${EXPERIMENT_NAME}

python -m isaacgymenvs.train \
task=AllegroKukaLSTM \
task/env=pose_reaching \
task.env.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT=False \
++task.env.useSparseReward=False \
headless=True \
task.env.numEnvs=12288 \
train.params.config.minibatch_size=98304 \
multi_gpu=False \
train.params.config.good_reset_boundary=0 \
task.env.goodResetBoundary=0 \
train.params.config.use_others_experience=lf \
train.params.config.off_policy_ratio=1.0 \
train.params.config.expl_type=mixed_expl_learn_param \
train.params.config.expl_reward_type=entropy \
train.params.config.expl_coef_block_size=2048 \
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
task.env.dofSpeedScale=10.0 \
task.env.useRelativeControl=False \
task.task.randomize=False \
task.task.randomization_params.actor_params.object.scale.range=[0.999,1.001] \
task.task.randomization_params.actor_params.allegro.scale.range=[0.999,1.001] \
task.env.use_green_robot=True \
task.env.sanity_check_controls=False \
task.env.armMovingAverage=0.1 \
task.env.handMovingAverage=0.1 \
task.env.add_prev_targets_to_obs=False \
task.task.randomize=False \
task.env.forceScale=0.0 \
task.env.jointVelocityPenaltyScale=0 \
task.env.jointAccelerationPenaltyScale=0.001 \
# task.env.episodeLength=100 \