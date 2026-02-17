#!/bin/bash

CHECKPOINT=/juno/u/kedia/sapg/train_dir/allegro_kuka_reorientation/PRETRAIN/JOINT_POS_TARGETS_AS_BLUE_ROBOT_2025-11-13_16-18-18/runs/00_JOINT_POS_TARGETS_AS_BLUE_ROBOT_2025-11-13_16-18-18/best/model.pth
CUSTOM_EXPERIMENT_NAME="DR_RELATIVE_SLOW"
WANDB_GROUP="PLAY"

WANDB_ENTITY="kk837"
WANDB_PROJECT="sapg_allegro_kuka_reorientation"
OBJECT_TYPE="scanned_hammer_2_coacd"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/allegro_kuka_reorientation/${WANDB_GROUP}/${EXPERIMENT_NAME}

python -m isaacgymenvs.train \
task=AllegroKukaLSTM \
task/env=pose_reaching \
++task.env.useSparseReward=False \
headless=True \
task.env.numEnvs=24576 \
train.params.config.minibatch_size=98304 \
multi_gpu=False \
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
task.task.randomize=True \
checkpoint=${CHECKPOINT} \
task.env.turn_off_object_vel_obs_slowly=True \
task.env.turn_off_extra_obs_slowly=True \
task.env.use_obs_dropout=True \
task.env.dofSpeedScale=10 \
task.env.dofSpeedScaleFinal=1.0 \
task.env.curriculumSuccessRatio=0.6 \
test=True \
task.env.numEnvs=1 \
headless=False \
task.env.use_fixed_init_object_pose=True \
task.env.use_fixed_set_of_goal_states=True \
task.env.resetDofPosRandomIntervalFingers=0.0 \
task.env.resetDofPosRandomIntervalArm=0.0 \
task.env.resetDofVelRandomInterval=0.0 \
task.env.init_tyler_curriculum_scale=1.0 \
task.env.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT=True \
train.params.config.player.deterministic=True \