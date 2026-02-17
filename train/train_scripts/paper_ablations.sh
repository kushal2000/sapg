#!/bin/bash

fixedSizeKeypointReward=True
forceOnlyWhenLifted=True
forceScale=2

CUSTOM_EXPERIMENT_NAME="Ours"
WANDB_GROUP="1x"

WANDB_ENTITY="kk837"
WANDB_PROJECT="PAPER_ABLATIONS"
OBJECT_TYPE="tyler_handle_head"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

python -m isaacgymenvs.train \
task/env=reorientation \
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
task.env.useActionDelay=True \
task.env.useObjectStateDelayNoise=True \
task.env.jointVelocityObsNoiseStd=0.01 \
task.env.successSteps=10 \
task.env.goalSamplingType=delta \
task.env.dofSpeedScale=1.5 \
task.env.robotFriction=0.5 \
task.env.tableResetZRange=0.01 \
task.env.resetWhenDropped=True \
task.env.armMovingAverage=0.1 \
train.params.config.max_frames=100_000_000_000_000 \
task.env.fixedSizeKeypointReward=${fixedSizeKeypointReward} \
task.env.forceOnlyWhenLifted=${forceOnlyWhenLifted} \
task.env.forceScale=${forceScale} \
task.env.targetSuccessTolerance=0.075 \


