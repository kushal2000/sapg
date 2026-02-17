#!/bin/bash

CHECKPOINT=/share/portal/kk837/sapg/train_dir/final_asymmetric_runs/PRETRAIN/SYMMETRIC_ORIG_OBS_2025-12-05_21-03-15/runs/00_SYMMETRIC_ORIG_OBS_2025-12-05_21-03-15/last/model.pth
CUSTOM_EXPERIMENT_NAME="FASTER_TYLER_CURRICULUM_DELTA_GOAL"
WANDB_GROUP="hyperparamChanges"

WANDB_ENTITY="kk837"
WANDB_PROJECT="NEW_FINETUNING"
OBJECT_TYPE="cuboid"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

python -m isaacgymenvs.train \
task=AllegroKukaLSTM \
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
checkpoint=${CHECKPOINT} \
task.env.turn_off_object_vel_obs_slowly=True \
task.env.turn_off_extra_obs_slowly=True \
task.env.use_obs_dropout=True \
task.env.dofSpeedScale=10 \
task.env.dofSpeedScaleFinal=2.5 \
task.env.curriculumSuccessRatio=0.1 \
task.env.forceScale=0.0 \
task.task.randomize=False \
task.env.init_tyler_curriculum_scale=0.0 \
task.env.episodeLength=600 \
task.env.controlFrequencyInv=1 \
task.env.goalSamplingType=delta \
task.env.targetVolumeRegionScale=1.0 \
task.env.deltaGoalDistance=0.1 \
task.env.deltaRotationDegrees=45.0 \
task.env.timeToUpdateTylerCurriculum=5 \
task.env.updateStepSizeTylerCurriculum=0.1 \
# task.task.randomization_params.actor_params.object.scale.range=[0.9,1.1] \
# task.task.randomization_params.actor_params.allegro.scale.range=[0.9,1.1] \
# task.env.withTableForceSensor=True \