# SAPG: Split and Aggregate Policy Gradients (ICML 2024 Oral) 

# TYLER README (Last Updated: November 4, 2025)

## Rough Installation Instructions

We need two conda envs: one for training with isaacgym (requires Python 3.8) and one for sim2real (Robostack requires Python 3.11+).

### IsaacGym Env

```
conda create -n sapg_env python=3.8  # isaacgym requires Python 3.8
conda activate sapg_env

# Misc
pip install \
  absl-py aiohttp aiosignal async-timeout attrs backports.functools-lru-cache \
  blinker cachetools certifi cffi charset-normalizer click cryptography debugpy decorator \
  google-auth google-auth-oauthlib grpcio idna importlib-metadata ipykernel ipython \
  jupyter-client jupyter-core markdown markupsafe matplotlib matplotlib-inline nest-asyncio \
  numpy oauthlib packaging parso pexpect pickleshare pillow prompt-toolkit protobuf psutil \
  pyasn1 pyasn1-modules pycparser pygments pyjwt pyopenssl pysocks python-dateutil \
  torch torchvision pyyaml pyzmq requests requests-oauthlib rsa scipy setuptools six \
  tensorboard tensorboard-data-server tensorboard-plugin-wit tk tornado traitlets \
  typing-extensions urllib3 werkzeug wheel \
  antlr4-python3-runtime cloudpickle cycler docker-pycreds fasteners fonttools freetype-py \
  gitpython gym gym-notices hydra-core imageio jinja2 kiwisolver lxml matplotlib networkx \
  ninja omegaconf opencv-python pathtools promise pyglet pyopengl pyparsing pyrender pysdf \
  pyvirtualdisplay sentry-sdk setproctitle shortuuid smmap tensorboardx termcolor transforms3d \
  trimesh urdfpy wandb warp-lang

# More dependencies
pip install pytorch3d "imageio[ffmpeg]" yourdfpy viser pytorch_kinematics mujoco

# Download the Isaac Gym Preview 4 release from https://developer.nvidia.com/isaac-gym
cd isaacgym/python
pip install -e .
pip install numpy==1.23.0  # isaacgym does not support numpy 1.24+

# Install this repo's rl_games
cd <this repo>/rl_games
pip install -e .

# Install this repo
cd <this repo>
pip install -e .
```

### Sim2Real Env

We use RoboStack with ROS1 Noetic: https://robostack.github.io/noetic.html.

```
conda create -n sapg_ros_env -c conda-forge -c robostack-noetic ros-noetic-desktop python=3.11  # Robostack requires Python 3.11+
conda activate sapg_ros_env
conda config --env --add channels robostack-noetic

# Misc
pip install \
  absl-py aiohttp aiosignal async-timeout attrs backports.functools-lru-cache \
  blinker cachetools certifi cffi charset-normalizer click cryptography debugpy decorator \
  google-auth google-auth-oauthlib grpcio idna importlib-metadata ipykernel ipython \
  jupyter-client jupyter-core markdown markupsafe matplotlib matplotlib-inline nest-asyncio \
  numpy oauthlib packaging parso pexpect pickleshare pillow prompt-toolkit protobuf psutil \
  pyasn1 pyasn1-modules pycparser pygments pyjwt pyopenssl pysocks python-dateutil \
  torch torchvision pyyaml pyzmq requests requests-oauthlib rsa scipy setuptools six \
  tensorboard tensorboard-data-server tensorboard-plugin-wit tk tornado traitlets \
  typing-extensions urllib3 werkzeug wheel \
  antlr4-python3-runtime cloudpickle cycler docker-pycreds fasteners fonttools freetype-py \
  gitpython gym gym-notices hydra-core imageio jinja2 kiwisolver lxml matplotlib networkx \
  ninja omegaconf opencv-python pathtools promise pyglet pyopengl pyparsing pyrender pysdf \
  pyvirtualdisplay sentry-sdk setproctitle shortuuid smmap tensorboardx termcolor transforms3d \
  trimesh urdfpy wandb warp-lang

# Imageio for saving video
pip install "imageio[ffmpeg]"

# Viser
pip install viser

# Pytorch kinematics
pip install pytorch_kinematics

# Mujoco
pip install mujoco

# Install this repo's rl_games
cd <this repo>/rl_games
pip install -e .

# Install this repo
cd <this repo>
pip install -e .
```

## Pretrain (Standard Cuboid)

Either use our training script:
```
bash train/train_scripts/pretrain.sh
```

Or use the following command:

```
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
wandb_project=sapg_allegro_kuka_reorientation \
wandb_entity=<ENTITY_NAME> \
wandb_activate=True \
wandb_group=<GROUP_NAME> \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_<EXPERIMENT_NAME> \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/<EXPERIMENT_NAME> \
task.env.object_type='cuboid'
```


## Finetune (Marker)

Either use our training script:
```
bash train/train_scripts/finetune.sh
```

Or use the following command:

```
python -m \
isaacgymenvs.train \
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
wandb_project=sapg_allegro_kuka_reorientation \
wandb_entity=<ENTITY_NAME> \
wandb_activate=True \
wandb_group=<GROUP_NAME> \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_<EXPERIMENT_NAME> \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/<EXPERIMENT_NAME> \
checkpoint=/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/nn/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth \
task.env.object_type='040_large_marker' \
task.env.use_fixed_set_of_goal_states=False \
task.env.use_fixed_init_object_pose=False
```

## Play Pretrained Policy

Either use our training script:
```
bash train/train_scripts/play.sh
```

Or use the following command:

```
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
wandb_project=sapg_allegro_kuka_reorientation \
wandb_entity=<ENTITY_NAME> \
wandb_activate=True \
wandb_group=<GROUP_NAME> \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_<EXPERIMENT_NAME> \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/<EXPERIMENT_NAME> \
checkpoint=/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/nn/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth \
task.env.object_type='cuboid' \
task.env.use_fixed_set_of_goal_states=False \
task.env.use_fixed_init_object_pose=False \
test=True \
task.env.numEnvs=100 \
headless=False \
task.env.envSpacing=1.0 \
task.env.maxConsecutiveSuccesses=10 \
task.env.enableDebugVis=True
```


## Play Finetuned Policy (Screwdriver) With Fixed Goal: `task.env.use_fixed_set_of_goal_states=True`

```
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
wandb_project=sapg_allegro_kuka_reorientation \
wandb_entity=<ENTITY_NAME> \
wandb_activate=True \
wandb_group=<GROUP_NAME> \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_<EXPERIMENT_NAME> \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/<EXPERIMENT_NAME> \
checkpoint=/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/nn/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth \
task.env.object_type='044_flat_screwdriver' \
task.env.use_fixed_set_of_goal_states=True \
task.env.use_fixed_init_object_pose=False \
test=True \
task.env.numEnvs=100 \
headless=False \
checkpoint=/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/finetune_screwdriver_notfixedinit_notfixedgoal/runs/00_finetune_screwdriver_notfixedinit_notfixedgoal/best/model.pth \
task.env.envSpacing=1.0 \
task.env.maxConsecutiveSuccesses=10 \
task.env.enableDebugVis=True
```

# Sim2Sim

For Sim2Sim, there are a few "axes" of variation:

* Simulator: Mujoco vs. Isaac

* Communication: ROS (async different processes) vs. No ROS (same process synchronous)

Convention: `env` refers to a wrapper around simulator that has a `step` method and returns an observation.

## Mujoco

Standard Mujoco simulation:

```
python eval/sim2sim/mujoco_sim/mujoco_sim.py
```

Mujoco environment with no ROS:

```
python eval/sim2sim/mujoco_sim/mujoco_env_no_ros.py
```

Mujoco environment with ROS:

```
python eval/sim2sim/mujoco_sim/mujoco_env_ros.py
```

## Isaac

For Isaac, we can either have the env step with raw actions in `[-1, 1]` or with joint position targets.

Standard Isaac environment:

```
python eval/sim2sim/isaac_sim/isaac_env.py
```

Isaac environment with no ROS:

```
python eval/sim2sim/isaac_sim/isaac_env_no_ros.py
```

Isaac environment with no ROS and joint position targets:

```
python eval/sim2sim/isaac_sim/isaac_env_no_ros_joint_pos_targets.py
```

As of right now, we have not implemented the following because isaacgym requires Python 3.8 and robostack requires Python 3.10+ (can do this if use system-level ROS):

```
python eval/sim2sim/isaac_sim/isaac_env_ros_joint_pos_targets.py
```


# Sim2Real

For Sim2Real policy deployment, we will require at least 3 nodes:

1. RL Policy Node: Takes in observations, runs policy to get raw actions, converts to joint position targets, and publishes to robot.
2. Perception Node: Reads in RGB-D images, uses SAM2 and FoundationPose to get object pose, and publishes to robot.
3. Robot Node: Sends joint position targets to robot and publishes joint states to ROS.

(2) and (3) are not in this repo.

The RL Policy Node is in this repo:

```
python eval/sim2real/rl_policy_ros_node.py
```

Home robot:

```
python eval/sim2real/home_robot.py
```

Open-loop replay of joint position trajectory:

```
python eval/sim2real/replay_trajectory.py
```

Visualizer node:

```
python eval/sim2real/visualizer.py
```

If want to try without a real robot, you can either use mujoco sim2sim (will publish object pose and goal object pose):

```
python eval/sim2sim/mujoco_sim/mujoco_env_ros.py
```

Or fake_robot_ros_node.py (no physics, just interpolating to joint position targets):

```
python eval/sim2real/fake_robot_ros_node.py
```

Move hand to "limp" position:

```
rostopic pub /allegroHand_0/joint_cmd sensor_msgs/JointState "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
position: [0.03580158647006279, 1.190307500756139, 0.04091241471899582, -0.0020815739716152164, -0.003517249230515697, 1.2851153897506231, 0.044026046173861466, 0.014320749234448864, -0.026443060708318096, 1.3508007502819834, 0.019888673216377658, 0.0169404863577189, 1.3616900779442058, 0.01507557136958743, 0.1047496180391897, 0.009729167245470401]
velocity: []
effort: []"
```

# Viser

## URDF Files

Standard URDF visualization with viser (mostly from Viser example code):

```
python viser_urdf/viser_urdf.py
```

Standard URDF visualization with viser with additional visualization of each of the frames of the robot (X = Red, Y = Green, Z = Blue).

```
python viser_urdf/viser_urdf_with_frames.py
```

## MJCF XML Files

MJCF XML visualization with viser (motivated by MjLabs code but heavily simplified):

```
python viser_mujoco/viser_mj_model.py
```

MJCF XML visualization with viser with additional visualization of each of the frames of the robot (X = Red, Y = Green, Z = Blue).

```
python viser_mujoco/viser_mj_model_with_frames.py
```

Visualize the MujocoSim with viser.

```
python viser_mujoco/viser_sim.py
```

Visualize the MujocoSim and compare with the URDF visualization.

```
python viser_mujoco/viser_sim_urdf_comparison.py
```

# ORIGINAL README
[![arXiv](https://img.shields.io/badge/arXiv-2407.20230-df2a2a.svg)](https://arxiv.org/abs/2407.20230)
[![Static Badge](https://img.shields.io/badge/Project-sapg-a)](https://sapg-rl.github.io)
[![Python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the algorithm **Split and Aggregate Policy Gradients**. 

## Performance of SAPG

![SAPG training plots](figures/main_plot_new.png)

We evaluate SAPG on a variety of complex robotic tasks and find that it outperforms state-of-the-art algorithms such as DexPBT [[1]](#1) and PPO [[2]](#2). In all environments, SAPG obtains the highest asympototic successes/reward, while also being most sample-efficient in nearly all situations. 

## Training
Use one of the following commands to train a policy using SAPG for any of the IsaacGym environments

```bash
conda activate sapg
export LD_LIBRARY_PATH=$(conda info --base)/envs/sapg/lib:$LD_LIBRARY_PATH
# For Allegro Kuka tasks - Reorientation, Regrasping and Throw
./scripts/train_allegro_kuka.sh <TASK> <EXPERIMENT_PREFIX> 1 <NUM_ENVS> [] --sapg --lstm --num-expl-coef-blocks=<NUMBER_OF_SAPG_BLOCKS> --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=<ENTROPY_COEFFICIENT_SCALE>

# For Allegro Kuka Two Arms tasks - Reorientation and Regrasping
./scripts/train_allegro_kuka_two_arms.sh <TASK> <EXPERIMENT_PREFIX> 1 <NUM_ENVS> [] --sapg --lstm --num-expl-coef-blocks=<NUMBER_OF_SAPG_BLOCKS> --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=<ENTROPY_COEFFICIENT_SCALE>

# For Shadow Hand and Allegro Hand
./scripts/train.sh <ENV> <EXPERIMENT_PREFIX> 1 <NUM_ENVS> [] --sapg --lstm --num-expl-coef-blocks=<NUMBER_OF_SAPG_BLOCKS> --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=<ENTROPY_COEFFICIENT_SCALE>
```

### Distributed training

The code supports distributed training too. The template for multi-GPU training is as follows

```bash
# Distributed training for the AllegroKuka tasks 
./scripts/train_allegro_kuka.sh <TASK> <EXPERIMENT_PREFIX> <NUM_PROCESSES> <NUM_ENVS_PER_PROCESS> [] --sapg --lstm --num-expl-coef-blocks=<NUMBER_OF_SAPG_BLOCKS> --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=<ENTROPY_COEFFICIENT_SCALE> --multi-gpu
```

## Inference
To visualize performance of one of your checkpoints, execute run the following commands

```bash
conda activate sapg
export LD_LIBRARY_PATH=$(conda info --base)/envs/sapg/lib:$LD_LIBRARY_PATH
python3 play.py --checkpoint <PATH_TO_CHECKPOINT> --num_envs <NUM_ENVS>
```

**Note**: The path to the checkpoint must be its original path when the checkpoint was created to ensure that evaluation can be run using the correct config. 

## Quickstart

Clone the repository and create a Conda environment using the ```env.yaml``` file.
```bash
conda env create -f env.yaml
conda activate sapg
```

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym) and executing the following after unzipping the downloaded file
```bash
cd isaacgym/python
pip install -e .
```

Now, in the root folder of the repository, execute the following commands,
```bash
cd rl_games
pip install -e . 
cd ..
pip install -e .
```

### Reproducing performance
 
We provide the exact commands which can be used to reproduce the performance of policies trained with SAPG as well as PPO on different environments

```bash
# Allegro Kuka Regrasping
./scripts/train_allegro_kuka.sh regrasping "test" 1 24576 [] --sapg --lstm --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=none

./scripts/train_allegro_kuka.sh regrasping "test" 1 24576 [] --lstm --wandb-entity <ENTITY_NAME> # PPO

# Allegro Kuka Throw
./scripts/train_allegro_kuka.sh throw "test" 1 24576 [] --sapg --lstm --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=none

./scripts/train_allegro_kuka.sh throw "test" 1 24576 [] --lstm --wandb-entity <ENTITY_NAME> # PPO

# Allegro Kuka Reorientation
./scripts/train_allegro_kuka.sh reorientation "test" 1 24576 [] --sapg --lstm --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=0.005

./scripts/train_allegro_kuka.sh reorientation "test" 1 24576 [] --lstm --wandb-entity <ENTITY_NAME> # PPO

# Allegro Kuka Two Arms Reorientation (Multi-GPU run)
./scripts/train_allegro_kuka_two_arms.sh reorientation "test" 6 4104  [] --sapg --lstm --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=0.002 --multi-gpu

./scripts/train_allegro_kuka_two_arms.sh reorientation "test" 6 4104  [] --lstm --wandb-entity <ENTITY_NAME> --multi-gpu # PPO

# In-hand reorientation with Shadow Hand
./scripts/train.sh shadow_hand "test" 1 24576 [] --sapg --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=entropy --ir-coef-scale=0.005

./scripts/train.sh shadow_hand "test" 1 24576 [] --wandb-entity <ENTITY_NAME> # PPO

# In-hand reorientation with Allegro Hand
./scripts/train.sh allegro_hand "test" 1 24576 [] --sapg --num-expl-coef-blocks=6 --wandb-entity <ENTITY_NAME> --ir-type=none

./scripts/train.sh allegro_hand "test" 1 24576 [] --wandb-entity <ENTITY_NAME> # PPO

```

## Citation
If you find our code useful, please cite our work
```
@inproceedings{sapg2024,
  title     = {SAPG: Split and Aggregate Policy Gradients},
  author    = {Singla, Jayesh and Agarwal, Ananye and Pathak, Deepak},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML 2024)},
  month     = {July},
  year      = {2024},
  publisher = {PMLR},
}
```

## Acknowledgements
This implementation builds upon the the following codebases - 
1. [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
2. [rl_games](https://github.com/Denys88/rl_games)

## References

<small><small>
<a id="1">[1]</a> 
Petrenko, A., Allshire, A., State, G., Handa, A., & Makoviychuk, V. (2023). DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training. ArXiv, abs/2305.12127.
<a id="2">[2]</a>
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. ArXiv, abs/1707.06347.
</small></small>
