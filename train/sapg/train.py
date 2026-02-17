"""Training entry point for the simulator-agnostic SAPG pipeline.

Usage::

    # Isaac Gym (default)
    python -m train.sapg.train simulator=isaacgym

    # MuJoCo
    python -m train.sapg.train simulator=mujoco

    # Mochi
    python -m train.sapg.train simulator=mochi

    # With depth observations
    python -m train.sapg.train simulator=isaacgym env.useDepthObservation=True

Config resolution order:
    1. ``train/sapg/cfg/env.yaml``        — task-level config
    2. ``train/sapg/cfg/sim/<sim>.yaml``   — simulator-specific params
    3. ``train/sapg/cfg/train/ppo.yaml``   — rl_games training config
    4. CLI overrides
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CFG_DIR = Path(__file__).resolve().parent / "cfg"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    simulator: str = "isaacgym",
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """Load and merge the YAML config hierarchy.

    Returns a single flat dict with keys ``env``, ``sim``, ``train``, and
    top-level convenience keys (``device``, ``simulator``, etc.).
    """
    # Task-level config
    with open(_CFG_DIR / "env.yaml") as f:
        cfg = yaml.safe_load(f)

    # Simulator-specific config
    sim_path = _CFG_DIR / "sim" / f"{simulator}.yaml"
    if sim_path.exists():
        with open(sim_path) as f:
            sim_cfg = yaml.safe_load(f)
        _deep_merge(cfg, sim_cfg)
    else:
        raise FileNotFoundError(f"No sim config found at {sim_path}")

    # Training config
    with open(_CFG_DIR / "train" / "ppo.yaml") as f:
        train_cfg = yaml.safe_load(f)
    cfg["train"] = train_cfg

    # Defaults
    cfg.setdefault("simulator", simulator)
    cfg.setdefault("device", "cuda:0")
    cfg.setdefault("rl_device", "cuda:0")
    cfg.setdefault("sim_device", "cuda:0")
    cfg.setdefault("graphics_device_id", 0)
    cfg.setdefault("headless", True)
    cfg.setdefault("multi_gpu", False)
    cfg.setdefault("seed", 0)
    cfg.setdefault("test", False)
    cfg.setdefault("checkpoint", "")
    cfg.setdefault("sigma", "")
    cfg.setdefault("wandb_activate", False)
    cfg.setdefault("wandb_project", "sapg")
    cfg.setdefault("wandb_entity", "")
    cfg.setdefault("wandb_group", "")
    cfg.setdefault("wandb_name", "")

    # Apply CLI overrides
    if overrides:
        _deep_merge(cfg, overrides)

    return cfg


def parse_cli_overrides(args: list[str]) -> tuple[str, dict]:
    """Parse ``key=value`` pairs from CLI args.

    The special key ``simulator`` is extracted and returned separately.
    Nested keys use dot notation: ``env.numEnvs=4096``.
    """
    simulator = "isaacgym"
    overrides: dict = {}

    for arg in args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)

        # Try to cast to int/float/bool
        val = _auto_cast(val)

        if key == "simulator":
            simulator = val
            continue

        # Dot-notation → nested dict
        parts = key.split(".")
        d = overrides
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val

    return simulator, overrides


def _auto_cast(val: str):
    """Best-effort cast of a CLI string to int/float/bool/list."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    # Lists like [a,b,c]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1]
        items = [_auto_cast(s.strip().strip('"').strip("'")) for s in inner.split(",")]
        return items
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def create_backend(simulator: str):
    """Instantiate the correct :class:`SimBackend` for *simulator*."""
    if simulator == "isaacgym":
        from train.sapg.envs.backends.isaacgym_backend import IsaacGymBackend
        return IsaacGymBackend()
    elif simulator == "mujoco":
        from train.sapg.envs.backends.mujoco_backend import MuJoCoBackend
        return MuJoCoBackend()
    elif simulator == "mochi":
        from train.sapg.envs.backends.mochi_backend import MochiBackend
        return MochiBackend()
    else:
        raise ValueError(f"Unknown simulator: {simulator!r}")


# ---------------------------------------------------------------------------
# rl_games VecEnv wrapper
# ---------------------------------------------------------------------------

class SAPGVecEnv:
    """Thin adapter that bridges :class:`DexManipEnv` to rl_games' IVecEnv.

    rl_games expects a vecenv wrapper with ``step``, ``reset``,
    ``get_number_of_agents``, ``get_env_info``, and properties like
    ``observation_space``, ``action_space``.
    """

    def __init__(self, env):
        self.env = env

    # --- IVecEnv interface ---

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return self.env.get_env_info()

    def set_train_info(self, env_frames, *args, **kwargs):
        self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        return self.env.get_env_state()

    def set_env_state(self, state):
        self.env.set_env_state(state)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_agents(self):
        return 1


# ---------------------------------------------------------------------------
# Isaac Gym path — reuses existing AllegroKukaBase (migration bridge)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    """Run rl_games training with the given config."""
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder

    simulator = cfg.get("simulator", "isaacgym")
    multi_gpu = cfg.get("multi_gpu", False)

    # ---- Multi-GPU: read rank / world_size and override devices ----
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if multi_gpu:
        # CUDA_VISIBLE_DEVICES is set in main() so each process sees one GPU.
        # All processes use cuda:0 which maps to their assigned physical GPU.
        cfg["device"] = "cuda:0"
        cfg["rl_device"] = "cuda:0"
        cfg["sim_device"] = "cuda:0"
        cfg["graphics_device_id"] = 0
        print(f"[Multi-GPU] global_rank={global_rank} local_rank={local_rank} world_size={world_size}")

    rl_device = cfg.get("rl_device", "cuda:0")

    # ---- Seed ----
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    set_np_formatting()
    seed = set_seed(cfg.get("seed", 0), rank=global_rank)

    # ---- Create environment ----
    if simulator == "isaacgym":
        # Migration path: use existing Isaac Gym env classes directly.
        # This preserves full backward compatibility with the current codebase.
        # NOTE: isaacgym is imported in main() before torch to satisfy its
        # import-order requirement.
        from isaacgymenvs.utils.rlgames_utils import (
            RLGPUEnv,
            RLGPUAlgoObserver,
            MultiObserver,
            ComplexObsRLGPUEnv,
            get_rlgames_env_creator,
        )
        from isaacgymenvs.tasks import isaacgym_task_map

        task_name = cfg.get("task_name", "AllegroKukaLSTMAsymmetric")

        # Build the task_config dict that isaacgym_task_map expects.
        # The task classes read from cfg["env"], cfg["sim"], cfg["task"], etc.
        # We use Hydra compose to get the full default config, then overlay
        # our sapg config overrides on top.
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from isaacgymenvs.utils.reformat import omegaconf_to_dict

        import isaacgymenvs as _ige_pkg
        ig_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(_ige_pkg.__file__)), "cfg")

        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=ig_cfg_dir, version_base="1.1"):
            hydra_cfg = compose(config_name="config", overrides=[f"task={task_name}"])
            task_config = omegaconf_to_dict(hydra_cfg.task)

        # Apply user overrides from sapg config onto the task config
        if "env" in cfg:
            _deep_merge(task_config.get("env", {}), cfg["env"])
            if "env" not in task_config:
                task_config["env"] = cfg["env"]

        task_config["env"]["numEnvs"] = cfg["env"]["numEnvs"]

        def _create_ig_env(**kwargs):
            create_fn = get_rlgames_env_creator(
                seed=seed,
                task_config=task_config,
                task_name=task_config.get("name", task_name),
                sim_device=cfg.get("sim_device", "cuda:0"),
                rl_device=rl_device,
                graphics_device_id=cfg.get("graphics_device_id", 0),
                headless=cfg.get("headless", True),
                multi_gpu=multi_gpu,
                virtual_screen_capture=cfg.get("capture_video", False),
                force_render=cfg.get("force_render", False),
            )
            return create_fn()

        env_configurations.register("rlgpu", {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: _create_ig_env(**kwargs),
        })

        # Check if task uses dict observations
        ige_env_cls = isaacgym_task_map.get(task_name)
        dict_cls = (
            ige_env_cls.dict_obs_cls
            if ige_env_cls and hasattr(ige_env_cls, "dict_obs_cls")
            else False
        )

        if dict_cls:
            obs_spec = {}
            actor_net_cfg = cfg["train"]["params"]["network"]
            obs_spec["obs"] = {
                "names": list(actor_net_cfg.get("inputs", {}).keys()),
                "concat": actor_net_cfg.get("name") != "complex_net",
                "space_name": "observation_space",
            }
            if "central_value_config" in cfg["train"]["params"]["config"]:
                critic_net_cfg = cfg["train"]["params"]["config"]["central_value_config"]["network"]
                obs_spec["states"] = {
                    "names": list(critic_net_cfg.get("inputs", {}).keys()),
                    "concat": critic_net_cfg.get("name") != "complex_net",
                    "space_name": "state_space",
                }
            vecenv.register(
                "RLGPU",
                lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(
                    config_name, num_actors, obs_spec, **kwargs
                ),
            )
        else:
            vecenv.register(
                "RLGPU",
                lambda config_name, num_actors, **kwargs: RLGPUEnv(
                    config_name, num_actors, **kwargs
                ),
            )

        observers = [RLGPUAlgoObserver()]
    else:
        # ---- New backend path (MuJoCo / Mochi / future backends) ----
        from train.sapg.envs.env import DexManipEnv

        backend = create_backend(simulator)
        env = DexManipEnv(cfg, backend)

        # Backend creates sim
        backend.create(cfg, cfg["env"]["numEnvs"], cfg.get("device", "cpu"))
        env.allocate_buffers()

        if env.use_depth:
            backend.setup_depth_cameras(cfg)

        vec_env = SAPGVecEnv(env)

        env_configurations.register("rlgpu", {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: vec_env,
        })
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: vec_env,
        )

        from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver, MultiObserver
        observers = [RLGPUAlgoObserver()]

    # ---- Wandb (only rank 0 in multi-GPU) ----
    if cfg.get("wandb_activate", False) and global_rank == 0:
        from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
        wandb_observer = WandbAlgoObserver(cfg)
        observers.append(wandb_observer)

    # ---- Build rl_games runner ----
    from isaacgymenvs.learning import amp_continuous, amp_players, amp_models, amp_network_builder

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder(
            "amp_continuous",
            lambda **kwargs: amp_continuous.AMPAgent(**kwargs),
        )
        runner.player_factory.register_builder(
            "amp_continuous",
            lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs),
        )
        model_builder.register_model(
            "continuous_amp",
            lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
        )
        model_builder.register_network(
            "amp",
            lambda **kwargs: amp_network_builder.AMPBuilder(),
        )
        return runner

    # ---- Prepare rl_games config ----
    rlg_config_dict = cfg["train"]

    # Inject runtime values into train config
    train_cfg = rlg_config_dict["params"]["config"]
    train_cfg["device"] = rl_device
    train_cfg["num_actors"] = cfg["env"]["numEnvs"]
    train_cfg["multi_gpu"] = multi_gpu
    train_cfg.setdefault("population_based_training", False)
    train_cfg.setdefault("pbt_idx", None)
    if cfg.get("full_experiment_name"):
        train_cfg["full_experiment_name"] = cfg["full_experiment_name"]

    # Model size multiplier
    try:
        mult = rlg_config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if mult != 1:
            units = rlg_config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * mult
            print(f"Modified MLP units by x{mult} to {units}")
    except KeyError:
        pass

    print(f"Simulator: {simulator}")
    print(f"rl_device: {rl_device}")
    print(f"num_envs: {cfg['env']['numEnvs']}")

    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # ---- Save config ----
    if not cfg.get("test", False):
        experiment_dir = os.path.join(
            "runs", train_cfg.get("name", "DexManip")
        )
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            with open(os.path.join(experiment_dir, "cmd.txt"), "w") as f:
                f.write(" ".join(sys.argv))

    # ---- Run ----
    runner.run({
        "train": not cfg.get("test", False),
        "play": cfg.get("test", False),
        "checkpoint": cfg.get("checkpoint", ""),
        "sigma": cfg.get("sigma") if cfg.get("sigma") else None,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point.  Parses ``key=value`` arguments from sys.argv."""
    # Multi-GPU: pin each process to its own GPU via CUDA_VISIBLE_DEVICES
    # before any CUDA initialization. torchrun sets LOCAL_RANK per process.
    # This makes cuda:0 in each process map to a different physical GPU.
    _local_rank = os.getenv("LOCAL_RANK")
    if _local_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _local_rank

    simulator, overrides = parse_cli_overrides(sys.argv[1:])

    # Isaac Gym MUST be imported before torch (which rl_games imports).
    if simulator == "isaacgym":
        import isaacgym  # noqa: F401

    cfg = load_config(simulator=simulator, overrides=overrides)
    train(cfg)


if __name__ == "__main__":
    main()
