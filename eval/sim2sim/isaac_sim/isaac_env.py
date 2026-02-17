# gymtorch must be imported before torch
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip

from pathlib import Path
from typing import Optional, Dict, Any

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from eval.sim2real.rl_player_utils import (
    read_cfg_omegaconf,
)


def create_env_from_cfg(
    cfg: DictConfig,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    merge_with_default_config: bool = True,
    episode_length: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AllegroKukaBase:
    if merge_with_default_config:
        cfg = merge_cfg_with_default_config(cfg)
    return create_env_from_cfg(
        cfg=cfg,
        headless=headless,
        enable_viewer_sync_at_start=enable_viewer_sync_at_start,
        episode_length=episode_length,
        overrides=overrides,
    )


def create_env(
    config_path: str,
    device: str,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    merge_with_default_config: bool = True,
    episode_length: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AllegroKukaBase:
    cfg = read_cfg_omegaconf(config_path=config_path, device=device)

    if merge_with_default_config:
        cfg = merge_cfg_with_default_config(cfg)
    return create_env_from_cfg(
        cfg=cfg,
        headless=headless,
        enable_viewer_sync_at_start=enable_viewer_sync_at_start,
        episode_length=episode_length,
        overrides=overrides,
    )

def merge_cfg_with_default_config(cfg: DictConfig) -> DictConfig:
    # Use this if the config from config path is missing fields
    # For example, say we recently added a new field "object_friction" to the config
    # If this wasn't in the config file, this would normally fail
    # Merging with the default config will add this field with the default value
    print("Merging with default config")

    # Should be path of the isaacgymenvs/cfg directory relative to this file's directory
    with initialize(version_base="1.1", config_path="../../isaacgymenvs/cfg"):
        init_cfg = compose(config_name="config", overrides=["task=AllegroKukaLSTM"])

    # Disable struct mode to allow merging
    OmegaConf.set_struct(init_cfg, False)
    OmegaConf.set_struct(cfg, False)

    # Put cfg second to override init_cfg
    merged_cfg = OmegaConf.merge(init_cfg, cfg)
    assert isinstance(merged_cfg, DictConfig), (
        f"Expected DictConfig, got {type(merged_cfg)}"
    )

    # Print the differences
    diff = recursive_diff(
        OmegaConf.to_container(cfg, resolve=True),
        OmegaConf.to_container(merged_cfg, resolve=True),
    )
    print("Changes:")
    print("-" * 80)
    for key, change in diff.items():
        print(f"{key}: {change}")

    return merged_cfg


def create_env_from_cfg(
    cfg: DictConfig,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    episode_length: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AllegroKukaBase:
    # Modify the config
    cfg.headless = headless
    cfg.task.sim.enable_viewer_sync_at_start = enable_viewer_sync_at_start
    cfg.task.env.numEnvs = 1
    if episode_length is not None:
        cfg.task.env.episodeLength = episode_length

    # HACK: Assume that graphics_device_id should be 0
    # This is a pretty reasonable assumption because we are typically doing this testing on a workstation with 1 GPU
    cfg.graphics_device_id = 0

    # Modify the config for the task
    if overrides is not None:
        # Example: overrides = {"task.env.asset.kukaAllegro": "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted.urdf"}
        for key, value in overrides.items():
            if isinstance(value, str):
                value = f'"{value}"'
            else:
                value = str(value)
            eval_str = f"cfg.{key} = {value}"
            print(f"Evaluating: {eval_str}")
            exec(eval_str)

    print_dict(omegaconf_to_dict(cfg))

    # MODIFY_THINGS_HARDCODED = True
    # if MODIFY_THINGS_HARDCODED:
    #     cfg.task.env.asset.kukaAllegro = "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted.urdf"
    #     # cfg.task.env.allegroStiffness = 5.0
    #     # cfg.task.env.allegroDamping = 0.1
    #     cfg.task.env.allegroStiffness = 5.0
    #     cfg.task.env.allegroDamping = 0.25

    env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    return env


def recursive_diff(cfg1: dict, cfg2: dict, path: str = "") -> dict:
    """Recursively compare two DictConfigs and return differences."""
    differences = {}

    # Get the keys from both configs
    keys1 = set(cfg1.keys()) if isinstance(cfg1, dict) else set()
    keys2 = set(cfg2.keys()) if isinstance(cfg2, dict) else set()

    # Check for keys that are only in cfg1
    for key in keys1 - keys2:
        differences[f"{path}.{key}".lstrip(".")] = f"{cfg1[key]} -> None"

    # Check for keys that are only in cfg2
    for key in keys2 - keys1:
        differences[f"{path}.{key}".lstrip(".")] = f"None -> {cfg2[key]}"

    # Check for keys that are in both configs
    for key in keys1 & keys2:
        val1 = cfg1[key]
        val2 = cfg2[key]

        # Recursively compare dictionaries or lists
        if isinstance(val1, dict) and isinstance(val2, dict):
            diff = recursive_diff(val1, val2, path=f"{path}.{key}".lstrip("."))
            differences.update(diff)
        elif val1 != val2:
            # If values differ, record the difference
            differences[f"{path}.{key}".lstrip(".")] = f"{val1} -> {val2}"

    return differences


def main() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    CONFIG_PATH = Path(
        "/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()

    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
    )

    print(env)
    obs = env.reset()
    N_STEPS = 1000
    for _ in range(N_STEPS):
        action = torch.rand(
            (env.num_envs, env.num_acts), device=DEVICE, dtype=torch.float
        )
        obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
