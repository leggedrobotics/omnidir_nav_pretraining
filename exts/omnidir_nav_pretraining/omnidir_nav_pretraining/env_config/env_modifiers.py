from dataclasses import MISSING

from omni.isaac.lab.envs import ViewerCfg
from nav_collectors.terrain_analysis.terrain_analysis_cfg import TerrainAnalysisCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg

import omnidir_nav.mdp as mdp


def env_modifier_pre_init(cfg, args_cli):
    """Modify the environment config before initialization."""
    # change terrain config
    if args_cli.test_env == "plane":
        cfg.env_cfg.scene.terrain.terrain_type = "plane"

    cfg.env_cfg.commands.goal_command = cfg.pretraining_goal_command_cfg

    # Add observation group for data specific to pretraining.
    @configclass
    class PretrainingCfg(ObsGroup):
        """Observations for pretraining data group."""
        base_position = ObsTerm(func=mdp.base_position)
        base_orientation = ObsTerm(func=mdp.base_orientation_xyzw)
        base_collision = ObsTerm(
            func=mdp.base_collision,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["base", "RF_THIGH", "LF_THIGH", "RH_THIGH", "LH_THIGH"]
                ),
                "threshold": 1.0,
            },
        )
        goal_paths = ObsTerm(func=mdp.full_path_to_goal_robot_frame)

        def __post_init__(self):
            self.concatenate_terms = True

    cfg.env_cfg.observations.pretraining_state = PretrainingCfg()

    @configclass
    class DebugViewerCfg(ViewerCfg):
        """Configuration of the scene viewport camera."""

        eye: tuple[float, float, float] = (0.0, 7.0, 7.0)
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
        resolution: tuple[int, int] = (1280, 720)  # (1280, 720) HD, (1920, 1080) FHD
        origin_type: str = "asset_root"  # "world", "env", "asset_root"
        env_index: int = 1
        asset_name: str = "robot"

    cfg.env_cfg.viewer = DebugViewerCfg()

    cfg.env_cfg.scene.num_envs = args_cli.num_envs

    # Turn off curriculum
    cfg.env_cfg.curriculum = MISSING
    return cfg


def env_modifier_post_init(runner, args_cli):
    """Modify the environment config after initialization."""
    print(f"[INFO] Post-Init modifying Environment...")
    return runner
