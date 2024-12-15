from dataclasses import MISSING

import os
from omni.isaac.lab.envs import ViewerCfg
from nav_collectors.terrain_analysis.terrain_analysis_cfg import TerrainAnalysisCfg
from nav_tasks import NAVSUITE_TASKS_DATA_DIR
from nav_tasks.terrains import MazeTerrainCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainGeneratorCfg

from omnidir_nav.mdp.commands import PretrainingGoalCommandCfg
import omnidir_nav.mdp as mdp
import omnidir_nav.terrains as terrains
import omni.isaac.lab.sim as sim_utils


def env_modifier_pre_init(env_cfg, args_cli):
    """Modify the environment config before initialization."""
    # change terrain config
    if args_cli.test_env == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"

    env_cfg.commands.goal_command.min_path_length= None
    env_cfg.commands.goal_command.max_path_length= None

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

    env_cfg.observations.pretraining_state = PretrainingCfg()
    # Get raw sphere image.
    env_cfg.observations.policy.embedded_spherical_image.return_embedded = False
    #env_cfg.observations.policy.embedded_multicam_perceptnet_image.return_embedded = False

    @configclass
    class DebugViewerCfg(ViewerCfg):
        """Configuration of the scene viewport camera."""

        eye: tuple[float, float, float] = (0.0, 4.0, 7.0)
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
        resolution: tuple[int, int] = (1280, 720)  # (1280, 720) HD, (1920, 1080) FHD
        origin_type: str = "asset_root"  # "world", "env", "asset_root"
        env_index: int = 1
        asset_name: str = "robot"

    # env_cfg.viewer = DebugViewerCfg()

    # This is a hack so we can tell apart the successful trajectories from the failed ones.
    env_cfg.terminations.goal_reached.time_out = True
    env_cfg.terminations.time_out.time_out = False

    env_cfg.scene.num_envs = args_cli.num_envs

    # Turn off curriculum
    env_cfg.curriculum = MISSING
    return env_cfg


def env_modifier_post_init(runner, args_cli):
    """Modify the environment config after initialization."""
    print(f"[INFO] Post-Init modifying Environment...")
    return runner
