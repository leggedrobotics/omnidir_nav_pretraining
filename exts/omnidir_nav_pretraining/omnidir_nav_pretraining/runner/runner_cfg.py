from __future__ import annotations

import os
import math
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg

from omnidir_nav.env_config.env_cfg import OmnidirNavEnvCfg

# from omnidir_nav.model import OmnidirNavModelCfg
from omnidir_nav_pretraining.agent import PDAgentCfg
from omnidir_nav_pretraining.data_buffers import ReplayBufferCfg
from omnidir_nav_pretraining.data_buffers import OmnidirNavDatasetCfg

from omnidir_nav.mdp.commands import PretrainingGoalCommandCfg
from nav_collectors.terrain_analysis.terrain_analysis_cfg import TerrainAnalysisCfg


@configclass
class GlobalSettingsCfg:
    command_timestep: float = MISSING
    """Command timestep in seconds."""


@configclass
class OmnidirNavRunnerCfg:
    global_settings_cfg: GlobalSettingsCfg = GlobalSettingsCfg(
        command_timestep=2.0,
    )

    ############################################################
    # Model, Environment, Agent, Replay Buffer, Dataset Configs
    ############################################################
    # model_cfg: OmnidirNavModelCfg = MISSING
    """Model config class"""
    env_cfg: OmnidirNavEnvCfg = OmnidirNavEnvCfg()
    """Environment config class"""
    agent_cfg: PDAgentCfg = PDAgentCfg(
        min_points_within_lookahead=2,
        path_frame="robot",
        maxAccel=2.5,
        maxSpeed=0.75,
        waypointUpdateThre=0.3,
        dynamic_lookahead=True,
        debug_vis=True,
    )
    """Agent config class"""
    replay_buffer_cfg: ReplayBufferCfg = ReplayBufferCfg(
        trajectory_length=100,
        non_obs_state_dim=1, # 1 additional state dimension for the current waypoint index
    )
    """Replay buffer config class"""
    train_dataset_cfg: OmnidirNavDatasetCfg = OmnidirNavDatasetCfg()
    """Training dataset config class"""
    validation_dataset_cfg: OmnidirNavDatasetCfg = OmnidirNavDatasetCfg()
    """Validation dataset config class"""

    ############################################################
    # Environment Settings
    ############################################################

    pretraining_goal_command_cfg: PretrainingGoalCommandCfg = PretrainingGoalCommandCfg(
        asset_name="robot",
        z_offset_spawn=0.1,
        terrain_analysis=TerrainAnalysisCfg(
            mode="path",
            max_waypoints=10,
            raycaster_sensor="front_zed_camera",
            max_terrain_size=100.0,
            sample_points=5000,  # TODO: Increase this after testing
            height_diff_threshold=0.4,
            semantic_cost_mapping=None,
            viz_graph=False,
            viz_height_map=False,
            keep_paths_in_subterrain=True,
        ),
        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
        debug_vis=True,
    )

    ############################################################
    # Runner Settings
    ############################################################

    body_regex_contact_checking: str = ".*FOOT"
    """Regex to select the bodies for contact checking.
    
    During data collection, reset environments that haven't been in contact with the ground for a certain number of steps.
    This regex describes which bodies are used to determine whether the robot is in contact with the ground.
    """

    def __post_init__(self):
        print("[INFO] OmnidirNavRunnerCfg initialized.")
