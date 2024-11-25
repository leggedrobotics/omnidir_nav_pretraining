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
        trajectory_length=300,
        non_obs_state_dim=1, # 1 additional state dimension for the current waypoint index
    )
    """Replay buffer config class"""
    dataset_cfg: OmnidirNavDatasetCfg = OmnidirNavDatasetCfg(
        num_samples=100000,
        validation_split=0.1,
        batch_size=1000,
    )
    """Dataset config class"""

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
