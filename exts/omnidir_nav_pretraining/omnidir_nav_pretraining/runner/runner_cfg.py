from __future__ import annotations

import os
import math
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg

from omnidir_nav.env_cfg import OmnidirNavEnvCfg
from omnidir_nav.model import OmnidirNavModelCfg
import omnidir_nav.mdp as mdp
from omnidir_nav_pretraining.agent import PDAgentCfg
from omnidir_nav_pretraining.data_buffers import ReplayBufferCfg

from omnidir_nav_pretraining.trainer import TrainerBaseCfg


@configclass
class OmnidirNavRunnerCfg:
    model_cfg: OmnidirNavModelCfg = MISSING
    """Model config class"""
    env_cfg: OmnidirNavEnvCfg = MISSING
    """Environment config class"""
    trainer_cfg: TrainerBaseCfg = TrainerBaseCfg()
    """Trainer config class"""
    agent_cfg: PDAgentCfg = PDAgentCfg(
        max_beta=0.3,
        sigma_scale=0.3,
    )
    """Agent config class"""
    replay_buffer_cfg: ReplayBufferCfg = ReplayBufferCfg()
    """Replay buffer config class"""

    # general configurations
    collection_rounds: int = 20
    """Number of collection rounds. For each round, ``epochs`` number of epochs are trained."""

    body_regex_contact_checking: str = ".*FOOT"
    """Regex to select the bodies for contact checking.
    
    During data collection, reset environments that haven't been in contact with the ground for a certain number of steps.
    This regex describes which bodies are used to determine whether the robot is in contact with the ground.
    """

    def __post_init__(self):
        print("[INFO] OmnidirNavRunnerCfg initialized.")