

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class OmnidirNavDatasetCfg:
    num_samples: int = 1000
    min_traj_duration_steps: int = 2