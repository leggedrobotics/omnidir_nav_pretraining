

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class ReplayBufferCfg:
    # replay buffer
    trajectory_length: int = 20
    """How many commands steps are stored for each environment."""
    buffer_device: str = "cpu"
    """Device for the buffer.

    Note: leave at `cpu` for high-dimensional exteroceptive observations. Otherwise, use `cuda`.
    `cpu` can lead to reduced performance.
    """
    non_obs_state_dim: int = 0
    """The dimension of any additional non-observation state data that is added during data collection."""
