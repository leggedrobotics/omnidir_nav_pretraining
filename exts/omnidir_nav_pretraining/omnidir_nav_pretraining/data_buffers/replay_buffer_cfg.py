

from __future__ import annotations

from omni.isaac.lab.utils import configclass


@configclass
class ReplayBufferCfg:
    # replay buffer
    trajectory_length: int = 150
    """How many commands steps are stored for each environment."""
    buffer_device: str = "cpu"
    """Device for the buffer.

    Note: leave at `cpu` for high-dimensional exteroceptive observations. Otherwise, use `cuda`.
    `cpu` can lead to reduced performance.
    """
    exteroceptive_obs_precision: str = "float16"
    """Precision of the exteroceptive observations. Reduced to `float16` for memory reasons."""
