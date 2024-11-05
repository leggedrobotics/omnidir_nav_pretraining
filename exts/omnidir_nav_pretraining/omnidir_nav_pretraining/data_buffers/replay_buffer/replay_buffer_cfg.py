

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class ReplayBufferCfg:
    # TODO(kappi): move to global config
    history_length: int = 1

    # replay buffer
    trajectory_length: int = 20
    """How many commands steps are stored for each environment."""
    buffer_device: str = "cpu"
    """Device for the buffer.

    Note: leave at `cpu` for high-dimensional exteroceptive observations. Otherwise, use `cuda`.
    `cpu` can lead to reduced performance.
    """
