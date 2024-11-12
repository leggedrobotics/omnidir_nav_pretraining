from __future__ import annotations

from dataclasses import MISSING
from typing import Literal
import math
from omni.isaac.lab.utils import configclass

from .pd_agent import PDAgent


@configclass
class PDAgentCfg:
    class_type: type = PDAgent
    """Class type of the agent."""

    debug_vis: bool = False
    """Enable debug visualization."""

    robot_attr: str = "robot"
    """Name of the robot attribute from the environment."""

    path_frame: Literal["world", "robot"] = "world"
    """Frame in which the path is defined.
    - ``world``: the path is defined in the world frame. Also called ``odom``.
    - ``robot``: the path is defined in the robot frame. Also called ``base``.
    """

    lookAheadDistance: float = 1.0
    """The lookahead distance for the path follower."""
    two_way_drive: bool = False
    """Allow robot to use reverse gear."""
    switch_time_threshold: float = 1.0
    """Time threshold to switch between the forward and backward drive."""
    maxSpeed: float = 2.5#0.5
    """Maximum speed of the robot."""
    maxAccel: float = 5#2.5
    """Maximum acceleration of the robot."""
    joyYaw: float = 1.0
    """TODO: add description"""
    yawRateGain: float = 7.0
    """Gain for the yaw rate."""
    stopYawRateGain: float = 7.0
    """"""
    dt: float = 0.1
    """Time step path follower 10Hz"""

    waypointUpdateThre: float = 0.3

    maxYawRate: float = 90.0 * math.pi / 360
    dirDiffThre: float = 0.7
    stopDisThre: float = 0.4
    curWaypointSlowDisThre: float = 0.2
    slowDwnDisThre: float = 0.2
    slowRate1: float = 0.25
    slowRate2: float = 0.5
    noRotAtGoal: bool = True
    autonomyMode: bool = False
    dynamic_lookahead: bool = False
    min_points_within_lookahead: int = 3
    """Dynamic setting of the lookahead distance based on the number of points within the lookahead distance."""