# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Sutter and Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Sub-module containing command generators for the velocity-based locomotion task."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
import torch
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from .pd_agent_cfg import PDAgentCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class PDAgent(ABC):
    """Agent that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: PDAgentCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: PDAgentCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg (PDAgentCfg): The configuration of the command generator.
            env (BaseEnv): The environment.
        """
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        # -- buffers
        self.vehicleSpeed: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.switch_time: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.vehicleYawRate: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.navigation_forward: torch.Tensor = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.twist: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached: torch.Tensor = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.current_waypoint_idxs = torch.zeros((self.num_envs), device=self.device, dtype=torch.int)

        self.prev_paths = None

        # -- debug vis
        self._set_debug_vis(cfg.debug_vis)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator.

        This function resets the command generator. It should be called whenever the environment is reset.

        Args:
            env_ids (Optional[Sequence[int]], optional): The list of environment IDs to reset. Defaults to None.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
        # reset the buffers
        self.vehicleSpeed[env_ids] = 0.0
        self.switch_time[env_ids] = 0.0
        self.vehicleYawRate[env_ids] = 0.0
        self.navigation_forward[env_ids] = True
        self.twist[env_ids] = 0.0
        self.goal_reached[env_ids] = False

        self.current_waypoint_idxs[env_ids] = 0.0
        return {}

    def act(self, paths: torch.Tensor):
        """Compute the velocity command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. Num_envs is equal to
        the number of robots spawned in all environments.

        returns:
            torch.Tensor: The velocity command to be executed by the robot. Shape is (num_envs, 3).
        """
        # get number of pases of the paths
        num_envs, N, _ = paths.shape
        assert N > 0, "PDAgentGenerator: paths must have at least one poses."
        # define current maxSpeed for the velocities
        max_speed = torch.ones(num_envs, device=self.device) * self.cfg.maxSpeed

        # transform path in base/ robot frame if given in world frame
        paths_world = paths
        if self.cfg.path_frame == "world":
            paths = math_utils.quat_apply(
                math_utils.quat_inv(self.robot.data.root_quat_w[:, None, :].repeat(1, N, 1)),
                paths - self.robot.data.root_pos_w[:, None, :],
            )
        elif self.cfg.path_frame == "robot":
            paths_world = math_utils.quat_apply(
                self.robot.data.root_quat_w[:, None, :].repeat(1, N, 1), paths
            ) + self.robot.data.root_pos_w[:, None, :]

        if self.prev_paths is not None:
            tolerance = 0.5  # 50 cm
            # Compare paths_world and self.prev_paths within the tolerance
            new_paths = torch.any(torch.any(torch.abs(paths_world - self.prev_paths) > tolerance, dim=2), dim=1)
            # if torch.any(new_paths):
            #     print(f"New path detected for {torch.nonzero(new_paths)}")
            self.current_waypoint_idxs[new_paths] = 0
        self.prev_paths = paths_world

        # get distance that robot has to travel until last set waypoint
        distance_end_point = torch.linalg.norm(paths[:, -1, :2], axis=1)

        # Get the current waypoint for each agent
        cur_waypoints = paths[torch.arange(num_envs), self.current_waypoint_idxs]
        cur_waypoints_world = paths_world[torch.arange(num_envs), self.current_waypoint_idxs]

        path_direction = torch.atan2(cur_waypoints[:, 1], cur_waypoints[:, 0])
        direction_diff = -path_direction

        # decide whether to drive forward or backward
        if self.cfg.two_way_drive:
            switch_time_threshold_exceeded = self.sim.current_time - self.switch_time > self.cfg.switch_time_threshold
            # get index of robots that should switch direction
            switch_to_backward_idx = torch.all(
                torch.vstack(
                    (abs(direction_diff) > math.pi / 2, switch_time_threshold_exceeded, self.navigation_forward)
                ),
                dim=0,
            )
            switch_to_forward_idx = torch.all(
                torch.vstack(
                    (abs(direction_diff) < math.pi / 2, switch_time_threshold_exceeded, ~self.navigation_forward)
                ),
                dim=0,
            )
            # update buffers
            self.navigation_forward[switch_to_backward_idx] = False
            self.navigation_forward[switch_to_forward_idx] = True
            self.switch_time[switch_to_backward_idx] = self.sim.current_time
            self.switch_time[switch_to_forward_idx] = self.sim.current_time

        # adapt direction difference and maxSpeed depending on driving direction
        direction_diff[~self.navigation_forward] += math.pi
        limit_radians = torch.all(torch.vstack((direction_diff > math.pi, ~self.navigation_forward)), dim=0)
        direction_diff[limit_radians] -= 2 * math.pi
        max_speed[~self.navigation_forward] *= -1

        # determine yaw rate of robot
        vehicleYawRate = torch.zeros(num_envs, device=self.device)
        stop_yaw_rate_bool = abs(direction_diff) < 2.0 * self.cfg.maxAccel * self.cfg.dt
        vehicleYawRate[stop_yaw_rate_bool] = -self.cfg.stopYawRateGain * direction_diff[stop_yaw_rate_bool]
        vehicleYawRate[~stop_yaw_rate_bool] = -self.cfg.yawRateGain * direction_diff[~stop_yaw_rate_bool]

        # limit yaw rate of robot
        vehicleYawRate[vehicleYawRate > self.cfg.maxYawRate] = self.cfg.maxYawRate
        vehicleYawRate[vehicleYawRate < -self.cfg.maxYawRate] = -self.cfg.maxYawRate

        # catch special cases
        if not self.cfg.autonomyMode:
            vehicleYawRate[max_speed == 0.0] = self.cfg.maxYawRate * self.cfg.joyYaw
        if N <= 1:
            vehicleYawRate *= 0
            max_speed *= 0
        elif self.cfg.noRotAtGoal:
            vehicleYawRate[distance_end_point < self.cfg.stopDisThre] = 0.0

        # determine joyspeed at the end of the path
        slow_down_bool = distance_end_point / self.cfg.slowDwnDisThre < max_speed
        max_speed[slow_down_bool] *= distance_end_point[slow_down_bool] / self.cfg.slowDwnDisThre

        # update current waypoint
        distance_cur_waypoint = torch.linalg.norm(cur_waypoints[:, :2], axis=1)
        update_waypoint = distance_cur_waypoint < self.cfg.waypointUpdateThre
        update_waypoint = update_waypoint & (self.current_waypoint_idxs < paths.shape[1] - 1)
        self.current_waypoint_idxs[update_waypoint] += 1
        self._debug_vis_callback(self, cur_waypoints_world + torch.Tensor([0, 0, 0.3]).to(cur_waypoints.device))

        # update vehicle speed
        drive_at_max_speed = torch.all(
            torch.vstack(
                (abs(direction_diff) < self.cfg.dirDiffThre, distance_cur_waypoint > self.cfg.curWaypointSlowDisThre)
            ),
            dim=0,
        )
        increase_speed = torch.all(torch.vstack((self.vehicleSpeed < max_speed, drive_at_max_speed)), dim=0)
        decrease_speed = torch.all(torch.vstack((self.vehicleSpeed > max_speed, drive_at_max_speed)), dim=0)
        self.vehicleSpeed[increase_speed] += self.cfg.maxAccel * self.cfg.dt
        self.vehicleSpeed[decrease_speed] -= self.cfg.maxAccel * self.cfg.dt
        increase_speed = torch.all(torch.vstack((self.vehicleSpeed <= 0, ~drive_at_max_speed)), dim=0)
        decrease_speed = torch.all(torch.vstack((self.vehicleSpeed > 0, ~drive_at_max_speed)), dim=0)
        self.vehicleSpeed[increase_speed] += self.cfg.maxAccel * self.cfg.dt
        self.vehicleSpeed[decrease_speed] -= self.cfg.maxAccel * self.cfg.dt

        # update twist command
        self.twist[:, 0] = self.vehicleSpeed
        self.twist[abs(self.vehicleSpeed) < self.cfg.maxAccel * self.cfg.dt, 0] = 0.0
        self.twist[abs(self.vehicleSpeed) > self.cfg.maxSpeed, 0] = self.cfg.maxSpeed
        self.twist[:, 2] = vehicleYawRate

        return self.twist, self.current_waypoint_idxs

    def _set_debug_vis(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        # for each marker type check that the correct command properties exist eg. need spawn position for spawn marker
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = f"/Visuals/Command/position_waypoint"
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                # Change color of the marker
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1, 0.6, 0)
                self.waypoint_visualizer = VisualizationMarkers(marker_cfg)
                self.waypoint_visualizer.set_visibility(True)

        else:
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event, paths: torch.Tensor):
        """Callback function for the debug visualization."""

        if not hasattr(self, "waypoint_visualizer"):
            return

        # Display markers at each path waypoint
        self.waypoint_visualizer.visualize(paths)
