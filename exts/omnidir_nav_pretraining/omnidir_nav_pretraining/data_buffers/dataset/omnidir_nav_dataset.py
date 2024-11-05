from __future__ import annotations

import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils

from .omnidir_nav_dataset_cfg import OmnidirNavDatasetCfg

if TYPE_CHECKING:
    from omnidir_nav_pretraining.data_buffers.replay_buffer import ReplayBuffer, ReplayBufferCfg
    from omnidir_nav_pretraining.runner.runner_cfg import GlobalSettingsCfg


class OmnidirNavDataset(Dataset):
    def __init__(
        self,
        cfg: OmnidirNavDatasetCfg,
        global_settings_cfg: GlobalSettingsCfg,
        replay_buffer_cfg: ReplayBufferCfg,
        return_device: str,
    ):
        # save configs
        self.cfg: OmnidirNavDatasetCfg = cfg
        self.global_settings_cfg: GlobalSettingsCfg = global_settings_cfg
        self.replay_buffer_cfg: ReplayBufferCfg = replay_buffer_cfg
        self._actual_num_samples: int = self.cfg.num_samples
        self.return_device: str = return_device

    def __str__(self) -> str:
        msg = (
            "#############################################################################################\n"
            f"Dataset with command trajectory (length {self.replay_buffer_cfg.trajectory_length})"
            " contains\n"
            f"\tIntended Number: \t{self.cfg.num_samples}\n"
            f"\tActual Number  : \t{self._actual_num_samples}\n"
            f"\tReturn Device  : \t{self.return_device}\n"
            "#############################################################################################"
        )

        return msg

    ##
    # Properties
    ##

    @property
    def num_samples(self) -> int:
        return self._actual_num_samples

    ##
    # Operations
    ##

    def populate(
        self,
        replay_buffer: ReplayBuffer,
    ):
        """
        Update data in the buffer for specified indexes.

        Args:
            replay_buffer: The replay buffer to get the data from.
        """
        start_idx = self._sample_random_traj_idx(replay_buffer)
        # TODO(kappi): Select goal idxs

        ############################################################
        # Actions
        ############################################################
        self.actions = replay_buffer.actions[start_idx[:, 0], start_idx[:, 1]] 

        ############################################################
        # States
        ############################################################

        # TODO(kappi): get current state and use it to transform the previous and following states into the local robot 
        # frame shape: [N, 7] with [x, y, z, qx, qy, qz, qw]
        initial_states = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], 0][
            :, None, :7
        ]
      
        ############################################################
        # Observations
        ############################################################

        self.obs = replay_buffer.observations[
            start_idx[:, 0], start_idx[:, 1]
        ]

        ############################################################
        # Goals
        ############################################################

        #TODO(kappi): Implement goal extraction


        ############################################################
        # Filter data
        ############################################################

        # init keep index array
        keep_idx = torch.ones(
            self.state_history.shape[0],
            dtype=torch.bool,
            device=self.replay_buffer_cfg.buffer_device,
        )

        # TODO(kappi): Implement filtering of bad trajectories (collisions, falling off the plane etc.) by setting keep_idx to False

        self._filter_idx(keep_idx)

        ############################################################
        # Gather Statistics
        ############################################################

        # TODO(kappi): Possibly normalize data based on the statistics

        max_distance = torch.norm(self.states[:, -1, :2], dim=1)

        # get the maximum observed velocity
        lin_velocity = torch.abs(
            (self.states[:, 1:, :2] - self.states[:, :-1, :2])
            / self.global_settings_cfg.command_timestep
        )
        heading = torch.atan2(self.states[:, :, 2], self.states[:, :, 3])
        # enforce periodicity of the heading
        yaw_diff = torch.abs(heading[:, 1:] - heading[:, :-1])
        yaw_diff = math_utils.wrap_to_pi(yaw_diff)
        ang_velocity = torch.abs(yaw_diff / self.global_settings_cfg.command_timestep)
        max_velocity = torch.concatenate(
            [
                torch.max(lin_velocity.reshape(-1, 2), dim=0)[0],
                torch.max(ang_velocity.reshape(-1, 1), dim=0)[0],
            ],
            dim=0,
        )

        # get the maximum observed acceleration
        max_lin_acceleration = torch.max(
            torch.abs(
                (lin_velocity[:, 1:] - lin_velocity[:, :-1])
                / self.global_settings_cfg.command_timestep
            ).reshape(-1, 2),
            dim=0,
        )[0]
        max_ang_acceleration = torch.max(
            torch.abs(
                (ang_velocity[:, 1:] - ang_velocity[:, :-1])
                / self.global_settings_cfg.command_timestep
            ).reshape(-1, 1),
            dim=0,
        )[0]
        max_acceleration = torch.concatenate(
            [max_lin_acceleration, max_ang_acceleration], dim=0
        )

        # TODO(kappi): Check the maximum velocity is not more than the maximum commanded velocity, filter out the samples


        ############################################################
        # Check for nan and inf values
        ############################################################

        if torch.any(torch.isnan(self.states)) or torch.any(torch.isinf(self.states)):
            raise ValueError("Nan/ Inf values in states!")
        if torch.any(torch.isnan(self.state_history)) or torch.any(
            torch.isinf(self.state_history)
        ):
            raise ValueError("Nan/ Inf values in state history!")
        if torch.any(torch.isnan(self.obs)) or torch.any(
            torch.isinf(self.obs)
        ):
            raise ValueError("Nan/ Inf values in proprioceptive observations!")
        if torch.any(torch.isnan(self.actions)) or torch.any(torch.isinf(self.actions)):
            raise ValueError("Nan/ Inf values in actions!")

        ############################################################
        # Print meta information
        ############################################################
        self._print_statistics(max_distance, max_velocity, max_acceleration)
        

    """
    Private functions
    """

    def _print_statistics(self, max_distance: torch.Tensor, max_velocity: torch.Tensor, max_acceleration: torch.Tensor):
        """Print statistics of the dataset"""
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"

        # Add rows with formatted values
        table.add_row(
            (
                "Average max distance",
                f"{torch.mean(torch.abs(max_distance), dim=0).item():.4f}",
            )
        )
        table.add_row(
            ("Max velocity", [f"{v:.4f}" for v in max_velocity.cpu().tolist()])
        )
        table.add_row(
            ("Max acceleration", [f"{a:.4f}" for a in max_acceleration.cpu().tolist()])
        )

        # Print distance percentages
        for distance in range(1, int(torch.max(max_distance).item()) + 1):
            ratio = (
                torch.sum(
                    torch.all(
                        torch.vstack(
                            (max_distance - 1 < distance, max_distance > distance)
                        ),
                        dim=0,
                    )
                )
                / self.states.shape[0]
            )
            table.add_row(
                (f"Ratio between {distance-1} - {distance}m", f"{ratio.item():.4f}")
            )

        # Print table
        print(f"[INFO] Dataset Metrics {self.states.shape[0]} samples\n", table)


    def _sample_random_traj_idx(self, replay_buffer: ReplayBuffer):
        # sample random start indexes
        # collision not correctly captured at the last entry of the trajectory, exclude it from trajectories
        start_idx = torch.randint(
            1,
            self.replay_buffer_cfg.trajectory_length
            - self.cfg.min_traj_duration_steps
            - 1,
            (self.cfg.num_samples,),
            device=self.replay_buffer_cfg.buffer_device,
        )
        env_idx = torch.randint(
            0,
            replay_buffer.env.num_envs,
            (self.cfg.num_samples,),
            device=self.replay_buffer_cfg.buffer_device,
        )
        return torch.vstack([env_idx, start_idx]).T


    def _filter_idx(self, keep_idx: torch.Tensor):
        """Filter data and only keep the given indexes. After filtering, update the number of samples"""
        # filter data
        self.state_history = self.state_history[keep_idx]
        self.obs = self.obs[keep_idx]
        self.actions = self.actions[keep_idx]
        self.states = self.states[keep_idx]

        # update sample number
        self._actual_num_samples = torch.sum(keep_idx).item()


    """
    Properties called when accessing the data
    """

    def __len__(self):
        return self._actual_num_samples

    def __getitem__(self, index: int):
        # TODO(kappi): Implement correct model inputs
        return (
            # model inputs
            self.state_history[index],
            self.obs[index],
            self.actions[index],
            # model targets
            self.states[index],
            # eval data
        )
