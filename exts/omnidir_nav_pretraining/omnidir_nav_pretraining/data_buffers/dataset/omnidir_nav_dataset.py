from __future__ import annotations

import os
import pickle
from datetime import datetime
from omnidir_nav_pretraining import DATA_DIR
import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils

from .omnidir_nav_dataset_cfg import OmnidirNavDatasetCfg

if TYPE_CHECKING:
    from omnidir_nav_pretraining.data_buffers.replay_buffer import ReplayBuffer, ReplayBufferCfg
    from omnidir_nav_pretraining.runner.runner_cfg import GlobalSettingsCfg

# Start and end indices for the goal waypoints in the observation tensor.
GOAL_IDX_START = 30
GOAL_IDX_END = GOAL_IDX_START + 3

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

    def populate_and_save(
        self,
        replay_buffer: ReplayBuffer,
        num_waypoints: int,
    ):
        """
        Populate and save training and validation datasets from the replay buffer.

        Args:
            replay_buffer: The replay buffer to get the data from.
            num_waypoints: Number of waypoints to use in observations.
        """
        num_samples = self.cfg.num_samples
        num_batches = (num_samples + self.cfg.batch_size - 1) // self.cfg.batch_size
        validation_size = int(self.cfg.batch_size * self.cfg.validation_split)

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        dataset_base_path = os.path.join(DATA_DIR, f"data_collection_output/{timestamp}")

        # Create train and val subdirectories
        os.makedirs(os.path.join(dataset_base_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_base_path, "val"), exist_ok=True)

        for batch_idx in range(num_batches):
            # Compute batch indices
            start_idx = batch_idx * self.cfg.batch_size
            end_idx = min(start_idx + self.cfg.batch_size, num_samples)

            # Sample data for this batch
            self.cfg.num_samples = end_idx - start_idx
            self.populate(replay_buffer, num_waypoints, batch=batch_idx)

            # Split data into training and validation
            train_obs = self.obs[:-validation_size] if validation_size > 0 else self.obs
            train_prev_obs = self.prev_obs[:-validation_size] if validation_size > 0 else self.prev_obs
            train_actions = self.actions[:-validation_size] if validation_size > 0 else self.actions
            train_waypoints = self.paths[:-validation_size] if validation_size > 0 else self.paths

            val_obs = self.obs[-validation_size:] if validation_size > 0 else []
            val_prev_obs = self.prev_obs[-validation_size:] if validation_size > 0 else []
            val_actions = self.actions[-validation_size:] if validation_size > 0 else []
            val_waypoints = self.paths[-validation_size:] if validation_size > 0 else []

            # Save training dataset
            train_path = f"{dataset_base_path}/train/batch_{batch_idx}.pkl"
            with open(train_path, "wb") as fp:
                pickle.dump({"observations": train_obs, "actions": train_actions, "prev_observations": train_prev_obs, "waypoints": train_waypoints}, fp)

            # Save validation dataset
            if validation_size > 0:
                val_path = f"{dataset_base_path}/val/batch_{batch_idx}.pkl"
                with open(val_path, "wb") as fp:
                    pickle.dump({"observations": val_obs, "actions": val_actions, "prev_observations": val_prev_obs, "waypoints": val_waypoints}, fp)

            # Print status
            print(f"Batch {batch_idx + 1}/{num_batches} processed and saved.")

        # Restore original number of samples
        self.cfg.num_samples = num_samples


    def populate(
        self,
        replay_buffer: ReplayBuffer,
        num_waypoints: int,
        batch: int,
    ):
        """
        Update data in the buffer for specified indexes.

        Args:
            replay_buffer: The replay buffer to get the data from.
        """
        start_idx = self._sample_random_traj_idx(replay_buffer, seed=batch)

        ############################################################
        # Actions
        ############################################################
        self.actions = replay_buffer.actions[start_idx[:, 0], start_idx[:, 1]]

        ############################################################
        # States
        ############################################################

        # TODO(kappi): get current state and use it to transform the previous and following states into the local robot
        # frame shape: [N, 7] with [x, y, z, qx, qy, qz, qw]
        self.states = replay_buffer.states[start_idx[:, 0], start_idx[:, 1]]
        self.prev_states = replay_buffer.states[start_idx[:, 0], start_idx[:, 1] - 1]

        ############################################################
        # Observations
        ############################################################

        self.obs = replay_buffer.observations[start_idx[:, 0], start_idx[:, 1]]
        self.prev_obs = replay_buffer.observations[start_idx[:, 0], start_idx[:, 1] - 1]

        ############################################################
        # Goals
        ############################################################

        # The last entry in the state is the current waypoint index
        curr_waypoint_idx = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], -1].to(torch.int)
        # Extract the whole path of waypoints
        self.paths = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], -num_waypoints * 3 -1 :-1]
        self.paths = self.paths.view(self.paths.shape[0], num_waypoints, 3)

        # All waypoints can act as goals, except those from the past (before the current waypoint)
        # So for waypoints in the past, overwrite them with the last (goal) waypoint
        # Get the shape parameters
        num_envs, num_waypoints, _ = self.paths.shape

        # Calculate the range limits for each environment
        # num_waypoints - waypoint_idx gives the range size for each environment
        range_size = num_waypoints - curr_waypoint_idx

        # Generate a random offset within each environment's range size
        random_offsets = self._tensor_randint(0, range_size, (num_envs,)).to(torch.int)

        # Add the random offsets to waypoint_idx to get a random waypoint in the specified range
        random_valid_goals = self.paths[torch.arange(self.paths.shape[0]), curr_waypoint_idx + random_offsets]

        # Overwrite the observed goals with the random valid goals
        # self.obs[:, GOAL_IDX_START:GOAL_IDX_END] = random_valid_goals

        ############################################################
        # Filter data
        ############################################################

        # init keep index array
        keep_idx = torch.ones(
            self.obs.shape[0],
            dtype=torch.bool,
            device=self.replay_buffer_cfg.buffer_device,
        )

        # Filter out any sampleswhere the previous state is from a past goal.
        prev_waypoint_idx = self.prev_states[:,-1].to(torch.int)
        keep_idx &= prev_waypoint_idx <= curr_waypoint_idx

        # TODO(kappi): Implement filtering of bad trajectories (collisions, falling off the plane etc.) by setting keep_idx to False

        self._filter_idx(keep_idx)

        ############################################################
        # Gather Statistics
        ############################################################

        # TODO(kappi): Possibly normalize data based on the statistics

        # get the maximum observed velocity
        lin_velocity = torch.abs(
            (replay_buffer.states[:, 1:, :2] - replay_buffer.states[:, :-1, :2]) / self.global_settings_cfg.command_timestep
        )
        heading = torch.atan2(replay_buffer.states[:, :, 2], replay_buffer.states[:, :, 3])
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
            torch.abs((lin_velocity[:, 1:] - lin_velocity[:, :-1]) / self.global_settings_cfg.command_timestep).reshape(
                -1, 2
            ),
            dim=0,
        )[0]
        max_ang_acceleration = torch.max(
            torch.abs((ang_velocity[:, 1:] - ang_velocity[:, :-1]) / self.global_settings_cfg.command_timestep).reshape(
                -1, 1
            ),
            dim=0,
        )[0]
        max_acceleration = torch.concatenate([max_lin_acceleration, max_ang_acceleration], dim=0)

        # TODO(kappi): Check the maximum velocity is not more than the maximum commanded velocity, filter out the samples

        ############################################################
        # Check for nan and inf values
        ############################################################

        # if torch.any(torch.isnan(self.states)) or torch.any(torch.isinf(self.states)):
        #     raise ValueError("Nan/ Inf values in states!")
        # if torch.any(torch.isnan(self.obs)) or torch.any(torch.isinf(self.obs)):
        #     raise ValueError("Nan/ Inf values in proprioceptive observations!")
        # if torch.any(torch.isnan(self.actions)) or torch.any(torch.isinf(self.actions)):
        #     raise ValueError("Nan/ Inf values in actions!")

        ############################################################
        # Print meta information
        ############################################################
        self._print_statistics(max_velocity, max_acceleration)

    """
    Private functions
    """

    def _tensor_randint(self, low: int | torch.Tensor, high: torch.Tensor | None =None, size=None):
        if high is None:
            high = low
            low = 0
        if size is None:
            size = low.shape if isinstance(low, torch.Tensor) else high.shape
        return torch.randint(2**63 - 1, size=size) % (high - low) + low

    def _print_statistics(self, max_velocity: torch.Tensor, max_acceleration: torch.Tensor):
        """Print statistics of the dataset"""
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"

        # Add rows with formatted values
        table.add_row(("Max velocity", [f"{v:.4f}" for v in max_velocity.cpu().tolist()]))
        table.add_row(("Max acceleration", [f"{a:.4f}" for a in max_acceleration.cpu().tolist()]))

        # Print table
        print(f"[INFO] Dataset Metrics {self.states.shape[0]} samples\n", table)

    def _sample_random_traj_idx(self, replay_buffer: ReplayBuffer, seed = 0):
        # Force randomness
        torch.manual_seed(seed)
        # sample random start indexes
        start_idx = torch.randint(
            2,
            self.replay_buffer_cfg.trajectory_length - self.cfg.min_traj_duration_steps - 1,
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
        self.obs = self.obs[keep_idx]
        self.prev_obs = self.prev_obs[keep_idx]
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
            self.obs[index],
            self.actions[index],
        )
