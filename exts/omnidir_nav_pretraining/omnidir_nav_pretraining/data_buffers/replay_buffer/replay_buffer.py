from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv

from .replay_buffer_cfg import ReplayBufferCfg

if TYPE_CHECKING:
    from omnidir_nav_pretraining.runner.runner_cfg import GlobalSettingsCfg


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.
    """

    def __init__(
        self,
        cfg: ReplayBufferCfg,
        global_settings_cfg: GlobalSettingsCfg,
        env: ManagerBasedRLEnv,
    ):
        # get parameters
        self.cfg: ReplayBufferCfg = cfg
        self.global_settings_cfg = global_settings_cfg
        # get env
        self.env: ManagerBasedRLEnv = env

        # init buffers
        self._init_buffers()

        # parameters
        self._ALL_INDICES = torch.arange(self.env.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def data_collection_interval(self):
        """The interval at which data is collected.

        Defined as the number of steps (decimation x physics_dt) after which data is collected."""
        return self._data_collection_interval

    @property
    def history_collection_interval(self):
        """The interval at which history is collected.

        Defined as the number of steps (decimation x physics_dt) after which history is collected."""
        return self._history_collection_interval

    @property
    def env_buffer_filled(self):
        return self.fill_idx >= self.cfg.trajectory_length

    @property
    def is_filled(self) -> bool:
        """Whether the replay buffer is filled to capacity."""
        return torch.all(self.env_buffer_filled)

    @property
    def fill_ratio(self) -> float:
        """The ratio of the buffer that is filled."""
        return torch.mean(self.fill_idx / self.cfg.trajectory_length).item()

    @property
    def state_dim(self) -> tuple[int, ...]:
        return self.env.observation_manager.group_obs_dim["pretraining_state"]

    @property
    def observation_dim(self) -> tuple[int, ...]:
        return self.env.observation_manager.group_obs_dim["policy"]

    @property
    def action_dim(self) -> int:
        return self.env.action_manager.action.shape[1]

    @property
    def device(self) -> str:
        """The device of the replay buffer."""
        return self.cfg.buffer_device

    """
    Operations to fill the buffer
    """

    def add(
        self,
        states: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        active_envs: torch.Tensor,
    ):
        # for terminate environments, reset the step counter
        self.env_step_counter[dones] = 0

        # update local history buffers
        self._update_local_history_buffers(states, observations, active_envs)

        # update full trajectory buffers
        self._update_full_trajectory_buffers(actions, active_envs)

        # update step counter for all environments
        # NOTE: we only start the counting once all feet were in contact
        self.env_step_counter[active_envs.to(self.device)] += 1

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the buffer for the given environments"""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # reset the step counter
        self.env_step_counter[env_ids] = 0

        # reset the fill index
        self.fill_idx[env_ids] = 0

        # reset the local history buffers
        self.local_state_history[env_ids] = 0
        self.local_observation_history[env_ids] = 0

        # reset the full trajectory buffers
        self.states[env_ids] = 0
        self.observations[env_ids] = 0
        self.actions[env_ids] = 0

    def fill_leftover_envs(self):
        """Fill the buffer for the environments that are not yet filled"""
        # filter for the environments that are not yet filled
        envs_to_fill = self._ALL_INDICES[~self.env_buffer_filled].type(torch.long)
        env_fill_from_indices = self.fill_idx[~self.env_buffer_filled].type(torch.long)

        # Use already-full environemnts to fill up the buffer.
        source_env_idxs = self._ALL_INDICES[self.env_buffer_filled]

        if source_env_idxs.shape[0] < len(envs_to_fill):
            print("[Warning]: Not enough environments to fill the buffer. Repeating the source environments.")
            repeat_times = len(envs_to_fill) // source_env_idxs.shape[0]
            source_env_idxs = source_env_idxs.repeat(repeat_times + 1)[: len(envs_to_fill)]

        # fill the buffer
        for target_env_idx, source_env_idx, fill_idx in zip(envs_to_fill, source_env_idxs, env_fill_from_indices):
            use_until_idx = int(self.cfg.trajectory_length - fill_idx)
            self.states[target_env_idx, int(fill_idx) :] = self.states[source_env_idx, :use_until_idx]
            self.observations[target_env_idx, int(fill_idx) :] = self.observations[source_env_idx, :use_until_idx]

        # update the fill index
        self.fill_idx[envs_to_fill] = self.cfg.trajectory_length

    """
    Helper functions
    """

    def _init_buffers(self):
        # init collection intervals
        self._data_collection_interval = self.global_settings_cfg.command_timestep / self.env.step_dt
        if self._data_collection_interval % 1 != 0:
            print("[WARNING]: Data collection interval is not an integer. Can influence data collection.")
        # define after how many steps (decimation x physics_dt) the state/ obs should be written in the buffer
        self._history_collection_interval = self._data_collection_interval / self.cfg.history_length
        assert self._history_collection_interval >= 1, (
            "History collection frequency calculated as must be larger than "
            "physics frequency! Decrease history length as collection timestep is calculated by division of the "
            "command timestep by the history length"
        )
        if self._history_collection_interval % 1 != 0:
            print(
                "[WARNING]: History collection interval is not an integer. Will make data collection not equidistant,"
                "i.e. with an interval of 2.5 will sample at env step [3, 5, 8, 10, 13, ...]."
            )

        # full trajectory buffers
        # state is position, orientation, collision
        self.states = torch.zeros(
            (self.env.num_envs, self.cfg.trajectory_length, self.cfg.history_length, *(self.state_dim)),
            device=self.device,
        )
        # observations
        self.observations = torch.zeros(
            (
                self.env.num_envs,
                self.cfg.trajectory_length,
                self.cfg.history_length,
                *(self.observation_dim),
            ),
            device=self.device,
        )
        # actions are velocity commands
        self.actions = torch.zeros(
            (self.env.num_envs, self.cfg.trajectory_length, self.action_dim),
            device=self.device,
        )

        # local history buffers
        self.local_state_history = torch.zeros(
            (self.env.num_envs, self.cfg.history_length, *(self.state_dim)), device=self.device
        )
        self.local_observation_history = torch.zeros(
            (self.env.num_envs, self.cfg.history_length, *(self.observation_dim)),
            device=self.device,
        )

        # index buffers
        self.fill_idx = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self.env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

    def _update_local_history_buffers(
        self,
        state: torch.Tensor,
        observations: torch.Tensor,
        active_envs: torch.Tensor,
    ):
        # local robot state history buffer
        updatable_envs = (self.env_step_counter % self._history_collection_interval).type(torch.int) == 0  # noqa: E721
        # don't update if robot has not touched the ground yet (initial falling period after reset)
        updatable_envs[~active_envs] = False
        # write the current robot state into the buffer
        self.local_state_history[updatable_envs] = torch.roll(self.local_state_history[updatable_envs], 1, dims=1)
        self.local_state_history[updatable_envs, 0] = state[updatable_envs].to(self.device)
        # write the current observation into the buffer
        self.local_observation_history[updatable_envs] = torch.roll(
            self.local_observation_history[updatable_envs], 1, dims=1
        )
        self.local_observation_history[updatable_envs, 0] = observations[updatable_envs].to(self.device)

    def _update_full_trajectory_buffers(
        self,
        actions: torch.Tensor,
        active_envs: torch.Tensor,
    ):
        # for updatable environments, store the state, observations and action
        updatable_envs = (self.env_step_counter % self._data_collection_interval).type(torch.int) == 0  # noqa: E721
        # filter if step is 0
        updatable_envs[self.env_step_counter == 0] = False
        # don't update if robot has not touched the ground yet (initial falling period after reset)
        updatable_envs[~active_envs] = False

        # get the index of the updatable environments
        updatable_idxs = self._ALL_INDICES[updatable_envs]
        # check which environments are not complelty filled
        env_non_full = ~self.env_buffer_filled[updatable_idxs]
        updatable_idxs = updatable_idxs[env_non_full]
        # check if any environment to be updated
        if len(updatable_idxs) == 0:
            return

        # write state with history into buffer
        self.states[updatable_idxs, self.fill_idx[updatable_idxs]] = self.local_state_history[updatable_idxs].clone()
        # write observations with history into buffer
        self.observations[updatable_idxs, self.fill_idx[updatable_idxs]] = self.local_observation_history[
            updatable_idxs
        ].clone()
        # write actions into buffer
        self.actions[updatable_idxs, self.fill_idx[updatable_idxs]] = actions[updatable_idxs].to(self.device)

        # update fill index
        self.fill_idx[updatable_idxs] += 1
