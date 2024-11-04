

from __future__ import annotations

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from fdm.model.fdm_model_cfg import FDMBaseModelCfg

from .replay_buffer_cfg import ReplayBufferCfg


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.
    """

    def __init__(
        self,
        cfg: ReplayBufferCfg,
        model_cfg: FDMBaseModelCfg,
        env: ManagerBasedRLEnv,
    ):
        # get parameters
        self.cfg = cfg
        self.model_cfg = model_cfg
        # get env
        self.env: ManagerBasedRLEnv = env

        # for simplicity, introduce flag if exteroceptive observations are present
        if "fdm_obs_exteroceptive" in self.env.observation_manager.group_obs_dim:
            self._has_exteroceptive_observation = True
        else:
            self._has_exteroceptive_observation = False

        # for simplicity, introduce flag if additional exteroceptive observations are present
        if "fdm_add_obs_exteroceptive" in self.env.observation_manager.group_obs_dim:
            self._has_add_exteroceptive_observation = True
        else:
            self._has_add_exteroceptive_observation = False

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
        return self.env.observation_manager.group_obs_dim["fdm_state"]

    @property
    def proprioceptive_observation_dim(self) -> tuple[int, ...]:
        return self.env.observation_manager.group_obs_dim["fdm_obs_proprioception"]

    @property
    def exteroceptive_observation_dim(self) -> tuple[int, ...] | None:
        if self._has_exteroceptive_observation:
            return self.env.observation_manager.group_obs_dim["fdm_obs_exteroceptive"]
        else:
            return None

    @property
    def add_exteroceptive_observation_dim(self) -> tuple[int, ...] | None:
        if self._has_add_exteroceptive_observation:
            return self.env.observation_manager.group_obs_dim["fdm_add_obs_exteroceptive"]
        else:
            return None

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
        obersevations_proprioceptive: torch.Tensor,
        obersevations_exteroceptive: torch.Tensor | None,
        actions: torch.Tensor,
        dones: torch.Tensor,
        feet_contact: torch.Tensor,
        add_observation_exteroceptive: torch.Tensor | None = None,
    ):
        # for terminate environments, reset the step counter
        self.env_step_counter[dones] = 0

        # for colliding environments, record the last state before they get reset (will happen in the following sim step)
        # only record if its history buffer is filled and does not terminate in first rec step
        colliding_envs = states[..., 7].to(torch.bool)
        colliding_envs[self.env_step_counter < self._data_collection_interval] = False

        # update local history buffers
        self._update_local_history_buffers(colliding_envs, states, obersevations_proprioceptive, feet_contact)

        # update full trajectory buffers
        self._update_full_trajectory_buffers(
            colliding_envs, obersevations_exteroceptive, actions, feet_contact, add_observation_exteroceptive
        )

        # update step counter for all environments
        # NOTE: we only start the counting once all feet were in contact
        self.env_step_counter[feet_contact.to(self.device)] += 1

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
        self.local_proprioceptive_observation_history[env_ids] = 0

        # reset the full trajectory buffers
        self.states[env_ids] = 0
        self.observations_proprioceptive[env_ids] = 0
        self.actions[env_ids] = 0

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[env_ids] = 0
        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[env_ids] = 0

    def fill_leftover_envs(self):
        """Fill the buffer for the environments that are not yet filled"""
        # identify collision for each environment
        collision_indices = torch.nonzero(torch.any(self.states[..., 7], dim=-1))

        # get the last collision index
        collision_envs, unique_indices = torch.unique(collision_indices[:, 0], return_inverse=True)
        env_split_data = torch.split(collision_indices[:, 1], torch.bincount(unique_indices).tolist())
        collision_max_indices = torch.tensor([torch.max(env_indices) for env_indices in env_split_data])

        # Note: add 1 to collision_max_idx to avoid cropping the collision event
        collision_max_indices += 1

        # check if any environment has never collided
        not_collided_envs = list(set(self._ALL_INDICES.tolist()) - set(collision_envs.tolist()))
        collision_envs = torch.concatenate((collision_envs, torch.tensor(not_collided_envs)))
        collision_max_indices = torch.concatenate((collision_max_indices, torch.zeros(len(not_collided_envs))))

        # sort the collision environments and collision max indices
        collision_envs, sort_indices = collision_envs.sort()
        collision_max_indices = collision_max_indices[sort_indices]

        # filter for the environments that are not yet filled
        envs_to_fill = collision_envs[~self.env_buffer_filled].type(torch.long)
        env_fill_from_indices = collision_max_indices[~self.env_buffer_filled].type(torch.long)

        # randomly select an environment to fill up the buffer from the environments that have already filled the buffer
        source_env_idxs = self._ALL_INDICES[
            self.fill_idx >= self.cfg.trajectory_length - torch.min(env_fill_from_indices)
        ]
        if source_env_idxs.shape[0] < len(envs_to_fill):
            print("[Warning]: Not enough environments to fill the buffer. Repeating the source environments.")
            repeat_times = len(envs_to_fill) // source_env_idxs.shape[0]
            source_env_idxs = source_env_idxs.repeat(repeat_times + 1)[: len(envs_to_fill)]

        # fill the buffer
        for target_env_idx, source_env_idx, collision_max_idx in zip(
            envs_to_fill, source_env_idxs, env_fill_from_indices
        ):
            use_until_idx = int(self.cfg.trajectory_length - collision_max_idx)
            self.states[target_env_idx, int(collision_max_idx) :] = self.states[source_env_idx, :use_until_idx]
            self.observations_proprioceptive[target_env_idx, int(collision_max_idx) :] = (
                self.observations_proprioceptive[source_env_idx, :use_until_idx]
            )
            if self._has_exteroceptive_observation:
                self.observations_exteroceptive[target_env_idx, int(collision_max_idx) :] = (
                    self.observations_exteroceptive[source_env_idx, :use_until_idx]
                )
            self.actions[target_env_idx, int(collision_max_idx) :] = self.actions[source_env_idx, :use_until_idx]
            if self._has_add_exteroceptive_observation:
                self.add_observations_exteroceptive[target_env_idx, int(collision_max_idx) :] = (
                    self.add_observations_exteroceptive[source_env_idx, :use_until_idx]
                )

        # update the fill index
        self.fill_idx[envs_to_fill] = self.cfg.trajectory_length

    """
    Helper functions
    """

    def _init_buffers(self):
        # init collection intervals
        self._data_collection_interval = self.model_cfg.command_timestep / self.env.step_dt
        if self._data_collection_interval % 1 != 0:
            print("[WARNING]: Data collection interval is not an integer. Can influence data collection.")
        # define after how many steps (decimation x physics_dt) the state/ proprioceptive obs should be written in the buffer
        self._history_collection_interval = self._data_collection_interval / self.model_cfg.history_length
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
            (self.env.num_envs, self.cfg.trajectory_length, self.model_cfg.history_length, *(self.state_dim)),
            device=self.device,
        )
        # proprioceptive observations
        self.observations_proprioceptive = torch.zeros(
            (
                self.env.num_envs,
                self.cfg.trajectory_length,
                self.model_cfg.history_length,
                *(self.proprioceptive_observation_dim),
            ),
            device=self.device,
        )
        # exteroceptive observations
        if self._has_exteroceptive_observation:
            self.observations_exteroceptive = torch.zeros(
                (self.env.num_envs, self.cfg.trajectory_length, *(self.exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.observations_exteroceptive = None
        # additional exteroceptive observations
        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive = torch.zeros(
                (self.env.num_envs, self.cfg.trajectory_length, *(self.add_exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.add_observations_exteroceptive = None
        # actions are velocity commands
        self.actions = torch.zeros(
            (self.env.num_envs, self.cfg.trajectory_length, self.action_dim),
            device=self.device,
        )

        # local history buffers
        self.local_state_history = torch.zeros(
            (self.env.num_envs, self.model_cfg.history_length, *(self.state_dim)), device=self.device
        )
        self.local_proprioceptive_observation_history = torch.zeros(
            (self.env.num_envs, self.model_cfg.history_length, *(self.proprioceptive_observation_dim)),
            device=self.device,
        )

        # index buffers
        self.fill_idx = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self.env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

    def _update_local_history_buffers(
        self,
        colliding_envs: torch.Tensor,
        state: torch.Tensor,
        obersevations_proprioceptive: torch.Tensor,
        feet_contact: torch.Tensor,
    ):
        # local robot state history buffer
        updatable_envs = (self.env_step_counter % self._history_collection_interval).type(torch.int) == 0  # noqa: E721
        # update if environment is colliding
        updatable_envs[colliding_envs] = True
        # don't update if robot has not touched the ground yet (initial falling period after reset)
        updatable_envs[~feet_contact] = False
        # write the current robot state into the buffer
        self.local_state_history[updatable_envs] = torch.roll(self.local_state_history[updatable_envs], 1, dims=1)
        self.local_state_history[updatable_envs, 0] = state[updatable_envs].to(self.device)
        # write the current proprioceptive observation into the buffer
        self.local_proprioceptive_observation_history[updatable_envs] = torch.roll(
            self.local_proprioceptive_observation_history[updatable_envs], 1, dims=1
        )
        self.local_proprioceptive_observation_history[updatable_envs, 0] = obersevations_proprioceptive[
            updatable_envs
        ].to(self.device)

    def _update_full_trajectory_buffers(
        self,
        colliding_envs: torch.Tensor,
        obersevations_exteroceptive: torch.Tensor | None,
        actions: torch.Tensor,
        feet_contact: torch.Tensor,
        add_observation_exteroceptive: torch.Tensor | None,
    ):
        # for updatable environments, store the state, proprioceptive observation, exteroceptive observation and action
        updatable_envs = (self.env_step_counter % self._data_collection_interval).type(torch.int) == 0  # noqa: E721
        # filter if step is 0
        updatable_envs[self.env_step_counter == 0] = False
        # enable if environment is colliding
        updatable_envs[colliding_envs] = True
        updatable_envs[self.env_step_counter == self._data_collection_interval] = ~colliding_envs[
            self.env_step_counter == self._data_collection_interval
        ].to(self.device)
        # don't update if robot has not touched the ground yet (initial falling period after reset)
        updatable_envs[~feet_contact] = False
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
        # write propriocpetive observations with history into buffer
        self.observations_proprioceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
            self.local_proprioceptive_observation_history[updatable_idxs].clone()
        )
        # write exteroceptive observations into buffer
        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
                obersevations_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )
        # write actions into buffer
        self.actions[updatable_idxs, self.fill_idx[updatable_idxs]] = actions[updatable_idxs].to(self.device)

        # write additional exteroceptive observations into buffer if requested
        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
                add_observation_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )
        # update fill index
        self.fill_idx[updatable_idxs] += 1
