
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdm.runner import FDMRunner

    from .base_agent_cfg import AgentCfg


class PDAgent(ABC):
    def __init__(self, cfg: AgentCfg, runner: FDMRunner):
        self.cfg: AgentCfg = cfg
        self._runner: FDMRunner = runner

        # init buffers
        self._init_buffers()

    """
    Properties
    """

    @property
    def device(self):
        return self._runner.env.device

    @property
    def action_dim(self):
        return self._runner.env.action_manager.action.shape[1]

    @property
    def resample_interval(self):
        return self._runner.cfg.model_cfg.command_timestep / self._runner.env.step_dt

    """
    Operations
    """

    def act(self, obs: dict, dones: torch.Tensor, feet_contact: torch.Tensor):
        """Get the next actions for all environments.

        Args:
            obs: The current observation.
            dones: The done flags for all environments. Specifies the terminated environments for which the
                next command has to be resampled before the official resampling period.

        Returns:
            The next action for all environments. For all non-resampled environments, this will be the
                same as the previous action.
        """
        # for colliding environments, reset the action that is applied in the next step call and that will be recorded
        # for the current state
        # the action is applied to the step that resets the environment and to further steps until the next recording
        # after the collision
        colliding_envs = obs["fdm_state"][..., 7].to(torch.bool)
        if torch.any(colliding_envs):
            self.reset(obs=obs, env_ids=self._ALL_INDICES[colliding_envs], return_actions=False)

        # reset env counter when env is reset in simulation
        self.env_step_counter[dones] = 0

        # determine which environments should be updated depending on the sim time
        # NOTE: filter all environments on the first step
        # NOTE: filter all environments where not all feet have touched the ground yet
        updatable_envs = self.env_step_counter % self.resample_interval == 0
        updatable_envs[self.env_step_counter == 0] = False
        updatable_envs[~feet_contact] = False
        updatable_envs[self.env_step_counter == self.resample_interval] = ~colliding_envs[
            self.env_step_counter == self.resample_interval
        ]
        # for environments that should be resampled, increase the counter
        self._plan_step[updatable_envs] += 1

        # agents (e.g. sampling planner agent) can depend on the reset state of the environment, which is available
        # after the sim reset, i.e., when dones is True. Therefore, this agent should perform another reset before
        # the first time their _plan_step gets increased.
        # for these cases, the initial action given from the agent should be random and all further actions can be
        # more targeted
        planner_reset_after = updatable_envs & (self._plan_step == 1)
        if torch.any(planner_reset_after):
            self.plan_reset(obs=obs, env_ids=self._ALL_INDICES[planner_reset_after])

        # ensure to replan the environments out of a plan with their last step as new init
        env_to_replan = self._ALL_INDICES[self._plan_step >= (self.cfg.horizon - 1)]
        self.plan(env_ids=env_to_replan, obs=obs, random_init=False)
        self._plan_step[env_to_replan] = 0

        # in expectation of the next env step, increase the counter
        # NOTE: we only start the counting once all feet were in contact
        self.env_step_counter[feet_contact] += 1

        return self._plan[self._ALL_INDICES, self._plan_step]

    def reset(
        self, obs: dict | None = None, env_ids: torch.Tensor | None = None, return_actions: bool = True
    ) -> torch.Tensor | None:
        # handle case for all envs
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset buffers
        self._plan_step[env_ids] = 0
        self._plan[env_ids] = 0
        # plan
        self.plan(env_ids=env_ids, obs=obs, random_init=True)
        if return_actions:
            return self._plan[self._ALL_INDICES, self._plan_step]

    @abstractmethod
    def plan(self, obs: dict | None = None, env_ids: torch.Tensor | None = None, random_init: bool = True):
        pass

    def plan_reset(self, obs: dict, env_ids: torch.Tensor):
        """Replan for already reset environments in the simulator.

        Necessary for the sampling-planner agent that depends on the new observations from the reset environment."""
        pass

    def debug_viz(self, env_ids: list[int] | None = None):
        pass

    """
    Helper functions
    """

    def _init_buffers(self):
        self._ALL_INDICES = torch.arange(self._runner.env.num_envs, device=self.device, dtype=torch.long)
        # plan buffers
        self._plan_step = torch.zeros(self._runner.env.num_envs, device=self.device, dtype=torch.long)
        self._plan = torch.zeros((self._runner.env.num_envs, self.cfg.horizon, self.action_dim), device=self.device)
        # env step counter
        self.env_step_counter = torch.zeros(self._runner.env.num_envs, device=self.device, dtype=torch.long)
