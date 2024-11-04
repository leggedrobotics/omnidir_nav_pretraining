

from __future__ import annotations

import numpy as np
import os
import random
import time
import torch
from dataclasses import MISSING

import cv2

import carb

from omni.isaac.lab.envs import ManagerBasedRLEnv

from fdm.data_buffers import ReplayBuffer

from omnidir_nav_pretraining.agent import PDAgent
from omnidir_nav.model import OmnidirNavModel
from .runner_cfg import OmnidirNavRunnerCfg
from .trainer import Trainer

class OmnidirNavRunner:
    def __init__(self, cfg: OmnidirNavRunnerCfg, args_cli, eval: bool = False):
        self.cfg = cfg
        self.args_cli = args_cli
        self.eval = eval

        if self.eval:
            self.cfg.trainer_cfg.resume = True
            self.cfg.trainer_cfg.logging = False

        # override the resampling command of the command generator with `trainer_cfg.command_timestep`
        self.cfg.env_cfg.episode_length_s = self.cfg.model_cfg.command_timestep * (
            self.cfg.replay_buffer_cfg.trajectory_length + 1
        )

        # setup
        self.setup()

    """
    Properties
    """

    @property
    def device(self) -> str:
        """The device to use for training."""
        return self.env.device

    """
    Operations
    """

    def setup(self):
        # setup environment
        self.env: ManagerBasedRLEnv = ManagerBasedRLEnv(self.cfg.env_cfg)
        # setup model
        self.model: OmnidirNavModel = (
            self.cfg.model_cfg.class_type(cfg=self.cfg.model_cfg, device=self.device)
        )
        self.model.to(self.device)
        # setup replay buffer
        self.replay_buffer = ReplayBuffer(self.cfg.replay_buffer_cfg, self.model.cfg, self.env)
        # setup trainer
        self.trainer = Trainer(
            cfg=self.cfg.trainer_cfg,
            replay_buffer_cfg=self.cfg.replay_buffer_cfg,
            model=self.model,
            device=self.device,
            eval=self.eval,
        )
        self.agent = self.cfg.agent_cfg.class_type(self.cfg.agent_cfg, self)

        # setup buffers to track first contact with the ground after reset.
        self.feet_idx, _ = self.env.scene.sensors["contact_forces"].find_bodies(self.cfg.body_regex_contact_checking)
        self.feet_contact = torch.zeros(
            (self.env.num_envs, len(self.feet_idx)), dtype=torch.bool, device=self.env.device
        )
        self.feet_non_contact_counter = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

        print("[INFO]: Setup complete.")

    def close(self):
        self.env.close()
        del self.model
        del self.replay_buffer
        del self.trainer
        if self.agent is not MISSING and self.agent is not None:
            del self.agent
        
    """
    Helper functions
    """

    @torch.inference_mode()
    def collect(self, eval: bool = False):
        """Collect data from the environment and store it in the trainer's storage."""
        print("[INFO]: Collecting data...")
        # reset environment
        with torch.inference_mode():
            obs, _ = self.env.reset(random.randint(0, 1000000))
            # the contact sensor is delayed, execute delay+1 steps to reset all environments correctly
            if torch.any(obs["fdm_state"][..., 7]):
                for _ in range(self.cfg.env_cfg.scene.contact_forces.history_length - self.cfg.env_cfg.decimation + 1):
                    obs, _, dones, _, _ = self.env.step(torch.zeros(self.env.num_envs, 3, device=self.env.device))
                if dones.sum() != 0:
                    carb.log_warn("Environments should not be done after reset.")

        # reset replay buffer
        self.replay_buffer.reset()
        # reset agent
        actions = self.agent.reset(obs)

        # reset feet contact amd counter
        self.feet_contact[:] = False
        self.feet_non_contact_counter[:] = 0

        # collect data
        sim_time = 0.0
        process_time = 0.0
        plan_time = 0.0
        collect_time = []
        info_counter = 1
        step_counter = 0

        while  not self.replay_buffer.is_filled:
            ###
            # Step the environment
            ###
            sim_start = time.time()
            with torch.inference_mode():
                obs, _, dones, _, _ = self.env.step(actions.clone())
            sim_time += time.time() - sim_start

            ###
            # Determine feet contact
            ###
            # Note: only start recording and changing actions when all feet have touched the ground
            # dones here will include any environments that have been reset due to not touching the ground for a while.
            # obs_after_reset is observations recomputed after resetting environments that haven't touched the ground
            # for a while. If none are reset, obs_after_reset isn't computed to save time, and the original observations
            # are used.
            feet_all_contact, dones, obs_after_reset = self._feet_contact_handler(dones)
            obs = obs_after_reset if obs_after_reset is not None else obs

            ###
            # Plan the actions for the current state
            ###
            # Note: We do this before updating the replay buffer because each entry in the buffer is the
            # state-action pair where the action is the response to the state.
            plan_start = time.time()

            # get actions
            actions = self.agent.act(obs, dones.to(torch.bool).clone(), feet_contact=feet_all_contact)

            plan_time += time.time() - plan_start

            ###
            # Update replay buffer
            ###
            update_buffer_start = time.time()
            self.replay_buffer.add(
                states=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                obersevations_exteroceptive=(
                    obs["fdm_obs_exteroceptive"].clone() if "fdm_obs_exteroceptive" in obs else None
                ),
                actions=actions.clone(),
                dones=dones.to(torch.bool).clone(),
                feet_contact=feet_all_contact,
                add_observation_exteroceptive=(
                    obs["fdm_add_obs_exteroceptive"] if "fdm_add_obs_exteroceptive" in obs else None
                ),
            )
            process_time += time.time() - update_buffer_start

            ###
            # Update timers
            ###

            # print fill ratio information
            if self.replay_buffer.fill_ratio > 0.1 * info_counter:
                print(
                    f"[INFO] Fill ratio: {self.replay_buffer.fill_ratio:.2f} \tPlan time: \t{plan_time:.2f}s \tSim"
                    f" time: \t{sim_time:.2f}s \tUpdate time: \t{process_time:.2f}s"
                )
                # save overall time
                collect_time.append(plan_time + sim_time + process_time)
                # reset times
                plan_time = 0.0
                sim_time = 0.0
                process_time = 0.0
                info_counter += 1

            step_counter += 1
            if step_counter % 1000 == 0:
                print(f"[INFO] Step {step_counter} completed.")

            ###
            # Break if some environments take too long to be filled
            ###

            if (
                not self.replay_buffer.is_filled
                and self.replay_buffer.fill_ratio > 0.95
                and plan_time + sim_time + process_time > 1.5 * np.mean(collect_time)
            ):
                print("[WARNING]: Collection took too long for some environments. Stopping collection.")
                self.replay_buffer.fill_leftover_envs()
                break

        # slice into samples and populate storage of trainer
        if eval:
            _, max_vel, max_acc = self.trainer.val_dataset.populate(replay_buffer=self.replay_buffer)

        else:
            _, max_vel, max_acc = self.trainer.train_dataset.populate(replay_buffer=self.replay_buffer)

        # save depth images as debug info
        if self.args_cli.debug and self.args_cli.env == "depth" and not eval:
            os.makedirs(self.trainer.log_dir + "/debug", exist_ok=True)
            for idx in range(min(self.trainer.train_dataset.num_samples, 100)):
                depth_img = self.trainer.train_dataset.obs_exteroceptive[idx, :, :, 0].cpu().numpy() * 1000
                depth_img = depth_img.astype(np.uint16)
                cv2.imwrite(self.trainer.log_dir + f"/debug/depth_{idx}.png", depth_img)

        print("[INFO]: Data collection complete.")

    @torch.inference_mode()
    def _feet_contact_handler(
        self, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]] | None]:
        # update for done environments
        self.feet_contact[dones] = False
        self.feet_non_contact_counter[dones] = 0
        # determine which feet are in contact
        self.feet_contact[
            torch.norm(self.env.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_idx], dim=-1) > 1
        ] = True
        feet_all_contact = torch.all(self.feet_contact, dim=-1)
        # feet non-contact counter
        self.feet_non_contact_counter[feet_all_contact] = 0
        self.feet_non_contact_counter[~feet_all_contact] += 1
        # reset for envs that have not touched the ground for a while
        obs_after_reset = None
        reset_envs = self.feet_non_contact_counter > 200
        if torch.any(reset_envs):
            print("[WARNING]: Resetting environments that have not touched the ground for a while.")
            # NOTE: this does not affect the data collection, as it will only start when all feet are in contact
            with torch.inference_mode():
                self.env._reset_idx(self.agent._ALL_INDICES[reset_envs])
                # compute observations
                # note: done after reset to get the correct observations for reset envs
                obs_after_reset = self.env.observation_manager.compute()
            # reset counter for these environments and add to done environments
            dones[reset_envs] = True
            self.feet_non_contact_counter[reset_envs] = 0

        return feet_all_contact, dones, obs_after_reset
