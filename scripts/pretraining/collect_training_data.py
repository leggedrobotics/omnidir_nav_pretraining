

"""Script to test data collection for a navigation environment."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect Training Data in Testing env.")

parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to simulate.")
parser.add_argument("--test_env", type=str, default="plane", help="Environment to collect data for.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Non-Isaac Sim Script begins."""

import os
import pickle
import torch
from dataclasses import MISSING
from datetime import datetime

import omni

from omnidir_nav_pretraining import OmnidirNavRunner, OmnidirNavRunnerCfg, env_modifier_pre_init, env_modifier_post_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # init runner cfg
    cfg: OmnidirNavRunnerCfg = OmnidirNavRunnerCfg()

    cfg.env_cfg = env_modifier_pre_init(cfg.env_cfg, args_cli=args_cli)

    # create a new stage
    omni.usd.get_context().new_stage()
    # init runner
    runner = OmnidirNavRunner(cfg=cfg, args_cli=args_cli)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)

    print(f"[INFO] Collecting data for {args_cli.test_env}")

    # collect dataset
    runner.collect()

    print("[INFO] Data collection complete. Shutting down...")

if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
