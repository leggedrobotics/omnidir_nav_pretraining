"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--actor_critic_path", type=str, default=None, help="Where to load actor critic weights from.")
parser.add_argument("--vit_path", type=str, default=None, help="Where to load vision transformer weights from.")
parser.add_argument("--test_env", type=str, default="normal", help="Override terrians with 'plane' for testing")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import MISSING
import datetime
import gymnasium as gym
import os
import torch

from rsl_rl.modules import ActorCritic

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omnidir_nav_pretraining import env_modifier_pre_init
from omnidir_nav_pretraining import efficientformerv2_s1

SPHERE_IMAGE_HEIGHT = 64
SPHERE_IMAGE_SIDES = 6
IMAGE_START_IDX = 33

def extract_image_data(obs):
    """Extract image data from observations."""
    # Slice batch
    batch_image_data = obs[:, IMAGE_START_IDX:]
    reshaped_image_data = batch_image_data.view(
        batch_image_data.shape[0], SPHERE_IMAGE_SIDES, SPHERE_IMAGE_HEIGHT, SPHERE_IMAGE_HEIGHT, 2
    )
    
    # Reshape and process each batch
    reshaped_image_data = reshaped_image_data[:, :4, :, :, :]

    front = reshaped_image_data[:, 0]  # Shape: [batch, 64, 64]
    back = reshaped_image_data[:, 1]  # Shape: [batch, 64, 64]
    left = reshaped_image_data[:, 2]  # Shape: [batch, 64, 64]
    right = reshaped_image_data[:, 3]  # Shape: [batch, 64, 64]

    # Step 2: Arrange the images in the desired order
    # Combine the images in the top row and bottom row
    top_row = torch.cat((front, right), dim=2)  # Shape: [batch, 64, 128]
    bottom_row = torch.cat((back, left), dim=2)  # Shape: [batch, 64, 128]

    del front, left, back, right

    # Step 3: Stack the rows vertically
    image_data = torch.cat((top_row, bottom_row), dim=1)

    ##### MISTAKE!!! REMOVE IF NOT USING THE 12-03 DATA ####
    # Normalize the image data between 0 and 1
    # image_data[image_data > 40.0] = 40.0
    # image_data = image_data / 40.0

    # # Step 4: Overwrite values of 0.0 with -1.0
    # # TODO(kappi): Remove when the data is collected again.
    # image_data[image_data == 0.0] = -1.0
    ##### MISTAKE!!! REMOVE IF NOT USING THE 12-03 DATA ####

    # Add a third channel
    # TODO(Kappi): Just train with 2
    image_data = torch.cat((image_data, image_data[...,0:1]), dim=-1)
    # Make the image data 3 channels first.
    batch_image_data = image_data.permute(0, 3, 1, 2)

    batch_non_image_data = obs[: , :IMAGE_START_IDX]

    return batch_image_data, batch_non_image_data

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg = env_modifier_pre_init(env_cfg, args_cli)
    env_cfg.observations.pretraining_state = None
    env_cfg.observations.policy.embedded_spherical_image.return_embedded = False
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    actor_critic_path = args_cli.actor_critic_path
    vit_path = args_cli.vit_path

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    EMBEDDING_SIZE = 1000
    NON_IMAGE_DATA_SIZE = 33
    num_actor_obs = EMBEDDING_SIZE + NON_IMAGE_DATA_SIZE

    # Load ActorCritic
    train_cfg = agent_cfg.to_dict()
    policy_cfg = train_cfg["policy"]
    actor_critic: ActorCritic = ActorCritic(num_actor_obs, num_actor_obs, env.num_actions, **policy_cfg).to(
        args_cli.device
    )

    print(f"[INFO]: Loading model checkpoint from: {actor_critic_path}")
    # Load the saved state dictionary
    actor_critic.load_state_dict(torch.load(actor_critic_path))

    # Set models to evaluation mode
    actor_critic.eval()

    vit_model = efficientformerv2_s1(pretrained=False, resolution=128, distillation = False).to(args_cli.device)
    print(f"[INFO]: Loading model checkpoint from: {vit_path}")
    # Load the saved state dictionaries
    vit_model.load_state_dict(torch.load(vit_path))
    vit_model.eval()

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            image_data, non_image_data = extract_image_data(obs)
            goal = non_image_data[:, -3:]
            embeddings = vit_model.forward_omnidir(image_data, goal)
            combined_input = torch.cat((non_image_data, embeddings), dim=-1)
            # agent stepping
            actions = actor_critic.actor(combined_input)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
