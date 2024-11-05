from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg

import omnidir_nav.mdp as mdp


def env_modifier_pre_init(cfg, args_cli):
    """Modify the environment config before initialization."""
    # change terrain config
    if args_cli.test_env == "plane":
        cfg.env_cfg.scene.terrain.terrain_type = "plane"

    # TODO(kappi): Make goal command able to return the waypoints for an agent, as well as just the goal for observations.
    # restrict goal generator to be purely goal-generated without any planner
    # cfg.env_cfg.commands.command = mdp.ConsecutiveGoalCommandCfg(
    #     resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
    #     terrain_analysis=TERRAIN_ANALYSIS_CFG,
    # )

    # Add observation group for data specific to pretraining.
    @configclass
    class PretrainingCfg(ObsGroup):
        """Observations for pretraining data group."""

        base_position = ObsTerm(func=mdp.base_position)
        base_orientation = ObsTerm(func=mdp.base_orientation_xyzw)
        base_collision = ObsTerm(
            func=mdp.base_collision,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["base", "RF_THIGH", "LF_THIGH", "RH_THIGH", "LH_THIGH"]
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    cfg.env_cfg.observations.pretraining_state = PretrainingCfg()

    cfg.env_cfg.scene.num_envs = args_cli.num_envs

    # Turn off curriculum
    cfg.env_cfg.curriculum = MISSING
    return cfg


def env_modifier_post_init(runner, args_cli):
    """Modify the environment config after initialization."""
    print(f"[INFO] Post-Init modifying Environment...")
    return runner
