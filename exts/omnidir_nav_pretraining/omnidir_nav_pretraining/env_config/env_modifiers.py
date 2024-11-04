# env_modifier_pre_init, env_modifier_post_init

def env_modifier_pre_init(cfg, args_cli):
    """Modify the environment config before initialization."""
        # change terrain config
    cfg.env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        #     project_uvw=True,
        # ),
        debug_vis=False,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )
    # restrict goal generator to be purely goal-generated without any planner
    cfg.env_cfg.commands.command = mdp.ConsecutiveGoalCommandCfg(
        resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        terrain_analysis=TERRAIN_ANALYSIS_CFG,
    )
    if hasattr(cfg.env_cfg.observations, "planner_obs"):
        cfg.env_cfg.observations.planner_obs.goal.func = mdp.goal_command_w_se2
        cfg.env_cfg.observations.planner_obs.goal.params = {"command_name": "command"}
    cfg.env_cfg.curriculum = MISSING
    return cfg

def env_modifier_post_init(cfg, args_cli):
    """Modify the environment config after initialization."""
    cfg.env_cfg.scene.num_envs = args_cli.num_envs
    return cfg