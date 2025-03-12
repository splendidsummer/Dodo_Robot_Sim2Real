import gymnasium as gym

from dodo_bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_cfg import PointFootPPORunnerCfg
from dodo_bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_mlp_cfg import PFFlatPPORunnerMlpCfg

from . import dodo_env_cfg, pointfoot_env_cfg

##
# Create PPO runners for RSL-RL
##

pf_blind_flat_runner_cfg = PointFootPPORunnerCfg()
pf_blind_flat_runner_cfg.experiment_name = "pf_blind_flat"

pf_blind_rough_runner_cfg = PointFootPPORunnerCfg()
pf_blind_rough_runner_cfg.experiment_name = "pf_blind_rough"

pf_blind_stairs_runner_cfg = PointFootPPORunnerCfg()
pf_blind_stairs_runner_cfg.experiment_name = "pf_blind_stairs"

pf_mlp_blind_flat_runner_cfg = PFFlatPPORunnerMlpCfg()
pf_mlp_blind_flat_runner_cfg.experiment_name = "pf_mlp_blind_flat"

##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################

gym.register(
    id="Isaac-PF-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_flat_runner_cfg,
    },
)

#############################
# PF Blind Rough Environment
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

##############################
# PF Blind Stairs Environment
##############################

gym.register(
    id="Isaac-PF-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairsEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)


#############################
# PF Blind Rough Environment v1
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfgv1,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfgv1_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

#############################
# PF Blind Flat Environment v1
#############################

gym.register(
    id="Isaac-PF-Blind-Flat-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pf_mlp_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Flat-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_mlp_blind_flat_runner_cfg,
    },
)
