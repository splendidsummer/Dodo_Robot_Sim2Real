from . import dodo_env_cfg
import gymnasium as gym

from dodo_bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_cfg import DodoPPORunnerCfg
from dodo_bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_mlp_cfg import DodoPPORunnerMlpCfg

from . import dodo_env_cfg

##
# Create PPO runners for RSL-RL
##
dodo_blind_flat_runner_cfg = DodoPPORunnerCfg()
dodo_blind_flat_runner_cfg.experiment_name = "dodo_blind_flat"

dodo_blind_rough_runner_cfg = DodoPPORunnerCfg()
dodo_blind_rough_runner_cfg.experiment_name = "dodo_blind_rough"

dodo_blind_stairs_runner_cfg = DodoPPORunnerCfg()
dodo_blind_stairs_runner_cfg.experiment_name = "dodo_blind_stairs"

dodo_mlp_blind_flat_runner_cfg = DodoPPORunnerMlpCfg()
dodo_mlp_blind_flat_runner_cfg.experiment_name = "dodo_mlp_blind_flat"

##
# Register Gym environments
##

############################
# Dodo Blind Flat Environment
############################

gym.register(
    id="Isaac-Dodo-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": dodo_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Dodo-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": dodo_blind_flat_runner_cfg,
    },
)

#############################
# Dodo Blind Rough Environment
#############################

gym.register(
    id="Isaac-Dodo-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": dodo_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Dodo-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": dodo_blind_rough_runner_cfg,
    },
)

###########################################
# Dodo Blind Stairs Environment
###########################################

gym.register(
    id="Isaac-Dodo-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindStairsEnvCfg,
        "rsl_rl_cfg_entry_point": dodo_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Dodo-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": dodo_blind_stairs_runner_cfg,
    },
)


#####################################################
# Dodo Blind Rough Environment v1
#####################################################

gym.register(
    id="Isaac-Dodo-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindRoughEnvCfgv1,
        "rsl_rl_cfg_entry_point": dodo_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Dodo-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindRoughEnvCfgv1_PLAY,
        "rsl_rl_cfg_entry_point": dodo_blind_rough_runner_cfg,
    },
)

##################################################
# Dodo Blind Flat Environment v1
##################################################

gym.register(
    id="Isaac-Dodo-Blind-Flat-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": dodo_mlp_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Dodo-Blind-Flat-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dodo_env_cfg.DodoBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": dodo_mlp_blind_flat_runner_cfg,
    },
)
