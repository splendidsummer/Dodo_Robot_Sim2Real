import math

from omni.isaac.lab.utils import configclass

from dodo_bipedal_locomotion.assets.config.dodorobot_cfg import DODOROBOT_CFG
from dodo_bipedal_locomotion.tasks.locomotion.cfg.rough_env_cfg import RoughEnvCfg
from dodo_bipedal_locomotion.tasks.locomotion.cfg.flat_env_cfg import FlatEnvCfg
from dodo_bipedal_locomotion.tasks.locomotion.cfg.test_terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)



from dodo_bipedal_locomotion.tasks.locomotion.cfg.test_env_cfg import PFEnvCfg
from dodo_bipedal_locomotion.tasks.locomotion.cfg.test_terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

######################
# Base Environment
######################


@configclass
class DodoBaseEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DODOROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
      
        self.scene.robot.init_state.joint_pos = {
                "Left_HIP_AA": 0.0,
                "Left_THIGH_FE": 0.0,
                "Left_KNEE_FE": 0.0,
                "Left_FOOT_ANKLE": 0.0,
                "Right_HIP_AA": 0.0,
                "Right_THIGH_FE": 0.0,
                "Right_SHIN_FE": 0.0,
                "Right_FOOT_ANKLE": 0.0,
        }
        
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
 

@configclass
class DodoBaseEnvCfg_PLAY(DodoBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


#########################################
# Dodo Robot Blind Flat Environment
#########################################


@configclass
class DodoBlindFlatEnvCfg(DodoBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class DodoBlindFlatEnvCfg_PLAY(DodoBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
