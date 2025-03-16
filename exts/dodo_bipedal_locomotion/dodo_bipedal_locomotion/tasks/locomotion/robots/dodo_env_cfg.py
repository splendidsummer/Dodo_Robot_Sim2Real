import math

from omni.isaac.lab.utils import configclass

from dodo_bipedal_locomotion.assets.config.dodorobot_cfg import DODOROBOT_CFG
from dodo_bipedal_locomotion.tasks.locomotion.cfg.rough_env_cfg import RoughEnvCfg
from dodo_bipedal_locomotion.tasks.locomotion.cfg.test_env_cfg import DodoEnvCfg
from dodo_bipedal_locomotion.tasks.locomotion.cfg.test_terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

#####################################
# Dodo Robot Base Environment
#####################################


@configclass
class DodoBaseEnvCfg(DodoEnvCfg):
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


############################################
# Dodo Robot Blind Flat Environment
############################################ 


@configclass
class DodoBlindFlatEnvCfg(DodoBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.rewards.pen_flat_orientation = None

        self.curriculum.terrain_levels = None


@configclass
class DodoBlindFlatEnvCfg_PLAY(DodoBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


####################################################
# Dodo Robot Blind Rough Environment
#################################################### 


@configclass
class DodoBlindRoughEnvCfg(DodoBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.pen_flat_orientation = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class DodoBlindRoughEnvCfg_PLAY(DodoBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


############################################
# Dodo Robot Blind Stairs Environment
############################################


@configclass
class DodoBlindStairsEnvCfg(DodoBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.rewards.rew_lin_vel_xy.weight = 2.0
        self.rewards.rew_ang_vel_z.weight = 1.5
        self.rewards.pen_lin_vel_z.weight = -1.0
        self.rewards.pen_ang_vel_xy.weight = -0.05
        self.rewards.pen_joint_deviation.weight = -0.2
        self.rewards.pen_action_rate.weight = -0.01
        self.rewards.pen_flat_orientation.weight = -2.5
        self.rewards.pen_undesired_contacts.weight = -1.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class DodoBlindStairsEnvCfg_PLAY(DodoBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# Pointfoot Blind Rough Environment v1
#############################


@configclass
class DodoBlindRoughEnvCfgv1(RoughEnvCfg):
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

        self.rewards.pen_flat_orientation = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class DodoBlindRoughEnvCfgv1_PLAY(DodoBlindRoughEnvCfgv1):
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
