import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg 

current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/DODO_ROBOT/robot1.usd")

DODOROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "Left_HIP_AA": 0.0,
            "Left_THIGH_FE": 0.0,
            "Left_KNEE_FE": 0.0,
            "Left_FOOT_ANKLE": 0.0,
            "Right_HIP_AA": 0.0,
            "Right_THIGH_FE": 0.0,
            "Right_SHIN_FE": 0.0,
            "Right_FOOT_ANKLE": 0.0,
        }, 

        joint_vel={".*": 0.0},
    ),

    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "Left_HIP_AA",
                "Left_THIGH_FE",
                "Left_KNEE_FE",
                "Left_FOOT_ANKLE",
                "Right_HIP_AA",
                "Right_THIGH_FE",
                "Right_SHIN_FE",
                "Right_FOOT_ANKLE",
            ],

            effort_limit=300,
            velocity_limit=100.0,       
            stiffness={
                "Left_HIP_AA": 40.0,
                "Left_THIGH_FE": 40.0,
                "Left_KNEE_FE": 40.0,
                "Left_FOOT_ANKLE": 40.0,
                "Right_HIP_AA": 40.0,
                "Right_THIGH_FE": 40.0,
                "Right_SHIN_FE": 40.0,
                "Right_FOOT_ANKLE": 40.0,
            }, 
            damping={
                "Left_HIP_AA": 2.5,
                "Left_THIGH_FE": 2.5,
                "Left_KNEE_FE": 2.5,
                "Left_FOOT_ANKLE": 2.5,
                "Right_HIP_AA": 2.5,
                "Right_THIGH_FE": 2.5,
                "Right_SHIN_FE": 2.5,
                "Right_FOOT_ANKLE": 2.5,
            }, 
            min_delay=4,  # TODO: modify this value according to the real robot
            max_delay=10,

        ),
    },
)