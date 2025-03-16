"""
Rainbow RL configuration file for bipedal locomotion.

"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import MISSING


@configclass
class HistoryEncoderCfg:

    """Configuration for the PPO history encoder networks."""

    class_name = "HistoryEncoder"
    """The history class name. Default is HistoryEncoder."""

    history_length: int = MISSING  # type: ignore
    """The length of the privileged steps of obs and actions."""
    

@configclass  
class RainbowRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type = "RainbowRunner"

    num_steps_per_env = 24
    max_iterations = 3001
    save_interval = 200
    experiment_name = "bipedal_locomotion"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 128, 64],

        # critic_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[256, 128, 64],

        activation="elu",
        # activation="relu", 
    )
    # TODO: Figure out why the error is raised  
    encoder = HistoryEncoderCfg(
        class_name="HistoryEncoder",  # type: ignore
        history_length=10)  # type: ignore
    
    # Hyperparameters for the PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="EncodedPPO", 
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
