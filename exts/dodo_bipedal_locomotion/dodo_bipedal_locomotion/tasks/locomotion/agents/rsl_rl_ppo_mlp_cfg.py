from omni.isaac.lab.utils import configclass

from dodo_bipedal_locomotion.utils.wrappers.rsl_rl import (
    MlpModuleCfg,
    RslRlOnPolicyRunnerMlpCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmMlpCfg,
)


@configclass
class DodoPPORunnerMlpCfg(RslRlOnPolicyRunnerMlpCfg):
    runner_type = "OnPolicyRunnerMlp"
    num_steps_per_env = 24
    max_iterations = 3001
    save_interval = 200
    experiment_name = "pf_flat"
    # TODO: To check whether the normalization parameters are correct
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
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
    mlp_cfg = MlpModuleCfg(
        latent_dim=64,
        privileged_encoder_hidden_dims=[512, 256],
        proprio_encoder_hidden_dims=[512, 256],
        activation="elu",
        output_normalize=1,
        orthogonal_init=True,
    )


class DodoRoughPPORunnerMlpCfg(DodoPPORunnerMlpCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dodo_rough"
        self.runner_type = "OnPolicyRunnerMlp"
