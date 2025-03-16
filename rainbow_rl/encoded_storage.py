#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class EncodedRolloutStorage:
    """
    Args:
        num_envs (int): Number of environments.
        num_transitions_per_env (int): Number of transitions per environment.
        obs_shape (tuple): Shape of the observations. [num_transitions_per_env, num_envs, obs_dim ]
        privileged_steps(int): if not None, the time steps for the privileged observations.
        privileged_obs_shape (tuple): Shape of the privileged observations: if not None, the critic will use these separate observations. 
        actions_shape (tuple): Shape of the actions.
        device (str, optional): Device to store the tensors. Defaults to "cpu".
    """
    class Transition:
        """
        The obs and actions here are assumed to be tensors of shape [num_envs, obs_dim] 
        and [num_envs, action_dim], respectively.  
        """
        def __init__(self):
            self.observations = None
            self.continuous_obs = None  # Continuous steps of obs for the actor  
            self.continuous_actions = None  # Continuous steps of actions for the actor
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
            self, 
            num_envs, 
            num_transitions_per_env, 
            obs_shape,
            privileged_steps,
            privileged_obs_shape, 
            actions_shape, 
            device="cpu"
            ):
            
        self.device = device

        self.obs_shape = obs_shape
        # The number of frames of continuous obs and actions
        #  for the encoder == self.encoder.history_length 
        self.privileged_steps = privileged_steps
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        
        # NOTE: The privileged_observations is used when the critic uses a different obs from the actor
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None

        # NOTE: The continuous steps of obs for the actor is initialized to zeros 
        self.privileged_steps_obs = torch.zeros(
            num_transitions_per_env, num_envs, 
            self.privileged_steps, *obs_shape, device=self.device)
            
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.priviledged_steps_actions = torch.zeros(
            num_transitions_per_env, num_envs, 
            self.privileged_steps, *actions_shape, device=self.device)
        
        # NOTE: The dones are initialized to bool values. 
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        
        # The step variable is used to keep track of the number of transitions 
        # that have been added to the storage.
        self.step = 0
 
    def add_transitions(self, transition: Transition):
        
        # The number of transitions per environment is the first dimension of the observations tensor
        #  and the second dimension of the actions tensor is num_envs  
        num_transitions_per_env = self.observations.shape[0]

        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        if (self.privileged_steps > num_transitions_per_env).any(): 
            raise ValueError("The number of continuous steps is greater than the number of transitions per env")

        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)

        self.privileged_steps_obs[self.step].copy_(transition.continuous_obs)
        
        self.priviledged_steps_actions[self.step].copy_(transition.continuous_actions)

        self.actions[self.step].copy_(transition.actions)
        # TODO: Figure out why to manipulate the shape of rewards, dones, actions_log_prob
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # number of transitions per rollout 
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # Shuffle the indices of the transitions to make sure that the mini-batches are different at each epoch
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        # flatten [num_transitions_per_env, num_envs, ...] to [num_transitions_per_env * num_envs, ...]
        observations = self.observations.flatten(0, 1)

        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        continuous_obs = self.privileged_steps_obs.flatten(0, 1).transpose(1, 2)
        
        actions = self.actions.flatten(0, 1)
        continuous_actions = self.priviledged_steps_actions.flatten(0, 1).transpose(1, 2)

        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # obs_batch = observations[batch_idx]
                continuous_obs_batch = continuous_obs[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                continuous_actions_batch = continuous_actions[batch_idx]

                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield continuous_obs_batch, critic_observations_batch, \
                    actions_batch, continuous_actions_batch, target_values_batch, advantages_batch, \
                    returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch