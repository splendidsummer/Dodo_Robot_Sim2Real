import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.distributions import Categorical 


class HistoryEncoder(nn.Module):
    def __init__(self, activation_fn, obs_size, actions_size, tsteps, 
                 num_transition_per_env,  output_size):
        super(HistoryEncoder, self).__init__()
        input_size = obs_size + actions_size  
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = (obs_size + actions_size) * tsteps
        # output_size = latent_dim == input_dim of policyNet 
        self.output_shape = output_size

        # We are not going the encoder constructed by linear layers  
        # self.encoder = nn.Sequential(
        #         nn.Linear(input_size, 128), self.activation_fn(),
        #         nn.Linear(128, 32), self.activation_fn()
        #         )

        if tsteps >= 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif 50>= tsteps >= 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif  20 >= tsteps >= 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()



    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std


class MultivariateGaussianDiagonalCovariance2(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance2, self).__init__()
        assert(dim == 12)
        self.dim = dim
        self.std_param = nn.Parameter(init_std * torch.ones(dim // 2))
        self.distribution = None

    def sample(self, logits):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std_param.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std_param.data = new_std


