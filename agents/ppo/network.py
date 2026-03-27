import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F



def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Encoder(nn.Module):
    def __init__(self, in_shape, out_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv1d(in_shape[0], 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv1d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute conv output size
        with torch.inference_mode():
            output_size = self.conv(torch.zeros(1, in_shape[0], in_shape[1] - 1)).shape[1]
        self.fc = layer_init(nn.Linear(output_size, out_size))

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        
        self.obs_encoder = Encoder(obs_shape, 128)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128 + 1, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1)),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(128 + 1, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, act_shape[0])),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape[0]))

    def get_value(self, obs):
        obs_sig = obs[:, :, :-1]
        obs_ite = obs[:, :, -1:]
        obs_sig_encoding = F.relu(self.obs_encoder(obs_sig))
        x = torch.cat([obs_sig_encoding, obs_ite.squeeze(2)], dim=1)
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        obs_sig = obs[:, :, :-1]
        obs_ite = obs[:, :, -1:]
        obs_sig_encoding = F.relu(self.obs_encoder(obs_sig))
        x = torch.cat([obs_sig_encoding, obs_ite.squeeze(2)], dim=1)
        
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        normal = Normal(action_mean, action_std)
        if action is None:
            action = normal.sample()
        return action, normal.log_prob(action).sum(1, keepdim=True), normal.entropy().sum(1), self.critic(x), action_mean