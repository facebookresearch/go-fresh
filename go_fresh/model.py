# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# def weights_init_(m):
#    """Custom weight init for Conv2D and Linear layers. -- Denis Version"""
#    if isinstance(m, nn.Linear):
#        nn.init.orthogonal_(m.weight.data)
#        m.bias.data.fill_(0.0)
#    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
#        assert m.weight.size(2) == m.weight.size(3)
#        m.weight.data.fill_(0.0)
#        m.bias.data.fill_(0.0)
#        mid = m.weight.size(2) // 2
#        gain = nn.init.calculate_gain('relu')
#        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class FCHead(nn.Module):
    def __init__(self, obs_shape, cfg):
        super(FCHead, self).__init__()
        self.obs_size = obs_shape[0]
        if cfg.remove_velocity:
            self.obs_size = cfg.dims_to_keep
        modules = [nn.Linear(self.obs_size, cfg.hidden_size), nn.Tanh()]
        for _ in range(1, cfg.n_layers - 1):
            modules.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(cfg.hidden_size, cfg.out_size))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x[..., :self.obs_size])


class RNetPlusHead(nn.Module):
    def __init__(self, obs_shape, cfg):
        super(RNetPlusHead, self).__init__()
        self.normalize = cfg.normalize
        n, m = obs_shape[1], obs_shape[2]
        def d(x): return ((((x - 1) // 2 - 1) // 2 - 1) // 2 - 1) // 2
        conv_outdim = d(m) * d(n) * 64
        modules = [
            nn.Conv2d(3, 8, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(conv_outdim, cfg.out_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, state):
        if self.normalize:
            state = state / 255
        return self.net(state)


class RNetHead(nn.Module):
    def __init__(self, obs_shape, cfg):
        super(RNetHead, self).__init__()
        self.normalize = cfg.normalize
        n, m = obs_shape[1], obs_shape[2]
        def d(x): return (((x - 1) // 2 - 1) // 2 - 1) // 2
        conv_outdim = d(m) * d(n) * 32
        modules = [
            nn.Conv2d(3, 8, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(conv_outdim, cfg.out_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, state):
        if self.normalize:
            state = state / 255
        return self.net(state)


class SkewFitHead(nn.Module):
    def __init__(self, obs_shape, cfg):
        super(SkewFitHead, self).__init__()
        self.normalize = cfg.normalize
        conv_outdim = {48: 576, 64: 1024}
        modules = [
            nn.Conv2d(3, 16, (5, 5), stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_outdim[obs_shape[1]], cfg.out_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, state):
        if self.normalize:
            state = state / 255
        return self.net(state)


_HEADS = {
    "skew_fit": SkewFitHead,
    "rnet": RNetHead,
    "rnet+": RNetPlusHead,
    "fc": FCHead,
}


class QNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, cfg):
        super(QNetwork, self).__init__()
        head_cfg = cfg.head
        self.frame_stack = cfg.frame_stack
        self.obs_shape = obs_shape
        self.obs_head = _HEADS[head_cfg.type](self.obs_shape, head_cfg)
        self.goal_head = _HEADS[head_cfg.type](self.obs_shape, head_cfg)

        num_inputs = head_cfg.out_size * (1 + self.frame_stack)
        hidden_dim = cfg.hidden_size

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        bsz = state.size(0)
        state_rs = state[:, :self.frame_stack].reshape(
            bsz * self.frame_stack, *self.obs_shape
        )
        obs_feat = self.obs_head(state_rs)
        obs_feat = obs_feat.view(bsz, self.frame_stack, -1).flatten(1)
        goal_feat = self.goal_head(state[:, -1])
        xu = torch.cat([obs_feat, goal_feat, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape, num_actions, cfg):
        super(GaussianPolicy, self).__init__()
        head_cfg = cfg.head
        self.obs_shape = obs_shape
        self.frame_stack = cfg.frame_stack
        self.obs_head = _HEADS[head_cfg.type](self.obs_shape, head_cfg)
        self.goal_head = _HEADS[head_cfg.type](self.obs_shape, head_cfg)

        num_inputs = head_cfg.out_size * (1 + self.frame_stack)
        hidden_dim = cfg.hidden_size

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        bsz = state.size(0)
        state_rs = state[:, :self.frame_stack].reshape(
            bsz * self.frame_stack, *self.obs_shape
        )
        obs_feat = self.obs_head(state_rs)
        obs_feat = obs_feat.view(bsz, self.frame_stack, -1).flatten(1)
        goal_feat = self.goal_head(state[:, -1])
        x = torch.cat([obs_feat, goal_feat], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_shape, num_actions, cfg):
        super(DeterministicPolicy, self).__init__()
        head_cfg = cfg.head
        self.obs_head = _HEADS[head_cfg.type](obs_shape, head_cfg)
        self.goal_head = _HEADS[head_cfg.type](obs_shape, head_cfg)

        num_inputs = head_cfg.out_size * 2
        hidden_dim = cfg.hidden_size

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state):
        obs_feat = self.obs_head(state[:, 0])
        goal_feat = self.goal_head(state[:, 1])
        x = torch.cat([obs_feat, goal_feat], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
