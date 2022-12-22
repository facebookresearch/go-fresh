# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from .utils import TORCH_DTYPE


class ReplayBuffer(object):
    def __init__(self, cfg, space_info):
        self.cfg = cfg
        self.capacity = cfg.capacity
        self.space_info = space_info
        self.device = "cpu"
        self.nframes = cfg.frame_stack + 1  # 1 extra for goal
        self.states = torch.empty(
            (self.capacity, self.nframes, *space_info["shape"]["obs"]),
            dtype=TORCH_DTYPE[space_info["type"]["obs"]]
        )
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(
            (self.capacity, space_info["action_dim"]), dtype=torch.float32
        )
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.not_dones = torch.empty((self.capacity, 1), dtype=torch.bool)

    def write(self, i, state, action, reward, next_state, done=False):
        assert i < self.capacity
        self.actions[i] = torch.from_numpy(action)
        self.rewards[i] = self.cfg.reward_scaling * reward
        self.not_dones[i] = not done
        for j in range(self.nframes):
            self.states[i, j] = torch.from_numpy(state[j])
            self.next_states[i, j] = torch.from_numpy(next_state[j])

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)
        states = self.states[idxs].float()
        actions = self.actions[idxs].float()
        rewards = self.rewards[idxs].float()
        next_states = self.next_states[idxs].float()
        mask = self.not_dones[idxs]
        return states, actions, rewards, next_states, mask

    def share_memory(self):
        self.states.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.next_states.share_memory_()
        self.not_dones.share_memory_()

    def to(self, device):
        if self.device == device:
            return
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.not_dones = self.not_dones.to(device)
        self.device = device

    def __len__(self):
        return self.capacity
