import torch
import numpy as np

import utils


class ReplayBuffer(object):
    def __init__(self, cfg, space_info):
        self.cfg = cfg
        self.capacity = cfg.capacity
        self.space_info = space_info
        self.device = "cpu"
        self.nframes = cfg.frame_stack + 1  # 1 extra for goal
        self.states = torch.empty(
            (self.capacity, self.nframes, *space_info["shape"]["obs"]),
            dtype=utils.TORCH_DTYPE[space_info["type"]["obs"]]
        )
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(
            (self.capacity, space_info["action_dim"]), dtype=torch.float32
        )
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32)

    def write(self, i, state, action, reward, next_state):
        assert i < self.capacity
        self.actions[i] = torch.from_numpy(action)
        self.rewards[i] = self.cfg.reward_scaling * reward
        for j in range(self.nframes):
            self.states[i, j] = torch.from_numpy(state[j])
            self.next_states[i, j] = torch.from_numpy(next_state[j])

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)
        states = self.states[idxs].float()
        actions = self.actions[idxs].float()
        rewards = self.rewards[idxs].float()
        next_states = self.next_states[idxs].float()
        mask = torch.ones((batch_size, 1), device=self.device)
        return states, actions, rewards, next_states, mask

    def share_memory(self):
        self.states.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.next_states.share_memory_()

    def to(self, device):
        if self.device == device:
            return
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.device = device

    def __len__(self):
        return self.capacity
