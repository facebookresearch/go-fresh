import torch
import numpy as np

import utils

class ReplayBuffer(object):
    def __init__(self, cfg, space_info, device):
        self.cfg = cfg

        self.capacity = cfg.capacity
        self.device = device
        obs_shape = space_info['shape']['obs']
        obs_dtype = utils.TORCH_DTYPE[space_info['type']['obs']]
        self.space_info = space_info

        self.states = torch.empty((self.capacity, 2, *obs_shape),
                dtype=obs_dtype, device=device)
        self.next_states = torch.empty((self.capacity, 2, *obs_shape),
                dtype=obs_dtype, device=device)
        self.actions = torch.empty((self.capacity, space_info['action_dim']),
                dtype=torch.float32, device=device)
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32,
                device=device)

        self.idx = 0
        self.full = False

    def process_state(self, state):
        return np.stack((state['obs'], state['goal_obs']))

    def push(self, state, action, reward, next_state):
        proc_state = self.process_state(state)
        proc_next_state = self.process_state(next_state)
        self.states[self.idx] = torch.from_numpy(proc_state)
        self.actions[self.idx] = torch.from_numpy(action)
        self.rewards[self.idx] = self.cfg.reward_scaling * reward
        self.next_states[self.idx] = torch.from_numpy(proc_next_state)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)
        states = self.states[idxs].float()
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs].float()
        mask = torch.ones((batch_size, 1), device=self.device)
        return states, actions, rewards, next_states, mask

    def flush(self):
        self.idx = 0
        self.full = False

    def is_full(self):
        return self.full

    def __len__(self):
        return self.capacity if self.full else self.idx
