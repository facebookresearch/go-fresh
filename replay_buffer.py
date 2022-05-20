import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, cfg, space_info):
        self.cfg = cfg

        self.capacity = cfg.capacity
        self.space_info = space_info

        self.states = np.empty(
            (self.capacity, 2, *space_info["shape"]["obs"]),
            dtype=space_info["type"]["obs"]
        )
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty((self.capacity, space_info["action_dim"]), dtype=float)
        self.rewards = np.empty((self.capacity, 1), dtype=float)

        self.is_torch = False
        self.idx = 0
        self.full = False

    def process_state(self, state):
        return np.stack((state["obs"], state["goal_obs"]))

    def push(self, state, action, reward, next_state):
        assert not self.is_torch
        self.states[self.idx] = self.process_state(state)
        self.actions[self.idx] = action
        self.rewards[self.idx] = self.cfg.reward_scaling * reward
        self.next_states[self.idx] = self.process_state(next_state)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        assert self.is_torch
        idxs = np.random.randint(0, len(self), size=batch_size)
        states = self.states[idxs].float()
        actions = self.actions[idxs].float()
        rewards = self.rewards[idxs].float()
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

    def to_torch(self, device):
        if self.is_torch:
            return
        self.states = torch.from_numpy(self.states).to(device)
        self.actions = torch.from_numpy(self.actions).to(device)
        self.rewards = torch.from_numpy(self.rewards).to(device)
        self.next_states = torch.from_numpy(self.next_states).to(device)
        self.device = device
        self.is_torch = True

    def to_numpy(self):
        if not self.is_torch:
            return
        self.states = self.states.cpu().numpy()
        self.actions = self.actions.cpu().numpy()
        self.rewards = self.rewards.cpu().numpy()
        self.next_states = self.next_states.cpu().numpy()
        self.is_torch = False
