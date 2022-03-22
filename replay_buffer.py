import os
import torch
import numpy as np

import utils

class ReplayBuffer(object):
    def __init__(self, cfg, space_info, device, log=None):
        self.cfg = cfg
        self.print_fn = print if log is None else log.info

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
        obs = state[self.space_info['key']['obs']]
        goal = state[self.space_info['key']['goal']]
        return np.stack((obs, goal))

    def push(self, state, action, reward, next_state):
        proc_state = self.process_state(state)
        proc_next_state = self.process_state(next_state)
        self.states[self.idx] = torch.from_numpy(proc_state)
        self.actions[self.idx] = torch.from_numpy(action)
        self.rewards[self.idx] = reward
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

    def fill(self, exploration_buffer, reward_fn):
        self.print_fn("filling replay buffer")
        for i in range(self.capacity):
            goal, _, _ = exploration_buffer.get_random_obs()
            _, traj_idx, step = exploration_buffer.get_random_obs()
            state = {'state': exploration_buffer.obss[traj_idx][step], 'goal':
                    goal}
            next_state = {'state': exploration_buffer.obss[traj_idx][step + 1],
                    'goal': goal}
            reward = self.cfg.reward_scaling * reward_fn(next_state['state'],
                    goal)
            self.push(state, exploration_buffer.actions[traj_idx][step + 1],
                    reward, next_state)

    def fill_unsup(self, exploration_buffer, graph_dist):
        self.print_fn("filling replay buffer")
        for i in range(self.capacity):
            g_obs, g1, g2 = exploration_buffer.get_random_obs()
            s_obs, s1, s2 = exploration_buffer.get_random_obs()
            state = {'state': s_obs, 'goal': g_obs}
            next_state = {'state': exploration_buffer.obss[s1][s2 + 1],
                    'goal': g_obs}
            nn_s = exploration_buffer.NN[s1, s2+1]
            nn_g = exploration_buffer.NN[g1, g2]
            reward = - self.cfg.reward_scaling * graph_dist[nn_s, nn_g]
            self.push(state, exploration_buffer.actions[s1][s2 + 1], reward,
                    next_state)

    def flush(self):
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    #def recompute_rewards(self, rnet, rs, batch_size):
    #    rnet.model.eval()
    #    for i in range(0, len(self), batch_size):
    #        j = min(i + batch_size, len(self))
    #        ns_batch = self.next_states[i:j]
    #        if rnet.cfg.buffer.remove_velocity:
    #            ns_batch = ns_batch[:, :, :3]
    #        self.rewards[i:j, 0] = rnet.compute_batch_reward(ns_batch[:,
    #            0].float(), ns_batch[:, 1].float())
    #        if rnet.cfg.env.rewards.sigmoid:
    #            self.rewards[i:j, 0] = torch.sigmoid(self.rewards[i:j, 0])
    #        self.rewards[i:j, 0] *= rs
    #        if rnet.cfg.env.rewards.use_diff:
    #            assert not rnet.cfg.rewards.sigmoid
    #            s_batch = self.states[i:j]
    #            if rnet.cfg.buffer.remove_velocity:
    #                s_batch = s_batch[:, :, :3]
    #            self.rewards[i:j, 0] -= rs * rnet.compute_batch_reward(
    #                    s_batch[:, 0].float(), s_batch[:, 1].float())

    #def save_buffer(self, logs_dir, return_path=False):
    #    save_path = os.path.join(logs_dir, "replay_buffer.pkl")
    #    print('Saving buffer to {}'.format(save_path))

    #    with open(save_path, 'wb') as f:
    #        pickle.dump(self.buffer, f)
    #    if return_path:
    #        return save_path

    #def load_buffer(self, save_path):
    #    print('Loading buffer from {}'.format(save_path))

    #    with open(save_path, "rb") as f:
    #        self.buffer = pickle.load(f)
    #        self.position = len(self.buffer) % self.capacity
