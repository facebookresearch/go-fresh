import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import utils

from rnet.model import RNetModel
from rnet.oracle_curri import Memory

class RNetBuffer(torch.utils.data.Dataset):
    def __init__(self, exploration_buffer, cfg):
        self.traj_buff = exploration_buffer
        self.traj_len = self.traj_buff.traj_len
        self.cfg = cfg

    def sample(self):
        return self.traj_buff.sample()

    def get_size(self):
        return len(self.traj_buff)

    def get_traj(self, traj_idx):
        return self.traj_buff.get_traj(traj_idx)

    def get_obs(self, traj_idx, obs_idx):
        return self.traj_buff.get_obs(traj_idx, obs_idx)

    def get_state(self, traj_idx, state_idx):
        return self.traj_buff.get_state(traj_idx, state_idx)

    def get_obss_array(self):
        return self.traj_buff.get_obss_array()

    def get_states_array(self):
        return self.traj_buff.get_states_array()

    def get_break_condition(self, i1, i2, pos, in_traj):
        if pos:
            return True
        else:
            if in_traj:
                return abs(i1 - i2) > self.cfg.thresh
            else:
                return True

    def get_search_interval(self, i1, pos):
        assert 2 * self.cfg.thresh < self.traj_len - 1, "negative sample might not exist!"
        low = 0
        high = self.traj_len - 1
        if pos:
            low = max(low, i1 - self.cfg.thresh)
            high = min(i1 + self.cfg.thresh, high)
        return low, high

    def set_num_pairs(self, num_pairs):
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, i, enforce_neg=False):
        pos = 0 if enforce_neg else random.randint(0, 1)
        traj1 = self.sample()['obs']
        i1 = random.randint(0, self.traj_len - 1)

        in_traj = pos or random.random() < self.cfg.in_traj_ratio
        traj2 = traj1 if in_traj else self.sample()['obs']

        interval = self.get_search_interval(i1, pos)
        while True:
            i2 = random.randint(*interval)
            if self.get_break_condition(i1, i2, pos, in_traj):
                break

        if self.cfg.check_neg_obs:
            if not pos and (traj1[i1] == traj2[i2]).all():
                return self.__getitem__(i, enforce_neg=True)

        if random.random() > 0.5:
            return traj1[i1], traj2[i2], pos
        else:
            return traj2[i2], traj1[i1], pos

class RNetMemory(Memory):
    def __init__(self, cfg, space_info, feat_size, device):
        super().__init__(cfg, space_info)
        self.cfg = cfg
        self.device = device
        obs_dtype = utils.TORCH_DTYPE[space_info['type']['obs']]
        self.obss = torch.zeros((cfg.capacity, *space_info['shape']['obs']),
                dtype=obs_dtype).to(device)
        self.embs = torch.zeros((cfg.capacity, feat_size)).to(device)

    def embed_memory(self, rnet_model):
        rnet_model.eval()
        with torch.no_grad():
            self.embs[:len(self)] = rnet_model.get_embedding(
                self.obss[:len(self)].float())

    def compare_embeddings(self, emb, rnet_model):
        emb_batch = emb.repeat(len(self), 1)
        with torch.no_grad():
            return rnet_model.compare_embeddings(emb_batch,
                    self.embs[:len(self)], batchwise=True)[:, 0]

    def compute_novelty(self, rnet_model, x):
        rnet_model.eval()
        x = x.unsqueeze(0)
        with torch.no_grad():
            e = rnet_model.get_embedding(x.float())
        rnet_values = self.compare_embeddings(e, rnet_model)
        argmax = torch.argmax(rnet_values).item()
        return e, - (rnet_values[argmax] - self.cfg.thresh).item(), argmax

    def compute_rnet_values(self, rnet_model):
        #rnet_model.eval()
        assert not rnet_model.training
        rnet_values = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            rnet_values[i] = self.compare_embeddings(self.embs[i],
                    rnet_model).cpu()
        return rnet_values

    def get_NN(self, rnet_model, e, mask=None):
        rnet_values = self.compare_embeddings(e, rnet_model)
        if not mask is None:
            rnet_values[mask] = - np.inf
        argmax = torch.argmax(rnet_values).item()
        return argmax, rnet_values[argmax].item()

    def compute_pairwise_dist(self, env, rnet_model):
        return - self.compute_rnet_values(rnet_model)

    def add_first_obs(self, rnet_model, obs, state):
        with torch.no_grad():
            emb = rnet_model.get_embedding(obs.unsqueeze(0).float())
        _ = self.add(obs, state, emb)

    def add(self, obs, state, emb=None):
        i = super().add(obs, state)
        if not emb is None:
            self.embs[i] = emb
        return i

class RNet:
    def __init__(self, cfg, space_info, device, exploration_buffer=None):
        self.cfg = cfg
        self.space_info = space_info
        self.device = device

        if not exploration_buffer is None:
            self.buffer = RNetBuffer(exploration_buffer, cfg.buffer)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.memory = RNetMemory(cfg.memory, self.space_info, cfg.feat_size,
                self.device)
        self.model = RNetModel(cfg.model, self.space_info, cfg.feat_size)
        self.model = self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay)
        print(self.model)

    def update(self, env, num_pairs, num_epochs):
        rnet_stats = self.update_model(num_pairs=num_pairs,
                num_epochs=num_epochs)
        self.update_memory(env)
        env.set_goals(self.memory.get_goals())
        if self.cfg.env.rewards.graph:
            self.compute_graph_dist(env)
        return rnet_stats

    def compute_graph_dist(self, env):
        return self.memory.compute_graph_dist(env, self.model)

    def process_state(self, state):
        state = {k: np.stack([s[k] for s in state]) for k in state[0]}
        obs = torch.from_numpy(state[self.space_info['key']['obs']]).to(self.device).float()
        goal = torch.from_numpy(state[self.space_info['key']['goal']]).to(self.device).float()
        return obs, goal

    def eval_state(self, state):
        obs, goal = self.process_state(state)
        return self.compute_batch_reward(obs, goal)

    def compute_batch_reward(self, obs, goal):
        assert not self.model.training
        if self.cfg.env.rewards.graph:
            out = torch.zeros(obs.size(0))
            with torch.no_grad():
                emb_obs = self.model.get_embedding(obs)
                emb_goal = self.model.get_embedding(goal)
            to_recompute = []
            nearest_inds = []
            for i in range(obs.size(0)):
                i_obs = self.memory.get_NN(self.model, emb_obs[i])[0]
                nearest_inds.append(i_obs)
                i_goal = self.memory.get_NN(self.model, emb_goal[i])[0]
                out[i] = - self.memory.graph_dist[i_obs, i_goal]
                if out [i] == - np.inf:
                    to_recompute.append(i)
            with torch.no_grad():
                out[to_recompute] = self.model(obs[to_recompute],
                        goal[to_recompute], batchwise=True).cpu()[:, 0]

            if self.cfg.env.rewards.graph_mode == "edge_goal":
                with torch.no_grad():
                    out += torch.sigmoid(self.model.compare_embeddings(emb_obs, emb_goal, batchwise=True)[:, 0]).cpu() - 1
            elif self.cfg.env.rewards.graph_mode == "edge_nearest":
                nearest_inds = torch.tensor(nearest_inds, device=emb_obs.device)
                nearest_embs = self.memory.embs.index_select(0, nearest_inds)
                out += torch.sigmoid(self.model.compare_embeddings(emb_obs, nearest_embs, batchwise=True)[:, 0]).cpu() - 1
        else:
            with torch.no_grad():
                out = self.model(obs, goal, batchwise=True)[:, 0]
        return out

    def compute_reward(self, state, next_state):
        reward = self.eval_state(next_state)
        if self.cfg.env.rewards.use_diff:
            reward -= self.eval_state(state)
        if self.cfg.env.rewards.sigmoid:
            reward = torch.sigmoid(reward)
        return reward

    def get_kNN(self, x, all_obs, k=5):
        x = x.unsqueeze(0)
        with torch.no_grad():
            e = self.model.get_embedding(x.float())
            all_emb = self.model.get_embedding(all_obs)
            e_batch = e.repeat(all_emb.size(0), 1)
            rnet_values = self.model.compare_embeddings(e_batch, all_emb,
                    batchwise=True)[:, 0]
        return torch.topk(rnet_values, k=k, sorted=True)

    def update_memory(self, env, num_trajs=-1, traj_len=-1):
        self.model.eval()
        self.memory.adj_matrix = np.pad(self.memory.adj_matrix, (0,
            self.cfg.memory.capacity - len(self.memory)), 'constant')
        if len(self.memory) == 0:
            x = torch.from_numpy(self.buffer.get_obs(0, 0)).to(self.device)
            self.memory.add_first_obs(self.model, x, self.buffer.get_state(0, 0))
        elif not self.cfg.memory.oracle:
            self.memory.embed_memory(self.model)
        num_trajs = self.buffer.get_size() if num_trajs == -1 else num_trajs
        traj_len = self.buffer.traj_len if traj_len == -1 else traj_len
        for traj_idx in tqdm(range(num_trajs), desc="Updating Memory"):
            traj = self.buffer.get_traj(traj_idx)
            prev_nearest = -1
            for i in range(0, traj_len, self.cfg.buffer.skip):
                if self.cfg.memory.oracle:
                    assert not self.cfg.env.rewards.graph
                    d = np.apply_along_axis(env.oracle_distance, 1,
                            self.memory.states, traj['state'][i])
                    if (d > env.oracle_thresh).all():
                        x = torch.from_numpy(traj['obs'][i]).to(self.device)
                        _ = self.memory.add(x, traj['state'][i], emb=None)
                else:
                    x = torch.from_numpy(traj['obs'][i]).to(self.device)
                    e, novelty, nearest = self.memory.compute_novelty(self.model, x)
                    if novelty > 0:
                        if not self.cfg.memory.oracle_check:
                            nearest = self.memory.add(x, traj['state'][i], e)
                            print("node added from traj", traj_idx)
                        else:
                            d = np.apply_along_axis(env.oracle_distance, 1,
                                    self.memory.states, traj['state'][i])
                            if (d > env.oracle_thresh).all():
                                nearest = self.memory.add(x, traj['state'][i], e)
                    if (self.cfg.memory.graph.from_buffer and not prev_nearest
                            == -1):
                        self.memory.adj_matrix[nearest, prev_nearest] = True
                        self.memory.adj_matrix[prev_nearest, nearest] = True
                    prev_nearest = nearest
        if self.cfg.memory.graph.from_buffer:
            self.memory.adj_matrix = self.memory.adj_matrix[:len(self.memory),
                    :len(self.memory)]

    def update_model(self, num_pairs, num_epochs):
        self.model.train()
        self.buffer.set_num_pairs(num_pairs)
        dataloader = torch.utils.data.DataLoader(self.buffer,
                batch_size=self.cfg.train.batch_size, num_workers=32)
        for epoch in range(num_epochs):
            stats = {'rnet_loss': 0.0, 'rnet_acc': 0.0}
            for data in dataloader:
                obs1, obs2, labels = data
                obs1 = obs1.to(self.device)
                obs2 = obs2.to(self.device)
                labels = labels.to(self.device).float()

                self.optim.zero_grad()
                outputs = self.model(obs1.float(), obs2.float(), batchwise=True)[:, 0]
                preds = outputs > 0
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()

                stats['rnet_loss'] += loss.item() * labels.shape[0]
                stats['rnet_acc'] += torch.sum(preds == labels.data)

            for k in stats:
                stats[k] /= num_pairs
            print("rnet epoch {} - loss {:.2f} - acc {:.2f}".format(epoch,
                stats['rnet_loss'], stats['rnet_acc']))
        self.model.eval()
        stats['rnet_updates'] = 1
        return stats

    def get_states_array(self):
        return self.buffer.get_states_array()

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, f'model.pth')
        print('Saving rnet model to {}'.format(model_path))
        self.model.save(model_path)

        memory_path = os.path.join(save_dir, f'memory.npy')
        print('Saving rnet memory to {}'.format(memory_path))
        self.memory.save(memory_path)

    def load(self, save_dir):
        model_path = os.path.join(save_dir, f'model.pth')
        print('Loading rnet model from {}'.format(model_path))
        self.model.load(model_path)

        memory_path = os.path.join(save_dir, f'memory.npy')
        print('Loading rnet memory from {}'.format(memory_path))
        self.memory.load(memory_path)
