import os
import random
import torch
import numpy as np
import scipy.special as ssp
import scipy.sparse.csgraph as csg

from tqdm import tqdm

class Memory:
    def __init__(self, cfg, space_info):
        self.cfg = cfg
        self.space_info = space_info
        self.size = 0
        self.obss = np.zeros((cfg.capacity, *space_info['shape']['obs']),
                dtype=space_info['type']['obs'])
        self.states = np.zeros((cfg.capacity, *space_info['shape']['state']))
        self.epoch_added = np.zeros(cfg.capacity, dtype=int)
        self.epoch = 0
        self.adj_matrix = np.zeros((cfg.capacity, cfg.capacity), dtype=bool)
        self.graph_dist = None
        self.graph_pred = None

    def __len__(self):
        return self.size

    def add(self, obs, state):
        if len(self) < self.cfg.capacity:
            i = len(self)
            self.size += 1
        else:
            i = random.randint(0, len(self) - 1)
            if self.cfg.graph.from_buffer:
                self.adj_matrix[i, :] = False
                self.adj_matrix[:, i] = False

        self.obss[i] = obs
        self.states[i] = state
        self.epoch_added[i] = self.epoch
        return i

    def flush(self):
        self.size = 0
        self.adj_matrix = np.zeros((cfg.capacity, cfg.capacity), dtype=bool)

    def to_numpy(self, arr):
        if torch.is_tensor(arr):
            return arr.cpu().numpy()
        return arr

    def from_numpy(self, arr, dest):
        if torch.is_tensor(dest):
            return torch.from_numpy(arr)
        return arr

    def compute_graph_dist(self, env, rnet_model):
        if not self.cfg.graph.from_buffer:
            pairwise_dist = self.compute_pairwise_dist(env, rnet_model)
            pairwise_dist = (pairwise_dist + pairwise_dist.transpose()) / 2

            self.adj_matrix = np.zeros_like(pairwise_dist, dtype=bool)
            self.adj_matrix[pairwise_dist < self.cfg.graph.thresh] = True

        #ensure graph is connected
        while True:
            n_components, labels = csg.connected_components(self.adj_matrix,
                    directed=False, return_labels=True)
            if n_components == 1:
                break
            components_count = np.bincount(labels)
            component_to_drop = np.argmin(components_count)
            component_idx = np.where(labels == component_to_drop)[0]
            max_value = - np.inf
            for i in range(len(component_idx)):
                argmax, value = self.get_NN(rnet_model,
                        self.embs[component_idx[i]], mask=component_idx)
                if value > max_value:
                    max_value = value
                    edge_to_add = (component_idx[i], argmax)
            self.adj_matrix[edge_to_add] = True
            self.adj_matrix[edge_to_add[::-1]] = True

        self.graph_dist, self.graph_pred = csg.floyd_warshall(self.adj_matrix,
                return_predecessors=True)

    def get_goals(self):
        return {
                'state': self.states[:len(self)],
                'image': self.to_numpy(self.obss[:len(self)]),
        }

    def vis_sample(self, n=None, num_cols=10):
        assert self.space_info['obs_type'] == 'rgb'
        if not n:
            n = len(self)
        assert n <= len(self)
        num_rows = n // num_cols + int(n % num_cols != 0)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 *
            num_rows))
        for i in range(n):
            img = self.to_numpy(self.obss[i]).transpose((1, 2, 0))
            ax[i // num_cols, i % num_cols].imshow(img)
            ax[i // num_cols, i % num_cols].axis('off')
            ax[i // num_cols, i % num_cols].set_title(str(i))
        return fig

    def save(self, path, embs=False):
        to_save = {
            'obss': self.to_numpy(self.obss[:len(self)]),
            'states': self.states[:len(self)],
            'adj_matrix': self.adj_matrix,
            'graph_dist': self.graph_dist,
            'graph_pred': self.graph_pred,
        }
        if embs:
            to_save['embs'] = self.to_numpy(self.embs[:len(self)])
        np.save(path, to_save)

    def load(self, path):
        memory = np.load(path, allow_pickle=True).tolist()
        self.size = memory['obss'].shape[0]
        self.obss[:len(self)] = self.from_numpy(memory['obss'],
                self.obss[:len(self)])
        self.states[:len(self)] = memory['states']
        self.adj_matrix = memory.get('adj_matrix')
        self.graph_dist = memory.get('graph_dist')
        self.graph_pred = memory.get('graph_pred')
        if 'embs' in memory:
            self.embs = torch.from_numpy(memory.get('embs'))

class OracleMemory(Memory):
    def update(self, traj_buffer, env):
        if self.cfg.reset:
            self.flush()
        for i in tqdm(range(len(traj_buffer)), desc='update oracle memory'):
            for j in range(traj_buffer.cfg.traj_len):
                if len(self) == 0:
                    self.add(traj_buffer.obss[i, j], traj_buffer.states[i, j])
                    continue
                d = np.apply_along_axis(env.oracle_distance, 1,
                        self.states[:len(self)], traj_buffer.states[i, j])
                if (d > env.oracle_thresh).all():
                    self.add(traj_buffer.obss[i, j], traj_buffer.states[i, j])

    def compute_pairwise_dist(self, env, rnet_model):
        dist = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            dist[i] = np.apply_along_axis(env.oracle_distance, 1,
                    self.states[:len(self)], self.states[i])
        return dist

class OracleCurri:
    def __init__(self, exploration_buffer, cfg, space_info):
        self.cfg = cfg
        self.buffer = exploration_buffer
        self.memory = OracleMemory(cfg.memory, space_info,
                cfg.env.goal_sampling)

    def get_states_array(self):
        return self.buffer.get_states_array()

    def update(self, env, num_pairs=None, num_epochs=None):
        self.update_memory(env)
        env.set_goals(self.memory.get_goals())
        self.flush_buffer()
        if self.cfg.env.rewards.graph:
            self.compute_graph_dist(env)
        return {}

    def compute_graph_dist(self, env):
        return self.memory.compute_graph_dist(env, rnet_model=None)

    def update_memory(self, env):
        self.memory.update(self.buffer, env)

    def save(self, logs_dir, total_numsteps):
        save_dir = os.path.join(logs_dir, 'oracle')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #buffer_path = os.path.join(save_dir, f'buffer_{total_numsteps}.npy')
        #print('Saving oracle buffer to {}'.format(buffer_path))
        #self.buffer.save(buffer_path)

        memory_path = os.path.join(save_dir, f'memory_{total_numsteps}.npy')
        print('Saving oracle memory to {}'.format(memory_path))
        self.memory.save(memory_path)

    def load(self, logs_dir, total_numsteps):
        save_dir = os.path.join(logs_dir, 'oracle')

        #buffer_path = os.path.join(save_dir, f'buffer_{total_numsteps}.npy')
        #print('Loading oracle buffer from {}'.format(buffer_path))
        #self.buffer.load(buffer_path)

        memory_path = os.path.join(save_dir, f'memory_{total_numsteps}.npy')
        print('Loading oracle memory from {}'.format(memory_path))
        self.memory.load(memory_path)
