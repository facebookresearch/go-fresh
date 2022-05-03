import torch
import random
import numpy as np
import scipy.sparse.csgraph as csg

class Memory:
    def __init__(self, cfg, space_info):
        self.cfg = cfg
        self.space_info = space_info
        self.size = 0
        self.obss = np.zeros((cfg.capacity, *space_info['shape']['obs']),
                dtype=space_info['type']['obs'])
        self.states = np.zeros((cfg.capacity, *space_info['shape']['state']))

    def __len__(self):
        return self.size

    def add(self, obs, state):
        if len(self) < self.cfg.capacity:
            i = len(self)
            self.size += 1
        else:
            i = random.randint(0, len(self) - 1)
        self.obss[i] = obs
        self.states[i] = state
        return i

    def flush(self):
        self.size = 0

    def to_numpy(self, arr):
        if torch.is_tensor(arr):
            return arr.cpu().numpy()
        return arr

    def from_numpy(self, arr, dest):
        if torch.is_tensor(dest):
            return torch.from_numpy(arr)
        return arr

    def get_goals(self):
        return {
                'state': self.states[:len(self)],
                f'{self.space_info["obs_type"]}_obs': self.to_numpy(self.obss[:len(self)]),
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

    def dict_to_save(self):
        return {
            'obss': self.to_numpy(self.obss[:len(self)]),
            'states': self.states[:len(self)],
        }

    def save(self, path):
        to_save = self.dict_to_save()
        np.save(path, to_save)

    def load(self, path):
        memory = np.load(path, allow_pickle=True).tolist()
        self.size = memory['obss'].shape[0]
        self.obss[:len(self)] = self.from_numpy(memory['obss'],
                self.obss[:len(self)])
        self.states[:len(self)] = memory['states']
        return memory

class GraphMemory(Memory):
    def __init__(self, cfg, space_info):
        super().__init__(cfg, space_info)
        self.adj_matrix = np.zeros((cfg.capacity, cfg.capacity), dtype=bool)

    def add(self, obs, state):
        i = super().add(obs, state)
        self.adj_matrix[i, :] = False
        self.adj_matrix[:, i] = False
        return i

    def add_edge(self, prev_NNi, prev_NNo, NNi, NNo):
        if not self.cfg.directed:
            assert prev_NNi == prev_NNo
            assert NNi == NNo
            self.add_single_edge(prev_NNi, NNi)
            self.add_single_edge(NNi, prev_NNi)
            return
        self.add_single_edge(prev_NNi, prev_NNo)
        self.add_single_edge(NNi, NNo)
        self.add_single_edge(prev_NNi, NNo)

    def add_single_edge(self, i, j):
        if not i == -1 and not j == -1 and not i == j:
            self.adj_matrix[i, j] = True

    def flush(self):
        super().flush()
        self.adj_matrix = np.zeros((cfg.capacity, cfg.capacity), dtype=bool)

    def compute_dist(self):
        self.dist, self.pred = csg.floyd_warshall(self.adj_matrix,
                return_predecessors=True, directed=self.cfg.directed)

    def dict_to_save(self):
        to_save = super().dict_to_save()
        to_save.update({
            'adj_matrix': self.adj_matrix,
            'dist': self.dist,
            'pred': self.pred,
        })
        return to_save

    def load(self, path):
        memory = super().load(path)
        self.adj_matrix = memory.get('adj_matrix')
        self.dist = memory.get('dist')
        self.pred = memory.get('pred')
        if self.dist is None:
            self.compute_dist()
        return memory
