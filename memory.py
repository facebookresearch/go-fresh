import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.csgraph as csg


class Memory:
    def __init__(self, cfg, space_info):
        self.cfg = cfg
        self.space_info = space_info
        self.size = 0
        self.obss = np.zeros(
            (cfg.capacity, *space_info["shape"]["obs"]), dtype=space_info["type"]["obs"]
        )
        self.states = np.zeros((cfg.capacity, *space_info["shape"]["state"]))

    def __len__(self):
        return self.size

    def add(self, obs, state):
        if len(self) < self.cfg.capacity:
            i = len(self)
            self.size += 1
        else:
            i = np.random.randint(0, len(self) - 1)
        self.obss[i] = obs
        self.states[i] = state
        return i

    def get_obs(self, i):
        return self.obss[i]

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
            "state": self.states[: len(self)],
            f'{self.space_info["obs_type"]}_obs': self.to_numpy(self.obss[: len(self)]),
        }

    def vis_sample(self, n=None, num_cols=10):
        assert self.space_info["obs_type"] == "rgb"
        if not n:
            n = len(self)
        assert n <= len(self)
        num_rows = n // num_cols + int(n % num_cols != 0)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
        for i in range(n):
            img = self.to_numpy(self.obss[i]).transpose((1, 2, 0))
            ax[i // num_cols, i % num_cols].imshow(img)
            ax[i // num_cols, i % num_cols].axis("off")
            ax[i // num_cols, i % num_cols].set_title(str(i))
        return fig

    def dict_to_save(self):
        return {
            "obss": self.to_numpy(self.obss[: len(self)]),
            "states": self.states[: len(self)],
        }

    def save(self, path):
        to_save = self.dict_to_save()
        np.save(path, to_save)

    def load(self, path):
        memory = np.load(path, allow_pickle=True).tolist()
        self.size = memory["obss"].shape[0]
        self.obss[: len(self)] = self.from_numpy(memory["obss"], self.obss[: len(self)])
        self.states[: len(self)] = memory["states"]
        return memory


class GraphMemory(Memory):
    def __init__(self, cfg, space_info):
        super().__init__(cfg, space_info)

    def init_adj_matrix(self, N=None):
        if N is None:
            N = self.cfg.capacity
        self.adj_matrix = np.zeros((N, N), dtype=bool)

    def transition2edges(self, prev_NNi, prev_NNo, NNi, NNo):
        edges_to_add = []
        if not self.cfg.directed:
            assert prev_NNi == prev_NNo
            assert NNi == NNo
            edges_to_add = [(prev_NNi, NNi), (NNi, prev_NNi)]
        else:
            edges_to_add = [(prev_NNi, prev_NNo), (NNi, NNo), (prev_NNi, NNo)]
        for edge in edges_to_add:
            self.add_edge(*edge)
        return edges_to_add

    def add_edge(self, i, j):
        if not i == -1 and not j == -1 and not i == j:
            self.adj_matrix[i, j] = True
            if not self.cfg.directed:
                self.adj_matrix[j, i] = True

    def flush(self):
        super().flush()
        self.adj_matrix = np.zeros((self.cfg.capacity, self.cfg.capacity), dtype=bool)

    def retrieve_path(self, start, end):
        if self.pred[start, end] == -9999:
            return [start]
        path = self.retrieve_path(start, self.pred[start, end])
        path.append(end)
        return path

    def get_nb_connected_components(self, return_labels=False):
        return csg.connected_components(
            self.adj_matrix,
            directed=self.cfg.directed,
            connection="strong",
            return_labels=return_labels,
        )

    def compute_dist(self):
        self.dist, self.pred = csg.floyd_warshall(
            self.adj_matrix, return_predecessors=True, directed=self.cfg.directed
        )

    def dict_to_save(self):
        to_save = super().dict_to_save()
        to_save.update(
            {"adj_matrix": self.adj_matrix, "dist": self.dist, "pred": self.pred}
        )
        return to_save

    def load(self, path):
        memory = super().load(path)
        self.adj_matrix = memory.get("adj_matrix")
        self.dist = memory.get("dist")
        self.pred = memory.get("pred")
        if self.dist is None:
            self.compute_dist()
        return memory
