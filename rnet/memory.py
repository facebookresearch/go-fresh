import numpy as np
import torch
import scipy.sparse.csgraph as csg

import utils

from memory import GraphMemory


class RNetMemory(GraphMemory):
    def __init__(self, cfg, space_info, feat_size, device):
        super().__init__(cfg, space_info)
        self.device = device
        obs_dtype = utils.TORCH_DTYPE[space_info["type"]["obs"]]
        self.obss = torch.zeros(
            (cfg.capacity, *space_info["shape"]["obs"]), dtype=obs_dtype
        ).to(device)
        self.embs = torch.zeros((cfg.capacity, feat_size)).to(device)

    def add(self, obs, state, emb):
        i = super().add(obs, state)
        self.embs[i] = emb
        return i

    def add_first_obs(self, rnet_model, obs, state):
        assert not rnet_model.training
        with torch.no_grad():
            emb = rnet_model.get_embedding(obs.unsqueeze(0).float())
        _ = self.add(obs, state, emb)

    def embed_memory(self, rnet_model):
        assert not rnet_model.training
        with torch.no_grad():
            self.embs[: len(self)] = rnet_model.get_embedding(
                self.obss[: len(self)].float()
            )

    def compare_embeddings(self, emb, rnet_model, both_dir=False):
        assert not rnet_model.training
        emb_batch = emb.expand(len(self), -1)
        with torch.no_grad():
            if both_dir:
                left_batch = torch.cat((emb_batch, self.embs[: len(self)]))
                right_batch = torch.cat((self.embs[: len(self)], emb_batch))
                return rnet_model.compare_embeddings(
                    left_batch, right_batch, batchwise=True
                )[:, 0]
            else:
                return rnet_model.compare_embeddings(
                    emb_batch, self.embs[: len(self)], batchwise=True
                )[:, 0]

    def compute_novelty(self, rnet_model, e):
        assert not rnet_model.training
        rnet_values = self.compare_embeddings(e, rnet_model, both_dir=self.cfg.directed)
        if self.cfg.directed:
            # both directions need to be novel
            NNi = torch.argmax(rnet_values[: len(self)]).item()
            NNo = torch.argmax(rnet_values[len(self):]).item()
            nov = -(
                max(rnet_values[NNo + len(self)], rnet_values[NNi]) - self.cfg.thresh
            ).item()
            return nov, NNi, NNo
        else:
            NN = torch.argmax(rnet_values).item()
            return - (rnet_values[NN] - self.cfg.thresh).item(), NN, NN

    def compute_rnet_values(self, rnet_model):
        rnet_values = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            rnet_values[i] = self.compare_embeddings(self.embs[i], rnet_model).cpu()
        return rnet_values

    def get_NN(self, rnet_model, e, mask=None):
        rnet_values = self.compare_embeddings(e, rnet_model, both_dir=self.cfg.directed)
        if mask is not None:
            rnet_values[mask] = -np.inf
        NN = torch.argmax(rnet_values).item() % len(self)
        return NN, rnet_values[NN].item()

    def get_batch_NN(self, rnet_model, e):
        batch_size = e.size(0)
        batch_e = torch.repeat_interleave(e, len(self), dim=0)
        memory_batch = self.embs[: len(self)].repeat(batch_size, 1)
        if self.cfg.directed:
            left_batch = torch.cat((batch_e, memory_batch))
            right_batch = torch.cat((memory_batch, batch_e))
            with torch.no_grad():
                rnet_vals = rnet_model.compare_embeddings(
                    left_batch, right_batch, batchwise=True
                )[:, 0]
            NN = torch.zeros(batch_size).long()
            for i in range(batch_size):
                intrvl = np.arange(i * len(self), (i + 1) * len(self))
                idx = np.concatenate((intrvl, batch_size * len(self) + intrvl))
                NN[i] = torch.argmax(rnet_vals[idx]).item() % len(self)
            return NN, rnet_vals[NN]
        else:
            with torch.no_grad():
                rnet_vals = rnet_model.compare_embeddings(
                    batch_e, memory_batch, batchwise=True
                )[:, 0]
            NN = [torch.argmax(rnet_vals[
                    i * len(self): (i + 1) * len(self)
                ]).item() for i in range(batch_size)]
            return torch.tensor(NN), rnet_vals[NN]

    def connect_graph(self, rnet_model):
        while True:
            n_components, labels = csg.connected_components(
                self.adj_matrix, directed=self.cfg.directed, return_labels=True
            )
            print(f"number of connected components: {n_components}")
            if n_components == 1:
                break
            components_count = np.bincount(labels)
            component_to_drop = np.argmin(components_count)
            component_idx = np.where(labels == component_to_drop)[0]
            max_value = -np.inf
            for i in range(len(component_idx)):
                argmax, value = self.get_NN(
                    rnet_model, self.embs[component_idx[i]], mask=component_idx
                )
                if value > max_value:
                    max_value = value
                    edge_to_add = (component_idx[i], argmax)
            self.add_edge(*edge_to_add)

    def dict_to_save(self):
        to_save = super().dict_to_save()
        to_save["embs"] = self.to_numpy(self.embs[: len(self)])
        return to_save

    def load(self, path):
        memory = super().load(path)
        self.embs = torch.from_numpy(memory.get("embs"))
        return memory
