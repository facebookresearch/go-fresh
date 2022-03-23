import torch
import scipy.sparse.csgraph as csg

import utils

from memory import GraphMemory 

class RNetMemory(GraphMemory):
    def __init__(self, cfg, space_info, feat_size, device):
        super().__init__(cfg, space_info)
        self.cfg = cfg
        self.device = device
        obs_dtype = utils.TORCH_DTYPE[space_info['type']['obs']]
        self.obss = torch.zeros((cfg.capacity, *space_info['shape']['obs']),
                dtype=obs_dtype).to(device)
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
            self.embs[:len(self)] = rnet_model.get_embedding(
                self.obss[:len(self)].float())

    def compare_embeddings(self, emb, rnet_model):
        assert not rnet_model.training
        emb_batch = emb.repeat(len(self), 1)
        with torch.no_grad():
            return rnet_model.compare_embeddings(emb_batch,
                    self.embs[:len(self)], batchwise=True)[:, 0]

    def compute_novelty(self, rnet_model, x):
        assert not rnet_model.training
        x = x.unsqueeze(0)
        with torch.no_grad():
            e = rnet_model.get_embedding(x.float())
        rnet_values = self.compare_embeddings(e, rnet_model)
        argmax = torch.argmax(rnet_values).item()
        return e, - (rnet_values[argmax] - self.cfg.thresh).item(), argmax

    def compute_rnet_values(self, rnet_model):
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

    def connect_graph(self, rnet_model):
        while True:
            n_components, labels = csg.connected_components(self.adj_matrix,
                    directed=False, return_labels=True)
            print(f"number of connected components: {n_components}")
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
            self.add_edge(*edge_to_add)

    def dict_to_save(self):
        to_save = super().dict_to_save()
        to_save['embs'] = self.to_numpy(self.embs[:len(self)])
        return to_save

    def load(self, path):
        memory = super().load(path)
        self.embs = torch.from_numpy(memory.get('embs'))
        return memory
