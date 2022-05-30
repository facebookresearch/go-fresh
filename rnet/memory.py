import numpy as np
import torch
import tqdm

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
        self.nn_out = None
        self.nn_in = None
        self.edge2rb = None

    def add(self, obs, state, emb):
        i = super().add(obs, state)
        self.embs[i] = emb

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

    def compare_embeddings(self, emb, rnet_model, incoming_dir=False):
        assert not rnet_model.training
        emb_batch = emb.expand(len(self), -1)
        with torch.no_grad():
            return rnet_model.compare_embeddings(
                emb_batch,
                self.embs[: len(self)],
                batchwise=True,
                reverse_dir=incoming_dir
            )[:, 0]

    def compute_novelty(self, rnet_model, e, incoming_dir=False):
        assert not rnet_model.training
        rnet_values = self.compare_embeddings(e, rnet_model, incoming_dir=incoming_dir)
        rnet_max, _ = torch.max(rnet_values, dim=0)
        nov = -(rnet_max.item() - self.cfg.thresh)
        return nov

    def compute_rnet_values(self, rnet_model):
        rnet_values = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            rnet_values[i] = self.compare_embeddings(self.embs[i], rnet_model).cpu()
        return rnet_values

    def get_NN(self, rnet_model, e, mask=None, incoming_dir=False):
        rnet_values = self.compare_embeddings(e, rnet_model, incoming_dir=incoming_dir)
        if mask is not None:
            rnet_values[mask] = -np.inf
        rnet_max, NN = torch.max(rnet_values, dim=0)
        return NN.item(), rnet_max.item()

    def get_batch_NN(self, rnet_model, e, incoming_dir=False):
        bsz = e.size(0)
        memsz = len(self)
        batch_e = torch.repeat_interleave(e, memsz, dim=0)
        memory_batch = self.embs[:memsz].repeat(bsz, 1)
        with torch.no_grad():
            rnet_vals = rnet_model.compare_embeddings(
                batch_e, memory_batch, batchwise=True, reverse_dir=incoming_dir
            )[:, 0]
        return rnet_vals.view(bsz, memsz).max(dim=1)[1].cpu()

    def build(self, model, expl_buffer):
        assert not model.training
        expl_buffer.embs = expl_buffer.embs.to(self.device)
        x = torch.from_numpy(expl_buffer.get_obs(0, 0)).to(self.device)
        self.add_first_obs(model, x, expl_buffer.get_state(0, 0))

        obs_count = 0
        for traj_idx in tqdm.tqdm(range(len(expl_buffer)), desc="Updating Memory"):
            traj = expl_buffer.get_traj(traj_idx)
            for i in range(expl_buffer.traj_len):
                if np.random.random() > self.cfg.node_skip:
                    continue
                obs_count += 1
                e = expl_buffer.embs[traj_idx, i].unsqueeze(0)
                novelty_o = self.compute_novelty(model, e)
                if self.cfg.directed:
                    novelty_i = self.compute_novelty(model, e, incoming_dir=True)
                else:
                    novelty_i = novelty_o

                if novelty_o > 0 and novelty_i > 0:
                    x = torch.from_numpy(traj["obs"][i]).to(self.device)
                    _ = self.add(x, traj["state"][i], e)
        print("num obs seen:", obs_count)
        print("memory_len :", len(self))

    def compute_NN(self, embs, model):
        num_trajs, traj_len = embs.shape[0], embs.shape[1]
        self.embs = self.embs.to(self.device)
        nn = {"out": np.zeros((num_trajs, traj_len), dtype=int)}
        if self.cfg.directed:
            nn["in"] = np.zeros((num_trajs, traj_len), dtype=int)
        bsz = self.cfg.NN_batch_size
        for traj_idx in tqdm.tqdm(range(num_trajs), desc="computing NN"):
            for i in range(0, traj_len, bsz):
                j = min(i + bsz, traj_len)
                nn["out"][traj_idx, i:j] = self.get_batch_NN(
                    model, embs[traj_idx, i:j]
                )
                if self.cfg.directed:
                    nn["in"][traj_idx, i:j] = self.get_batch_NN(
                        model, embs[traj_idx, i:j], incoming_dir=True
                    )
        return nn

    def set_nn(self, nn):
        self.nn_out = nn["out"]
        if self.cfg.directed:
            self.nn_in = nn["in"]

    def compute_edges(self, model):
        self.edge2rb = {}
        num_trajs, traj_len = self.nn_out.shape
        self.init_adj_matrix(N=len(self))
        for i in tqdm.tqdm(range(num_trajs), desc="computing edges"):
            for j in range(traj_len - 1):
                if np.random.random() > self.cfg.edge_skip:
                    continue
                prev_NNi = (
                    self.nn_in[i, j] if self.cfg.directed else self.nn_out[i, j]
                )
                prev_NNo = self.nn_out[i, j]
                NNi = (
                    self.nn_in[i, j + 1] if self.cfg.directed else (
                        self.nn_out[i, j + 1]
                    )
                )
                NNo = self.nn_out[i, j + 1]
                edges = self.transition2edges(prev_NNi, prev_NNo, NNi, NNo)
                for edge in edges:
                    if edge[0] == edge[1]:
                        continue
                    if edge not in self.edge2rb:
                        self.edge2rb[edge] = []
                    self.edge2rb[edge].append((i, j))

        # make sure any memory node can be reached from any other
        model.eval()
        self.connect_graph(model)
        self.compute_dist()

    def get_component_shortest_edge(
        self, component_idx, rnet_model, mask=None, incoming_dir=False
    ):
        """ given a set nodes, find a node has the nearest edge outside of the set """
        if mask is None:
            mask = component_idx
        max_value = -np.inf
        edge = None
        for i in range(len(component_idx)):
            argmax, value = self.get_NN(
                rnet_model,
                self.embs[component_idx[i]],
                mask=mask,
                incoming_dir=incoming_dir
            )
            if value > max_value:
                max_value = value
                if incoming_dir:
                    edge = (argmax, component_idx[i])
                else:
                    edge = (component_idx[i], argmax)
        return edge, max_value

    def connect_graph(self, rnet_model):
        while True:
            n_components, labels = self.get_nb_connected_components(return_labels=True)
            if n_components == 1:
                break
            components_count = np.bincount(labels)
            component_to_drop = np.argmin(components_count)
            component_idx = np.where(labels == component_to_drop)[0]
            print(
                f"There are {n_components} components. Connecting {component_to_drop}."
            )
            if not self.cfg.directed:
                edge_to_add, _ = self.get_component_shortest_edge(
                    component_idx, rnet_model
                )
                self.add_edge(*edge_to_add)
            else:
                # need to figure out which direction needs to be added
                self.compute_dist()
                # maskout nodes that are already connected in that direction
                mask_o = np.where(self.dist[component_idx[0], :] < np.inf)[0]
                mask_i = np.where(self.dist[:, component_idx[0]] < np.inf)[0]

                edge_o, val_o = self.get_component_shortest_edge(
                    component_idx, rnet_model, mask=mask_o
                )
                edge_i, val_i = self.get_component_shortest_edge(
                    component_idx, rnet_model, mask=mask_i, incoming_dir=True
                )
                if val_o > val_i:
                    self.add_edge(*edge_o)
                else:
                    self.add_edge(*edge_i)

    def dict_to_save(self):
        to_save = super().dict_to_save()
        to_save["embs"] = self.to_numpy(self.embs[:len(self)])
        to_save["nn_out"] = self.nn_out
        to_save["nn_in"] = self.nn_in
        to_save["edge2rb"] = self.edge2rb
        return to_save

    def load(self, path):
        memory = super().load(path)
        self.embs = torch.from_numpy(memory.get("embs"))
        self.nn_out = memory.get("nn_out")
        self.nn_in = memory.get("nn_in")
        self.edge2rb = memory.get("edge2rb")
        return memory
