import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.decomposition import PCA

class FeatureEncoder(nn.Module):
    def __init__(self, cfg, space_info):
        super(FeatureEncoder, self).__init__()
        self.cfg = cfg
        self.obs_type = space_info['obs_type']
        obs_shape = space_info['shape']['obs']

        if space_info['obs_type'] == 'vec':
            obs_size = obs_shape[0]
            if self.cfg.remove_velocity:
                obs_size = 3
            modules = [nn.Linear(obs_size, cfg.hidden_size),
                    nn.BatchNorm1d(num_features=cfg.hidden_size), nn.Tanh()]
            for _ in range(1, cfg.n_layers - 1):
                modules.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
                modules.append(nn.BatchNorm1d(num_features=cfg.hidden_size))
                modules.append(nn.Tanh())
            modules.append(nn.Linear(cfg.hidden_size, cfg.feat_size))

        elif space_info['obs_type'] == 'rgb':
            n, m = obs_shape[1], obs_shape[2]
            d = lambda x: (((x - 1)//2 - 1)//2 - 1)//2
            conv_outdim = d(m)*d(n)*32
            modules = [
                    nn.Conv2d(3, 8, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(8, 16, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Flatten(),
                    nn.Linear(conv_outdim, cfg.feat_size),
            ]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        if self.obs_type == "rgb":
            x = x / 255.
        if self.cfg.remove_velocity:
            x = x[:, :3]
        return self.net(x)

class RNetModel(nn.Module):
    def __init__(self, cfg, space_info):
        super(RNetModel, self).__init__()

        self.feat_size = cfg.feat_size
        self.encoder = FeatureEncoder(cfg, space_info)
        self.comparator_type = cfg.comparator
        self.bias = nn.Parameter(torch.zeros(1))
        if self.comparator_type == "net":
            self.comparator = nn.Sequential(
                nn.Linear(2 * cfg.feat_size, cfg.feat_size),
                nn.BatchNorm1d(num_features=cfg.feat_size),
                nn.ReLU(),
                nn.Linear(cfg.feat_size, 2),
            )
        elif self.comparator_type == "net_sym":
            self.lin = nn.Linear(cfg.feat_size, cfg.feat_size)
            self.comparator = nn.Sequential(
                nn.Linear(cfg.feat_size, cfg.feat_size),
                nn.BatchNorm1d(num_features=cfg.feat_size),
                nn.ReLU(),
                nn.Linear(cfg.feat_size, 2),
            )
        elif self.comparator_type == "dot-W":
            self.W = nn.Parameter(torch.eye(cfg.feat_size))

    def forward(self, x1, x2=None, batchwise=False):
        e1 = self.encoder(x1)
        if x2 is None:
            return self.compare_embeddings(e1, e1, equal=True,
                    batchwise=batchwise)
        e2 = self.encoder(x2)
        return self.compare_embeddings(e1, e2, batchwise=batchwise)

    def get_embedding(self, x):
        return self.encoder(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def compare_embeddings(self, e1, e2, equal=False, batchwise=False):
        if self.comparator_type == 'dot':
            if batchwise:
                logits = torch.bmm(e1.unsqueeze(1),
                        e2.unsqueeze(2))[:, :, 0]
            else:
                logits = torch.matmul(e1, e2.T)
            return logits - self.bias

        elif self.comparator_type == 'cosine':
            if batchwise:
                return (F.cosine_similarity(e1, e2)).unsqueeze(1) - self.bias
            else:
                raise NotImplementedError

        elif self.comparator_type == 'dot-W':
            proj2 = torch.matmul(self.W, e2.T)
            if batchwise:
                logits = torch.bmm(e1.unsqueeze(1),
                        proj2.T.unsqueeze(2))[:, :, 0]
            else:
                    logits = torch.matmul(e1, proj2)
            return logits - self.bias

        elif self.comparator_type == 'L2':
            if batchwise:
                raise NotImplementedError
            if equal:
                e2 = None
            return self.pairwise_distances(e1, e2)

        elif self.comparator_type == 'net':
            if batchwise:
                x = torch.cat((e1, e2), 1)
                return self.comparator(x) - self.bias
            else:
                raise NotImplementedError

        elif self.comparator_type == 'net_sym':
            if batchwise:
                x1 = self.lin(e1)
                x2 = self.lin(e2)
                x = torch.abs(x1 - x2)
                return self.comparator(x) - self.bias
            else:
                raise NotImplementedError

    def pca_embeddings(self, obss):
        self.eval()
        with torch.no_grad():
            emb = self.get_embedding(torch.from_numpy(obss).float())
        pca = PCA(n_components=2)
        pca.fit(emb)
        return pca.transform(emb)

    def compute_values(self, comp_obs, obss):
        self.eval()
        with torch.no_grad():
            comp_obs = torch.from_numpy(comp_obs).float()
            obss = torch.from_numpy(obss).float()
            rnet_values = self(comp_obs.repeat(obss.size(0), 1), obss,
                    batchwise=True)[:, 0]
        return rnet_values.numpy()
