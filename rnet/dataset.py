import torch
import random

class RNetPairsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, traj_buffer, num_pairs, traj_range):
        self.cfg = cfg
        self.num_pairs = num_pairs
        self.traj_buffer = traj_buffer
        self.traj_len = self.traj_buffer.traj_len
        self.traj_range = traj_range

    def get_break_condition(self, i1, i2, pos, in_traj):
        if pos:
            return True
        else:
            if in_traj:
                return abs(i1 - i2) > self.cfg.thresh
            else:
                return True

    def get_search_interval(self, i1, pos, in_traj):
        assert 2 * self.cfg.thresh < self.traj_len - 1, "negative sample might not exist!"
        low = 0
        high = self.traj_len - 1
        if pos:
            low = max(low, i1 - self.cfg.thresh)
            high = min(i1 + self.cfg.thresh, high)
        elif in_traj and self.cfg.neg_thresh > 0:
            low = max(low, i1 - self.cfg.neg_thresh)
            high = min(i1 + self.cfg.neg_thresh, high)
        return low, high

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, i, enforce_neg=False):
        pos = 0 if enforce_neg else random.randint(0, 1)
        traj1 = self.traj_buffer.sample(range=self.traj_range)['obs']
        i1 = random.randint(0, self.traj_len - 1)

        in_traj = pos or random.random() < self.cfg.in_traj_ratio
        traj2 = traj1 if in_traj else self.traj_buffer.sample(range=self.traj_range)['obs']

        interval = self.get_search_interval(i1, pos, in_traj)
        while True:
            i2 = random.randint(*interval)
            if self.get_break_condition(i1, i2, pos, in_traj):
                break

        if pos and not self.cfg.symmetric:
            imin, imax = min(i1, i2), max(i1, i2)
            return traj1[imin], traj1[imax], pos
        if random.random() > 0.5:
            return traj1[i1], traj2[i2], pos
        return traj2[i2], traj1[i1], pos


class RNetPairsSplitDataset:
    def __init__(self, cfg, traj_buffer):
        N = len(traj_buffer)
        N_val = int(N * cfg.valid_ratio)
        N_train = N - N_val
        train_range = [0, N_train]
        val_range = [N_train, N]
        self.train = RNetPairsDataset(cfg, traj_buffer, cfg.num_pairs.train, train_range)
        self.val = RNetPairsDataset(cfg, traj_buffer, cfg.num_pairs.val, val_range)
