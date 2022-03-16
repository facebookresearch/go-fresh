import os

import numpy as np
import multiprocessing as mp

class ExplorationBuffer(object):
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.load_data(cfg.data_dir)

    def read_x(self, path):
        with np.load(path) as data:
            return data["observation"], data["action"]
    
    def parallel_read(self, files, num_procs):
        with mp.Pool(num_procs) as pool:
            x_list = pool.map(self.read_x, files)
        traj_buffer = {"observation": [], "action": []}
        for x in x_list:
            obs, a = x
            traj_buffer["observation"].append(obs)
            traj_buffer["action"].append(a)
        traj_buffer = {k: np.stack(v) for k, v in traj_buffer.items()}
        return traj_buffer
    
    def load_data(self, data_dir):
        self.log.info("loading exploration buffer")
        ep_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        x_list = self.parallel_read(ep_files, num_procs=self.cfg.num_procs)
        self.obss = x_list["observation"]
        self.actions = x_list["action"]
        self.traj_len = self.obss.shape[1]

    def __len__(self):
        return self.obss.shape[0]

    def get_random_obs(self):
        traj_idx = np.random.randint(len(self))
        step = np.random.randint(self.traj_len - 1)
        return self.obss[traj_idx][step], traj_idx, step
