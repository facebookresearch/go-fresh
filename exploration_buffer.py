import os
import random
import numpy as np
import multiprocessing as mp

class ExplorationBuffer(object):
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.print_fn = print if log is None else log.info
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
        self.print_fn("loading exploration buffer")
        ep_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        x_list = self.parallel_read(ep_files, num_procs=self.cfg.num_procs)
        self.obss = x_list["observation"]
        self.states = x_list["observation"]
        self.actions = x_list["action"]
        self.traj_len = self.obss.shape[1]

    def __len__(self):
        return self.obss.shape[0]

    def get_random_obs(self):
        traj_idx = np.random.randint(len(self))
        step = np.random.randint(self.traj_len - 1)
        return self.obss[traj_idx][step], traj_idx, step

    def sample(self):
        i = random.randint(0, len(self) - 1)
        return self.get_traj(i)

    def get_traj(self, traj_idx):
        assert traj_idx < len(self), "invalid traj_idx"
        return {'obs': self.obss[traj_idx],
                'state': self.states[traj_idx]}

    def get_obs(self, traj_idx, obs_idx):
        assert obs_idx < self.traj_len, "invalid obs_idx"
        return self.get_traj(traj_idx)['obs'][obs_idx]

    def get_state(self, traj_idx, state_idx):
        assert state_idx < self.traj_len, "invalid state_idx"
        return self.get_traj(traj_idx)['state'][state_idx]

    def get_obss_array(self):
        return self.obss

    def get_states_array(self):
        return self.states
