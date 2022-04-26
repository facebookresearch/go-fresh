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
            return dict(data)

    def parallel_read(self, files, num_procs):
        with mp.Pool(num_procs) as pool:
            x_list = pool.map(self.read_x, files)
        traj_buffer = {}
        for x in x_list:
            for k, v in x.items():
                if not k in traj_buffer:
                    traj_buffer[k] = []
                traj_buffer[k].append(v)
        traj_buffer = {k: np.stack(v) for k, v in traj_buffer.items()}
        return traj_buffer

    def seq_img(self, files, num_procs, env):
        for path in files:
            with np.load(path) as data:
                print(path)
                state_traj = data["physics"]
                data["image"] = []
                for i in range(len(state_traj)):
                    env.unwrapped._env.physics.set_state(state)
                    env.unwrapped._env.physics.forward()
                    data["image"].append(env.render(mode='rgb_array'))
                data["image"] = np.stack(data["image"])
                print(data["image"].shape)
                print(data["image"].dtype)
            np.save(path, data)
    
    def load_data(self, data_dir):
        self.print_fn("loading exploration buffer")
        ep_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        # exclude hidden files
        ep_files = [f for f in ep_files if not os.path.basename(f).startswith('.')]
        x_list = self.parallel_read(ep_files, num_procs=self.cfg.num_procs)
        self.obss = x_list["observation"]
        if "physics" in x_list:
            self.states = x_list["physics"]
        else:
            self.states = x_list["observation"]
        self.actions = x_list["action"]
        self.traj_len = self.obss.shape[1]

    def __len__(self):
        return self.obss.shape[0]

    def get_random_obs(self):
        traj_idx = np.random.randint(len(self))
        step = np.random.randint(self.traj_len - 1)
        return self.obss[traj_idx][step], traj_idx, step

    def sample(self, range=None):
        if range is None:
            i = random.randint(0, len(self) - 1)
        else:
            i = random.randint(range[0], range[1] - 1)
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
