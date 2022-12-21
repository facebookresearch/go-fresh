import pathlib
import logging
import numpy as np
import multiprocessing as mp


log = logging.getLogger(__name__)


class ExplorationBuffer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.print_fn = log.info
        dir_path = pathlib.Path(__file__).resolve().parent.parent
        self.load_data(dir_path / cfg.data_dir)
        self.embs = None

    def read_x(self, path):
        with np.load(path) as data:
            return dict(data)

    def parallel_read(self, files, num_procs):
        with mp.Pool(num_procs) as pool:
            x_list = pool.map(self.read_x, files)
        traj_buffer = {}
        for x in x_list:
            for k, v in x.items():
                if k not in traj_buffer:
                    traj_buffer[k] = []
                traj_buffer[k].append(v)
        traj_buffer = {k: np.stack(v) for k, v in traj_buffer.items()}
        return traj_buffer

    def load_data(self, data_dir):
        self.print_fn(f"loading exploration buffer from {data_dir}")
        # exclude hidden files
        ep_files = [
            f
            for f in data_dir.iterdir()
            if not any(part.startswith(".") for part in f.parts)
        ]
        ep_files.sort()  # make sure ordering is always the same
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

    def get_random_obs(self, not_last=False, frame_stack=1):
        max_step = self.traj_len - 1 if not_last else self.traj_len
        traj_idx = np.random.randint(len(self))
        step = np.random.randint(frame_stack - 1, max_step)
        return self.get_obs(traj_idx, step, frame_stack=frame_stack), traj_idx, step

    def sample(self, range=None):
        if range is None:
            i = np.random.randint(0, len(self) - 1)
        else:
            i = np.random.randint(range[0], range[1] - 1)
        return self.get_traj(i)

    def get_traj(self, traj_idx):
        assert traj_idx < len(self), "invalid traj_idx"
        return {"obs": self.obss[traj_idx], "state": self.states[traj_idx]}

    def get_obs(self, traj_idx, step, frame_stack=1):
        assert step < self.traj_len, "invalid step"
        if frame_stack == 1:
            return self.get_traj(traj_idx)["obs"][step]
        else:
            obs_stack = []
            for t in range(frame_stack):
                stack_step = step + t - frame_stack + 1
                stack_step = max(0, stack_step)
                obs_stack.append(self.get_traj(traj_idx)["obs"][stack_step])
            return obs_stack

    def get_state(self, traj_idx, step):
        assert step < self.traj_len, "invalid step"
        return self.get_traj(traj_idx)["state"][step]

    def get_obss_array(self):
        return self.obss

    def get_states_array(self):
        return self.states

    def set_embs(self, embs):
        self.embs = embs
