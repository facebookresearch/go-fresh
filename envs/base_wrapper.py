import os
import gym
import numpy as np
import matplotlib.pyplot as plt

WORK_DIR = "/private/home/linamezghani/code/pytorch-soft-actor-critic/"


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class BaseWrapper(gym.core.Wrapper):
    def __init__(self, env, cfg, space_info):
        super().__init__(env)
        self.cfg = cfg
        self.t = 0
        self._max_episode_steps = self.env._max_episode_steps
        self.set_observation_space(space_info)
        self.load_topline_goals()
        self.set_info_keys()

    def seed(self, seed):
        super().seed(seed)
        self.action_space.seed(seed)
        # np.random.seed(seed)

    def load_topline_goals(self):
        goals_file = os.path.join(WORK_DIR, f"envs/topline_goals/{self.cfg.id}.npy")
        if os.path.isfile(goals_file):
            self.set_goals(np.load(goals_file, allow_pickle=True).tolist())
        else:
            print("goals_file not found... garbage goals")
            self.set_goals(
                {
                    x: np.zeros((1, *self.observation_space[x].shape))
                    for x in ["state", "image"]
                }
            )
        goals = self.get_goals()
        self.set_goals(goals)

    def save_topline_goals(self, goals):
        goals_file = os.path.join(WORK_DIR, f"envs/topline_goals/{self.cfg.id}.npy")
        np.save(goals_file, goals)

    def init_observation_space(self, space_info):
        if self.cfg.obs.type == "vec":
            low, high = -np.inf, np.inf
            dtype = np.float32
        else:
            low, high = 0, 255
            dtype = np.uint8
        self.observation_space = gym.spaces.dict.Dict(
            {
                "state": gym.spaces.box.Box(
                    -np.inf, np.inf, space_info["shape"]["state"]
                ),
                "obs": gym.spaces.box.Box(
                    low, high, space_info["shape"]["obs"], dtype=dtype
                ),
            }
        )

    def set_observation_space(self, space_info):
        self.init_observation_space(space_info)
        assert isinstance(self.observation_space, gym.spaces.dict.Dict)
        self.observation_space.spaces["time"] = gym.spaces.Box(np.zeros(1), np.ones(1))

    def set_info_keys(self):
        self.info_keys = ["oracle_distance", "oracle_success"]

    def get_s0(self):
        obs = self.reset()
        return obs["obs"]

    def compute_metrics(self, obs):
        metrics = {}
        metrics["oracle_distance"] = self.oracle_distance(
            obs["state"], obs["goal_state"]
        )
        metrics["oracle_success"] = bool(
            metrics["oracle_distance"] < self.cfg.success_thresh
        )
        return metrics

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.t += self.cfg.action_repeat
        obs = self.process_obs(obs)
        metrics = self.compute_metrics(obs)
        reward = self.process_reward(reward, metrics)
        info = self.process_info(info, metrics)
        return obs, reward, done, info

    def reset(self, goal_idx=None):
        self.t = 0
        if goal_idx is None:
            goal_idx = self.sample_goal_idx()
        self.set_goal_idx(goal_idx)
        obs = super().reset()
        if self.cfg.random_start_pos:
            self.set_random_pos()
            obs = self.get_state()
        obs = self.process_obs(obs)
        return obs

    def get_image_from_obs(self, obs):
        raise NotImplementedError

    def get_state_from_obs(self, obs):
        raise NotImplementedError

    def set_random_pos(self):
        raise NotImplementedError

    def process_obs(self, obs):
        new_obs = {"time": self.t / self._max_episode_steps}
        goal_idx = self.get_goal_idx()
        new_obs["state"] = self.get_state_from_obs(obs)
        new_obs["goal_state"] = self.get_goals()["state"][goal_idx]
        if self.cfg.obs.type == "rgb":
            new_obs["obs"] = self.get_image_from_obs(obs)
        else:
            new_obs["obs"] = new_obs["state"]
        new_obs["goal_obs"] = self.get_goals()[f"{self.cfg.obs.type}_obs"][goal_idx]
        return new_obs

    def process_info(self, info, metrics):
        if "TimeLimit.truncated" in info:
            del info["TimeLimit.truncated"]
        info.update(metrics)
        return info

    def process_reward(self, reward, metrics):
        reward = -metrics["oracle_distance"]
        return reward

    def oracle_distance(self, x1, x2):
        raise NotImplementedError

    def set_goals(self, goals):
        self.goals = goals
        self.num_goals = self.goals["state"].shape[0]

    def get_goals(self):
        return self.goals

    def set_goal_idx(self, goal_idx):
        self.goal_idx = goal_idx

    def get_goal_idx(self):
        return self.goal_idx

    def sample_goal_idx(self):
        return np.random.choice(self.num_goals, size=1)[0]

    def init_stats_goal_idx(self):
        return {
            x: np.zeros(self.num_goals)
            for x in ["count", "oracle_distance", "oracle_success"]
        }

    def add_stats_goal_idx(self, stats, info):
        stats["count"][self.goal_idx] += 1
        stats["oracle_distance"][self.goal_idx] += info["oracle_distance"]
        stats["oracle_success"][self.goal_idx] += info["oracle_success"]
        return stats

    def plot_goals(self):
        return

    def plot_buffer(self, buffer_states):
        return

    def plot_stats_goal_idx(self, stats):
        return

    def plot_adj_matrix(self, curri):
        fig = plt.figure()
        sc = plt.imshow(curri.memory.adj_matrix.astype(int))
        plt.colorbar(sc)
        return fig

    def plot_graph_dist(self, curri):
        return

    def plot_graph(self, curri):
        return
