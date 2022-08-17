import gym
import torch
import multiworld
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import collections as mc

import envs.pusher_utils as utils

from gym.wrappers.time_limit import TimeLimit
from envs.base_wrapper import BaseWrapper


class PusherWrapper(BaseWrapper):
    def __init__(self, env, cfg, space_info):
        super().__init__(env, cfg, space_info)
        self.xlim = (-0.20, 0.20)
        self.ylim = (0.45, 0.75)

    def get_state_from_obs(self, obs):
        return obs['state_observation']

    def get_image(self):
        return self.render(
            mode="rgb_array", height=self.cfg.obs.rgb_size, width=self.cfg.obs.rgb_size
        ).transpose((2, 0, 1))

    def get_image_from_obs(self, obs):
        self.set_state(obs)
        return self.get_image()

    def set_state(self, state):
        self.set_hand_xy(state[:2])
        self.set_puck_xy(state[2:])

    def compute_metrics(self, obs):
        metrics = super().compute_metrics(obs)
        metrics['hand_distance'] = np.linalg.norm(
            obs['state'][:2] - obs['goal_state'][:2]
        )
        metrics['puck_distance'] = np.linalg.norm(
            obs['state'][2:] - obs['goal_state'][2:]
        )
        return metrics

    def oracle_distance(self, x1, x2):
        return utils.oracle_distance(x1, x2)

    def process_info(self, info, metrics):
        return super().process_info({}, metrics)

    def plot_goals(self):
        fig = plt.figure()
        goals = self.get_goals()
        plt.scatter(goals['state'][:, 2], goals['state'][:, 3])
        plt.grid()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        return fig

    def plot_stats_goal_idx(self, stats):
        goals = self.get_goals()
        fig = plt.figure()
        sc = plt.scatter(goals['state'][:, 2], goals['state'][:, 3], c=stats)
        plt.colorbar(sc)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.grid()
        return fig

    def plot_buffer(self, buffer_states):
        a = buffer_states.reshape((-1, buffer_states.shape[-1]))
        fig = plt.figure()
        plt.hist2d(a[:, 2], a[:, 3], bins=20, range=(self.xlim, self.ylim))
        plt.title('puck')
        return fig

    def plot_graph_dist(self, curri):
        fig = plt.figure()
        states = curri.memory.states[:len(curri.memory)]
        i = np.random.randint(0, len(curri.memory))
        sc = plt.scatter(states[:, 2], states[:, 3], c=-curri.memory.graph_dist[i])
        plt.colorbar(sc)
        plt.scatter(states[i, 2], states[i, 3], c='red', marker='x', s=100)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.grid()
        return fig

    def plot_values(
        self, rnet_model, comp_obs, comp_state, obss, states, pos=False, sigmoid=False
    ):
        rnet_values = rnet_model.compute_values(comp_obs, obss)
        if sigmoid:
            rnet_values = torch.sigmoid(torch.from_numpy(rnet_values)).numpy()
        if pos:
            rnet_values = rnet_values > 0
        fig, ax = plt.subplots(1, 2,  figsize=(12, 4))
        sc = ax[0].scatter(states[:, 2], states[:, 3], c=rnet_values)
        fig.colorbar(sc)
        ax[0].scatter(comp_state[2], comp_state[3], c='red', marker='x', s=100)
        ax[0].set_title('puck')
        ax[1].scatter(states[:, 0], states[:, 1], c=rnet_values)
        ax[1].scatter(comp_state[0], comp_state[1], c='red', marker='x', s=100)
        ax[1].set_title('hand')
        return fig

    def plot_graph(self, memory):
        fig, ax = plt.subplots()
        states = memory.states[: len(memory)]
        lines = []
        for i in range(len(memory)):
            for j in range(i):
                if memory.adj_matrix[i, j] or memory.adj_matrix[j, i]:
                    lines.append([states[i, 2:], states[j, 2:]])
        lc = mc.LineCollection(lines, color="gray", alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(states[:, 2], states[:, 3])
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.grid()
        return fig


def make_pusher_env(env_cfg, space_info):
    multiworld.register_all_envs()
    env = gym.make('SawyerPushNIPSEasy-v0')
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = PusherWrapper(env, env_cfg, space_info)
    return env
