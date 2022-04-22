import gym
import dmc2gym
import numpy as np

from gym.wrappers.time_limit import TimeLimit

import envs.walker_utils as utils

from envs.base_wrapper import BaseWrapper

class WalkerWrapper(BaseWrapper):
    def process_obs(self, obs):
        obs = super().process_obs(obs)
        obs['state'] = self.unwrapped._env.physics.data.qpos
        return obs

    def get_state(self):
        physics = self.unwrapped._env.physics
        obs = self.unwrapped._env._task.get_observation(physics)
        return dmc2gym.wrappers._flatten_obs(obs)

    def get_state_from_obs(self, obs):
        return obs

    def get_image_from_obs(self, obs):
        return self.get_image()

    def get_image(self):
        return self.render(mode='rgb_array', height=self.cfg.obs.rgb_size,
                width=self.cfg.obs.rgb_size).transpose((2, 0, 1))

    def oracle_distance(self, x1, x2):
        def get_su(_goal):
            dist = np.abs(x1 - _goal)
            dist = dist[..., [0, 2, 3, 4, 6, 7]]
            dist[...,1] = utils.shortest_angle(dist[...,1])
            return dist.max(-1)
        return min(get_su(x2), get_su(x2[..., [0, 1, 2, 6, 7, 8, 3, 4, 5]]))

    def process_info(self, info, metrics):
        super().process_info(info, metrics)
        del info['discount']
        del info['internal_state']
        info_goal = {}
        for g in range(self.num_goals):
            info_goal[f'count-goal{g}'] = int(g == self.goal_idx)
            for k, v in info.items():
                info_goal[f'{k}-goal{g}'] = v if g == self.goal_idx else 0
        info.update(info_goal)
        return info

    def set_info_keys(self):
        super().set_info_keys()
        to_add = []
        for g in range(self.num_goals):
            to_add.append(f'count-goal{g}')
            for k in self.info_keys:
                to_add.append(f'{k}-goal{g}')
        self.info_keys += to_add

def make_walker_env(env_cfg, space_info):
    env = dmc2gym.make("walker", "walk", frame_skip=env_cfg.action_repeat)
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = WalkerWrapper(env, env_cfg, space_info)
    return env
