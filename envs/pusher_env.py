import gym
import multiworld
import numpy as np

from gym.wrappers.time_limit import TimeLimit

import envs.pusher_utils as utils
from envs.base_wrapper import BaseWrapper


class PusherWrapper(BaseWrapper):
    def get_state_from_obs(self, obs):
        return obs['state_observation']

    def get_image(self):
        return self.render(
            mode="rgb_array", height=self.cfg.obs.rgb_size, width=self.cfg.obs.rgb_size
        ).transpose((2, 0, 1))

    def get_image_from_obs(self, _):
        return self.get_image()

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


def make_pusher_env(env_cfg, space_info):
    multiworld.register_all_envs()
    env = gym.make('SawyerPushNIPSEasy-v0')
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = PusherWrapper(env, env_cfg, space_info)
    return env
