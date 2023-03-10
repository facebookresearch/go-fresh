# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dmc2gym

from gym.wrappers.time_limit import TimeLimit

from .. import BaseWrapper
from .walker_utils import oracle_distance as walker_distance


tasks = {"walker": "walk"}


class Dmc2GymWrapper(BaseWrapper):
    def process_obs(self, obs):
        obs = super().process_obs(obs)
        obs["state"] = self.unwrapped._env.physics.data.qpos
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
        return self.render(
            mode="rgb_array", height=self.cfg.obs.rgb_size, width=self.cfg.obs.rgb_size
        ).transpose((2, 0, 1))

    def process_info(self, info, metrics):
        super().process_info(info, metrics)
        del info["discount"]
        del info["internal_state"]
        info_goal = {}
        for g in range(self.num_goals):
            info_goal[f"count-goal{g}"] = int(g == self.goal_idx)
            for k, v in info.items():
                info_goal[f"{k}-goal{g}"] = v if g == self.goal_idx else 0
        info.update(info_goal)
        return info

    def oracle_distance(self, x1, x2):
        if self.cfg.id == "walker":
            return walker_distance(x1, x2)
        raise NotImplementedError

    def set_info_keys(self):
        super().set_info_keys()
        to_add = []
        for g in range(self.num_goals):
            to_add.append(f"count-goal{g}")
            for k in self.info_keys:
                to_add.append(f"{k}-goal{g}")
        self.info_keys += to_add


def make_dmc2gym_env(env_cfg, space_info, seed):
    kwargs = {"frame_skip": env_cfg.action_repeat, "seed": seed}
    env = dmc2gym.make(env_cfg.id, tasks[env_cfg.id], **kwargs)
    env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
    env = Dmc2GymWrapper(env, env_cfg, space_info)
    return env
