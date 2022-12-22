# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_wrapper import NormalizeActions
from .maze import maze_distance
from .dmc2gym import walker_distance
from .pusher import pusher_distance


def make_env(env_cfg, space_info, seed=0):
    if env_cfg.id.startswith("maze"):
        from .maze.maze_env import make_maze_env

        env = make_maze_env(env_cfg, space_info)
    elif env_cfg.id == "walker":
        from .dmc2gym.dmc2gym_env import make_dmc2gym_env

        env = make_dmc2gym_env(env_cfg, space_info, seed)
    elif env_cfg.id == "pusher":
        from .pusher.pusher_env import make_pusher_env

        env = make_pusher_env(env_cfg, space_info)
    else:
        raise ValueError(f"wrong env name {env_cfg.id}")
    env.seed(seed)
    env = NormalizeActions(env)
    return env


def oracle_reward(cfg, x1, x2):
    if cfg.env.id == 'maze_U4rooms':
        oracle_distance = maze_distance
    elif cfg.env.id == 'walker':
        oracle_distance = walker_distance
    elif cfg.env.id == 'pusher':
        oracle_distance = pusher_distance
    else:
        raise ValueError()

    if cfg.main.reward == "oracle_sparse":
        if oracle_distance(x1, x2) < cfg.env.success_thresh:
            return 0
        return -1
    elif cfg.main.reward == "oracle_dense":
        return -oracle_distance(x1, x2)
    else:
        raise ValueError()
