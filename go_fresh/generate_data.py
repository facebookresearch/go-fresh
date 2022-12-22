# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import argparse

import numpy as np
import multiprocessing as mp
from omegaconf import OmegaConf

from .envs import make_env
from .utils import get_space_info


def generate_episode(env, steps, ep_id, save_dir):
    ep_data = {
        "observation": np.zeros((steps + 1, *env.observation_space["obs"].shape)),
        "action": np.zeros((steps + 1, env.action_space.shape[0])),
    }
    obs = env.reset()
    ep_data["observation"][0] = obs["obs"]
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        ep_data["action"][step] = action
        ep_data["observation"][step] = obs["obs"]

    np.savez_compressed(save_dir / f"episode_{ep_id:06d}_{steps}.npz", **ep_data)


def generate_episodes(env, steps, ep_ids, seed, save_dir):
    """ep_ids is a tuple of [first_id, last_id)"""
    env.seed(seed)
    env.action_space.seed(seed)
    for ep_id in range(*ep_ids):
        generate_episode(env, steps, ep_id, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, required=True, help="ID of env")
    parser.add_argument(
        "--num-episodes", "-n", type=int, help="total number of episodes", default=10000
    )
    parser.add_argument(
        "--num-procs", "-p", type=int, help="number of procs", default=10
    )
    parser.add_argument("--ep-len", "-l", type=int, help="episode length")
    parser.add_argument("--data-dir", "-s", type=str, help="data dir", default='data')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    save_dir = data_dir / args.env
    save_dir.mkdir(exist_ok=True)

    cfg = OmegaConf.load(f"conf/env/{args.env}.yaml")
    cfg.env.random_start_pos = args.env == "maze"

    space_info = get_space_info(cfg.env.obs, cfg.env.action_dim)
    env = make_env(cfg.env, space_info)

    procs = []
    n = args.num_episodes // args.num_procs
    for i in range(args.num_procs):
        p = mp.Process(
            target=generate_episodes,
            args=(env, args.ep_len, (i * n, (i + 1) * n), i, save_dir),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
