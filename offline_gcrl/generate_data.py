import os
import argparse

import numpy as np
import multiprocessing as mp
from omegaconf import OmegaConf

import envs

def generate_episode(env, steps, ep_id, save_dir):
    ep_data = {
        'observation': np.zeros((steps + 1, *env.observation_space['state'].shape)),
        'action': np.zeros((steps + 1, env.action_space.shape[0]))
    }
    obs = env.reset()
    ep_data['observation'][0] = obs['state']
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        ep_data['action'][step] = action
        ep_data['observation'][step] = obs['state']
    
    np.savez_compressed(os.path.join(save_dir, f'episode_{ep_id:06d}_{steps}.npz'), **ep_data)
    
def generate_episodes(env, steps, ep_ids, seed, save_dir):
    """ep_ids is a tuple of [first_id, last_id)"""
    env.seed(seed)
    env.action_space.seed(seed)
    for ep_id in range(*ep_ids):
        generate_episode(env, steps, ep_id, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, required=True, help='ID of env')
    parser.add_argument('--num-episodes', '-n', type=int,
            help='total number of episodes')
    parser.add_argument('--num-procs', '-p', type=int, help='number of procs')
    parser.add_argument('--ep-len', '-l', type=int, help='episode length')
    parser.add_argument('--random-start-pos', '-r', action="store_true",
            default=False)
    parser.add_argument('--save-dir', '-s', type=str, required=True,
            help='save dir')
    args = parser.parse_args()

    n = args.num_episodes // args.num_procs

    cfg = OmegaConf.load('conf/config.yaml')
    env_cfg = OmegaConf.load(f'conf/env/{args.env}.yaml')
    cfg = OmegaConf.merge(cfg, env_cfg)
    cfg.env.random_start_pos = args.random_start_pos

    env = envs.make_env(cfg.env)

    procs = []
    for i in range(args.num_procs):
        p = mp.Process(target=generate_episodes, args=(env, args.ep_len,
                (i * n, (i + 1) * n), i, args.save_dir))
        p.start()
        procs.append(p)
    
    for p in procs:
        p.join()
