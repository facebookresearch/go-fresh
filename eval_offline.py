import torch
import itertools

import numpy as np
import multiprocessing as mp

import envs

def eval(agent, cfg, num_episodes, num_procs):
    agent.eval()

    ctx = mp.get_context("fork")
    barriers, buffers, n_eval_done = create_mputils(num_procs, (7,), (2,), ctx)

    procs = []
    for i in range(num_procs):
        p = ctx.Process(target=worker_eval, args=(cfg, i, buffers,
            barriers, n_eval_done, num_episodes,))
        p.start()
        procs.append(p)

    n_eval_done.value = 0
    eval_stat = {x: 0.0 for x in ['oracle_success', 'oracle_distance']}
    while n_eval_done.value < num_episodes:
        barriers["obs"].wait()
        with torch.no_grad():
            actions = agent.select_actions(buffers['obs'], evaluate=True)
            buffers["act"].copy_(actions)
        barriers["act"].wait()
        barriers["stp"].wait()
        for x in eval_stat:
            eval_stat[x] += buffers[x].sum().item()
    for p in procs:
        p.join()
    for x in eval_stat:
        eval_stat[x] /= n_eval_done.value
    return eval_stat

def worker_eval(cfg, i, buffers, barriers, num_eval_done, num_episodes):
    env = envs.make_env(cfg, seed=i)
    obs = env.reset()
    num_steps = 0
    while num_eval_done.value < num_episodes:
        buffers["obs"][i, 0] = torch.from_numpy(obs['state'])
        buffers["obs"][i, 1] = torch.from_numpy(obs['goal'])
        barriers["obs"].wait()
        barriers["act"].wait()
        obs, _, _, info = env.step(buffers["act"][i])
        num_steps += 1
        if num_steps >= env._max_episode_steps:
            for key in ['oracle_success', 'oracle_distance']:
                buffers[key][i] = info[key]
            with num_eval_done.get_lock():
                num_eval_done.value += 1
            obs = env.reset()
            num_steps = 0
        else:
            for key in ['oracle_success', 'oracle_distance']:
                buffers[key][i] = 0
        barriers["stp"].wait()

def create_mputils(n, obs_space, act_space, ctx):
    Barrier = ctx.Barrier
    Value = ctx.Value

    barriers = {
        "obs": Barrier(n + 1),
        "act": Barrier(n + 1),
        "stp": Barrier(n + 1),
    }

    num_eval_done = Value('i', 0)

    buffers = {}
    buffers["obs"] = torch.zeros((n, 2, *obs_space)) 
    buffers["act"] = torch.zeros((n, *act_space))
    for x in ['oracle_success', 'oracle_distance']:
        buffers[x] = torch.zeros(n)

    for k, v in buffers.items():
        v.share_memory_()
        buffers[k] = v

    return barriers, buffers, num_eval_done
