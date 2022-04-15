import torch
import multiprocessing as mp

import envs

def start_procs(cfg, space_info):
    ctx = mp.get_context("fork")
    buffers, barriers, n_eval_done, info_keys = create_mputils(cfg, space_info,
            ctx)
    procs = []
    for i in range(cfg.eval.num_procs):
        p = ctx.Process(target=worker_eval, args=(cfg, space_info, i, buffers,
            barriers, n_eval_done,))
        p.start()
        procs.append(p)
    return procs, buffers, barriers, n_eval_done, info_keys

def run(agent, num_episodes, buffers, barriers, n_eval_done, info_keys):
    agent.eval()
    n_eval_done.value = 0
    eval_stat = {x: 0.0 for x in info_keys}
    barriers["sta"].wait()
    while n_eval_done.value < num_episodes:
        barriers["obs"].wait()
        with torch.no_grad():
            actions = agent.select_actions(buffers['obs'], evaluate=True)
            buffers["act"].copy_(actions)
        barriers["act"].wait()
        barriers["stp"].wait()
        for x in eval_stat:
            eval_stat[x] += buffers[x].sum().item()

    for x in eval_stat:
        den = n_eval_done.value
        if 'room' in x or 'goal' in x:
            key = 'room' if 'room' in x else 'goal'
            den = eval_stat[f"count-{key}{x[x.find(key) + 4]}"]
            if 'count' in x:
                den = 1
        if den == 0:
            continue
        eval_stat[x] /= den

    barriers["end"].wait()
    return eval_stat

def worker_eval(cfg, space_info, i, buffers, barriers, n_eval_done):
    env = envs.make_env(cfg.env, space_info, seed=i)
    for epoch in range(cfg.optim.num_epochs):
        barriers["sta"].wait()
        obs = env.reset()
        num_steps = 0
        while n_eval_done.value < cfg.eval.num_episodes:
            buffers["obs"][i, 0] = torch.from_numpy(obs['obs'])
            buffers["obs"][i, 1] = torch.from_numpy(obs['goal_obs'])
            barriers["obs"].wait()
            barriers["act"].wait()
            obs, _, _, info = env.step(buffers["act"][i])
            num_steps += 1
            if num_steps >= env._max_episode_steps:
                for k, v in info.items():
                    buffers[k][i] = v
                with n_eval_done.get_lock():
                    n_eval_done.value += 1
                obs = env.reset()
                num_steps = 0
            else:
                for k in info:
                    buffers[k][i] = 0
            barriers["stp"].wait()
        barriers["end"].wait()

def create_mputils(cfg, space_info, ctx):
    n = cfg.eval.num_procs
    Barrier = ctx.Barrier
    Value = ctx.Value

    barriers = {
        "sta": Barrier(n + 1),
        "obs": Barrier(n + 1),
        "act": Barrier(n + 1),
        "stp": Barrier(n + 1),
        "end": Barrier(n + 1),
    }

    n_eval_done = Value('i', 0)

    env = envs.make_env(cfg.env, space_info)
    info_keys = env.info_keys
    env.close()

    buffers = {}
    buffers["obs"] = torch.zeros((n, 2, *space_info['shape']['obs']))
    buffers["act"] = torch.zeros((n, space_info['action_dim']))
    for x in info_keys:
        buffers[x] = torch.zeros(n)

    for k, v in buffers.items():
        v.share_memory_()
        buffers[k] = v

    return buffers, barriers, n_eval_done, info_keys
