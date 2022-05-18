import torch
import multiprocessing as mp
import numpy as np

import envs


def start_procs(cfg, space_info):
    ctx = mp.get_context("fork")
    buffers, barriers, n_eval_done, info_keys = create_mputils(cfg, space_info, ctx)
    procs = []
    for i in range(cfg.eval.num_procs):
        p = ctx.Process(
            target=worker_eval,
            args=(cfg, space_info, i, buffers, barriers, n_eval_done,),
        )
        p.start()
        procs.append(p)
    return procs, buffers, barriers, n_eval_done, info_keys


def run(
    agent, num_episodes, buffers, barriers, n_eval_done, info_keys, rnet_model, device
):
    agent.eval()
    n_eval_done.value = 0
    eval_stat = {x: 0.0 for x in info_keys}
    avg_rnet_val = 0
    barriers["sta"].wait()
    while n_eval_done.value < num_episodes:
        barriers["obs"].wait()
        with torch.no_grad():
            actions = agent.select_actions(buffers["obs"].to(device), evaluate=True)
            buffers["act"].copy_(actions)
        barriers["act"].wait()
        barriers["stp"].wait()
        if rnet_model is not None and buffers["done"].sum() > 0:
            with torch.no_grad():
                rnet_val = rnet_model(
                    buffers["final_obs"][:, 0].to(device),
                    buffers["final_obs"][:, 1].to(device),
                    batchwise=True
                )[:, 0].cpu()
                avg_rnet_val += (buffers["done"] * rnet_val).sum().item()
        barriers["rnv"].wait()
        for x in eval_stat:
            eval_stat[x] += buffers[x].sum().item()

    eval_stat["rnet_val"] = avg_rnet_val
    for x in eval_stat:
        den = n_eval_done.value
        if "room" in x or "goal" in x:
            key = "room" if "room" in x else "goal"
            den = eval_stat[f"count-{key}{x[x.find(key) + 4:]}"]
            if "count" in x:
                den = 1
        if den == 0:
            continue
        eval_stat[x] /= den

    barriers["end"].wait()
    return eval_stat


def worker_eval(cfg, space_info, i, buffers, barriers, n_eval_done):
    np.random.seed(i * cfg.main.seed)
    env = envs.make_env(cfg.env, space_info, seed=i)
    for _ in range((cfg.optim.num_epochs - 1) // cfg.eval.interval_epochs + 1):
        barriers["sta"].wait()
        goal_idx = cfg.eval.goal_idx if cfg.train.goal_strat == "one_goal" else None
        obs = env.reset(goal_idx=goal_idx)
        buffers["done"][buffers["done"] != 0] = 0
        num_steps = 0
        while n_eval_done.value < cfg.eval.num_episodes:
            buffers["obs"][i, 0] = torch.from_numpy(obs["obs"])
            buffers["obs"][i, 1] = torch.from_numpy(obs["goal_obs"])
            barriers["obs"].wait()
            barriers["act"].wait()
            obs, _, _, info = env.step(buffers["act"][i])
            num_steps += cfg.env.action_repeat
            if num_steps >= env._max_episode_steps:
                buffers["final_obs"][i, 0] = torch.from_numpy(obs["obs"])
                buffers["final_obs"][i, 1] = torch.from_numpy(obs["goal_obs"])
                buffers["done"][i] = 1

                for k, v in info.items():
                    buffers[k][i] = v
                with n_eval_done.get_lock():
                    n_eval_done.value += 1
                obs = env.reset(goal_idx=goal_idx)
                num_steps = 0
            else:
                buffers["done"][i] = 0
                for k in info:
                    buffers[k][i] = 0
            barriers["stp"].wait()
            barriers["rnv"].wait()
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
        "rnv": Barrier(n + 1),  # rnet values computation
        "end": Barrier(n + 1),
    }

    n_eval_done = Value("i", 0)

    env = envs.make_env(cfg.env, space_info)
    info_keys = env.info_keys
    env.close()

    buffers = {}
    buffers["obs"] = torch.zeros((n, 2, *space_info["shape"]["obs"]))
    buffers["final_obs"] = torch.zeros_like(buffers["obs"])
    buffers["act"] = torch.zeros((n, space_info["action_dim"]))
    buffers["done"] = torch.zeros(n, dtype=int)
    for x in info_keys:
        buffers[x] = torch.zeros(n)

    for k, v in buffers.items():
        v.share_memory_()
        buffers[k] = v

    return buffers, barriers, n_eval_done, info_keys
