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
    barriers["sta"].wait()
    while n_eval_done.value < num_episodes:
        barriers["obs"].wait()
        with torch.no_grad():
            actions = agent.select_actions(buffers["obs"].to(device), evaluate=True)
            buffers["act"].copy_(actions)
        barriers["act"].wait()
        barriers["stp"].wait()
        for x in eval_stat:
            eval_stat[x] += buffers[x].sum().item()
    barriers["end"].wait()

    if rnet_model is not None:
        with torch.no_grad():
            rnet_val = rnet_model(
                buffers["final_obs"][:, 0].to(device),
                buffers["final_obs"][:, 1].to(device),
                batchwise=True
            )[:, 0].cpu()
            eval_stat["rnet_val"] = rnet_val.sum().item()

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
    return eval_stat


def worker_eval(cfg, space_info, i, buffers, barriers, n_eval_done):
    np.random.seed(i * cfg.main.seed)
    env = envs.make_env(cfg.env, space_info, seed=i)
    for _ in range((cfg.optim.num_epochs - 1) // cfg.eval.interval_epochs + 1):
        barriers["sta"].wait()
        goal_idx = cfg.eval.goal_idx if cfg.train.goal_strat == "one_goal" else None
        obs = env.reset(goal_idx=goal_idx)
        if cfg.env.frame_stack > 1:
            obs_stack = [torch.from_numpy(obs["obs"].copy())] * cfg.env.frame_stack
        num_steps = 0
        while n_eval_done.value < cfg.eval.num_episodes:
            if cfg.env.frame_stack == 1:
                buffers["obs"][i, 0] = torch.from_numpy(obs["obs"].copy())
            else:
                obs_stack.pop(0)
                obs_stack.append(torch.from_numpy(obs["obs"].copy()))
                for j in range(cfg.env.frame_stack):
                    buffers["obs"][i, j] = obs_stack[j]
            buffers["obs"][i, -1] = torch.from_numpy(obs["goal_obs"].copy())
            barriers["obs"].wait()
            barriers["act"].wait()
            obs, _, _, info = env.step(buffers["act"][i])
            num_steps += cfg.env.action_repeat
            if num_steps >= env._max_episode_steps:
                with n_eval_done.get_lock():
                    episode_ind = n_eval_done.value
                    n_eval_done.value += 1
                buffers["final_obs"][episode_ind, 0] = torch.from_numpy(
                    obs["obs"].copy()
                )
                buffers["final_obs"][episode_ind, 1] = torch.from_numpy(
                    obs["goal_obs"].copy()
                )

                for k, v in info.items():
                    buffers[k][i] = v

                obs = env.reset(goal_idx=goal_idx)
                if cfg.env.frame_stack > 1:
                    obs_stack = [
                        torch.from_numpy(obs["obs"].copy())
                    ] * cfg.env.frame_stack
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

    n_eval_done = Value("i", 0)

    env = envs.make_env(cfg.env, space_info)
    info_keys = env.info_keys
    env.close()

    buffers = {}
    buffers["obs"] = torch.zeros(
        (n, 1 + cfg.env.frame_stack, *space_info["shape"]["obs"])
    )
    buffers["final_obs"] = torch.zeros(
        (cfg.eval.num_episodes, 2, *space_info["shape"]["obs"])
    )
    buffers["act"] = torch.zeros((n, space_info["action_dim"]))
    for x in info_keys:
        buffers[x] = torch.zeros(n)

    for k, v in buffers.items():
        v.share_memory_()
        buffers[k] = v

    return buffers, barriers, n_eval_done, info_keys
