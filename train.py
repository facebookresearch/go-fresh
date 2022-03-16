import torch
import hydra

import sac
import envs
import utils

import multiprocessing as mp
from replay_buffer import ReplayBuffer
from exploration_buffer import ExplorationBuffer

def start_procs(cfg):
    ctx = mp.get_context("fork")
    buffers, barriers, n_eval_done = create_mputils(cfg.eval.num_procs, (7,),
            (2,), ctx)

    procs = []
    for i in range(cfg.eval.num_procs):
        p = ctx.Process(target=worker_eval, args=(cfg, i, buffers, barriers,
            n_eval_done,))
        p.start()
        procs.append(p)
    return procs, buffers, barriers, n_eval_done

def eval_agent(agent, num_episodes, buffers, barriers, n_eval_done):
    agent.eval()
    n_eval_done.value = 0
    eval_stat = {x: 0.0 for x in ['oracle_success', 'oracle_distance']}
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
        eval_stat[x] /= n_eval_done.value
    barriers["end"].wait()
    return eval_stat

def worker_eval(cfg, i, buffers, barriers, n_eval_done):
    env = envs.make_env(cfg.env, seed=i)
    for epoch in range(cfg.optim.num_epochs):
        barriers["sta"].wait()
        obs = env.reset()
        num_steps = 0
        while n_eval_done.value < cfg.eval.num_episodes:
            buffers["obs"][i, 0] = torch.from_numpy(obs['state'])
            buffers["obs"][i, 1] = torch.from_numpy(obs['goal'])
            barriers["obs"].wait()
            barriers["act"].wait()
            obs, _, _, info = env.step(buffers["act"][i])
            num_steps += 1
            if num_steps >= env._max_episode_steps:
                for key in ['oracle_success', 'oracle_distance']:
                    buffers[key][i] = info[key]
                with n_eval_done.get_lock():
                    n_eval_done.value += 1
                obs = env.reset()
                num_steps = 0
            else:
                for key in ['oracle_success', 'oracle_distance']:
                    buffers[key][i] = 0
            barriers["stp"].wait()
        barriers["end"].wait()

def create_mputils(n, obs_space, act_space, ctx):
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

    buffers = {}
    buffers["obs"] = torch.zeros((n, 2, *obs_space))
    buffers["act"] = torch.zeros((n, *act_space))
    for x in ['oracle_success', 'oracle_distance']:
        buffers[x] = torch.zeros(n)

    for k, v in buffers.items():
        v.share_memory_()
        buffers[k] = v

    return buffers, barriers, n_eval_done

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    device = torch.device("cuda:0")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)

    exploration_buffer = ExplorationBuffer(cfg.exploration_buffer)

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)

    agent = sac.SAC(cfg.sac, space_info, device)

    procs, buffers, barriers, n_eval_done = start_procs(cfg)

    for epoch in range(cfg.optim.num_epochs):
        print(f"epoch: {epoch}")

        #eval
        eval_stats = eval_agent(agent, cfg.eval.num_episodes, buffers,
                barriers, n_eval_done)
        print(eval_stats)

        #train
        replay_buffer.flush()
        replay_buffer.fill(exploration_buffer, utils.oracle_reward)

        stats = agent.train_one_epoch(replay_buffer)
        print(stats)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
