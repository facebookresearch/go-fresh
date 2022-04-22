import os
import torch
import hydra
import logging

import sac
import eval
import utils

import rnet.utils as rnet_utils

from rnet.memory import RNetMemory
from logger import Logger
from replay_buffer import ReplayBuffer
from exploration_buffer import ExplorationBuffer


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)
    tb_log = Logger(cfg.main.logs_dir, cfg)

    log.info(f'exp name: {cfg.main.name}')

    device = torch.device("cuda:0")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)

    exploration_buffer = ExplorationBuffer(cfg.exploration_buffer, log)

    kwargs = {}
    if not cfg.main.oracle_reward:
        rnet_memory = RNetMemory(cfg.rnet.memory, space_info,
                cfg.rnet.model.feat_size, device)
        rnet_memory, NN = rnet_utils.load(cfg.rnet.load_from, rnet_memory)

        kwargs["NN"] = NN
        kwargs["graph_dist"] = rnet_memory.dist

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)

    agent = sac.SAC(cfg.sac, space_info, device, log)

    procs, buffers, barriers, n_eval_done, info_keys = eval.start_procs(cfg,
            space_info)

    num_updates = 0
    for epoch in range(cfg.optim.num_epochs):
        log.info(f"epoch: {epoch}")

        ## EVAL
        eval_stats = eval.run(agent, cfg.eval.num_episodes, buffers, barriers,
                n_eval_done, info_keys)
        log.info("eval " + ' - '.join([f"{k}: {v:.2f}" for k, v in
            eval_stats.items()]))
        tb_log.add_stats(eval_stats, epoch, 'eval')

        ## TRAIN
        replay_buffer.flush()
        rnet_utils.fill_replay_buffer(replay_buffer, exploration_buffer, cfg,
                **kwargs)

        train_stats = agent.train_one_epoch(replay_buffer)
        log.info("train " + ' - '.join([f"{k}: {v:.2f}" for k, v in
            train_stats.items()]))
        num_updates += train_stats['updates']
        train_stats['updates'] = num_updates
        tb_log.add_stats(train_stats, epoch, 'train')

        if epoch % cfg.main.save_interval == 0:
            agent.save_checkpoint(cfg.main.logs_dir, epoch)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
