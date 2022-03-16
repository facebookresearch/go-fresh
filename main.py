import os
import torch
import hydra
import logging

import sac
import eval
import utils

from logger import Logger
from replay_buffer import ReplayBuffer
from exploration_buffer import ExplorationBuffer


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):

    log = logging.getLogger(__name__)
    logs_dir = os.path.join('/checkpoint/linamezghani/offline-gcrl_logs',
            cfg.main.name)
    tb_log = Logger(logs_dir, cfg)

    device = torch.device("cuda:0")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)

    exploration_buffer = ExplorationBuffer(cfg.exploration_buffer, log)

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device, log)

    agent = sac.SAC(cfg.sac, space_info, device)

    procs, buffers, barriers, n_eval_done = eval.start_procs(cfg)

    num_updates = 0
    for epoch in range(cfg.optim.num_epochs):
        log.info(f"epoch: {epoch}")

        #eval
        eval_stats = eval.run(agent, cfg.eval.num_episodes, buffers, barriers,
                n_eval_done)
        log.info("eval " + ' - '.join([f"{k}: {v:.2f}" for k, v in
            eval_stats.items()]))
        tb_log.add_stats(eval_stats, epoch, 'eval')

        #train
        replay_buffer.flush()
        replay_buffer.fill(exploration_buffer, utils.oracle_reward)

        train_stats = agent.train_one_epoch(replay_buffer)
        log.info("train " + ' - '.join([f"{k}: {v:.2f}" for k, v in
            train_stats.items()]))
        num_updates += train_stats['updates']
        train_stats['updates'] = num_updates
        tb_log.add_stats(train_stats, epoch, 'train')

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
