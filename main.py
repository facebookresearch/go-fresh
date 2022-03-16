import torch
import hydra

import sac
import eval
import utils

from replay_buffer import ReplayBuffer
from exploration_buffer import ExplorationBuffer

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    device = torch.device("cuda:0")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)

    exploration_buffer = ExplorationBuffer(cfg.exploration_buffer)

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)

    agent = sac.SAC(cfg.sac, space_info, device)

    procs, buffers, barriers, n_eval_done = eval.start_procs(cfg)

    for epoch in range(cfg.optim.num_epochs):
        print(f"epoch: {epoch}")

        #eval
        eval_stats = eval.run(agent, cfg.eval.num_episodes, buffers, barriers,
                n_eval_done)
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
