import torch
import hydra

import sac
import utils
import eval_offline

from replay_buffer import ReplayBuffer
from exploration_buffer import ExplorationBuffer

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    device = torch.device("cuda:0")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)

    exploration_buffer = ExplorationBuffer(cfg.exploration_buffer)

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)
    replay_buffer.fill(exploration_buffer, utils.oracle_reward)

    agent = sac.SAC(cfg.sac, space_info, device)

    for epoch in range(cfg.optim.num_epochs):
        print(f"epoch: {epoch}")

        out = eval_offline.eval(agent, cfg.env, 500, 10)
        print(out)

        stats = agent.train_one_epoch(replay_buffer)
        print(stats)

if __name__ == "__main__":
    main()
