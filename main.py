from os import path
import numpy as np
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
from rnet.model import RNetModel
from rnet.dataset import RNetPairsSplitDataset
from rnet.utils import build_memory, compute_NN


def train_rnet(cfg, model, expl_buffer, tb_log, device):
    dataset = RNetPairsSplitDataset(cfg.rnet.dataset, expl_buffer)
    stats = rnet_utils.train(cfg.rnet.train, model, dataset, device, tb_log)


def train_memory(cfg, memory, model, expl_buffer, space_info, device):
    model.eval()
    memory = build_memory(cfg.rnet.memory, space_info, model, expl_buffer, device)
    memory.compute_dist()
    NN = compute_NN(expl_buffer, model, memory, device)
    return memory, NN


def train_policy(cfg, expl_buffer, memory, NN, space_info, device, log, tb_log):
    kwargs = {}
    if not cfg.main.oracle_reward:
        kwargs["NN"] = NN
        kwargs["graph_dist"] = memory.dist

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)

    agent = sac.SAC(cfg.sac, space_info, device, log)

    procs, buffers, barriers, n_eval_done, info_keys = eval.start_procs(cfg, space_info)

    num_updates = 0
    for epoch in range(cfg.optim.num_epochs):
        log.info(f"epoch: {epoch}")

        # EVAL
        eval_stats = eval.run(
            agent, cfg.eval.num_episodes, buffers, barriers, n_eval_done, info_keys
        )
        log.info("eval " + " - ".join([f"{k}: {v:.2f}" for k, v in eval_stats.items()]))
        tb_log.add_stats(eval_stats, epoch, "eval")

        # TRAIN
        replay_buffer.flush()
        rnet_utils.fill_replay_buffer(replay_buffer, expl_buffer, cfg, **kwargs)

        train_stats = agent.train_one_epoch(replay_buffer)
        log.info(
            "train " + " - ".join([f"{k}: {v:.2f}" for k, v in train_stats.items()])
        )
        num_updates += train_stats["updates"]
        train_stats["updates"] = num_updates
        tb_log.add_stats(train_stats, epoch, "train")

        if epoch % cfg.main.save_interval == 0:
            agent.save_checkpoint(cfg.main.logs_dir, epoch)

    for p in procs:
        p.join()


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)
    tb_log = Logger(cfg.main.logs_dir, cfg)
    log.info(f"exp name: {cfg.main.name}")

    device = torch.device("cuda")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)
    expl_buffer = ExplorationBuffer(cfg.exploration_buffer, log)

    # RNet
    rnet_model = RNetModel(cfg.rnet.model, space_info).to(device)
    log.info(rnet_model)
    rnet_path = path.join(cfg.main.logs_dir, "model.pth")
    if path.exists(rnet_path):
        log.info(f"Loading RNet from {rnet_path}")
        rnet_model.load(rnet_path)
    else:
        log.info("Training RNet")
        train_rnet(cfg, rnet_model, expl_buffer, tb_log, device)
        log.info(f"Saving RNet to {rnet_path}")
        rnet_model.save(rnet_path)

    # Memory and graph
    memory = RNetMemory(cfg.rnet.memory, space_info, rnet_model.feat_size, device)
    memory_path = path.join(cfg.main.logs_dir, "memory.npy")
    NN_path = path.join(cfg.main.logs_dir, "NN.npy")
    if path.exists(memory_path):
        log.info(f"Loading memory from {memory_path}")
        memory.load(memory_path)
        NN = np.load(NN_path)
    else:
        log.info("Training memory")
        NN = train_memory(cfg, memory, rnet_model, expl_buffer, space_info, device)
        memory.save(memory_path)
        np.save(NN_path, NN)
    log.info(f"Memory size: {len(memory)}")

    # Policy
    log.info("Training policy")
    train_policy(cfg, expl_buffer, memory, NN, space_info, device, log, tb_log)


if __name__ == "__main__":
    main()
