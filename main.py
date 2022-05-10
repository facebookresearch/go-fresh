from os import path, system
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
from rnet.utils import build_memory, compute_NN, embed_expl_buffer


log = logging.getLogger(__name__)


def train_rnet(cfg, model, expl_buffer, tb_log, device):
    dataset = RNetPairsSplitDataset(cfg.rnet.dataset, expl_buffer)
    _ = rnet_utils.train(cfg.rnet.train, model, dataset, device, tb_log)


def train_memory(cfg, model, explr_embs, expl_buffer, space_info, device):
    memory = build_memory(
        cfg.rnet.memory, explr_embs, space_info, model, expl_buffer, device
    )
    return memory


def train_policy(
    cfg,
    expl_buffer,
    explr_embs,
    rnet_model,
    memory,
    NN,
    space_info,
    device,
    tb_log
):
    kwargs = {}
    if cfg.main.reward in ["graph", "graph_sig"]:
        kwargs["NN"] = NN
        kwargs["graph_dist"] = memory.dist
    if cfg.main.reward in ["rnet", "graph_sig"]:
        kwargs["rnet_model"] = rnet_model
        kwargs["explr_embs"] = explr_embs

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info, device)

    agent = sac.SAC(cfg.sac, space_info, device)

    procs, buffers, barriers, n_eval_done, info_keys = eval.start_procs(cfg, space_info)

    num_updates = 0
    for epoch in range(cfg.optim.num_epochs):
        log.info(f"epoch: {epoch}")

        # TRAIN
        replay_buffer.flush()
        log.info("filling replay buffer")
        rnet_utils.fill_replay_buffer(replay_buffer, expl_buffer, cfg, **kwargs)

        log.info("train one epoch")
        train_stats = agent.train_one_epoch(replay_buffer)
        log.info(
            "train " + " - ".join([f"{k}: {v:.2f}" for k, v in train_stats.items()])
        )
        num_updates += train_stats["updates"]
        train_stats["updates"] = num_updates
        tb_log.add_stats(train_stats, epoch, "train")

        # EVAL
        eval_stats = eval.run(
            agent, cfg.eval.num_episodes, buffers, barriers, n_eval_done, info_keys
        )
        log.info("eval " + " - ".join([f"{k}: {v:.2f}" for k, v in eval_stats.items()]))
        tb_log.add_stats(eval_stats, epoch, "eval")

        if epoch % cfg.main.save_interval == 0:
            agent.save_checkpoint(cfg.main.logs_dir, epoch)

    for p in procs:
        p.join()


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    tb_log = Logger(cfg.main.logs_dir, cfg)
    log.info(f"exp name: {cfg.main.name}")

    # setup paths and load
    rnet_path = path.join(cfg.main.logs_dir, "model.pth")
    memory_path = path.join(cfg.main.logs_dir, "memory.npy")
    NN_path = path.join(cfg.main.logs_dir, "NN.npz")
    if cfg.main.load_from_dir is not None:
        for file in ["model.pth", "memory.npy", "NN.npz"]:
            load_path = path.join(cfg.main.load_from_dir, file)
            if path.exists(load_path):
                log.info(f"copying from {load_path}")
                system(f"cp {load_path} {cfg.main.logs_dir}/")

    device = torch.device("cuda")
    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)
    expl_buffer = ExplorationBuffer(cfg.exploration_buffer)

    if cfg.main.reward in ["rnet", "graph", "graph_sig"]:
        # RNet
        rnet_model = RNetModel(cfg.rnet.model, space_info).to(device)
        log.info(rnet_model)
        if path.exists(rnet_path):
            log.info(f"Loading RNet from {rnet_path}")
            rnet_model.load(rnet_path)
        else:
            log.info("Training RNet")
            train_rnet(cfg, rnet_model, expl_buffer, tb_log, device)
            log.info(f"Saving RNet to {rnet_path}")
            rnet_model.save(rnet_path)
    else:
        rnet_model = None

    if cfg.main.train_until == "rnet":
        return

    if cfg.main.reward in ["rnet", "graph", "graph_sig"]:
        explr_embs = embed_expl_buffer(expl_buffer, rnet_model, device)
        # Memory and graph
        if path.exists(memory_path):
            log.info(f"Loading memory from {memory_path}")
            memory = RNetMemory(
                cfg.rnet.memory, space_info, rnet_model.feat_size, device
            )
            memory.load(memory_path)
        else:
            log.info("Training memory")
            memory = train_memory(
                cfg=cfg,
                model=rnet_model,
                explr_embs=explr_embs,
                expl_buffer=expl_buffer,
                space_info=space_info,
                device=device,
            )
            memory.save(memory_path)
        log.info(f"Memory size: {len(memory)}")
        log.info(
            f"Number of connected components: {memory.get_nb_connected_components()}"
        )
    else:
        explr_embs = None
        memory = None

    if cfg.main.train_until == "memory":
        return

    if cfg.main.reward in ["graph", "graph_sig"]:
        # Nearest neigbhor
        if path.exists(NN_path):
            log.info(f"Loading NN from {NN_path}")
            NN = dict(np.load(NN_path))
        else:
            log.info("Computing NN")
            NN = compute_NN(explr_embs, rnet_model, memory, device)
            np.savez(NN_path, **NN)
    else:
        NN = None

    if cfg.main.train_until == "NN":
        return

    # Policy
    log.info("Training policy")
    train_policy(
        cfg=cfg,
        expl_buffer=expl_buffer,
        explr_embs=explr_embs,
        rnet_model=rnet_model,
        memory=memory,
        NN=NN,
        space_info=space_info,
        device=device,
        tb_log=tb_log,
    )


if __name__ == "__main__":
    main()
