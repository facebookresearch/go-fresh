import os
import logging
from tqdm import tqdm

import torch
import hydra
import numpy as np

import envs
import utils
import envs.walker_utils as walker_utils

from rnet.model import RNetModel
from rnet.memory import RNetMemory
from rnet.dataset import RNetPairsSplitDataset
from exploration_buffer import ExplorationBuffer
from rnet.utils import train, build_memory, save, compute_NN
from logger import Logger


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)
    log.info(cfg)
    tb_log = Logger(cfg.main.logs_dir, cfg)

    space_info = utils.get_space_info(cfg.env.obs, cfg.env.action_dim)
    device = torch.device("cuda")
    
    expl_buffer = ExplorationBuffer(cfg.exploration_buffer)
    dataset = RNetPairsSplitDataset(cfg.rnet.dataset, expl_buffer)
    
    model = RNetModel(cfg.rnet.model, space_info)
    log.info(model)
    model = model.to(device)

    stats = train(cfg.rnet.train, model, dataset, device, tb_log)

    model.eval()
    memory = RNetMemory(cfg.rnet.memory, space_info, model.feat_size, device)
    memory = build_memory(cfg.rnet.memory, space_info, model, expl_buffer, device)
    memory.compute_dist()
    log.info(f"Memory size: {len(memory)}")

    NN = compute_NN(expl_buffer, model, memory, device)

    save_dir = cfg.rnet.load_from
    save(save_dir, model, memory, NN)


if __name__ == "__main__":
    main()
