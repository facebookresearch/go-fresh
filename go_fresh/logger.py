# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import wandb
import pathlib

from omegaconf import OmegaConf


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.wandb.enable:
            wandb_dir = pathlib.Path(cfg.main.cwd) / "logs"
            wandb_dir.mkdir(exist_ok=True)
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.main.name,
                dir=wandb_dir,
                group=cfg.wandb.group,
            )
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb.config.update(cfg_dict)

    def add_stats(self, stats, epoch, tab):
        if self.cfg.wandb.enable:
            metrics = {"epoch": epoch}
            for k, v in stats.items():
                metrics[f"{tab}/{k}"] = v
            wandb.log(metrics)

    def close(self):
        if self.cfg.wandb.enable:
            wandb.finish()
