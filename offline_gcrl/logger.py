import os
import wandb

from omegaconf import OmegaConf


class Logger:
    def __init__(self, logs_dir, cfg):
        self.logs_dir = logs_dir
        self.cfg = cfg
        os.makedirs(logs_dir, exist_ok=True)
        if cfg.wandb.enable:
            os.makedirs(cfg.wandb.dir, exist_ok=True)
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.main.name,
                dir=cfg.wandb.dir,
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
