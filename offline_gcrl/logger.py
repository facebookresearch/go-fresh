import os
import wandb

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logs_dir, cfg):
        self.logs_dir = logs_dir
        self.cfg = cfg
        os.makedirs(logs_dir, exist_ok=True)
        if self.cfg.plot.type == "tb":
            self.writer = SummaryWriter(logs_dir)
            self.writer.add_text("config", OmegaConf.to_yaml(cfg))
        elif self.cfg.plot.type == "wandb":
            os.makedirs(cfg.plot.wandb_dir, exist_ok=True)
            wandb.init(
                project=cfg.plot.wandb_project,
                name=cfg.main.name,
                dir=cfg.plot.wandb_dir,
                group=cfg.plot.wandb_group,
            )
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb.config.update(cfg_dict)

    def add_stats(self, stats, epoch, tb_tab):
        if self.cfg.plot.type == "tb":
            for k, v in stats.items():
                self.writer.add_scalar(f"{tb_tab}/{k}", v, epoch)
        elif self.cfg.plot.type == "wandb":
            metrics = {"epoch": epoch}
            for k, v in stats.items():
                metrics[f"{tb_tab}/{k}"] = v
            wandb.log(metrics, step=epoch)

    def close(self):
        if self.cfg.plot.type == "wandb":
            wandb.finish()
