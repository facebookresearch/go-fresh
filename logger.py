from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, logs_dir, cfg):
        self.logs_dir = logs_dir
        self.writer = SummaryWriter(logs_dir)
        self.cfg = cfg
        self.writer.add_text('config', OmegaConf.to_yaml(cfg))

    def add_stats(self, stats, epoch, tb_tab):
        for k, v in stats.items():
            self.writer.add_scalar(f'{tb_tab}/{k}', v, epoch)
