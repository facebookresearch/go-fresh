import os
import logging
import torch

import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from model import GaussianPolicy, QNetwork, DeterministicPolicy


log = logging.getLogger(__name__)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SAC(object):
    def __init__(self, cfg, space_info, device):
        self.cfg = cfg
        self.space_info = space_info
        self.alpha = cfg.optim.entropy.alpha
        self.device = device

        self.critic = QNetwork(
            space_info["shape"]["obs"], space_info["action_dim"], cfg.policy
        )
        self.critic = self.critic.to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.optim.lr)

        self.critic_target = QNetwork(
            space_info["shape"]["obs"], space_info["action_dim"], cfg.policy
        )
        self.critic_target = self.critic_target.to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.cfg.policy.type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the
            # paper
            if self.cfg.optim.entropy.auto_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor((space_info["action_dim"],)).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=cfg.optim.entropy.lr)

            self.policy = GaussianPolicy(
                space_info["shape"]["obs"], space_info["action_dim"], cfg.policy
            ).to(self.device)

        else:
            self.alpha = 0
            self.cfg.optim.entropy.auto_tuning = False
            self.policy = DeterministicPolicy(
                space_info["shape"]["obs"], space_info["action_dim"], cfg.policy
            ).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=cfg.optim.policy.lr)

    def select_action(self, state, evaluate=False):
        proc_state = np.stack((state["obs"], state["goal_obs"]))
        proc_state = torch.FloatTensor(proc_state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(proc_state)
        else:
            with torch.no_grad():
                _, _, action = self.policy.sample(proc_state)
        return action.detach().cpu().numpy()[0]

    def select_actions(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            with torch.no_grad():
                _, _, action = self.policy.sample(state)
        return action.detach().cpu()

    def update_parameters(self, memory, batch_size, updates):
        stats = {}

        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target -= self.alpha * next_state_log_pi
            next_q_value = (
                reward_batch + mask_batch * self.cfg.optim.gamma * min_qf_next_target
            )
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        stats["qf1_loss"] = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        stats["qf2_loss"] = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = stats["qf1_loss"] + stats["qf2_loss"]

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, stats["log_pi"], _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        stats["policy_loss"] = (
            (self.alpha * stats["log_pi"]) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        stats["policy_loss"].backward()
        self.policy_optim.step()

        if self.cfg.optim.entropy.auto_tuning:
            stats["alpha_loss"] = -(
                self.log_alpha * (stats["log_pi"] + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            stats["alpha_loss"].backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            stats["alpha_tlogs"] = self.alpha.clone()  # For TensorboardX logs
        else:
            stats["alpha_loss"] = torch.tensor(0.0).to(self.device)
            stats["alpha_tlogs"] = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.cfg.optim.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.cfg.optim.tau)

        stats["log_pi"] = stats["log_pi"].mean()
        for k, v in stats.items():
            stats[k] = v.item()
        return stats

    def train_one_epoch(self, replay_buffer):
        self.train()
        stats = {}
        for i in range(self.cfg.optim.num_updates_per_epoch):
            out = self.update_parameters(replay_buffer, self.cfg.optim.batch_size, i)
            for k, v in out.items():
                if k not in stats:
                    stats[k] = 0.0
                stats[k] += v
        for k, v in stats.items():
            stats[k] = v / self.cfg.optim.num_updates_per_epoch
        stats["updates"] = self.cfg.optim.num_updates_per_epoch
        return stats

    # Save model parameters
    def save_checkpoint(self, logs_dir, epoch):
        save_dir = os.path.join(logs_dir, "agent")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
        log.info("Saving agent to {}".format(save_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            save_path,
        )

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.policy.train()
        self.critic.train()
        self.critic_target.train()

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
