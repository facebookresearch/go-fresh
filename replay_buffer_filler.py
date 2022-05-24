import torch
import numpy as np

from torch import multiprocessing as mp

from envs import make_env, oracle_reward
from rnet.utils import compute_NN


class ReplayBufferFiller:
    def __init__(
        self,
        replay_buffer,
        expl_buffer,
        cfg,
        space_info,
        device,
        memory=None,
        NN=None,
        rnet_model=None,
    ):
        self.replay_buffer = replay_buffer
        self.expl_buffer = expl_buffer
        self.cfg = cfg
        self.space_info = space_info
        self.device = device
        self.memory = memory
        self.NN = NN
        self.rnet_model = rnet_model
        if cfg.train.goal_strat == "memory_bins":
            self.NN_dict = self.compute_NN_dict()
        elif cfg.train.goal_strat in ["one_goal", "all_goal"]:
            self.eval_goals = self.get_eval_goals()
        self.epoch = 0

    def compute_NN_dict(self):
        if self.cfg.rnet.memory.directed:
            NN_array = self.NN["incoming"]
        else:
            NN_array = self.NN["outgoing"]
        NN_dict = {x: [] for x in np.arange(len(self.memory))}
        for traj_idx in range(NN_array.shape[0]):
            for step in range(NN_array.shape[1]):
                NN_dict[NN_array[traj_idx, step]].append((traj_idx, step))
        return NN_dict

    def get_eval_goals(self):
        eval_goals = {x: None for x in ["obs", "state", "embs", "NN"]}
        env = make_env(self.cfg.env, self.space_info)
        eval_goals["obs"] = np.float32(env.get_goals()[f"{self.cfg.env.obs.type}_obs"])
        eval_goals["states"] = env.get_goals()["state"]
        if self.cfg.main.reward in ["rnet", "graph", "graph_sig"]:
            goal_obs_pt = torch.from_numpy(eval_goals["obs"]).to(self.device)
            eval_goals["embs"] = self.rnet_model.get_embedding(goal_obs_pt)
            if self.cfg.main.reward in ["graph", "graph_sig"]:
                eval_goals["NN"] = compute_NN(
                    eval_goals["embs"].unsqueeze(0),
                    self.rnet_model,
                    self.memory,
                    self.device
                )
        return eval_goals

    def get_goal_rb(self, g1, g2, g_obs):
        g_emb = self.expl_buffer.embs[g1, g2]
        g_state = self.expl_buffer.states[g1, g2]
        if self.cfg.rnet.memory.directed:
            g_NN = self.NN["incoming"][g1, g2]
        else:
            g_NN = self.NN["outgoing"][g1, g2]
        return g_obs, g_NN, g_emb, g_state

    def sample_goal_rb(self):
        g_obs, g1, g2 = self.expl_buffer.get_random_obs()
        return self.get_goal_rb(g1, g2, g_obs)

    def sample_goal_eval(self):
        if self.cfg.train.goal_strat == "one_goal":
            goal_idx = self.cfg.eval.goal_idx
        elif self.cfg.train.goal_strat == "all_goal":
            ngoals = np.size(self.eval_goals["obs"], 0)
            goal_idx = np.random.randint(ngoals)
        else:
            raise ValueError(f"invalid goal_strat: {self.cfg.train.goal_strat}")
        g_obs = self.eval_goals["obs"][goal_idx]
        g_emb = self.eval_goals["embs"][goal_idx]
        g_state = self.eval_goals["states"][goal_idx]
        if self.cfg.rnet.memory.directed:
            g_NN = self.eval_goals["NN"]["incoming"][0, goal_idx]
        else:
            g_NN = self.eval_goals["NN"]["outgoing"][0, goal_idx]
        return g_obs, g_NN, g_emb, g_state

    def sample_goal_memory(self):
        goal_idx = np.random.randint(len(self.memory))
        g_obs = self.memory.get_obs(goal_idx)
        g_emb = self.memory.embs[goal_idx]
        g_state = self.memory.states[goal_idx]
        g_NN = goal_idx
        return g_obs, g_NN, g_emb, g_state

    def sample_goal_memory_bins(self):
        goal_idx = np.random.randint(len(self.memory))
        idx = np.random.randint(len(self.NN_dict[goal_idx]))
        g1, g2 = self.NN_dict[goal_idx][idx]
        g_obs = self.expl_buffer.get_obs(g1, g2)
        return self.get_goal_rb(g1, g2, g_obs)

    def sample_goal(self):
        if self.cfg.train.goal_strat == "rb":
            return self.sample_goal_rb()
        elif self.cfg.train.goal_strat == "memory":
            return self.sample_goal_memory()
        elif self.cfg.train.goal_strat == "memory_bins":
            return self.sample_goal_memory_bins()
        else:
            return self.sample_goal_eval()

    def run(self):
        self.epoch += 1
        if self.memory is not None:
            self.memory.obss = self.memory.obss.to("cpu")

        ctx = mp.get_context("fork")
        self.idx = ctx.Value("i", 0)
        self.replay_buffer.share_memory()

        if self.cfg.main.reward in ["rnet", "graph_sig"]:
            # will compute rewards in parallel for efficiency
            self.s_embs = torch.empty(
                (len(self.replay_buffer), self.cfg.rnet.model.feat_size),
                dtype=torch.float32,
            )
            self.g_embs = torch.empty_like(self.s_embs)
            self.s_embs.share_memory_()
            self.g_embs.share_memory_()

        procs = []
        for proc_id in range(self.cfg.replay_buffer.num_procs):
            p = ctx.Process(target=self.worker_fill, args=(proc_id,))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        if self.cfg.main.reward in ["rnet", "graph_sig"]:
            with torch.no_grad():
                self.s_embs = self.s_embs.to(self.device)
                self.g_embs = self.g_embs.to(self.device)
                rval = self.rnet_model.compare_embeddings(
                    self.s_embs, self.g_embs, batchwise=True
                )
            rewards = rval[:, 0]
            assert rewards.size(0) == len(self.replay_buffer)
            if self.cfg.main.reward == "graph_sig":
                rewards = torch.sigmoid(rewards / self.cfg.main.reward_sigm_temp) - 1
                rewards *= self.cfg.main.reward_sigm_weight
            rewards = rewards.cpu()
            rewards *= self.cfg.replay_buffer.reward_scaling
            rewards += self.replay_buffer.rewards[:, 0]
            self.replay_buffer.rewards[:, 0].copy_(rewards)

    def get_safe_i(self):
        with self.idx.get_lock():
            if self.idx.value == len(self.replay_buffer):
                return -1
            i = self.idx.value
            self.idx.value += 1
            return i

    def worker_fill(self, proc_id):
        assert self.cfg.main.seed > self.cfg.replay_buffer.num_procs
        np.random.seed(self.cfg.main.seed * self.epoch + proc_id)
        while True:
            i = self.get_safe_i()
            if i == -1:
                break
            g_obs, g_NN, g_emb, g_state = self.sample_goal()

            s_obs, s1, s2 = self.expl_buffer.get_random_obs(not_last=True)
            next_s_obs = self.expl_buffer.get_obs(s1, s2 + 1)
            if self.cfg.main.reward in ["oracle_dense", "oracle_sparse"]:
                reward = oracle_reward(
                    self.cfg, self.expl_buffer.states[s1, s2 + 1], g_state
                )
            elif self.cfg.main.reward in ["rnet", "graph_sig"]:
                self.s_embs[i] = self.expl_buffer.embs[s1, s2 + 1]
                self.g_embs[i] = g_emb
                reward = 0  # will compute it later in parallel
            if self.cfg.main.reward in ["graph", "graph_sig"]:
                s_NN = self.NN["outgoing"][s1, s2 + 1]
                reward = -self.memory.dist[s_NN, g_NN]
            state = np.stack((s_obs, g_obs))
            next_state = np.stack((self.expl_buffer.get_obs(s1, s2 + 1), g_obs))
            action = self.expl_buffer.actions[s1, s2 + 1]
            self.replay_buffer.write(i, state, action, reward, next_state)

            if self.cfg.main.subgoal_transitions:
                assert self.cfg.main.reward in ["graph", "graph_sig"]
                subgoals = self.memory.retrieve_path(s_NN, g_NN)
                if self.cfg.train.goal_strat == "memory":
                    subgoals = subgoals[:-1]
                for subgoal in subgoals:
                    i = self.get_safe_i()
                    if i == -1:
                        break
                    reward = -self.memory.dist[s_NN, subgoal]
                    subgoal_obs = self.memory.get_obs(subgoal)
                    state = np.stack((s_obs, subgoal_obs))
                    next_state = np.stack((next_s_obs, subgoal_obs))
                    self.replay_buffer.write(i, state, action, reward, next_state)
                    if self.cfg.main.reward == "graph_sig":
                        self.s_embs[i] = self.expl_buffer.embs[s1, s2 + 1]
                        self.g_embs[i] = self.memory.embs[subgoal]
