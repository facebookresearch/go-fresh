import os
import torch
import random
import numpy as np

from tqdm import tqdm

from rnet.memory import RNetMemory
from envs import maze_utils, walker_utils, make_env

walker_goals = {8: (8290, 414), 10: (7079, 198)}


def train(cfg, model, dataset, device, tb_log=None):
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    dataloader = {
        "train": torch.utils.data.DataLoader(
            dataset.train,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers
        ),
        "val": torch.utils.data.DataLoader(
            dataset.val,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers
        ),
    }
    stats = {}
    for epoch in range(cfg.num_epochs):
        stats[epoch] = {}
        to_print = f"rnet epoch {epoch}"
        for phase in ["train", "val"]:
            stats[epoch][phase] = {"rnet_loss": 0.0, "rnet_acc": 0.0, "num_pairs": 0}
            if phase == "train":
                model.train()
            else:
                model.eval()
            for data in dataloader[phase]:
                obs1, obs2, labels = data
                obs1 = obs1.to(device)
                obs2 = obs2.to(device)
                labels = labels.to(device).float()

                if phase == "train":
                    optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(obs1.float(), obs2.float(), batchwise=True)[:, 0]
                preds = outputs > 0
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optim.step()

                stats[epoch][phase]["num_pairs"] += labels.shape[0]
                stats[epoch][phase]["rnet_loss"] += loss.item() * labels.shape[0]
                stats[epoch][phase]["rnet_acc"] += torch.sum(
                    preds == labels.data
                ).item()

            for k in stats[epoch][phase]:
                if k == "num_pairs":
                    continue
                stats[epoch][phase][k] /= stats[epoch][phase]["num_pairs"]
            to_print += " - {} loss {:.2f},  acc {:.2f}".format(
                phase, stats[epoch][phase]["rnet_loss"], stats[epoch][phase]["rnet_acc"]
            )
        print(to_print)
        if tb_log is not None:
            tb_log.add_stats(stats[epoch]["train"], epoch, "rnet/train")
            tb_log.add_stats(stats[epoch]["val"], epoch, "rnet/val")
    model.eval()
    return stats


def build_memory(cfg, explr_embs, space_info, model, exploration_buffer, device):
    model.eval()
    memory = RNetMemory(cfg, space_info, model.feat_size, device)

    x = torch.from_numpy(exploration_buffer.get_obs(0, 0)).to(device)
    memory.add_first_obs(model, x, exploration_buffer.get_state(0, 0))

    for traj_idx in tqdm(range(len(exploration_buffer)), desc="Updating Memory"):
        if random.random() > cfg.skip_traj:
            continue
        prev_NNi, prev_NNo = -1, -1
        traj = exploration_buffer.get_traj(traj_idx)
        for i in range(0, exploration_buffer.traj_len, cfg.skip):
            e = explr_embs[traj_idx, i].unsqueeze(0)
            novelty_o, NNo = memory.compute_novelty(model, e)
            if cfg.directed:
                novelty_i, NNi = memory.compute_novelty(model, e, incoming_dir=True)
            else:
                novelty_i, NNi = novelty_o, NNo

            if novelty_o > 0 and novelty_i > 0:
                x = torch.from_numpy(traj["obs"][i]).to(device)
                NN = memory.add(x, traj["state"][i], e)
                NNi, NNo = NN, NN
            memory.transition2edges(prev_NNi, prev_NNo, NNi, NNo)
            prev_NNi, prev_NNo = NNi, NNo

    memory.adj_matrix = memory.adj_matrix[:len(memory), :len(memory)]

    # make sure any memory node can be reached from any other
    memory.connect_graph(model)

    memory.compute_dist()
    return memory


def embed_expl_buffer(exploration_buffer, model, device):
    model.eval()
    num_trajs, traj_len = len(exploration_buffer), exploration_buffer.traj_len
    embs = torch.zeros((num_trajs, traj_len, model.feat_size), device=device)
    for traj_idx in tqdm(range(num_trajs), desc="embed exploration buffer"):
        traj = exploration_buffer.get_traj(traj_idx)["obs"]
        traj = torch.from_numpy(traj).float().to(device)
        with torch.no_grad():
            embs[traj_idx] = model.get_embedding(traj)
    return embs


def compute_NN(explr_embs, model, memory, device):
    num_trajs, traj_len = explr_embs.size()[:2]
    memory.embs = memory.embs.to(device)
    NN = {"outgoing": np.zeros((num_trajs, traj_len), dtype=int)}
    if memory.cfg.directed:
        NN["incoming"] = np.zeros((num_trajs, traj_len), dtype=int)
    bsz = memory.cfg.NN_batch_size
    for traj_idx in tqdm(range(num_trajs), desc="computing NN"):
        for i in range(0, traj_len, bsz):
            j = min(i + bsz, traj_len)
            NN["outgoing"][traj_idx, i:j] = memory.get_batch_NN(
                model, explr_embs[traj_idx, i:j]
            )
            if memory.cfg.directed:
                NN["incoming"][traj_idx, i:j] = memory.get_batch_NN(
                    model, explr_embs[traj_idx, i:j], incoming_dir=True
                )
    return NN


def get_eval_goals(cfg, memory, space_info, rnet_model, device):
    eval_goals = {x: None for x in ["obs", "state", "embs", "NN"]}
    env = make_env(cfg.env, space_info)
    eval_goals["obs"] = np.float32(env.get_goals()[f"{cfg.env.obs.type}_obs"])
    eval_goals["states"] = np.expand_dims(env.get_goals()["state"], 0)
    if cfg.main.reward in ["rnet", "graph", "graph_sig"]:
        goal_obs_pt = torch.from_numpy(eval_goals["obs"]).to(device)
        eval_goals["embs"] = rnet_model.get_embedding(goal_obs_pt)  # ngoals x emb_dim
        eval_goals["embs"] = eval_goals["embs"].unsqueeze(0)
        if cfg.main.reward in ["graph", "graph_sig"]:
            eval_goals["NN"] = compute_NN(
                eval_goals["embs"], rnet_model, memory, device
            )
    return eval_goals


def oracle_reward(cfg, x1, x2):
    if cfg.env.id == 'maze_U4rooms':
        oracle_distance = maze_utils.oracle_distance
    elif cfg.env.id == 'walker':
        oracle_distance = walker_utils.oracle_distance
    else:
        raise ValueError()

    if cfg.main.reward == "oracle_sparse":
        if oracle_distance(x1, x2) < cfg.env.success_thresh:
            return 0
        return -1
    elif cfg.main.reward == "oracle_dense":
        return -oracle_distance(x1, x2)
    else:
        raise ValueError()


def fill_replay_buffer(
    replay_buffer,
    exploration_buffer,
    cfg,
    NN=None,
    graph_dist=None,
    rnet_model=None,
    explr_embs=None,
    eval_goals=None
):
    replay_buffer.to("cpu")

    if cfg.main.reward in ["rnet", "graph_sig"]:
        # will compute rewards in parallel for efficiency
        assert len(replay_buffer) == 0
        s_emb, g_emb = [], []

    while not replay_buffer.is_full():
        if cfg.train.goal_strat == "rb":
            g_obs, g1, g2 = exploration_buffer.get_random_obs()
            goal_embs = explr_embs
            goal_states = exploration_buffer.states
            goal_NN = NN
        else:
            if cfg.train.goal_strat == "one_goal":
                goal_idx = cfg.eval.goal_idx
            elif cfg.train.goal_strat == "all_goal":
                ngoals = np.size(eval_goals["obs"], 0)
                goal_idx = np.random.randint(ngoals)
            else:
                raise ValueError(f"invalid goal_strat: {cfg.train.goal_strat}")
            g1, g2 = 0, goal_idx
            g_obs = eval_goals["obs"][goal_idx]
            goal_embs = eval_goals["embs"]
            goal_NN = eval_goals["NN"]
            goal_states = eval_goals["states"]
        s_obs, s1, s2 = exploration_buffer.get_random_obs()
        state = {"obs": s_obs, "goal_obs": g_obs}
        next_state = {"obs": exploration_buffer.obss[s1, s2 + 1], "goal_obs": g_obs}
        if cfg.main.reward in ["oracle_dense", "oracle_sparse"]:
            reward = oracle_reward(
                cfg, exploration_buffer.states[s1, s2 + 1], goal_states[g1, g2]
            )
        if cfg.main.reward in ["rnet", "graph_sig"]:
            s_emb.append(explr_embs[s1, s2 + 1])
            g_emb.append(goal_embs[g1, g2])
            reward = 0  # will compute it later in parallel
        if cfg.main.reward in ["graph", "graph_sig"]:
            s_NN = NN["outgoing"][s1, s2 + 1]
            if cfg.rnet.memory.directed:
                g_NN = goal_NN["incoming"][g1, g2]
            else:
                g_NN = goal_NN["outgoing"][g1, g2]
            reward = -graph_dist[s_NN, g_NN]
        replay_buffer.push(
            state, exploration_buffer.actions[s1, s2 + 1], reward, next_state
        )

    replay_buffer.to(replay_buffer.device)

    if cfg.main.reward in ["rnet", "graph_sig"]:
        assert replay_buffer.is_full()
        s_emb = torch.stack(s_emb)
        g_emb = torch.stack(g_emb)
        with torch.no_grad():
            rval = rnet_model.compare_embeddings(s_emb, g_emb, batchwise=True)
        rewards = rval[:, 0]
        assert rewards.size(0) == len(replay_buffer)
        if cfg.main.reward == "graph_sig":
            rewards = torch.sigmoid(rewards / cfg.main.reward_sigm_temp) - 1
        rewards *= cfg.replay_buffer.reward_scaling
        rewards += replay_buffer.rewards[:, 0]
        replay_buffer.rewards[:, 0].copy_(rewards)


def save(save_dir, model, memory, NN):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving rnet objects to ", save_dir)

    model_path = os.path.join(save_dir, "model.pth")
    model.save(model_path)

    memory_path = os.path.join(save_dir, "memory.npy")
    memory.save(memory_path)

    NN_path = os.path.join(save_dir, "NN.npz")
    np.savez(NN_path, **NN)


def load(save_dir, memory, model=None):
    print("Loading rnet objects from ", save_dir)
    if model is not None:
        model_path = os.path.join(save_dir, "model.pth")
        if os.path.exists(model_path):
            model.load(model_path)
        else:
            print("model path not found")

    memory_path = os.path.join(save_dir, "memory.npy")
    if os.path.exists(memory_path):
        memory.load(memory_path)
    else:
        print("memory path not found")

    NN_path = os.path.join(save_dir, "NN.npz")
    if os.path.exists(NN_path):
        NN = np.load(NN_path)
    else:
        print("NN path not found")
        NN = None

    if model is None:
        return memory, NN
    return memory, NN, model
