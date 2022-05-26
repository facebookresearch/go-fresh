import os
import torch
import numpy as np

from tqdm import tqdm

from rnet.memory import RNetMemory


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


def build_memory(cfg, space_info, model, expl_buffer, device):
    model.eval()
    memory = RNetMemory(cfg, space_info, model.feat_size, device)

    x = torch.from_numpy(expl_buffer.get_obs(0, 0)).to(device)
    memory.add_first_obs(model, x, expl_buffer.get_state(0, 0))

    for traj_idx in tqdm(range(len(expl_buffer)), desc="Updating Memory"):
        if np.random.random() > cfg.skip_traj:
            continue
        prev_NNi, prev_NNo = -1, -1
        traj = expl_buffer.get_traj(traj_idx)
        for i in range(0, expl_buffer.traj_len, cfg.skip):
            e = expl_buffer.embs[traj_idx, i].unsqueeze(0)
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


def embed_expl_buffer(expl_buffer, model, device):
    model.eval()
    num_trajs, traj_len = len(expl_buffer), expl_buffer.traj_len
    embs = torch.zeros((num_trajs, traj_len, model.feat_size), device=device)
    for traj_idx in tqdm(range(num_trajs), desc="embed exploration buffer"):
        traj = expl_buffer.get_traj(traj_idx)["obs"]
        traj = torch.from_numpy(traj).float().to(device)
        with torch.no_grad():
            embs[traj_idx] = model.get_embedding(traj)
    return embs.cpu()


def compute_NN(embs, model, memory, device):
    num_trajs, traj_len = embs.shape[0], embs.shape[1]
    memory.embs = memory.embs.to(device)
    NN = {"outgoing": np.zeros((num_trajs, traj_len), dtype=int)}
    if memory.cfg.directed:
        NN["incoming"] = np.zeros((num_trajs, traj_len), dtype=int)
    bsz = memory.cfg.NN_batch_size
    for traj_idx in tqdm(range(num_trajs), desc="computing NN"):
        for i in range(0, traj_len, bsz):
            j = min(i + bsz, traj_len)
            NN["outgoing"][traj_idx, i:j] = memory.get_batch_NN(
                model, embs[traj_idx, i:j]
            )
            if memory.cfg.directed:
                NN["incoming"][traj_idx, i:j] = memory.get_batch_NN(
                    model, embs[traj_idx, i:j], incoming_dir=True
                )
    return NN


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
