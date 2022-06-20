import os
import tqdm
import torch


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


def embed_expl_buffer(expl_buffer, model, device):
    model.eval()
    num_trajs, traj_len = len(expl_buffer), expl_buffer.traj_len
    embs = torch.zeros((num_trajs, traj_len, model.feat_size), device=device)
    for traj_idx in tqdm.tqdm(range(num_trajs), desc="embed exploration buffer"):
        traj = expl_buffer.get_traj(traj_idx)["obs"]
        traj = torch.from_numpy(traj).float().to(device)
        with torch.no_grad():
            embs[traj_idx] = model.get_embedding(traj)
    return embs.cpu()


def save(save_dir, model, memory):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving rnet objects to ", save_dir)

    model_path = os.path.join(save_dir, "model.pth")
    model.save(model_path)

    memory_path = os.path.join(save_dir, "memory.npy")
    memory.save(memory_path)


def load(save_dir, memory, model):
    print("Loading rnet objects from ", save_dir)
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

    return memory, model
