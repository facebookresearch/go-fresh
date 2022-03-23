import torch
import random

from tqdm import tqdm

from rnet.memory import RNetMemory

def train(cfg, model, dataset, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay)
    dataloader = {
            'train': torch.utils.data.DataLoader(dataset.train,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers),
            'val': torch.utils.data.DataLoader(dataset.val,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    }
    stats = {}
    for epoch in range(cfg.num_epochs):
        stats[epoch] = {}
        to_print = f"rnet epoch {epoch}"
        for phase in ['train', 'val']:
            stats[epoch][phase] = {'rnet_loss': 0.0, 'rnet_acc': 0.0,
                    'num_pairs': 0}
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for data in dataloader[phase]:
                obs1, obs2, labels = data
                obs1 = obs1.to(device)
                obs2 = obs2.to(device)
                labels = labels.to(device).float()

                if phase == 'train':
                    optim.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(obs1.float(), obs2.float(),
                            batchwise=True)[:, 0]
                preds = outputs > 0
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optim.step()

                stats[epoch][phase]['num_pairs'] += labels.shape[0]
                stats[epoch][phase]['rnet_loss'] += loss.item() * labels.shape[0]
                stats[epoch][phase]['rnet_acc'] += torch.sum(preds == labels.data)

            for k in stats[epoch][phase]:
                if k == 'num_pairs':
                    continue
                stats[epoch][phase][k] /= stats[epoch][phase]['num_pairs']
            to_print += " - {} loss {:.2f},  acc {:.2f}".format(phase,
                stats[epoch][phase]['rnet_loss'], stats[epoch][phase]['rnet_acc'])
        print(to_print)
    model.eval()
    return stats

def build_memory(cfg, space_info, model, exploration_buffer, device):
    model.eval()
    memory = RNetMemory(cfg, space_info, model.feat_size, device)

    x = torch.from_numpy(exploration_buffer.get_obs(0, 0)).to(device)
    memory.add_first_obs(model, x, exploration_buffer.get_state(0, 0))

    for traj_idx in tqdm(range(len(exploration_buffer)), desc="Updating Memory"):
        if random.random() > cfg.skip_traj:
            continue
        prev_NN = -1
        traj = exploration_buffer.get_traj(traj_idx)
        for i in range(0, exploration_buffer.traj_len, cfg.skip):
            x = torch.from_numpy(traj['obs'][i]).to(device)
            e, novelty, NN = memory.compute_novelty(model, x)
            if novelty > 0:
                NN = memory.add(x, traj['state'][i], e)
            memory.add_edge(NN, prev_NN)
            prev_NN = NN

    memory.adj_matrix = memory.adj_matrix[:len(memory), :len(memory)]
    memory.compute_dist()
    return memory
