import os
import torch
import random
import numpy as np

from tqdm import tqdm

from rnet.memory import RNetMemory
from envs.maze_utils import oracle_distance

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

def compute_NN(exploration_buffer, model, memory, device):
    model.eval()
    num_trajs, traj_len = len(exploration_buffer), exploration_buffer.traj_len
    embs = torch.zeros((num_trajs, traj_len, model.feat_size))
    for traj_idx in tqdm(range(num_trajs), desc="embed exploration buffer"):
        traj = exploration_buffer.get_traj(traj_idx)['obs']
        traj = torch.from_numpy(traj).float().to(device)
        with torch.no_grad():
            embs[traj_idx] = model.get_embedding(traj)

    memory.embs = memory.embs.to(device)
    NN = np.zeros((num_trajs, traj_len), dtype=int)
    skip = memory.cfg.skip
    for traj_idx in tqdm(range(num_trajs), desc="computing NN"):
        for i in range(0, traj_len, skip):
            j = i + skip // 2 if i + skip // 2 < traj_len else i
            NN[traj_idx][i:i + skip] = memory.get_NN(model,
                    embs[traj_idx][j].to(device))[0]
    return NN

def oracle_reward(x1, x2):
    return - oracle_distance(x1, x2)

def fill_replay_buffer(replay_buffer, exploration_buffer, oracle=False,
        NN=None, graph_dist=None):
    print("filling replay buffer")
    while not replay_buffer.is_full():
        g_obs, g1, g2 = exploration_buffer.get_random_obs()
        s_obs, s1, s2 = exploration_buffer.get_random_obs()
        state = {'obs': s_obs, 'goal_obs': g_obs}
        next_state = {'obs': exploration_buffer.obss[s1][s2 + 1], 'goal_obs':
                g_obs}
        if oracle:
            reward = oracle_reward(exploration_buffer.states[s1][s2 + 1],
                    exploration_buffer.states[g1][g2])
        else:
            reward = - graph_dist[NN[s1, s2+1], NN[g1, g2]]
        replay_buffer.push(state, exploration_buffer.actions[s1][s2 + 1],
                reward, next_state)

def save(save_dir, model, memory, NN):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving rnet objects to ', save_dir)

    model_path = os.path.join(save_dir, 'model.pth')
    model.save(model_path)

    memory_path = os.path.join(save_dir, 'memory.npy')
    memory.save(memory_path)

    NN_path = os.path.join(save_dir, 'NN.npy')
    np.save(NN_path, NN)

def load(save_dir, memory, model=None):
    print('Loading rnet objects from ', save_dir)
    if not model is None:
        model_path = os.path.join(save_dir, 'model.pth')
        model.load(model_path)

    memory_path = os.path.join(save_dir, 'memory.npy')
    memory.load(memory_path)

    NN_path = os.path.join(save_dir, 'NN.npy')
    NN = np.load(NN_path)

    if model is None:
        return memory, NN
    return memory, NN, model
