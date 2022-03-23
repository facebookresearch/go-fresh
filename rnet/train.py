import torch

def main(cfg, model, dataset, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay)
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    for epoch in range(cfg.num_epochs):
        stats = {'rnet_loss': 0.0, 'rnet_acc': 0.0, 'num_pairs': 0}
        for data in dataloader:
            obs1, obs2, labels = data
            obs1 = obs1.to(device)
            obs2 = obs2.to(device)
            labels = labels.to(device).float()

            optim.zero_grad()
            outputs = model(obs1.float(), obs2.float(),
                    batchwise=True)[:, 0]
            preds = outputs > 0
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            stats['num_pairs'] += labels.shape[0]
            stats['rnet_loss'] += loss.item() * labels.shape[0]
            stats['rnet_acc'] += torch.sum(preds == labels.data)

        for k in stats:
            if k == 'num_pairs':
                continue
            stats[k] /= stats['num_pairs']
        print("rnet epoch {} - loss {:.2f} - acc {:.2f}".format(epoch,
            stats['rnet_loss'], stats['rnet_acc']))
    model.eval()
    return stats
