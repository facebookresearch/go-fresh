from os import path, system
import pathlib
import torch
import hydra
import logging

from . import eval

from .sac import SAC
from .envs import make_env
from .logger import Logger
from .replay_buffer import ReplayBuffer
from .utils import fix_seed, get_space_info
from .exploration_buffer import ExplorationBuffer
from .replay_buffer_filler import ReplayBufferFiller
from .rnet import rnet_utils, RNetMemory, RNetModel, RNetPairsSplitDataset


log = logging.getLogger(__name__)


def train_rnet(cfg, model, expl_buffer, vis_log, device):
    dataset = RNetPairsSplitDataset(cfg.rnet.dataset, expl_buffer)
    rnet_utils.train(cfg.rnet.train, model, dataset, device, vis_log)


def train_policy(
    cfg,
    expl_buffer,
    rnet_model,
    memory,
    space_info,
    device,
    vis_log,
):
    env = make_env(cfg.env, space_info)
    uniform_action_fn = env.action_space.sample
    env.close()

    replay_buffer = ReplayBuffer(cfg.replay_buffer, space_info)
    agent = SAC(cfg.sac, space_info, device)
    replay_buffer_filler = ReplayBufferFiller(
        replay_buffer,
        expl_buffer,
        cfg,
        space_info,
        device,
        memory,
        rnet_model,
        agent=agent,
        uniform_action_fn=uniform_action_fn
    )

    procs, buffers, barriers, n_eval_done, info_keys = eval.start_procs(cfg, space_info)

    num_updates = 0
    for epoch in range(cfg.optim.num_epochs):
        log.info(f"epoch: {epoch}")

        # TRAIN
        replay_buffer.to("cpu")
        log.info("filling replay buffer")
        replay_buffer_filler.run()

        log.info("train one epoch")
        replay_buffer.to(device)
        train_stats = agent.train_one_epoch(replay_buffer)
        log.info(
            "train " + " - ".join([f"{k}: {v:.2f}" for k, v in train_stats.items()])
        )
        num_updates += train_stats["updates"]
        train_stats["updates"] = num_updates
        vis_log.add_stats(train_stats, epoch, "train")

        # EVAL
        if epoch % cfg.eval.interval_epochs == 0:
            eval_stats = eval.run(
                agent,
                cfg.eval.num_episodes,
                buffers,
                barriers,
                n_eval_done,
                info_keys,
                rnet_model,
                device
            )
            log.info(
                "eval " + " - ".join([f"{k}: {v:.2f}" for k, v in eval_stats.items()])
            )
            vis_log.add_stats(eval_stats, epoch, "eval")

        if epoch % cfg.main.save_interval == 0:
            agent.save_checkpoint(cfg.main.logs_dir, epoch)

    for p in procs:
        p.join()


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg):
    cfg.main.cwd = hydra.utils.get_original_cwd()
    logs_dir = pathlib.Path(cfg.main.cwd) / "logs" / cfg.main.name
    cfg.main.logs_dir = logs_dir.as_posix()
    logs_dir.mkdir(exist_ok=True)
    vis_log = Logger(cfg)
    log.info(f"exp name: {cfg.main.name}")

    fix_seed(cfg.main.seed)

    # setup paths and load
    rnet_path = logs_dir / "model.pth"
    embs_path = logs_dir / "embs.pth"
    memory_path = logs_dir / "memory.npy"
    if cfg.main.load_from_dir is not None:
        for file in ["model.pth", "memory.npy", "embs.pth"]:
            load_path = path.join(cfg.main.load_from_dir, file)
            if path.exists(load_path):
                log.info(f"copying from {load_path}")
                system(f"cp {load_path} {logs_dir}/")

    device = torch.device("cuda")
    space_info = get_space_info(cfg.env.obs, cfg.env.action_dim)
    expl_buffer = ExplorationBuffer(cfg)

    if cfg.main.reward in ["rnet", "graph", "graph_sig"]:
        # RNet
        rnet_model = RNetModel(cfg.rnet.model, space_info).to(device)
        log.info(rnet_model)
        if path.exists(rnet_path):
            log.info(f"Loading RNet from {rnet_path}")
            rnet_model.load(rnet_path, device=device)
        else:
            log.info("Training RNet")
            train_rnet(cfg, rnet_model, expl_buffer, vis_log, device)
            log.info(f"Saving RNet to {rnet_path}")
            rnet_model.save(rnet_path)
    else:
        rnet_model = None

    if cfg.main.train_until == "rnet":
        vis_log.close()
        return

    if cfg.main.reward in ["rnet", "graph", "graph_sig"]:
        # Exploration buffer embeddings
        if path.exists(embs_path):
            log.info(f"Loading embeddings from {embs_path}")
            embs = torch.load(embs_path)
        else:
            log.info("Embedding exploration_buffer")
            embs = rnet_utils.embed_expl_buffer(expl_buffer, rnet_model, device)
            torch.save(embs, embs_path)
        expl_buffer.set_embs(embs)

        # Memory and graph
        memory = RNetMemory(cfg.rnet.memory, space_info, rnet_model.feat_size, device)
        if path.exists(memory_path):
            log.info(f"Loading memory from {memory_path}")
            memory.load(memory_path)
        else:
            log.info("Training memory")
            memory.build(rnet_model, expl_buffer)
        if cfg.main.reward in ["graph", "graph_sig"]:
            # Nearest neigbhor
            if memory.nn_out is None:
                log.info("Computing NN")
                nn = memory.compute_NN(expl_buffer.embs, rnet_model)
                expl_buffer.embs = expl_buffer.embs.to("cpu")
                memory.set_nn(nn)
            if memory.edge2rb is None:
                log.info("Computing graph")
                memory.compute_edges(rnet_model)
        memory.embs = memory.embs.to("cpu")
        memory.save(memory_path)
        log.info(f"Memory size: {len(memory)}")
        log.info(
            f"Number of connected components: {memory.get_nb_connected_components()}"
        )
    else:
        memory = None

    if cfg.main.train_until == "memory":
        vis_log.close()
        return

    # Policy
    log.info("Training policy")
    train_policy(
        cfg=cfg,
        expl_buffer=expl_buffer,
        rnet_model=rnet_model,
        memory=memory,
        space_info=space_info,
        device=device,
        vis_log=vis_log,
    )

    vis_log.close()


if __name__ == "__main__":
    main()
