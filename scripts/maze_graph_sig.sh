#!/bin/bash

python main.py \
    +exp=maze_rnet_sym \
    main.suffix='undirected-rs\${replay_buffer.reward_scaling}-gamma\${sac.optim.gamma}' \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/logs/20220614-090425_maze_U4rooms_rnet_sym_new_s234_r0' \
    main.reward=graph_sig \
    main.edge_transitions=True \
    main.subgoal_transitions=True \
    main.seed=234,123,345 \
    replay_buffer.reward_scaling=0.1,0.5 \
    plot.type=wandb \
    main.reward_sigm_temp=1 \
    sac.optim.lr=0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.05 \
    sac.optim.gamma=0.9 \
    eval.interval_epochs=10 \
    hydra.launcher.partition=devlab \
    --multirun \
