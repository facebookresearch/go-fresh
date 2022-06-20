#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rnet-rewards-rs\${replay_buffer.reward_scaling}" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/logs/20220530-051512_maze_U4rooms_rs0.01-lr0.001-bs1024-updates1000-alpha0.01-gamma0.99-graph_sig-reward-temp-1.0-subgoalFalse-asym-fix-seed_cudnn_s234_r0/' \
    main.reward=rnet \
    main.seed=234,123,345 \
    replay_buffer.reward_scaling=0.05,0.1,0.5 \
    plot.type=wandb \
    main.reward_sigm_temp=1 \
    sac.optim.lr=0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.05 \
    sac.optim.gamma=0.95 \
    eval.interval_epochs=10 \
    hydra.launcher.partition=devlab \
    --multirun \
