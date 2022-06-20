#!/bin/bash

python main.py \
    +exp=walker_rnet_thresh \
    main.suffix="unsup-rnet_rewards-final-vec-with-velo_rb1M" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/trained_rnet/walker_memonly_thresh2_neg0_skip0.05_new-graph' \
    rnet.model.hidden_size=128 \
    rnet.model.comp_n_layers=4 \
    rnet.model.feat_size=128 \
    env.action_repeat=2 \
    main.reward=rnet \
    replay_buffer.capacity=1000000 \
    replay_buffer.num_procs=20 \
    sac.policy.head.remove_velocity=False \
    plot.type=wandb \
    main.seed=234,123,345 \
    train.goal_strat=rb \
    eval.interval_epochs=20 \
    sac.optim.batch_size=2048 \
    replay_buffer.reward_scaling=0.5 \
    sac.optim.lr=0.0003 \
    sac.optim.entropy.alpha=0.01 \
    sac.optim.gamma=0.95 \
    hydra.launcher.cpus_per_task=20 \
    optim.num_epochs=5001 \
    hydra.launcher.partition=learnlab \
    --multirun \