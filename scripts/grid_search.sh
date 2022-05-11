#!/bin/bash

python main.py \
    +exp=walker_rnet_thresh \
    main.suffix="\${main.reward}-reward-rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-action_repeat\${env.action_repeat}-1goal\${eval.goal_idx}" \
    rnet.model.remove_velocity=True \
    eval.goal_idx=8,10 \
    train.goal_strat='one_goal' \
    main.load_from_dir='/checkpoint/sainbar/offline-gcrl/logs/20220509-173335_walker_memonly_thresh2_neg0_skip0.05' \
    rnet.model.hidden_size=128 \
    rnet.model.comp_n_layers=4 \
    rnet.model.feat_size=128 \
    main.reward=graph,graph_sig,oracle_sparse \
    env.action_repeat=2 \
    replay_buffer.reward_scaling=1.0,0.1,0.01 \
    sac.optim.lr=0.0005,0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.008,0.01 \
    sac.optim.gamma=0.85,0.9,0.95,0.99 \
    --multirun \
