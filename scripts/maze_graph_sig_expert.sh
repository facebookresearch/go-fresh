#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="expert-cheap-rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-temp-\${main.reward_sigm_temp}-subgoal\${main.subgoal_transitions}-asym-fix-seed_cudnn" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/logs/20220613-101123_maze_U4rooms_expert-rs0.01-lr0.001-bs1024-updates1000-alpha0.01-gamma0.99-graph_sig-reward-temp-1.0-subgoalTrue-asym-fix-seed_cudnn_s234_r0' \
    exploration_buffer.data_dir='/checkpoint/linamezghani/offline-gcrl/data/maze_U4rooms/expert'\
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
    sac.optim.gamma=0.9,0.95 \
    eval.interval_epochs=10 \
    hydra.launcher.partition=devlab \
    --multirun \
