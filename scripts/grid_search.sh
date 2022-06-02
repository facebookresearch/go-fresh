#!/bin/bash

python main.py \
    +exp=walker_rnet_thresh \
    main.suffix="unsup-grid-vec-rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${train.goal_strat}-edge-trans\${main.edge_transitions}-subgoal-trans\${main.subgoal_transitions}-with-velo_rb1M" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/trained_rnet/walker_memonly_thresh2_neg0_skip0.05_new-graph' \
    rnet.model.hidden_size=128 \
    rnet.model.comp_n_layers=4 \
    rnet.model.feat_size=128 \
    env.action_repeat=2 \
    main.reward=graph_sig \
    replay_buffer.capacity=1000000 \
    replay_buffer.num_procs=20 \
    main.edge_transitions=True \
    sac.policy.head.remove_velocity=False \
    plot.type=wandb \
    main.seed=234,123,345 \
    train.goal_strat=rb \
    main.subgoal_transitions=True \
    main.reward_sigm_temp=1 \
    eval.interval_epochs=20 \
    sac.optim.batch_size=2048 \
    replay_buffer.reward_scaling=0.3,0.5,0.1 \
    sac.optim.lr=0.0001,0.00005,0.0003 \
    sac.optim.entropy.alpha=0.001,0.005,0.01 \
    sac.optim.gamma=0.9,0.95 \
    hydra.launcher.cpus_per_task=20 \
    --multirun \
