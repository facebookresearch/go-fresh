#!/bin/bash

python main.py \
    +exp=walker_rnet_thresh \
    main.suffix="\${main.reward}-reward-rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-unsup-graph_sig-temp\${main.reward_sigm_temp}-subgoal\${main.subgoal_transitions}-\${train.goal_strat}" \
    main.load_from_dir='/checkpoint/sainbar/offline-gcrl/logs/20220509-173335_walker_memonly_thresh2_neg0_skip0.05' \
    rnet.model.remove_velocity=True \
    rnet.model.hidden_size=128 \
    rnet.model.comp_n_layers=4 \
    rnet.model.feat_size=128 \
    plot.type=wandb \
    main.reward=graph_sig \
    train.goal_strat=rb,memory_bins \
    main.subgoal_transitions=True \
    main.reward_sigm_temp=0.1,0.5,1,3,5 \
    env.action_repeat=2 \
    eval.interval_epochs=20 \
    sac.optim.batch_size=2048,4096 \
    replay_buffer.reward_scaling=0.3,0.8,1.0,2.0 \
    sac.optim.lr=0.0001 \
    sac.optim.entropy.alpha=0.005,0.001,0.0001,0.01 \
    sac.optim.gamma=0.9 \
    --multirun \
