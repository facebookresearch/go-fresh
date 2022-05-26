#!/bin/bash

python main.py \
    +exp=walker_rnet_thresh \
    main.suffix="unsup-grid-vec-\${train.goal_strat}-remove_velocity\${sac.policy.head.remove_velocity}-stack\${env.frame_stack}" \
    main.load_from_dir='/checkpoint/sainbar/offline-gcrl/logs/20220509-173335_walker_memonly_thresh2_neg0_skip0.05' \
    rnet.model.hidden_size=128 \
    rnet.model.comp_n_layers=4 \
    rnet.model.feat_size=128 \
    env.action_repeat=2 \
    main.reward=graph_sig \
    sac.policy.head.remove_velocity=True,False \
    env.frame_stack=1,2,3,4 \
    main.run=0,1,2 \
    plot.type=wandb \
    train.goal_strat=rb,memory_bins \
    main.subgoal_transitions=False \
    main.reward_sigm_temp=1 \
    eval.interval_epochs=20 \
    sac.optim.batch_size=2048 \
    replay_buffer.reward_scaling=0.3 \
    sac.optim.lr=0.0001 \
    sac.optim.entropy.alpha=0.001 \
    sac.optim.gamma=0.9 \
    hydra.launcher.partition=devlab \
    --multirun \
