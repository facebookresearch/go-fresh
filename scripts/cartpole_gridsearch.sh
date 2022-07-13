#!/bin/bash

python main.py \
    +exp=cartpole_rnet \
    main.suffix="unsup_random-search_len\${env.max_episode_steps}_\${exploration_buffer.algo}_rs\${replay_buffer.reward_scaling}_lr\${sac.optim.lr}_alpha\${sac.optim.entropy.alpha}_gamma\${sac.optim.gamma}" \
    main.load_from_dir="/checkpoint/linamezghani/offline-gcrl/logs/20220712-072542_cartpole_rnet-thresh2_\${exploration_buffer.algo}_s234_r0" \
    main.reward=graph_sig \
    replay_buffer.capacity=1000000 \
    replay_buffer.num_procs=20 \
    main.edge_transitions=True \
    sac.policy.head.remove_velocity=False \
    main.subgoal_transitions=True \
    train.goal_strat=rb \
    main.reward_sigm_temp=1 \
    eval.interval_epochs=20 \
    sac.optim.batch_size=2048 \
    replay_buffer.reward_scaling=0.5,1 \
    sac.optim.lr=0.0003,0.0001 \
    sac.optim.entropy.alpha=0.01,0.05,0.005 \
    sac.optim.gamma=0.95,0.99 \
    optim.num_epochs=5001 \
    hydra.launcher.cpus_per_task=20 \
    env.max_episode_steps=1000,100 \
    main.seed=234,123,345 \
    exploration_buffer.algo=proto,random,icm_apt \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
