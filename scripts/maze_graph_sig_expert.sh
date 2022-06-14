#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-neg-action-\${replay_buffer.neg_action}-neg-goal-\${replay_buffer.neg_goal}-expert-rb1M" \
    main.reward=act_model \
    exploration_buffer.data_dir="/checkpoint/linamezghani/offline-gcrl/data/maze_U4rooms/expert" \
    replay_buffer.neg_action=null,policy,uniform \
    replay_buffer.neg_goal=zero,critic \
    replay_buffer.reward_scaling=1,10 \
    sac.optim.lr=0.00005,0.0001 \
    sac.optim.batch_size=2048 \
    sac.optim.entropy.alpha=0.001 \
    sac.optim.gamma=0.99,0.95 \
    optim.num_epochs=5001 \
    eval.interval_epochs=10 \
    replay_buffer.num_procs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    hydra.launcher.cpus_per_task=20 \
    --multirun \
