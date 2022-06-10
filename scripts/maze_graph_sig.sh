#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-neg-action-\${replay_buffer.neg_action}-neg-goal-\${replay_buffer.neg_goal}-cut100-rb1M" \
    main.reward=act_model \
    replay_buffer.neg_action=null,policy,uniform \
    replay_buffer.neg_goal=zero,critic \
    replay_buffer.reward_scaling=100,200,500 \
    sac.optim.lr=0.0001,0.00005 \
    sac.optim.batch_size=2048 \
    sac.optim.entropy.alpha=0.001,0.01,0.005 \
    sac.optim.gamma=0.95,0.99 \
    optim.num_epochs=5001 \
    eval.interval_epochs=10 \
    replay_buffer.num_procs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    hydra.launcher.cpus_per_task=20 \
    --multirun \
