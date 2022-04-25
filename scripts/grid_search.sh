#!/bin/bash

python main.py \
    +env=walker \
    main.suffix="oracle\${main.oracle_reward}-rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-action_repeat\${env.action_repeat}-1goal-run\${main.run}" \
    train.goal_strat='one_goal' \
    eval.goal_idx=8 \
    env.action_repeat=2 \
    replay_buffer.reward_scaling=1.0 \
    sac.optim.lr=0.0005 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.008 \
    sac.optim.gamma=0.85 \
    #main.run=0,1,2,3,4 \
    #hydra.launcher.partition=devlab \
    #--multirun \
