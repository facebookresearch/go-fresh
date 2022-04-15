#!/bin/bash

python main.py \
    +env=maze_U4rooms \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-oracle_reward-check" \
    replay_buffer.reward_scaling=0.01 \
    sac.optim.lr=0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.08 \
    sac.optim.gamma=0.95 \
    main.oracle_reward=True \
    #--multirun \
