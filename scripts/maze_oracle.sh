#!/bin/bash

python main.py \
    +exp=maze_rnet_sym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-rnet-sym" \
    main.reward=rnet \
    replay_buffer.reward_scaling=0.01,0.1,1.0 \
    sac.optim.lr=0.0003,0.0001,0.0006 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.08,0.05,0.03 \
    sac.optim.gamma=0.95,0.9 \
    --multirun \
