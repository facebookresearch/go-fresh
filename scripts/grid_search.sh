#!/bin/bash

python main.py \
    +env=maze_U4rooms \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-grid_search" \
    replay_buffer.reward_scaling=0.01,0.1,0.001 \
    sac.optim.lr=0.00001,0.0001,0.0005,0.001,0.005 \
    sac.optim.batch_size=1024,2048,512 \
    sac.optim.num_updates_per_epoch=1000,2000,10000 \
    sac.optim.entropy.alpha=0.01,0.05,0.001,0.0001 \
    sac.optim.gamma=0.99,0.95,0.9 \
    --multirun \
