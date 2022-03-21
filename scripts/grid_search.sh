#!/bin/bash

python main.py \
    +env=maze_U4rooms \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-better_dist-fixed_start_pos-grid_search" \
    exploration_buffer.data_dir="/checkpoint/linamezghani/offline-gcrl/data/maze_U4rooms/random" \
    replay_buffer.reward_scaling=0.01,0.1,0.001 \
    sac.optim.lr=0.0001,0.0003,0.00008 \
    sac.optim.batch_size=1024,2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.03,0.05,0.08 \
    sac.optim.gamma=0.99,0.95 \
    --multirun \
