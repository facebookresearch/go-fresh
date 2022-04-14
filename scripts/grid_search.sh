#!/bin/bash

python main.py \
    +env=maze_U4rooms \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-rnet_graph_dist-grid_search" \
    replay_buffer.reward_scaling=0.1 \
    sac.optim.lr=0.0001,0.0003,0.00008 \
    sac.optim.batch_size=1024,2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.05 \
    sac.optim.gamma=0.95 \
    --multirun \
