#!/bin/bash

python main.py \
    +env=pusher \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-debug" \
    main.reward=oracle_dense \
    replay_buffer.reward_scaling=1.0 \
    sac.optim.lr=0.0003,0.0001,0.0006 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.08,0.05,0.03 \
    sac.optim.gamma=0.95,0.9,0.99 \
    replay_buffer.num_procs=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
