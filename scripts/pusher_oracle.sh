#!/bin/bash

python main.py \
    +env=pusher \
    main.suffix="lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward" \
    main.reward=oracle_dense \
    replay_buffer.reward_scaling=1.0 \
    sac.optim.lr=0.001,0.0006,0.003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.01,0.03,0.005 \
    sac.optim.gamma=0.95 \
    replay_buffer.num_procs=20 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
