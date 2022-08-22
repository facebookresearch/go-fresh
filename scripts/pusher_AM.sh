#!/bin/bash

python main.py \
    +env=pusher \
    main.suffix="baseline-rs\${replay_buffer.reward_scaling}-\${replay_buffer.algo}-lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward" \
    main.reward=act_model \
    replay_buffer.algo=AM,HER,HERu \
    replay_buffer.reward_scaling=1,10 \
    sac.optim.lr=0.001,0.0001,0.0005 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.01,0.001,0.005 \
    sac.optim.gamma=0.95,0.99 \
    replay_buffer.num_procs=20 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    main.seed=234,123,345 \
    plot.type=wandb \
    hydra.launcher.partition=learnlab \
    --multirun \
