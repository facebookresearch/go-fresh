#!/bin/bash

python main.py \
    +env=walker \
    main.suffix="baseline-\${replay_buffer.algo}" \
    replay_buffer.algo=AM,HER,HERu \
    main.reward=act_model \
    env.action_repeat=2 \
    main.reward=act_model \
    sac.optim.batch_size=2048 \
    replay_buffer.reward_scaling=10 \
    sac.optim.lr=0.0001 \
    sac.optim.entropy.alpha=0.001 \
    main.seed=234,123,345 \
    sac.optim.gamma=0.99 \
    optim.num_epochs=5001 \
    plot.type=wandb \
    eval.interval_epochs=20 \
    hydra.launcher.cpus_per_task=20 \
    replay_buffer.num_procs=20 \
    hydra.launcher.partition=learnlab \
    --multirun \
