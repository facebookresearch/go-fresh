#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="expert-baseline-\${replay_buffer.algo}-rs\${replay_buffer.reward_scaling}-alpha\${sac.optim.entropy.alpha}" \
    exploration_buffer.data_dir="/checkpoint/linamezghani/offline-gcrl/data/maze_U4rooms/expert" \
    main.reward=act_model \
    replay_buffer.algo=AM,HER,HERu \
    replay_buffer.reward_scaling=0.1,1 \
    sac.optim.lr=0.0001 \
    sac.optim.batch_size=2048 \
    sac.optim.entropy.alpha=0.001,0.01 \
    sac.optim.gamma=0.99 \
    optim.num_epochs=1001 \
    main.seed=234,123,345 \
    eval.interval_epochs=10 \
    replay_buffer.num_procs=20 \
    plot.type=wandb \
    hydra.launcher.partition=learnlab \
    hydra.launcher.cpus_per_task=20 \
    --multirun \
