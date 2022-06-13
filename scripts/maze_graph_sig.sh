#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="baseline-HER-final-neg-action-\${replay_buffer.neg_action}-neg-goal-\${replay_buffer.neg_goal}" \
    main.reward=act_model \
    replay_buffer.neg_action=null \
    replay_buffer.neg_goal=zero \
    replay_buffer.reward_scaling=10 \
    sac.optim.lr=0.0001 \
    sac.optim.batch_size=2048 \
    sac.optim.entropy.alpha=0.001 \
    sac.optim.gamma=0.99 \
    optim.num_epochs=5001 \
    main.seed=234,123,345 \
    eval.interval_epochs=10 \
    replay_buffer.num_procs=20 \
    plot.type=wandb \
    hydra.launcher.partition=learnlab \
    hydra.launcher.cpus_per_task=20 \
    --multirun \
