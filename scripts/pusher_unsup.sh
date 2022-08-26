#!/bin/bash

python main.py \
    +exp=pusher_rnet \
    main.suffix="unsup_random-search_rs\${replay_buffer.reward_scaling}_lr\${sac.optim.lr}_alpha\${sac.optim.entropy.alpha}-rnet_weight\${main.reward_sigm_weight}" \
    main.reward=graph_sig \
    train.goal_strat=rb \
    main.edge_transitions=False \
    main.subgoal_transitions=True \
    main.reward_sigm_weight=10,100 \
    replay_buffer.reward_scaling=0.1,0.01 \
    sac.optim.lr=0.0005,0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.00001,0.0005,0.001,0.0001,0.00005 \
    main.seed=234,123,345 \
    sac.optim.gamma=0.95 \
    replay_buffer.num_procs=20 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
