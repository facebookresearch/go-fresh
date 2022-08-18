#!/bin/bash

python main.py \
    +exp=pusher_rnet \
    main.suffix="unsup_random-search_rs\${replay_buffer.reward_scaling}_lr\${sac.optim.lr}_alpha\${sac.optim.entropy.alpha}_gamma\${sac.optim.gamma}-trans-sg\${main.edge_transitions}-e\${main.subgoal_transitions}" \
    main.reward=graph_sig \
    train.goal_strat=rb \
    main.edge_transitions=True,False \
    main.subgoal_transitions=True,False \
    replay_buffer.reward_scaling=0.1,1.0 \
    sac.optim.lr=0.001,0.0005 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.01,0.005 \
    sac.optim.gamma=0.95,0.99 \
    replay_buffer.num_procs=20 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=learnlab \
    --multirun \
