#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-done-neg-uniform-act-fixQ-rb1M" \
    main.reward=act_model \
    replay_buffer.neg_action=policy \
    #replay_buffer.reward_scaling=10.0,50.0,100.0 \
    #sac.optim.lr=0.0003,0.0001 \
    #sac.optim.batch_size=2048 \
    #sac.optim.entropy.alpha=0.05,0.01,0.08,0.005 \
    #sac.optim.gamma=0.95,0.99 \
    #eval.interval_epochs=10 \
    #replay_buffer.num_procs=20 \
    #plot.type=wandb \
    #hydra.launcher.partition=devlab \
    #hydra.launcher.cpus_per_task=20 \
    #--multirun \
