#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-temp-\${main.reward_sigm_temp}-subgoal\${main.subgoal_transitions}-asym-mptest-alpha" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/trained_rnet/maze_asym-fixed/' \
    main.run=0 \
    main.reward=graph_sig \
    main.subgoal_transitions=True,False \
    plot.type=wandb \
    main.reward_sigm_temp=1 \
    replay_buffer.reward_scaling=0.1 \
    sac.optim.lr=0.0003 \
    sac.optim.batch_size=2048 \
    sac.optim.num_updates_per_epoch=1000 \
    sac.optim.entropy.alpha=0.05,0.01,0.005,0.001,0.0005,0.0001,0.1,0.5 \
    sac.optim.gamma=0.9 \
    eval.interval_epochs=10 \
    hydra.launcher.partition=devlab \
    --multirun \
