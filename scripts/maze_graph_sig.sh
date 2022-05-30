#!/bin/bash

python main.py \
    +exp=maze_rnet_asym \
    main.suffix="rs\${replay_buffer.reward_scaling}-lr\${sac.optim.lr}-bs\${sac.optim.batch_size}-updates\${sac.optim.num_updates_per_epoch}-alpha\${sac.optim.entropy.alpha}-gamma\${sac.optim.gamma}-\${main.reward}-reward-temp-\${main.reward_sigm_temp}-subgoal\${main.subgoal_transitions}-asym-fix-seed_cudnn" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/trained_rnet/maze_asym-fixed-new/' \
    main.run=0 \
    main.reward=graph_sig \
    env.frame_stack=1 \
    #main.edge_transitions=True \
    #main.subgoal_transitions=False,True \
    #replay_buffer.reward_scaling=0.1 \
    #plot.type=wandb \
    #main.reward_sigm_temp=1 \
    #sac.optim.lr=0.0003 \
    #sac.optim.batch_size=2048 \
    #sac.optim.num_updates_per_epoch=1000 \
    #sac.optim.entropy.alpha=0.05 \
    #sac.optim.gamma=0.9 \
    #eval.interval_epochs=10 \
    #hydra.launcher.partition=devlab \
    #--multirun \
