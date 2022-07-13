#!/bin/bash

python main.py \
    +env=cartpole\
    main.suffix="rnet-thresh2_\${exploration_buffer.algo}" \
    main.train_until=memory \
    main.reward=graph \
    rnet.dataset.thresh=2 \
    rnet.dataset.symmetric=False \
    rnet.model.remove_velocity=True \
    exploration_buffer.algo=proto,random,icm_apt \
    rnet.train.num_epochs=20 \
    rnet.memory.directed=True \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
