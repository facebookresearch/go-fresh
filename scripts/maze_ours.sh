#!/bin/bash

python -m offline_gcrl.main \
    +exp=maze_ours \
    main.suffix=ours_load-rnet \
    plot.type=wandb \
    main.seed=234,123,345 \
    hydra.launcher.cpus_per_task=20 \
    hydra.launcher.partition=devlab \
    --multirun \
