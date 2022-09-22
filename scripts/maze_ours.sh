#!/bin/bash

python -m offline_gcrl.main \
    +exp=maze_ours \
    main.suffix=ours_load-rnet \
    main.load_from_dir=/checkpoint/linamezghani/offline-gcrl/trained_rnet/maze_asym-fixed-new/ \
    plot.type=wandb \
    main.seed=234,123,345 \
    hydra.launcher.cpus_per_task=20 \
    hydra.launcher.partition=devlab \
    --multirun \
