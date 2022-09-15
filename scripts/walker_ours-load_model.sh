#!/bin/bash

python -m offline_gcrl.main \
    +exp=walker_ours \
    main.suffix="walker_ours-load_model" \
    main.load_from_dir='/checkpoint/linamezghani/offline-gcrl/trained_rnet/walker_memonly_thresh2_neg0_skip0.05_new-graph' \
    plot.type=wandb \
    main.seed=234,123,345 \
    hydra.launcher.cpus_per_task=20 \
    hydra.launcher.partition=devlab \
    --multirun \
