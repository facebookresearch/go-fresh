#!/bin/bash

python -m offline_gcrl.main \
    +exp=pusher_ours \
    main.seed=234,123,345 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
