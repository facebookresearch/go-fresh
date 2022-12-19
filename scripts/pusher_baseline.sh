#!/bin/bash

python -m offline_gcrl.main \
    +exp=pusher_baseline \
    replay_buffer.algo=AM,HER,HERu \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    main.seed=234,123,345 \
    plot.type=wandb \
    hydra.launcher.partition=devlab \
    --multirun \
