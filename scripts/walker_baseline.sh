#!/bin/bash

python -m offline_gcrl.main \
    +exp=walker_baseline \
    replay_buffer.algo=HER \
    # replay_buffer.algo=AM,HER,HERu \
    # main.seed=234,123,345 \
    # plot.type=wandb \
    # eval.interval_epochs=20 \
    # hydra.launcher.cpus_per_task=20 \
    # hydra.launcher.partition=learnlab \
    # --multirun \
