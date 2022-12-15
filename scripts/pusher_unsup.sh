#!/bin/bash

python -m offline_gcrl.main \
    +exp=pusher_ours \
    main.suffix="unsup_edge\${main.edge_transitions}-thresh\${rnet.dataset.thresh}-rnet-lr\${rnet.train.lr}" \
    main.reward=graph_sig \
    rnet.dataset.thresh=5,10 \
    rnet.train.lr=0.001,0.0003 \
    main.edge_transitions=True,False \
    main.seed=234,123,345 \
    hydra.launcher.cpus_per_task=20 \
    eval.interval_epochs=20 \
    plot.type=wandb \
    hydra.launcher.partition=learnlab \
    --multirun \
