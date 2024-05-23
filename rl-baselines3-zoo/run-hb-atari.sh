#!/usr/bin/env bash
# 1: env
# 2: seed
# 3: mom
python -m rl_zoo3.train \
    --algo a2c \
    --env $1 \
    --conf-file ./hyperparams/a2c.yml \
    --tensorboard-log ./logs-atari/hb-mom=$3 \
    --seed $2 \
    --hyperparams policy_kwargs:"dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5, momentum=$3, nesterov=False))"\
    --log-folder logs-atari/hb-mom=$3