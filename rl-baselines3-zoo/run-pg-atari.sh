#!/usr/bin/env bash
# 1: env
# 2: seed
# 3: mom
python -m rl_zoo3.train \
    --algo a2c \
    --env $1 \
    --conf-file ./hyperparams/a2c.yml \
    --tensorboard-log ./logs-atari/pg \
    --seed $2 \
    --log-folder logs-atari/pg