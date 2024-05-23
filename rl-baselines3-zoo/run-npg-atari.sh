#!/usr/bin/env bash
# 1: env
# 2: seed
# 3: mom
python -m rl_zoo3.train \
    --algo trpo \
    --env $1 \
    --tensorboard-log ./logs-atari/trpo \
    --seed $2 \
    --hyperparams n_envs:2 learning_rate:0.0001 \
    --log-folder logs-atari/trpo