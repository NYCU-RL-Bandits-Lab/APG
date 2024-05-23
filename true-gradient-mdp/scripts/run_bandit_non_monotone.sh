#!/bin/bash
# change path if needed
LIST=`ls $search_dir`
if echo ${LIST[*]} | grep -qw "main.py"; then
    echo "found"
else
    echo "cd .."
    cd ..
    LIST=`ls $search_dir`
    if echo ${LIST[*]} | grep -qw "main.py"; then
        echo "found"
    else
        echo "not found"
        exit 1
    fi
fi
# activate env
source /home/m-ru/anaconda3/bin/activate
source /home/m-ru/anaconda3/bin/activate APG
# param
env="bandit_non_monotone"
now="$(date +'%Y%m%d-%H:%M:%S')"
# run
for algo in "NAPG"
do
    python3 main.py --log_root ./logs \
                    --fname "$env"_napg \
                    --gamma 0.0 \
                    --chunk_size 1 \
                    --epoch 500 \
                    --env ./mdp_env/$env.yaml \
                    --algo $algo
done