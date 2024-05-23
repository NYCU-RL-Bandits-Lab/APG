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
env="mdp_5s5a_hard"
now="$(date +'%Y%m%d-%H:%M:%S')"
# run
for algo in "APG" "PG"
do
    python3 main.py --log_root ./logs \
                    --fname "$env"_stochastic \
                    --gamma 0.9 \
                    --chunk_size 1000 \
                    --epoch 1000000 \
                    --env ./mdp_env/$env.yaml \
                    --algo $algo \
                    --stochastic \
                    --seed_num 50
done