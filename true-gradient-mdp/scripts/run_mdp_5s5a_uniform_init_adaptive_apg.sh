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
env="mdp_5s5a_uniform"
now="$(date +'%Y%m%d-%H:%M:%S')"
# run
for algo in "APG_adaptive"
do
    python3 main.py --log_root ./logs \
                    --fname "$env"_APG_adaptive_apg \
                    --gamma 0.9 \
                    --chunk_size 1 \
                    --epoch 3000 \
                    --env ./mdp_env/$env.yaml \
                    --algo $algo
done